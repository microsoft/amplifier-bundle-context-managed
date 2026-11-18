"""Host-admitted model transitions using the existing checkpoint transaction.

There is no writer or authority inference here. Hosts retain the source unit and
durably reserve each logical call in their existing recovery journal. Unknown
reservations must be reconciled by the host, never replayed from saved progress.
"""
import copy
import json

from amplifier_core import ChatRequest

from .checkpoint import classify_transition, digest
from .errors import CompactionError
from .recovery import _check_guard, _guarded, recover_native, settled_boundaries


FIELDS = ("summary", "summary_omitted_indices", "summary_identity", "summary_failure",
          "_summary_progress", "_restored_progress", "evidence_refs", "recovery",
          "checkpoint_status")


async def prepare_transition(context, *, provider, count_view, identity, save_progress,
                             install, guard, retain_contents, target_bytes, progress,
                             source_revision, source_checkpoint, admit_call, max_calls,
                             transition_source):
    """Prepare originals or regenerate target state; activate only after fit.

    ``admit_call(report)`` must durably reserve the exact operation before returning
    True. The host enforces the total allowance across restarts, including unknown
    and failed operations. ``save_progress`` acknowledges the matching reservation
    only after storing its derived record. A callback must raise to reject a save.
    Source-native provider envelopes are never validated by or sent to the target.
    Classification verifies record structure/history/configuration only; the host
    binds the previously accepted source bytes/provenance separately.
    """
    from .boundary import BoundaryContextManager

    if not all(callable(cb) for cb in (count_view, save_progress, install, guard, admit_call)):
        raise TypeError("Transitions require count_view, save_progress, install, guard and admit_call")
    if (source_checkpoint is None) == (transition_source is None):
        raise ValueError("Supply exactly one source_checkpoint or transition_source")
    _check_guard(guard)
    async with context.lock:
        source, revision = copy.deepcopy(context.messages), context.revision
        source_digest = digest(source)
        config = copy.deepcopy(context.config)
        configuration = context.checkpoint_configuration()
        target = copy.deepcopy(identity)
        record = copy.deepcopy(source_checkpoint)
        origin = copy.deepcopy(transition_source)
        progress_input, progress = progress, copy.deepcopy(progress)
        originals = {key: copy.deepcopy(getattr(context, key)) for key in FIELDS}
        factory, operations = context.factory, context.active_operations
        preserve, getter, summarizer = (context.preserve_evidence,
                                       context.checkpoint_identity, context.summary_provider)
        utility = utility_key = None
        provider_key = (getattr(provider, "name", type(provider).__name__),
                        getattr(provider, "default_model", None))

        def check():
            _check_guard(guard)
            if (context.revision != revision or digest(context.messages) != source_digest
                    or context.config != config or context.checkpoint_configuration() != configuration
                    or identity != target or source_checkpoint != record or transition_source != origin
                    or progress_input != progress
                    or context.factory is not factory or context.active_operations is not operations
                    or context.preserve_evidence is not preserve or context.checkpoint_identity is not getter
                    or context.summary_provider is not summarizer
                    or utility is not None and utility_key != (
                        getattr(utility, "name", type(utility).__name__),
                        getattr(utility, "default_model", None))
                    or utility is not None and summarizer is not None and utility_key != (
                        getattr(summarizer(), "name", type(utility).__name__),
                        getattr(summarizer(), "default_model", None))
                    or getter is not None and getter() != target
                    or provider_key != (getattr(provider, "name", type(provider).__name__),
                                        getattr(provider, "default_model", None))
                    or any(getattr(context, key) != value for key, value in originals.items())):
                raise RuntimeError("Transition source or request contract changed; active state was not replaced")

        check()
        if not config.get("durable_checkpoints") or not callable(preserve):
            raise CompactionError("recovery_evidence_required", "Transitions require host-preserved originals.")
        if record is not None:
            classification = classify_transition(context, record, target)
            if classification["status"] != "transition_required":
                raise CompactionError("transition_source_invalid",
                                      "Source checkpoint does not verify a different-model transition.")
            binding = {"messages": len(source), "sha256": source_digest}
            source_sha = record["sha256"]
            if any(originals[key] for key in ("summary", "_summary_progress", "_restored_progress")):
                active_source = BoundaryContextManager(config)
                active_source.messages = copy.deepcopy(source)
                for key in FIELDS:
                    setattr(active_source, key, copy.deepcopy(originals[key]))
                if active_source.export_checkpoint(record["identity"]) != record:
                    raise CompactionError("transition_source_invalid",
                                          "The supplied source record does not match the active source unit.")
        else:
            if (not isinstance(target, dict) or not all(
                    isinstance(target.get(k), str) and target[k].strip() for k in ("provider", "model"))
                    or not isinstance(origin, dict) or set(origin) != {"kind", "sourceRevision"}
                    or origin["kind"] != "canonical-originals"):
                raise CompactionError("transition_source_invalid", "Canonical-originals intent is invalid.")
            binding = copy.deepcopy(origin["sourceRevision"])
            source_sha = None
            # A checkpointless fork is not an escape from admitted derived work.
            if originals["summary"] or originals["_summary_progress"] or originals["_restored_progress"]:
                raise CompactionError("transition_source_invalid", "Derived source state requires its exact checkpoint.")
        if source_revision is not None:
            # The journal binds the whole admitted prefix; the source checkpoint
            # may cover only an earlier compatible prefix. Appended foreground
            # input stays protected outside that journal boundary.
            if (not isinstance(source_revision, dict)
                    or type(source_revision.get("messages")) is not int
                    or source_revision["messages"] < (
                        record["sourceRevision"]["messages"] if record else binding["messages"])):
                raise CompactionError("recovery_source_changed", "Transition source bindings disagree.")
            binding = copy.deepcopy(source_revision)
        if (not isinstance(binding, dict) or type(binding.get("messages")) is not int
                or not 0 < binding["messages"] <= len(source)
                or binding.get("sha256") != digest(source[:binding["messages"]])):
            raise CompactionError("recovery_source_changed", "The transition's canonical source changed.")
        configured_cap = config.get("summary_max_calls", 16)
        cap = configured_cap if max_calls is None else max_calls
        if (type(configured_cap) is not int or configured_cap < 1
                or type(cap) is not int or not 0 <= cap <= configured_cap):
            raise ValueError("Transition max_calls must be nonnegative and within summary_max_calls")

        staged = BoundaryContextManager(config, context.hooks)
        staged.messages, staged.revision = copy.deepcopy(source), revision
        staged.checkpoint_identity = lambda: copy.deepcopy(target)
        fitter = staged._fitter()
        policy_limit = fitter._calculate_budget(None, provider)
        prompt = await _guarded(check, factory) if factory else None
        check()
        pending = operations() if operations else None
        # Use the same synchronous operation contract as ordinary preparation.
        if hasattr(pending, "__await__"):
            if hasattr(pending, "close"):
                pending.close()
            raise TypeError("Transition active_operations must be synchronous")
        manifest = ("Pending operations (observations, not authorization):\n" +
                    json.dumps(pending)) if pending else None
        refs = await _guarded(check, preserve, copy.deepcopy(source))
        if not isinstance(refs, list):
            raise CompactionError("recovery_evidence_required", "The host did not preserve full originals.")
        staged.evidence_refs = copy.deepcopy(refs)
        carry = staged._carry(staged.messages)
        retain = tuple(retain_contents)

        def view(candidate):
            rows = candidate._assembled(candidate.messages, candidate.summary, retain,
                                        candidate._carry(candidate.messages),
                                        candidate.summary_omitted_indices)
            if factory:
                rows = [{"role": "system", "content": prompt}] + [
                    row for row in rows if row.get("role") != "system"
                    or (row.get("metadata") or {}).get("source") == "hook"]
            if manifest:
                rows.append({"role": "user", "content": manifest, "metadata": {
                    "ephemeral": True, "persisted": True, "source": "context-managed-operations"}})
            return fitter._strip_internal_metadata(rows)

        template = None
        hard_limit = None

        async def measure(rows):
            nonlocal template, hard_limit
            envelope = await _guarded(check, count_view, copy.deepcopy(rows))
            if not isinstance(envelope, dict) or not isinstance(envelope.get("dispatch"), ChatRequest):
                raise TypeError("Transition count_view must return an actual ChatRequest")
            measured = fitter._measured_budget_decision(envelope)
            if measured is None:
                raise CompactionError("native_measurement_unavailable",
                                      "Transitions require authoritative full-request measurement.")
            request = envelope["dispatch"]
            count, _estimate, limit, _source = measured
            if request.model != target["model"]:
                raise CompactionError("recovery_request_changed", "Target request model does not match selection.")
            if template is None:
                template, hard_limit = request.model_copy(deep=True), limit
            else:
                instructions = lambda req: [row.model_dump() for row in req.messages
                                           if row.role in {"system", "developer"}]
                if (template.model_dump(exclude={"messages"}) != request.model_dump(exclude={"messages"})
                        or instructions(template) != instructions(request) or limit != hard_limit
                        or fitter._calculate_budget(None, provider) != policy_limit):
                    raise CompactionError("recovery_request_changed",
                                          "Transition changed request options, instructions or allowance.")
            return count, envelope

        before, initial = await measure(view(staged))
        limit = min(policy_limit, hard_limit)
        count_calls = 1

        def result_for(final_view, envelope, after, status, sha=None):
            verification = {"status": status, "sourceRevision": copy.deepcopy(binding),
                            "targetIdentity": copy.deepcopy(target),
                            "requestSha256": digest(envelope["dispatch"].model_dump(mode="json")),
                            "before": before, "after": after, "limit": limit}
            if sha:
                verification["sha256"] = sha
            return {"base_view": final_view, "final_attempt": envelope,
                    "outcome": "transition_fitted" if status == "prepared-originals" else "transitioned",
                    "measured_before": before, "measured_after": after,
                    "policy_budget": limit, "trigger": limit, "target": limit,
                    "count_calls": count_calls, "transaction": None, "recovery": verification}

        def activate(candidate, envelope, snapshot):
            activation = {key: copy.deepcopy(getattr(candidate, key)) for key in FIELDS}
            check()
            if envelope != snapshot:
                raise CompactionError("recovery_request_changed", "Measured request changed before activation.")
            for key, value in activation.items():
                setattr(context, key, value)

        if progress is None and initial["budget_decision"]["estimated_input_tokens"] <= limit:
            final_view = view(staged)
            snapshot = copy.deepcopy(initial)
            check()
            result = result_for(final_view, initial, before, "prepared-originals")
            staged.checkpoint_status = {"status": "prepared-originals", "originalsAvailable": True}
            if initial != snapshot:
                raise CompactionError("recovery_request_changed", "Measured originals changed before activation.")
            activate(staged, initial, snapshot)
            return result

        # Check the irreducible current envelope before buying any replacement:
        # current instructions/tools/options, carried reminders and protected tail.
        end = staged._boundary(staged.messages, retain, ceiling=binding["messages"])
        if not end:
            raise CompactionError("recovery_no_eligible_work", "No settled prefix can be transitioned.")
        native = callable(getattr(provider, "supports_native_compaction", None)) and provider.supports_native_compaction()
        if progress is not None:
            restored = staged.restore_checkpoint(copy.deepcopy(progress), target)
            if restored["status"] not in {"restored", "resuming"}:
                raise CompactionError("summary_progress_incompatible", "Target progress is incompatible.")
            if staged.summary and staged.summary[0] > end:
                raise CompactionError("summary_progress_incompatible", "Target progress crosses the protected prefix.")
            if (staged.summary and isinstance(staged.summary[1], dict) != native
                    or native and staged._restored_progress):
                raise CompactionError("summary_progress_incompatible",
                                      "Target progress uses an incompatible compaction method.")
            if staged._restored_progress:
                saved_end = staged._restored_progress["source_revision"]["messages"]
                if (saved_end > end or staged._boundary(staged.messages, retain,
                                                       ceiling=saved_end) != saved_end):
                    raise CompactionError("summary_progress_incompatible",
                                          "Portable progress crosses the protected boundary.")
                # Finish precisely the accepted source first; the appended tail
                # stays outside this job, even if more history is now eligible.
                end = saved_end
            staged.evidence_refs = copy.deepcopy(refs)
        required = {id(row) for row in carry}
        objective = next((i for i, row in enumerate(staged.messages)
                          if staged._human(row)), None) if not native else None
        minimal = [row for i, row in enumerate(staged.messages)
                   if i >= end or row.get("role") in {"system", "developer"}
                   or i == objective or id(row) in required or staged._retained_reminder(row, retain)]
        floor = BoundaryContextManager(config)
        floor.messages = copy.deepcopy(minimal)
        _floor_count, floor_envelope = await measure(view(floor))
        count_calls += 1
        if floor_envelope["budget_decision"]["estimated_input_tokens"] > limit:
            raise CompactionError("native_input_oversized",
                                  "Required target instructions, tools and recent history cannot fit.")

        calls, last_call = 0, None

        async def paid(method, request, callback, **kwargs):
            nonlocal calls, last_call
            check()
            if calls >= cap:
                raise CompactionError("summary_work_paused", "The transition's call allowance is exhausted.")
            report = {"method": method, "sourceCheckpointSha256": source_sha,
                      "targetIdentity": copy.deepcopy(target), "sequence": calls + 1,
                      "requestSha256": digest(request.model_dump(mode="json")),
                      "maxOutputTokens": request.max_output_tokens}
            accepted = await _guarded(check, admit_call, copy.deepcopy(report))
            if accepted is not True:
                raise CompactionError("transition_call_unconfirmed", "The host did not reserve this exact call.")
            calls += 1
            last_call = report
            return await _guarded(check, callback, request, **kwargs)

        class BoundedProvider:
            def __init__(self, original):
                self.original = original
                if not callable(getattr(original, "request_budget", None)):
                    self.request_budget = None

            def __getattr__(self, name):
                return getattr(self.original, name)

            async def request_budget(self, *args, **kwargs):
                return await _guarded(check, self.original.request_budget, *args, **kwargs)

            async def complete(self, request, **kwargs):
                return await paid("semantic", request, self.original.complete, **kwargs)

            async def compact_context(self, request):
                return await paid("native", request, self.original.compact_context)

        bounded = BoundedProvider(provider)

        async def save(record, report):
            check()
            report = {**report, "call": copy.deepcopy(last_call), "disposition": "completed"}
            ack = await _guarded(check, save_progress, copy.deepcopy(record), report)
            if ack is False:
                raise CompactionError("checkpoint_persist_failed", "The host refused transition progress.")

        complete_progress = (staged.summary is not None and staged.summary[0] == end
                             and not staged._restored_progress)
        if complete_progress:
            candidate = staged.export_checkpoint(target)
        elif native:
            if not config.get("native_compaction", True):
                raise CompactionError("native_checkpoint_invalid", "Advertised native compaction is disabled.")
            excluded = {i for i, row in enumerate(staged.messages)
                        if id(row) in required or staged._retained_reminder(row, retain)}
            # Never infer a completed outcome for an unresolved/queued exchange.
            if end not in settled_boundaries(staged.messages):
                raise CompactionError("native_checkpoint_invalid", "The eligible prefix has an unresolved exchange.")
            compact_template = template.model_copy(deep=True, update={
                "max_output_tokens": config.get("native_compaction_max_output_tokens", 4096)})
            candidate = await recover_native(staged, bounded, compact_template, target,
                save_progress=save, target_bytes=target_bytes,
                progress=staged.export_checkpoint(target) if staged.summary else None,
                through_message=end, exclude_indices=excluded, guard=check)
        else:
            utility = summarizer() if summarizer else provider
            utility_key = (getattr(utility, "name", type(utility).__name__),
                           getattr(utility, "default_model", None))
            check()
            if getattr(utility, "native_bundle_live", False):
                raise RuntimeError("Native live providers require a separate context.summary_provider for compaction")
            async def persist():
                record = staged.export_checkpoint(target)
                if record is None:
                    raise CompactionError("checkpoint_persist_failed", "Staged portable progress has no record.")
                await save(record, {"throughMessage": end})
            staged.persist_checkpoint = persist
            # Existing portable generation, prompt/source partitioning, output
            # policy, progress compatibility and no-native-fallback semantics.
            portable = BoundedProvider(utility)
            portable_source = staged._assembled(staged.messages[:end], staged.summary,
                                               retain, carry, staged.summary_omitted_indices)
            stats = {"calls": 0}
            text = await staged._summarize(portable, portable_source, revision, stats,
                fingerprint=staged._summary_fingerprint(portable_source, target, portable),
                source_revision={"messages": end, "sha256": digest(staged.messages[:end])},
                identity=target)
            staged.summary, staged.summary_identity = (end, text), copy.deepcopy(target)
            staged.summary_omitted_indices = []
            staged._summary_progress = staged._restored_progress = None
            candidate = staged.export_checkpoint(target)
            if candidate is None:
                raise CompactionError("transition_source_invalid", "No completed target checkpoint was produced.")

        verified = BoundaryContextManager(config)
        verified.messages = copy.deepcopy(source)
        restored = verified.restore_checkpoint(candidate, target)
        if restored["status"] != "restored" or verified.summary is None:
            raise CompactionError("transition_source_invalid", "Target checkpoint did not verify.")
        if native and not provider.validate_compacted_context(verified.summary[1]):
            raise CompactionError("native_checkpoint_invalid", "Target native envelope did not verify.")
        final_view = view(verified)
        after, final_envelope = await measure(final_view)
        count_calls += 1
        if after >= before:
            raise CompactionError("native_no_reduction", "Transition did not reduce the complete target request.")
        if final_envelope["budget_decision"]["estimated_input_tokens"] > limit:
            raise CompactionError("native_input_oversized", "Transition still exceeds the target allowance.")
        verification = result_for(final_view, final_envelope, after, "installed", candidate["sha256"])
        snapshot = copy.deepcopy(final_envelope)
        check()
        acknowledged = await _guarded(check, install, copy.deepcopy(candidate),
                                      copy.deepcopy(verification["recovery"]))
        if isinstance(acknowledged, dict) and set(acknowledged) == {"sha256"}:
            acknowledged = acknowledged["sha256"]
        if type(acknowledged) is not str or acknowledged != candidate["sha256"]:
            raise CompactionError("recovery_install_unconfirmed", "Reconcile the host's checkpoint installation receipt.")
        if snapshot != final_envelope:
            raise CompactionError("recovery_request_changed", "Measured request changed during installation.")
        verified.checkpoint_status = {"status": "ready", "throughMessage": verified.summary[0],
                                      "originalsAvailable": True}
        activate(verified, final_envelope, snapshot)
        return verification