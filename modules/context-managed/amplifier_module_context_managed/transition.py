"""Host-admitted model transitions using the existing checkpoint transaction.

There is no writer or authority inference here. Hosts retain the source unit and
durably reserve each logical call in their existing recovery journal. Unknown
reservations must be reconciled by the host, never replayed from saved progress.
"""
import copy
import json

from amplifier_core import ChatRequest

from .checkpoint import classify_transition, digest, handoff_summary, native_handoff_eligible
from .errors import CompactionError
from .measurement import estimate_pressure, measured_limit, supports_native_for
from .interrupted import project as project_interrupted, unanswered
from .recovery import _check_guard, _guarded, recover_native, settled_boundaries


FIELDS = ("summary", "summary_omitted_indices", "summary_identity", "summary_failure",
          "_summary_progress", "_restored_progress", "evidence_refs", "recovery",
          "checkpoint_status", "_checkpoint_legacy_foreground")


def _remaining_allowance(configured, requested, grant, progress):
    if type(configured) is not int or configured < 1:
        raise ValueError("Invalid configured summary call allowance")
    if grant is None:
        cap = configured if requested is None else requested
        if type(cap) is not int or not 0 <= cap <= configured:
            raise ValueError("Transition max_calls must be nonnegative and within summary_max_calls")
        return cap
    # The host owns user authorization and every durable reservation. This
    # opt-in contract only resumes confirmed progress within the stated total.
    if (not isinstance(grant, dict)
            or set(grant) != {"version", "additionalCalls", "originalMaxCalls", "confirmedCalls", "authorizationId"}
            or type(grant["version"]) is not int or grant["version"] != 1
            or not isinstance(grant["authorizationId"], str) or not grant["authorizationId"].strip()
            or len(grant["authorizationId"]) > 200 or not isinstance(progress, dict)):
        raise ValueError("Invalid additional transition call allowance")
    extra, original, confirmed = grant["additionalCalls"], grant["originalMaxCalls"], grant["confirmedCalls"]
    if (type(original) is not int or not 1 <= original <= configured
            or type(extra) is not int or not 1 <= extra <= 32
            or type(confirmed) is not int or not original <= confirmed <= original + extra
            or type(requested) is not int or not 0 <= requested <= original + extra - confirmed):
        raise ValueError("Transition max_calls exceeds the confirmed additional allowance")
    return requested


async def prepare_transition(context, *, provider, count_view, identity, save_progress,
                             install, guard, retain_contents, target_bytes, progress,
                             source_revision, source_checkpoint, admit_call, max_calls,
                             transition_source, installed_candidate_sha256, additional_call_allowance=None):
    """Prepare originals or regenerate target state; activate only after fit.

    ``admit_call(report)`` must durably reserve the exact operation before returning
    True. The host enforces the total allowance across restarts, including unknown
    and failed operations. ``save_progress`` acknowledges the matching reservation
    only after storing its derived record. A callback must raise to reject a save.
    Only a same-connection handoff qualified by the mounted provider can reuse
    a completed source-native window.
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
        authorized = copy.deepcopy(additional_call_allowance)
        readback_sha = installed_candidate_sha256
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
                    or additional_call_allowance != authorized or progress_input != progress
                    or installed_candidate_sha256 != readback_sha
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

        def active_pinned_readback():
            if not (
                    readback_sha is not None and isinstance(progress, dict)
                    and progress.get("sha256") == readback_sha
                    and progress.get("summary") is not None and progress.get("progress") is None
                    and originals["summary_identity"] == target
                    and originals["_summary_progress"] is None
                    and originals["_restored_progress"] is None):
                return False
            active = BoundaryContextManager(config)
            active.messages = copy.deepcopy(source)
            for key in FIELDS:
                setattr(active, key, copy.deepcopy(originals[key]))
            return active.export_checkpoint(target) == progress

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
                if not active_pinned_readback() and active_source.export_checkpoint(record["identity"]) != record:
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
            if (any(originals[key] for key in ("summary", "_summary_progress", "_restored_progress"))
                    and not active_pinned_readback()):
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
        cap = _remaining_allowance(config.get("summary_max_calls", 16), max_calls,
                                   additional_call_allowance, progress)

        if readback_sha is not None and (
                not isinstance(progress, dict) or progress.get("sha256") != readback_sha
                or progress.get("progress") is not None or progress.get("summary") is None):
            raise CompactionError("transition_candidate_invalid",
                                  "Readback requires the exact host-pinned completed candidate.")

        staged = BoundaryContextManager(config, context.hooks)
        staged.messages, staged.revision = copy.deepcopy(source), revision
        staged.checkpoint_identity = lambda: copy.deepcopy(target)
        fitter = staged._fitter()
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
        limit = None

        async def measure(rows):
            nonlocal template, limit
            envelope = await _guarded(check, count_view, copy.deepcopy(rows))
            if not isinstance(envelope, dict) or not isinstance(envelope.get("dispatch"), ChatRequest):
                raise TypeError("Transition count_view must return an actual ChatRequest")
            measured = fitter._measured_budget_decision(envelope)
            if measured is None:
                raise CompactionError("native_measurement_unavailable",
                                      "Transitions require authoritative full-request measurement.")
            request = envelope["dispatch"]
            count, _estimate, provider_limit, _source = measured
            effective_limit = measured_limit(config, provider_limit, fitter._calculate_budget(None, provider))
            # Selected-provider adapters deliberately leave the loop's dispatch
            # unchanged while applying their bound model to cached wire requests.
            # Preserve that dispatch identity; only an absent model may use the
            # explicitly guarded provider default, never override an explicit one.
            effective_model = request.model if request.model is not None else provider_key[1]
            if (effective_model != target["model"]
                    or request.model is None and (
                        envelope.get("effective_identity") != target
                        or getter is None or getter() != target)):
                raise CompactionError("recovery_request_changed", "Target request model does not match selection.")
            if template is None:
                template, limit = request.model_copy(deep=True), effective_limit
            else:
                instructions = lambda req: [row.model_dump() for row in req.messages
                                           if row.role in {"system", "developer"}]
                if (template.model_dump(exclude={"messages"}) != request.model_dump(exclude={"messages"})
                        or instructions(template) != instructions(request) or effective_limit < limit):
                    raise CompactionError("recovery_request_changed",
                                          "Transition changed request options, instructions or allowance.")
            return count, envelope

        before = None
        count_calls = 0

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

        async def install_fit(verified, candidate, final_view, envelope, after):
            result = result_for(final_view, envelope, after, "installed", candidate["sha256"])
            snapshot = copy.deepcopy(envelope)
            check()
            acknowledged = await _guarded(check, install, copy.deepcopy(candidate),
                                          copy.deepcopy(result["recovery"]))
            if isinstance(acknowledged, dict) and set(acknowledged) == {"sha256"}:
                acknowledged = acknowledged["sha256"]
            if type(acknowledged) is not str or acknowledged != candidate["sha256"]:
                raise CompactionError("recovery_install_unconfirmed",
                                      "Reconcile the host's checkpoint installation receipt.")
            verified.checkpoint_status = {"status": "ready", "throughMessage": verified.summary[0],
                                          "originalsAvailable": True}
            activate(verified, envelope, snapshot)
            return result

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
            if request.model and request.model != target["model"]:
                report["requestModel"] = request.model
            accepted = await _guarded(check, admit_call, copy.deepcopy(report))
            if accepted is not True:
                raise CompactionError("transition_call_unconfirmed", "The host did not reserve this exact call.")
            calls += 1
            last_call = report
            return await _guarded(check, callback, request, **kwargs)

        async def save(record, report):
            check()
            report = {**report, "call": copy.deepcopy(last_call), "disposition": "completed"}
            ack = await _guarded(check, save_progress, copy.deepcopy(record), report)
            if ack is False:
                raise CompactionError("checkpoint_persist_failed", "The host refused transition progress.")

        rebind = getattr(provider, "rebind_compacted_context", None)
        summary = (record or {}).get("summary") or {}
        same_connection = record and record["identity"]["provider"] == target["provider"]
        native = supports_native_for(provider, target["model"])
        portable_base = None
        reused = None
        if progress is None and record and summary and record.get("progress") is None:
            base = handoff_summary(staged, provider, target, source_identity=record["identity"],
                summary=(summary["throughMessage"], summary.get("message") or summary.get("text")),
                omitted=summary.get("omittedIndices", []), retain=retain, ceiling=binding["messages"])
            if base is not None and isinstance(base[1], str):
                portable_base = base
            else:
                reused = base

        async def try_checkpoint(base, *, renewed=False, saved=None):
            nonlocal count_calls, before, initial
            existing = BoundaryContextManager(config)
            existing.messages = copy.deepcopy(source)
            previous = saved if saved is not None else record
            if existing.restore_checkpoint(previous, previous["identity"])["status"] != "restored":
                raise CompactionError("transition_source_invalid", "The source checkpoint could not be restored.")
            existing._checkpoint_legacy_foreground = None
            existing.summary = base
            if isinstance(base[1], str) and summary.get("kind") == "native":
                existing.summary_omitted_indices = []
                if (existing.recovery or {}).get("kind") == "native-recovery":
                    existing.recovery = None
            existing.summary_identity = copy.deepcopy(target)
            if saved is None:
                existing.evidence_refs = copy.deepcopy(refs)
            candidate = saved if saved is not None else existing.export_checkpoint(target)
            if renewed:
                await save(candidate, {"throughMessage": base[0]})
            rows = view(existing)
            after, envelope = await measure(rows)
            count_calls += 1
            if envelope["budget_decision"]["estimated_input_tokens"] <= limit:
                return await install_fit(existing, candidate, rows, envelope, after)
            if not isinstance(base[1], str):
                before, initial = after, envelope
            return None

        initial = None
        if (record and not record.get("progress") and summary.get("kind") == "native"
                and progress is None and not (portable_base or reused)):
            planner = getattr(provider, "checkpoint_renewal_request", None)
            renew = getattr(provider, "renew_compacted_context", None)
            through = summary["throughMessage"]
            if callable(planner) and callable(renew) and native_handoff_eligible(
                    staged, provider, target, record["identity"], through, summary.get("omittedIndices", []),
                    retain, binding["messages"]):
                request = planner(summary["message"], source_model=record["identity"]["model"], model=target["model"])
                if request is not None:
                    window = await paid("native", request, renew, model=target["model"])
                    if not provider.validate_compacted_context(window):
                        raise CompactionError("native_checkpoint_invalid", "Renewed checkpoint did not verify.")
                    reused = through, window
        saved = None
        if progress and not progress.get("progress") and (progress.get("summary") or {}).get("kind") == "native":
            summary_progress = progress["summary"]
            if (progress.get("identity") == target and native_handoff_eligible(
                    staged, provider, target, target, summary_progress["throughMessage"],
                    summary_progress.get("omittedIndices", []), retain, binding["messages"])
                    and provider.validate_compacted_context(summary_progress["message"])):
                saved = progress
                reused = summary_progress["throughMessage"], summary_progress["message"]

        if reused is not None:
            result = await try_checkpoint(reused, renewed=last_call is not None, saved=saved)
            if result is not None:
                return result

        if initial is None:
            before, initial = await measure(view(staged))
            count_calls += 1
        source_request_snapshot = copy.deepcopy(initial)

        if progress is None and initial["budget_decision"]["estimated_input_tokens"] <= limit:
            final_view = view(staged)
            check()
            result = result_for(final_view, initial, before, "prepared-originals")
            staged.checkpoint_status = {"status": "prepared-originals", "originalsAvailable": True}
            if initial != source_request_snapshot:
                raise CompactionError("recovery_request_changed", "Measured originals changed before activation.")
            activate(staged, initial, source_request_snapshot)
            return result

        # Check the irreducible current envelope before buying any replacement:
        # current instructions/tools/options, carried reminders and protected tail.
        end = staged._boundary(staged.messages, retain, ceiling=binding["messages"])
        if not end:
            raise CompactionError("recovery_no_eligible_work", "No settled prefix can be transitioned.")
        # An incompatible native window cannot be sent to a foreign model.
        # A fresh cross-connection handoff is plain text, even when that
        # destination can create its own native checkpoints for later turns.
        portable_handoff = origin is not None or not same_connection or (callable(rebind) and reused is None)
        if progress is None and portable_base:
            result = await try_checkpoint(portable_base)
            if result is not None:
                return result
            staged.summary, staged.summary_identity = portable_base, copy.deepcopy(target)
            if (record.get("recovery") or {}).get("kind") != "native-recovery":
                staged.recovery = copy.deepcopy(record.get("recovery"))
            native = False
        elif progress is None and portable_handoff:
            native = False
        elif progress is None and reused is not None:
            # The qualified window fits the destination protocol but its tail
            # has grown. Extend that verified prefix instead of starting over.
            staged.summary, staged.summary_identity = reused, copy.deepcopy(target)
            staged.summary_omitted_indices = copy.deepcopy(summary.get("omittedIndices", []))
            staged.recovery = copy.deepcopy(record.get("recovery"))
        if progress is not None:
            # Resume the exact method recorded by the host. A provider gaining
            # native support cannot invalidate an in-flight readable handoff.
            if (progress.get("summary") or {}).get("text") or progress.get("progress"):
                native = False
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
            if readback_sha is not None:
                # A final installed candidate is not unfinished generation.
                # New foreground input may advance eligibility, but readback
                # never extends its exact original coverage or digest.
                through = staged.summary[0]
                if staged._boundary(staged.messages, retain, ceiling=through) != through:
                    raise CompactionError("transition_candidate_invalid",
                                          "Pinned candidate crosses the current retention boundary.")
                end = through
            if native and staged.summary:
                # Reuse cannot silently consume newly required retained rows
                # that the recorded target window did not leave outside.
                through = staged.summary[0]
                if any((id(row) in {id(item) for item in carry}
                        or staged._retained_reminder(row, retain))
                       and i not in staged.summary_omitted_indices
                       for i, row in enumerate(staged.messages[:through])):
                    raise CompactionError("recovery_retention_changed",
                                          "Target progress cannot verify current retained rows.")
        required = {id(row) for row in carry}
        objective = next((i for i, row in enumerate(staged.messages)
                          if staged._human(row)), None) if not native else None
        async def measure_tail(cut):
            nonlocal count_calls
            floor = BoundaryContextManager(config)
            floor.messages = [copy.deepcopy(row) for i, row in enumerate(staged.messages)
                              if i >= cut or row.get("role") in {"system", "developer"} or i == objective
                              or id(row) in required or staged._retained_reminder(row, retain)]
            rows = project_interrupted(view(floor), unanswered(staged.messages, staged._human))
            _, envelope = await measure(rows)
            count_calls += 1
            return envelope["budget_decision"]["estimated_input_tokens"]

        floor_count = await measure_tail(end)
        if floor_count > limit:
            raise CompactionError("native_input_oversized",
                                  "Required target instructions, tools and recent history cannot fit.")
        if progress is None and portable_handoff and not native:
            # Plan a larger verbatim tail locally, then check it once. Do not
            # binary search the remote tokenizer or split a protocol exchange.
            # A final measurement below always verifies the real generated note.
            reserve = config.get("summary_max_output_tokens") or 8192
            room = limit - floor_count - reserve
            base = staged.summary[0] if staged.summary else 0
            if room > 0:
                # Calibrate the cheap row estimate against the exact original
                # request (including tool and provider serialization overhead).
                scale = max(1, before / max(1, estimate_pressure(fitter, staged.messages)))
                cuts = staged._boundaries(staged.messages, retain, ceiling=end)
                proposed, added = end, 0
                for cut in reversed(cuts):
                    if cut <= base:
                        break
                    added += estimate_pressure(fitter, staged.messages[cut:proposed])
                    if added * scale > room:
                        break
                    proposed = cut
                if proposed < end and await measure_tail(proposed) <= limit - reserve:
                    end = proposed

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

        fit_envelope = None
        complete_progress = (staged.summary is not None and staged.summary[0] == end
                             and not staged._restored_progress)
        if complete_progress and progress is not None:
            # Exact bytes/digest matter for lost-ACK readback. Freshly preserved
            # references do not authorize rewriting an already installed record.
            candidate = copy.deepcopy(progress)
        elif native:
            staged.evidence_refs = copy.deepcopy(refs)
            if not config.get("native_compaction", True):
                raise CompactionError("native_checkpoint_invalid", "Advertised native compaction is disabled.")
            excluded = {i for i, row in enumerate(staged.messages)
                        if id(row) in required or staged._retained_reminder(row, retain)}
            # Never infer a completed outcome for an unresolved/queued exchange.
            if end not in settled_boundaries(staged.messages):
                raise CompactionError("native_checkpoint_invalid", "The eligible prefix has an unresolved exchange.")
            from .native_budget import native_output_limit
            output = await native_output_limit(provider, template.model, config, initial["budget_decision"])
            check()
            compact_template = template.model_copy(deep=True, update={"max_output_tokens": output})
            async def remainder_fits():
                # A confirmed native prefix may already make the intact suffix
                # fit. Measure the exact foreground envelope, including tools.
                nonlocal count_calls, fit_envelope
                through = staged.summary[0]
                if staged._boundary(staged.messages, retain, ceiling=through) != through:
                    return False
                rows = view(staged)
                count, envelope = await measure(rows)
                count_calls += 1
                if count < before and envelope["budget_decision"]["estimated_input_tokens"] <= limit:
                    fit_envelope = rows, count, envelope
                    return True
                return False

            candidate = await recover_native(staged, bounded, compact_template, target,
                save_progress=save, target_bytes=target_bytes,
                progress=staged.export_checkpoint(target) if staged.summary else None,
                through_message=end, exclude_indices=excluded, guard=check,
                stop_when_fit=remainder_fits if progress is not None or reused is not None else None,
                installable_boundaries=(staged._boundaries(staged.messages, retain, ceiling=end)
                                        if progress is not None or reused is not None else None))
        else:
            staged.evidence_refs = copy.deepcopy(refs)
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
        if fit_envelope is not None and final_view == fit_envelope[0]:
            # No intervening provider or host call since this exact final
            # measurement. Retain its bound dispatch/options in the receipt.
            check()
            _, after, final_envelope = fit_envelope
        else:
            after, final_envelope = await measure(final_view)
            count_calls += 1
        if after >= before:
            raise CompactionError("native_no_reduction", "Transition did not reduce the complete target request.")
        if final_envelope["budget_decision"]["estimated_input_tokens"] > limit:
            raise CompactionError("native_input_oversized", "Transition still exceeds the target allowance.")
        return await install_fit(verified, candidate, final_view, final_envelope, after)
