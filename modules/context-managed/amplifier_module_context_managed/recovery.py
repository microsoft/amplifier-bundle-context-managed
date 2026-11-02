"""Explicit, offline native recovery of oversized legacy history.

This is called explicitly, including when a host consumes a one-shot recovery
intent during ordinary preparation. It is never an automatic failure fallback.
The caller owns exclusive history access and installation of the final derived
checkpoint. No foreground inference or tools run here. Eligible source messages
are covered in order; explicitly retained rows stay outside native windows.
"""
import copy
import inspect
import json

from amplifier_core import ChatRequest, Message

from .checkpoint import digest
from .errors import CompactionError
from .summary import request_fits


def _check_guard(guard):
    """A guard is synchronous: return None/True, or raise to reject stale work."""
    if guard is None:
        return
    result = guard()
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("Recovery guard must be synchronous")
    if result is not None and result is not True:
        raise RuntimeError("Recovery guard rejected the current request")


async def _guarded(check_guard, callback, *args, **kwargs):
    """Check both sides of a callback, including a failed/cancelled await."""
    _check_guard(check_guard)
    try:
        value = callback(*args, **kwargs)
        value = await value if inspect.isawaitable(value) else value
    except BaseException:
        # Observe invalidation without replacing the original failure/cancellation.
        try:
            _check_guard(check_guard)
        except BaseException:
            pass
        raise
    else:
        _check_guard(check_guard)
        return value


def _retention_binding(excluded, through):
    """Bind exclusions to canonical offsets, not to the filtered batch view."""
    return {"kind": "native-recovery", "version": 1,
            "retentionSha256": digest(sorted(i for i in excluded if i < through))}


def _pending_receipt(content):
    """Queued results are protocol observations, not completed exchanges."""
    if isinstance(content, list):
        return any(_pending_receipt(block.get("text", ""))
                   for block in content if isinstance(block, dict) and block.get("type") == "text")
    try:
        receipt = json.loads(content) if isinstance(content, str) else content
    except (ValueError, TypeError):
        return False
    return (isinstance(receipt, dict) and receipt.get("status") in {"queued", "pending"}
            and bool(receipt.get("job_id")))


def settled_boundaries(messages, start=0):
    """Complete protocol exchanges, including batches of parallel tool calls."""
    pending = set()
    boundaries = []
    for index, row in enumerate(messages):
        calls = list(row.get("tool_calls") or [])
        content = row.get("content")
        if isinstance(content, list):
            calls += [block for block in content if block.get("type") in {"tool_call", "tool_use"}]
        for call in calls:
            if call.get("id"):
                pending.add(call["id"])
        if row.get("role") == "tool":
            if not _pending_receipt(content):
                pending.discard(row.get("tool_call_id"))
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result" and not _pending_receipt(block.get("content")):
                    pending.discard(block.get("tool_use_id"))
        if not pending and index + 1 > start:
            boundaries.append(index + 1)
    return boundaries


async def recover_native(context, provider, template: ChatRequest, identity, *,
                         save_progress, target_bytes=1_500_000, progress=None,
                         through_message=None, exclude_indices=(), guard=None):
    """Build a derived checkpoint, using a compatible saved prefix if supplied.

    Byte size chooses a tentative batch only. Authoritative provider preflight
    must accept each exact request. Oversize batches shrink at settled exchange
    boundaries; even the smallest rejected batch fails without dropping input.

    ``save_progress`` may be sync or async. Its return value is ignored; raise
    to reject a save. Export/save failure or cancellation restores the prior
    accepted summary and identity, not completed provider calls or storage writes.

    ``through_message`` bounds work without truncating canonical history.
    ``exclude_indices`` retains canonical offsets/digests while omitting those
    rows from batches. Bounded/excluding recovery requires matching retention
    metadata on supplied progress. ``guard`` fences every count, compact and save.
    """
    _check_guard(guard)
    if not provider.supports_native_compaction():
        raise CompactionError("native_checkpoint_invalid", "Explicit native recovery requires native support.")
    if type(target_bytes) is not int or target_bytes < 1:
        raise ValueError("target_bytes must be positive")
    source = copy.deepcopy(context.messages)
    source_digest = digest(source)
    stop = len(source) if through_message is None else through_message
    if type(stop) is not int or not 0 <= stop <= len(source):
        raise ValueError("through_message must be a canonical source boundary")
    excluded = set(exclude_indices)
    if any(type(i) is not int or not 0 <= i < len(source) for i in excluded):
        raise ValueError("exclude_indices must contain canonical source indices")
    saved_recovery = progress.get("recovery") if isinstance(progress, dict) else None
    bind_retention = (through_message is not None or bool(excluded)
                      or (isinstance(saved_recovery, dict) and saved_recovery.get("kind") == "native-recovery"))
    boundaries = [end for end in settled_boundaries(source) if end <= stop]
    if stop and (not boundaries or boundaries[-1] != stop):
        raise CompactionError("native_checkpoint_invalid", "History contains an unresolved tool exchange; resolve its actual outcome before recovery.")
    if progress is not None:
        # Restore is destructive by design for a newly mounted context. Rejecting
        # supplied recovery progress must not erase an already accepted checkpoint
        # or its failure latch in this caller-owned context.
        fields = ("summary", "summary_identity", "summary_failure", "_summary_progress",
                  "_restored_progress", "evidence_refs", "recovery", "checkpoint_status")
        prior = {key: copy.deepcopy(getattr(context, key)) for key in fields}
        try:
            state = context.restore_checkpoint(progress, identity)
            if (state["status"] != "restored" or context.summary is None
                    or not isinstance(context.summary[1], dict)
                    or context.summary[0] not in boundaries
                    or not provider.validate_compacted_context(context.summary[1])):
                raise CompactionError("native_checkpoint_invalid", "Recovery progress is incompatible with this history or model.")
            if bind_retention and context.recovery != _retention_binding(excluded, context.summary[0]):
                raise CompactionError("recovery_retention_changed", "Saved progress does not verify the required retained canonical rows.")
        except BaseException:
            for key, value in prior.items():
                setattr(context, key, value)
            raise
    elif context.summary is not None:
        raise ValueError("Pass the existing checkpoint explicitly as recovery progress")
    start, previous = context.summary if context.summary else (0, None)
    if previous and not provider.validate_compacted_context(previous):
        raise CompactionError("native_checkpoint_invalid", "The provider rejected the recovery checkpoint.")
    while start < stop:
        candidates = [end for end in boundaries if end > start]
        if excluded:
            included = [i for i in range(start, stop) if i not in excluded]
            if not included:
                raise CompactionError("recovery_no_eligible_work", "Only retained canonical rows remain in the recovery prefix.")
            # Never count an empty leading batch or leave an excluded-only tail
            # requiring a redundant compaction of the preceding native window.
            candidates = [end for end in candidates if end > included[0]
                          and (end == stop or end <= included[-1])]
        total = 0
        end = candidates[0]
        allowed = set(candidates)
        for index in range(start, stop):
            if index not in excluded:
                total += len(json.dumps(source[index], ensure_ascii=False).encode())
            if index + 1 in allowed:
                if total > target_bytes and index + 1 != candidates[0]:
                    break
                end = index + 1
        candidate_index = candidates.index(end)
        while True:
            rows = ([copy.deepcopy(previous)] if previous else []) + [
                copy.deepcopy(source[i]) for i in range(start, end) if i not in excluded]
            # Current prompt instructions belong to the request envelope, not
            # whichever historical batch happens to contain an old prompt.
            instructions = [row.model_copy(deep=True) for row in template.messages
                            if row.role in {"system", "developer"}]
            roles = {row.role for row in instructions}
            request = template.model_copy(deep=True, update={
                "messages": instructions + [Message(**row) for row in rows if row.get("role") not in roles],
                "metadata": {**(template.metadata or {}), "purpose": "context-compaction",
                             "stream": False, "native_compaction_request_context": True}})
            fits, decision = await _guarded(guard, request_fits, provider, request)
            if decision is None or decision.get("measurement_kind") != "provider_count":
                raise CompactionError("native_measurement_unavailable", "Recovery requires authoritative native input measurement.")
            if fits:
                break
            if candidate_index == 0:
                raise CompactionError("native_input_oversized", "The prior native window plus the next complete exchange exceeds the allowance. Recovery stopped with its last checkpoint preserved.")
            candidate_index //= 2
            end = candidates[candidate_index]
        result = await _guarded(guard, provider.compact_context, request)
        message = result.get("message")
        if result.get("kind") != "native" or not isinstance(message, dict) or not provider.validate_compacted_context(message):
            raise CompactionError("native_checkpoint_invalid", "Recovery returned an invalid native checkpoint.")
        if digest(context.messages) != source_digest:
            raise RuntimeError("History changed during recovery; new checkpoint was not installed")
        prior_summary, prior_identity, prior_recovery = context.summary, context.summary_identity, context.recovery
        try:
            context.summary = (end, copy.deepcopy(message))
            context.summary_identity = copy.deepcopy(identity)
            if bind_retention:
                context.recovery = _retention_binding(excluded, end)
            record = context.export_checkpoint(identity)
            await _guarded(guard, save_progress, record, {"fromMessage": start, "throughMessage": end,
                "totalMessages": len(source), "inputTokens": decision["input_tokens"],
                "inputLimit": decision["input_limit_tokens"]})
        except BaseException:
            context.summary, context.summary_identity = prior_summary, prior_identity
            context.recovery = prior_recovery
            raise
        start, previous = end, message
    return context.export_checkpoint(identity)


async def recover_request(context, *, provider, count_view, identity, save_progress,
                          install, guard, retain_contents=(), target_bytes=1_500_000,
                          progress=None, source_revision=None):
    """Explicitly recover, verify and install a native checkpoint under the lock.

    ``count_view`` must only rebuild/count the complete ordinary ChatRequest:
    no request hooks, completion, tools, fitting, or output-reserve relief. Its
    {dispatch, budget_decision} uses the request fitter's authoritative validator.
    The host's mandatory synchronous ``guard`` must fence runtime/provider and
    callback-envelope ownership; it returns None/True or raises. Local history,
    configuration, identity, factory and derived-state changes are also fenced.

    Originals are preserved by the existing host callback before derivation.
    Progress saves contain private native state and are NOT installation.
    Supplied progress must bind retained canonical indices; an already active
    checkpoint without that binding is usable only when none of its covered
    rows require exclusion. Invalid progress never silently restarts recovery.

    ``install(record, verification)`` must acknowledge the exact record SHA-256,
    as a string or {"sha256": ...}. Only then are prevalidated fields activated
    synchronously. Failure/cancellation leaves active context untouched by this
    operation, but cannot undo provider calls or host writes. In particular, an
    install followed by guard rejection requires host receipt reconciliation.

    ``source_revision`` optionally bounds work to an explicitly authorized
    canonical prefix; appended current input remains outside that boundary.
    Returns the ordinary measured-view envelope plus bounded recovery evidence.
    Only an explicit host intent may invoke it during ordinary preparation.
    """
    from .boundary import BoundaryContextManager

    if not all(callable(callback) for callback in (count_view, save_progress, install, guard)):
        raise TypeError("Recovery requires count_view, save_progress, install and a synchronous guard")
    _check_guard(guard)
    await context.lock.acquire()
    try:
        _check_guard(guard)
        # Snapshot before the first factory/provider/host callback can suspend.
        source, revision = copy.deepcopy(context.messages), context.revision
        source_digest = digest(source)
        maximum = len(source)
        if source_revision is not None:
            if (not isinstance(source_revision, dict)
                    or type(source_revision.get("messages")) is not int
                    or not 0 < source_revision["messages"] <= len(source)
                    or digest(source[:source_revision["messages"]]) != source_revision.get("sha256")):
                raise CompactionError("recovery_source_changed", "The explicitly authorized recovery prefix changed.")
            maximum = source_revision["messages"]
        config = copy.deepcopy(context.config)
        configuration = context.checkpoint_configuration()
        identity_input, identity = identity, copy.deepcopy(identity)
        progress = copy.deepcopy(progress)
        retain = tuple(retain_contents)
        factory, operations = context.factory, context.active_operations
        identity_getter, preserve = context.checkpoint_identity, context.preserve_evidence
        provider_identity = (getattr(provider, "name", type(provider).__name__),
                             getattr(provider, "default_model", None))
        fields = ("summary", "summary_identity", "summary_failure", "_summary_progress",
                  "_restored_progress", "evidence_refs", "recovery", "checkpoint_status")
        originals = {key: copy.deepcopy(getattr(context, key)) for key in fields}

        def check():
            _check_guard(guard)
            if (context.revision != revision or digest(context.messages) != source_digest
                    or context.config != config or identity_input != identity
                    or context.checkpoint_configuration() != configuration
                    or context.factory is not factory or context.active_operations is not operations
                    or context.checkpoint_identity is not identity_getter
                    or context.preserve_evidence is not preserve
                    or (identity_getter is not None and identity_getter() != identity)
                    or provider_identity != (getattr(provider, "name", type(provider).__name__),
                                             getattr(provider, "default_model", None))
                    or any(getattr(context, key) != value for key, value in originals.items())):
                raise RuntimeError("Recovery source or request contract changed; active checkpoint was not replaced")

        check()
        if not config.get("durable_checkpoints") or not callable(preserve):
            raise CompactionError("recovery_evidence_required", "Explicit recovery requires durable checkpoints and the host preserve_evidence callback.")
        if not isinstance(identity, dict) or not identity.get("provider") or not identity.get("model"):
            raise ValueError("Recovery identity needs an explicit provider and model")
        if (not config.get("native_compaction", True)
                or not callable(getattr(provider, "supports_native_compaction", None))
                or not provider.supports_native_compaction()
                or not callable(getattr(provider, "validate_compacted_context", None))
                or not callable(getattr(provider, "compact_context", None))):
            raise CompactionError("native_checkpoint_invalid", "Explicit recovery requires enabled native compaction and checkpoint validation.")
        if type(target_bytes) is not int or target_bytes < 1:
            raise ValueError("target_bytes must be positive")

        def fresh():
            clone = BoundaryContextManager(copy.deepcopy(config))
            clone.messages, clone.revision = copy.deepcopy(source), revision
            return clone

        baseline = fresh()
        carry = baseline._carry(source)
        excluded = {i for i, row in enumerate(source)
                    if row in carry or baseline._retained_reminder(row, retain)}
        end = min(baseline._boundary(source, retain), maximum)
        boundaries = settled_boundaries(source)

        def validate(record):
            clone = fresh()
            state = clone.restore_checkpoint(record, identity)
            if (state["status"] != "restored" or clone.summary is None
                    or not isinstance(clone.summary[1], dict)
                    or clone.summary[0] not in boundaries
                    or not provider.validate_compacted_context(clone.summary[1])):
                raise CompactionError("native_checkpoint_invalid", "Recovery checkpoint does not verify against canonical history and provider identity.")
            through = clone.summary[0]
            if any(isinstance(row.get("content"), str) and row["content"] in retain
                   and not clone._retained_reminder(row, retain) for row in source[:through]):
                raise CompactionError("recovery_retention_changed", "A required non-reminder is inside prior checkpoint coverage; recovery cannot silently shift or restart that coverage.")
            return clone

        active_record = None
        if originals["summary"] is not None:
            for key in ("summary", "summary_identity", "evidence_refs", "recovery"):
                setattr(baseline, key, copy.deepcopy(originals[key]))
            active_record = baseline.export_checkpoint(identity)
            baseline = validate(active_record)
        prior_end = baseline.summary[0] if baseline.summary else 0
        if not end or end <= prior_end:
            raise CompactionError("recovery_no_eligible_work", "No new eligible canonical prefix is available before the protected recent history and retention boundary.")
        if end not in boundaries:
            raise CompactionError("native_checkpoint_invalid", "The eligible prefix crosses an unresolved tool exchange; recovery cannot infer its outcome.")

        seed = progress if progress is not None else active_record
        staged = validate(seed) if seed is not None else fresh()
        if seed is not None:
            through = staged.summary[0]
            if not prior_end <= through <= end:
                raise CompactionError("native_checkpoint_invalid", "Recovery progress would rewind accepted coverage or cross the eligible prefix.")
            binding = _retention_binding(excluded, through)
            if staged.recovery != binding:
                # Only the actual accepted checkpoint may lack new metadata.
                # Opaque saved progress alone cannot prove exclusion compatibility.
                if (seed != active_record or staged.recovery is not None
                        or any(i < through for i in excluded)):
                    raise CompactionError("recovery_retention_changed", "Saved progress cannot verify the current retained canonical indices.")
                staged.recovery = binding

        fitter = baseline._fitter()
        policy_limit = fitter._calculate_budget(None, provider)
        system_prompt = await _guarded(check, factory) if factory else None
        # Like _prepare, operation observations are one synchronous snapshot.
        check()
        pending = operations() if operations else None
        if inspect.isawaitable(pending):
            if inspect.iscoroutine(pending):
                pending.close()
            raise TypeError("Recovery active_operations must be synchronous")
        manifest = ("Pending operations (observations, not authorization):\n" + json.dumps(pending)) if pending else None
        check()
        refs = await _guarded(check, preserve, copy.deepcopy(source))
        if not isinstance(refs, list):
            raise CompactionError("recovery_evidence_required", "The host did not return durable references for the full canonical history.")
        staged.evidence_refs = copy.deepcopy(refs)

        def view(summary):
            rows = baseline._assembled(source, summary, retain, carry)
            if factory:
                rows = [{"role": "system", "content": system_prompt}] + [
                    row for row in rows if row.get("role") != "system"
                    or (row.get("metadata") or {}).get("source") == "hook"]
            if manifest:
                rows.append({"role": "user", "content": manifest,
                    "metadata": {"ephemeral": True, "persisted": True, "source": "context-managed-operations"}})
            return fitter._strip_internal_metadata(rows)

        async def measure(rows):
            envelope = await _guarded(check, count_view, copy.deepcopy(rows))
            if not isinstance(envelope, dict) or not isinstance(envelope.get("dispatch"), ChatRequest):
                raise TypeError("Recovery count_view must return an actual ChatRequest in dispatch")
            measured = fitter._measured_budget_decision(envelope)
            if measured is None:
                raise CompactionError("native_measurement_unavailable", "Recovery requires authoritative measurement of the complete ordinary request.")
            count, _estimate, limit, _source = measured
            return count, limit, envelope

        before, hard_limit, before_envelope = await _guarded(check, measure, view(baseline.summary))
        template = before_envelope["dispatch"].model_copy(deep=True)
        before_decision = copy.deepcopy(before_envelope["budget_decision"])
        limit = min(policy_limit, hard_limit)
        # Seed with freshly preserved references, never with a caller's stale refs.
        seed = staged.export_checkpoint(identity) if staged.summary else None
        record = await _guarded(check, recover_native, staged, provider, template, identity,
            save_progress=save_progress, target_bytes=target_bytes, progress=seed,
            through_message=end, exclude_indices=excluded, guard=check)
        # Restore the exact export against full originals on a disposable context.
        # Active restore would erase latches/progress even on rejection.
        validated = validate(record)
        if (validated.summary[0] != end or validated.recovery != _retention_binding(excluded, end)
                or validated.export_checkpoint(identity) != record):
            raise CompactionError("native_checkpoint_invalid", "The candidate checkpoint did not round-trip exactly.")
        final_view = view(validated.summary)
        after, final_limit, final_envelope = await _guarded(check, measure, final_view)
        final_request, after_decision = final_envelope["dispatch"], final_envelope["budget_decision"]
        instructions = lambda request: [row.model_dump() for row in request.messages
                                        if row.role in {"system", "developer"}]
        if (template.model_dump(exclude={"messages"}) != final_request.model_dump(exclude={"messages"})
                or instructions(template) != instructions(final_request)
                or before_decision.get("max_output_tokens") != after_decision.get("max_output_tokens")
                or final_limit != hard_limit or fitter._calculate_budget(None, provider) != policy_limit):
            raise CompactionError("recovery_request_changed", "Recovery verification changed the request envelope, output reservation or hard allowance.")
        if after >= before:
            raise CompactionError("native_no_reduction", "The recovered ordinary request did not strictly reduce the effective prior request.")
        # The authoritative count proves reduction; the conservative estimate
        # still owns hard safety, exactly as in ordinary RequestContext fitting.
        if after_decision["estimated_input_tokens"] > limit:
            raise CompactionError("native_input_oversized", "The recovered ordinary request still exceeds its unchanged hard allowance.")
        verification = {"status": "verified", "before": before, "after": after,
                        "limit": limit, "throughMessage": end, "sha256": record["sha256"]}
        validated.checkpoint_status = {"status": "ready", "throughMessage": end, "originalsAvailable": True}
        activation = {key: copy.deepcopy(getattr(validated, key)) for key in fields}
        final_snapshot = copy.deepcopy(final_envelope)
        acknowledged = await _guarded(check, install, copy.deepcopy(record), copy.deepcopy(verification))
        if isinstance(acknowledged, dict) and set(acknowledged) == {"sha256"}:
            acknowledged = acknowledged.get("sha256")
        if type(acknowledged) is not str or acknowledged != record["sha256"]:
            raise CompactionError("recovery_install_unconfirmed", "The host did not acknowledge this exact checkpoint; active state is unchanged and storage needs reconciliation.")
        check()
        if final_envelope != final_snapshot:
            raise CompactionError("recovery_request_changed", "The measured request changed during checkpoint installation; reconcile the stored checkpoint before continuing.")
        # No await and no destructive global restore between approval and activation.
        for key, value in activation.items():
            setattr(context, key, value)
        return {"base_view": final_view, "final_attempt": final_envelope,
                "outcome": "recovered", "measured_before": before, "measured_after": after,
                "policy_budget": limit, "trigger": limit, "target": limit,
                "count_calls": 2, "transaction": None,
                "recovery": {**verification, "status": "installed"}}
    finally:
        context.lock.release()
