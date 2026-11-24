"""Zero-paid projection of a host-authorized completed plain-text checkpoint."""
import copy

from .checkpoint import handoff_summary


async def prepare(context, offered, target, provider, messages, revision, token_budget,
                  retain, system_prompt, operations):
    from .boundary import BoundaryContextManager

    record, stage = offered
    if not isinstance(record, dict) or not isinstance(record.get("summary"), dict):
        return None
    source = record.get("identity", {})
    if (not target or not isinstance(source, dict) or source.get("provider") != target["provider"]
            or source.get("model") == target["model"] or record.get("progress") or record.get("recovery")
            or not isinstance(record["summary"].get("text"), str)):
        return None

    scratch = BoundaryContextManager(copy.deepcopy(context.config))
    scratch.messages = messages
    if scratch.restore_checkpoint(record, source)["status"] != "restored":
        return None
    if context.summary and (context.summary_identity != source or context.summary != scratch.summary):
        return None
    if not isinstance(scratch.summary[1], str) or not handoff_summary(scratch, provider, target,
            source_identity=source, summary=scratch.summary, omitted=[], retain=retain, ceiling=len(messages)):
        return None

    active = copy.deepcopy(context._checkpoint_state())
    def guard():
        context._check_stop()
        if (context.revision != revision or context.config != scratch.config
                or context.checkpoint_identity() != target or context._checkpoint_state() != active
                or context._summary_progress or context._restored_progress or context._ordinary_transition
                or (context.active_operations and context.active_operations() != operations)):
            raise RuntimeError("Request changed during portable checkpoint handoff")

    guard()
    scratch.checkpoint_identity = lambda: target
    scratch.summary_identity = copy.deepcopy(target)
    scratch.factory = context.factory
    # Never purchase a replacement while success of the target request is unknown.
    fitter, protected = await scratch._prepare(None, token_budget, retain,
                                               snapshot=(system_prompt, operations))
    guard()
    candidate = scratch.export_checkpoint(target)
    if not candidate or not scratch.summary:
        return None
    note = next(row["content"] for i, row in scratch._assembly_rows(
        messages, scratch.summary, retain, scratch._carry(messages), []) if i is None)

    def stage_request():
        guard()
        stage(candidate, note, guard)

    fitter._transition_guard = guard
    fitter._transition_commit = stage_request
    return fitter, protected
