"""Explicit, offline native recovery of oversized legacy history.

This is never called by ordinary request preparation or as a failure fallback.
The caller owns exclusive history access and installation of the final derived
checkpoint. No foreground inference or tools run here. Every source message is
covered in order; provider-returned windows are kept intact.
"""
import copy
import inspect
import json

from amplifier_core import ChatRequest, Message

from .checkpoint import digest
from .errors import CompactionError
from .summary import request_fits


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
            try:
                receipt = json.loads(row.get("content", ""))
            except (TypeError, ValueError):
                receipt = None
            if not (isinstance(receipt, dict) and receipt.get("status") in {"queued", "pending"} and receipt.get("job_id")):
                pending.discard(row.get("tool_call_id"))
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    pending.discard(block.get("tool_use_id"))
        if not pending and index + 1 > start:
            boundaries.append(index + 1)
    return boundaries


async def recover_native(context, provider, template: ChatRequest, identity, *,
                         save_progress, target_bytes=1_500_000, progress=None):
    """Build a derived checkpoint, using a compatible saved prefix if supplied.

    Byte size chooses a tentative batch only. Authoritative provider preflight
    must accept each exact request. Oversize batches shrink at settled exchange
    boundaries; even the smallest rejected batch fails without dropping input.
    """
    if not provider.supports_native_compaction():
        raise CompactionError("native_checkpoint_invalid", "Explicit native recovery requires native support.")
    if type(target_bytes) is not int or target_bytes < 1:
        raise ValueError("target_bytes must be positive")
    source = copy.deepcopy(context.messages)
    source_digest = digest(source)
    if progress is not None:
        state = context.restore_checkpoint(progress, identity)
        if state["status"] != "restored" or not isinstance(context.summary[1], dict):
            raise CompactionError("native_checkpoint_invalid", "Recovery progress is incompatible with this history or model.")
    elif context.summary is not None:
        raise ValueError("Pass the existing checkpoint explicitly as recovery progress")
    start, previous = context.summary if context.summary else (0, None)
    if previous and not provider.validate_compacted_context(previous):
        raise CompactionError("native_checkpoint_invalid", "The provider rejected the recovery checkpoint.")
    boundaries = settled_boundaries(source, start)
    if start < len(source) and (not boundaries or boundaries[-1] != len(source)):
        raise CompactionError("native_checkpoint_invalid", "History contains an unresolved tool exchange; resolve its actual outcome before recovery.")
    while start < len(source):
        candidates = [end for end in boundaries if end > start]
        total = 0
        end = candidates[0]
        allowed = set(candidates)
        for index in range(start, len(source)):
            total += len(json.dumps(source[index], ensure_ascii=False).encode())
            if index + 1 in allowed:
                if total > target_bytes and index + 1 != candidates[0]:
                    break
                end = index + 1
        candidate_index = candidates.index(end)
        while True:
            rows = ([copy.deepcopy(previous)] if previous else []) + copy.deepcopy(source[start:end])
            request = template.model_copy(deep=True, update={
                "messages": [Message(**row) for row in rows],
                "metadata": {**(template.metadata or {}), "purpose": "context-compaction",
                             "stream": False, "native_compaction_request_context": True}})
            fits, decision = await request_fits(provider, request)
            if decision is None or decision.get("measurement_kind") != "provider_count":
                raise CompactionError("native_measurement_unavailable", "Recovery requires authoritative native input measurement.")
            if fits:
                break
            if candidate_index == 0:
                raise CompactionError("native_input_oversized", "The prior native window plus the next complete exchange exceeds the allowance. Recovery stopped with its last checkpoint preserved.")
            candidate_index //= 2
            end = candidates[candidate_index]
        result = await provider.compact_context(request)
        message = result.get("message")
        if result.get("kind") != "native" or not isinstance(message, dict) or not provider.validate_compacted_context(message):
            raise CompactionError("native_checkpoint_invalid", "Recovery returned an invalid native checkpoint.")
        if digest(context.messages) != source_digest:
            raise RuntimeError("History changed during recovery; new checkpoint was not installed")
        context.summary = (end, copy.deepcopy(message))
        context.summary_identity = copy.deepcopy(identity)
        record = context.export_checkpoint(identity)
        saved = save_progress(record, {"fromMessage": start, "throughMessage": end,
            "totalMessages": len(source), "inputTokens": decision["input_tokens"],
            "inputLimit": decision["input_limit_tokens"]})
        if inspect.isawaitable(saved):
            await saved
        start, previous = end, message
    return context.export_checkpoint(identity)
