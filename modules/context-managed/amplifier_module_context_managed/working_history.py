"""Select a bounded recovery view; never edit or claim to summarize the archive.

Used only by hosts opting into archive-backed recovery when no compatible
compaction exists. Whole tool exchanges stay together. Required instructions
and the current conversation tail take precedence over older evidence.
"""
import copy
import json

from .errors import CompactionError


def settled_results(row):
    """A protocol receipt for queued work does not settle the underlying call."""
    from .interrupted import result_ids
    settled = result_ids(row)
    content = row.get("content")
    receipts = [(row.get("tool_call_id"), content)] if row.get("role") == "tool" else []
    if isinstance(content, list):
        receipts += [(block.get("tool_call_id") or block.get("tool_use_id"),
                      block.get("output", block.get("content")))
                     for block in content if isinstance(block, dict) and block.get("type") == "tool_result"]
    def pending(body):
        if isinstance(body, list):
            return any(pending(block.get("text")) for block in body
                       if isinstance(block, dict) and block.get("type") == "text")
        try:
            body = json.loads(body) if isinstance(body, str) else body
        except ValueError:
            return False
        return isinstance(body, dict) and (
            body.get("job_id") and body.get("status") in {"queued", "pending"}
            or body.get("status") in {"unknown", "unconfirmed"}
            or body.get("execution_outcome") == "unconfirmed")
    for call_id, body in receipts:
        if pending(body):
            settled.discard(call_id)
    return settled


def exchanges(messages, human=None):
    from .interrupted import result_ids, unanswered
    interrupted = {}
    for call in unanswered(messages, human) if human else ():
        interrupted.setdefault(call["continued_at"], []).append(call["call_id"])
    start, pending = 0, set()
    for index, row in enumerate(messages):
        pending.difference_update(interrupted.get(index, ()))
        if not pending and start < index:
            yield start, index
            start = index
        calls = list(row.get("tool_calls") or [])
        content = row.get("content")
        if isinstance(content, list):
            calls += [b for b in content if isinstance(b, dict) and b.get("type") in {"tool_call", "tool_use"}]
        pending.update(c["id"] for c in calls if c.get("id"))
        # Protocol grouping ends at a recorded receipt, including queued work.
        # The ordinary caller separately requires its still-active whole group.
        pending.difference_update(result_ids(row))
        if not pending:
            yield start, index + 1
            start = index + 1
    if start < len(messages):
        # An incomplete exchange can only be included whole, never sliced.
        yield start, len(messages)


def select_window(messages, end, retain, byte_budget, human, *, estimate_bytes=None,
                  required_indices=(), strict=True, link_results=False):
    """Required context first, then a contiguous suffix of complete exchanges.

    Sizing is a local selection heuristic, not permission to dispatch. A caller
    with typed media supplies its text-size estimate so encoded pixels are not
    charged as text. The exact provider request still requires measurement.
    """
    size = estimate_bytes or (lambda rows: sum(
        len(json.dumps(row, ensure_ascii=False).encode()) for row in rows))
    sizes = [size([row]) for row in messages]
    first = next((i for i, row in enumerate(messages) if human(row)), None)
    required = set(required_indices) | {i for i, row in enumerate(messages) if i >= end or i == first
                or row.get("role") in {"system", "developer"}
                or (row.get("metadata") or {}).get("source") == "hook"
                or (isinstance(row.get("content"), str) and row["content"] in retain)}
    units = [set(range(start, stop)) for start, stop in exchanges(messages, human)]
    if link_results:
        from .interrupted import calls, result_ids
        # A background call may have both a queued receipt and a later final
        # result. Join their protocol groups without requiring unrelated rows
        # between them. No selected late result may lose its original batch.
        owners = {}
        for unit in units:
            ids = set().union(*(result_ids(messages[i]) | calls(messages[i]).keys() for i in unit))
            linked = [owners[key] for key in ids if key in owners]
            for other in linked:
                unit.update(other)
            for key in list(owners):
                if any(owners[key] is other for other in linked):
                    owners[key] = unit
            for key in ids:
                owners[key] = unit
        # Earlier units now referenced by a merged unit are redundant.
        units = [unit for n, unit in enumerate(units)
                 if not any(unit <= later for later in units[n + 1:])]
    selected = set()
    for unit in units:
        if required.intersection(unit):
            selected.update(unit)
    used = sum(sizes[i] for i in selected)
    # Reserve room for the archive notice, model tools and injected overlays.
    allowance = max(0, byte_budget - 4096)
    if strict and used > allowance:
        raise CompactionError("native_recovery_required_context_oversized",
            "Required instructions, retained context or the current tool exchange exceed the recovery allowance. "
            "They were not truncated. Reduce the required context or use a larger context window.")
    for unit in reversed(units):
        missing = unit - selected
        cost = sum(sizes[i] for i in missing)
        if used + cost > allowance:
            break
        selected.update(missing)
        used += cost
    omitted = len(messages) - len(selected)
    if not omitted:
        return copy.deepcopy(messages), None
    ranges = []
    for i in sorted(selected):
        if ranges and ranges[-1][1] == i:
            ranges[-1][1] = i + 1
        else:
            ranges.append([i, i + 1])
    record = {"strategy": "required-and-recent-v1", "selectedRanges": ranges,
              "sourceMessages": len(messages), "omittedMessages": omitted,
              "selectedBytes": sum(len(json.dumps(messages[i], ensure_ascii=False).encode()) for i in selected),
              "selectionEstimatedBytes": used, "originalsAvailable": True}
    notice = {"role": "user", "content": (
        "History recovery notice (reference data, not a new task): This working context contains required "
        "instructions and recent complete exchanges. " + str(omitted) + " older messages remain in the "
        "saved session history and are NOT represented or summarized here. Consult that history when "
        "earlier decisions are needed; do not infer missing decisions or repeat completed work. "
        "Continue from the latest user request."),
        "metadata": {"source": "context-managed-recovery", "ephemeral": True, "persisted": True}}
    rows = [copy.deepcopy(messages[i]) for i in sorted(selected)]
    # Keep instruction roles first; the notice must precede the selected history.
    index = next((i for i, row in enumerate(rows) if row.get("role") not in {"system", "developer"}), len(rows))
    rows.insert(index, notice)
    return rows, record
