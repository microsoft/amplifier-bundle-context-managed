"""Select a bounded recovery view; never edit or claim to summarize the archive.

Used only by hosts opting into archive-backed recovery when no compatible
compaction exists. Whole tool exchanges stay together. Required instructions
and the current conversation tail take precedence over older evidence.
"""
import copy
import json

from .errors import CompactionError


def exchanges(messages):
    start, pending = 0, set()
    for index, row in enumerate(messages):
        calls = list(row.get("tool_calls") or [])
        content = row.get("content")
        if isinstance(content, list):
            calls += [b for b in content if isinstance(b, dict) and b.get("type") in {"tool_call", "tool_use"}]
        pending.update(c["id"] for c in calls if c.get("id"))
        if row.get("role") == "tool":
            pending.discard(row.get("tool_call_id"))
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    pending.discard(block.get("tool_use_id"))
        if not pending:
            yield start, index + 1
            start = index + 1
    if start < len(messages):
        # An incomplete exchange can only be included whole, never sliced.
        yield start, len(messages)


def select_window(messages, end, retain, byte_budget, human):
    """Required context first, then a contiguous suffix of complete exchanges.

    Byte sizing is a conservative local selection heuristic, not permission to
    dispatch. The caller must measure the exact provider request afterwards.
    """
    sizes = [len(json.dumps(row, ensure_ascii=False).encode()) for row in messages]
    first = next((i for i, row in enumerate(messages) if human(row)), None)
    required = {i for i, row in enumerate(messages) if i >= end or i == first
                or row.get("role") in {"system", "developer"}
                or (row.get("metadata") or {}).get("source") == "hook"
                or (isinstance(row.get("content"), str) and row["content"] in retain)}
    units = list(exchanges(messages))
    selected = set()
    for start, stop in units:
        if any(i in required for i in range(start, stop)):
            selected.update(range(start, stop))
    used = sum(sizes[i] for i in selected)
    # Reserve room for the archive notice, model tools and injected overlays.
    allowance = max(0, byte_budget - 4096)
    if used > allowance:
        raise CompactionError("native_recovery_required_context_oversized",
            "Required instructions, retained context or the current tool exchange exceed the recovery allowance. "
            "They were not truncated. Reduce the required context or use a larger context window.")
    for start, stop in reversed(units):
        missing = set(range(start, stop)) - selected
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
              "selectedBytes": used, "originalsAvailable": True}
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
