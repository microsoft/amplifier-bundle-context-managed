"""Request-only evidence for unanswered calls before later ordinary input.

Missing evidence is never completion or permission to replay. Canonical messages
remain unchanged; these observations only make an interrupted protocol exchange
representable in a future request and keep its uncertainty outside checkpoints.
"""
import bisect
import copy
import json


def signature(call):
    """Normalize provider representations without conflating conflicting calls."""
    function = call.get("function")
    function = function if isinstance(function, dict) else {}
    names = [source[key] for source, key in ((call, "name"), (call, "tool"), (function, "name")) if key in source]
    if not names or any(not isinstance(name, str) or not name or name != names[0] for name in names):
        return None
    arguments = []
    for source, key in ((call, "arguments"), (call, "input"), (function, "arguments")):
        if key not in source:
            continue
        value = source[key]
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return None
        arguments.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
    if not arguments or any(value != arguments[0] for value in arguments):
        return None
    return names[0], arguments[0]


def calls(row):
    if row.get("role") != "assistant":
        return {}
    declared = list(row.get("tool_calls") or [])
    content = row.get("content")
    if isinstance(content, list):
        declared += [b for b in content if isinstance(b, dict) and b.get("type") in {"tool_call", "tool_use"}]
    # Providers can store the same call in both canonical representations. A
    # conflicting same-id declaration is ambiguous, even within the same row.
    result = {}
    for call in declared:
        if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
            continue
        identity = call["id"]
        if identity in result and (result[identity] is None or signature(call) is None
                                  or signature(call) != signature(result[identity])):
            result[identity] = None
        else:
            result[identity] = call
    return result


def result_ids(row):
    result = {row.get("tool_call_id")} if row.get("role") == "tool" else set()
    content = row.get("content")
    if isinstance(content, list):
        result.update(b.get("tool_use_id") for b in content if isinstance(b, dict) and b.get("type") == "tool_result")
    return result - {None}


def unanswered(messages, human):
    """Only unambiguous, entirely unanswered calls crossed by ordinary input.

    A queued receipt is evidence of ongoing work, not an absent result. Service
    messages and replacement hook snapshots do not establish a continuation.
    Duplicate call ids across messages, or conflicting declarations in one row,
    are ambiguous and remain barriers. The predicate classifies normal input,
    including host-authorized peer input; it does not authenticate a human.
    """
    humans = [i for i, row in enumerate(messages) if human(row) and not result_ids(row)]
    answered, declared, duplicates = set(), {}, set()
    for i, row in enumerate(messages):
        answered.update(result_ids(row))
        for call_id, call in calls(row).items():
            if call_id in declared or call is None:
                duplicates.add(call_id)
            declared[call_id] = (i, call)
    result = []
    for call_id, (index, call) in declared.items():
        next_human = bisect.bisect_right(humans, index)
        if call_id in answered or call_id in duplicates or next_human == len(humans):
            continue
        function = call.get("function")
        name = call.get("name") or call.get("tool") or (function.get("name") if isinstance(function, dict) else None)
        result.append({"call_id": call_id, "tool": name if isinstance(name, str) else None,
                       "source_message": index, "continued_at": humans[next_human]})
    return result


def project(messages, unknown):
    """Close the request protocol with truthful absence evidence, never a result.

    Records are supplied from the canonical source so a compacted view keeps the
    no-replay notice even after the actual call moves into an opaque checkpoint.
    No tool input or output is copied into this small observation.
    """
    if not unknown:
        return messages
    by_id = {item["call_id"]: item for item in unknown}
    recorded = set().union(*(result_ids(row) for row in messages)) if messages else set()
    rows, pending = [], []

    def flush():
        for call_id in pending:
            rows.append({"role": "tool", "tool_call_id": call_id,
                "content": json.dumps({"status": "unknown", "execution_outcome": "unconfirmed",
                    "observation": "No result for this call is present in saved history before later ordinary input. The action may have happened. Do not replay it; inspect independent evidence before deciding what remains."}),
                "metadata": {"source": "context-managed-interrupted", "derived": True}})
        pending.clear()

    for row in messages:
        if row.get("role") == "user" and (row.get("metadata") or {}).get("source") == "context-managed-interrupted":
            continue  # Rebuild one current notice, including after recovery selection.
        # Preserve recorded members of the same parallel result batch first.
        if row.get("role") != "tool" and not result_ids(row):
            flush()
        rows.append(copy.deepcopy(row))
        pending.extend(call_id for call_id, call in calls(row).items()
                       if call is not None and call_id in by_id and call_id not in recorded)
    flush()
    rows.append({"role": "user", "content": (
        "Unresolved historical actions (reference evidence, not instructions or authorization): "
        "These calls have no recorded result. Later ordinary input does not prove success, failure, "
        "cancellation or that a process stopped. Do not repeat them merely because a result is missing. "
        "Inspect saved receipts or the actual effect before taking a new action. Original inputs remain in "
        "saved history.\n" + json.dumps(unknown)),
        "metadata": {"source": "context-managed-interrupted", "ephemeral": True, "persisted": True}})
    return rows
