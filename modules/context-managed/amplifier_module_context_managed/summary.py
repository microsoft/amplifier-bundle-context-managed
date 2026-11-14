"""Bounded public conversation data for semantic continuation notes.

This is not Responses API native compaction. Opaque provider state and private
reasoning belong to the canonical transcript, never to a plain-text summary.
"""

import copy
import inspect
import json
from decimal import Decimal, InvalidOperation

from amplifier_core import ChatRequest, Message

from .errors import CompactionError


def _has_media_refs(rows):
    for row in rows:
        metadata = row.get("metadata") if isinstance(row, dict) else getattr(row, "metadata", None)
        if isinstance(metadata, dict) and "experimental.media.v1" in metadata:
            # Only the exact empty envelope is known not to contain refs.
            # Malformed or unknown shapes must not bypass media admission.
            if metadata["experimental.media.v1"] != {"refs": []}:
                return True
    return False


def _valid_media_witness(evidence):
    if not isinstance(evidence, dict):
        return False
    digest = evidence.get("projection_digest")
    return (type(evidence.get("v")) is int and evidence["v"] == 1
            and evidence.get("validated") is True and isinstance(digest, str)
            and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest))


def _text_fallback_messages(evidence):
    """Only consume the owner's detached text count view, never private pixels."""
    if not _valid_media_witness(evidence):
        return None
    rows = evidence.get("messages")
    if not isinstance(rows, list) or not rows:
        return None
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"role", "content", "name", "tool_call_id"}:
            return None
        if (row.get("role") not in {"system", "developer", "user", "assistant", "tool", "function"}
                or any(row.get(key) is not None and not isinstance(row[key], str)
                       for key in ("name", "tool_call_id"))):
            return None
        content = row.get("content")
        if not isinstance(content, str):
            if not isinstance(content, list) or any(
                    not isinstance(block, dict) or set(block) - {"type", "text", "visibility"}
                    or block.get("type") != "text" or not isinstance(block.get("text"), str)
                    or block.get("visibility") not in (None, "internal", "developer", "user")
                    for block in content):
                return None
    return copy.deepcopy(rows)


async def require_media_budget(provider, request, decision, *, messages=(), request_options=None):
    """Require an exact witness; return a detached text-only fallback count view.

    The public text projection is useful for semantic pressure, not hard media
    safety. This module neither resolves references nor substitutes a different
    request after counting. The host alone can freeze unavailable text privately
    at the original positions, while the canonical request keeps its refs.
    """
    rows = list(messages)
    if isinstance(request, ChatRequest):
        rows.extend(request.messages)
    if not _has_media_refs(rows):
        return
    valid = (isinstance(decision, dict)
             and type(decision.get("estimated_input_tokens")) is int
             and decision["estimated_input_tokens"] >= 0
             and type(decision.get("input_limit_tokens")) is int
             and decision["input_limit_tokens"] > 0)
    witness = getattr(provider, "media_budget_witness", None)
    if valid and isinstance(request, ChatRequest) and callable(witness):
        try:
            evidence = (witness(request, request_options=request_options)
                        if request_options is not None else witness(request))
            if inspect.isawaitable(evidence):
                evidence = await evidence
        except Exception:
            evidence = None  # Do not expose private request data in diagnostics.
        if _valid_media_witness(evidence):
            return
    # A warning-only private projection needs no image count. Its owner must
    # validate exact canonical/options/authority identity and freeze that same
    # private candidate for dispatch. Count the returned text, not canonical refs.
    fallback = getattr(provider, "media_text_fallback_witness", None)
    if decision is None and isinstance(request, ChatRequest) and callable(fallback):
        try:
            evidence = fallback(request, request_options=request_options)
            if inspect.isawaitable(evidence):
                evidence = await evidence
        except Exception:
            evidence = None
        rows = _text_fallback_messages(evidence)
        if rows is not None:
            return rows
    raise CompactionError(
        "media_budget_unavailable",
        "Media references require an exact media_budget_witness with a valid budget, "
        "or an owner-validated media_text_fallback_witness for the private warning-only projection. "
        "Restore that adapter or supply a position-preserving unavailable-text request before counting; "
        "a text-only estimate cannot admit unresolved media.")


def add_usage(stats, usage):
    """Preserve independent cache buckets and only sum known provider costs."""
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return
    stats["usage_calls"] = stats.get("usage_calls", 0) + 1
    totals = stats.setdefault("usage", {})
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    ):
        value = usage.get(key)
        if type(value) is int and value >= 0:
            totals[key] = totals.get(key, 0) + value
    try:
        cost = Decimal(str(usage.get("cost_usd")))
        if cost.is_finite() and cost >= 0:
            totals["cost_usd"] = str(Decimal(totals.get("cost_usd", "0")) + cost)
            stats["priced_calls"] = stats.get("priced_calls", 0) + 1
    except InvalidOperation:
        pass


def public_messages(messages):
    """Keep task evidence without serializing duplicate or opaque wire state."""
    result = []
    for message in messages:
        row = {
            key: copy.deepcopy(message[key])
            for key in ("role", "name", "tool_call_id")
            if key in message
        }
        content = message.get("content")
        calls = message.get("tool_calls") or []
        if isinstance(content, list):
            blocks = []
            call_ids = set()
            for block in content:
                kind = block.get("type")
                if kind in {"thinking", "reasoning", "redacted_thinking"}:
                    continue
                if kind in {"text", "tool_call", "tool_use", "tool_result"}:
                    blocks.append(copy.deepcopy(block))
                    if kind in {"tool_call", "tool_use"}:
                        call_ids.add(block.get("id"))
                else:
                    # Binary media is available in originals. Encoding it as
                    # text neither describes the image nor preserves its meaning.
                    blocks.append(
                        {
                            "type": "text",
                            "text": f"[{kind or 'media'} content retained in original transcript]",
                        }
                    )
            row["content"] = blocks
            calls = [call for call in calls if call.get("id") not in call_ids]
        else:
            row["content"] = copy.deepcopy(content)
        if calls:
            row["tool_calls"] = copy.deepcopy(calls)
        metadata = message.get("metadata") or {}
        provenance = {
            key: copy.deepcopy(metadata[key])
            for key in ("amplifier_input", "source", "ephemeral", "persisted")
            if key in metadata
        }
        if provenance:
            row["metadata"] = provenance
        result.append(row)
    return result


def source_fragments(messages, max_chars):
    """Yield ordered, bounded fragments, without changing canonical messages.

    Fragments are *quoted source data*, not tool protocol messages. A large
    record may span fragments; indexes let the summarizer join its evidence.
    Only the completed note is eligible for commit at a safe turn boundary.
    A missing character limit prepares one complete prefix for request preflight.
    """
    if max_chars is None:
        text = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                         for row in messages)
        if text:
            yield text
        return
    pending = []
    size = 0
    for index, row in enumerate(messages):
        text = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        if len(text) > max_chars:
            if pending:
                yield "\n".join(pending)
                pending, size = [], 0
            total = (len(text) + max_chars - 1) // max_chars
            for part, start in enumerate(range(0, len(text), max_chars), 1):
                yield (
                    f"Source record {index}, fragment {part}/{total}:\n"
                    + text[start : start + max_chars]
                )
        else:
            if pending and size + len(text) + 1 > max_chars:
                yield "\n".join(pending)
                pending, size = [], 0
            pending.append(text)
            size += len(text) + 1
    if pending:
        yield "\n".join(pending)


def summary_request(prompt, source, previous, config):
    content = (
        "Previous continuation note (reference data):\n" + previous + "\n\n"
        if previous
        else ""
    )
    content += (
        "Next ordered conversation data. Incorporate this evidence into the continuation note:\n"
        + source
    )
    return ChatRequest(
        messages=[
            Message(role="system", content=prompt),
            Message(role="user", content=content),
        ],
        model=config.get("summarization_model"),
        max_output_tokens=config.get("summary_max_output_tokens", 8192),
        reasoning_effort=config.get("summary_reasoning_effort", "low"),
        stream=False,
        metadata={"purpose": "context-compaction", "stream": False},
    )


async def request_fits(provider, request, *, request_options=None, estimate_tokens=None, fallback_budget=None):
    """Use the provider's assembled-request budget when it supports one."""
    check = getattr(provider, "request_budget", None)
    decision = None
    if callable(check):
        kwargs = {"request_options": request_options} if request_options is not None else {}
        decision = check(request, context_estimate=len(request.messages[-1].content), **kwargs)
        if inspect.isawaitable(decision):
            decision = await decision
    fallback_rows = await require_media_budget(provider, request, decision, request_options=request_options)
    if fallback_rows is not None:
        if estimate_tokens is None or type(fallback_budget) is not int or fallback_budget <= 0:
            raise CompactionError("media_budget_unavailable",
                                  "The warning-only projection needs the caller's text-budget validation.")
        count = estimate_tokens(fallback_rows)
        return count <= fallback_budget, {
            "input_tokens": count, "input_limit_tokens": fallback_budget,
            "output_reserve": request.max_output_tokens, "measurement_kind": "public_estimate"}
    if not isinstance(decision, dict):
        return True, None
    measured = decision.get("estimated_input_tokens")
    limit = decision.get("input_limit_tokens")
    if type(measured) is not int or measured < 0 or type(limit) is not int or limit <= 0:
        return True, None
    measurement = decision.get("measurement")
    counted = (isinstance(measurement, dict) and measurement.get("kind") == "provider_count"
               and isinstance(measurement.get("source"), str) and bool(measurement["source"])
               and type(measurement.get("input_tokens")) is int
               and measurement["input_tokens"] == measured)
    return measured <= limit, {
        "input_tokens": measured,
        "input_limit_tokens": limit,
        "output_reserve": request.max_output_tokens,
        "measurement_kind": "provider_count" if counted else "provider_estimate",
    }
