"""Bounded public conversation data for semantic continuation notes.

This is not Responses API native compaction. Opaque provider state and private
reasoning belong to the canonical transcript, never to a plain-text summary.
"""

import copy
import inspect
import json
from decimal import Decimal, InvalidOperation

from amplifier_core import ChatRequest, Message

from .checkpoint import digest


class SummaryInputTooLarge(Exception):
    """Only a local input-fit refusal authorizes splitting summary source."""


async def summary_config(provider, config):
    """Resolve an absent-only output policy without model discovery or inference.

    Published defaults select a conservative workload profile, not a promise of
    model capability. The provider still owns the selected model's hard ceiling
    and complete assembled-request admission. In particular, a wrapper may
    advertise a foreground allowance here; only an explicit summary override
    may exceed it. An implicit policy is never raised above that advertisement.
    """
    resolved = copy.deepcopy(config)
    explicit = config.get("summary_max_output_tokens")
    if "summary_max_output_tokens" in config and (type(explicit) is not int or explicit <= 0):
        raise ValueError("summary_max_output_tokens must be a positive integer")
    defaults = {}
    info = None
    getter = getattr(provider, "get_info", None)
    if callable(getter):
        info = getter()
        if inspect.isawaitable(info):
            info = await info
        defaults = info.get("defaults", {}) if isinstance(info, dict) else getattr(info, "defaults", {})
    defaults = defaults if isinstance(defaults, dict) else {}
    model = config.get("summarization_model") or getattr(provider, "default_model", None)
    # Ignore an explicitly different model. Equality is only advisory: wrappers
    # may overlay model/output defaults while retaining an inherited window.
    bound = isinstance(model, str) and defaults.get("model") == model
    window, output = (defaults.get("context_window"), defaults.get("max_output_tokens")) if bound else (None, None)
    output = output if type(output) is int and output > 0 else None
    large = type(window) is int and window >= 131072 and output is not None and output >= 32768
    allowance = explicit if explicit is not None else (32768 if large else 8192)
    if explicit is None and output is not None:
        allowance = min(allowance, output)
    resolved["summary_max_output_tokens"] = allowance
    return resolved, info


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


def source_fragments(messages, max_chars, *, legacy_labels=False):
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
            # Reserve the longest label before slicing. Older saved partitions
            # added the label afterward, overfilling every full part.
            width = max_chars
            total = (len(text) + width - 1) // width
            if not legacy_labels:
                while True:
                    width = max_chars - len(f"Source record {index}, fragment {total}/{total}:\n")
                    if width <= 0:
                        raise ValueError("Summary fragment limit cannot hold its source label")
                    required = (len(text) + width - 1) // width
                    if required == total:
                        break
                    total = required
            for part, start in enumerate(range(0, len(text), width), 1):
                yield (
                    f"Source record {index}, fragment {part}/{total}:\n"
                    + text[start : start + width]
                )
        else:
            if pending and size + len(text) + 1 > max_chars:
                yield "\n".join(pending)
                pending, size = [], 0
            pending.append(text)
            size += len(text) + 1
    if pending:
        yield "\n".join(pending)


def summary_parts(messages, limit, saved=None):
    """A saved note can resume only its exact original partition."""
    parts = list(source_fragments(messages, limit))
    if saved is None or digest(parts) == saved["parts_digest"]:
        return parts
    if limit is not None and any(part.startswith("Source record ") for part in parts):
        del parts
        parts = list(source_fragments(messages, limit, legacy_labels=True))
        if digest(parts) == saved["parts_digest"]:
            return parts
    return None


def summary_prompt(prompt, config):
    """Render a soft target for the whole visible note, not a generation cap.

    Keep the base checkpoint prompt stable: a completed note remains useful.
    Callers bind this rendered policy separately before resuming partial work.
    """
    if "summary_target_tokens" not in config:
        return prompt
    target = config["summary_target_tokens"]
    if type(target) is not int or target <= 0:
        raise ValueError("summary_target_tokens must be a positive integer")
    return prompt + (
        "\n\nRewrite one complete cumulative continuation note that replaces the previous note; "
        "do not append a supplement or reproduce its wording. Merge repetition and compress "
        "settled work into concise outcomes with exact evidence references. "
        f"Soft target for the entire visible continuation note: approximately {target} tokens, "
        "including retained facts and new evidence, not additional tokens per fragment. "
        "This is not the generation allowance. Preserve required continuation facts; "
        "do not truncate text or discard unresolved work to meet the target."
    )


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
            Message(role="system", content=summary_prompt(prompt, config)),
            Message(role="user", content=content),
        ],
        model=config.get("summarization_model"),
        max_output_tokens=config.get("summary_max_output_tokens", 8192),
        reasoning_effort=config.get("summary_reasoning_effort", "low"),
        stream=False,
        metadata={"purpose": "context-compaction", "stream": False},
    )


async def request_fits(provider, request):
    """Use the provider's assembled-request budget when it supports one."""
    check = getattr(provider, "request_budget", None)
    if not callable(check):
        return True, None
    decision = check(request, context_estimate=len(request.messages[-1].content))
    if inspect.isawaitable(decision):
        decision = await decision
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
               # A provider may add a safety reserve to its raw count. Keep
               # that guarded estimate for admission; do not mistake the
               # valid smaller raw measurement for an estimate-only result.
               and 0 <= measurement["input_tokens"] <= measured)
    metadata = request.metadata or {}
    portable = (metadata.get("purpose") == "context-compaction"
                and "native_compaction_request_context" not in metadata)
    reserve = request.max_output_tokens
    reserve = reserve if type(reserve) is int and reserve > 0 else 0
    effective = decision.get("max_output_tokens")
    if portable and "max_output_tokens" in decision:
        if type(effective) is not int or effective <= 0:
            raise ValueError("Provider summary output reservation is invalid")
        # This decision belongs to this exact request_budget invocation. A
        # smaller effective cap can be the provider's own clamp. A larger
        # advertisement is inconsistent with the caller cap: reserve all of it
        # rather than lowering it to manufacture admission. Never change the
        # request or increase its caller-owned generation allowance.
        reserve = effective
    # Portable summaries conservatively reserve output even when a provider's
    # input allowance is independent (Anthropic) or already output-subtracted.
    # The latter can split earlier, never admit a larger request. Native
    # compaction has its own operation-specific accounting contract.
    fits = measured <= limit and (not portable or reserve > 0 and measured + reserve <= limit)
    return fits, {
        "input_tokens": measured,
        "input_limit_tokens": limit,
        "output_reserve": reserve if portable else request.max_output_tokens,
        "measurement_kind": "provider_count" if counted else "provider_estimate",
    }
