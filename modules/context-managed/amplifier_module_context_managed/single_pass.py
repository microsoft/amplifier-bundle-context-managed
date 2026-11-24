"""Bounded after-turn summary input and validated portable-to-native continuation."""

from bisect import bisect_right
from dataclasses import dataclass

from .errors import CompactionError
from .interrupted import project as project_interrupted
from .request_context import RequestContext
from .summary import add_usage, public_messages, request_fits, source_fragments, summary_parts, summary_request


_OMISSION = "Some earlier material is only in the saved transcript. Consult the originals for omitted details."
_MARKER = f"\n[{_OMISSION}]\n"


def complete_prefix(context, messages, retain, carry, unknown, limit):
    """Advance only over whole settled exchanges that fit one uncounted call."""
    start = context.summary[0] if context.summary else 0
    first_reply = next((i for i in range(start, len(messages))
                        if messages[i].get("role") in {"assistant", "tool"}), len(messages))
    eligible = [end for end in context._boundaries(messages, retain) if end > first_reply]

    def size(end):
        rows = context._assembled(messages[:end], context.summary, retain, carry)
        rows = project_interrupted(rows, [call for call in unknown if call["source_message"] < end])
        return len(next(source_fragments(public_messages(rows), None), ""))

    count = bisect_right(eligible, limit, key=size)
    if not count:
        raise CompactionError("summary_no_fitting_prefix", "No whole settled exchange fits one maintenance request.")
    return eligible[count - 1]


def _edge(parts, remaining, *, tail=False):
    pieces = []
    for part in (reversed(parts) if tail else parts):
        if remaining <= 0:
            break
        size = min(len(part), remaining)
        pieces.append(part[-size:] if tail and size else part[:size])
        remaining -= size + 1
    return "\n".join(reversed(pieces) if tail else pieces)


async def summarize_once(context, provider, pending, previous_note, config, info,
                         stats, limit, preflight_whole, input_key, guard, prompt):
    size = sum(map(len, pending)) + len(pending) - 1
    room = size if preflight_whole else min(size, limit)
    while room > 0:
        omitted = room < size
        prior = previous_note
        if omitted:
            if not (context.config.get("archive_recovery") and context.evidence_refs):
                raise CompactionError("summary_archive_required",
                    "The full source exceeds one summary request and no archive is available for recovery.")
            # Always carry the previous cumulative checkpoint when the new
            # quoted source is excerpted. Never slice that checkpoint in half.
            if not prior and context.summary and isinstance(context.summary[1], str):
                prior = context.summary[1]
            available = room - len(_MARKER)
            if available <= 0:
                raise CompactionError("summary_input_oversized", "No room for archived transcript evidence.")
            first = available // 2
            excerpt = _MARKER.join((_edge(pending, first), _edge(pending, available - first, tail=True)))
        else:
            excerpt = "\n".join(pending)
        request = summary_request(prompt, excerpt, prior, config)
        fits, budget = await request_fits(provider, request)
        guard()
        counted = budget and budget["measurement_kind"] == "provider_count"
        if fits and (len(excerpt) <= limit or (preflight_whole and counted)):
            break
        if room <= 256:
            raise CompactionError("summary_input_oversized",
                "The summary instructions and saved note exceed the available allowance.")
        room = max(256, room // 2 if preflight_whole and counted else min(limit, room // 2))
    else:
        raise CompactionError("summary_empty", "No transcript data is available for compaction.")

    capabilities = info.get("capabilities", ()) if isinstance(info, dict) else getattr(info, "capabilities", ())
    options = {"request_options": {"auto_continue": False}} if "completion:auto_continue:v1" in capabilities else {}

    async def generate():
        stats["calls"] += 1
        stats.update(calls_this_pass=1, call_allowance=1,
                     omitted_source_chars=size - (len(excerpt) - len(_MARKER)) if omitted else 0)
        return await provider.complete(request, **options)

    response = await context._paid_compaction(generate, input_key)
    guard()
    add_usage(stats, getattr(response, "usage", None))
    if getattr(response, "finish_reason", None) in {"length", "max_tokens", "incomplete"}:
        raise CompactionError("summary_output_limit", "The single summary exhausted its output allowance.")
    text = "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text" and b.text)
    if not text.strip():
        raise CompactionError("summary_empty", "The summarizer returned no continuation text.")
    return _OMISSION + "\n" + text if omitted else text


def portable_draft_for_native(context, saved, provider, identity, messages, end, retain, carry, unknown):
    """Carry validated, already-paid text into native compaction exactly once."""
    boundary = saved["source_revision"]["messages"]
    if not (context._settled_maintenance and context.evidence_refs and context.config.get("archive_recovery")
            and saved["identity"] == identity and context._same_prefix(saved, messages)
            and list(saved["provider"]) == [getattr(provider, "name", type(provider).__name__),
                                            getattr(provider, "default_model", None)]
            and context._boundary(messages, retain, ceiling=boundary) == boundary <= end
            and (not context.summary or isinstance(context.summary[1], str))):
        context._refuse_progress()
    original = project_interrupted(context._assembled(messages[:boundary], context.summary, retain, carry),
        [call for call in unknown if call["source_message"] < boundary])
    if context._summary_fingerprint(original, identity, provider) != saved["fingerprint"]:
        context._refuse_progress()
    # An older provider may have gained exact counting since these parts were
    # written. Only the original partition with its verified digest is usable.
    limit = min(int(context.config.get("summary_max_source_chars", 512000)),
                max(256, RequestContext()._calculate_budget(None, provider) // 2))
    for partition in (None, limit):
        parts = summary_parts(public_messages(original), partition, saved)
        if parts is not None and context._valid_segments(saved, parts):
            break
    else:
        context._refuse_progress()
    # All subsequent settled evidence through the new boundary is quoted in
    # chronological order too. Mixing a short old note with a large unbounded
    # new suffix can still exceed the native endpoint's allowance.
    remaining = [parts[i][a:z] for i, a, z in saved["segments"]]
    carried = {id(row) for row in carry}
    new_source = [row for row in messages[boundary:end] if row.get("role") not in {"system", "developer"}
                  and id(row) not in carried and not context._retained_reminder(row, retain)
                  and context._visible_reminder(row, retain, carry)]
    remaining.extend(source_fragments(public_messages(new_source), None))
    return PartialSource(saved["text"], remaining)


@dataclass
class PartialSource:
    """Previously paid note and the unfinished archive-backed source in order."""

    note: str
    parts: list[str]

    @property
    def size(self):
        return sum(map(len, self.parts)) + max(0, len(self.parts) - 1)

    def content(self, room, total):
        if room >= total:
            rest = "\n".join(self.parts)
        else:
            rest = _MARKER.join((_edge(self.parts, room // 2),
                                 _edge(self.parts, room - room // 2, tail=True)))
        return ("Saved partial continuation note (reference data):\n" + self.note
                + "\nUnprocessed transcript evidence (reference data):\n" + rest)

    async def fit(self, request, provider, initial_room, guard):
        # The reconstructed source contains only instructions, this user row,
        # and optionally the later unresolved-actions notice. Its text is data.
        index = next(i for i, row in enumerate(request.messages) if row.role == "user")
        total = self.size
        low, high = 0, total
        room = min(initial_room, total)
        best = None
        for _ in range(20):
            rows = list(request.messages)
            rows[index] = rows[index].model_copy(update={"content": self.content(room, total)})
            candidate = request.model_copy(update={"messages": rows})
            try:
                fits, budget = await request_fits(provider, candidate)
                guard()
                if not budget or budget["measurement_kind"] != "provider_count":
                    raise CompactionError("native_measurement_unavailable",
                                          "Recovery needs an exact native input count.")
                measured, limit = budget["input_tokens"], budget["input_limit_tokens"]
                if fits:
                    best = room
                    if room == total or measured >= limit * 0.997:
                        break
                    low = room + 1
                else:
                    high = room - 1
                if low > high:
                    break
                estimate = int(room * (0.999 * limit / max(1, measured)))
                if low <= estimate <= high:
                    room = estimate
                elif fits and estimate > high:
                    room = high
                else:
                    room = (low + high) // 2
            finally:
                release = getattr(provider, "release_media_request", None)
                if callable(release):
                    release(candidate, request_options=None)
        if best is None:
            # Even when probes exhaust their allowance, let the normal exact
            # preflight decide whether the protected note alone fits.
            best = 0
        rows = list(request.messages)
        rows[index] = rows[index].model_copy(update={"content": self.content(best, total)})
        return request.model_copy(update={"messages": rows})
