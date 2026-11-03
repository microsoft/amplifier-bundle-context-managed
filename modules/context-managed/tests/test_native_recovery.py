import copy
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message, ToolSpec
from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.checkpoint import digest
from amplifier_module_context_managed.errors import CompactionError
from amplifier_module_context_managed.recovery import recover_native, settled_boundaries

IDENTITY = {"provider": "fixture", "model": "native"}


def native_window(number):
    return {"role": "user", "content": "opaque checkpoint", "metadata": {
        "source": "context-managed", "ephemeral": True, "persisted": True,
        "canonical_window": [{"type": "compaction", "id": number}, {"type": "message", "retained": True}]}}


def fixture():
    provider = SimpleNamespace(supports_native_compaction=lambda: True,
        validate_compacted_context=lambda message: bool(message.get("metadata", {}).get("canonical_window")))
    async def count(request, **kwargs):
        size = sum(len(str(row.content)) for row in request.messages)
        return {"estimated_input_tokens": size, "input_limit_tokens": 1000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": size}}
    provider.request_budget = AsyncMock(side_effect=count)
    provider.compact_context = AsyncMock(side_effect=lambda request: {"kind": "native", "message": native_window(provider.compact_context.await_count)})
    return provider


def history():
    return [{"role": role, "content": text} for role, text in [
        ("user", "original objective " * 10), ("assistant", "progress " * 30),
        ("user", "new constraint " * 10), ("assistant", "evidence " * 30),
        ("user", "Please pause")]]


def context(rows):
    result = BoundaryContextManager({"durable_checkpoints": True})
    result.messages = copy.deepcopy(rows)
    return result


@pytest.mark.asyncio
async def test_recovery_carries_complete_windows_and_preserves_every_source_message():
    rows = history(); ctx = context(rows); provider = fixture(); saved = []
    result = await recover_native(ctx, provider, ChatRequest(messages=[], model="native"), IDENTITY,
        save_progress=lambda record, report: saved.append((record, report)), target_bytes=350)
    assert len(saved) > 1
    covered = []
    previous = None
    for call, (record, report) in zip(provider.compact_context.await_args_list, saved):
        messages = [row.model_dump(exclude_none=True) for row in call.args[0].messages]
        if previous:
            assert messages[0]["metadata"] == previous["metadata"]
            messages = messages[1:]
        covered.extend((row["role"], row["content"]) for row in messages)
        previous = record["summary"]["message"]
        assert report["inputTokens"] <= report["inputLimit"]
    assert covered == [(row["role"], row["content"]) for row in rows]
    assert ctx.messages == rows
    assert result["summary"]["throughMessage"] == len(rows)
    restored = context(rows)
    assert restored.restore_checkpoint(result, IDENTITY)["status"] == "restored"
    assert await recover_native(restored, provider, ChatRequest(messages=[]), IDENTITY,
        save_progress=lambda *args: None, progress=result) == result


@pytest.mark.asyncio
@pytest.mark.parametrize("current_roles", [
    ("system", "developer"), ("system",), ("developer",),
])
async def test_recovery_preserves_template_instructions_and_options_in_every_count_and_batch(current_roles):
    historical_instructions = [
        {"role": "system", "content": "Stale system instructions."},
        {"role": "developer", "content": "Stale developer instructions."},
    ]
    rows = historical_instructions + history() + [
        {"role": "system", "content": "Later stale system instructions."},
        {"role": "developer", "content": "Later stale developer instructions."},
    ] + history()
    current = [
        Message(role="system", content="Current system instructions.", metadata={"owner": "host"}),
        Message(role="system", content="Current system safety constraints."),
        Message(role="developer", content="Current developer instructions."),
    ]
    current = [row for row in current if row.role in current_roles]
    template = ChatRequest(
        messages=current + [Message(role="user", content="Uncommitted template overlay.")],
        model="native",
        tools=[ToolSpec(name="fixture", description="Synthetic tool.", parameters={"type": "object"})],
        tool_choice="auto", reasoning_effort="high", max_output_tokens=73,
        metadata={"owner": "loop", "options": {"retain": ["selected"]}, "stream": True},
    )
    original_template = copy.deepcopy(template.model_dump())
    ctx, provider, saved = context(rows), fixture(), []
    result = await recover_native(
        ctx, provider, template, IDENTITY, target_bytes=10_000,
        save_progress=lambda record, report: saved.append(copy.deepcopy((record, report))),
    )
    # Force authoritative rejection/shrinking as well as multiple accepted batches.
    assert len(saved) > 1
    assert provider.request_budget.await_count > provider.compact_context.await_count == len(saved)
    counts = [call.args[0] for call in provider.request_budget.await_args_list]
    batches = [call.args[0] for call in provider.compact_context.await_args_list]
    for request in counts + batches:
        assert request.model_dump(exclude={"messages", "metadata"}) == template.model_dump(
            exclude={"messages", "metadata"})
        assert request.metadata == {
            **template.metadata, "purpose": "context-compaction", "stream": False,
            "native_compaction_request_context": True,
        }
        instructions = [row.model_dump(exclude_none=True) for row in request.messages
                        if row.role in current_roles]
        assert instructions == [row.model_dump(exclude_none=True) for row in current]
        assert not any(row.content == "Uncommitted template overlay." for row in request.messages)
    for request in batches:
        assert any(request.model_dump() == counted.model_dump() for counted in counts)
    # A supplied instruction role overrides only that role, not other canonical roles.
    remaining_roles = {"system", "developer"} - set(current_roles)
    assert [(row.role, row.content) for request in batches for row in request.messages
            if row.role in remaining_roles] == [
        (row["role"], row["content"]) for row in rows if row["role"] in remaining_roles]
    through = 0
    for record, report in saved:
        assert report["fromMessage"] == through
        assert through < report["throughMessage"] <= len(rows)
        through = report["throughMessage"]
        # Superseded instructions remain canonical evidence, included in coverage.
        assert record["sourceRevision"] == {"messages": through, "sha256": digest(rows[:through])}
    assert through == result["summary"]["throughMessage"] == len(rows)
    assert ctx.messages == rows
    assert template.model_dump() == original_template


@pytest.mark.asyncio
async def test_recovery_preserves_template_empty_instruction_fallback_to_canonical_roles():
    rows = [
        {"role": "system", "content": "Canonical system instructions."},
        {"role": "developer", "content": "Canonical developer instructions."},
    ] + history()
    ctx, provider, saved = context(rows), fixture(), []
    result = await recover_native(
        ctx, provider, ChatRequest(messages=[]), IDENTITY, target_bytes=350,
        save_progress=lambda record, report: saved.append(copy.deepcopy(record)),
    )
    assert len(saved) > 1
    assert [(row.role, row.content)
            for call in provider.compact_context.await_args_list for row in call.args[0].messages
            if row.role in {"system", "developer"}] == [
        (row["role"], row["content"]) for row in rows if row["role"] in {"system", "developer"}]
    assert result["sourceRevision"] == {"messages": len(rows), "sha256": digest(rows)}
    assert ctx.messages == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", [
    "invalid_digest", "invalid_envelope", "identity", "history", "configuration",
    "portable_summary", "provider_validation",
])
async def test_rejected_recovery_progress_preserves_all_existing_active_state(rejection):
    rows = history()
    ctx, provider, save = context(rows), fixture(), AsyncMock()
    ctx.summary = (1, native_window("active"))
    ctx.summary_identity = copy.deepcopy(IDENTITY)
    ctx.evidence_refs = [{"kind": "transcript", "id": "active-evidence"}]
    ctx.recovery = {"kind": "active-recovery", "throughMessage": 1}
    source_revision = {"messages": len(rows), "sha256": digest(rows)}
    ctx.summary_failure = {
        "fingerprint": "active-failure", "source_revision": copy.deepcopy(source_revision),
        "attempts": 2, "retryable": False, "retry_after": 123,
    }
    ctx._summary_progress = {
        "fingerprint": "active-progress", "source_revision": copy.deepcopy(source_revision),
        "identity": copy.deepcopy(IDENTITY), "provider": ("fixture", "native"),
        "configuration": ctx.checkpoint_configuration(), "config": copy.deepcopy(ctx.config),
        "parts_digest": digest(["tail"]), "segments": [[0, 0, 4]], "pending": ["tail"],
        "text": "Accepted partial evidence.", "calls": 2, "completed": 1,
    }
    ctx._restored_progress = copy.deepcopy(ctx._summary_progress)
    ctx._restored_progress["text"] = "Previously restored partial evidence."
    ctx.checkpoint_status = {
        "status": "failed", "originalsAvailable": True,
        "failure": {"code": "native_input_oversized", "attempt": 2},
    }
    ctx.last_budget = {"input_tokens": 1200, "input_limit_tokens": 1000}
    ctx.revision = 7
    fields = (
        "messages", "revision", "summary", "summary_identity", "summary_failure",
        "_summary_progress", "_restored_progress", "evidence_refs", "checkpoint_status",
        "recovery", "last_budget", "is_compacting",
    )
    before = copy.deepcopy({name: getattr(ctx, name) for name in fields})
    checkpoint_before = copy.deepcopy(ctx.export_checkpoint(IDENTITY))

    # Build a distinct, structurally valid candidate before applying one rejection.
    candidate = context(rows)
    candidate.summary = (2, native_window("candidate"))
    candidate.summary_identity = copy.deepcopy(IDENTITY)
    candidate.evidence_refs = [{"kind": "transcript", "id": "candidate-evidence"}]
    candidate.recovery = {"kind": "candidate-recovery", "throughMessage": 2}
    if rejection == "invalid_envelope":
        candidate.summary[1]["content"] = ""
    elif rejection == "identity":
        candidate.summary_identity = {**IDENTITY, "model": "different-model"}
    elif rejection == "history":
        candidate.messages[0]["content"] = "A different canonical prefix."
    elif rejection == "configuration":
        candidate.config["summary_max_source_chars"] = 2048
    elif rejection == "portable_summary":
        candidate.summary = (2, "A portable note is not a native checkpoint.")
    elif rejection == "provider_validation":
        provider.validate_compacted_context = lambda message: False
    progress = candidate.export_checkpoint(candidate.summary_identity)
    if rejection == "invalid_digest":
        progress["sha256"] = "0" * 64
    original_progress = copy.deepcopy(progress)

    with pytest.raises(CompactionError, match="native_checkpoint_invalid"):
        await recover_native(
            ctx, provider, ChatRequest(messages=[]), IDENTITY,
            progress=progress, save_progress=save,
        )
    assert {name: getattr(ctx, name) for name in fields} == before
    assert ctx.export_checkpoint(IDENTITY) == checkpoint_before
    assert {name: getattr(ctx, name) for name in fields} == before
    assert progress == original_progress
    provider.request_budget.assert_not_awaited()
    provider.compact_context.assert_not_awaited()
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_failure_retains_progress_and_resumes_only_new_suffix():
    rows = history(); ctx = context(rows); provider = fixture(); saved = []
    async def compact(request):
        if provider.compact_context.await_count == 2:
            raise RuntimeError("provider unavailable")
        return {"kind": "native", "message": native_window(1)}
    provider.compact_context.side_effect = compact
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await recover_native(ctx, provider, ChatRequest(messages=[]), IDENTITY,
            save_progress=lambda record, report: saved.append(record), target_bytes=350)
    assert len(saved) == 1
    provider = fixture()
    recovered = await recover_native(context(rows), provider, ChatRequest(messages=[]), IDENTITY,
        save_progress=lambda *args: None, progress=saved[-1], target_bytes=350)
    first = provider.compact_context.await_args_list[0].args[0]
    assert first.messages[1].content == rows[saved[-1]["summary"]["throughMessage"]]["content"]
    assert recovered["summary"]["throughMessage"] == len(rows)


def test_tool_batches_are_indivisible():
    rows = [{"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "tool_call_id": "b", "content": "b"},
            {"role": "tool", "tool_call_id": "a", "content": "a"}]
    assert settled_boundaries(rows) == [3]
    assert settled_boundaries(rows[:2]) == []


@pytest.mark.asyncio
async def test_single_oversized_exchange_fails_without_discard_or_model_call():
    rows = [{"role": "user", "content": "x" * 2000}]; ctx = context(rows); provider = fixture()
    with pytest.raises(CompactionError, match="native_input_oversized"):
        await recover_native(ctx, provider, ChatRequest(messages=[]), IDENTITY, save_progress=lambda *args: None)
    provider.compact_context.assert_not_awaited()
    assert ctx.messages == rows and ctx.summary is None


@pytest.mark.asyncio
async def test_recovery_detects_concurrent_history_change_before_saving():
    rows = history(); ctx = context(rows); provider = fixture(); save = AsyncMock()
    async def mutate(request):
        ctx.messages.append({"role": "user", "content": "new correction"})
        return {"kind": "native", "message": native_window(1)}
    provider.compact_context.side_effect = mutate
    with pytest.raises(RuntimeError, match="History changed"):
        await recover_native(ctx, provider, ChatRequest(messages=[]), IDENTITY, save_progress=save)
    save.assert_not_awaited()


def test_queued_background_work_is_not_a_settled_recovery_boundary():
    rows = [{"role": "assistant", "content": "", "tool_calls": [{"id": "job"}]},
            {"role": "tool", "tool_call_id": "job", "content": '{"status":"queued","job_id":"background"}'}]
    assert settled_boundaries(rows) == []
    rows[1]["content"] = 'completed receipt'
    assert settled_boundaries(rows) == [2]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "pending"])
@pytest.mark.parametrize("payload_shape", ["json_string", "text_blocks"])
async def test_embedded_pending_tool_results_remain_unresolved_until_parallel_batch_completes(status, payload_shape):
    receipt = json.dumps({"status": status, "job_id": "background"})
    payload = receipt if payload_shape == "json_string" else [{"type": "text", "text": receipt}]
    rows = [
        {"role": "user", "content": "Original objective."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "a", "name": "fixture", "input": {}},
            {"type": "tool_use", "id": "b", "name": "fixture", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "b", "content": payload},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "Completed a."},
        ]},
        {"role": "user", "content": "Later input does not resolve the background outcome."},
    ]
    original = copy.deepcopy(rows)
    assert settled_boundaries(rows) == [1]
    assert settled_boundaries(rows, start=1) == []
    ctx, provider, save = context(rows), fixture(), AsyncMock()
    with pytest.raises(CompactionError, match="native_checkpoint_invalid"):
        await recover_native(
            ctx, provider, ChatRequest(messages=[]), IDENTITY,
            target_bytes=1, save_progress=save,
        )
    provider.request_budget.assert_not_awaited()
    provider.compact_context.assert_not_awaited()
    save.assert_not_awaited()
    assert ctx.messages == rows == original
    assert ctx.summary is None and ctx.summary_identity is None

    completed = json.dumps({"status": "completed", "job_id": "background"})
    rows[2]["content"][0]["content"] = (
        completed if payload_shape == "json_string" else [{"type": "text", "text": completed}]
    )
    # Out-of-order parallel results settle only after both calls have completed.
    assert settled_boundaries(rows[:3]) == [1]
    assert settled_boundaries(rows[:4]) == [1, 4]
    assert settled_boundaries(rows) == [1, 4, 5]
    assert settled_boundaries(rows, start=1) == [4, 5]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync_error", "async_error", "cancel", "second_error"])
async def test_failed_progress_save_restores_last_accepted_state(mode):
    rows = history()
    ctx, provider, accepted = context(rows), fixture(), []
    initial = (ctx.summary, ctx.summary_identity)
    failure = asyncio.CancelledError("save cancelled") if mode == "cancel" else OSError("save failed")

    def save(record, report):
        if mode == "second_error" and not accepted:
            accepted.append(copy.deepcopy((ctx.summary, ctx.summary_identity)))
            return
        raise failure

    async def async_save(record, report):
        await asyncio.sleep(0)
        return save(record, report)

    with pytest.raises(type(failure)) as caught:
        await recover_native(
            ctx, provider, ChatRequest(messages=[]), IDENTITY, target_bytes=350,
            save_progress=async_save if mode in {"async_error", "cancel"} else save,
        )
    assert caught.value is failure
    assert ctx.messages == rows
    assert (ctx.summary, ctx.summary_identity) == (accepted[0] if accepted else initial)
    assert provider.compact_context.await_count == (2 if accepted else 1)
    # The completed provider calls remain real; only rejected summary state rolls back.
    if accepted:
        record = ctx.export_checkpoint(IDENTITY)
        assert record["summary"]["throughMessage"] == accepted[0][0][0]
        restored = context(rows)
        assert restored.restore_checkpoint(record, IDENTITY)["status"] == "restored"
    else:
        assert ctx.export_checkpoint(IDENTITY) is None
        await recover_native(
            ctx, provider, ChatRequest(messages=[]), IDENTITY,
            save_progress=lambda *args: None, target_bytes=350,
        )
        assert ctx.summary[0] == len(rows)


@pytest.mark.asyncio
async def test_progress_export_failure_restores_prior_state(monkeypatch):
    rows = history()
    ctx, provider = context(rows), fixture()
    failure = OSError("export failed")
    save = AsyncMock()

    def reject_export(identity):
        raise failure

    monkeypatch.setattr(ctx, "export_checkpoint", reject_export)
    with pytest.raises(OSError) as caught:
        await recover_native(
            ctx, provider, ChatRequest(messages=[]), IDENTITY, save_progress=save,
        )
    assert caught.value is failure
    assert ctx.summary is None and ctx.summary_identity is None
    assert ctx.messages == rows and provider.compact_context.await_count == 1
    save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("value", [None, True, False])
async def test_progress_callback_return_value_remains_ignored(asynchronous, value):
    rows = history()
    ctx, provider, saved = context(rows), fixture(), []

    def save(record, report):
        saved.append(copy.deepcopy(record))
        return value

    async def async_save(record, report):
        return save(record, report)

    result = await recover_native(
        ctx, provider, ChatRequest(messages=[]), IDENTITY, target_bytes=350,
        save_progress=async_save if asynchronous else save,
    )
    assert result == saved[-1]
    assert result["summary"]["throughMessage"] == len(rows)
    assert ctx.messages == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
async def test_rejected_recovery_progress_must_end_at_settled_boundary(pending):
    rows = [{"role": "user", "content": "objective"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a",
             "content": '{"status":"queued","job_id":"j"}' if pending else "completed"}]
    ctx, provider = context(rows), fixture()
    ctx.summary = (1, native_window("accepted"))
    ctx.summary_identity = copy.deepcopy(IDENTITY)
    ctx.summary_failure = {"fingerprint": "preserve", "retryable": False}
    before = copy.deepcopy((ctx.summary, ctx.summary_identity, ctx.summary_failure, ctx.checkpoint_status))
    candidate = context(rows)
    candidate.summary = (len(rows) if pending else 2, native_window("invalid-boundary"))
    candidate.summary_identity = copy.deepcopy(IDENTITY)
    with pytest.raises(CompactionError, match="native_checkpoint_invalid"):
        await recover_native(ctx, provider, ChatRequest(messages=[]), IDENTITY,
                             progress=candidate.export_checkpoint(IDENTITY), save_progress=lambda *args: None)
    assert (ctx.summary, ctx.summary_identity, ctx.summary_failure, ctx.checkpoint_status) == before
    assert ctx.messages == rows
    provider.request_budget.assert_not_awaited()
    provider.compact_context.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", ["", []])
async def test_recovery_empty_final_content_reaches_authoritative_measurement(empty):
    rows = [{"role": "user", "content": "objective"}, {"role": "assistant", "content": empty}]
    ctx, provider = context(rows), fixture()
    await recover_native(ctx, provider, ChatRequest(messages=[]), IDENTITY, save_progress=lambda *args: None)
    assert provider.request_budget.await_count == provider.compact_context.await_count == 1
    assert provider.request_budget.await_args.args[0].messages[-1].content == empty
    assert ctx.messages == rows


# Explicit request recovery is a separate contract from recover_native's
# prefix-building helper. These tests do not execute the ordinary foreground loop.
REQUEST_FACTS = (
    "Objective: export ledger.",
    "Decision: CSV only.",
    "Completed: write-1; receipt R-17.",
    "Artifact: ledger.csv.",
    "Remaining: verify checksum.",
    "Constraint: never replay write-1.",
)
REQUEST_NOTE = " ".join(REQUEST_FACTS)
REQUEST_REMINDER = "Current reminder: keep secrets private."
REQUEST_LIMIT = 4000
REQUEST_STATE_FIELDS = (
    "messages", "revision", "summary", "summary_identity", "summary_failure",
    "_summary_progress", "_restored_progress", "evidence_refs",
    "checkpoint_status", "recovery", "last_budget", "is_compacting",
    "summary_omitted_indices",
)


def request_recovery_history():
    return [
        {"role": "user", "content": REQUEST_NOTE + " historical detail" * 100},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "write-1", "name": "fixture", "arguments": {"path": "ledger.csv"}}]},
        {"role": "tool", "tool_call_id": "write-1",
         "content": "Completed: receipt R-17, ledger.csv." + " verified output" * 100},
        *[{"role": "assistant", "content": f"Evidence section {i}." + " source detail" * 120}
          for i in range(3)],
        {"role": "user", "content": "Keep CSV only. Do not repeat the completed write."},
        {"role": "assistant", "content": "Checksum verification is still pending."},
        {"role": "user", "content": "Resume only after a verified checkpoint."},
        {"role": "user", "content": REQUEST_REMINDER, "metadata": {
            "ephemeral": True, "persisted": True, "reminder_placement": "tail"}},
    ]


def request_recovery_window(label="candidate"):
    message = native_window(label)
    # A real short continuation note, not an opaque placeholder whose measured
    # size merely happens to shrink. The synthetic native payload carries it too.
    message["content"] = REQUEST_NOTE
    message["metadata"]["canonical_window"] = [
        {"type": "compaction", "id": label, "summary": REQUEST_NOTE, "padding": ""}]
    return message


def request_recovery_dispatch(view):
    return ChatRequest(
        messages=[
            Message(role="system", content="Current system safety instructions."),
            Message(role="developer", content="Current developer CSV instructions."),
            *[Message(**copy.deepcopy(row)) for row in view],
            Message(role="user", content="Current request overlay: retain receipt R-17.",
                    metadata={"ephemeral": True}),
        ],
        model="native",
        tools=[ToolSpec(name="fixture", description="Completed write; do not replay.",
                        parameters={"type": "object", "properties": {
                            "path": {"type": "string"}}})],
        tool_choice="auto", reasoning_effort="high", max_output_tokens=73,
        metadata={"owner": "loop", "stream": True, "options": {"retain": ["selected"]}},
    )


def request_recovery_size(request):
    """Deterministic content-only count, including native state and envelope."""
    rows = []
    windows = []
    for row in request.messages:
        rows.append(row.model_dump(exclude_none=True, exclude={"metadata"}))
        if (row.metadata or {}).get("canonical_window"):
            windows.append(row.metadata["canonical_window"])
    return len(json.dumps({
        "messages": rows, "canonical_windows": windows,
        "options": request.model_dump(exclude={"messages", "metadata"}),
    }, sort_keys=True, separators=(",", ":")))


def request_recovery_state(ctx):
    return copy.deepcopy({name: getattr(ctx, name) for name in REQUEST_STATE_FIELDS})


def request_recovery_fixture(*, prior=True, rows=None):
    rows = request_recovery_history() if rows is None else rows
    ctx, provider = context(rows), fixture()
    if prior:
        ctx.summary = (3, request_recovery_window("accepted"))
        ctx.summary_identity = copy.deepcopy(IDENTITY)
    ctx.evidence_refs = [{"kind": "transcript", "id": "previous-evidence"}]
    ctx.summary_failure = {
        "fingerprint": "unchanged-retry-latch", "retryable": False, "attempts": 2,
        "retry_after": 123, "source_revision": {"messages": len(rows), "sha256": digest(rows)},
    }
    ctx.checkpoint_status = {"status": "failed", "originalsAvailable": True,
                             "failure": {"code": "native_input_oversized", "attempt": 2}}
    ctx.last_budget = {"input_tokens": 12000, "input_limit_tokens": REQUEST_LIMIT}
    ctx.revision = 7
    ctx.checkpoint_identity = lambda: copy.deepcopy(IDENTITY)
    ctx.preserve_evidence = AsyncMock(return_value=[
        {"kind": "transcript", "id": "preserved-originals", "sha256": digest(rows)}])
    ctx._prepare = AsyncMock(side_effect=AssertionError("Recovery must not prepare a request"))
    ctx._summarize = AsyncMock(side_effect=AssertionError("No portable fallback"))
    ctx.persist_checkpoint = AsyncMock(side_effect=AssertionError("Use explicit install ACK"))
    provider.complete = AsyncMock(side_effect=AssertionError("No foreground completion"))
    provider.stream = AsyncMock(side_effect=AssertionError("No foreground streaming"))
    state = SimpleNamespace(
        ctx=ctx, provider=provider, before=request_recovery_state(ctx),
        window=request_recovery_window(), views=[], envelopes=[], saved=[], installed=[],
        measurement_failure=None, change_contract=None, invalidated=False,
        effects={"write-1": 1},
    )
    state.checkpoint_before = copy.deepcopy(ctx.export_checkpoint(IDENTITY))
    state.baseline = ctx._assembled(copy.deepcopy(rows), copy.deepcopy(ctx.summary),
                                   (REQUEST_REMINDER,))
    state.end = ctx._boundary(rows, (REQUEST_REMINDER,))

    def guard():
        if state.invalidated:
            raise RuntimeError("Captured request contract changed")

    async def budget(request, **kwargs):
        size = request_recovery_size(request)
        return {"estimated_input_tokens": size, "input_limit_tokens": REQUEST_LIMIT,
                "measurement": {"kind": "provider_count", "source": "fixture",
                                "input_tokens": size}}

    async def count_view(view):
        assert request_recovery_state(ctx) == state.before
        request = request_recovery_dispatch(view)
        candidate = any(
            any(item.get("id") == "candidate" for item in
                (row.get("metadata") or {}).get("canonical_window", []))
            for row in view)
        if candidate and state.change_contract == "options":
            request = request.model_copy(update={"max_output_tokens": 74})
        decision = await provider.request_budget(request)
        if state.measurement_failure == ("after" if candidate else "before"):
            decision.pop("measurement")
        if candidate and state.change_contract == "allowance":
            decision["input_limit_tokens"] += 1
        envelope = {"dispatch": request, "budget_decision": decision}
        state.views.append(copy.deepcopy(view))
        state.envelopes.append(copy.deepcopy(envelope))
        return envelope

    async def compact(request):
        assert request_recovery_state(ctx) == state.before
        if state.change_contract == "guard":
            state.invalidated = True
        elif state.change_contract == "identity":
            ctx.checkpoint_identity = lambda: {**IDENTITY, "model": "changed-model"}
        elif state.change_contract == "configuration":
            ctx.config["summary_max_source_chars"] = 2048
        # Only the batch containing the eligible prefix's final source row
        # returns the final payload. Intermediate windows remain small. This
        # lets oversized/equal final-window tests reach the final request count
        # instead of failing on the next auxiliary batch.
        reaches_end = any(row.content == rows[state.end - 1]["content"]
                          for row in request.messages)
        window = state.window if reaches_end else request_recovery_window("intermediate")
        return {"kind": "native", "message": copy.deepcopy(window)}

    async def save(record, report):
        assert request_recovery_state(ctx) == state.before
        state.saved.append(copy.deepcopy((record, report)))

    async def install(record, report=None):
        assert request_recovery_state(ctx) == state.before
        state.installed.append(copy.deepcopy(record))
        return {"sha256": record["sha256"]}

    provider.request_budget = AsyncMock(side_effect=budget)
    provider.compact_context = AsyncMock(side_effect=compact)
    state.count_view = AsyncMock(side_effect=count_view)
    state.save = AsyncMock(side_effect=save)
    state.install = AsyncMock(side_effect=install)
    state.guard = guard
    # This sentinel is not an end-to-end no-replay proof; the ordinary-loop test
    # must exercise the real tool dispatcher. Never call it to seed the counter.
    async def completed_effect(*args, **kwargs):
        state.effects["write-1"] += 1
        raise AssertionError("A completed effect was replayed")
    ctx.tool_executor = AsyncMock(side_effect=completed_effect)
    return state


async def run_request_recovery(state, **overrides):
    # Local import leaves all pre-existing helper cases independently collectable
    # while the explicit API is being implemented in the other worker's files.
    from amplifier_module_context_managed.recovery import recover_request

    arguments = {
        "provider": state.provider, "count_view": state.count_view, "identity": IDENTITY,
        "save_progress": state.save, "install": state.install, "guard": state.guard,
        "retain_contents": (REQUEST_REMINDER,), "target_bytes": 2500,
    }
    arguments.update(overrides)
    return await recover_request(state.ctx, **arguments)


def assert_request_recovery_no_replay(state):
    state.ctx._prepare.assert_not_called()
    state.ctx._summarize.assert_not_called()
    state.ctx.persist_checkpoint.assert_not_called()
    state.ctx.tool_executor.assert_not_called()
    state.provider.complete.assert_not_called()
    state.provider.stream.assert_not_called()
    assert state.effects == {"write-1": 1}


def assert_request_recovery_unchanged(state):
    assert request_recovery_state(state.ctx) == state.before
    assert state.ctx.export_checkpoint(IDENTITY) == state.checkpoint_before
    assert request_recovery_state(state.ctx) == state.before
    assert_request_recovery_no_replay(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", [False, True])
async def test_explicit_recovery_installs_only_acknowledged_measured_candidate(prior, tmp_path):
    state = request_recovery_fixture(prior=prior)
    ctx = state.ctx
    checkpoint_path = tmp_path / "private-checkpoint.json"

    async def install(record, report=None):
        assert_request_recovery_unchanged(state)
        assert state.envelopes
        measured = state.envelopes[-1]["budget_decision"]
        assert measured["measurement"]["input_tokens"] <= REQUEST_LIMIT
        assert record["sha256"] == digest({k: v for k, v in record.items() if k != "sha256"})
        # Model the host's private atomic store, then acknowledge the exact bytes.
        staging = tmp_path / "private-checkpoint.pending"
        staging.touch(mode=0o600)
        staging.write_text(json.dumps(record), encoding="utf-8")
        staging.replace(checkpoint_path)
        stored = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert stored == record
        assert checkpoint_path.stat().st_mode & 0o777 == 0o600
        state.installed.append(copy.deepcopy(stored))
        await asyncio.sleep(0)
        assert_request_recovery_unchanged(state)
        return {"sha256": stored["sha256"]}

    state.install.side_effect = install
    result = await run_request_recovery(state)
    state.install.assert_awaited_once()
    assert state.views[0] == state.baseline
    assert len(state.views) >= 2
    assert ctx.summary[0] == state.end == 6
    assert ctx.summary[1] == state.window
    assert ctx.summary_identity == IDENTITY
    assert ctx.messages == state.before["messages"] and ctx.revision == state.before["revision"]
    assert ctx.summary_failure is None
    record = state.installed[0]
    assert ctx.export_checkpoint(IDENTITY) == record
    assert record["sourceRevision"] == {
        "messages": state.end, "sha256": digest(ctx.messages[:state.end])}
    assert record["evidenceRefs"] == ctx.evidence_refs == ctx.preserve_evidence.return_value
    ctx.preserve_evidence.assert_awaited_once_with(state.before["messages"])
    final = ctx._assembled(ctx.messages, ctx.summary, (REQUEST_REMINDER,))
    assert state.views[-1] == final
    for row in ctx.messages[state.end:]:
        assert final.count(row) == 1
    assert all(fact in ctx.summary[1]["content"] for fact in REQUEST_FACTS)
    before, after = (state.envelopes[i]["budget_decision"]["measurement"]["input_tokens"]
                     for i in (0, -1))
    assert before > REQUEST_LIMIT >= after
    assert after < before
    assert result["recovery"] == {
        "status": "installed", "before": before, "after": after, "limit": REQUEST_LIMIT,
        "throughMessage": state.end, "sha256": state.installed[0]["sha256"],
    }
    assert result["base_view"] == final
    assert result["final_attempt"]["dispatch"] == state.envelopes[-1]["dispatch"]
    assert result["outcome"] == "recovered" and result["transaction"] is None
    assert len(state.saved) > 1
    for record, _report in state.saved:
        assert record["summary"]["throughMessage"] <= state.end
        assert record["summary"]["throughMessage"] in settled_boundaries(ctx.messages)
        assert record["sourceRevision"]["sha256"] == digest(
            ctx.messages[:record["summary"]["throughMessage"]])
        assert record["evidenceRefs"] == ctx.preserve_evidence.return_value
    baseline_request = state.envelopes[0]["dispatch"]
    for envelope in state.envelopes:
        dispatch = envelope["dispatch"]
        assert dispatch.model_dump(exclude={"messages"}) == baseline_request.model_dump(exclude={"messages"})
        assert dispatch.messages[:2] == baseline_request.messages[:2]
        assert dispatch.messages[-1] == baseline_request.messages[-1]
        assert envelope["budget_decision"]["input_limit_tokens"] == REQUEST_LIMIT
        assert envelope["budget_decision"]["measurement"]["input_tokens"] == request_recovery_size(dispatch)
    for call in state.provider.compact_context.await_args_list:
        request = call.args[0]
        assert request_recovery_size(request) <= REQUEST_LIMIT
        assert any(request.model_dump() == counted.args[0].model_dump()
                   for counted in state.provider.request_budget.await_args_list)
        assert request.model_dump(exclude={"messages", "metadata"}) == baseline_request.model_dump(
            exclude={"messages", "metadata"})
        assert request.metadata == {
            **baseline_request.metadata, "purpose": "context-compaction", "stream": False,
            "native_compaction_request_context": True,
        }
        assert request.messages[:2] == baseline_request.messages[:2]
        assert all(row.content not in {item["content"] for item in ctx.messages[state.end:]}
                   for row in request.messages)
        assert baseline_request.messages[-1] not in request.messages
    assert_request_recovery_no_replay(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("restore", [False, True], ids=["direct", "restored"])
async def test_explicit_recovery_preserves_carried_input_after_carry_rotation(restore):
    carried = {"role": "user", "content": "A: Keep audit stamp ALPHA-17 on every export."}
    rows = [carried]
    for i in range(6):
        call_id = "write-1" if i == 0 else f"check-{i}"
        rows.extend([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "name": "fixture", "arguments": {}}]},
            {"role": "tool", "tool_call_id": call_id,
             "content": f"Completed {call_id}. " + (
                 REQUEST_NOTE + "x" * 1600 if i < 2 else "Verified.")},
        ])
    state = request_recovery_fixture(prior=False, rows=rows)
    ctx = state.ctx
    assert state.end == 5  # A and two exchanges covered; four exchanges stay verbatim.
    recovered = await run_request_recovery(state)
    assert recovered["recovery"]["status"] == "installed"
    assert ctx.summary[0] == state.end
    assert recovered["base_view"].count(carried) == 1
    assert carried["content"] not in json.dumps(ctx.summary[1])
    for call in state.provider.compact_context.await_args_list:
        assert all(row.content != carried["content"] for row in call.args[0].messages)
    assert recovered["measured_before"] > REQUEST_LIMIT >= recovered["measured_after"]
    assert recovered["measured_after"] < recovered["measured_before"]
    assert ctx.messages == rows
    assert ctx.revision == state.before["revision"]
    assert_request_recovery_no_replay(state)
    compactions = state.provider.compact_context.await_count
    assert compactions > 0

    # Exercise real ordinary preparation, not the recovery-only rejection sentinel.
    del ctx._prepare
    later = [
        {"role": "user", "content": "B: Check the CSV header."},
        {"role": "user", "content": "C: Check the row count."},
    ]
    for row in later:
        await ctx.add_message(row)
    assert ctx._carry(ctx.messages) == later
    # Fit must remain possible WITH A, not become possible only by dropping it.
    assert request_recovery_size(request_recovery_dispatch(
        recovered["base_view"] + later)) <= REQUEST_LIMIT
    record = json.loads(json.dumps(ctx.export_checkpoint(IDENTITY)))
    assert record["sourceRevision"] == {"messages": 5, "sha256": digest(rows[:5])}
    if restore:
        ctx = context(state.ctx.messages)
        for name in ("checkpoint_identity", "preserve_evidence", "persist_checkpoint",
                     "_summarize", "tool_executor"):
            setattr(ctx, name, getattr(state.ctx, name))
        assert ctx.restore_checkpoint(record, IDENTITY)["status"] == "restored"
    revision = ctx.revision

    async def count_view(view):
        # The recovery fixture's callback guards the OLD active state; ordinary
        # preparation needs a fresh callback with the same serialized count.
        request = request_recovery_dispatch(view)
        return {"dispatch": request,
                "budget_decision": await state.provider.request_budget(request)}

    ordinary = await ctx.get_measured_request_view(
        provider=state.provider, retain_contents=(REQUEST_REMINDER,), count_view=count_view)
    assert state.provider.compact_context.await_count == compactions
    assert ctx.messages == rows + later
    assert ctx.revision == revision
    assert ctx.export_checkpoint(IDENTITY) == record
    for envelope in [*state.envelopes, ordinary["final_attempt"]]:
        assert envelope["dispatch"].max_output_tokens == 73
        decision = envelope["budget_decision"]
        assert decision["input_limit_tokens"] == REQUEST_LIMIT
        assert decision["measurement"]["input_tokens"] == request_recovery_size(envelope["dispatch"])
    assert ordinary["final_attempt"]["budget_decision"]["estimated_input_tokens"] <= REQUEST_LIMIT
    state.ctx._summarize.assert_not_called()
    state.ctx.persist_checkpoint.assert_not_called()
    state.ctx.tool_executor.assert_not_called()
    state.provider.complete.assert_not_called()
    state.provider.stream.assert_not_called()
    assert state.effects == {"write-1": 1}
    assert ordinary["base_view"].count(carried) == 1
    assert sum(row.role == "user" and row.content == carried["content"]
               for row in ordinary["final_attempt"]["dispatch"].messages) == 1


def omitted_input_fixture():
    carried = {"role": "user", "content": "A: Keep audit stamp ALPHA-17 on every export."}
    rows = [carried]
    for i in range(6):
        rows += [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"settled-{i}", "name": "fixture", "arguments": {}}]},
            {"role": "tool", "tool_call_id": f"settled-{i}",
             "content": REQUEST_NOTE + "x" * 1600 if i < 2 else "Completed."},
        ]
    state = request_recovery_fixture(prior=False, rows=rows)
    # Set the unchanged auxiliary reservation before creating any checkpoint.
    state.ctx.config["native_compaction_max_output_tokens"] = 73
    state.before = request_recovery_state(state.ctx)
    return state, carried


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["accepted", "nonreducing", "cancelled", "observer_restore"])
async def test_omitted_input_adjacent_compaction_commits_provenance_only_on_acceptance(outcome):
    state, carried = omitted_input_fixture()
    ctx, provider = state.ctx, state.provider
    await run_request_recovery(state)
    assert ctx.summary_omitted_indices == [0]
    del ctx._prepare
    for text in ("B: Check headers.", "C: Check rows."):
        await ctx.add_message({"role": "user", "content": text})
    for i in range(6):
        await ctx.add_message({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"adjacent-{i}", "name": "fixture", "arguments": {}}]})
        await ctx.add_message({"role": "tool", "tool_call_id": f"adjacent-{i}",
                               "content": "New settled evidence. " + ("z" * 600 if i < 2 else "done")})
    originals = copy.deepcopy(ctx.messages)
    accepted = ctx.export_checkpoint(IDENTITY)
    assert ctx._carry(originals) == originals[13:15]
    calls = provider.compact_context.await_count
    counts = []

    async def count_view(view):
        request = request_recovery_dispatch(view)
        counts.append(copy.deepcopy(request))
        return {"dispatch": request, "budget_decision": await provider.request_budget(request)}

    async def compact(request):
        assert sum(row.content == carried["content"] for row in request.messages) == 1
        assert not any(row.content in ("B: Check headers.", "C: Check rows.") for row in request.messages)
        assert [row.content for row in request.messages if row.role == "developer"] == [
            "Current developer CSV instructions."]
        assert request.max_output_tokens == 73
        assert request_recovery_size(request) <= REQUEST_LIMIT
        assert ctx.export_checkpoint(IDENTITY) == accepted
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        window = request_recovery_window("adjacent")
        window["content"] += " " + carried["content"]
        window["metadata"]["canonical_window"][0]["summary"] = window["content"]
        if outcome == "nonreducing":
            window["metadata"]["canonical_window"][0]["padding"] = "n" * 10000
        return {"kind": "native", "message": window}

    provider.compact_context.side_effect = compact
    if outcome == "observer_restore":
        async def observer(event, data):
            if event == "context:compaction_finished" and data.get("outcome") == "completed":
                assert ctx.restore_checkpoint(accepted, IDENTITY)["status"] == "restored"
        ctx.hooks = SimpleNamespace(emit=AsyncMock(side_effect=observer))
    # Ordinary persistence is allowed only after candidate acceptance.
    ctx.persist_checkpoint = AsyncMock()
    if outcome != "accepted":
        error = (asyncio.CancelledError if outcome == "cancelled" else
                 RuntimeError if outcome == "observer_restore" else CompactionError)
        with pytest.raises(error) as caught:
            await ctx.get_measured_request_view(provider=provider, retain_contents=(), count_view=count_view)
        if outcome == "nonreducing":
            assert caught.value.code == "native_no_reduction"
        if outcome == "observer_restore":
            assert "changed" in str(caught.value).lower()
        assert ctx.export_checkpoint(IDENTITY) == accepted
        assert ctx.summary_omitted_indices == [0]
    else:
        result = await ctx.get_measured_request_view(provider=provider, retain_contents=(), count_view=count_view)
        assert ctx.summary[0] == 19 > accepted["summary"]["throughMessage"]
        assert ctx.summary_omitted_indices == [13, 14]
        assert result["final_attempt"]["budget_decision"]["estimated_input_tokens"] <= REQUEST_LIMIT
        dispatch = result["final_attempt"]["dispatch"]
        assert request_recovery_size(dispatch) < request_recovery_size(counts[0])
        for content in ("B: Check headers.", "C: Check rows."):
            assert sum(row.content == content for row in dispatch.messages) == 1
        assert not any(row.content == carried["content"] for row in dispatch.messages)
        assert sum(carried["content"] in str(row.content) for row in dispatch.messages) == 1
        record = json.loads(json.dumps(ctx.export_checkpoint(IDENTITY)))
        restored = BoundaryContextManager(copy.deepcopy(ctx.config))
        restored.messages = copy.deepcopy(originals)
        assert restored.restore_checkpoint(record, IDENTITY)["status"] == "restored"
        assert restored.summary_omitted_indices == ctx.summary_omitted_indices
        assert restored._assembled(originals, restored.summary) == ctx._assembled(originals, ctx.summary)
    assert provider.compact_context.await_count == calls + 1
    assert ctx.messages == originals
    assert state.effects == {"write-1": 1}
    ctx.tool_executor.assert_not_called()
    ctx._summarize.assert_not_called()
    provider.complete.assert_not_called()
    provider.stream.assert_not_called()


def test_native_assembly_does_not_resurrect_equal_valued_covered_carry():
    rows = [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "Already in native state."},
        {"role": "user", "content": "continue"},
        {"role": "user", "content": "Finish."},
    ]
    ctx = context(rows)
    ctx.summary = (2, request_recovery_window("accepted"))
    ctx.summary_identity = copy.deepcopy(IDENTITY)
    assert ctx.messages[0] == ctx.messages[2] and ctx.messages[0] is not ctx.messages[2]
    view = ctx._assembled(ctx.messages, ctx.summary)
    assert sum(row.get("content") == "continue" for row in view) == 1
    # The sidecar must distinguish actual carry from a duplicate-valued old row.
    indexed = ctx._assembly_rows(ctx.messages[:3], ctx.summary, (), ctx._carry(ctx.messages), [])
    assert 0 not in [i for i, _ in indexed]
    assert 2 in [i for i, _ in indexed]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_save", [False, True])
async def test_resumed_helper_preserves_old_omissions_after_carry_rotates(fail_save):
    state, carried = omitted_input_fixture()
    ctx, provider = state.ctx, state.provider
    await run_request_recovery(state)
    for text in ("B: continue", "C: continue"):
        await ctx.add_message({"role": "user", "content": text})
    prior = ctx.export_checkpoint(IDENTITY)
    originals = copy.deepcopy(ctx.messages)
    calls = provider.compact_context.await_count
    provider.compact_context.side_effect = lambda request: {
        "kind": "native", "message": request_recovery_window("resumed")}

    async def save(record, report):
        assert record["summary"]["omittedIndices"] == [0]
        if fail_save:
            raise asyncio.CancelledError()

    arguments = dict(progress=prior, through_message=9, exclude_indices=(),
                     save_progress=save, guard=lambda: None)
    template = request_recovery_dispatch([])
    if fail_save:
        with pytest.raises(asyncio.CancelledError):
            await recover_native(ctx, provider, template, IDENTITY, **arguments)
        assert ctx.export_checkpoint(IDENTITY) == prior
    else:
        record = await recover_native(ctx, provider, template, IDENTITY, **arguments)
        assert record["summary"]["throughMessage"] == 9
        assert record["summary"]["omittedIndices"] == [0]
    assert provider.compact_context.await_count == calls + 1
    assert ctx._assembled(ctx.messages, ctx.summary).count(carried) == 1
    assert ctx.messages == originals
    assert state.effects == {"write-1": 1}
    ctx.tool_executor.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("indices", [[True], [-1], [5], [0, 0], [1, 0], "0", None])
async def test_checkpoint_rejects_malformed_omission_provenance(indices):
    state, _ = omitted_input_fixture()
    await run_request_recovery(state)
    record = state.ctx.export_checkpoint(IDENTITY)
    record["summary"]["omittedIndices"] = indices
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    restored = BoundaryContextManager(copy.deepcopy(state.ctx.config))
    restored.messages = copy.deepcopy(state.ctx.messages)
    assert restored.restore_checkpoint(record, IDENTITY)["status"] == "rejected"
    assert restored.summary is None and restored.summary_omitted_indices == []
    assert restored.messages == state.ctx.messages


@pytest.mark.asyncio
async def test_final_count_cannot_return_after_same_revision_checkpoint_change(monkeypatch):
    from amplifier_module_context_managed.request_context import RequestContext

    state, _ = omitted_input_fixture()
    ctx = state.ctx
    await run_request_recovery(state)
    del ctx._prepare
    revision = ctx.revision
    original = RequestContext.get_measured_request_view

    async def finish_then_invalidate(fitter, **kwargs):
        result = await original(fitter, **kwargs)
        # An independently restored derived state need not edit canonical rows.
        ctx.summary_omitted_indices = []
        assert ctx.revision == revision
        return result

    monkeypatch.setattr(RequestContext, "get_measured_request_view", finish_then_invalidate)

    async def count_view(view):
        dispatch = request_recovery_dispatch(view)
        return {"dispatch": dispatch, "budget_decision": await state.provider.request_budget(dispatch)}

    calls = state.provider.compact_context.await_count
    with pytest.raises(RuntimeError, match="Checkpoint changed"):
        await ctx.get_measured_request_view(provider=state.provider, retain_contents=(), count_view=count_view)
    assert state.provider.compact_context.await_count == calls
    assert state.effects == {"write-1": 1}
    ctx.tool_executor.assert_not_called()
    state.provider.complete.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_legacy_recovery_hash_requires_reconstructible_omissions(empty):
    state = request_recovery_fixture(prior=False) if empty else omitted_input_fixture()[0]
    await run_request_recovery(state)
    assert state.ctx.summary_omitted_indices == ([] if empty else [0])
    record = state.ctx.export_checkpoint(IDENTITY)
    del record["summary"]["omittedIndices"]
    record["summary"].pop("version", None)
    record["configurationVersion"] = 2
    # A legacy empty binding is safe; a nonempty one cannot reveal its offsets.
    record["recovery"] = {"kind": "native-recovery", "version": 1,
                          "retentionSha256": digest([] if empty else [0])}
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    restored = BoundaryContextManager(copy.deepcopy(state.ctx.config))
    restored.messages = copy.deepcopy(state.ctx.messages)
    assert restored.restore_checkpoint(record, IDENTITY)["status"] == ("restored" if empty else "rejected")
    assert restored.summary_omitted_indices == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["oversized", "equal", "missing_before", "missing_after"])
async def test_explicit_recovery_rejects_unusable_final_measurement(failure):
    rows = request_recovery_history()
    if failure == "equal":
        # A fitting but equal-sized result isolates strict reduction from fit.
        for row in rows[3:6]:
            row["content"] = row["content"][:350]
    state = request_recovery_fixture(rows=rows)
    before = request_recovery_size(request_recovery_dispatch(state.baseline))
    if failure in {"oversized", "equal"}:
        candidate = state.ctx._assembled(state.ctx.messages, (state.end, state.window),
                                         (REQUEST_REMINDER,))
        after = request_recovery_size(request_recovery_dispatch(candidate))
        target = REQUEST_LIMIT + 100 if failure == "oversized" else before
        if failure == "oversized":
            assert before > target > REQUEST_LIMIT > after
        else:
            assert after < target == before <= REQUEST_LIMIT
        state.window["metadata"]["canonical_window"][0]["padding"] = "x" * (target - after)
        padded = state.ctx._assembled(state.ctx.messages, (state.end, state.window),
                                      (REQUEST_REMINDER,))
        assert request_recovery_size(request_recovery_dispatch(padded)) == target
    else:
        state.measurement_failure = failure.removeprefix("missing_")
    # Padding is the final provider-returned payload, not an ordinal-dependent
    # count that would mask an incorrect assembled view.
    with pytest.raises(CompactionError) as caught:
        await run_request_recovery(state, target_bytes=100_000)
    assert caught.value.code == {
        "oversized": "native_input_oversized", "equal": "native_no_reduction",
        "missing_before": "native_measurement_unavailable",
        "missing_after": "native_measurement_unavailable",
    }[failure]
    state.install.assert_not_called()
    if failure == "missing_before":
        state.provider.compact_context.assert_not_called()
    else:
        state.provider.compact_context.assert_awaited()
        assert len(state.views) >= 2
        assert state.views[-1] == state.ctx._assembled(
            state.ctx.messages, (state.end, state.window), (REQUEST_REMINDER,))
        final = state.envelopes[-1]["budget_decision"]
        if failure == "missing_after":
            assert "measurement" not in final
        else:
            assert final["measurement"]["input_tokens"] == target
            assert final["input_limit_tokens"] == REQUEST_LIMIT
    assert_request_recovery_unchanged(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["guard", "identity", "configuration", "options", "allowance"])
async def test_explicit_recovery_rejects_changed_request_contract(change):
    state = request_recovery_fixture()
    state.change_contract = change
    with pytest.raises(RuntimeError):
        await run_request_recovery(state)
    state.install.assert_not_called()
    state.provider.compact_context.assert_awaited()
    if change == "configuration":
        # The external configuration change is real; recovery must not undo it.
        # Export recomputes the configuration digest, so compare active fields,
        # not a newly exported record against an older configuration's digest.
        assert state.ctx.config["summary_max_source_chars"] == 2048
        assert request_recovery_state(state.ctx) == state.before
        assert_request_recovery_no_replay(state)
    else:
        assert_request_recovery_unchanged(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    "provider_cancel", "save_cancel", "install_cancel", "save_error",
    "ack_none", "ack_wrong_digest", "ack_boolean",
])
async def test_explicit_recovery_preserves_active_checkpoint_until_exact_install_ack(failure):
    state = request_recovery_fixture()
    error = OSError("Private progress store failed") if failure == "save_error" else asyncio.CancelledError()
    if failure == "provider_cancel":
        state.provider.compact_context.side_effect = error
    elif failure in {"save_cancel", "save_error"}:
        state.save.side_effect = error
    elif failure == "install_cancel":
        state.install.side_effect = error
    else:
        state.install.side_effect = None
        state.install.return_value = {
            "ack_none": None, "ack_wrong_digest": {"sha256": "0" * 64}, "ack_boolean": True,
        }[failure]
    expected = RuntimeError if failure.startswith("ack_") else type(error)
    with pytest.raises(expected) as caught:
        await run_request_recovery(state)
    if not failure.startswith("ack_"):
        assert caught.value is error
    if failure.startswith("ack_") or failure == "install_cancel":
        state.install.assert_awaited_once()
        assert state.saved  # Completed auxiliary work is real, not installed state.
    else:
        state.install.assert_not_called()
    assert_request_recovery_unchanged(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["missing_result", "queued_result", "indivisible"])
async def test_explicit_recovery_never_compacts_an_unsettled_or_split_exchange(boundary):
    rows = request_recovery_history()
    if boundary == "missing_result":
        del rows[2]
    elif boundary == "queued_result":
        rows[2]["content"] = json.dumps({"status": "queued", "job_id": "job-17"})
    else:
        rows = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "name": "fixture", "arguments": {}},
                {"id": "b", "name": "fixture", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "b", "content": "completed b " * 1000},
            {"role": "tool", "tool_call_id": "a", "content": "completed a"},
            {"role": "user", "content": "Original objective."},
            {"role": "user", "content": "Previous current input."},
            {"role": "user", "content": "Latest current input."},
        ]
        assert settled_boundaries(rows)[0] == 3
    state = request_recovery_fixture(prior=False, rows=rows)
    with pytest.raises(CompactionError):
        await run_request_recovery(state, target_bytes=1)
    state.provider.compact_context.assert_not_called()
    state.save.assert_not_called()
    state.install.assert_not_called()
    assert_request_recovery_unchanged(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["evidence_error", "missing_evidence_callback", "unsupported_native"])
async def test_explicit_recovery_requires_evidence_and_never_uses_portable_fallback(failure):
    state = request_recovery_fixture()
    if failure == "evidence_error":
        state.ctx.preserve_evidence.side_effect = OSError("Evidence unavailable")
    elif failure == "missing_evidence_callback":
        state.ctx.preserve_evidence = None
    else:
        state.provider.supports_native_compaction = lambda: False
    with pytest.raises(OSError if failure == "evidence_error" else RuntimeError):
        await run_request_recovery(state)
    state.provider.compact_context.assert_not_called()
    state.save.assert_not_called()
    state.install.assert_not_called()
    assert_request_recovery_unchanged(state)


@pytest.mark.asyncio
async def test_explicit_recovery_authorized_prefix_rejects_changed_source():
    state = request_recovery_fixture()
    with pytest.raises(CompactionError, match="recovery_source_changed"):
        await run_request_recovery(state, source_revision={"messages": 6, "sha256": "0" * 64})
    state.provider.request_budget.assert_not_called()
    state.install.assert_not_called()
    assert_request_recovery_unchanged(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("fits", [False, True])
async def test_explicit_recovery_authorized_prefix_keeps_appended_evidence_outside_window(fits):
    rows = request_recovery_history()
    if fits:
        for row in rows[4:6]:
            row["content"] = row["content"][:40]
    state = request_recovery_fixture(prior=False, rows=rows)
    source_revision = {"messages": 4, "sha256": digest(state.ctx.messages[:4])}
    if fits:
        result = await run_request_recovery(state, source_revision=source_revision)
        assert result["recovery"]["throughMessage"] == 4
        view = result["base_view"]
    else:
        with pytest.raises(CompactionError, match="native_input_oversized"):
            await run_request_recovery(state, source_revision=source_revision)
        assert_request_recovery_unchanged(state)
        state.install.assert_not_called()
        view = state.views[-1]
    for row in state.ctx.messages[4:]:
        assert row in view
    for call in state.provider.compact_context.await_args_list:
        assert not any(row.content == state.ctx.messages[4]["content"] for row in call.args[0].messages)


@pytest.mark.asyncio
async def test_explicit_recovery_callback_cancellation_is_not_replaced_by_guard_failure():
    state = request_recovery_fixture()
    cancelled = asyncio.CancelledError("external stop")
    async def cancel(request):
        state.invalidated = True
        raise cancelled
    state.provider.compact_context.side_effect = cancel
    with pytest.raises(asyncio.CancelledError) as caught:
        await run_request_recovery(state)
    assert caught.value is cancelled
    assert_request_recovery_unchanged(state)


@pytest.mark.asyncio
async def test_pending_recovery_hook_runs_only_when_host_supplies_it():
    from amplifier_module_context_managed.boundary import mount_boundary
    caps = {}
    coordinator = SimpleNamespace(
        hooks=None, get_capability=caps.get,
        register_capability=lambda name, value: caps.__setitem__(name, value),
        register_contributor=lambda *args: None)
    async def mount(name, value):
        coordinator.context = value
    coordinator.mount = mount
    await mount_boundary(coordinator, {"durable_checkpoints": True})
    ctx = coordinator.context
    ctx._prepare = AsyncMock(side_effect=RuntimeError("ordinary path"))
    count_view = AsyncMock()
    with pytest.raises(RuntimeError, match="ordinary path"):
        await ctx.get_measured_request_view(provider=None, retain_contents=[], count_view=count_view)
    result = {"sentinel": "host explicitly requested recovery"}
    async def pending(**kwargs):
        assert not ctx.lock.locked()
        assert kwargs["count_view"] is count_view
        return result
    caps["context.recovery.run_pending"] = AsyncMock(side_effect=pending)
    ctx._prepare.reset_mock()
    assert await ctx.get_measured_request_view(
        provider=None, retain_contents=[], count_view=count_view) is result
    ctx._prepare.assert_not_called()
    count_view.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", [False, True])
async def test_explicit_recovery_respects_conservative_hard_safety_estimate(prior):
    state = request_recovery_fixture(prior=prior)
    count = state.count_view.side_effect
    async def conservative(view):
        result = await count(view)
        if any((row.get("metadata") or {}).get("canonical_window", [{}])[0].get("id") == "candidate"
               for row in view):
            assert result["budget_decision"]["measurement"]["input_tokens"] < REQUEST_LIMIT
            result["budget_decision"]["estimated_input_tokens"] = REQUEST_LIMIT + 500
        return result
    state.count_view.side_effect = conservative
    with pytest.raises(CompactionError, match="native_input_oversized"):
        await run_request_recovery(state)
    state.install.assert_not_called()
    assert_request_recovery_unchanged(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", [False, True])
async def test_explicit_recovery_preserves_covered_canonical_developer_instructions(prior):
    rows = request_recovery_history()
    rows[3] = {"role": "developer", "content": "Canonical developer: retain exact receipt R-17."}
    state = request_recovery_fixture(prior=prior, rows=rows)
    result = await run_request_recovery(state)
    current = [row for row in result["base_view"] if row["role"] == "developer"]
    assert current == [rows[3]]
    ordinary = state.ctx._assembled(state.ctx.messages, state.ctx.summary, (REQUEST_REMINDER,))
    assert [row for row in ordinary if row["role"] == "developer"] == current
    assert state.ctx.messages == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["request", "decision"])
async def test_explicit_recovery_rejects_final_envelope_mutation_during_install(field):
    state = request_recovery_fixture()
    count = state.count_view.side_effect
    envelopes = []
    async def retain(view):
        result = await count(view)
        envelopes.append(result)
        return result
    async def mutate(record, verification):
        await asyncio.sleep(0)
        if field == "request":
            envelopes[-1]["dispatch"].messages[-1].content = "Changed after count"
        else:
            envelopes[-1]["budget_decision"]["input_limit_tokens"] += 1
        return {"sha256": record["sha256"]}
    state.count_view.side_effect = retain
    state.install.side_effect = mutate
    with pytest.raises(CompactionError, match="recovery_request_changed"):
        await run_request_recovery(state)
    state.install.assert_awaited_once()  # Storage may have committed: not rollback.
    assert_request_recovery_unchanged(state)
