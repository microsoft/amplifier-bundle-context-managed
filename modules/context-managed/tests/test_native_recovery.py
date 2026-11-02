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
