import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_module_context_managed.boundary import BoundaryContextManager
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
