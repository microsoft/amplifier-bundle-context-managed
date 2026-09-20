import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_module_context_managed.boundary import BoundaryContextManager


def conversation():
    return [{"role": role, "content": text} for role, text in [
        ("user", "Build a report"), ("assistant", "Evidence from earlier research. " * 500),
        ("user", "Keep local files"), ("assistant", "Progress " * 100),
        ("user", "Use the revised requirement")]]


def provider(complete=None):
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="Objective: report. Keep local files. Earlier research complete.")])
    return SimpleNamespace(complete=complete or AsyncMock(return_value=response))


@pytest.mark.asyncio
async def test_visible_compaction_preserves_canonical_and_latest_correction():
    hooks = SimpleNamespace(emit=AsyncMock())
    context = BoundaryContextManager({"max_tokens": 6000, "summarize_trigger": 0.2}, hooks)
    original = conversation()
    await context.set_messages(original)
    view = await context.get_messages_for_request(provider=provider())
    assert "Continuation note" in str(view)
    assert "Use the revised requirement" in str(view)
    assert await context.get_messages() == original
    assert [call.args[0] for call in hooks.emit.call_args_list if call.args[0].startswith("context:compaction_")] == ["context:compaction_started", "context:compaction_finished"]
    original[0]["content"] = "mutated by caller"
    assert (await context.get_messages())[0]["content"] == "Build a report"


@pytest.mark.asyncio
async def test_restore_during_compaction_discards_stale_note():
    entered, release = asyncio.Event(), asyncio.Event()
    async def complete(request):
        entered.set()
        await release.wait()
        return await provider().complete(request)
    context = BoundaryContextManager({"max_tokens": 6000, "summarize_trigger": 0.2})
    await context.set_messages(conversation())
    task = asyncio.create_task(context.get_messages_for_request(provider=provider(complete)))
    await entered.wait()
    restored = [{"role": "user", "content": "Restored authoritative history"}]
    await context.set_messages(restored)
    release.set()
    with pytest.raises(RuntimeError, match="stale summary"):
        await task
    assert context.summary is None
    assert await context.get_messages() == restored
    assert not context.is_compacting


@pytest.mark.asyncio
async def test_cancel_compaction_keeps_originals_and_closes_progress():
    entered = asyncio.Event()
    async def complete(request):
        entered.set()
        await asyncio.Event().wait()
    hooks = SimpleNamespace(emit=AsyncMock())
    context = BoundaryContextManager({"max_tokens": 6000, "summarize_trigger": 0.2}, hooks)
    original = conversation()
    await context.set_messages(original)
    task = asyncio.create_task(context.get_messages_for_request(provider=provider(complete)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await context.get_messages() == original
    assert context.summary is None
    assert hooks.emit.call_args.args[1]["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_pending_job_is_outside_summary_and_original_output_is_lossless():
    context = BoundaryContextManager()
    messages = conversation()
    messages[1:1] = [{"role": "assistant", "content": "", "tool_calls": [{"id": "pending", "name": "delegate", "arguments": {}}]},
                       {"role": "tool", "tool_call_id": "pending", "content": '{"status":"queued","job_id":"j"}'}]
    assert context._boundary(messages, []) == 0
    payload = "x" * 200000
    messages.append({"role": "tool", "content": payload, "tool_call_id": "big"})
    await context.set_messages(messages)
    assert (await context.get_messages())[-1]["content"] == payload


@pytest.mark.asyncio
async def test_measured_request_preserves_active_jobs_and_exact_dispatch():
    context = BoundaryContextManager({"max_tokens": 1000, "compaction_notice_enabled": False})
    context.active_operations = lambda: [{"job_id": "pending-job", "call_id": "original-call"}]
    await context.add_message({"role": "user", "content": "Continue the task"})
    dispatch = object()
    async def count_view(view):
        assert "pending-job" in str(view)
        return {"dispatch": dispatch, "budget_decision": {
            "estimated_input_tokens": 200, "input_limit_tokens": 1000,
            "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 200}}}
    result = await context.get_measured_request_view(provider=None, retain_contents=[], count_view=count_view)
    assert result["final_attempt"]["dispatch"] is dispatch
    assert len(await context.get_messages()) == 1
