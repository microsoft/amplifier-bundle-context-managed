import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_module_context_managed.boundary import BoundaryContextManager


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, measured", [("actual", True), ("estimate", False)])
async def test_mount_advertises_provider_count_only_in_actual_mode(mode, measured):
    from amplifier_module_context_managed.boundary import mount_boundary
    capabilities = {}
    coordinator = SimpleNamespace(mount=AsyncMock(), hooks=None,
        get_capability=capabilities.get,
        register_capability=lambda name, value: capabilities.__setitem__(name, value),
        register_contributor=lambda *args: None)
    await mount_boundary(coordinator, {"token_meter": mode})
    assert ("context.measured_request_view" in capabilities) is measured
    assert callable(capabilities["context.request_retention"])


def conversation():
    return [{"role": role, "content": text} for role, text in [
        ("user", "Build a report"), ("assistant", "Evidence from earlier research. " * 500),
        ("user", "Keep local files"), ("assistant", "Progress " * 100),
        ("user", "Use the revised requirement")]]


def test_structured_service_observations_do_not_become_human_turn_boundaries():
    context = BoundaryContextManager()
    observation = {"role": "user", "content": "External observation: result", "metadata": {
        "amplifier_input": {"version": 1, "kind": "service", "source": "worker", "id": "report"}}}
    assert not context._human(observation)
    # A user quoting the same text remains a human input; text is not provenance.
    assert context._human({"role": "user", "content": observation["content"]})
    assert context._boundary([{"role": "user", "content": "Original task"}, observation, observation], []) == 0


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


@pytest.mark.asyncio
async def test_required_persisted_reminder_survives_two_compactions_and_restore():
    reminder = {"role": "user", "content": "Required policy\r\nKeep originals — 保留.\n",
                "metadata": {"ephemeral": True, "persisted": True, "reminder_placement": "pre_user"}}
    config = {"max_tokens": 6000, "summarize_trigger": .1, "durable_checkpoints": True}
    context = BoundaryContextManager(config)
    identity = {"provider": "configured", "model": "unchanged"}
    context.checkpoint_identity = lambda: identity
    original = conversation()
    original.insert(2, reminder)
    await context.set_messages(original)
    model = provider()
    retain = [reminder["content"]]
    view = await context.get_messages_for_request_retaining(provider=model, retain_contents=retain)
    first_boundary = context.summary[0]
    assert first_boundary > 2
    assert [row for row in view if row.get("content") == reminder["content"]] == [reminder]
    appended = [{"role": "assistant", "content": "More verified research " * 500},
                {"role": "user", "content": "Retain the correction and continue"}]
    for row in appended:
        await context.add_message(row)
    view = await context.get_messages_for_request_retaining(provider=model, retain_contents=retain)
    assert model.complete.await_count == 2
    assert context.summary[0] > first_boundary
    assert [row for row in view if row.get("content") == reminder["content"]] == [reminder]
    assert await context.get_messages() == original + appended
    saved = context.export_checkpoint(identity)
    restored = BoundaryContextManager(config)
    restored.checkpoint_identity = lambda: identity
    await restored.set_messages(original + appended)
    assert restored.restore_checkpoint(saved, identity)["status"] == "restored"
    view = await restored.get_messages_for_request_retaining(retain_contents=retain)
    assert restored.summary == context.summary
    assert [row for row in view if row.get("content") == reminder["content"]] == [reminder]
    assert await restored.get_messages() == original + appended


@pytest.mark.parametrize("metadata", [{}, {"ephemeral": True}, {"persisted": True},
                                     {"ephemeral": "true", "persisted": True}])
def test_required_ordinary_or_unproven_reminder_still_blocks_boundary(metadata):
    context = BoundaryContextManager()
    messages = conversation()
    messages.insert(2, {"role": "user", "content": "Required policy", "metadata": metadata})
    assert context._boundary(messages, ["Required policy"]) <= 2


def test_preserved_reminder_does_not_release_unresolved_tool_call_barrier():
    context = BoundaryContextManager()
    messages = conversation()
    messages[1:1] = [{"role": "assistant", "content": "", "tool_calls": [{"id": "pending", "name": "delegate", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "pending", "content": '{"status":"queued","job_id":"job"}'},
        {"role": "user", "content": "Required policy", "metadata": {"ephemeral": True, "persisted": True}}]
    assert context._boundary(messages, ["Required policy"]) == 0
