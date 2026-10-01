"""Interrupted semantic work must remain bounded and never become history authority."""
import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import LLMError
from amplifier_module_context_managed.boundary import BoundaryContextManager


IDENTITY = {"provider": "fixture", "model": "fixture-model"}


def history():
    return [
        {"role": "system", "content": "Never replay completed tools."},
        {"role": "user", "content": "Preserve project ORBIT and original evidence."},
        {"role": "assistant", "content": "FIRST verified evidence. " + "x" * 4000 + " LAST verified evidence."},
        {"role": "user", "content": "Correction: preserve all originals."},
        {"role": "assistant", "content": "Recent completed work."},
        {"role": "user", "content": "Finish the current report."},
    ]


def response(text="Verified evidence; all originals preserved."):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           usage={"input_tokens": 100, "output_tokens": 10})


async def context(**config):
    hooks = SimpleNamespace(emit=AsyncMock())
    manager = BoundaryContextManager({"max_tokens": 10000, "summarize_trigger": .01,
        "summary_max_source_chars": 1200, "summary_retry_delay": 0,
        "durable_checkpoints": True, **config}, hooks)
    manager.checkpoint_identity = lambda: copy.deepcopy(IDENTITY)
    await manager.set_messages(history())
    return manager


def finished(manager):
    return [call.args[1] for call in manager.hooks.emit.call_args_list
            if call.args[0] == "context:compaction_finished"]


@pytest.mark.asyncio
async def test_legacy_fragment_and_work_timers_do_not_interrupt_healthy_summary():
    manager = await context(summary_timeout=.001, summary_total_work_timeout=.001)
    requests = []

    async def complete(request):
        requests.append(request)
        await asyncio.sleep(.03)
        return response()

    original = await manager.get_messages()
    await manager._prepare(SimpleNamespace(complete=complete), None, [])
    assert len(requests) >= 4  # Every healthy fragment exceeds the retired timers.
    assert manager.summary[0] == 3
    assert manager._summary_progress is None
    assert await manager.get_messages() == original
    event = finished(manager)[-1]
    assert event["outcome"] == "completed"
    assert event["completed_fragments"] == len(requests)
    assert event["remaining_fragments"] == 0
    assert manager.export_checkpoint(IDENTITY)["summary"]["throughMessage"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError("transport timeout"), LLMError("provider unavailable", retryable=True)])
async def test_provider_failure_resumes_completed_fragments_without_exporting_partial_note(failure):
    manager = await context()
    requests = []

    async def complete(request):
        requests.append(request)
        if len(requests) == 2:
            raise failure
        return response("FIRST fragment verified; originals preserved.")

    model = SimpleNamespace(complete=complete)
    original = await manager.get_messages()
    await manager._prepare(model, None, [])
    assert manager.summary is None and manager.export_checkpoint(IDENTITY)["summary"] is None
    assert manager._summary_progress["completed"] == 1
    first_event = finished(manager)[-1]
    assert first_event["failure"]["type"] == type(failure).__name__
    assert "timeout" not in first_event  # No fabricated context-owned deadline.
    assert first_event["completed_fragments"] == 1
    assert first_event["remaining_fragments"] > 0
    await manager._prepare(model, None, [])
    assert requests[2].messages[-1].content == requests[1].messages[-1].content
    assert "FIRST fragment verified" in requests[2].messages[-1].content
    assert requests[0].messages[-1].content != requests[2].messages[-1].content
    assert manager.summary is not None and manager._summary_progress is None
    assert await manager.get_messages() == original
    final = finished(manager)[-1]
    assert final["outcome"] == "completed" and final["resumed_fragments"] == 1
    assert final["summary_calls_total"] == final["completed_fragments"] + 1


@pytest.mark.asyncio
async def test_call_allowance_resumes_saved_work_in_later_pass_without_repeating_paid_pieces():
    manager = await context(summary_max_calls=2)
    model = SimpleNamespace(complete=AsyncMock(return_value=response()))
    await manager._prepare(model, None, [])
    assert manager.summary is None and manager._summary_progress["completed"] == 2
    assert finished(manager)[-1]["work_limit_exhausted"] is True
    first_requests = [c.args[0].messages[-1].content for c in model.complete.call_args_list]
    for _ in range(8):
        if manager.summary: break
        await manager._prepare(model, None, [])
    assert manager.summary is not None
    later = [c.args[0].messages[-1].content for c in model.complete.call_args_list[2:]]
    assert all(r not in first_requests for r in later)
    assert finished(manager)[-1]["resumed_fragments"] >= 2


@pytest.mark.asyncio
async def test_successful_progress_resets_stalled_attempts_without_resetting_total_work():
    manager = await context()
    calls = 0

    async def complete(request):
        nonlocal calls
        calls += 1
        if calls <= 6 and calls % 2 == 0:
            raise LLMError("temporary", retryable=True)
        return response()

    model = SimpleNamespace(complete=complete)
    for _ in range(3):
        await manager._prepare(model, None, [])
        assert manager.summary_failure["attempts"] == 1
    await manager._prepare(model, None, [])
    assert manager.summary is not None
    event = finished(manager)[-1]
    assert event["summary_calls_total"] == calls
    assert event["resumed_fragments"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["prefix", "tail"])
async def test_inflight_replacement_aborts_request_and_retains_only_unchanged_prefix_work(change):
    manager = await context()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def complete(request):
        calls.append(request)
        if len(calls) == 2:
            entered.set()
            await release.wait()
        return response()

    model = SimpleNamespace(complete=complete)
    task = asyncio.create_task(manager.get_messages_for_request(provider=model))
    await entered.wait()
    replacement = await manager.get_messages()
    replacement[2 if change == "prefix" else -2]["content"] += " Authoritative correction."
    await manager.set_messages(replacement)
    release.set()
    with pytest.raises(RuntimeError, match="History changed"):
        await task
    assert manager.summary is None
    assert manager._summary_progress is None if change == "prefix" else manager._summary_progress["completed"] == 1
    await manager._prepare(model, None, [])
    assert manager.summary is not None
    assert finished(manager)[-1]["resumed_fragments"] == (1 if change == "tail" else 0)
    assert await manager.get_messages() == replacement


@pytest.mark.asyncio
async def test_cancellation_during_next_preflight_preserves_only_completed_private_work():
    manager = await context()
    entered = asyncio.Event()
    budgets = 0

    async def request_budget(request, **kwargs):
        nonlocal budgets
        if (request.metadata or {}).get("purpose") == "context-compaction":
            budgets += 1
            if budgets == 2:
                entered.set()
                await asyncio.Event().wait()
        return {}

    model = SimpleNamespace(complete=AsyncMock(return_value=response()), request_budget=request_budget)
    task = asyncio.create_task(manager.get_messages_for_request(provider=model))
    await entered.wait()
    assert manager._summary_progress["completed"] == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager._summary_progress["completed"] == 1 and manager.summary is None
    assert manager.export_checkpoint(IDENTITY)["summary"] is None
    assert await manager.get_messages() == history()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["config", "identity", "required_content"])
async def test_draft_invalidates_on_changed_summary_contract(change):
    manager = await context()
    model = SimpleNamespace(complete=AsyncMock(side_effect=[response(), LLMError("temporary", retryable=True)]))
    await manager._prepare(model, None, [])
    assert manager._summary_progress["completed"] == 1
    retain = []
    if change == "config":
        manager.config["summary_target_tokens"] = 1400
    elif change == "identity":
        manager.checkpoint_identity = lambda: {**IDENTITY, "model": "another-model"}
    else:
        retain = [history()[2]["content"]]
    model.complete = AsyncMock(return_value=response())
    await manager._prepare(model, None, retain)
    assert finished(manager)[-1]["resumed_fragments"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, .001, 120, 600])
async def test_saved_legacy_deadlines_are_retired_without_configuration_failure(value):
    manager = await context(summary_timeout=value, summary_total_work_timeout=value,
                            native_compaction_timeout=value)
    for key in ("summary_timeout", "summary_total_work_timeout", "native_compaction_timeout"):
        assert key not in manager.config


@pytest.mark.asyncio
async def test_idle_tail_replacement_preserves_private_completed_work():
    manager = await context()
    model = SimpleNamespace(complete=AsyncMock(side_effect=[response(), LLMError("temporary", retryable=True)]))
    await manager._prepare(model, None, [])
    draft = copy.deepcopy(manager._summary_progress)
    failure = copy.deepcopy(manager.summary_failure)
    replacement = await manager.get_messages()
    replacement[-2]["content"] += " A new verified result."
    await manager.set_messages(replacement)
    assert manager._summary_progress == draft and manager.summary_failure == failure
    model.complete = AsyncMock(return_value=response())
    await manager._prepare(model, None, [])
    assert manager.summary is not None and manager._summary_progress is None


@pytest.mark.asyncio
async def test_native_failure_is_not_repeated_when_resuming_portable_draft():
    manager = await context()
    model = SimpleNamespace(supports_native_compaction=lambda: True,
        validate_compacted_context=lambda row: True,
        request_budget=lambda request, **kwargs: {"measurement": {"kind": "provider_count", "input_tokens": 9000}},
        compact_context=AsyncMock(side_effect=LLMError("native unavailable", retryable=False)),
        complete=AsyncMock(side_effect=[response(), LLMError("temporary", retryable=True)]))
    await manager._prepare(model, None, [])
    assert manager._summary_progress["completed"] == 1
    model.complete = AsyncMock(return_value=response())
    await manager._prepare(model, None, [])
    assert model.compact_context.await_count == 1
    assert manager.summary is not None
    assert finished(manager)[-1]["native_skipped_for_staged_semantic"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["config", "identity", "provider"])
async def test_inflight_summary_contract_change_discards_draft_without_committing(change):
    manager = await context()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def complete(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            await release.wait()
        return response()

    model = SimpleNamespace(name="fixture", default_model="fixture-model", complete=complete)
    task = asyncio.create_task(manager._prepare(model, None, []))
    await entered.wait()
    if change == "config":
        manager.config["summary_target_tokens"] = 1400
    elif change == "identity":
        manager.checkpoint_identity = lambda: {**IDENTITY, "model": "new-model"}
    else:
        model.default_model = "another-model"
    release.set()
    with pytest.raises(RuntimeError, match="contract changed"):
        await task
    assert manager._summary_progress is None and manager.summary is None
    assert manager.export_checkpoint(IDENTITY) is None
