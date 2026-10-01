"""Restart resumes derived work, never a model call or a canonical tool."""
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import LLMError
from amplifier_module_context_managed.checkpoint import digest
from .test_summary_work_budget import context, response, IDENTITY, history, finished


@pytest.mark.asyncio
async def test_restart_reuses_completed_piece_without_copying_pending_source():
    first = await context()
    saved = []
    first.persist_checkpoint = lambda: saved.append(copy.deepcopy(first.export_checkpoint(IDENTITY)))
    model = SimpleNamespace(name="fixture", default_model="fixture-model",
        complete=AsyncMock(side_effect=[response("FIRST evidence verified"), LLMError("offline", retryable=True)]))
    await first._prepare(model, None, [])
    assert len(saved) == 1 and saved[0]["summary"] is None
    assert "pending" not in saved[0]["progress"] and "FIRST evidence verified" == saved[0]["progress"]["text"]
    second = await context()
    assert second.restore_checkpoint(saved[0], IDENTITY)["status"] == "resuming"
    assert second.summary is None and await second.get_messages() == history()
    model.complete = AsyncMock(return_value=response())
    await second._prepare(model, None, [])
    assert second.summary is not None and finished(second)[-1]["resumed_fragments"] == 1
    assert "FIRST evidence verified" in model.complete.call_args_list[0].args[0].messages[-1].content
    assert model.complete.await_count == finished(second)[-1]["completed_fragments"] - 1


@pytest.mark.asyncio
async def test_bounded_summary_option_does_not_change_chat_and_truncation_is_not_committed():
    manager = await context()
    limited = response("Incomplete note")
    limited.finish_reason = "length"
    model = SimpleNamespace(get_info=lambda: {"capabilities": ["completion:auto_continue:v1"]},
        complete=AsyncMock(return_value=limited))
    await manager._prepare(model, None, [])
    assert model.complete.call_args.kwargs == {"request_options": {"auto_continue": False}}
    assert manager.summary is None and manager.export_checkpoint(IDENTITY) is None
    assert finished(manager)[-1]["usage"]["output_tokens"] == 10


@pytest.mark.asyncio
async def test_harmless_trigger_change_keeps_checkpoint_but_source_and_model_still_guard_it():
    manager = await context()
    await manager._prepare(SimpleNamespace(complete=AsyncMock(return_value=response())), None, [])
    record = manager.export_checkpoint(IDENTITY)
    second = await context(summarize_trigger=.9, summary_retry_delay=999, summary_max_calls=3)
    assert second.restore_checkpoint(record, IDENTITY)["status"] == "restored"
    third = await context()
    assert third.restore_checkpoint(record, {**IDENTITY, "model": "other"})["status"] == "rejected"
    changed = history();changed[0]["content"] = "New constraint"
    await third.set_messages(changed)
    assert third.restore_checkpoint(record, IDENTITY)["status"] == "rejected"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["fingerprint", "provider", "overlapping_segments"])
async def test_incompatible_regenerated_progress_is_explained_and_never_skips_source(change):
    first = await context()
    model = SimpleNamespace(name="fixture", default_model="fixture-model",
        complete=AsyncMock(side_effect=[response("PAID completed evidence"), LLMError("offline", retryable=True)]))
    await first._prepare(model, None, [])
    record = first.export_checkpoint(IDENTITY)
    if change == "fingerprint":
        record["progress"]["fingerprint"] = "changed"
    elif change == "provider":
        record["progress"]["provider"] = ["different-utility", "other-model"]
    else:
        record["progress"]["segments"].insert(1, record["progress"]["segments"][0])
    record["sha256"] = digest({key: value for key, value in record.items() if key != "sha256"})
    second = await context()
    assert second.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
    model.complete = AsyncMock(return_value=response())
    await second._prepare(model, None, [])
    events = [call.args[1] for call in second.hooks.emit.call_args_list
              if call.args[0] == "context:checkpoint_progress_rejected"]
    assert len(events) == 1 and events[0]["originalsAvailable"] is True
    assert "PAID completed evidence" not in model.complete.call_args_list[0].args[0].messages[-1].content
    assert finished(second)[-1]["resumed_fragments"] == 0
    assert second.summary is not None and await second.get_messages() == history()


@pytest.mark.asyncio
async def test_legacy_checkpoint_keeps_its_original_strict_configuration_contract():
    from amplifier_module_context_managed.boundary import SUMMARY_PROMPT
    first = await context()
    await first._prepare(SimpleNamespace(complete=AsyncMock(return_value=response())), None, [])
    record = first.export_checkpoint(IDENTITY)
    record.pop("configurationVersion")
    record.pop("progress")
    record["configuration"] = digest({"config": first.config, "prompt": SUMMARY_PROMPT})
    record["sha256"] = digest({key: value for key, value in record.items() if key != "sha256"})
    second = await context()
    assert second.restore_checkpoint(record, IDENTITY)["status"] == "restored"
    changed = await context(summary_retry_delay=999)
    assert changed.restore_checkpoint(record, IDENTITY)["status"] == "rejected"
