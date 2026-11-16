"""Soft cumulative-note guidance; no live inference or semantic-quality claim."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_module_context_managed.boundary import BoundaryContextManager, SUMMARY_PROMPT
from amplifier_module_context_managed.checkpoint import digest
from amplifier_module_context_managed.errors import CompactionError
from amplifier_module_context_managed.summary import summary_request
from .test_summary_work_budget import context, response, IDENTITY, history, finished


@pytest.mark.parametrize("target", [1500, 3000])
def test_configured_target_guides_whole_replacement_not_generation_allowance(target):
    config = {"summary_target_tokens": target, "summary_max_output_tokens": 32768,
              "summarization_model": "chosen", "summary_reasoning_effort": "low"}
    request = summary_request(SUMMARY_PROMPT, "NEXT source", "PAID previous note", config)
    prompt = request.messages[0].content
    assert f"approximately {target} tokens" in prompt
    assert "entire visible continuation note" in prompt
    assert "not additional tokens per fragment" in prompt
    assert "replaces the previous note" in prompt
    assert "not truncate" in prompt
    assert request.messages[1].content.count("PAID previous note") == 1
    assert request.messages[1].content.endswith("NEXT source")
    assert request.max_output_tokens == 32768
    assert request.model == "chosen" and request.reasoning_effort == "low"
    assert request.stream is False
    assert config["summary_target_tokens"] == target


def test_absent_target_retains_exact_legacy_request():
    assert summary_request(SUMMARY_PROMPT, "SOURCE", "OLD", {}).messages[0].content == SUMMARY_PROMPT


@pytest.mark.parametrize("target", [0, -1, True, None, "1500", 1.5])
@pytest.mark.asyncio
async def test_invalid_newly_active_target_fails_before_count_or_completion(target):
    manager = await context(summary_target_tokens=target)
    model = SimpleNamespace(complete=AsyncMock(), request_budget=AsyncMock())
    with pytest.raises((CompactionError, ValueError), match="summary_target_tokens"):
        await manager._prepare(model, None, [])
    model.complete.assert_not_awaited()
    model.request_budget.assert_not_awaited()
    assert await manager.get_messages() == history()


@pytest.mark.asyncio
async def test_every_fragment_rewrites_whole_note_and_does_not_clip_long_complete_output():
    manager = await context(summary_target_tokens=15, summary_max_output_tokens=32768)
    original = await manager.get_messages()
    calls = []

    async def complete(request):
        calls.append(request)
        assert "approximately 15 tokens" in request.messages[0].content
        if len(calls) > 1:
            assert ("COMPLETE " + str(len(calls) - 1) + " " + "evidence " * 30) in request.messages[1].content
        return response("COMPLETE " + str(len(calls)) + " " + "evidence " * 30)

    await manager._prepare(SimpleNamespace(complete=complete), None, [])
    assert len(calls) > 2
    assert manager.summary[1] == "COMPLETE " + str(len(calls)) + " " + "evidence " * 30
    assert await manager.get_messages() == original
    assert {call.max_output_tokens for call in calls} == {32768}


@pytest.mark.asyncio
async def test_prompt_bound_partial_exports_v5_and_resumes_without_repeating_paid_piece():
    first = await context(summary_target_tokens=1500, summary_max_calls=1)
    model = SimpleNamespace(name="fixture", default_model="fixture-model",
                            complete=AsyncMock(return_value=response("PAID complete part")))
    with pytest.raises(CompactionError):
        await first._prepare(model, None, [])
    record = first.export_checkpoint(IDENTITY)
    assert record["configurationVersion"] == 5
    assert record["progress"]["generation_contract"]["prompt_sha256"] == digest(model.complete.call_args.args[0].messages[0].content)
    second = await context(summary_target_tokens=1500)
    assert second.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
    model.complete = AsyncMock(return_value=response())
    await second._prepare(model, None, [])
    assert "PAID complete part" in model.complete.call_args_list[0].args[0].messages[1].content
    assert finished(second)[-1]["resumed_fragments"] == 1
    assert second.summary and await second.get_messages() == history()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["different_digest", "missing_digest", "extra_key", "invalid_digest"])
async def test_v5_contract_is_strict_and_changed_prompt_cannot_resume(change):
    first = await context(summary_target_tokens=1500, summary_max_calls=1)
    model = SimpleNamespace(name="fixture", default_model="fixture-model", complete=AsyncMock(return_value=response()))
    with pytest.raises(CompactionError):
        await first._prepare(model, None, [])
    record = first.export_checkpoint(IDENTITY)
    contract = record["progress"]["generation_contract"]
    if change == "different_digest": contract["prompt_sha256"] = "0" * 64
    elif change == "missing_digest": contract.pop("prompt_sha256")
    elif change == "extra_key": contract["unknown"] = True
    else: contract["prompt_sha256"] = "not-a-digest"
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    second = await context(summary_target_tokens=1500)
    status = second.restore_checkpoint(record, IDENTITY)
    if change != "different_digest":
        assert status["status"] == "rejected"
        return
    assert status["status"] == "resuming"
    model.complete = AsyncMock(return_value=response())
    await second._prepare(model, None, [])
    assert finished(second)[-1]["resumed_fragments"] == 0
    assert any("prompt" in call.args[1].get("reason", "").lower() for call in second.hooks.emit.call_args_list
               if call.args[0] == "context:checkpoint_progress_rejected")


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["max_tokens", "length", "incomplete"])
async def test_third_output_limit_retains_two_complete_pieces_and_originals_without_retry(finish):
    manager = await context(summary_target_tokens=1500, summary_max_output_tokens=32768)
    limited = response("TRUNCATED third note"); limited.finish_reason = finish
    model = SimpleNamespace(name="fixture", default_model="fixture-model", complete=AsyncMock(
        side_effect=[response("FIRST complete"), response("SECOND complete"), limited]))
    saved = []
    manager.persist_checkpoint = lambda: saved.append(copy.deepcopy(manager.export_checkpoint(IDENTITY)))
    with pytest.raises(CompactionError, match="output allowance"):
        await manager._prepare(model, None, [])
    record = manager.export_checkpoint(IDENTITY)
    assert model.complete.await_count == 3
    assert len(saved) == 2
    assert record["progress"]["completed"] == 2 and record["progress"]["calls"] == 3
    assert record["progress"]["text"] == "SECOND complete"
    assert record["summary"] is None
    assert await manager.get_messages() == history()
    with pytest.raises(CompactionError, match="previously failed"):
        await manager._prepare(model, None, [])
    assert model.complete.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["completed", "completed_with_partial"])
async def test_frozen_396325_completed_note_remains_usable_without_generation(name):
    records = json.loads(Path(__file__).with_name("summary_note_target_legacy.json").read_text())
    fixture = records[name]
    assert records["source_commit"] == "396325a6f9af7dfa96718c9a84fc1b36b74a8681"
    manager = BoundaryContextManager(fixture["config"])
    manager.checkpoint_identity = lambda: copy.deepcopy(IDENTITY)
    await manager.set_messages(fixture["messages"])
    original = await manager.get_messages()
    status = manager.restore_checkpoint(fixture["checkpoint"], IDENTITY)
    assert status["status"] in {"restored", "resuming"}
    before = copy.deepcopy(manager.summary)
    model = SimpleNamespace(name="fixture", default_model="fixture-model", complete=AsyncMock())
    await manager._prepare(model, None, [])
    model.complete.assert_not_awaited()
    assert manager.summary == before and await manager.get_messages() == original


@pytest.mark.asyncio
async def test_frozen_396325_partial_rejected_under_new_policy_without_skipping_source():
    fixtures = json.loads(Path(__file__).with_name("summary_note_target_legacy.json").read_text())
    fixture = fixtures["partial_only"]
    # The frozen record stopped after one call. This inert work-per-pass option
    # permits the test to finish recomputation without changing note identity.
    manager = BoundaryContextManager({**fixture["config"], "summary_max_calls": 16},
                                     SimpleNamespace(emit=AsyncMock()))
    manager.checkpoint_identity = lambda: copy.deepcopy(IDENTITY)
    await manager.set_messages(fixture["messages"])
    assert manager.restore_checkpoint(fixture["checkpoint"], IDENTITY)["status"] == "resuming"
    model = SimpleNamespace(name="fixture", default_model="fixture-model", complete=AsyncMock(return_value=response()))
    await manager._prepare(model, None, [])
    assert "LEGACY paid note" not in model.complete.call_args_list[0].args[0].messages[1].content
    assert finished(manager)[-1]["resumed_fragments"] == 0
    assert await manager.get_messages() == fixture["messages"]


@pytest.mark.asyncio
async def test_legacy_failure_fingerprint_does_not_latch_corrected_prompt_policy():
    fixture = json.loads(Path(__file__).with_name("summary_note_target_legacy.json").read_text())["partial_only"]
    manager = BoundaryContextManager({**fixture["config"], "summary_max_calls": 16},
                                     SimpleNamespace(emit=AsyncMock()))
    manager.checkpoint_identity = lambda: copy.deepcopy(IDENTITY)
    await manager.set_messages(fixture["messages"])
    old_progress = fixture["checkpoint"]["progress"]
    manager.summary_failure = {"fingerprint": old_progress["fingerprint"],
        "source_revision": old_progress["source_revision"], "retryable": False}
    model = SimpleNamespace(name="fixture", default_model="fixture-model",
                            complete=AsyncMock(return_value=response()))
    await manager._prepare(model, None, [])
    assert model.complete.await_count > 1 and manager.summary is not None
    assert await manager.get_messages() == fixture["messages"]


@pytest.mark.asyncio
async def test_completed_plus_old_partial_keeps_completed_note_when_corrected_work_fails():
    fixture = json.loads(Path(__file__).with_name("summary_note_target_legacy.json").read_text())["completed_with_partial"]
    original_record = copy.deepcopy(fixture["checkpoint"])
    manager = BoundaryContextManager({**fixture["config"], "summarize_trigger": .01,
                                      "summary_max_calls": 16}, SimpleNamespace(emit=AsyncMock()))
    manager.checkpoint_identity = lambda: copy.deepcopy(IDENTITY)
    await manager.set_messages(fixture["messages"])
    assert manager.restore_checkpoint(fixture["checkpoint"], IDENTITY)["status"] == "resuming"
    completed = copy.deepcopy(manager.summary)
    limited = response("TRUNCATED replacement"); limited.finish_reason = "max_tokens"
    model = SimpleNamespace(name="fixture", default_model="fixture-model",
                            complete=AsyncMock(return_value=limited))
    with pytest.raises(CompactionError, match="output allowance"):
        await manager._prepare(model, None, [])
    assert model.complete.await_count == 1
    assert "LEGACY next paid note" not in model.complete.call_args.args[0].messages[1].content
    assert manager.summary == completed
    assert manager.export_checkpoint(IDENTITY)["summary"] == original_record["summary"]
    assert fixture["checkpoint"] == original_record
    assert await manager.get_messages() == fixture["messages"]


@pytest.mark.asyncio
async def test_target_prompt_is_counted_and_dispatched_as_one_unchanged_request():
    manager = await context(summary_target_tokens=1500, summary_max_output_tokens=32768)
    counts, calls = [], []

    async def count(request, **kwargs):
        counts.append(request.model_dump())
        assert "approximately 1500 tokens" in request.messages[0].content
        return {"estimated_input_tokens": 100, "input_limit_tokens": 1000000,
                "max_output_tokens": 32768,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 100}}

    async def complete(request):
        assert request.model_dump() == counts[-1]
        calls.append(request.model_dump())
        return response()

    await manager._prepare(SimpleNamespace(complete=complete, request_budget=count), None, [])
    # Local source-cap probes may be counted and split without dispatch. Each
    # paid completion above still requires exact equality with its latest probe.
    assert len(calls) > 1 and len(counts) >= len(calls)
    assert all(request["max_output_tokens"] == 32768 for request in calls)
