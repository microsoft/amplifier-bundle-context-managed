"""A later admitted preparation must not repay an unchanged summary prefix."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_module_context_managed.checkpoint import digest
from amplifier_module_context_managed.errors import CompactionError
from .test_summary_work_budget import context, response, IDENTITY, finished


def model():
    return SimpleNamespace(name="fixture", default_model="fixture-model",
                           complete=AsyncMock(return_value=response()))


async def large_context(**options):
    manager = await context(**options)
    rows = await manager.get_messages()
    rows[2]["content"] = "START evidence " + "".join(f"row-{n:05d};" for n in range(4000)) + " END evidence"
    await manager.set_messages(rows)
    return manager


async def append_tail(manager):
    rows = await manager.get_messages()
    rows += [{"role": "assistant", "content": "No additional operations."},
             {"role": "user", "content": "Continue the saved summary only."}]
    await manager.set_messages(rows)
    return rows


@pytest.mark.asyncio
async def test_default_sixteen_is_a_nonretryable_pause_not_a_provider_failure():
    manager = await large_context()
    provider = model()
    saved = []
    manager.persist_checkpoint = lambda: saved.append(copy.deepcopy(manager.export_checkpoint(IDENTITY)))
    originals = await manager.get_messages()
    with pytest.raises(CompactionError) as error:
        await manager._prepare(provider, None, [])
    assert error.value.code == "summary_work_paused"
    assert error.value.retryable is False
    assert provider.complete.await_count == 16
    assert len(saved) == 16 and saved[-1]["progress"]["completed"] == 16
    assert manager.summary is None and manager.summary_failure is None
    assert manager.checkpoint_status["status"] == "paused"
    event = finished(manager)[-1]
    assert event["outcome"] == "paused" and "failure" not in event
    assert event["calls_this_pass"] == event["call_allowance"] == 16
    assert event["summary_calls_total"] == 16
    assert event["source_chars_completed"] + event["source_chars_pending"] == event["source_chars_total"]
    assert event["source_chars_pending"] > 0 and event["partition_final"] is False
    assert await manager.get_messages() == originals


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("completed_calls", [2, 16])
async def test_append_finishes_original_job_without_rebinding_or_repeating_paid_pieces(restart, completed_calls):
    first = await large_context(summary_max_calls=completed_calls)
    provider = model()
    with pytest.raises(CompactionError):
        await first._prepare(provider, None, [])
    record = copy.deepcopy(first.export_checkpoint(IDENTITY))
    checkpoint_hash = digest(record)
    old_fingerprint = record["progress"]["fingerprint"]
    old_end = record["progress"]["source_revision"]["messages"]
    paid = [c.args[0].model_dump() for c in provider.complete.call_args_list]
    manager = await large_context(summary_max_calls=128) if restart else first
    if restart:
        assert manager.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
    else:
        manager.config["summary_max_calls"] = 128  # Explicit caller allowance; compatibility-inert.
    rows = await append_tail(manager)
    assert manager._boundary(rows, []) > old_end
    provider.complete.reset_mock()
    fitter, _ = await manager._prepare(provider, None, [])
    assert manager.summary[0] == old_end  # Appended source is NOT silently added to this job.
    assert finished(manager)[-1]["resumed_fragments"] == completed_calls
    assert all(c.args[0].model_dump() not in paid for c in provider.complete.call_args_list)
    assert manager._summary_progress is None and manager.summary_failure is None
    assert digest(record) == checkpoint_hash
    assert record["progress"]["fingerprint"] == old_fingerprint
    assert await manager.get_messages() == rows
    assert (await fitter.get_messages())[-len(rows[old_end:]):] == rows[old_end:]


@pytest.mark.asyncio
@pytest.mark.parametrize("difference", ["fingerprint", "parts", "segments", "provider", "generation", "required"])
async def test_new_tail_does_not_silently_rebind_incompatible_paid_job(difference):
    first = await large_context(summary_max_calls=2)
    provider = model()
    with pytest.raises(CompactionError):
        await first._prepare(provider, None, [])
    record = copy.deepcopy(first.export_checkpoint(IDENTITY))
    if difference == "fingerprint": record["progress"]["fingerprint"] = "0" * 64
    elif difference == "parts": record["progress"]["parts_digest"] = "0" * 64
    elif difference == "segments": record["progress"]["segments"].insert(0, record["progress"]["segments"][0])
    elif difference == "provider": record["progress"]["provider"][0] = "other-utility"
    elif difference == "generation": record["progress"]["generation_contract"]["max_output_tokens"] += 1
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    manager = await large_context()
    assert manager.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
    rows = await append_tail(manager)
    before = digest(record)
    provider.complete.reset_mock()
    with pytest.raises(CompactionError) as error:
        await manager._prepare(provider, None, [rows[1]["content"]] if difference == "required" else [])
    assert error.value.code == "summary_progress_incompatible"
    assert not provider.complete.await_count
    assert digest(record) == before and await manager.get_messages() == rows
    assert manager.summary is None


@pytest.mark.asyncio
async def test_split_progress_conserves_characters_and_never_calls_it_fixed_total_work():
    manager = await large_context(summary_max_calls=2)
    provider = model()
    with pytest.raises(CompactionError):
        await manager._prepare(provider, None, [])
    progress = [c.args[1] for c in manager.hooks.emit.call_args_list
                if c.args[0] == "context:compaction_progress"]
    assert len(progress) == 2
    for data in progress:
        assert data["source_chars_completed"] + data["source_chars_pending"] == data["source_chars_total"]
        assert data["pending_segments"] == data["remaining_parts"]
        assert data["largest_pending_segment_chars"] > 0
        assert data["partition_final"] is False
        assert data["source_character_basis"] == "serialized_summary_source_parts"
    assert progress[1]["source_chars_completed"] > progress[0]["source_chars_completed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("advance", [False, True])
@pytest.mark.parametrize("difference", ["prefix", "identity", "config", "model", "provider", "prompt"])
async def test_all_paid_contract_mismatches_refuse_without_replacement(restart, advance, difference):
    manager = await large_context(summary_max_calls=2)
    provider = model()
    with pytest.raises(CompactionError):
        await manager._prepare(provider, None, [])
    record = copy.deepcopy(manager.export_checkpoint(IDENTITY))
    if restart:
        second = await large_context(summary_max_calls=2)
        assert second.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
        manager = second
    if advance:
        await append_tail(manager)
    if difference == "prefix":
        rows = await manager.get_messages()
        rows[2]["content"] += " A changed covered fact."
        await manager.set_messages(rows)
    elif difference == "identity":
        manager.checkpoint_identity = lambda: {**IDENTITY, "model": "changed"}
    elif difference == "config":
        manager.config["summary_reasoning_effort"] = "high"
    elif difference == "model":
        provider.default_model = "changed"
    elif difference == "provider":
        provider.name = "changed"
    else:
        manager.config["summary_target_tokens"] = 1500
    originals = await manager.get_messages()
    before = copy.deepcopy(manager._summary_progress or manager._restored_progress)
    provider.complete.reset_mock()
    writes = []
    manager.persist_checkpoint = lambda: writes.append(True)
    with pytest.raises(CompactionError, match="summary_progress_incompatible") as caught:
        await manager._prepare(provider, None, [])
    assert type(caught.value) is CompactionError and caught.value.retryable is False
    provider.complete.assert_not_awaited()
    assert not writes
    assert (manager._summary_progress or manager._restored_progress) == before
    assert await manager.get_messages() == originals
    assert record["progress"]["completed"] == 2


class CountedFragments:
    """Offline count oracle; refusals split whole input lazily, not in advance."""
    name = "fixture"
    default_model = "fixture-model"

    def __init__(self):
        self.counted = []
        self.completed = []
        self.accepted_sources = []

    async def request_budget(self, request, **unused):
        self.counted.append(copy.deepcopy(request.model_dump()))
        raw = sum(len(row.content) for row in request.messages)
        return {"estimated_input_tokens": raw + 4096, "input_limit_tokens": 45000,
                "max_output_tokens": request.max_output_tokens,
                "measurement": {"kind": "provider_count", "source": "offline-fixture", "input_tokens": raw}}

    async def complete(self, request):
        body = request.model_dump()
        assert body == self.counted[-1]  # Same rendered, counted request is sent.
        assert request.max_output_tokens == 32768 and request.reasoning_effort == "low"
        assert "approximately 1500 tokens" in request.messages[0].content
        assert sum(len(row.content) for row in request.messages) + 4096 + 32768 <= 45000
        self.completed.append(body)
        source = request.messages[-1].content.split(
            "Next ordered conversation data. Incorporate this evidence into the continuation note:\n", 1)[1]
        self.accepted_sources.append(source)
        return response()


@pytest.mark.asyncio
async def test_lazy_unequal_ancestors_pause_at_sixteen_and_restore_exact_coverage():
    manager = await large_context(summary_max_output_tokens=32768, summary_target_tokens=1500)
    manager.config.pop("summary_max_source_chars")  # One whole initial source, like the incident.
    rows = await manager.get_messages()
    rows[2]["content"] = "".join(f"fact-{n:06d};" for n in range(40000))
    await manager.set_messages(rows)
    original = copy.deepcopy(rows)
    provider = CountedFragments()
    with pytest.raises(CompactionError, match="summary_work_paused"):
        await manager._prepare(provider, None, [])
    assert len(provider.completed) == 16
    record = manager.export_checkpoint(IDENTITY)
    pending = record["progress"]["segments"]
    assert len(pending) >= 2 and {row[0] for row in pending} == {0}
    assert max(z-a for _, a, z in pending) > min(z-a for _, a, z in pending)
    first = finished(manager)[-1]
    assert first["source_chars_completed"] < first["source_chars_pending"]
    assert first["source_chars_completed"] == sum(map(len, provider.accepted_sources))
    covered = first["source_chars_completed"]
    assert pending[0][1] == covered
    assert pending[-1][2] == first["source_chars_total"]
    assert manager.summary is None
    restored = await large_context(summary_max_output_tokens=32768, summary_target_tokens=1500)
    restored.config.pop("summary_max_source_chars")
    await restored.set_messages(original)
    assert restored.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
    await append_tail(restored)
    old_end = record["progress"]["source_revision"]["messages"]
    # Each call below is an explicit later offline preparation, not an automatic
    # provider retry or an authorization policy implemented by this module.
    for _ in range(10):
        before = len(provider.completed)
        try:
            await restored._prepare(provider, None, [])
        except CompactionError as error:
            assert error.code == "summary_work_paused" and error.retryable is False
        assert len(provider.completed) - before <= 16
        current = finished(restored)[-1]
        assert current["source_chars_completed"] == sum(map(len, provider.accepted_sources))
        assert current["source_chars_completed"] + current["source_chars_pending"] == first["source_chars_total"]
        if restored.summary:
            break
    assert restored.summary[0] == old_end
    assert len(provider.completed) > 16
    assert len(set(map(digest, provider.completed))) == len(provider.completed)
    from amplifier_module_context_managed.summary import public_messages, source_fragments
    original_parts = list(source_fragments(public_messages(original[:old_end]), None))
    assert record["progress"]["parts_digest"] == digest(original_parts)
    assert "".join(provider.accepted_sources) == "".join(original_parts)
    assert current["source_chars_pending"] == 0
    assert (await restored.get_messages())[:len(original)] == original


@pytest.mark.asyncio
async def test_public_final_measured_fit_refuses_large_tail_after_finishing_old_job_once():
    from amplifier_core import ChatRequest, Message
    from amplifier_core.llm_errors import ContextLengthError

    manager = await large_context(summary_max_calls=2)
    provider = model()
    with pytest.raises(CompactionError):
        await manager._prepare(provider, None, [])
    record = manager.export_checkpoint(IDENTITY)
    old_end = record["progress"]["source_revision"]["messages"]
    manager.config["summary_max_calls"] = 128  # Explicit synthetic caller allowance.
    rows = await append_tail(manager)
    rows[-2]["content"] = "LARGE UNSUMMARIZED TAIL " + "z" * 100000
    await manager.set_messages(rows)
    counted = []

    async def count_view(view):
        counted.append(copy.deepcopy(view))
        raw = sum(len(row.get("content") or "") for row in view)
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]),
                "budget_decision": {"estimated_input_tokens": raw, "input_limit_tokens": 10000,
                    "measurement": {"kind": "provider_count", "source": "offline", "input_tokens": raw}}}

    provider.complete.reset_mock()
    with pytest.raises(ContextLengthError):
        await manager.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    assert manager.summary[0] == old_end and manager._summary_progress is None
    events = finished(manager)
    assert len(events) == 2 and events[-1]["outcome"] == "completed"
    assert events[-1]["resumed_fragments"] == 2
    assert len(provider.complete.call_args_list) == events[-1]["completed_fragments_this_pass"]
    assert all((call.args[0].metadata or {}).get("purpose") == "context-compaction"
               for call in provider.complete.call_args_list)  # No foreground send.
    assert counted[-1][-len(rows[old_end:]):] == rows[old_end:]
    assert await manager.get_messages() == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [4, 5])
@pytest.mark.parametrize("base", ["partial_only", "completed_with_partial"])
async def test_frozen_accepted_v4_v5_progress_resumes_without_replaying_baseline_piece(version, base):
    from amplifier_module_context_managed.boundary import BoundaryContextManager

    packet = json.loads(Path(__file__).with_name("summary_work_pause_baseline.json").read_text())
    assert packet["source_commit"] == "0de9bd2ea366e0ad10a3fd668c1d62c398baaefd"
    fixture = packet["fixtures"][f"v{version}_{base}"]
    record = fixture["checkpoint"]
    before = digest(fixture)
    assert record["configurationVersion"] == version
    assert record["sha256"] == digest({k: v for k, v in record.items() if k != "sha256"})
    manager = BoundaryContextManager({**fixture["config"], "summary_max_calls": 16},
                                     SimpleNamespace(emit=AsyncMock()))
    manager.checkpoint_identity = lambda: copy.deepcopy(fixture["identity"])
    await manager.set_messages(fixture["messages"])
    assert manager.restore_checkpoint(record, fixture["identity"])["status"] == "resuming"
    previous = copy.deepcopy(manager.summary)
    provider = model()
    await manager._prepare(provider, None, [])
    assert finished(manager)[-1]["resumed_fragments"] == 1
    # Pending ancestors can split again under current preflight. Prove exact
    # remaining bytes, not a fixed number of future calls.
    from amplifier_module_context_managed.summary import public_messages, source_fragments
    carry = manager._carry(fixture["messages"])
    source = manager._assembled(fixture["messages"][:record["progress"]["source_revision"]["messages"]],
                               previous, [], carry)
    parts = list(source_fragments(public_messages(source), fixture["config"]["summary_max_source_chars"]))
    assert digest(parts) == record["progress"]["parts_digest"]
    expected = "".join(parts[i][a:z] for i, a, z in record["progress"]["segments"])
    actual = "".join(call.args[0].messages[-1].content.split(
        "Next ordered conversation data. Incorporate this evidence into the continuation note:\n", 1)[1]
        for call in provider.complete.call_args_list)
    assert actual == expected
    assert record["progress"]["text"] in provider.complete.call_args_list[0].args[0].messages[-1].content
    baseline_requests = {row["request_sha256"] for row in fixture["successful_paid_requests"]}
    assert all(digest(row.args[0].model_dump()) not in baseline_requests
               for row in provider.complete.call_args_list)
    assert manager.summary[0] == record["progress"]["source_revision"]["messages"]
    assert (previous is not None) == (base == "completed_with_partial")
    if previous:
        assert previous[0] < manager.summary[0]
    assert digest(fixture) == before and await manager.get_messages() == fixture["messages"]
