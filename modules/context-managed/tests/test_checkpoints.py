import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from amplifier_core import AmplifierSession, HookResult
from amplifier_module_context_managed.boundary import BoundaryContextManager, mount_boundary
from amplifier_module_context_managed.checkpoint import digest
from amplifier_module_context_managed.errors import CompactionError

IDENTITY = {"provider": "fixture", "model": "model-a"}
CONFIG = {"durable_checkpoints": True, "max_tokens": 6000, "summarize_trigger": .2}


def messages():
    return [{"role": "user", "content": "Build a report; preserve the original files."},
            {"role": "assistant", "content": "Evidence " * 1800},
            {"role": "user", "content": "Correction: use the revised constraints."},
            {"role": "assistant", "content": "Progress " * 100},
            {"role": "user", "content": "Continue independent work; question q-1 is unanswered."}]


def context():
    result = BoundaryContextManager(CONFIG)
    result.checkpoint_identity = lambda: copy.deepcopy(IDENTITY)
    return result


def provider():
    return SimpleNamespace(complete=AsyncMock(return_value=SimpleNamespace(content=[
        SimpleNamespace(type="text", text="Report objective; originals preserved; pending question q-1; operation op-1 outcome unknown.")])) )


@pytest.mark.asyncio
async def test_multiple_compactions_round_trip_with_exact_prefix_and_new_correction():
    first = context()
    original = messages()
    await first.set_messages(original)
    await first.get_messages_for_request(provider=provider())
    for i in range(3):
        await first.add_message({"role": "assistant", "content": "Additional evidence " * 1000})
        await first.add_message({"role": "user", "content": f"Latest correction {i}; keep artifact file-{i}.txt"})
        await first.get_messages_for_request(provider=provider())
    full = await first.get_messages()
    serialized = json.dumps(first.export_checkpoint(IDENTITY))
    restored = context()
    suffix = {"role": "user", "content": "Newest unsummarized correction"}
    await restored.set_messages(full + [suffix])
    assert restored.restore_checkpoint(json.loads(serialized), IDENTITY)["status"] == "restored"
    view = await restored.get_messages_for_request()
    assert "Newest unsummarized correction" in str(view)
    assert "Latest correction 2" in str(view)
    assert "operation op-1 outcome unknown" in str(view)
    assert await restored.get_messages() == full + [suffix]
    assert full[:len(original)] == original


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["prefix", "model", "format", "digest", "config", "shorter"])
async def test_incompatible_checkpoint_fails_visibly_without_replacing_originals(change):
    first = context()
    await first.set_messages(messages())
    await first.get_messages_for_request(provider=provider())
    record = first.export_checkpoint(IDENTITY)
    authoritative = messages()
    identity = copy.deepcopy(IDENTITY)
    target = context()
    if change == "prefix": authoritative[0]["content"] = "Revised objective"
    if change == "shorter": authoritative = authoritative[:1]
    if change == "model": identity["model"] = "model-b"
    if change == "format": record["format"] = "future"
    if change == "digest": record["summary"]["text"] = "Tampered note"
    if change == "config": target.config["summary_target_tokens"] = 99
    await target.set_messages(authoritative)
    result = target.restore_checkpoint(record, identity)
    assert result["status"] == "rejected" and result["originalsAvailable"]
    assert target.summary is None
    assert await target.get_messages() == authoritative


@pytest.mark.asyncio
async def test_host_commits_original_large_payload_before_fitter_and_checkpoint_refs():
    value = "original-large-output-" * 20000
    full = messages()
    full.insert(2, {"role": "tool", "tool_call_id": "large", "content": value})
    model = provider()
    target = context()
    committed = []
    async def preserve(rows):
        assert model.complete.await_count == 0
        committed.extend(copy.deepcopy(rows))
        return [{"kind": "transcript", "message": 2, "sha256": "fixture-reference"}]
    target.preserve_evidence = preserve
    await target.set_messages(full)
    await target.get_messages_for_request(provider=model)
    assert committed[2]["content"] == value
    assert (await target.get_messages())[2]["content"] == value
    record = target.export_checkpoint(IDENTITY)
    assert record["evidenceRefs"][0]["kind"] == "transcript"


@pytest.mark.asyncio
async def test_checkpoint_mount_is_opt_in_and_never_fits_without_preservation_host():
    for enabled in (False, True):
        capabilities = {}
        mounted = {}
        async def mount(name, item): mounted[name] = item
        coordinator = SimpleNamespace(mount=mount, hooks=None, get_capability=capabilities.get,
            register_capability=lambda name, value: capabilities.__setitem__(name, value), register_contributor=lambda *args: None)
        await mount_boundary(coordinator, {**CONFIG, "durable_checkpoints": enabled})
        assert ("context.checkpoint.restore" in capabilities) == enabled
        await mounted["context"].set_messages(messages())
        if enabled:
            with pytest.raises(RuntimeError, match="preserve_evidence"):
                await mounted["context"].get_messages_for_request(provider=provider())
        else:
            assert mounted["context"].export_checkpoint(IDENTITY) is None


async def mounted_checkpoint_context():
    session = AmplifierSession({"session": {
        "orchestrator": "loop-streaming", "context": "context-managed"}})
    coordinator = session.coordinator
    events = []

    async def observe(kind, data):
        events.append((kind, copy.deepcopy(data)))
        return HookResult()

    async def preserve(rows):
        return [{"kind": "fixture", "sha256": digest(copy.deepcopy(rows))}]

    for kind in ("context:compaction_progress", "context:compaction_finished"):
        coordinator.hooks.register(kind, observe)
    coordinator.register_capability("context.checkpoint_identity", lambda: copy.deepcopy(IDENTITY))
    coordinator.register_capability("context.preserve_evidence", preserve)
    await mount_boundary(coordinator, {**CONFIG,
        "summary_max_source_chars": 1200, "summary_max_calls": 64})
    target = coordinator.get("context")
    await target.set_messages(messages())
    return session, target, events


def checkpoint_provider():
    model = provider()
    model.complete.return_value.usage = {"input_tokens": 100, "output_tokens": 10}
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ack", [None, True])
async def test_optional_checkpoint_callback_accepts_compatible_acknowledgements(asynchronous, ack):
    session, target, events = await mounted_checkpoint_context()
    saved = []

    def persist():
        saved.append(target.export_checkpoint(IDENTITY))
        return ack

    async def persist_async():
        return persist()

    session.coordinator.register_capability("context.persist_checkpoint",
        persist_async if asynchronous else persist)
    model = checkpoint_provider()
    await target.get_messages_for_request(provider=model)
    assert len(saved) == model.complete.await_count > 1
    assert [record["progress"]["completed"] for record in saved] == list(range(1, len(saved) + 1))
    assert target.summary is not None
    assert await target.get_messages() == messages()
    assert events[-1][1]["outcome"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raises", [False, True])
async def test_failed_checkpoint_callback_rolls_back_first_piece(asynchronous, raises):
    session, target, events = await mounted_checkpoint_context()
    staged = {}

    def persist():
        progress = target._summary_progress
        staged.update(progress=progress, pending=progress["pending"], segments=progress["segments"])
        assert target.export_checkpoint(IDENTITY)["progress"]["completed"] == 1
        if raises:
            raise OSError("private-payload")
        return False

    async def persist_async():
        return persist()

    session.coordinator.register_capability("context.persist_checkpoint",
        persist_async if asynchronous else persist)
    model = checkpoint_provider()
    with pytest.raises(CompactionError, match="checkpoint_persist_failed") as caught:
        await target.get_messages_for_request(provider=model)
    assert caught.value.code == "checkpoint_persist_failed"
    assert "private-payload" not in str(caught.value) + json.dumps(events)
    progress = staged["progress"]
    assert progress["pending"] is staged["pending"]
    assert progress["segments"] is staged["segments"]
    first = json.dumps(messages()[0], ensure_ascii=False, separators=(",", ":"))
    assert progress["pending"][0] == first
    assert progress["segments"][0] == [0, 0, len(first)]
    assert progress["text"] == "" and progress["completed"] == 0
    assert progress["calls"] == model.complete.await_count == 1
    assert target.summary is None and target.export_checkpoint(IDENTITY) is None
    assert await target.get_messages() == messages()
    assert not [data for kind, data in events if kind == "context:compaction_progress"]
    final = events[-1][1]
    assert final["outcome"] == "failed"
    assert final["failure"]["code"] == "checkpoint_persist_failed"
    assert final["completed_fragments"] == final["completed_fragments_this_pass"] == 0
    assert final["remaining_fragments"] == len(progress["pending"])
    assert final["calls"] == final["summary_calls_total"] == final["usage_calls"] == 1
    assert final["usage"] == {"input_tokens": 100, "output_tokens": 10}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    False, OSError("private-payload"), asyncio.CancelledError(),
    CompactionError("host_checkpoint_failed", "Restore checkpoint storage and retry.")])
async def test_late_checkpoint_failure_preserves_accepted_progress_and_host_file(tmp_path, failure):
    session, target, events = await mounted_checkpoint_context()
    previous_summary = target.summary = (1, "Previously accepted continuation note.")
    target.summary_identity = copy.deepcopy(IDENTITY)
    path = tmp_path / "checkpoint.json"
    accepted = {}
    attempts = 0

    async def persist():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            progress = target._summary_progress
            record = target.export_checkpoint(IDENTITY)
            candidate = tmp_path / "checkpoint.next"
            candidate.write_text(json.dumps(record), encoding="utf-8")
            candidate.replace(path)
            accepted.update(progress=progress, pending=progress["pending"],
                segments=progress["segments"], record=record, contents=path.read_bytes())
            return True
        assert target._summary_progress is accepted["progress"]
        if isinstance(failure, BaseException):
            raise failure
        return failure

    session.coordinator.register_capability("context.persist_checkpoint", persist)
    model = checkpoint_provider()
    first = model.complete.return_value
    second = copy.deepcopy(first)
    second.content[0].text = "Unaccepted replacement note."

    def complete(request):
        if model.complete.await_count == 1:
            return first
        # Preflight may split a remaining segment before making the next call.
        progress = target._summary_progress
        accepted["pending_before_failure"] = copy.deepcopy(progress["pending"])
        accepted["segments_before_failure"] = copy.deepcopy(progress["segments"])
        return second

    model.complete.side_effect = complete
    expected = asyncio.CancelledError if isinstance(failure, asyncio.CancelledError) else CompactionError
    with pytest.raises(expected) as caught:
        await target.get_messages_for_request(provider=model)
    if isinstance(failure, (asyncio.CancelledError, CompactionError)):
        assert caught.value is failure
    if isinstance(caught.value, CompactionError):
        assert caught.value.code == (
            failure.code if isinstance(failure, CompactionError) else "checkpoint_persist_failed")
        assert "private-payload" not in str(caught.value)
    assert "private-payload" not in json.dumps(events)
    progress = target._summary_progress
    assert progress is accepted["progress"]
    assert progress["pending"] is accepted["pending"]
    assert progress["segments"] is accepted["segments"]
    assert progress["pending"] == accepted["pending_before_failure"]
    assert progress["segments"] == accepted["segments_before_failure"]
    assert progress["text"] == accepted["record"]["progress"]["text"]
    assert progress["completed"] == 1 and progress["calls"] == model.complete.await_count == 2
    assert target.summary is previous_summary
    assert await target.get_messages() == messages()
    assert path.read_bytes() == accepted["contents"]
    record = json.loads(path.read_bytes())
    assert record["sha256"] == digest({key: value for key, value in record.items() if key != "sha256"})
    assert [data["completed_parts"] for kind, data in events
            if kind == "context:compaction_progress"] == [1]
    final = events[-1][1]
    assert final["outcome"] == ("cancelled" if expected is asyncio.CancelledError else "failed")
    assert final["completed_fragments"] == final["completed_fragments_this_pass"] == 1
    assert final["remaining_fragments"] == len(progress["pending"])
    assert final["calls"] == final["summary_calls_total"] == final["usage_calls"] == 2
    assert final["usage"] == {"input_tokens": 200, "output_tokens": 20}


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
async def test_optional_checkpoint_writer_can_be_absent_including_restored_view(mounted):
    if mounted:
        session, target, _ = await mounted_checkpoint_context()
        assert session.coordinator.get_capability("context.persist_checkpoint") is None
    else:
        target = context()
        await target.set_messages(messages())
        assert target.persist_checkpoint is None
    await target.get_messages_for_request(provider=checkpoint_provider())
    assert target.summary is not None
    record = target.export_checkpoint(IDENTITY)
    assert target.restore_checkpoint(record, IDENTITY)["status"] == "restored"
    assert await target.get_messages_for_request()
    assert await target.get_messages() == messages()
