import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from amplifier_module_context_managed.boundary import BoundaryContextManager, mount_boundary

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
