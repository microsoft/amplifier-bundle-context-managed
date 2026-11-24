"""Only a verified plain note enters an uncounted target request, without adopting it."""
import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.checkpoint import digest

from .test_summary_work_budget import IDENTITY, context, history, response


TARGET = {**IDENTITY, "model": "other"}


def model(identity):
    return SimpleNamespace(name=identity["provider"], default_model=identity["model"],
                           complete=AsyncMock(return_value=response()))


async def original():
    manager = await context()
    generated = Mock()
    manager._host_capability = {"context.portable_handoff.generated": generated}.get
    provider = model(IDENTITY)
    await manager.get_messages_for_request(provider=provider)
    generated.assert_called_once()
    witness, record = generated.call_args.args
    assert witness is provider and record == manager.export_checkpoint(IDENTITY)
    return manager, record


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [False, True])
async def test_only_target_success_can_promote_the_staged_note(warm):
    first, saved = await original()
    unchanged = copy.deepcopy(saved)
    manager = first if warm else await context()
    selected = dict(TARGET)
    manager.checkpoint_identity = lambda: dict(selected)
    stage, generated = Mock(), Mock()
    offer = AsyncMock(return_value=(saved, stage))
    manager._host_capability = {"context.portable_handoff.source": offer,
                                "context.portable_handoff.generated": generated}.get
    provider = model(TARGET)
    result = await manager.get_messages_for_request(provider=provider)
    provider.complete.assert_not_awaited()
    generated.assert_not_called()
    offer.assert_awaited_once()
    candidate, note, guard = stage.call_args.args
    assert sum(row.get("content") == note for row in result) == 1
    assert saved["summary"]["text"] in note
    assert candidate["identity"] == TARGET
    assert (candidate["summary"], candidate["sourceRevision"]) == (saved["summary"], saved["sourceRevision"])
    assert manager.summary_identity == (IDENTITY if warm else None)
    assert saved == unchanged and await manager.get_messages() == history()

    manager._is_cancelled = lambda: True
    with pytest.raises(asyncio.CancelledError):
        guard()
    manager._is_cancelled = None
    selected["model"] = "changed"
    with pytest.raises(RuntimeError, match="handoff"):
        guard()
    selected.update(TARGET)
    guard()
    assert manager.restore_checkpoint(candidate, TARGET)["status"] == "restored"
    assert manager.summary_identity == TARGET and manager.export_checkpoint(TARGET) == candidate


@pytest.mark.asyncio
@pytest.mark.parametrize("difference", ["account", "digest", "config", "prefix", "native", "partial"])
async def test_unqualified_source_never_stages_or_relabels_a_note(difference):
    _, record = await original()
    params = {"summary_retry_delay": 0}
    manager = await context(**params)
    if difference == "account":
        record["identity"]["provider"] = "other-account"
    elif difference == "config":
        record["configuration"] = "different"
    elif difference == "prefix":
        rows = history()
        rows[0]["content"] += " changed"
        await manager.set_messages(rows)
    elif difference == "native":
        record["summary"] = {"throughMessage": record["sourceRevision"]["messages"],
                             "kind": "native", "message": {"role": "user", "content": "opaque"}}
    elif difference == "partial":
        record["progress"] = {"completed": 1}
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    if difference == "digest":
        record["sha256"] = "wrong"
    stage = Mock()
    offer = AsyncMock(return_value=(record, stage))
    manager._host_capability = {"context.portable_handoff.source": offer}.get
    manager.checkpoint_identity = lambda: copy.deepcopy(TARGET)
    provider = model(TARGET)
    await manager.get_messages_for_request(provider=provider)
    stage.assert_not_called()
    assert provider.complete.await_count > 0


@pytest.mark.asyncio
async def test_paid_partial_never_becomes_evidence_of_a_new_creating_connection():
    from .test_resumable_work_pause import frozen_fixture
    fixture = frozen_fixture("partial_only")
    manager = BoundaryContextManager({**fixture["config"], "summary_max_calls": 16})
    manager.checkpoint_identity = lambda: copy.deepcopy(fixture["identity"])
    await manager.set_messages(fixture["messages"])
    assert manager.restore_checkpoint(fixture["checkpoint"], fixture["identity"])["status"] == "resuming"
    generated, offer = Mock(), AsyncMock()
    manager._host_capability = {"context.portable_handoff.generated": generated,
                                "context.portable_handoff.source": offer}.get
    await manager.get_messages_for_request(provider=model(fixture["identity"]))
    generated.assert_not_called()
    offer.assert_not_awaited()
    assert manager.summary is not None


@pytest.mark.asyncio
async def test_later_compaction_reads_the_portable_note_and_new_history_only():
    _, saved = await original()
    manager = await context()
    manager.checkpoint_identity = lambda: copy.deepcopy(TARGET)
    provider = model(TARGET)
    stage = Mock()
    manager._host_capability = {"context.portable_handoff.source": AsyncMock(return_value=(saved, stage))}.get
    await manager.get_messages_for_request(provider=provider)
    candidate, _, guard = stage.call_args.args
    guard()
    assert manager.restore_checkpoint(candidate, TARGET)["status"] == "restored"
    manager._host_capability = None

    await manager.add_message({"role": "assistant", "content": "NEW evidence " + "y" * 4500})
    await manager.add_message({"role": "user", "content": "Next objective"})
    await manager.add_message({"role": "assistant", "content": "Acknowledged"})
    await manager.add_message({"role": "user", "content": "Continue"})
    await manager.get_messages_for_request(provider=provider)
    requests = [call.args[0].messages[-1].content for call in provider.complete.call_args_list]
    assert any("NEW evidence" in source for source in requests)
    assert any(saved["summary"]["text"] in source for source in requests)
    assert all("FIRST verified evidence" not in source for source in requests)
    assert manager.summary[0] > saved["summary"]["throughMessage"]


@pytest.mark.asyncio
async def test_measured_provider_keeps_its_existing_transition_owner():
    manager = await context()
    provider = model(IDENTITY)
    provider.request_budget = AsyncMock(return_value={})
    offer = AsyncMock()
    manager._host_capability = {"context.portable_handoff.source": offer}.get
    await manager.get_messages_for_request(provider=provider)
    offer.assert_not_awaited()
    assert provider.request_budget.await_count > 0


@pytest.mark.asyncio
async def test_target_fit_refusal_preserves_source_without_speculative_compaction():
    from amplifier_core.llm_errors import ContextLengthError

    manager, record = await original()
    source = copy.deepcopy(manager.summary)
    manager.checkpoint_identity = lambda: copy.deepcopy(TARGET)
    stage = Mock()
    manager._host_capability = {"context.portable_handoff.source": AsyncMock(return_value=(record, stage))}.get
    provider = model(TARGET)
    with pytest.raises(ContextLengthError):
        await manager.get_messages_for_request(provider=provider, token_budget=1)
    assert manager.summary == source and manager.summary_identity == IDENTITY
    provider.complete.assert_not_awaited()
    stage.assert_not_called()
