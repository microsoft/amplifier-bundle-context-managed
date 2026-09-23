"""Canonical replacement must invalidate only checkpoints whose evidence changed."""
import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from amplifier_module_context_managed.boundary import BoundaryContextManager

IDENTITY = {"provider": "fixture", "model": "fixture-model"}


def history():
    return [
        {"role": "user", "content": "Preserve the original objective.", "metadata": {"flag": True}},
        {"role": "assistant", "content": "Verified historical evidence. " * 1000},
        {"role": "user", "content": "Correction: preserve all originals."},
        {"role": "assistant", "content": "Second turn completed."},
        {"role": "user", "content": "Finish the current task."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "current-call", "name": "delegate", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "current-call", "content": json.dumps(
            {"status": "pending", "job_id": "fixture-job"})},
    ]


def provider(method):
    native = {"role": "user", "content": "Derived native fixture", "metadata": {
        "source": "context-managed", "ephemeral": True, "persisted": True, "opaque": "fixture"}}
    model = SimpleNamespace(name=IDENTITY["provider"], default_model=IDENTITY["model"],
        complete=AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(type="text",
            text="Original objective; historical evidence verified; originals must be preserved.")])))
    if method == "native":
        model.supports_native_compaction = lambda: True
        model.validate_compacted_context = lambda row: bool((row.get("metadata") or {}).get("opaque"))
        model.compact_context = AsyncMock(return_value={"kind": "native", "message": native})
        model.request_budget = lambda request, **kw: {"measurement": {"kind": "provider_count",
            "input_tokens": 100 if any((row.metadata or {}).get("opaque") for row in request.messages) else 9000}}
    return model


def calls(model, method):
    return model.compact_context.await_count if method == "native" else model.complete.await_count


async def compacted(method):
    manager = BoundaryContextManager({"max_tokens": 10000, "summarize_trigger": .2,
                                      "durable_checkpoints": True})
    manager.checkpoint_identity = lambda: copy.deepcopy(IDENTITY)
    model = provider(method)
    await manager.set_messages(history())
    await manager._prepare(model, None, [])
    assert manager.summary[0] == 2
    assert calls(model, method) == 1
    return manager, model


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["native", "semantic"])
@pytest.mark.parametrize("change", ["identical", "result", "append", "shorter_tail", "only_prefix"])
async def test_exact_covered_prefix_survives_authoritative_suffix_replacement(method, change):
    manager, model = await compacted(method)
    replacement = await manager.get_messages()
    if change == "result": replacement[-1]["content"] = "Delegated task completed; result verified."
    if change == "append": replacement.append({"role": "assistant", "content": "Additional progress."})
    if change == "shorter_tail": replacement = replacement[:-2]
    if change == "only_prefix": replacement = replacement[:2]
    summary, identity, revision = copy.deepcopy(manager.summary), copy.deepcopy(manager.summary_identity), manager.revision
    manager.evidence_refs = [{"kind": "transcript", "sha256": "old-whole-history"}]
    await manager.set_messages(replacement)
    assert manager.summary == summary and manager.summary_identity == identity
    assert manager.checkpoint_status["status"] == "ready"
    assert manager.revision == revision + 1
    assert manager.evidence_refs == []
    await manager._prepare(model, None, [])
    assert calls(model, method) == 1
    assert await manager.get_messages() == replacement
    replacement[0]["content"] = "Caller-owned mutation"
    assert (await manager.get_messages())[0]["content"] != replacement[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["native", "semantic"])
@pytest.mark.parametrize("change", ["content", "metadata", "scalar_type", "order", "shorter_prefix", "clear"])
async def test_any_covered_canonical_change_discards_checkpoint(method, change):
    manager, _ = await compacted(method)
    replacement = await manager.get_messages()
    if change == "content": replacement[1]["content"] = "Corrected historical evidence."
    if change == "metadata": replacement[0]["metadata"]["new"] = "source identity changed"
    if change == "scalar_type": replacement[0]["metadata"]["flag"] = 1  # Python True == 1 is not canonical identity.
    if change == "order": replacement[:2] = reversed(replacement[:2])
    if change == "shorter_prefix": replacement = replacement[:1]
    if change == "clear": replacement = []
    await manager.set_messages(replacement)
    assert manager.summary is None and manager.summary_identity is None
    assert manager.checkpoint_status["status"] == "invalidated"
    assert manager.export_checkpoint(IDENTITY) is None
    assert await manager.get_messages() == replacement


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["native", "semantic"])
async def test_replacement_refreshes_evidence_and_checkpoint_restores_against_new_tail(method):
    manager, model = await compacted(method)
    original = await manager.get_messages()
    old_record = manager.export_checkpoint(IDENTITY)
    replacement = copy.deepcopy(original)
    replacement[-1]["content"] = "Exact new completed result"
    await manager.set_messages(replacement)
    async def preserve(rows):
        assert rows == replacement
        return [{"kind": "transcript", "sha256": "new-whole-history"}]
    manager.preserve_evidence = preserve
    await manager._prepare(model, None, [])
    record = manager.export_checkpoint(IDENTITY)
    assert record["sourceRevision"] == old_record["sourceRevision"]
    assert record["summary"] == old_record["summary"]
    assert record["evidenceRefs"] == [{"kind": "transcript", "sha256": "new-whole-history"}]
    restored = BoundaryContextManager(manager.config)
    restored.checkpoint_identity = manager.checkpoint_identity
    await restored.set_messages(replacement)
    assert restored.restore_checkpoint(record, IDENTITY)["status"] == "restored"
    await restored.set_messages(replacement + [{"role": "assistant", "content": "Further progress."}])
    assert restored.summary == manager.summary
    await restored._prepare(model, None, [])
    assert calls(model, method) == 1
    assert (await restored.get_messages())[-2]["content"] == "Exact new completed result"
    assert original == history()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["native", "semantic"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_suffix_replacement_keeps_committed_checkpoint_but_rejects_inflight_work(method, cancel):
    manager, model = await compacted(method)
    previous = copy.deepcopy(manager.summary)
    # Start a new compactable human turn; force this test's next boundary attempt.
    await manager.add_message({"role": "user", "content": "Next human turn."})
    manager.config.update(summarize_trigger=0, native_min_new_tokens=0)
    entered, release = asyncio.Event(), asyncio.Event()
    operation = model.compact_context if method == "native" else model.complete
    result = operation.return_value
    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        return result
    operation.side_effect = delayed
    task = asyncio.create_task(manager._prepare(model, None, []))
    await asyncio.wait_for(entered.wait(), 1)
    replacement = await manager.get_messages()
    replacement[-2]["content"] = "Background task completed during preparation."
    await manager.set_messages(replacement)
    assert manager.summary == previous
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    else:
        release.set()
        with pytest.raises(RuntimeError, match="stale summary"): await task
    assert manager.summary == previous
    assert await manager.get_messages() == replacement
    assert not manager.is_compacting


@pytest.mark.asyncio
@pytest.mark.parametrize("method, guard", [
    ("native", "identity"), ("semantic", "identity"),
    ("native", "required_original"), ("semantic", "required_original"),
    ("native", "native_transport"), ("native", "native_count"),
])
async def test_preserving_prefix_does_not_bypass_request_boundary_guards(method, guard):
    manager, model = await compacted(method)
    replacement = await manager.get_messages()
    replacement[-1]["content"] = "Completed result"
    await manager.set_messages(replacement)
    manager.config["summarize_trigger"] = 100  # Observe invalidation without making another checkpoint.
    retain = []
    if guard == "identity": manager.checkpoint_identity = lambda: {**IDENTITY, "model": "changed-model"}
    if guard == "required_original": retain = [replacement[1]["content"]]
    if guard == "native_transport": model.validate_compacted_context = lambda row: False
    if guard == "native_count": model.request_budget = lambda *args, **kwargs: {}
    await manager._prepare(model, None, retain)
    assert manager.summary is None
    assert await manager.get_messages() == replacement
