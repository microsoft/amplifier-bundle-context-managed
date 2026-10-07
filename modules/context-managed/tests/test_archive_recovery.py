import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.errors import CompactionError
from amplifier_module_context_managed.working_history import select_window


def history():
    rows = [{"role": "system", "content": "Required instructions"},
            {"role": "user", "content": "Original objective"}]
    for i in range(100):
        rows.extend([
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "name": "read", "arguments": {}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "archived evidence " * 500},
            {"role": "assistant", "content": f"Completed step {i}"}])
    rows.extend([{"role": "user", "content": "Latest correction"},
                 {"role": "assistant", "content": "Saved; waiting"},
                 {"role": "user", "content": "Resume"}])
    return rows


def test_selection_keeps_required_and_latest_whole_exchanges_and_truthful_notice():
    original = history()
    original.insert(40, {"role": "user", "content": "Hook requirement", "metadata": {"source": "hook"}})
    before = copy.deepcopy(original)
    rows, record = select_window(original, len(original)-3, [], 60000, BoundaryContextManager._human)
    assert original == before
    text = str(rows)
    assert all(s in text for s in ["Original objective", "Required instructions", "Hook requirement", "Resume", "Latest correction", "NOT represented or summarized"])
    calls = {c["id"] for row in rows for c in row.get("tool_calls", [])}
    receipts = {row["tool_call_id"] for row in rows if row.get("role") == "tool"}
    assert calls == receipts
    assert record["omittedMessages"] > 200
    assert record["selectedRanges"][-1][1] == len(original)


def test_required_context_is_not_silently_clipped():
    with pytest.raises(CompactionError, match="required_context_oversized"):
        select_window([{"role": "system", "content": "x" * 50000}], 1, [], 10000, BoundaryContextManager._human)


def model():
    compacted = {"role": "user", "content": "Native state with archive notice", "metadata": {
        "source": "context-managed", "ephemeral": True, "persisted": True, "opaque": True}}
    requests = []
    def count(request, **kwargs):
        requests.append(request)
        size = len(request.model_dump_json()) // 4
        return {"estimated_input_tokens": size, "input_limit_tokens": 60000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": size}}
    return SimpleNamespace(name="native", default_model="fixture", supports_native_compaction=lambda: True,
        validate_compacted_context=lambda row: bool(row.get("metadata", {}).get("opaque")),
        request_budget=count, requests=requests, compact_context=AsyncMock(return_value={"kind": "native", "message": compacted}),
        complete=AsyncMock())


async def manager(original):
    context = BoundaryContextManager({"durable_checkpoints": True, "archive_recovery": True, "max_tokens": 60000})
    context.checkpoint_identity = lambda: {"provider": "native", "model": "fixture"}
    context.preserve_evidence = AsyncMock(return_value=[])
    await context.set_messages(original)
    return context


@pytest.mark.asyncio
async def test_oversized_resume_counts_only_bounded_input_and_reuses_native_after_restart():
    original = history()
    context = await manager(original)
    provider = model()
    view = await context.get_messages_for_request(provider=provider)
    assert "Resume" in str(view)
    assert await context.get_messages() == original
    assert max(len(r.model_dump_json()) for r in provider.requests) < 120000
    provider.compact_context.assert_awaited_once()
    provider.complete.assert_not_awaited()
    saved = context.export_checkpoint(context.checkpoint_identity())
    assert saved["recovery"]["omittedMessages"] > 200
    restored = await manager(original + [{"role": "user", "content": "Status?"}])
    assert restored.restore_checkpoint(saved, context.checkpoint_identity())["status"] == "restored"
    await restored.get_messages_for_request(provider=provider)
    provider.compact_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_recovery_error_stops_without_summary_or_foreground():
    context = await manager(history())
    provider = model()
    provider.compact_context.side_effect = RuntimeError("provider failure")
    with pytest.raises(CompactionError, match="native_failed"):
        await context.get_messages_for_request(provider=provider)
    provider.complete.assert_not_awaited()
    assert context.summary is None


@pytest.mark.asyncio
async def test_compatible_native_prefix_is_kept_when_uncompacted_tail_is_oversized():
    original = history()
    context = await manager(original)
    provider = model()
    native = provider.compact_context.return_value['message']
    context.summary = (2, copy.deepcopy(native))
    context.summary_identity = context.checkpoint_identity()
    await context.get_messages_for_request(provider=provider)
    request = provider.compact_context.call_args.args[0]
    assert request.messages[1 if request.messages[0].role == 'system' else 0].metadata['opaque']
    assert context.recovery['previousThroughMessage'] == 2
    assert all(a >= 2 for a,b in context.recovery['selectedRanges'])
    assert await context.get_messages() == original
    assert max(len(r.model_dump_json()) for r in provider.requests) < 120000
