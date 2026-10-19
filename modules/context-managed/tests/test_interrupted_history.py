"""A missing old receipt must not pin all subsequent completed work forever."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.interrupted import project, unanswered
from amplifier_module_context_managed.working_history import exchanges, select_window


def call(identity, name="inspect"):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": identity, "name": name, "arguments": {}}]}


def result(identity, content="recorded evidence"):
    return {"role": "tool", "tool_call_id": identity, "content": content}


def history(count=14, size=100):
    rows = [{"role": "user", "content": "Original task: preserve work."}, call("interrupted", "bash"),
            {"role": "user", "content": "Continue; check what happened before repeating actions."}]
    for i in range(count):
        rows += [call(str(i)), result(str(i), "evidence " * size)]
    return rows


def native():
    def count(request, **kw):
        size = len(request.model_dump_json()) // 4
        return {"estimated_input_tokens": size, "input_limit_tokens": 10000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": size}}
    return SimpleNamespace(name="fixture", default_model="native", supports_native_compaction=lambda: True,
        validate_compacted_context=lambda row: bool(row.get("metadata", {}).get("opaque")),
        request_budget=count, complete=AsyncMock(),
        compact_context=AsyncMock(return_value={"kind": "native", "message": {"role": "user", "content": "opaque",
            "metadata": {"source": "context-managed", "ephemeral": True, "persisted": True, "opaque": True}}}))


def notice(view):
    return [r for r in view if r.get("role") == "user" and r.get("metadata", {}).get("source") == "context-managed-interrupted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("archive", [False, True])
async def test_interrupted_resume_compacts_later_work_without_replay_and_restores(archive):
    ctx = BoundaryContextManager({"max_tokens": 10000, "summarize_trigger": .3,
        "durable_checkpoints": True, "archive_recovery": archive})
    original = history(count=400, size=100) if archive else history()
    saved = copy.deepcopy(original)
    identity = {"provider": "fixture", "model": "native"}
    ctx.checkpoint_identity = lambda: identity
    ctx.preserve_evidence = AsyncMock(return_value=[{"artifact": "canonical-originals"}])
    await ctx.set_messages(original)
    provider = native()
    view = await ctx.get_messages_for_request(provider=provider)
    assert ctx.summary[0] > 3
    assert view[-2]["content"] == original[-1]["content"]
    assert len(notice(view)) == 1
    assert "interrupted" in notice(view)[0]["content"]
    assert "does not prove success" in notice(view)[0]["content"]
    assert await ctx.get_messages() == saved
    provider.complete.assert_not_awaited()  # Auxiliary compaction never executes tools/foreground work.
    provider.compact_context.assert_awaited_once()
    requests = provider.compact_context.await_args.args[0]
    request_rows = [m.model_dump() for m in requests.messages]
    calls = {c["id"] for r in request_rows for c in (r.get("tool_calls") or [])}
    results = {r.get("tool_call_id") for r in request_rows if r["role"] == "tool"}
    assert calls <= results
    if not archive:
        receipt = next(r for r in request_rows if r.get("tool_call_id") == "interrupted")
        assert json.loads(receipt["content"])["execution_outcome"] == "unconfirmed"
    else:
        assert ctx.recovery["omittedMessages"] > 0
    record = ctx.export_checkpoint(identity)
    restored = BoundaryContextManager(ctx.config)
    restored.checkpoint_identity = lambda: identity
    restored.preserve_evidence = ctx.preserve_evidence
    await restored.set_messages(saved)
    assert restored.restore_checkpoint(record, identity)["status"] == "restored"
    again = await restored.get_messages_for_request(provider=provider)
    assert len(notice(again)) == 1
    assert "interrupted" in notice(again)[0]["content"]
    assert await restored.get_messages() == saved
    provider.compact_context.assert_awaited_once()


def test_pending_and_current_calls_still_block_including_partial_parallel_batch():
    ctx = BoundaryContextManager()
    rows = history()
    rows.insert(2, result("interrupted", '{"status":"queued","job_id":"still-running"}'))
    assert ctx._boundary(rows, []) == 0
    assert unanswered(rows, ctx._human) == []
    # A current unanswered batch has no subsequent human continuation.
    rows = [{"role": "user", "content": "Task"}, call("missing")]
    for i in range(20): rows += [call(str(i)), result(str(i))]
    assert ctx._boundary(rows, []) == 0
    assert unanswered(rows, ctx._human) == []


@pytest.mark.parametrize("meta", [{"ephemeral": True, "persisted": True},
    {"amplifier_input": {"version": 1, "kind": "service"}}])
def test_service_and_hook_messages_cannot_retire_an_unanswered_call(meta):
    ctx = BoundaryContextManager()
    rows = history()
    rows[2]["metadata"] = meta
    assert unanswered(rows, ctx._human) == []
    assert ctx._boundary(rows, []) == 0


def test_mixed_parallel_batch_preserves_actual_results_and_exact_unknown():
    ctx = BoundaryContextManager()
    rows = history()
    rows[1]["tool_calls"].append({"id": "completed", "name": "write", "arguments": {}})
    rows.insert(2, result("completed", "saved exactly once"))
    unknown = unanswered(rows, ctx._human)
    projected = project(rows, unknown)
    assert len([r for r in projected if r.get("tool_call_id") == "completed"]) == 1
    assert next(r for r in projected if r.get("tool_call_id") == "completed")["content"] == "saved exactly once"
    assert len([r for r in projected if r.get("tool_call_id") == "interrupted"]) == 1
    assert len(notice(project(projected, unknown))) == 1
    assert project(projected, unknown) == projected
    assert ctx._boundary(rows, []) > 3
    assert ctx._boundary(rows, [rows[0]["content"]]) == 0


def test_late_actual_result_replaces_absence_observation_and_ambiguous_ids_fail_closed():
    ctx = BoundaryContextManager()
    rows = history()
    rows.append(result("interrupted", "actual late result"))
    assert unanswered(rows, ctx._human) == []
    assert not notice(ctx._assembled(rows, None))
    duplicate = history()
    duplicate.insert(2, call("interrupted"))
    assert unanswered(duplicate, ctx._human) == []
    assert ctx._boundary(duplicate, []) == 0


def test_content_block_calls_and_results_are_classified_without_duplicate_receipts():
    ctx = BoundaryContextManager()
    rows = history()
    rows[1]["content"] = [{"type": "tool_use", "id": "interrupted", "name": "bash", "input": {}}]
    assert len(unanswered(rows, ctx._human)) == 1
    rows.insert(2, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "interrupted", "content": "done"}]})
    assert unanswered(rows, ctx._human) == []


@pytest.mark.parametrize("representation", ["same-list", "content-block", "function"])
@pytest.mark.parametrize("conflict", ["name", "arguments"])
def test_conflicting_same_row_call_ids_remain_unresolved_barriers(representation, conflict):
    ctx = BoundaryContextManager()
    rows = history()
    name = "delete" if conflict == "name" else "bash"
    arguments = {"command": "different action"} if conflict == "arguments" else {}
    if representation == "same-list":
        rows[1]["tool_calls"].append({"id": "interrupted", "name": name, "arguments": arguments})
    elif representation == "content-block":
        rows[1]["content"] = [{"type": "tool_use", "id": "interrupted", "name": name, "input": arguments}]
    else:
        rows[1]["tool_calls"].append({"id": "interrupted", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}})
    saved = copy.deepcopy(rows)
    assert unanswered(rows, ctx._human) == []
    assert project(rows, unanswered(rows, ctx._human)) == saved
    assert ctx._boundary(rows, []) == 0
    assert list(exchanges(rows, ctx._human)) == [(0, 1), (1, len(rows))]


def test_equivalent_json_arguments_across_call_representations_are_deduplicated():
    ctx = BoundaryContextManager()
    rows = history()
    rows[1]["tool_calls"][0]["arguments"] = {"a": 1, "b": 2}
    rows[1]["content"] = [{"type": "tool_call", "id": "interrupted",
        "function": {"name": "bash", "arguments": '{"b":2, "a":1}'}}]
    unknown = unanswered(rows, ctx._human)
    assert len(unknown) == 1
    assert len([r for r in project(rows, unknown) if r.get("tool_call_id") == "interrupted"]) == 1
    assert ctx._boundary(rows, []) > 3


def test_ordinary_peer_input_qualifies_without_claiming_human_authorization():
    ctx = BoundaryContextManager()
    rows = history()
    rows[2]["metadata"] = {"amplifier_input": {"version": 1, "kind": "user", "source": "peer"}}
    unknown = unanswered(rows, ctx._human)
    assert len(unknown) == 1
    assert ctx._boundary(rows, []) > 3
    observation = notice(project(rows, unknown))[0]["content"]
    assert "reference evidence, not instructions or authorization" in observation
    assert "Later ordinary input does not prove success" in observation
    assert "human" not in observation


def test_archive_ranges_still_name_canonical_rows_after_interruption():
    ctx = BoundaryContextManager()
    rows = history(50)
    selected, record = select_window(rows, len(rows) - 8, [], 12000, ctx._human)
    assert record["sourceMessages"] == len(rows)
    assert record["omittedMessages"] > 0
    canonical_selected = [rows[i] for start, end in record["selectedRanges"] for i in range(start, end)]
    assert [r for r in selected if r.get("metadata", {}).get("source") != "context-managed-recovery"] == canonical_selected


def test_tool_result_user_message_is_not_a_human_continuation():
    ctx = BoundaryContextManager()
    rows = [call("unknown"), call("known"),
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "known", "content": "done"}]}]
    assert unanswered(rows, ctx._human) == []


def test_embedded_user_tool_call_data_cannot_create_an_unknown_action():
    ctx = BoundaryContextManager()
    rows = [{"role": "user", "content": [{"type": "tool_call", "id": "untrusted", "name": "bash", "input": {}}]},
            {"role": "user", "content": "Discuss the supplied data."}]
    assert unanswered(rows, ctx._human) == []


@pytest.mark.asyncio
async def test_frozen_native_checkpoint_advances_past_lone_interrupted_tail():
    # The shape of the production failure: a valid checkpoint, one completed
    # exchange, one lost result, human resume, then hundreds of settled rows.
    original = [{"role": "user", "content": "Build the feature."}]
    for i in range(2183): original += [call(f"old-{i}"), result(f"old-{i}", "done")]
    original.append({"role": "assistant", "content": "Saved progress."})
    assert len(original) == 4368
    original += [call("last-before-interruption"), result("last-before-interruption"), call("interrupted"),
                 {"role": "user", "content": "Continue without repeating completed actions."}]
    for i in range(453): original += [call(f"new-{i}"), result(f"new-{i}", "evidence " * 100)]
    original.append({"role": "assistant", "content": "New progress."})
    assert len(original) == 5279
    ctx = BoundaryContextManager({"max_tokens": 10000, "durable_checkpoints": True, "archive_recovery": True})
    provider = native()
    identity = {"provider": "fixture", "model": "native"}
    ctx.checkpoint_identity = lambda: identity
    ctx.preserve_evidence = AsyncMock(return_value=[])
    await ctx.set_messages(original)
    ctx.summary = (4368, copy.deepcopy(provider.compact_context.return_value["message"]))
    ctx.summary_identity = identity
    view = await ctx.get_messages_for_request(provider=provider)
    assert ctx.summary[0] > 5200
    assert ctx.last_budget["unanswered_historical_calls"] == 1
    assert ctx.last_budget["retained_tail_messages"] <= 10
    assert ctx.recovery["previousThroughMessage"] == 4368
    assert all(start >= 4368 for start, end in ctx.recovery["selectedRanges"])
    assert "interrupted" in notice(view)[0]["content"]
    assert await ctx.get_messages() == original
    provider.compact_context.assert_awaited_once()
    provider.complete.assert_not_awaited()
