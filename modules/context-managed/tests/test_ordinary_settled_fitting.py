"""Ordinary request fitting sheds evidence, not the task or unresolved actions."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import ContextLengthError

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.interrupted import calls, result_ids


def exchange(identity, size=8000, structured=False):
    call = {"id": identity, "name": "delegate", "arguments": {"instructions": "a" * size}}
    if structured:
        return [
            {"role": "assistant", "content": [{"type": "tool_call", "id": identity,
                "name": "delegate", "input": call["arguments"]}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_call_id": identity,
                "output": {"observations": {"nested": ["e" * size]}}}]},
        ]
    return [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": identity, "content": json.dumps({
            "delegated": {"observations": ["e" * size]}, "status": "completed"})},
    ]


def history(structured=False):
    rows = [{"role": "system", "content": "Required instructions: preserve Stop and never replay."},
            {"role": "user", "content": "Current objective: finish autonomously."}]
    for i in range(12):
        rows.extend(exchange(f"settled-{i}", structured=structured))
        rows.append({"role": "user", "content": "Settled service observation " + "s" * 1000,
            "metadata": {"ephemeral": True, "source": "service"}})
    return rows


class Provider:
    name = "fixture"
    default_model = "bounded"

    def __init__(self):
        self.counted = []
        self.completed = []
        self.summaries = 0

    async def request_budget(self, request, **kwargs):
        self.counted.append(request.model_copy(deep=True))
        size = len(request.model_dump_json().encode())
        return {"estimated_input_tokens": size, "input_limit_tokens": 16000,
            "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": size}}

    async def complete(self, request, **kwargs):
        if (request.metadata or {}).get("purpose") == "context-compaction":
            self.summaries += 1
            raise TimeoutError("Unconfirmed paid operation; never retry automatically")
        assert request == self.counted[-1], "Warning, tools and overlays must be in the counted dispatch"
        assert len(request.model_dump_json().encode()) <= 16000
        self.completed.append(request.model_copy(deep=True))
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Ordinary response completed.")])


async def manager(rows, **config):
    context = BoundaryContextManager({"foreground_continuation": True, "max_tokens": 16000,
        "compaction_notice_enabled": False, **config})
    await context.set_messages(rows)
    return context


async def measured(context, model, **kwargs):
    async def count(view):
        request = ChatRequest(messages=[Message(**row) for row in view])
        return {"dispatch": request, "budget_decision": await model.request_budget(request)}
    return await context.get_measured_request_view(
        provider=model, retain_contents=[], count_view=count, **kwargs)


def assert_protocol(rows):
    declared = set().union(*(calls(row).keys() for row in rows))
    results = set().union(*(result_ids(row) for row in rows))
    assert results <= declared, "Selected results must not lose their calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_many_settled_exchanges_after_only_input_complete_two_ordinary_requests(structured):
    rows = history(structured)
    before = copy.deepcopy(rows)
    context = await manager(rows)
    model = Provider()
    for _ in range(2):
        result = await measured(context, model)
        response = await model.complete(result["final_attempt"]["dispatch"])
        assert response.content[0].text == "Ordinary response completed."
        view = result["base_view"]
        assert result["continuation"] == "incomplete_history"
        assert "Current objective" in str(view) and "Required instructions" in str(view)
        assert len(str(view)) < 16000
        assert_protocol(view)
        warning = [row for row in view
            if (row.get("metadata") or {}).get("source") == "context-managed-continuation"]
        assert len(warning) == 1 and warning[0]["role"] == "developer"
        assert warning[0]["metadata"]["ephemeral"] is True
        assert "not represented or summarized" in warning[0]["content"]
    assert len(model.completed) == 2
    assert model.summaries <= 1
    assert await context.get_messages() == rows == before


@pytest.mark.asyncio
async def test_structured_result_before_fresh_short_input_is_not_carried_as_input():
    rows = [{"role": "user", "content": "Original objective"}]
    rows += exchange("old-structured", size=64000, structured=True)
    rows += [{"role": "user", "content": "Fresh short input"}]
    context = await manager(rows)
    assert not context._human(rows[2])
    assert rows[2] not in context._carry(rows)
    result = await measured(context, Provider())
    assert "Fresh short input" in str(result["base_view"])
    assert "old-structured" not in str(result["base_view"])
    assert await context.get_messages() == rows


@pytest.mark.asyncio
async def test_settled_ephemeral_recovery_reference_can_be_omitted():
    rows = history()
    rows.append({"role": "user", "content": "Recovered result reference " + "r" * 64000,
        "metadata": {"source": "session-health-recovery", "ephemeral": True, "persisted": True}})
    context = await manager(rows)
    result = await measured(context, Provider())
    assert "Recovered result reference" not in str(result["base_view"])
    assert await context.get_messages() == rows


@pytest.mark.asyncio
async def test_recovery_all_user_reference_rows_do_not_pin_archive_or_latest_image():
    rows = [{"role": "system", "content": "Required recovery instructions"},
            {"role": "user", "content": "Original recovery objective"}]
    for i in range(475):
        rows.append({"role": "user", "content": "Reference data " + json.dumps({
            "sourceRole": "tool" if i % 2 else "assistant", "toolResult": "r" * 2200}),
            "metadata": {"amplifier_recovery_reference": True, "ephemeral": True}})
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
        "data": "x" * 1_389_000}}
    rows.append({"role": "user", "content": [
        {"type": "text", "text": "True current user input; recover the report."}, image]})
    context = await manager(rows)
    model = Provider()
    async def count_media(request, **kwargs):
        model.counted.append(request.model_copy(deep=True))
        wire = request.model_dump(mode="json")
        for row in wire["messages"]:
            if isinstance(row["content"], list):
                for block in row["content"]:
                    if block["type"] == "image":
                        assert block["source"]["data"] == image["source"]["data"]
                        block["source"]["data"] = "<counted media: 1000 tokens>"
        size = len(json.dumps(wire).encode())
        return {"estimated_input_tokens": size, "input_limit_tokens": 67904,
            "measurement": {"kind": "provider_count", "source": "fixture.media", "input_tokens": size}}
    model.request_budget = count_media
    result = await measured(context, model)
    view = result["base_view"]
    assert "True current user input" in str(view)
    assert image in view[-1]["content"]
    assert sum(bool((row.get("metadata") or {}).get("amplifier_recovery_reference"))
               for row in view) < 30
    assert await context.get_messages() == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["unanswered", "queued", "structured-queued"])
async def test_unresolved_parallel_sibling_keeps_whole_group_and_fails_if_required_too_large(state):
    rows = history()
    rows += [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "done", "name": "read", "arguments": {"large": "x" * 24000}},
        {"id": "pending", "name": "write", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "done", "content": "Completed sibling"}]
    receipt = json.dumps({"job_id": "job-original", "status": "queued"})
    if state == "queued":
        rows.append({"role": "tool", "tool_call_id": "pending", "content": receipt})
    elif state == "structured-queued":
        rows.append({"role": "user", "content": [
            {"type": "tool_result", "tool_call_id": "pending", "output": receipt}]})
    context = await manager(rows)
    model = Provider()
    with pytest.raises(ContextLengthError):
        await measured(context, model)
    assert not model.completed
    assert await context.get_messages() == rows


@pytest.mark.asyncio
async def test_small_unresolved_group_and_current_input_survive_while_settled_groups_drop():
    rows = history() + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "small-active", "name": "write", "arguments": {"target": "original"}}]}]
    context = await manager(rows)
    context.active_operations = lambda: [{"job_id": "active-1", "status": "pending"}]
    result = await measured(context, Provider())
    assert "small-active" in str(result["base_view"])
    assert "active-1" in str(result["base_view"])
    assert_protocol(result["base_view"])
    assert await context.get_messages() == rows


@pytest.mark.asyncio
async def test_historical_unknown_keeps_no_replay_notice_without_replaying_call():
    rows = history() + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "uncertain-original", "name": "write", "arguments": {"large": "x" * 24000}}]},
        {"role": "user", "content": "New actual input; report status only."}]
    context = await manager(rows)
    result = await measured(context, Provider())
    view = result["base_view"]
    assert "uncertain-original" in str(view)
    assert "Do not repeat" in str(view)
    assert_protocol(view)
    assert await context.get_messages() == rows


@pytest.mark.asyncio
async def test_warning_is_counted_and_cannot_be_added_after_admission():
    context = await manager(history())
    seen = []
    async def count(view):
        seen.append(copy.deepcopy(view))
        warning = any((row.get("metadata") or {}).get("source") == "context-managed-continuation"
                      for row in view)
        size = 16001 if warning or len(str(view)) > 16000 else 100
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]),
            "budget_decision": {"estimated_input_tokens": size, "input_limit_tokens": 16000,
            "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": size}}}
    with pytest.raises(ContextLengthError):
        await context.get_measured_request_view(provider=None, retain_contents=[], count_view=count)
    assert any(any((row.get("metadata") or {}).get("source") == "context-managed-continuation"
                   for row in view) for view in seen)


@pytest.mark.asyncio
async def test_stop_before_preparation_permits_no_count_or_compaction():
    context = await manager(history())
    context._is_cancelled = lambda: True
    model = Provider()
    import asyncio
    with pytest.raises(asyncio.CancelledError):
        await measured(context, model)
    assert not model.counted and not model.completed and not model.summaries


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("blocks", [False, True])
@pytest.mark.parametrize("status", ["queued", "pending", "unknown"])
async def test_old_unsettled_receipt_and_completed_sibling_survive_fresh_input(structured, blocks, status):
    rows = history()
    rows += [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "settled-sibling", "name": "read", "arguments": {}},
        {"id": "unsettled-sibling", "name": "write", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "settled-sibling", "content": "Complete"}]
    body = json.dumps({"status": status, "job_id": "original-job",
        "execution_outcome": "unconfirmed"})
    payload = [{"type": "text", "text": body}] if blocks else body
    rows.append({"role": "user", "content": [{"type": "tool_result",
        "tool_call_id": "unsettled-sibling", "output": payload}]} if structured else
        {"role": "tool", "tool_call_id": "unsettled-sibling", "content": payload})
    rows += [{"role": "user", "content": "Current status input"}]
    context = await manager(rows)
    result = await measured(context, Provider())
    view = result["base_view"]
    assert all(text in str(view) for text in ("settled-sibling", "unsettled-sibling", "original-job"))
    assert_protocol(view)
    assert await context.get_messages() == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["history", "config", "model", "checkpoint", "stop", "operations"])
async def test_contract_change_while_counting_never_returns_stale_dispatch(mutation):
    rows = history()
    context = await manager(rows)
    model = Provider()
    original = model.request_budget
    count_calls = 0
    async def changing_count(request, **kwargs):
        nonlocal count_calls
        count_calls += 1
        if count_calls == 2:
            if mutation == "history":
                await context.add_message({"role": "user", "content": "New input"})
            elif mutation == "config":
                context.config["max_tokens"] = 12345
            elif mutation == "model":
                model.default_model = "changed"
            elif mutation == "checkpoint":
                context.summary = (1, "Concurrent checkpoint")
            elif mutation == "stop":
                context._is_cancelled = lambda: True
            elif mutation == "operations":
                context.active_operations = lambda: [{"job_id": "new-job"}]
        return await original(request, **kwargs)
    model.request_budget = changing_count
    import asyncio
    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        await measured(context, model)
    assert not model.completed


@pytest.mark.asyncio
async def test_dynamic_prompt_materialized_once_across_failed_compaction_and_fitting():
    context = await manager(history())
    prompt = AsyncMock(return_value="Current captured system instructions")
    context.factory = prompt
    result = await measured(context, Provider())
    assert prompt.await_count == 1
    assert "Current captured system instructions" in str(result["base_view"])


@pytest.mark.asyncio
async def test_evidence_failure_is_not_optional_compaction_failure():
    from amplifier_module_context_managed.errors import CompactionError
    context = await manager(history(), durable_checkpoints=True)
    context.preserve_evidence = AsyncMock(side_effect=CompactionError("native_failed", "Storage failed"))
    model = Provider()
    with pytest.raises(RuntimeError, match="evidence preservation"):
        await measured(context, model)
    assert not model.counted and not model.completed and not model.summaries


@pytest.mark.asyncio
async def test_appended_settled_tools_cannot_rebuy_unconfirmed_compaction():
    context = await manager(history(), summary_retry_delay=0)
    model = Provider()
    # Give the utility enough room to actually reach one paid fake operation.
    async def utility_budget(request, **kwargs):
        return {"estimated_input_tokens": 100, "input_limit_tokens": 1_000_000,
            "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 100}}
    utility = Provider()
    utility.request_budget = utility_budget
    context.summary_provider = lambda: utility
    for i in range(2):
        result = await measured(context, model)
        await model.complete(result["final_attempt"]["dispatch"])
        for row in exchange(f"appended-{i}"):
            await context.add_message(row)
    assert len(model.completed) == 2
    assert utility.summaries == 1
    assert context._foreground_unconfirmed


@pytest.mark.asyncio
async def test_required_hook_retain_and_mixed_result_text_groups_not_orphaned():
    rows = history() + exchange("mixed", size=10, structured=True)
    rows[-1]["content"].append({"type": "text", "text": "Actual current correction alongside receipt"})
    rows += [{"role": "user", "content": "Required hook", "metadata": {"source": "hook"}},
             {"role": "assistant", "content": "Explicit retention"}]
    context = await manager(rows)
    model = Provider()
    async def count(view):
        request = ChatRequest(messages=[Message(**row) for row in view])
        return {"dispatch": request, "budget_decision": await model.request_budget(request)}
    result = await context.get_measured_request_view(
        provider=model, retain_contents=["Explicit retention"], count_view=count)
    assert all(text in str(result["base_view"])
        for text in ("Actual current correction", "Required hook", "Explicit retention", "mixed"))
    assert_protocol(result["base_view"])


@pytest.mark.parametrize("id_key", ["tool_use_id", "tool_call_id"])
def test_result_only_classification_supports_both_provider_forms(id_key):
    context = BoundaryContextManager()
    result = {"role": "user", "content": [{"type": "tool_result", id_key: "original", "content": "x" * 64000}]}
    assert not context._human(result)
    assert result_ids(result) == {"original"}
    assert context._carry([result, {"role": "user", "content": "Fresh input"}]) == [
        {"role": "user", "content": "Fresh input"}]


@pytest.mark.asyncio
async def test_hard_fit_without_provider_count_uses_complete_conservative_request():
    rows = history()
    context = await manager(rows)
    view = await context.get_messages_for_request_retaining(retain_contents=[], hard_fit=True)
    assert len(ChatRequest(messages=[Message(**row) for row in view]).model_dump_json()) < 16000
    assert any((row.get("metadata") or {}).get("source") == "context-managed-continuation" for row in view)
    assert await context.get_messages() == rows


@pytest.mark.asyncio
async def test_nonreducing_native_compaction_is_optional_but_never_retried():
    context = await manager(history())
    model = Provider()
    native = {"role": "user", "content": "n" * 200000, "metadata": {
        "source": "context-managed", "ephemeral": True, "persisted": True, "opaque": True}}
    model.supports_native_compaction = lambda: True
    model.validate_compacted_context = lambda row: bool(row.get("metadata", {}).get("opaque"))
    model.compact_context = AsyncMock(return_value={"kind": "native", "message": native})
    # A separate native input limit permits buying the one non-reducing result.
    original = model.request_budget
    async def native_budget(request, **kwargs):
        if (request.metadata or {}).get("purpose") == "context-compaction":
            return {"estimated_input_tokens": 100, "input_limit_tokens": 1_000_000,
                "measurement": {"kind": "provider_count", "source": "fixture.native", "input_tokens": 100}}
        return await original(request, **kwargs)
    model.request_budget = native_budget
    for _ in range(2):
        result = await measured(context, model)
        await model.complete(result["final_attempt"]["dispatch"])
    assert model.compact_context.await_count == 1
    assert model.summaries == 0 and len(model.completed) == 2
    assert context.summary is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["history", "model", "config"])
async def test_native_count_contract_change_prevents_paid_dispatch(mutation):
    context = await manager(history())
    model = Provider()
    model.supports_native_compaction = lambda: True
    model.validate_compacted_context = lambda row: True
    model.compact_context = AsyncMock(return_value={})
    original = model.request_budget
    async def changing_native_budget(request, **kwargs):
        if (request.metadata or {}).get("purpose") == "context-compaction":
            if mutation == "history":
                await context.add_message({"role": "user", "content": "New current input"})
            elif mutation == "model":
                model.default_model = "Changed"
            else:
                context.config["max_tokens"] = 12345
            return {"estimated_input_tokens": 100, "input_limit_tokens": 1_000_000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 100}}
        return await original(request, **kwargs)
    model.request_budget = changing_native_budget
    with pytest.raises(RuntimeError):
        await measured(context, model)
    model.compact_context.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("large", [False, True])
async def test_queued_then_appended_final_result_keeps_declaration_and_parallel_group(large):
    rows = history() + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "async-call", "name": "write", "arguments": {"large": "x" * (24000 if large else 10)}},
        {"id": "async-sibling", "name": "read", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "async-sibling", "content": "Complete sibling"},
        {"role": "tool", "tool_call_id": "async-call", "content": json.dumps({
            "job_id": "job-original", "status": "queued"})},
        {"role": "assistant", "content": "Waiting for original job."},
        {"role": "tool", "tool_call_id": "async-call", "content": json.dumps({
            "job_id": "job-original", "status": "completed", "output": "Final result"})},
        {"role": "user", "content": "Current status only"}]
    context = await manager(rows)
    result = await measured(context, Provider())
    view = result["base_view"]
    assert_protocol(view)
    if "async-call" in str(view):
        assert "async-sibling" in str(view) and "Complete sibling" in str(view)
    assert await context.get_messages() == rows


@pytest.mark.asyncio
async def test_missing_optional_utility_provider_does_not_block_ordinary_response():
    context = await manager(history())
    def missing():
        raise RuntimeError("Host has no utility provider")
    context.summary_provider = missing
    model = Provider()
    result = await measured(context, model)
    await model.complete(result["final_attempt"]["dispatch"])
    assert len(model.completed) == 1 and model.summaries == 0


@pytest.mark.asyncio
async def test_storage_callback_compaction_error_still_blocks_fallback():
    from amplifier_module_context_managed.errors import CompactionError
    context = await manager(history(), durable_checkpoints=True)
    utility = Provider()
    async def budget(request, **kwargs):
        return {"estimated_input_tokens": 100, "input_limit_tokens": 1_000_000,
            "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 100}}
    async def complete(request, **kwargs):
        utility.summaries += 1
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="A returned summary")])
    utility.request_budget, utility.complete = budget, complete
    context.summary_provider = lambda: utility
    context.preserve_evidence = AsyncMock(return_value=[])
    context.persist_checkpoint = AsyncMock(side_effect=CompactionError("native_failed", "Storage failed"))
    model = Provider()
    with pytest.raises(CompactionError, match="checkpoint_persist_failed"):
        await measured(context, model)
    assert utility.summaries == 1 and not model.completed