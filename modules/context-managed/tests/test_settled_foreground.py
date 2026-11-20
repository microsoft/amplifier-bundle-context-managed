"""Foreground selection and independent checkpoint ownership across a tool boundary."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import ContextLengthError

from .test_autonomous_compaction import provider as native_provider
from .test_ordinary_pressure_count import TokenCountError
from .test_ordinary_settled_fitting import exchange, manager, measured, Provider


def long_history():
    rows = [{"role": "system", "content": "Required instructions"},
            {"role": "user", "content": "Original goal"}]
    for index in range(120):
        rows.extend(exchange(str(index), size=800))
    rows.append({"role": "user", "content": "Give me a short readiness report."})
    return rows


def attempt(rows, size, limit=16000):
    return {"dispatch": ChatRequest(messages=[Message(**row) for row in rows]),
            "budget_decision": {"estimated_input_tokens": size, "input_limit_tokens": limit,
                "input_limit_independent_of_output": True, "measurement": {
                    "kind": "provider_count", "source": "fixture", "input_tokens": size}}}


@pytest.mark.asyncio
@pytest.mark.parametrize("after_tool", [False, True])
async def test_bounded_foreground_measured_before_and_after_completed_tool(after_tool):
    history = long_history()
    if after_tool:
        history.extend(exchange("completed-bash", size=32))
    ctx = await manager(history, compaction_schedule="after_turn")
    provider = Provider()
    sizes = []
    async def count(rows):
        sizes.append(len(rows))
        assert len(rows) < len(history)
        if len(sizes) == 1:
            return attempt(rows, 17000)
        if len(sizes) == 2:
            error = ContextLengthError("Counter timed out")
            error.context_count_unavailable = True
            raise error
        return attempt(rows, 4000)
    fit_output = AsyncMock(side_effect=AssertionError("Independent input limit"))
    result = await ctx.get_measured_request_view(provider=provider, count_view=count,
        retain_contents=[], fit_output=fit_output)
    assert len(sizes) == 3 and sizes[0] > sizes[1] > sizes[2]
    assert result["continuation"] == "incomplete_history"
    assert result["final_attempt"]["budget_decision"]["measurement"]["input_tokens"] == 4000
    if after_tool:
        assert result["base_view"][-2:] == history[-2:]
    assert await ctx.get_messages() == history
    assert provider.summaries == 0
    fit_output.assert_not_awaited()


@pytest.mark.asyncio
async def test_counter_outage_and_required_overflow_are_distinct():
    history = long_history()
    ctx = await manager(history, compaction_schedule="after_turn")
    failure = ContextLengthError("Counter unavailable")
    failure.context_count_unavailable = True
    calls = 0
    async def count(rows):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return {"dispatch": attempt(rows, 1)["dispatch"], "budget_decision": None}
    with pytest.raises(ContextLengthError) as raised:
        await ctx.get_measured_request_view(provider=Provider(), retain_contents=[], count_view=count)
    assert raised.value is failure
    assert calls > 1
    async def too_large(rows):
        return attempt(rows, 20000)
    with pytest.raises(ContextLengthError) as raised:
        await ctx.get_measured_request_view(provider=Provider(), retain_contents=[], count_view=too_large)
    assert raised.value.context_input_limit == 16000
    assert raised.value.context_input_tokens == 20000


@pytest.mark.asyncio
@pytest.mark.parametrize("supersede", [None, "foreground", "stop"])
async def test_settled_compaction_never_blocks_foreground_or_commits_superseded_work(supersede):
    rows = long_history()[:18] + [{"role": "user", "content": "What changed?"},
                                 {"role": "assistant", "content": "Collected results."},
                                 {"role": "user", "content": "Answer now."}]
    ctx = await manager(rows, compaction_schedule="after_turn", summarize_trigger=.01,
                        native_min_new_tokens=0)
    provider = native_provider()
    provider.request_budget = lambda request, **kwargs: attempt([], 1000)["budget_decision"]
    response = provider.compact_context.return_value
    started, finish, cancelled, release = (asyncio.Event() for _ in range(4))
    async def generation(*args, **kwargs):
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()  # A remote completion can outlive local Stop.
        return response
    provider.compact_context = AsyncMock(side_effect=generation)
    async def count(rows):
        return attempt(rows, len(str(rows)) // 4, 16000)
    async def foreground():
        return await ctx.get_measured_request_view(provider=provider, count_view=count, retain_contents=[])
    await foreground()
    provider.compact_context.assert_not_called()
    ctx.hooks = SimpleNamespace(emit=AsyncMock())
    assert ctx.schedule_settled_compaction(provider=provider, count_view=count, retain_contents=[])
    assert not ctx.schedule_settled_compaction(provider=provider, count_view=count, retain_contents=[])
    task = ctx._maintenance_task
    await asyncio.wait_for(started.wait(), 2)
    if supersede:
        if supersede == "foreground":
            assert (await asyncio.wait_for(foreground(), 2))["final_attempt"]
        else:
            ctx.cancel_settled_compaction()
        await asyncio.wait_for(cancelled.wait(), 2)
        assert ctx.summary is None
        release.set()
    else:
        finish.set()
    await asyncio.gather(task, return_exceptions=True)
    assert bool(ctx.summary) == (supersede is None)
    assert provider.compact_context.await_count == 1
    assert not ctx.schedule_settled_compaction(provider=provider, count_view=count, retain_contents=[])
    assert ctx.hooks.emit.await_count
    assert all(event.args[1]["context_work"] == "settled_maintenance"
               for event in ctx.hooks.emit.await_args_list)


@pytest.mark.asyncio
async def test_oversized_completed_result_is_addressable_with_its_real_call_id():
    rows = long_history() + exchange("once", size=12)
    rows[-1]["content"] = "Complete fixture output. " * 10000
    ctx = await manager(rows, compaction_schedule="after_turn")
    result = await measured(ctx, Provider())
    visible = result["base_view"][-1]
    assert visible["role"] == "tool" and visible["tool_call_id"] == "once"
    assert "SHA-256" in visible["content"] and "Do not execute" in visible["content"]
    assert (await ctx.get_messages())[-1] == rows[-1]


@pytest.mark.asyncio
async def test_provider_overflow_can_reference_the_only_completed_exchange():
    rows = [{"role": "user", "content": "What did the tool find?"}] + exchange("once", size=12)
    rows[-1]["content"] = "Full saved tool output. " * 10000
    ctx = await manager(rows, compaction_schedule="after_turn")
    model = Provider()
    result = await measured(ctx, model, reason="provider_overflow")
    visible = result["base_view"]
    assert rows[0] in visible and visible[-2] == rows[-2]
    assert visible[-1]["tool_call_id"] == "once"
    assert "SHA-256" in visible[-1]["content"] and "Do not execute" in visible[-1]["content"]
    assert len(model.counted) == 1
    assert model.counted[0] == result["final_attempt"]["dispatch"]
    assert (await ctx.get_messages()) == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("native_decision", ["oversized", "unavailable"])
async def test_mac_native_oversized_preflight_is_independent_of_ordinary_dispatch(native_decision):
    # Reproduce the observed 163-row shape and one previously executed tool.
    # Values, tool data and the measured allowances here are synthetic.
    rows = [{"role": "user", "content": "Inspect this project."}]
    for index in range(79):
        rows.extend([{"role": "user", "content": f"Question {index}"},
                     {"role": "assistant", "content": "Saved evidence. " * 240}])
    rows.append({"role": "user", "content": "Short current question."})
    rows.extend(exchange("already-completed", size=15))
    rows.insert(0, {"role": "system", "content": "Inspect recorded results."})
    assert len(rows) == 163
    ctx = await manager(rows, compaction_schedule="after_turn", summarize_trigger=.01)
    model = native_provider()
    native_counts = []
    async def native_budget(request, **kwargs):
        native_counts.append(request)
        return attempt([], 30000)["budget_decision"] if native_decision == "oversized" else None
    model.release_media_request = Mock()
    model.request_budget = native_budget
    async def count(view):
        return attempt(view, len(str(view)) // 4)
    ordinary = await ctx.get_measured_request_view(provider=model, retain_contents=[], count_view=count)
    assert ordinary["final_attempt"]["budget_decision"]["estimated_input_tokens"] <= 16000
    assert ordinary["base_view"][-2:] == rows[-2:]
    assert not native_counts
    assert ctx.schedule_settled_compaction(provider=model, count_view=count, retain_contents=[])
    await asyncio.gather(ctx._maintenance_task, return_exceptions=True)
    assert len(native_counts) == 1
    model.compact_context.assert_not_awaited()
    assert (ctx._foreground_native_refusal is not None) == (native_decision == "oversized")
    if native_decision == "oversized":
        assert ctx.checkpoint_status["lastNativeFailure"]["code"] == "native_input_oversized"
        await ctx.add_message({"role": "user", "content": "Fresh input."})
        await ctx.set_messages([*await ctx.get_messages(), {"role": "user", "content": "This input cannot fit."}])

        async def too_large(view):
            return attempt(view, 20000)

        with pytest.raises(ContextLengthError):
            await ctx.get_measured_request_view(provider=model, retain_contents=[], count_view=too_large)
        assert ctx.checkpoint_status["lastNativeFailure"]["code"] == "native_input_oversized"
        assert (await ctx.get_messages())[:-2] == rows
    model.release_media_request.assert_called_once_with(native_counts[0], request_options=None)
    assert (await ctx.get_messages())[:len(rows)] == rows
    if native_decision == "oversized":
        await ctx.set_messages([{"role": "user", "content": "Different history."}])
        assert "lastNativeFailure" not in ctx.checkpoint_status


@pytest.mark.asyncio
@pytest.mark.parametrize("category,should_continue", [("timeout", True), ("authentication", False)])
async def test_installed_openai_count_contract_remains_usable_after_turn(category, should_continue):
    ctx = await manager(long_history(), compaction_schedule="after_turn")
    failure = TokenCountError(category=category, retryable=should_continue)
    counts = []

    async def count(rows):
        counts.append(len(rows))
        if len(counts) == 1:
            raise failure
        return attempt(rows, 200)

    request = ctx.get_measured_request_view(provider=Provider(), retain_contents=[], count_view=count)
    if should_continue:
        assert (await request)["final_attempt"]["budget_decision"]["estimated_input_tokens"] == 200
        assert counts[0] > counts[1]
    else:
        with pytest.raises(TokenCountError) as raised:
            await request
        assert raised.value is failure
        assert len(counts) == 1
