"""Only a completed ordinary reply can schedule this uncounted checkpoint."""
import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from .test_summary_work_budget import IDENTITY, context, response


async def setup(**config):
    options = {"max_tokens": 20000, "summarize_trigger": .7, "summary_max_source_chars": 12500,
               "archive_recovery": True, "foreground_continuation": True, "compaction_schedule": "after_turn"}
    ctx = await context(**{**options, **config})
    ctx.preserve_evidence = AsyncMock(return_value=[])
    rows = [{"role": "user", "content": "Keep the ORBIT launch fact."},
            {"role": "assistant", "content": "Understood."}]
    for i in range(9):
        rows.extend([{"role": "user", "content": f"Evidence {i}: " + "x" * 5900},
                     {"role": "assistant", "content": f"Read {i}."}])
    await ctx.set_messages(rows)
    model = SimpleNamespace(name=IDENTITY["provider"], default_model=IDENTITY["model"],
                            complete=AsyncMock(return_value=response("ORBIT grounded note")))
    return ctx, model, rows


def offer(ctx, provider, input_tokens=13000, output_tokens=30, prompt_characters=5000):
    return ctx.ordinary_settled_response(provider=provider, retain_contents=[], input_tokens=input_tokens,
                                         output_tokens=output_tokens, prompt_characters=prompt_characters)


@pytest.mark.asyncio
async def test_anticipated_crossing_commits_only_one_fitting_whole_prefix_and_reopens():
    ctx, model, rows = await setup()
    saved = []
    ctx.persist_checkpoint = lambda: saved.append(copy.deepcopy(ctx.export_checkpoint(IDENTITY)))
    assert offer(ctx, model, input_tokens=12000) is None
    settle = offer(ctx, model)
    model.complete.assert_not_called()
    assert ctx._maintenance_task is None and settle() is True and settle() is False
    await ctx._maintenance_task
    model.complete.assert_awaited_once()
    assert len(saved) == 1 and saved[0]["summary"]["text"] == "ORBIT grounded note"
    end = saved[0]["summary"]["throughMessage"]
    assert 2 < end < len(rows) - 4 and end % 2 == 0
    quoted = model.complete.call_args.args[0].messages[1].content
    assert all(f"Evidence {i}:" in quoted for i in range(end // 2 - 1))
    assert f"Evidence {end // 2 - 1}:" not in quoted
    assert await ctx.get_messages() == rows
    reopened, _, _ = await setup()
    assert reopened.restore_checkpoint(saved[0], IDENTITY)["status"] == "restored"
    await reopened.add_message({"role": "user", "content": "Continue from the evidence."})
    view = await reopened.get_messages_for_request(provider=model)
    assert any("ORBIT grounded note" in m["content"] for m in view)
    assert all(row in view for row in rows[end:])
    model.complete.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["input", "stop", "selection", "configuration", "operations", "partial"])
async def test_intervening_change_supersedes_exact_settled_turn(change):
    ctx, model, _ = await setup()
    settle = offer(ctx, model)
    if change == "input":
        await ctx.add_message({"role": "user", "content": "Different work"})
    elif change == "stop":
        ctx._is_cancelled = lambda: True
    elif change == "selection":
        ctx.checkpoint_identity = lambda: {**IDENTITY, "model": "other"}
    elif change == "configuration":
        ctx.config["output_reserve_fraction"] = .8
    elif change == "operations":
        ctx.active_operations = lambda: [{"job_id": "new"}]
    else:
        ctx._restored_progress = {"completed": 1}
    assert settle() is False and ctx._maintenance_task is None
    model.complete.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["other_model", "opaque", "restored_partial", "uncertain", "measured", "native",
                                     "opt_out", "unknown_usage", "zero_usage", "impossible_usage", "no_prompt"])
async def test_ineligible_evidence_starts_no_optional_work(source):
    ctx, model, _ = await setup()
    usage = {}
    if source == "other_model":
        ctx.summary, ctx.summary_identity = (2, "Prior note"), {**IDENTITY, "model": "other"}
    elif source == "opaque":
        ctx.summary, ctx.summary_identity = (2, {"kind": "native"}), IDENTITY
    elif source == "restored_partial":
        ctx._restored_progress = {"completed": 1}
    elif source == "uncertain":
        ctx._foreground_unconfirmed = True
    elif source == "measured":
        model.request_budget = AsyncMock()
    elif source == "native":
        model.supports_native_compaction = lambda: True
    elif source == "opt_out":
        ctx.config["compaction_schedule"] = "foreground"
    elif source == "unknown_usage":
        usage["input_tokens"] = None
    elif source == "impossible_usage":
        usage["input_tokens"] = 4_000_000
    elif source == "zero_usage":
        usage["input_tokens"] = 0
    else:
        usage["prompt_characters"] = 0
    assert offer(ctx, model, **usage) is None
    assert ctx._maintenance_task is None and not ctx._maintenance_requested
    model.complete.assert_not_called()


@pytest.mark.asyncio
async def test_forecast_uses_current_configured_reserve():
    ctx, model, _ = await setup(output_reserve_fraction=.1)
    model.get_info = lambda: SimpleNamespace(defaults={"context_window": 24000, "max_output_tokens": 8000})
    assert offer(ctx, model, input_tokens=9100) is None
    ctx.config["output_reserve_fraction"] = .75
    assert callable(offer(ctx, model, input_tokens=9100))
    model.complete.assert_not_called()


@pytest.mark.asyncio
async def test_oversized_first_exchange_is_never_excerpted_or_promoted():
    ctx, model, rows = await setup(summary_max_source_chars=400)
    await ctx.set_messages(rows[2:])
    settle = offer(ctx, model)
    assert settle()
    await asyncio.gather(ctx._maintenance_task, return_exceptions=True)
    model.complete.assert_not_called()
    assert ctx.summary is None and await ctx.get_messages() == rows[2:]


@pytest.mark.asyncio
@pytest.mark.parametrize("overhead", ["fixed_tools", "fixed_system", "retained_initial"])
async def test_large_constant_envelope_and_small_history_does_not_buy_summaries(overhead):
    ctx, model, _ = await setup(summary_max_source_chars=100000)
    rows = [{"role": role, "content": "Ok"} for _ in range(10) for role in ("user", "assistant")]
    if overhead == "fixed_system":
        rows.insert(0, {"role": "system", "content": "I" * 56000})
    elif overhead == "retained_initial":
        rows[0]["content"] = "I" * 56000
    await ctx.set_messages(rows)
    for _ in range(2):
        settle = offer(ctx, model, input_tokens=15000, prompt_characters=2)
        assert settle()
        await ctx._maintenance_task
        assert ctx.summary is None and await ctx.get_messages() == rows
        rows.extend([{"role": "user", "content": "Ok"}, {"role": "assistant", "content": "Ok"}])
        for row in rows[-2:]:
            await ctx.add_message(row)
    model.complete.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["stop", "new_input"])
async def test_cancelled_after_turn_answer_cannot_publish_a_late_summary(interrupt):
    ctx, model, rows = await setup()
    entered = asyncio.Event()
    never = asyncio.Event()

    async def late_summary(_request):
        entered.set()
        try:
            await never.wait()
        except asyncio.CancelledError:
            return response("Completed after cancellation")

    ctx.persist_checkpoint = AsyncMock()
    model.complete = AsyncMock(side_effect=late_summary)
    assert offer(ctx, model)()
    task = ctx._maintenance_task
    await asyncio.wait_for(entered.wait(), 2)
    if interrupt == "new_input":
        row = {"role": "user", "content": "Continue my work"}
        await ctx.add_message(row)
        rows.append(row)
    else:
        ctx._is_cancelled = lambda: True
        ctx.cancel_settled_compaction()
    await asyncio.gather(task, return_exceptions=True)
    assert ctx.summary is None and await ctx.get_messages() == rows
    ctx.persist_checkpoint.assert_not_called()
    model.complete.assert_awaited_once()
