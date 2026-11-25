"""An oversized native request can only resume on new, fitting evidence."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import ContextLengthError

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.errors import CompactionError
from .test_model_transition import native_message, provider, rows, SOURCE


def decision(tokens):
    return {"estimated_input_tokens": tokens, "input_limit_tokens": 1000,
            "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": tokens}}


@pytest.mark.asyncio
@pytest.mark.parametrize("foreground", [False, True])
@pytest.mark.parametrize("counted", [True, False], ids=["measured", "estimate"])
async def test_appended_inputs_hold_refusal_and_changed_dispatch_can_resume(foreground, counted):
    hooks = SimpleNamespace(emit=AsyncMock())
    context = BoundaryContextManager({"max_tokens": 1000, "summarize_trigger": 0.1,
                                      "foreground_continuation": foreground}, hooks)
    originals = rows()
    await context.set_messages(originals)
    model = provider(native=True, identity=SOURCE)
    model.native_compaction_requires_request_context = True
    model.complete = AsyncMock(return_value="continued")
    model.compact_context = AsyncMock(return_value={"kind": "native", "message": native_message(SOURCE["model"])})
    async def native_budget(request, **options):
        if request.temperature == 0.25:
            return None
        result = decision(500 if request.temperature == 0.5 else 1100)
        if not counted:
            del result["measurement"]
        return result

    model.request_budget = AsyncMock(side_effect=native_budget)
    temperature = None

    async def count(view):
        native_checkpoint = any((row.get("metadata") or {}).get("fixture:opaque") for row in view)
        request = ChatRequest(model=SOURCE["model"], messages=[Message(**row) for row in view],
                              temperature=temperature)
        return {"dispatch": request, "budget_decision": decision(50 if native_checkpoint else 1200)}

    async def prepare():
        return await context.get_measured_request_view(provider=model, retain_contents=[], count_view=count)

    for step in ("initial", "identical retry", "first append", "second append"):
        if "append" in step:
            row = {"role": "user", "content": step}
            originals.append(row)
            if step == "first append":
                await context.add_message(row)
            else:
                await context.set_messages(originals)
        with pytest.raises((CompactionError, ContextLengthError)) as failed:
            await prepare()
        error = failed.value
        if foreground:
            assert error.context_compaction_code == "native_input_oversized"
            assert error.__cause__.code == "native_input_oversized"
        else:
            assert error.code == "native_input_oversized"
        assert context.checkpoint_status["failure"]["code"] == "native_input_oversized"
        assert context.summary is None and not context.is_compacting
        model.compact_context.assert_not_awaited()

    events = [call.args[0] for call in hooks.emit.await_args_list]
    assert events.count("context:compaction_started") == events.count("context:compaction_finished") == 1
    # Only an actual provider count can be reused for an identical dispatch.
    assert model.request_budget.await_count == (3 if counted else 4)
    temperature = 0.25
    with pytest.raises((ContextLengthError, CompactionError)):
        await prepare()
    model.compact_context.assert_not_awaited()
    assert context.checkpoint_status["failure"]["code"] == "native_input_oversized"
    temperature = 0.5  # same canonical source and model, different physical request
    fitted = await prepare()
    await model.complete(fitted["final_attempt"]["dispatch"])
    model.compact_context.assert_awaited_once()
    model.complete.assert_awaited_once()
    assert context.summary[0] > 0 and context.checkpoint_status["status"] == "ready"
    assert await context.get_messages() == originals


@pytest.mark.asyncio
async def test_known_native_refusal_allows_explicit_recovery_and_the_next_foreground_request():
    from .test_native_recovery import (
        IDENTITY, REQUEST_REMINDER, request_recovery_dispatch, request_recovery_fixture,
        request_recovery_state, run_request_recovery,
    )

    state = request_recovery_fixture(prior=False)
    context, model = state.ctx, state.provider
    del context._prepare  # The recovery-only fixture fences ordinary preparation.
    context.summary_failure = None
    context.config["summarize_trigger"] = 0.1
    originals = await context.get_messages()

    async def ordinary_count(view):
        request = request_recovery_dispatch(view)
        return {"dispatch": request, "budget_decision": await model.request_budget(request)}

    async def prepare():
        return await context.get_measured_request_view(
            provider=model, retain_contents=[REQUEST_REMINDER], count_view=ordinary_count)

    for _ in range(2):
        with pytest.raises(CompactionError, match="native_input_oversized"):
            await prepare()
    model.compact_context.assert_not_awaited()
    state.before = request_recovery_state(context)
    recovered = await run_request_recovery(state)
    assert recovered["recovery"]["status"] == "installed"
    assert state.installed[-1] == context.export_checkpoint(IDENTITY)

    fitted = await prepare()
    model.complete = AsyncMock(return_value="continued")
    await model.complete(fitted["final_attempt"]["dispatch"])
    assert await context.get_messages() == originals
    assert fitted["final_attempt"]["dispatch"] == recovered["final_attempt"]["dispatch"]
    assert state.effects == {"write-1": 1}
    context.tool_executor.assert_not_awaited()
