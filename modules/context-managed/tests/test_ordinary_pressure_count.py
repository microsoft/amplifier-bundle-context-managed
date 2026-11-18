"""Optional pressure failures cannot admit an uncounted foreground request."""
import asyncio
import copy

import pytest
from amplifier_core import ChatRequest, Message

from .test_ordinary_settled_fitting import Provider, history, manager


class TokenCountError(RuntimeError):
    def __init__(self, category="timeout", retryable=True):
        self.count_failure = {"category": category, "retryable": retryable, "attempts": 3}
        super().__init__("Could not check conversation size.")


TokenCountError.__module__ = "amplifier_module_provider_openai._token_count"


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["timeout", "connection", "rate_limit", "service"])
async def test_optional_pressure_failure_selects_bounded_view_but_counts_exact_final(category):
    rows = history()
    context = await manager(rows)
    provider = Provider()
    counted = []

    async def count(view):
        counted.append(copy.deepcopy(view))
        if len(counted) == 1:
            raise TokenCountError(category)
        # The oversized pressure request must not just be counted a second time.
        assert len(str(view)) < len(str(counted[0]))
        request = ChatRequest(messages=[Message(**row) for row in view])
        return {"dispatch": request, "budget_decision": await provider.request_budget(request)}

    for _ in range(2):
        counted.clear()
        result = await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count)
        await provider.complete(result["final_attempt"]["dispatch"])
    assert len(provider.completed) == 2 and provider.summaries <= 1
    assert await context.get_messages() == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("final", ["timeout", "unmeasured", "estimate"])
async def test_final_count_failure_after_optional_pressure_failure_is_terminal(final):
    context = await manager(history())
    provider = Provider()
    calls = 0
    error = TokenCountError()

    async def count(view):
        nonlocal calls
        calls += 1
        if calls == 1 or final == "timeout":
            raise error
        request = ChatRequest(messages=[Message(**row) for row in view])
        decision = None if final == "unmeasured" else {
            "estimated_input_tokens": 1, "input_limit_tokens": 16000}
        return {"dispatch": request, "budget_decision": decision}

    with pytest.raises(TokenCountError) as caught:
        await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count)
    assert caught.value is error
    assert calls == 2 and not provider.completed and provider.summaries == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["history", "config", "model", "operations", "stop"])
async def test_optional_count_failure_cannot_bypass_contract_or_stop(mutation):
    context = await manager(history())
    provider = Provider()
    calls = 0

    async def count(view):
        nonlocal calls
        calls += 1
        if mutation == "history":
            await context.add_message({"role": "user", "content": "New input"})
        elif mutation == "config":
            context.config["max_tokens"] = 1000
        elif mutation == "model":
            provider.default_model = "different"
        elif mutation == "operations":
            context.active_operations = lambda: [{"job_id": "New unresolved operation"}]
        else:
            context._is_cancelled = lambda: True
        raise TokenCountError()

    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count)
    assert calls == 1 and not provider.completed and provider.summaries == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(), RuntimeError(),
    TokenCountError("authentication", False), TokenCountError("quota", False),
    TokenCountError("invalid_response", False)])
async def test_only_typed_transient_optional_pressure_error_is_recoverable(error):
    context = await manager(history())
    calls = 0

    async def count(view):
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(type(error)):
        await context.get_measured_request_view(provider=Provider(), retain_contents=[], count_view=count)
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["none", "source", "model", "config"])
async def test_known_native_input_refusal_is_not_repeated_or_used_as_final_admission(change):
    from unittest.mock import AsyncMock
    from .test_ordinary_settled_fitting import measured

    context = await manager(history())
    model = Provider()
    model.supports_native_compaction = lambda: True
    model.validate_compacted_context = lambda row: True
    model.compact_context = AsyncMock(side_effect=AssertionError("No paid replay"))
    original = model.request_budget
    native_counts = 0

    async def count(request, **kwargs):
        nonlocal native_counts
        if (request.metadata or {}).get("purpose") == "context-compaction":
            native_counts += 1
            return {"estimated_input_tokens": 20000, "input_limit_tokens": 16000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 20000}}
        return await original(request, **kwargs)

    model.request_budget = count
    first = await measured(context, model)
    await model.complete(first["final_attempt"]["dispatch"])
    # Exercise the exact optional-preflight memo independently of the older
    # summary-fingerprint cooldown (which may also suppress this request).
    context.summary_failure = None
    if change == "source":
        # Settled evidence changes the eligible source, not the latest input.
        context.messages[2]["tool_calls"][0]["arguments"]["instructions"] += " Changed"
        context.revision += 1
    elif change == "model":
        model.default_model = "changed"
    elif change == "config":
        context.config["native_compaction_max_output_tokens"] = 1024
    second = await measured(context, model)
    await model.complete(second["final_attempt"]["dispatch"])
    assert native_counts == (1 if change == "none" else 2)
    assert len(model.completed) == 2
    model.compact_context.assert_not_awaited()