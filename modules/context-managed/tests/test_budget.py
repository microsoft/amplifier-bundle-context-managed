"""
Budget calculation and token estimation tests for ManagedContextManager.

Verifies priority resolution chain for _calculate_budget() and chars/4
token estimation heuristic used throughout the module.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_context_managed import ManagedContextManager


class TestCalculateBudget:
    """Verify budget calculation priority chain: explicit > model_info > provider defaults > fallback."""

    def test_explicit_token_budget_takes_priority(self):
        """Explicit token_budget overrides all provider-based calculations."""
        ctx = ManagedContextManager()
        provider = MagicMock()

        result = ctx._calculate_budget(token_budget=50000, provider=provider)

        assert result == 50000
        # Provider should not be consulted when explicit budget is given
        provider.get_model_info.assert_not_called()

    def test_provider_model_info(self):
        """Provider.get_model_info() drives budget: context_window - int(max_output*0.5) - 4096."""
        ctx = ManagedContextManager()

        model_info = SimpleNamespace(context_window=200000, max_output_tokens=8192)
        provider = MagicMock()
        provider.get_model_info.return_value = model_info

        result = ctx._calculate_budget(token_budget=None, provider=provider)

        # 200000 - int(8192 * 0.5) - 4096 = 200000 - 4096 - 4096 = 191808
        assert result == 191808

    def test_provider_info_defaults_fallback(self):
        """Falls back to provider.get_info().defaults when get_model_info returns None."""
        ctx = ManagedContextManager()

        provider = MagicMock()
        provider.get_model_info.return_value = None
        provider_info = SimpleNamespace(
            defaults={"context_window": 128000, "max_output_tokens": 4096}
        )
        provider.get_info.return_value = provider_info

        result = ctx._calculate_budget(token_budget=None, provider=provider)

        # 128000 - int(4096 * 0.5) - 4096 = 128000 - 2048 - 4096 = 121856
        assert result == 121856

    def test_fallback_to_max_tokens(self):
        """With no provider, falls back to self.max_tokens (default 200000)."""
        ctx = ManagedContextManager()

        result = ctx._calculate_budget(token_budget=None, provider=None)

        assert result == 200000

    def test_custom_max_tokens(self):
        """ManagedContextManager with custom max_tokens uses that as fallback."""
        ctx = ManagedContextManager(max_tokens=100000)

        result = ctx._calculate_budget(token_budget=None, provider=None)

        assert result == 100000

    def test_provider_exception_falls_back(self):
        """Provider raising RuntimeError falls back to self.max_tokens."""
        ctx = ManagedContextManager()

        provider = MagicMock()
        provider.get_model_info.side_effect = RuntimeError("provider unavailable")

        result = ctx._calculate_budget(token_budget=None, provider=provider)

        assert result == 200000


class TestTokenEstimation:
    """Verify chars/4 token estimation heuristic."""

    def test_estimate_tokens_basic(self):
        """Single-message estimation uses chars/4 heuristic."""
        ctx = ManagedContextManager()
        message = {"role": "user", "content": "hello"}

        result = ctx._estimate_tokens([message])

        expected = len(str(message)) // 4
        assert result == expected

    def test_estimate_tokens_single(self):
        """_estimate_tokens_single matches _estimate_tokens for a single-element list."""
        ctx = ManagedContextManager()
        message = {
            "role": "assistant",
            "content": "This is a longer response with more content.",
        }

        list_result = ctx._estimate_tokens([message])
        single_result = ctx._estimate_tokens_single(message)

        assert list_result == single_result

    def test_running_token_estimate_updates(self):
        """Initial running token estimate is 0 on a fresh context."""
        ctx = ManagedContextManager()

        assert ctx._running_token_estimate == 0

    @pytest.mark.asyncio
    async def test_running_estimate_after_messages(self):
        """After adding 2 messages, running estimate grows above 0."""
        ctx = ManagedContextManager()

        assert ctx._running_token_estimate == 0

        await ctx.add_message({"role": "user", "content": "Hello, world!"})
        after_first = ctx._running_token_estimate
        assert after_first > 0

        await ctx.add_message(
            {"role": "assistant", "content": "Hi there, how can I help?"}
        )
        after_second = ctx._running_token_estimate
        assert after_second > after_first


class TestBudgetPressure:
    """Verify budget pressure event emission at the pressure_warning threshold."""

    @pytest.mark.asyncio
    async def test_pressure_event_at_threshold(self):
        """Event emitted with correct data when usage crosses 70% threshold."""
        hooks = MagicMock()
        hooks.emit = AsyncMock()
        ctx = ManagedContextManager(max_tokens=100, pressure_warning=0.70, hooks=hooks)

        # Add 20 messages with padding — easily accumulates enough tokens to
        # cross 70 tokens on a 100-token budget (each message ~25 tokens after
        # timestamp injection).
        for i in range(20):
            await ctx.add_message({"role": "user", "content": f"msg{i}" + "x" * 10})

        await ctx.get_messages_for_request()

        pressure_calls = [
            c
            for c in hooks.emit.call_args_list
            if c.args[0] == "context:budget_pressure"
        ]
        assert len(pressure_calls) == 1
        event_data = pressure_calls[0].args[1]
        assert event_data["usage_fraction"] >= 0.70
        assert "token_count" in event_data
        assert "budget" in event_data

    @pytest.mark.asyncio
    async def test_pressure_event_not_emitted_below_threshold(self):
        """No budget_pressure event when usage is well below the threshold."""
        hooks = MagicMock()
        hooks.emit = AsyncMock()
        ctx = ManagedContextManager(
            max_tokens=200000, pressure_warning=0.70, hooks=hooks
        )

        # One small message — negligible fraction of 200 000-token budget
        await ctx.add_message({"role": "user", "content": "hello"})

        await ctx.get_messages_for_request()

        pressure_calls = [
            c
            for c in hooks.emit.call_args_list
            if c.args[0] == "context:budget_pressure"
        ]
        assert len(pressure_calls) == 0

    @pytest.mark.asyncio
    async def test_pressure_event_emitted_only_once(self):
        """Budget pressure event is emitted exactly once even after multiple requests."""
        hooks = MagicMock()
        hooks.emit = AsyncMock()
        ctx = ManagedContextManager(max_tokens=100, pressure_warning=0.70, hooks=hooks)

        # Cross the threshold
        for i in range(20):
            await ctx.add_message({"role": "user", "content": f"msg{i}" + "x" * 10})

        # Multiple requests while still above threshold
        await ctx.get_messages_for_request()
        await ctx.get_messages_for_request()
        await ctx.get_messages_for_request()

        pressure_calls = [
            c
            for c in hooks.emit.call_args_list
            if c.args[0] == "context:budget_pressure"
        ]
        assert len(pressure_calls) == 1

    @pytest.mark.asyncio
    async def test_pressure_reset_after_clear(self):
        """Pressure flag resets after clear(), allowing a second emission on re-crossing."""
        hooks = MagicMock()
        hooks.emit = AsyncMock()
        ctx = ManagedContextManager(max_tokens=100, pressure_warning=0.70, hooks=hooks)

        # First crossing
        for i in range(20):
            await ctx.add_message({"role": "user", "content": f"msg{i}" + "x" * 10})
        await ctx.get_messages_for_request()

        # clear() archives transcript and resets _pressure_emitted
        await ctx.clear()

        # Second crossing after clear
        for i in range(20):
            await ctx.add_message({"role": "user", "content": f"msg{i}" + "x" * 10})
        await ctx.get_messages_for_request()

        pressure_calls = [
            c
            for c in hooks.emit.call_args_list
            if c.args[0] == "context:budget_pressure"
        ]
        assert len(pressure_calls) == 2
