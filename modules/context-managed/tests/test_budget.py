"""
Budget calculation and token estimation tests for ManagedContextManager.

Verifies priority resolution chain for _calculate_budget() and chars/4
token estimation heuristic used throughout the module.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

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
