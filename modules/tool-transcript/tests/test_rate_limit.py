"""
Tests for rate limiting in ReadTranscriptTool.execute().

Verifies that the per-turn call counter correctly enforces limits, resets
on demand, applies custom limits, and tracks independently per instance.
"""

import pytest

from amplifier_module_tool_transcript import ReadTranscriptTool


class TestRateLimiting:
    """Tests for rate limiting behavior in ReadTranscriptTool.execute()."""

    @pytest.mark.asyncio
    async def test_first_three_calls_succeed(self, mock_coordinator):
        """The first 3 calls with default rate_limit_per_turn=3 all return success=True."""
        tool = ReadTranscriptTool(mock_coordinator)

        result1 = await tool.execute({})
        result2 = await tool.execute({})
        result3 = await tool.execute({})

        assert result1.success is True
        assert result2.success is True
        assert result3.success is True

    @pytest.mark.asyncio
    async def test_fourth_call_rate_limited(self, mock_coordinator):
        """The 4th call exceeds the default limit (3) and returns success=False with 'rate limit' in output."""
        tool = ReadTranscriptTool(mock_coordinator)

        # Exhaust the 3-call limit
        await tool.execute({})
        await tool.execute({})
        await tool.execute({})

        # 4th call should be rate limited
        result = await tool.execute({})

        assert result.success is False
        assert isinstance(result.output, str)
        assert "rate limit" in result.output.lower()

    @pytest.mark.asyncio
    async def test_reset_allows_more_calls(self, mock_coordinator):
        """After exhausting the limit, reset_rate_limit() allows the next call to succeed."""
        tool = ReadTranscriptTool(mock_coordinator)

        # Exhaust the 3-call limit
        await tool.execute({})
        await tool.execute({})
        await tool.execute({})

        # Reset the per-turn counter
        tool.reset_rate_limit()

        # Next call should succeed again
        result = await tool.execute({})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_custom_rate_limit(self, mock_coordinator):
        """With rate_limit_per_turn=1, the 1st call succeeds and the 2nd call fails."""
        tool = ReadTranscriptTool(mock_coordinator, rate_limit_per_turn=1)

        result1 = await tool.execute({})
        result2 = await tool.execute({})

        assert result1.success is True
        assert result2.success is False
        assert isinstance(result2.output, str)
        assert "rate limit" in result2.output.lower()

    @pytest.mark.asyncio
    async def test_counter_tracks_per_instance(self, mock_coordinator):
        """Two separate instances each maintain their own independent rate limit counter."""
        tool1 = ReadTranscriptTool(mock_coordinator, rate_limit_per_turn=1)
        tool2 = ReadTranscriptTool(mock_coordinator, rate_limit_per_turn=1)

        # Exhaust tool1's limit
        await tool1.execute({})

        # tool2 has its own fresh counter — its first call should still succeed
        result = await tool2.execute({})
        assert result.success is True
