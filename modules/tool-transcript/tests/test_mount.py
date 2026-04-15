"""
Tests for mount() registration in tool-transcript module.

Verifies that mount() correctly registers the ReadTranscriptTool with the
coordinator, passes config values to the tool, and handles missing config.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from amplifier_module_tool_transcript import mount, ReadTranscriptTool


class TestMount:
    """Tests for mount() function registration behavior."""

    def _make_coordinator(self) -> MagicMock:
        """Create a MagicMock coordinator with AsyncMock mount and get_capability returning None."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.get_capability = MagicMock(return_value=None)
        return coordinator

    @pytest.mark.asyncio
    async def test_mount_calls_coordinator(self):
        """mount() calls coordinator.mount with ('tools', ReadTranscriptTool instance, name='read_transcript')."""
        coordinator = self._make_coordinator()

        await mount(coordinator)

        coordinator.mount.assert_called_once()
        call_args = coordinator.mount.call_args
        # First positional arg is "tools"
        assert call_args[0][0] == "tools"
        # Second positional arg is a ReadTranscriptTool instance
        assert isinstance(call_args[0][1], ReadTranscriptTool)
        # Keyword arg name is "read_transcript"
        assert call_args[1]["name"] == "read_transcript"

    @pytest.mark.asyncio
    async def test_mount_passes_rate_limit_config(self):
        """config={'rate_limit_per_turn': 5} results in tool._rate_limit_per_turn == 5."""
        coordinator = self._make_coordinator()

        await mount(coordinator, config={"rate_limit_per_turn": 5})

        # Retrieve the tool instance from the call
        call_args = coordinator.mount.call_args
        tool = call_args[0][1]
        assert tool._rate_limit_per_turn == 5

    @pytest.mark.asyncio
    async def test_mount_default_rate_limit(self):
        """No config results in tool._rate_limit_per_turn == 3 (the default)."""
        coordinator = self._make_coordinator()

        await mount(coordinator)

        # Retrieve the tool instance from the call
        call_args = coordinator.mount.call_args
        tool = call_args[0][1]
        assert tool._rate_limit_per_turn == 3

    @pytest.mark.asyncio
    async def test_mount_no_config(self):
        """config=None succeeds and coordinator.mount is called exactly once."""
        coordinator = self._make_coordinator()

        await mount(coordinator, config=None)

        coordinator.mount.assert_called_once()
