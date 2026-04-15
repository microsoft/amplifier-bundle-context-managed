"""
Integration tests for the full context module -> transcript tool cycle.

Tests the complete cycle: ManagedContextManager writes transcript entries,
ReadTranscriptTool reads them back via the coordinator's capability registry.

Uses real ManagedContextManager from amplifier_module_context_managed to verify
end-to-end behaviour without mocking the file I/O layer.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_context_managed import ManagedContextManager
from amplifier_module_tool_transcript import ReadTranscriptTool


def _make_coordinator(session_dir):
    """Create a MagicMock coordinator with a real capabilities registry.

    register_capability stores values, get_capability retrieves them.
    mount is an AsyncMock. session is a SimpleNamespace with session_dir.
    """
    capabilities: dict = {}

    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.register_capability = MagicMock(
        side_effect=lambda name, value: capabilities.__setitem__(name, value)
    )
    coordinator.get_capability = MagicMock(
        side_effect=lambda name: capabilities.get(name)
    )
    coordinator.session = SimpleNamespace(session_dir=str(session_dir))

    return coordinator


class TestContextToTranscriptCycle:
    """Integration tests for the context module -> transcript tool cycle."""

    @pytest.mark.asyncio
    async def test_write_then_read(self, tmp_path):
        """add 4 messages via ctx.add_message, tool reads 2 turns with correct content."""
        coordinator = _make_coordinator(tmp_path)
        ctx = ManagedContextManager(session_dir=tmp_path)

        # Register transcript path so the tool can discover it
        coordinator.register_capability(
            "context_transcript_path", str(ctx.transcript_path)
        )

        # Add 4 messages (2 turns: user+assistant pairs)
        await ctx.add_message({"role": "user", "content": "What is Python?"})
        await ctx.add_message(
            {"role": "assistant", "content": "Python is a programming language."}
        )
        await ctx.add_message({"role": "user", "content": "What is asyncio?"})
        await ctx.add_message(
            {"role": "assistant", "content": "asyncio is for async programming."}
        )

        # Tool reads the transcript written by the context module
        tool = ReadTranscriptTool(coordinator)
        result = await tool.execute({})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "Turn 1" in result.output
        assert "Turn 2" in result.output
        assert "What is Python?" in result.output
        assert "What is asyncio?" in result.output

    @pytest.mark.asyncio
    async def test_read_specific_turn(self, tmp_path):
        """3 turns, start_turn=2 end_turn=2 returns only turn 2."""
        coordinator = _make_coordinator(tmp_path)
        ctx = ManagedContextManager(session_dir=tmp_path)

        coordinator.register_capability(
            "context_transcript_path", str(ctx.transcript_path)
        )

        # Add 3 turns (6 messages: user+assistant for each)
        await ctx.add_message({"role": "user", "content": "First question here"})
        await ctx.add_message({"role": "assistant", "content": "First answer here"})
        await ctx.add_message({"role": "user", "content": "Second question here"})
        await ctx.add_message({"role": "assistant", "content": "Second answer here"})
        await ctx.add_message({"role": "user", "content": "Third question here"})
        await ctx.add_message({"role": "assistant", "content": "Third answer here"})

        # Read only turn 2
        tool = ReadTranscriptTool(coordinator)
        result = await tool.execute({"start_turn": 2, "end_turn": 2})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "Turn 2" in result.output
        assert "Turn 1" not in result.output
        assert "Turn 3" not in result.output
        assert "Second question here" in result.output

    @pytest.mark.asyncio
    async def test_search_across_conversation(self, tmp_path):
        """search='login' returns only matching turn, not README turn."""
        coordinator = _make_coordinator(tmp_path)
        ctx = ManagedContextManager(session_dir=tmp_path)

        coordinator.register_capability(
            "context_transcript_path", str(ctx.transcript_path)
        )

        # Add a turn with README content (should NOT match "login")
        await ctx.add_message(
            {"role": "user", "content": "Read the README file for me"}
        )
        await ctx.add_message(
            {
                "role": "assistant",
                "content": "The README describes the project structure.",
            }
        )

        # Add a turn with login content (should match)
        await ctx.add_message(
            {"role": "user", "content": "How do I implement the login feature?"}
        )
        await ctx.add_message(
            {"role": "assistant", "content": "The login system uses JWT tokens."}
        )

        # Search for "login" -- only the second turn should appear
        tool = ReadTranscriptTool(coordinator)
        result = await tool.execute({"search": "login"})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "login" in result.output.lower()
        assert "README" not in result.output

    @pytest.mark.asyncio
    async def test_clear_then_read_empty(self, tmp_path):
        """add messages, ctx.clear(), tool sees empty."""
        coordinator = _make_coordinator(tmp_path)
        ctx = ManagedContextManager(session_dir=tmp_path)

        coordinator.register_capability(
            "context_transcript_path", str(ctx.transcript_path)
        )

        # Add some messages
        await ctx.add_message({"role": "user", "content": "Hello world"})
        await ctx.add_message({"role": "assistant", "content": "Hi there"})

        # Clear the context -- archives old transcript, writes fresh header
        await ctx.clear()

        # Tool reads from the now-empty transcript (just a header, no turns)
        tool = ReadTranscriptTool(coordinator)
        result = await tool.execute({})

        assert result.success is True
        assert isinstance(result.output, str)
        # After clear() the fresh transcript has no conversation turns
        assert "Turn 1" not in result.output

    @pytest.mark.asyncio
    async def test_resume_then_read(self, tmp_path):
        """create ctx1, add messages, create ctx2 from same dir, _load_from_transcript, tool reads original messages."""
        coordinator = _make_coordinator(tmp_path)

        # ctx1 writes the original messages to transcript
        ctx1 = ManagedContextManager(session_dir=tmp_path)
        coordinator.register_capability(
            "context_transcript_path", str(ctx1.transcript_path)
        )

        await ctx1.add_message({"role": "user", "content": "Original first question"})
        await ctx1.add_message(
            {"role": "assistant", "content": "Original first answer"}
        )
        await ctx1.add_message({"role": "user", "content": "Original second question"})
        await ctx1.add_message(
            {"role": "assistant", "content": "Original second answer"}
        )

        # ctx2 resumes from the same session directory
        ctx2 = ManagedContextManager(session_dir=tmp_path)
        await ctx2._load_from_transcript()

        # Tool should still read the original messages written by ctx1
        tool = ReadTranscriptTool(coordinator)
        result = await tool.execute({})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "Turn 1" in result.output
        assert "Turn 2" in result.output
        assert "Original first question" in result.output
        assert "Original second question" in result.output
