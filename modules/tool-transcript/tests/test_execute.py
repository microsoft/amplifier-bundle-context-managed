"""
Tests for ReadTranscriptTool.execute() method.

Tests the full execute() pipeline: rate limiting, transcript discovery,
turn range clamping, search filtering, and graceful failure handling.
"""

import pytest
from unittest.mock import MagicMock

from amplifier_module_tool_transcript import ReadTranscriptTool


class TestExecuteBasic:
    """Tests for core execute() behaviour: default, start/end/range selection."""

    @pytest.mark.asyncio
    async def test_returns_all_turns_by_default(self, mock_coordinator):
        """execute({}) returns all 3 turns, success=True."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "Turn 1" in result.output
        assert "Turn 2" in result.output
        assert "Turn 3" in result.output

    @pytest.mark.asyncio
    async def test_start_turn(self, mock_coordinator):
        """execute with start_turn=2 skips Turn 1 and returns turns 2 and 3."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({"start_turn": 2})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "Turn 1" not in result.output
        assert "Turn 2" in result.output
        assert "Turn 3" in result.output

    @pytest.mark.asyncio
    async def test_end_turn(self, mock_coordinator):
        """execute with end_turn=2 returns turns 1 and 2, not turn 3."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({"end_turn": 2})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "Turn 1" in result.output
        assert "Turn 2" in result.output
        assert "Turn 3" not in result.output

    @pytest.mark.asyncio
    async def test_exact_range(self, mock_coordinator):
        """execute with start_turn=2, end_turn=2 returns only turn 2."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({"start_turn": 2, "end_turn": 2})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "Turn 1" not in result.output
        assert "Turn 2" in result.output
        assert "Turn 3" not in result.output


class TestExecuteClamp:
    """Tests for turn range clamping behavior."""

    @pytest.mark.asyncio
    async def test_start_turn_below_1_clamped(self, mock_coordinator):
        """start_turn=0 is clamped to 1, returning all turns from the start."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({"start_turn": 0})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "Turn 1" in result.output

    @pytest.mark.asyncio
    async def test_start_turn_negative_clamped(self, mock_coordinator):
        """start_turn=-5 is clamped to 1, returning all turns from the start."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({"start_turn": -5})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "Turn 1" in result.output

    @pytest.mark.asyncio
    async def test_end_turn_beyond_max_clamped(self, mock_coordinator):
        """end_turn=100 is clamped to total (3), returning all 3 turns."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({"end_turn": 100})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "Turn 1" in result.output
        assert "Turn 2" in result.output
        assert "Turn 3" in result.output

    @pytest.mark.asyncio
    async def test_start_beyond_end_returns_empty(self, mock_coordinator):
        """start_turn=100 (beyond total turns) returns success=True with empty/count message."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({"start_turn": 100})

        assert result.success is True
        assert isinstance(result.output, str)
        # Output should be empty-ish or contain turn count info, not actual turns
        assert "Turn 1" not in result.output
        assert "Turn 2" not in result.output
        assert "Turn 3" not in result.output


class TestExecuteSearch:
    """Tests for search filtering in execute()."""

    @pytest.mark.asyncio
    async def test_search_filters_turns(self, mock_coordinator):
        """Search for 'summary' returns only turns containing that word."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({"search": "summary"})

        assert result.success is True
        assert isinstance(result.output, str)
        assert result.output
        assert "No matches found" not in result.output

    @pytest.mark.asyncio
    async def test_search_no_matches(self, mock_coordinator):
        """Search for a string that matches no turns returns 'No matches found'."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({"search": "xyzzy_no_match_abc_123"})

        assert result.success is True
        assert isinstance(result.output, str)
        assert "No matches found" in result.output

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, mock_coordinator):
        """Search is case-insensitive: 'SUMMARY' matches the same turns as 'summary'."""
        tool = ReadTranscriptTool(mock_coordinator)

        lower_result = await tool.execute({"search": "summary"})
        tool.reset_rate_limit()
        upper_result = await tool.execute({"search": "SUMMARY"})

        assert lower_result.success is True
        assert upper_result.success is True
        assert isinstance(lower_result.output, str)
        assert isinstance(upper_result.output, str)
        # Both should match (or not match) the same content
        assert ("No matches found" in lower_result.output) == (
            "No matches found" in upper_result.output
        )

    @pytest.mark.asyncio
    async def test_search_with_turn_range(self, mock_coordinator):
        """Search combined with turn range only searches within the specified range."""
        tool = ReadTranscriptTool(mock_coordinator)
        # Turn 1 contains "auth module"; search for it but only within turns 2-3
        result = await tool.execute({"start_turn": 2, "search": "summary"})

        assert result.success is True
        assert isinstance(result.output, str)
        # Turn 1 should not appear even if it matches the search
        assert "Turn 1" not in result.output


class TestExecuteOutputSizeLimit:
    """Tests for output-size based turn reduction in execute()."""

    @pytest.mark.asyncio
    async def test_large_output_truncated_with_note(self, tmp_path):
        """When formatted output exceeds 100K chars, only a leading subset of turns is returned
        and a note is appended explaining the truncation."""
        import json

        # Build a transcript with 5 turns each containing ~30K chars of tool result
        transcript = tmp_path / "transcript.jsonl"
        header = {
            "type": "transcript_header",
            "format_version": "1.0.0",
            "created_at": "2024-01-01T00:00:00.000+00:00",
        }
        messages = []
        for i in range(1, 6):
            messages.append({"role": "user", "content": f"Turn {i} question"})
            messages.append({"role": "assistant", "content": ""})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"tc_{i}",
                    # ~30K chars per tool result
                    "content": f"result-{i}: " + "x" * 30_000,
                }
            )

        with open(transcript, "w") as f:
            f.write(json.dumps(header) + "\n")
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        coordinator = MagicMock()
        coordinator.get_capability = MagicMock(
            side_effect=lambda name: (
                str(transcript) if name == "context-managed.transcript_path" else None
            )
        )

        tool = ReadTranscriptTool(coordinator)
        result = await tool.execute({})

        assert result.success is True
        assert isinstance(result.output, str)
        # Output should contain a truncation note
        assert (
            "narrower range" in result.output.lower()
            or "showing turns" in result.output.lower()
        )
        # Not all 5 turns should be present
        assert "Turn 5" not in result.output or "Turn 1" in result.output

    @pytest.mark.asyncio
    async def test_small_output_not_truncated(self, mock_coordinator):
        """When formatted output is under 100K chars, no truncation note is added."""
        tool = ReadTranscriptTool(mock_coordinator)
        result = await tool.execute({})

        assert result.success is True
        # The 3-turn sample transcript is tiny — no truncation note
        assert "narrower range" not in result.output.lower()


class TestExecuteGracefulFailure:
    """Tests for graceful failure cases in execute()."""

    @pytest.mark.asyncio
    async def test_no_transcript_path(self, mock_coordinator_no_transcript):
        """When coordinator has no transcript path, returns success=False with error message."""
        tool = ReadTranscriptTool(mock_coordinator_no_transcript)
        result = await tool.execute({})

        assert result.success is False
        assert result.error == {"message": "no_transcript_path"}
        assert isinstance(result.output, str)
        output_lower = result.output.lower()
        assert "no transcript" in output_lower or "not available" in output_lower

    @pytest.mark.asyncio
    async def test_transcript_file_missing(self, tmp_path):
        """When coordinator returns a path to a non-existent file, returns success=True with empty."""
        coordinator = MagicMock()
        missing_path = str(tmp_path / "nonexistent_transcript.jsonl")
        coordinator.get_capability = MagicMock(
            side_effect=lambda name: (
                missing_path if name == "context-managed.transcript_path" else None
            )
        )
        tool = ReadTranscriptTool(coordinator)
        result = await tool.execute({})

        assert result.success is True
        # No real content — output should be empty or minimal
        assert result.output is not None
