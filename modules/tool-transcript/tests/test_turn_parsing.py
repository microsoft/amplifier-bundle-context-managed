"""
Turn parsing tests for ReadTranscriptTool._parse_transcript.

Tests the logic that groups transcript lines into turns, where a turn
starts with a user message and includes all subsequent messages until
the next user message.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from amplifier_module_tool_transcript import ReadTranscriptTool


def _write_transcript(path: Path, messages: list[dict]) -> Path:
    """Write a transcript.jsonl file with a header line followed by messages."""
    path.mkdir(parents=True, exist_ok=True)
    transcript = path / "transcript.jsonl"
    header = {
        "type": "transcript_header",
        "format_version": "1.0.0",
        "created_at": "2024-01-01T00:00:00.000+00:00",
    }
    with open(transcript, "w") as f:
        f.write(json.dumps(header) + "\n")
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return transcript


class TestTurnParsing:
    """Tests for ReadTranscriptTool._parse_transcript."""

    def setup_method(self):
        """Create a tool instance for each test."""
        coordinator = MagicMock()
        self.tool = ReadTranscriptTool(coordinator)

    # ------------------------------------------------------------------
    # Tests using the standard 3-turn sample transcript
    # ------------------------------------------------------------------

    @pytest.fixture
    def sample_transcript(self, tmp_path):
        """A transcript.jsonl with 3 turns of conversation.

        Turn 1: user asks about auth, assistant responds
        Turn 2: user asks to fix a bug, assistant uses a tool, tool responds,
                assistant responds again
        Turn 3: user asks for summary, assistant responds
        """
        messages = [
            {"role": "user", "content": "How does the auth module work?"},
            {
                "role": "assistant",
                "content": "The auth module uses JWT tokens for session management.",
            },
            {
                "role": "user",
                "content": "There's a bug in auth.py line 42. Can you fix it?",
            },
            {
                "role": "assistant",
                "content": "I'll look at that file.",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "auth.py"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "def authenticate(token):\n    return verify(token)",
                "tool_call_id": "tc_1",
            },
            {
                "role": "assistant",
                "content": "I found the issue on line 42. The token validation was missing.",
            },
            {"role": "user", "content": "Can you give me a summary of what we did?"},
            {
                "role": "assistant",
                "content": "We reviewed the auth module and fixed the token validation bug.",
            },
        ]
        return _write_transcript(tmp_path, messages)

    def test_parse_turns_from_sample(self, sample_transcript):
        """Sample transcript must parse into exactly 3 turns."""
        turns = self.tool._parse_transcript(str(sample_transcript))
        assert len(turns) == 3

    def test_turn_1_starts_with_user(self, sample_transcript):
        """Turn 1 (index 0) starts with a user message containing 'auth module'."""
        turns = self.tool._parse_transcript(str(sample_transcript))
        first_msg = turns[0][0]
        assert first_msg["role"] == "user"
        assert "auth module" in first_msg["content"]

    def test_turn_2_includes_tool_pair(self, sample_transcript):
        """Turn 2 (index 1) contains [user, assistant, tool, assistant]."""
        turns = self.tool._parse_transcript(str(sample_transcript))
        turn2 = turns[1]
        roles = [msg["role"] for msg in turn2]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_turn_3_is_final(self, sample_transcript):
        """Turn 3 (index 2) has exactly 2 messages: user then assistant."""
        turns = self.tool._parse_transcript(str(sample_transcript))
        turn3 = turns[2]
        assert len(turn3) == 2
        assert turn3[0]["role"] == "user"
        assert turn3[1]["role"] == "assistant"

    # ------------------------------------------------------------------
    # Edge-case tests using inline transcript construction
    # ------------------------------------------------------------------

    def test_empty_transcript(self, tmp_path):
        """A transcript with only a header line returns an empty list."""
        transcript = _write_transcript(tmp_path, messages=[])
        turns = self.tool._parse_transcript(str(transcript))
        assert turns == []

    def test_single_user_message(self, tmp_path):
        """A single user message produces exactly 1 turn."""
        messages = [{"role": "user", "content": "Hello, world."}]
        transcript = _write_transcript(tmp_path, messages)
        turns = self.tool._parse_transcript(str(transcript))
        assert len(turns) == 1
        assert turns[0][0]["role"] == "user"

    def test_skips_summary_markers(self, tmp_path):
        """Summary markers (metadata.type=='context_managed_summary') are skipped.

        Without skipping the marker this transcript would parse as 3 turns
        (the summary marker has role='user', which would start a new turn).
        With skipping it must parse as exactly 2 turns.
        """
        messages = [
            {"role": "user", "content": "First question."},
            {"role": "assistant", "content": "First answer."},
            # This summary marker has role='user' but must be skipped, so it
            # must NOT start a new turn.
            {
                "role": "user",
                "content": "Summary of the above.",
                "metadata": {"type": "context_managed_summary"},
            },
            {"role": "user", "content": "Second question."},
            {"role": "assistant", "content": "Second answer."},
        ]
        transcript = _write_transcript(tmp_path, messages)
        turns = self.tool._parse_transcript(str(transcript))
        assert len(turns) == 2

    def test_nonexistent_file_returns_empty(self, tmp_path):
        """Calling _parse_transcript with a path that does not exist returns []."""
        nonexistent = str(tmp_path / "does_not_exist.jsonl")
        turns = self.tool._parse_transcript(nonexistent)
        assert turns == []
