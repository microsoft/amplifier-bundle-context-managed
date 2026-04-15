"""
Output formatting tests for ReadTranscriptTool._format_turns.

Tests the logic that formats a list of turns into human-readable text,
including turn labels, role labels, tool calls, tool results, content
truncation, and empty list handling.
"""

from unittest.mock import MagicMock


from amplifier_module_tool_transcript import ReadTranscriptTool


class TestOutputFormatting:
    """Tests for ReadTranscriptTool._format_turns."""

    def setup_method(self):
        """Create a tool instance for each test."""
        coordinator = MagicMock()
        self.tool = ReadTranscriptTool(coordinator)

    def test_basic_format(self):
        """Turn 1 with user 'hello' and assistant 'hi there' formats correctly."""
        turns = [
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        ]
        result = self.tool._format_turns(turns, start_turn=1)

        # Should contain turn header
        assert "Turn 1" in result
        # Should contain role labels
        assert "[user]" in result
        assert "[assistant]" in result
        # Should contain message content
        assert "hello" in result
        assert "hi there" in result

    def test_multi_turn_format(self):
        """start_turn=5 shows Turn 5 and Turn 6 for a 2-turn list."""
        turns = [
            [
                {"role": "user", "content": "question five"},
                {"role": "assistant", "content": "answer five"},
            ],
            [
                {"role": "user", "content": "question six"},
                {"role": "assistant", "content": "answer six"},
            ],
        ]
        result = self.tool._format_turns(turns, start_turn=5)

        # Both turns should be labeled correctly
        assert "Turn 5" in result
        assert "Turn 6" in result
        # Should NOT show Turn 1 or Turn 4
        assert "Turn 1" not in result
        assert "Turn 4" not in result

    def test_truncates_long_content(self):
        """Content longer than 2000 chars is truncated with a note showing char count."""
        long_content = "x" * 5000
        turns = [
            [
                {"role": "user", "content": long_content},
            ]
        ]
        result = self.tool._format_turns(turns, start_turn=1)

        # Should mention truncation
        assert "truncated" in result.lower()
        # Should show original length — either 5,000 or 5000 is acceptable
        assert "5,000" in result or "5000" in result

    def test_tool_call_formatted(self):
        """Assistant tool calls show the function name; tool results show tool_call_id."""
        turns = [
            [
                {"role": "user", "content": "Read a file"},
                {
                    "role": "assistant",
                    "content": "I'll read that.",
                    "tool_calls": [
                        {
                            "id": "tc_42",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "foo.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "file contents here",
                    "tool_call_id": "tc_42",
                },
            ]
        ]
        result = self.tool._format_turns(turns, start_turn=1)

        # Tool call should reference the function name
        assert "read_file" in result
        # Tool result should use the "[tool" prefix pattern
        assert "[tool" in result.lower()

    def test_empty_turns_list(self):
        """An empty turns list returns a message indicating no content was found."""
        result = self.tool._format_turns([], start_turn=1)

        # Should indicate emptiness — 'no' or 'empty' must appear (case-insensitive)
        result_lower = result.lower()
        assert "no" in result_lower or "empty" in result_lower
