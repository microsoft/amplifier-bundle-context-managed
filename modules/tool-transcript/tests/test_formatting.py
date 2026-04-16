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

    def test_tool_result_content_shown(self):
        """Tool result messages show full content, not just the id."""
        turns = [
            [
                {"role": "user", "content": "Read auth.py"},
                {
                    "role": "assistant",
                    "content": "I'll read that.",
                    "tool_calls": [
                        {
                            "id": "tc_99",
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
                    "tool_call_id": "tc_99",
                },
            ]
        ]
        result = self.tool._format_turns(turns, start_turn=1)

        # Content should be present, not just the ID
        assert "def authenticate" in result
        assert "return verify" in result
        # ID should also appear
        assert "tc_99" in result

    def test_tool_call_block_in_assistant_content(self):
        """tool_call content blocks in assistant messages are shown as [tool_call: name(input)]."""
        turns = [
            [
                {"role": "user", "content": "Search for TODO comments"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll search for that."},
                        {
                            "type": "tool_call",
                            "id": "tc_block",
                            "name": "grep",
                            "input": {"pattern": "TODO", "path": "src/"},
                        },
                    ],
                },
            ]
        ]
        result = self.tool._format_turns(turns, start_turn=1)

        assert "[tool_call: grep(" in result
        assert "TODO" in result
        assert "I'll search for that." in result

    def test_tool_calls_field_shown_both_formats(self):
        """tool_calls field entries are shown for both {function.name} and {name/tool} formats."""
        # OpenAI format: {function: {name, arguments}}
        turns_openai = [
            [
                {"role": "user", "content": "Run tests"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_oai",
                            "function": {
                                "name": "run_tests",
                                "arguments": '{"suite":"unit"}',
                            },
                        }
                    ],
                },
            ]
        ]
        result = self.tool._format_turns(turns_openai, start_turn=1)
        assert "run_tests" in result
        assert "unit" in result

        # Direct format: {name, input}
        turns_direct = [
            [
                {"role": "user", "content": "Check code"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_direct",
                            "name": "check_code",
                            "input": {"path": "main.py"},
                        }
                    ],
                },
            ]
        ]
        result2 = self.tool._format_turns(turns_direct, start_turn=1)
        assert "check_code" in result2
        assert "main.py" in result2
