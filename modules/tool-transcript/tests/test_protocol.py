"""
Protocol compliance tests for ReadTranscriptTool.

Verifies the Amplifier Tool protocol: name, description, input_schema,
and execute() all meet the required contract.
"""

import pytest
from unittest.mock import MagicMock

from amplifier_module_tool_transcript import ReadTranscriptTool
from amplifier_core import ToolResult


class TestToolProtocol:
    """Protocol compliance tests for ReadTranscriptTool."""

    def setup_method(self):
        """Create a tool instance for each test."""
        coordinator = MagicMock()
        self.tool = ReadTranscriptTool(coordinator)

    def test_has_name(self):
        """Tool name must be 'read_transcript'."""
        assert self.tool.name == "read_transcript"

    def test_has_description(self):
        """Description must be a str with len>50 and contain 'summar'."""
        desc = self.tool.description
        assert isinstance(desc, str)
        assert len(desc) > 50
        assert "summar" in desc.lower()

    def test_has_input_schema(self):
        """Input schema must be a dict with type='object' and start_turn/end_turn/search."""
        schema = self.tool.input_schema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        props = schema.get("properties", {})
        assert "start_turn" in props
        assert "end_turn" in props
        assert "search" in props

    def test_input_schema_no_required_fields(self):
        """All input schema fields must be optional (no required fields)."""
        schema = self.tool.input_schema
        required = schema.get("required", [])
        assert required == []

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self):
        """execute() must return a ToolResult instance."""
        result = await self.tool.execute({})
        assert isinstance(result, ToolResult)
