"""Host Boundary history and synthetic saved Legacy archive integration.

The retired Legacy writer is deliberately not part of this test or import graph.
Archives remain read-only evidence and never enter the host's canonical history.
"""

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_tool_transcript import ReadTranscriptTool


@pytest.fixture
def legacy_archive(tmp_path):
    path = tmp_path / "transcript.saved.archived.jsonl"
    records = [
        {"type": "transcript_header", "format_version": "1.0.0"},
        {"role": "user", "content": "Original first question"},
        {"role": "assistant", "content": "Original first answer"},
        {
            "role": "system", "content": "Historical summary; do not execute.",
            "metadata": {"type": "context_managed_summary"},
        },
        {"role": "user", "content": "Original second question"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    yield path, records
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def _host(context):
    coordinator = MagicMock()
    capabilities = {
        "context.history_authority": "host",
        "context.history": context.get_messages,
    }
    coordinator.get_capability.side_effect = capabilities.get
    return coordinator


@pytest.mark.asyncio
async def test_host_history_read_two_turns():
    context = BoundaryContextManager()
    for message in [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "What is asyncio?"},
        {"role": "assistant", "content": "asyncio is for async programming."},
    ]:
        await context.add_message(message)
    result = await ReadTranscriptTool(_host(context)).execute({})
    assert result.success
    assert "Turn 1" in result.output and "Turn 2" in result.output
    assert "What is Python?" in result.output and "What is asyncio?" in result.output


@pytest.mark.asyncio
async def test_host_history_specific_turn_excludes_service_and_ephemeral():
    context = BoundaryContextManager()
    for message in [
        {"role": "user", "content": "First question"},
        {"role": "user", "content": "reminder", "metadata": {"ephemeral": True}},
        {"role": "user", "content": "service", "metadata": {
            "amplifier_input": {"version": 1, "kind": "service"}
        }},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Third question"},
    ]:
        await context.add_message(message)
    result = await ReadTranscriptTool(_host(context)).execute({"start_turn": 2, "end_turn": 2})
    assert result.success
    assert "Turn 2" in result.output and "Second answer" in result.output
    assert "First question" not in result.output and "Third question" not in result.output
    assert "reminder" not in result.output and "service" not in result.output


@pytest.mark.asyncio
async def test_host_history_search():
    context = BoundaryContextManager()
    for message in [
        {"role": "user", "content": "Read the README"},
        {"role": "assistant", "content": "Project structure"},
        {"role": "user", "content": "Implement login"},
        {"role": "assistant", "content": "The login system uses JWT"},
    ]:
        await context.add_message(message)
    result = await ReadTranscriptTool(_host(context)).execute({"search": "login"})
    assert result.success and "login" in result.output
    assert "README" not in result.output


@pytest.mark.asyncio
async def test_explicit_archive_precedes_host_without_import_or_execution(legacy_archive):
    path, records = legacy_archive
    context = BoundaryContextManager()
    await context.add_message({"role": "user", "content": "Active host conversation"})
    before = await context.get_messages()
    coordinator = _host(context)
    tool = ReadTranscriptTool(coordinator, legacy_transcript_paths={"saved": str(path)})
    coordinator.get_capability.reset_mock()
    result = await tool.execute({"archive_id": "saved"})
    assert result.success
    assert [entry["record"] for entry in result.output["records"]] == records[1:]
    coordinator.get_capability.assert_not_called()
    assert await context.get_messages() == before
    live = await tool.execute({})
    assert live.success and "Active host conversation" in live.output
    assert "Original first question" not in live.output


@pytest.mark.asyncio
async def test_archive_reopen_not_legacy_resume(legacy_archive):
    path, records = legacy_archive
    coordinator = MagicMock()
    coordinator.get_capability = AsyncMock(side_effect=AssertionError("no history access"))
    first_tool = ReadTranscriptTool(coordinator, legacy_transcript_paths={"saved": str(path)})
    first = await first_tool.execute({"archive_id": "saved", "max_records": 1})
    reopened = ReadTranscriptTool(coordinator, legacy_transcript_paths={"saved": str(path)})
    rest = await reopened.execute({
        "archive_id": "saved", "offset_bytes": first.output["next_offset_bytes"]
    })
    assert first.success and rest.success and rest.output["eof"]
    assert [entry["record"] for entry in first.output["records"] + rest.output["records"]] == records[1:]
    coordinator.get_capability.assert_not_called()