"""
Persistence tests for ManagedContextManager.

Verifies JSONL transcript creation, header format, message persistence,
reading, header/summary exclusion, no-disk mode, and malformed line handling.
"""

import json
import time
from pathlib import Path

import pytest

from amplifier_module_context_managed import (
    ManagedContextManager,
    TRANSCRIPT_FORMAT_VERSION,
)


class TestTranscriptCreation:
    """Tests for transcript file creation and header format."""

    @pytest.mark.asyncio
    async def test_transcript_created_on_first_message(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """transcript.jsonl doesn't exist before, exists after add_message."""
        transcript_path = tmp_session_dir / "transcript.jsonl"
        assert not transcript_path.exists()

        await context.add_message({"role": "user", "content": "hello"})

        assert transcript_path.exists()

    @pytest.mark.asyncio
    async def test_transcript_has_header(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """First line is JSON with type='transcript_header', correct format_version, and 'created_at'."""
        await context.add_message({"role": "user", "content": "hello"})

        transcript_path = tmp_session_dir / "transcript.jsonl"
        with open(transcript_path) as f:
            first_line = f.readline().strip()

        header = json.loads(first_line)
        assert header["type"] == "transcript_header"
        assert header["format_version"] == TRANSCRIPT_FORMAT_VERSION
        assert "created_at" in header


class TestMessagePersistence:
    """Tests for message persistence to JSONL."""

    @pytest.mark.asyncio
    async def test_messages_persisted_to_jsonl(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """Add 2 messages, read file, verify 3 lines: header + 2 messages with correct roles/content."""
        await context.add_message({"role": "user", "content": "first message"})
        await context.add_message({"role": "assistant", "content": "second message"})

        transcript_path = tmp_session_dir / "transcript.jsonl"
        with open(transcript_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 3  # header + 2 messages

        header = json.loads(lines[0])
        assert header["type"] == "transcript_header"

        msg1 = json.loads(lines[1])
        assert msg1["role"] == "user"
        assert msg1["content"] == "first message"

        msg2 = json.loads(lines[2])
        assert msg2["role"] == "assistant"
        assert msg2["content"] == "second message"

    @pytest.mark.asyncio
    async def test_get_messages_reads_from_transcript(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """Add 2 messages, get_messages returns them correctly."""
        await context.add_message({"role": "user", "content": "first message"})
        await context.add_message({"role": "assistant", "content": "second message"})

        messages = await context.get_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "first message"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "second message"

    @pytest.mark.asyncio
    async def test_get_messages_excludes_header(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """Add 1 message, get_messages returns exactly 1 message not the header."""
        await context.add_message({"role": "user", "content": "test message"})

        messages = await context.get_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test message"


class TestSummaryMarkerExclusion:
    """Tests for exclusion of summary markers from get_messages."""

    @pytest.mark.asyncio
    async def test_get_messages_excludes_summary_markers(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """Add message, manually write summary marker to transcript, add another message, get_messages returns only the 2 conversation messages."""
        await context.add_message({"role": "user", "content": "first message"})

        # Manually write a summary marker directly to the transcript file
        transcript_path = tmp_session_dir / "transcript.jsonl"
        summary_marker = {
            "role": "system",
            "content": "summary of earlier conversation",
            "metadata": {"type": "context_managed_summary", "tier": 0},
        }
        with open(transcript_path, "a") as f:
            f.write(json.dumps(summary_marker) + "\n")

        await context.add_message({"role": "assistant", "content": "second message"})

        messages = await context.get_messages()

        assert len(messages) == 2
        assert messages[0]["content"] == "first message"
        assert messages[1]["content"] == "second message"


class TestNoDiskMode:
    """Tests for no-disk mode (in-memory only)."""

    @pytest.mark.asyncio
    async def test_no_disk_mode_works(self, context_no_disk: ManagedContextManager):
        """context_no_disk add and get works without any session directory."""
        await context_no_disk.add_message(
            {"role": "user", "content": "no disk message"}
        )
        await context_no_disk.add_message(
            {"role": "assistant", "content": "no disk reply"}
        )

        messages = await context_no_disk.get_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "no disk message"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "no disk reply"


class TestMalformedLineHandling:
    """Tests for graceful handling of malformed lines in transcript."""

    @pytest.mark.asyncio
    async def test_transcript_handles_malformed_lines(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """Add message, inject non-JSON line, add another message, get_messages returns both valid messages."""
        await context.add_message({"role": "user", "content": "first message"})

        # Inject a non-JSON line directly into the transcript file
        transcript_path = tmp_session_dir / "transcript.jsonl"
        with open(transcript_path, "a") as f:
            f.write("this is not valid JSON!!!\n")

        await context.add_message({"role": "assistant", "content": "second message"})

        messages = await context.get_messages()

        assert len(messages) == 2
        assert messages[0]["content"] == "first message"
        assert messages[1]["content"] == "second message"


class TestLargeToolResultHandling:
    """Tests for large tool result pointer file creation and in-memory truncation."""

    @pytest.mark.asyncio
    async def test_large_tool_result_creates_pointer_file(self, tmp_session_dir: Path):
        """Large tool result creates pointer file in tool_results/ with full content."""
        context = ManagedContextManager(
            session_dir=tmp_session_dir, large_result_threshold=100
        )
        large_content = "x" * 200
        await context.add_message(
            {
                "role": "tool",
                "content": large_content,
                "tool_call_id": "call_abc",
            }
        )

        tool_results_dir = tmp_session_dir / "tool_results"
        assert tool_results_dir.exists()
        files = list(tool_results_dir.iterdir())
        assert len(files) == 1
        assert "call_abc" in files[0].name
        assert files[0].read_text() == large_content

    @pytest.mark.asyncio
    async def test_large_result_truncated_in_memory(self, tmp_session_dir: Path):
        """Large tool result is truncated in memory with [truncated: marker."""
        context = ManagedContextManager(
            session_dir=tmp_session_dir, large_result_threshold=100
        )
        large_content = "x" * 200
        await context.add_message(
            {
                "role": "tool",
                "content": large_content,
                "tool_call_id": "call_abc",
            }
        )

        messages = await context.get_messages()
        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) < 200
        assert "[truncated:" in content

    @pytest.mark.asyncio
    async def test_large_result_metadata_has_pointer(self, tmp_session_dir: Path):
        """Large tool result metadata has full_result_path and original_length=200."""
        context = ManagedContextManager(
            session_dir=tmp_session_dir, large_result_threshold=100
        )
        large_content = "x" * 200
        await context.add_message(
            {
                "role": "tool",
                "content": large_content,
                "tool_call_id": "call_abc",
            }
        )

        messages = await context.get_messages()
        meta = messages[0].get("metadata") or {}
        assert "full_result_path" in meta
        assert meta["original_length"] == 200

    @pytest.mark.asyncio
    async def test_small_tool_result_not_truncated(self, tmp_session_dir: Path):
        """Small tool result is not truncated and has no full_result_path in metadata."""
        context = ManagedContextManager(
            session_dir=tmp_session_dir, large_result_threshold=100
        )
        await context.add_message(
            {
                "role": "tool",
                "content": "small result",
                "tool_call_id": "call_small",
            }
        )

        messages = await context.get_messages()
        assert messages[0]["content"] == "small result"
        meta = messages[0].get("metadata") or {}
        assert "full_result_path" not in meta

    @pytest.mark.asyncio
    async def test_non_tool_messages_not_affected_by_large_result(
        self, tmp_session_dir: Path
    ):
        """Non-tool messages with large content are not truncated."""
        context = ManagedContextManager(
            session_dir=tmp_session_dir, large_result_threshold=100
        )
        large_content = "x" * 200
        await context.add_message(
            {
                "role": "user",
                "content": large_content,
            }
        )

        messages = await context.get_messages()
        assert messages[0]["content"] == large_content
        assert len(messages[0]["content"]) == 200


class TestClearAndArchive:
    """Tests for clear() method: archiving transcript, resetting state, no-disk mode."""

    @pytest.mark.asyncio
    async def test_clear_archives_transcript(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """Add message, verify transcript exists, clear, verify original transcript
        gone, verify transcript.*.archived.jsonl exists."""
        await context.add_message({"role": "user", "content": "hello"})

        transcript_path = tmp_session_dir / "transcript.jsonl"
        assert transcript_path.exists()

        await context.clear()

        # Verify archive file exists
        archived_files = list(tmp_session_dir.glob("transcript.*.archived.jsonl"))
        assert len(archived_files) == 1

        # Verify the archive name contains a valid timestamp (YYYYMMDDTHHMMSS)
        archive_name = archived_files[0].name
        timestamp_str = archive_name.split(".")[1]
        parsed = time.strptime(timestamp_str, "%Y%m%dT%H%M%S")
        assert parsed is not None

    @pytest.mark.asyncio
    async def test_clear_creates_fresh_transcript(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """Add message, clear, verify new transcript.jsonl exists with just 1 line
        being the header."""
        await context.add_message({"role": "user", "content": "hello"})

        await context.clear()

        transcript_path = tmp_session_dir / "transcript.jsonl"
        assert transcript_path.exists()

        with open(transcript_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 1
        header = json.loads(lines[0])
        assert header["type"] == "transcript_header"

    @pytest.mark.asyncio
    async def test_clear_resets_in_memory_state(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """Add 2 messages, clear, verify get_messages returns empty,
        _running_token_estimate==0, _message_index==0, _loaded_from_transcript==False,
        _pressure_emitted==False."""
        await context.add_message({"role": "user", "content": "first"})
        await context.add_message({"role": "assistant", "content": "second"})

        await context.clear()

        messages = await context.get_messages()
        assert messages == []
        assert context._running_token_estimate == 0
        assert context._message_index == 0
        assert context._loaded_from_transcript is False
        assert context._pressure_emitted is False

    @pytest.mark.asyncio
    async def test_clear_archives_tool_results(self, tmp_session_dir: Path):
        """Create ManagedContextManager with large_result_threshold=10, add tool message
        with 100-char content, verify tool_results exists, clear, verify tool_results
        gone, verify tool_results.*.archived exists."""
        context = ManagedContextManager(
            session_dir=tmp_session_dir, large_result_threshold=10
        )
        large_content = "x" * 100
        await context.add_message(
            {
                "role": "tool",
                "content": large_content,
                "tool_call_id": "call_test",
            }
        )

        tool_results_dir = tmp_session_dir / "tool_results"
        assert tool_results_dir.exists()

        await context.clear()

        # tool_results/ should be gone (renamed to archive)
        assert not tool_results_dir.exists()

        # tool_results.*.archived should exist
        archived_dirs = list(tmp_session_dir.glob("tool_results.*.archived"))
        assert len(archived_dirs) == 1

    @pytest.mark.asyncio
    async def test_clear_when_no_transcript(
        self, context_no_disk: ManagedContextManager
    ):
        """context_no_disk, add message, clear doesn't raise, get_messages returns empty."""
        await context_no_disk.add_message({"role": "user", "content": "hello"})

        # Should not raise
        await context_no_disk.clear()

        messages = await context_no_disk.get_messages()
        assert messages == []


class TestFormatVersioning:
    """Tests verifying TRANSCRIPT_FORMAT_VERSION in transcript headers."""

    @pytest.mark.asyncio
    async def test_fresh_transcript_has_version(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """Add message, read first line, verify format_version == TRANSCRIPT_FORMAT_VERSION == '1.0.0'."""
        await context.add_message({"role": "user", "content": "hello"})

        transcript_path = tmp_session_dir / "transcript.jsonl"
        with open(transcript_path) as f:
            first_line = f.readline().strip()

        header = json.loads(first_line)
        assert TRANSCRIPT_FORMAT_VERSION == "1.0.0"
        assert header["format_version"] == TRANSCRIPT_FORMAT_VERSION

    @pytest.mark.asyncio
    async def test_clear_writes_new_header(
        self, context: ManagedContextManager, tmp_session_dir: Path
    ):
        """Add message, clear, read first line of new transcript, verify type=='transcript_header'
        and format_version==TRANSCRIPT_FORMAT_VERSION."""
        await context.add_message({"role": "user", "content": "hello"})

        await context.clear()

        transcript_path = tmp_session_dir / "transcript.jsonl"
        with open(transcript_path) as f:
            first_line = f.readline().strip()

        header = json.loads(first_line)
        assert header["type"] == "transcript_header"
        assert header["format_version"] == TRANSCRIPT_FORMAT_VERSION
