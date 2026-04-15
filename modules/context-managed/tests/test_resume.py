"""
Session resume tests for ManagedContextManager.

Verifies transcript loading, loaded flag, token estimate, no-file behavior,
set_messages NO-OP after resume, appending after resume, and summary marker
exclusion during resume.
"""

import json
from pathlib import Path

import pytest

from amplifier_module_context_managed import (
    ManagedContextManager,
    TRANSCRIPT_FORMAT_VERSION,
)


def _write_transcript(session_dir: Path, messages: list[dict]) -> None:
    """Write a transcript.jsonl file with a header followed by the given messages."""
    transcript_path = session_dir / "transcript.jsonl"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with open(transcript_path, "w") as f:
        header = {
            "type": "transcript_header",
            "format_version": TRANSCRIPT_FORMAT_VERSION,
            "created_at": "2024-01-01T00:00:00.000+00:00",
        }
        f.write(json.dumps(header) + "\n")
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


class TestResumeLoadsMessages:
    """Tests for loading messages from a transcript on resume."""

    @pytest.mark.asyncio
    async def test_resume_loads_messages(self, tmp_session_dir: Path):
        """Write 3 messages to transcript, create new MCM, load, verify get_messages returns 3 with correct content."""
        messages = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "second message"},
            {"role": "user", "content": "third message"},
        ]
        _write_transcript(tmp_session_dir, messages)

        ctx = ManagedContextManager(session_dir=tmp_session_dir)
        await ctx._load_from_transcript()

        result = await ctx.get_messages()

        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "first message"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "second message"
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "third message"

    @pytest.mark.asyncio
    async def test_resume_sets_loaded_flag(self, tmp_session_dir: Path):
        """Write 1 message, load, verify _loaded_from_transcript is True."""
        _write_transcript(tmp_session_dir, [{"role": "user", "content": "hello"}])

        ctx = ManagedContextManager(session_dir=tmp_session_dir)
        await ctx._load_from_transcript()

        assert ctx._loaded_from_transcript is True

    @pytest.mark.asyncio
    async def test_resume_updates_token_estimate(self, tmp_session_dir: Path):
        """Write 1 message, load, verify _running_token_estimate > 0."""
        _write_transcript(tmp_session_dir, [{"role": "user", "content": "hello world"}])

        ctx = ManagedContextManager(session_dir=tmp_session_dir)
        await ctx._load_from_transcript()

        assert ctx._running_token_estimate > 0


class TestResumeNoFile:
    """Tests for no-op behavior when no transcript file exists."""

    @pytest.mark.asyncio
    async def test_resume_no_file_is_noop(self, tmp_session_dir: Path):
        """No transcript file, load, verify _loaded_from_transcript is False and get_messages returns empty."""
        ctx = ManagedContextManager(session_dir=tmp_session_dir)
        await ctx._load_from_transcript()

        assert ctx._loaded_from_transcript is False
        result = await ctx.get_messages()
        assert result == []


class TestSetMessagesAfterResume:
    """Tests for set_messages behavior after resume."""

    @pytest.mark.asyncio
    async def test_set_messages_noop_after_resume(self, tmp_session_dir: Path):
        """Write 'From transcript', load, call set_messages with 'From kernel', verify get_messages returns 'From transcript'."""
        _write_transcript(
            tmp_session_dir, [{"role": "user", "content": "From transcript"}]
        )

        ctx = ManagedContextManager(session_dir=tmp_session_dir)
        await ctx._load_from_transcript()

        # set_messages should be NO-OP since we loaded from transcript
        await ctx.set_messages([{"role": "user", "content": "From kernel"}])

        result = await ctx.get_messages()

        assert len(result) == 1
        assert result[0]["content"] == "From transcript"
        # Explicitly verify 'From kernel' is NOT present
        contents = [msg["content"] for msg in result]
        assert "From kernel" not in contents

    @pytest.mark.asyncio
    async def test_set_messages_accepted_when_fresh(self, tmp_session_dir: Path):
        """No transcript, load is noop, set_messages with 'From kernel', verify get_messages returns 'From kernel'."""
        ctx = ManagedContextManager(session_dir=tmp_session_dir)
        await ctx._load_from_transcript()  # no-op since no file

        await ctx.set_messages([{"role": "user", "content": "From kernel"}])

        result = await ctx.get_messages()

        assert len(result) == 1
        assert result[0]["content"] == "From kernel"


class TestResumeAndAppend:
    """Tests for appending new messages after resume."""

    @pytest.mark.asyncio
    async def test_resume_then_add_new_messages(self, tmp_session_dir: Path):
        """Write 'Old msg', load, add_message 'New reply', verify get_messages returns both in order."""
        _write_transcript(tmp_session_dir, [{"role": "user", "content": "Old msg"}])

        ctx = ManagedContextManager(session_dir=tmp_session_dir)
        await ctx._load_from_transcript()

        await ctx.add_message({"role": "assistant", "content": "New reply"})

        result = await ctx.get_messages()

        assert len(result) == 2
        assert result[0]["content"] == "Old msg"
        assert result[1]["content"] == "New reply"


class TestResumeSummaryMarkerExclusion:
    """Tests for exclusion of summary markers during resume."""

    @pytest.mark.asyncio
    async def test_resume_skips_summary_markers(self, tmp_session_dir: Path):
        """Write user msg + summary marker + assistant msg, load, verify get_messages returns 2 messages excluding summary."""
        messages = [
            {"role": "user", "content": "user message"},
            {
                "role": "system",
                "content": "summary of earlier conversation",
                "metadata": {"type": "context_managed_summary", "tier": 0},
            },
            {"role": "assistant", "content": "assistant message"},
        ]
        _write_transcript(tmp_session_dir, messages)

        ctx = ManagedContextManager(session_dir=tmp_session_dir)
        await ctx._load_from_transcript()

        result = await ctx.get_messages()

        assert len(result) == 2
        contents = [msg["content"] for msg in result]
        assert "user message" in contents
        assert "assistant message" in contents
        assert "summary of earlier conversation" not in contents
