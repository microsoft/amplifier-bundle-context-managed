"""
Test fixtures for tool-transcript module tests.

Provides:
- tmp_session_dir: temporary session directory
- sample_transcript: a transcript.jsonl file with known content
- mock_coordinator: a mock coordinator with get_capability support
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


TRANSCRIPT_FORMAT_VERSION = "1.0.0"


def _write_transcript(path: Path, messages: list[dict]) -> Path:
    """Write a transcript.jsonl file with header + messages."""
    path.mkdir(parents=True, exist_ok=True)
    transcript = path / "transcript.jsonl"
    header = {
        "type": "transcript_header",
        "format_version": TRANSCRIPT_FORMAT_VERSION,
        "created_at": "2024-01-01T00:00:00.000+00:00",
    }
    with open(transcript, "w") as f:
        f.write(json.dumps(header) + "\n")
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return transcript


@pytest.fixture
def tmp_session_dir(tmp_path):
    """Temporary session directory for transcript files."""
    return tmp_path


@pytest.fixture
def sample_transcript(tmp_session_dir):
    """A transcript.jsonl with 3 turns of conversation.

    Turn 1: user asks about auth, assistant responds
    Turn 2: user asks to fix a bug, assistant uses a tool, tool responds
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
    path = _write_transcript(tmp_session_dir, messages)
    return path


@pytest.fixture
def mock_coordinator(tmp_session_dir, sample_transcript):
    """Mock coordinator that returns a transcript path via get_capability.

    Uses the namespaced key "context-managed.transcript_path" that the
    context-managed module registers at mount time.
    """
    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(
        side_effect=lambda name: (
            str(sample_transcript)
            if name == "context-managed.transcript_path"
            else None
        )
    )
    return coordinator


@pytest.fixture
def mock_coordinator_no_transcript():
    """Mock coordinator with no transcript path registered."""
    coordinator = MagicMock()
    coordinator.get_capability = MagicMock(return_value=None)
    return coordinator
