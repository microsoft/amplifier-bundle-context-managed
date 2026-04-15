"""
Mount function tests for the context-managed module.

Verifies coordinator registration, transcript path capability,
config passthrough, no-session behavior, and resume on mount.

Fix 3 note: mount() uses coordinator.session_id + CLI slug algorithm to
compute the session directory, NOT coordinator.session.session_dir (which
was always None because AmplifierSession has no session_dir attribute).
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import amplifier_module_context_managed as module
from amplifier_module_context_managed import TRANSCRIPT_FORMAT_VERSION


def _session_dir_for(home: Path, session_id: str, working_dir: Path) -> Path:
    """Compute the session directory the mount() function would produce.

    Mirrors the CLI slug algorithm from amplifier_app_cli/project_utils.py
    and the discovery logic in mount().
    """
    cwd = Path(working_dir).resolve()
    slug = str(cwd).replace("/", "-").replace("\\", "-").replace(":", "")
    if not slug.startswith("-"):
        slug = "-" + slug
    return home / ".amplifier" / "projects" / slug / "sessions" / session_id


@pytest.mark.asyncio
async def test_mount_registers_context(tmp_path, monkeypatch):
    """mount() calls coordinator.mount once with ('context', instance) where instance has add_message."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session_id = "test-session-abc"
    coordinator.get_capability = MagicMock(
        side_effect=lambda name: (
            str(tmp_path) if name == "session.working_dir" else None
        )
    )
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    coordinator.mount.assert_called_once()
    name, instance = coordinator.mount.call_args[0]
    assert name == "context"
    assert hasattr(instance, "add_message")


@pytest.mark.asyncio
async def test_mount_registers_transcript_path(tmp_path, monkeypatch):
    """mount() calls register_capability with 'context-managed.transcript_path' and path to transcript.jsonl."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session_id = "test-session-abc"
    coordinator.get_capability = MagicMock(
        side_effect=lambda name: (
            str(tmp_path) if name == "session.working_dir" else None
        )
    )
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    coordinator.register_capability.assert_called_once()
    cap_name, cap_value = coordinator.register_capability.call_args[0]
    assert cap_name == "context-managed.transcript_path"
    assert "transcript.jsonl" in cap_value
    # Must be inside the context-managed/ subdirectory
    assert "context-managed" in cap_value


@pytest.mark.asyncio
async def test_mount_passes_config():
    """mount() with config passes those values to the mounted instance."""
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    # No session_id → no session directory → no capability registered
    coordinator.session_id = None
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.register_capability = MagicMock()

    config = {"max_tokens": 100000, "pressure_warning": 0.60}

    await module.mount(coordinator, config=config)

    _, instance = coordinator.mount.call_args[0]
    assert instance.max_tokens == 100000
    assert instance.pressure_warning == 0.60


@pytest.mark.asyncio
async def test_mount_no_session_dir():
    """mount() with session_id=None succeeds and does not call register_capability."""
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session_id = None
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    coordinator.mount.assert_called_once()
    coordinator.register_capability.assert_not_called()


@pytest.mark.asyncio
async def test_mount_resumes_from_existing_transcript(tmp_path, monkeypatch):
    """When transcript exists at the computed session path, mounted instance loads messages from it."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    session_id = "test-resume-session"
    working_dir = tmp_path

    # Compute the exact path mount() will derive
    session_dir = _session_dir_for(tmp_path, session_id, working_dir)
    transcript_dir = session_dir / "context-managed"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / "transcript.jsonl"

    # Write a transcript with one message
    header = {
        "type": "transcript_header",
        "format_version": TRANSCRIPT_FORMAT_VERSION,
        "created_at": "2024-01-01T00:00:00.000+00:00",
    }
    message = {"role": "user", "content": "existing message"}
    with open(transcript_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        f.write(json.dumps(message) + "\n")

    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session_id = session_id
    coordinator.get_capability = MagicMock(
        side_effect=lambda name: (
            str(working_dir) if name == "session.working_dir" else None
        )
    )
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    _, instance = coordinator.mount.call_args[0]
    messages = await instance.get_messages()
    assert len(messages) == 1
