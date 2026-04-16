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
    """mount() calls register_capability with 'context-managed.transcript_path' and path to transcript.jsonl.

    mount() also always registers 'observability.events' for hooks-logging auto-discovery,
    so register_capability is now called twice when a session_id is present.
    """
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

    # Two register_capability calls: observability.events + transcript_path
    assert coordinator.register_capability.call_count == 2
    calls_by_key = {
        call[0][0]: call[0][1]
        for call in coordinator.register_capability.call_args_list
    }
    assert "context-managed.transcript_path" in calls_by_key
    cap_value = calls_by_key["context-managed.transcript_path"]
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
    """mount() with session_id=None registers observability.events but not transcript_path.

    The observability.events capability is always registered (hooks-logging needs it
    regardless of whether there is a session directory).  The transcript path capability
    is only registered when a real session directory can be computed (i.e. session_id
    is present), so it must NOT appear when session_id is None.
    """
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session_id = None
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    coordinator.mount.assert_called_once()
    # observability.events is always registered; transcript_path is not
    call_names = [call[0][0] for call in coordinator.register_capability.call_args_list]
    assert "observability.events" in call_names
    assert "context-managed.transcript_path" not in call_names


@pytest.mark.asyncio
async def test_mount_registers_observability_events():
    """mount() registers our four context:* events under 'observability.events' capability.

    hooks-logging discovers module events by calling coordinator.get_capability(
    'observability.events') during its own mount().  Without this registration our
    custom events fire into the void because no handler is ever subscribed to them.
    The registration must happen unconditionally (even with no session_id) so
    hooks-logging sees the events regardless of storage availability.
    """
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session_id = None  # No session — events registration must still happen
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    calls_by_key = {
        call[0][0]: call[0][1]
        for call in coordinator.register_capability.call_args_list
    }
    assert "observability.events" in calls_by_key, (
        "mount() must register 'observability.events' so hooks-logging can "
        "subscribe to context:* events"
    )
    registered_events = calls_by_key["observability.events"]
    for expected in [
        "context:budget_pressure",
        "context:pre_summarize",
        "context:post_summarize",
        "context:compaction",
    ]:
        assert expected in registered_events, (
            f"Expected '{expected}' in registered observability.events, "
            f"got: {registered_events}"
        )


@pytest.mark.asyncio
async def test_mount_merges_existing_observability_events():
    """mount() merges our events with any existing 'observability.events' entries.

    If another module already registered events before us, we must not
    discard them — we append our events to the existing list.
    """
    existing = ["kernel:turn_start", "kernel:turn_end"]

    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session_id = None
    coordinator.get_capability = MagicMock(
        side_effect=lambda name: existing if name == "observability.events" else None
    )
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    calls_by_key = {
        call[0][0]: call[0][1]
        for call in coordinator.register_capability.call_args_list
    }
    registered = calls_by_key["observability.events"]
    # Existing events preserved
    for ev in existing:
        assert ev in registered, f"Existing event '{ev}' was dropped"
    # Our events appended
    for ev in ["context:budget_pressure", "context:pre_summarize"]:
        assert ev in registered, f"Our event '{ev}' missing from merged list"


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
