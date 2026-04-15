"""
Mount function tests for the context-managed module.

Verifies coordinator registration, transcript path capability,
config passthrough, no-session behavior, and resume on mount.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import amplifier_module_context_managed as module
from amplifier_module_context_managed import TRANSCRIPT_FORMAT_VERSION


@pytest.mark.asyncio
async def test_mount_registers_context(tmp_path):
    """mount() calls coordinator.mount once with ('context', instance) where instance has add_message."""
    session = SimpleNamespace(session_dir=tmp_path)
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session = session
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    coordinator.mount.assert_called_once()
    name, instance = coordinator.mount.call_args[0]
    assert name == "context"
    assert hasattr(instance, "add_message")


@pytest.mark.asyncio
async def test_mount_registers_transcript_path(tmp_path):
    """mount() calls register_capability with 'context_transcript_path' and path to transcript.jsonl."""
    session = SimpleNamespace(session_dir=tmp_path)
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session = session
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    coordinator.register_capability.assert_called_once()
    cap_name, cap_value = coordinator.register_capability.call_args[0]
    assert cap_name == "context_transcript_path"
    assert "transcript.jsonl" in cap_value


@pytest.mark.asyncio
async def test_mount_passes_config():
    """mount() with config passes those values to the mounted instance."""
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session = None
    coordinator.register_capability = MagicMock()

    config = {"max_tokens": 100000, "pressure_warning": 0.60}

    await module.mount(coordinator, config=config)

    _, instance = coordinator.mount.call_args[0]
    assert instance.max_tokens == 100000
    assert instance.pressure_warning == 0.60


@pytest.mark.asyncio
async def test_mount_no_session_dir():
    """mount() with session=None succeeds and does not call register_capability."""
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session = None
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    coordinator.mount.assert_called_once()
    coordinator.register_capability.assert_not_called()


@pytest.mark.asyncio
async def test_mount_resumes_from_existing_transcript(tmp_path):
    """When transcript exists, mounted instance loads messages from it via get_messages."""
    # Write transcript with header + 1 message
    transcript_path = tmp_path / "transcript.jsonl"
    header = {
        "type": "transcript_header",
        "format_version": TRANSCRIPT_FORMAT_VERSION,
        "created_at": "2024-01-01T00:00:00.000+00:00",
    }
    message = {"role": "user", "content": "existing message"}
    with open(transcript_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        f.write(json.dumps(message) + "\n")

    session = SimpleNamespace(session_dir=tmp_path)
    coordinator = MagicMock()
    coordinator.mount = AsyncMock()
    coordinator.hooks = MagicMock()
    coordinator.session = session
    coordinator.register_capability = MagicMock()

    await module.mount(coordinator)

    _, instance = coordinator.mount.call_args[0]
    messages = await instance.get_messages()
    assert len(messages) == 1
