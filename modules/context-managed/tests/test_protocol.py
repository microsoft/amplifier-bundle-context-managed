"""
Protocol compliance tests for ManagedContextManager.

Verifies all ContextManager protocol methods exist, are callable,
and basic round-trip operations work correctly.
"""

import pytest
from datetime import datetime, UTC


class TestProtocolMethodsExist:
    """Verify all ContextManager protocol methods exist and are callable."""

    @pytest.mark.asyncio
    async def test_has_add_message(self, context_no_disk):
        assert hasattr(context_no_disk, "add_message")
        assert callable(context_no_disk.add_message)

    @pytest.mark.asyncio
    async def test_has_get_messages_for_request(self, context_no_disk):
        assert hasattr(context_no_disk, "get_messages_for_request")
        assert callable(context_no_disk.get_messages_for_request)

    @pytest.mark.asyncio
    async def test_has_get_messages(self, context_no_disk):
        assert hasattr(context_no_disk, "get_messages")
        assert callable(context_no_disk.get_messages)

    @pytest.mark.asyncio
    async def test_has_set_messages(self, context_no_disk):
        assert hasattr(context_no_disk, "set_messages")
        assert callable(context_no_disk.set_messages)

    @pytest.mark.asyncio
    async def test_has_clear(self, context_no_disk):
        assert hasattr(context_no_disk, "clear")
        assert callable(context_no_disk.clear)

    @pytest.mark.asyncio
    async def test_has_set_system_prompt_factory(self, context_no_disk):
        assert hasattr(context_no_disk, "set_system_prompt_factory")
        assert callable(context_no_disk.set_system_prompt_factory)


class TestProtocolBehavior:
    """Verify basic protocol operations work correctly."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_message(self, context_no_disk):
        """Round-trip: add a message then retrieve it, verify role and content."""
        message = {"role": "user", "content": "Hello, world!"}
        await context_no_disk.add_message(message)

        messages = await context_no_disk.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_get_messages_returns_copy(self, context_no_disk):
        """Two get_messages() calls return different list objects."""
        await context_no_disk.add_message({"role": "user", "content": "test"})

        first = await context_no_disk.get_messages()
        second = await context_no_disk.get_messages()

        assert first is not second

    @pytest.mark.asyncio
    async def test_clear_resets_state(self, context_no_disk):
        """Add messages, then clear, then get_messages returns empty."""
        await context_no_disk.add_message({"role": "user", "content": "message 1"})
        await context_no_disk.add_message({"role": "assistant", "content": "reply"})

        await context_no_disk.clear()

        messages = await context_no_disk.get_messages()
        assert messages == []

    @pytest.mark.asyncio
    async def test_set_messages_when_fresh(self, context_no_disk):
        """set_messages initializes from external list when no transcript loaded."""
        external_messages = [
            {"role": "user", "content": "external message 1"},
            {"role": "assistant", "content": "external reply"},
        ]
        await context_no_disk.set_messages(external_messages)

        messages = await context_no_disk.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "external message 1"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "external reply"

    @pytest.mark.asyncio
    async def test_get_messages_for_request_returns_messages(self, context_no_disk):
        """get_messages_for_request returns the conversation messages."""
        await context_no_disk.add_message({"role": "user", "content": "hello"})
        await context_no_disk.add_message({"role": "assistant", "content": "hi there"})

        result = await context_no_disk.get_messages_for_request()

        assert isinstance(result, list)
        assert len(result) >= 1
        # Verify conversation messages are present
        roles = [m["role"] for m in result]
        assert "user" in roles
        assert "assistant" in roles


class TestTimestampBehavior:
    """Verify timestamp injection and mutation behavior in add_message."""

    @pytest.mark.asyncio
    async def test_timestamp_injected_automatically(self, context_no_disk):
        """add_message without metadata injects metadata.timestamp as valid ISO datetime."""
        message = {"role": "user", "content": "hello"}
        await context_no_disk.add_message(message)

        messages = await context_no_disk.get_messages()
        assert len(messages) == 1
        stored = messages[0]
        assert "metadata" in stored
        assert "timestamp" in stored["metadata"]
        # Verify it's a valid ISO datetime string close to the current time
        ts = stored["metadata"]["timestamp"]
        parsed = datetime.fromisoformat(ts)  # Raises ValueError if not valid ISO
        age_seconds = abs((datetime.now(UTC) - parsed).total_seconds())
        assert age_seconds < 5  # Timestamp should be very recent

    @pytest.mark.asyncio
    async def test_timestamp_preserved_when_present(self, context_no_disk):
        """add_message with existing timestamp preserves it unchanged."""
        existing_ts = "2026-01-15T10:00:00.123+00:00"
        message = {
            "role": "user",
            "content": "hello",
            "metadata": {"timestamp": existing_ts},
        }
        await context_no_disk.add_message(message)

        messages = await context_no_disk.get_messages()
        assert len(messages) == 1
        stored = messages[0]
        assert stored["metadata"]["timestamp"] == existing_ts

    @pytest.mark.asyncio
    async def test_existing_metadata_preserved(self, context_no_disk):
        """add_message preserves existing metadata fields and adds timestamp."""
        message = {
            "role": "system",
            "content": "You are a helpful assistant.",
            "metadata": {"source": "hook", "custom": "value"},
        }
        await context_no_disk.add_message(message)

        messages = await context_no_disk.get_messages()
        assert len(messages) == 1
        stored = messages[0]
        assert stored["metadata"]["source"] == "hook"
        assert stored["metadata"]["custom"] == "value"
        assert "timestamp" in stored["metadata"]

    @pytest.mark.asyncio
    async def test_add_message_does_not_mutate_caller_dict(self, context_no_disk):
        """add_message does not mutate the caller's original dict."""
        original = {"role": "user", "content": "hello"}
        await context_no_disk.add_message(original)

        # Original dict must not have 'metadata' key added
        assert "metadata" not in original
