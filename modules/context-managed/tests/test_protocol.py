"""
Protocol compliance tests for ManagedContextManager.

Verifies all ContextManager protocol methods exist, are callable,
and basic round-trip operations work correctly.
"""

import pytest


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
