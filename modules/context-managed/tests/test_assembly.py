"""
Assembly tests for system prompt factory integration in ManagedContextManager.

Verifies position, per-request refresh, stored system message exclusion,
hook message preservation, cache hints, ordering, and empty states.
"""

import pytest
from unittest.mock import AsyncMock


class TestSystemPromptFactory:
    """Verify system prompt factory integration with message assembly."""

    @pytest.mark.asyncio
    async def test_factory_system_prompt_at_position_zero(self, context_no_disk):
        """Factory system prompt is placed at messages[0]."""
        factory = AsyncMock(return_value="You are a helpful assistant.")
        await context_no_disk.set_system_prompt_factory(factory)
        await context_no_disk.add_message({"role": "user", "content": "Hello"})

        messages = await context_no_disk.get_messages_for_request()

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_factory_called_every_request(self, context_no_disk):
        """Factory is called on every get_messages_for_request invocation."""
        counter = 0

        async def counting_factory() -> str:
            nonlocal counter
            counter += 1
            return f"System prompt version {counter}"

        await context_no_disk.set_system_prompt_factory(counting_factory)
        await context_no_disk.add_message({"role": "user", "content": "Hello"})

        result1 = await context_no_disk.get_messages_for_request()
        result2 = await context_no_disk.get_messages_for_request()

        assert counter == 2
        assert result1[0]["content"] == "System prompt version 1"
        assert result2[0]["content"] == "System prompt version 2"

    @pytest.mark.asyncio
    async def test_factory_excludes_stored_system_messages(self, context_no_disk):
        """When factory is set, stored system messages are excluded from assembly."""
        factory = AsyncMock(return_value="Factory system prompt")
        await context_no_disk.set_system_prompt_factory(factory)
        await context_no_disk.add_message(
            {"role": "system", "content": "Old system prompt"}
        )
        await context_no_disk.add_message({"role": "user", "content": "Hello"})

        messages = await context_no_disk.get_messages_for_request()

        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 1
        assert system_messages[0]["content"] == "Factory system prompt"

    @pytest.mark.asyncio
    async def test_hook_injected_system_messages_preserved(self, context_no_disk):
        """System messages with metadata.source='hook' are preserved when factory is set."""
        factory = AsyncMock(return_value="Factory system prompt")
        await context_no_disk.set_system_prompt_factory(factory)
        await context_no_disk.add_message(
            {
                "role": "system",
                "content": "Hook-injected context",
                "metadata": {"source": "hook"},
            }
        )
        await context_no_disk.add_message({"role": "user", "content": "Hello"})

        messages = await context_no_disk.get_messages_for_request()

        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 2
        system_contents = [m["content"] for m in system_messages]
        assert "Factory system prompt" in system_contents
        assert "Hook-injected context" in system_contents

    @pytest.mark.asyncio
    async def test_no_factory_uses_stored_system(self, context_no_disk):
        """Without factory, the stored system message is used at position 0."""
        await context_no_disk.add_message(
            {"role": "system", "content": "Stored system prompt"}
        )
        await context_no_disk.add_message({"role": "user", "content": "Hello"})

        messages = await context_no_disk.get_messages_for_request()

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Stored system prompt"

    @pytest.mark.asyncio
    async def test_system_prompt_gets_cache_hint(self, context_no_disk):
        """Factory-generated system prompt has metadata.cache_hint == 'breakpoint'."""
        factory = AsyncMock(return_value="You are a helpful assistant.")
        await context_no_disk.set_system_prompt_factory(factory)
        await context_no_disk.add_message({"role": "user", "content": "Hello"})

        messages = await context_no_disk.get_messages_for_request()

        assert messages[0]["role"] == "system"
        assert messages[0].get("metadata", {}).get("cache_hint") == "breakpoint"

    @pytest.mark.asyncio
    async def test_assembly_order(self, context_no_disk):
        """Messages are assembled in order: system, user Q1, assistant A1, user Q2."""
        factory = AsyncMock(return_value="System")
        await context_no_disk.set_system_prompt_factory(factory)
        await context_no_disk.add_message({"role": "user", "content": "Q1"})
        await context_no_disk.add_message({"role": "assistant", "content": "A1"})
        await context_no_disk.add_message({"role": "user", "content": "Q2"})

        messages = await context_no_disk.get_messages_for_request()

        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Q1"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "A1"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Q2"

    @pytest.mark.asyncio
    async def test_empty_conversation(self, context_no_disk):
        """With no messages and no factory, get_messages_for_request returns []."""
        messages = await context_no_disk.get_messages_for_request()

        assert messages == []

    @pytest.mark.asyncio
    async def test_empty_conversation_with_factory(self, context_no_disk):
        """With factory but no conversation messages, returns just [system]."""
        factory = AsyncMock(return_value="You are a helpful assistant.")
        await context_no_disk.set_system_prompt_factory(factory)

        messages = await context_no_disk.get_messages_for_request()

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
