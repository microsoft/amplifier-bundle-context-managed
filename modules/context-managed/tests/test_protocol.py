"""
Protocol compliance tests for BoundaryContextManager.

Verifies all ContextManager protocol methods exist, are callable,
and basic round-trip operations work correctly.
"""

import copy
import inspect
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ContextManager


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


class TestCanonicalHistory:
    """The host owns every original field, including timestamps and metadata."""

    @pytest.mark.asyncio
    async def test_no_timestamp_or_metadata_injected(self, context_no_disk):
        message = {"role": "user", "content": "hello"}
        await context_no_disk.add_message(message)
        assert await context_no_disk.get_messages() == [message]

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
        """add_message preserves all existing metadata without adding fields."""
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
        assert stored == message

    @pytest.mark.asyncio
    async def test_add_message_does_not_mutate_caller_dict(self, context_no_disk):
        """add_message does not mutate the caller's original dict."""
        original = {"role": "user", "content": "hello"}
        await context_no_disk.add_message(original)

        # Original dict must not have 'metadata' key added
        assert "metadata" not in original

    @pytest.mark.asyncio
    async def test_nested_history_is_detached_on_add_set_and_get(self, context_no_disk):
        original = {"role": "user", "content": [{"type": "text", "text": "Original"}],
                    "metadata": {"host": {"flags": [True], "unknown": None}},
                    "unknown_field": {"preserve": [1, 2]}}
        expected = copy.deepcopy(original)
        await context_no_disk.add_message(original)
        original["metadata"]["host"]["flags"].append(False)
        original["content"][0]["text"] = "Changed"
        assert await context_no_disk.get_messages() == [expected]
        rows = await context_no_disk.get_messages()
        rows[0]["unknown_field"]["preserve"].clear()
        assert await context_no_disk.get_messages() == [expected]
        await context_no_disk.set_messages([expected])
        expected["metadata"]["host"]["flags"].clear()
        assert (await context_no_disk.get_messages())[0]["metadata"]["host"]["flags"] == [True]


def test_context_manager_protocol_signatures(context_no_disk):
    # Core's protocol is not runtime_checkable. Check its callable contract
    # directly without mutating the shared protocol or weakening it to a stub.
    for name, declaration in vars(ContextManager).items():
        if name.startswith("_") or not inspect.iscoroutinefunction(declaration):
            continue
        method = getattr(context_no_disk, name)
        assert inspect.iscoroutinefunction(method)
        expected = list(inspect.signature(declaration).parameters.values())[1:]
        actual = inspect.signature(method)
        assert list(actual.parameters) == [parameter.name for parameter in expected]
        for parameter in expected:
            assert actual.parameters[parameter.name].kind == parameter.kind
            assert actual.parameters[parameter.name].default == parameter.default


class TestSystemPromptFactory:
    """Keep the applicable factory assembly guards from the retired executor."""

    @pytest.mark.asyncio
    async def test_factory_system_prompt_at_position_zero(self, context_no_disk):
        factory = AsyncMock(return_value="You are a helpful assistant.")
        await context_no_disk.set_system_prompt_factory(factory)
        await context_no_disk.add_message({"role": "user", "content": "Hello"})
        rows = await context_no_disk.get_messages_for_request()
        assert rows[0]["role"] == "system"
        assert rows[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_factory_called_every_request(self, context_no_disk):
        factory = AsyncMock(side_effect=["System prompt version 1", "System prompt version 2"])
        await context_no_disk.set_system_prompt_factory(factory)
        await context_no_disk.add_message({"role": "user", "content": "Hello"})
        first = await context_no_disk.get_messages_for_request()
        second = await context_no_disk.get_messages_for_request()
        assert factory.await_count == 2
        assert first[0]["content"] == "System prompt version 1"
        assert second[0]["content"] == "System prompt version 2"

    @pytest.mark.asyncio
    async def test_factory_excludes_stored_system_messages(self, context_no_disk):
        await context_no_disk.set_system_prompt_factory(AsyncMock(return_value="Factory system prompt"))
        original = [{"role": "system", "content": "Old system prompt"},
                    {"role": "user", "content": "Hello"}]
        await context_no_disk.set_messages(original)
        rows = await context_no_disk.get_messages_for_request()
        assert [row["content"] for row in rows if row["role"] == "system"] == ["Factory system prompt"]
        assert await context_no_disk.get_messages() == original

    @pytest.mark.asyncio
    async def test_hook_injected_system_messages_preserved(self, context_no_disk):
        await context_no_disk.set_system_prompt_factory(AsyncMock(return_value="Factory system prompt"))
        original = [{"role": "system", "content": "Hook-injected context", "metadata": {"source": "hook", "host": [1]}},
                    {"role": "user", "content": "Hello"}]
        await context_no_disk.set_messages(original)
        rows = await context_no_disk.get_messages_for_request()
        assert [row["content"] for row in rows if row["role"] == "system"] == [
            "Factory system prompt", "Hook-injected context"]
        assert await context_no_disk.get_messages() == original

    @pytest.mark.asyncio
    async def test_no_factory_uses_stored_system(self, context_no_disk):
        original = [{"role": "system", "content": "Stored system prompt"},
                    {"role": "user", "content": "Hello"}]
        await context_no_disk.set_messages(original)
        assert (await context_no_disk.get_messages_for_request())[0]["content"] == "Stored system prompt"
        assert await context_no_disk.get_messages() == original

    @pytest.mark.asyncio
    async def test_assembly_order(self, context_no_disk):
        await context_no_disk.set_system_prompt_factory(AsyncMock(return_value="System"))
        original = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "Q2"}]
        await context_no_disk.set_messages(original)
        rows = await context_no_disk.get_messages_for_request()
        assert [(row["role"], row["content"]) for row in rows] == [
            ("system", "System"), ("user", "Q1"), ("assistant", "A1"), ("user", "Q2")]
        assert await context_no_disk.get_messages() == original

    @pytest.mark.asyncio
    async def test_empty_conversation(self, context_no_disk):
        assert await context_no_disk.get_messages_for_request() == []

    @pytest.mark.asyncio
    async def test_empty_conversation_with_factory(self, context_no_disk):
        await context_no_disk.set_system_prompt_factory(AsyncMock(return_value="System"))
        rows = await context_no_disk.get_messages_for_request()
        assert len(rows) == 1 and rows[0]["role"] == "system" and rows[0]["content"] == "System"
        assert await context_no_disk.get_messages() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("image", [
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x" * 8500000}},
    {"type": "image", "source": {"type": "url", "url": "https://example.test/" + "x" * 10000}},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "x" * 10000}},
    {"type": "input_image", "image_url": "data:image/png;base64," + "x" * 10000},
])
async def test_typed_images_preserved_through_provider_measured_boundary(context_no_disk, image):
    original = [{"role": "user", "content": [{"type": "text", "text": "Describe it"}, image],
                 "metadata": {"host": "original"}}]
    before = copy.deepcopy(original)
    await context_no_disk.set_messages(original)

    async def count_view(rows):
        assert rows[0]["content"] == before[0]["content"]
        return {"dispatch": object(), "budget_decision": {
            "estimated_input_tokens": 200, "input_limit_tokens": 1000,
            "measurement": {"kind": "provider_count", "input_tokens": 200, "source": "fixture"}}}

    result = await context_no_disk.get_measured_request_view(
        provider=None, retain_contents=[], count_view=count_view)
    assert result["base_view"][0]["content"] == before[0]["content"]
    assert result["outcome"] == "unchanged"
    assert await context_no_disk.get_messages() == before
    assert original == before


@pytest.mark.asyncio
async def test_large_tool_results_preserved_without_pointer_filing(context_no_disk):
    original = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "large", "name": "fixture", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "large", "content": "Exact result " * 20000,
         "metadata": {"host": {"receipt": "r-1"}}}]
    await context_no_disk.set_messages(original)
    assert (await context_no_disk.get_messages_for_request(token_budget=200000))[-1]["content"] == original[-1]["content"]
    assert await context_no_disk.get_messages() == original


@pytest.mark.asyncio
async def test_large_typed_image_transport_does_not_trigger_text_compaction(context_no_disk):
    small = {"role": "user", "content": [
        {"type": "text", "text": "CURRENT"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}}]}
    large = copy.deepcopy(small)
    large["content"][1]["source"]["data"] = "x" * 8500000
    fitter = context_no_disk._fitter()
    assert fitter._estimate_tokens([large]) == fitter._estimate_tokens([small]) < 100
    await context_no_disk.set_messages([large])
    rows = await context_no_disk.get_messages_for_request(token_budget=1000)
    assert rows[0]["content"] == large["content"]
    assert context_no_disk.summary is None
    assert await context_no_disk.get_messages() == [large]


@pytest.mark.parametrize("content", [
    "data:image/png;base64," + "x" * 10000,
    [{"type": "text", "text": '{"type":"image","source":{"data":"' + "x" * 10000 + '"}}'}],
    [{"type": "tool_call", "name": "fixture", "arguments": {"type": "image", "data": "x" * 10000}}],
])
def test_quoted_images_and_tool_arguments_still_count_as_text(context_no_disk, content):
    original = {"role": "user", "content": content}
    before = copy.deepcopy(original)
    assert context_no_disk._fitter()._estimate_tokens([original]) > 1000
    assert original == before


@pytest.mark.parametrize("image", [
    {"type": "image", "source": {"type": "url", "url": "https://example.test/" + "x" * 10000}},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "x" * 10000}},
    {"type": "input_image", "image_url": "data:image/png;base64," + "x" * 10000},
])
def test_nested_tool_screenshot_transport_is_not_text(context_no_disk, image):
    original = {"role": "tool", "content": [{"type": "tool_result", "content": [image, image]}]}
    before = copy.deepcopy(original)
    assert context_no_disk._fitter()._estimate_tokens([original]) < 100
    assert original == before
