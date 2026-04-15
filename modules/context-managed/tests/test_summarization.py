"""
Tests for Phase 2 summarization dataclasses: SummaryResult and SummaryTier.

Task 1: Add SummaryResult and SummaryTier dataclasses.
Task 2: Default summarization prompt and _get_summarization_prompt().
Task 4: Trigger summarization with guard and threshold wiring.
Task 12: Tier reconstruction on resume.
Task 13: Clear handles active summarization, no-provider guard verified.
"""

import asyncio
import json

import pytest


class TestSummaryResultDataclass:
    """Tests for the SummaryResult dataclass."""

    def test_construct_with_required_fields(self):
        """SummaryResult can be constructed with all required fields."""
        from amplifier_module_context_managed import SummaryResult

        result = SummaryResult(
            summary_text="This is a summary.",
            turn_range=(1, 5),
            source_message_range=(0, 10),
        )
        assert result.summary_text == "This is a summary."
        assert result.turn_range == (1, 5)
        assert result.source_message_range == (0, 10)
        assert result.compression_passes == 1  # default value

    def test_custom_compression_passes(self):
        """SummaryResult accepts a custom compression_passes value."""
        from amplifier_module_context_managed import SummaryResult

        result = SummaryResult(
            summary_text="Compressed summary.",
            turn_range=(2, 8),
            source_message_range=(5, 20),
            compression_passes=3,
        )
        assert result.compression_passes == 3


class TestSummaryTierDataclass:
    """Tests for the SummaryTier dataclass."""

    def test_construct_with_all_fields(self):
        """SummaryTier can be constructed with all required fields."""
        from amplifier_module_context_managed import SummaryTier

        tier = SummaryTier(
            content="Tier content here.",
            turn_range=(0, 10),
            source_message_range=(0, 25),
            compression_passes=2,
            token_estimate=500,
        )
        assert tier.content == "Tier content here."
        assert tier.turn_range == (0, 10)
        assert tier.source_message_range == (0, 25)
        assert tier.compression_passes == 2
        assert tier.token_estimate == 500


class TestDefaultSummarizationPrompt:
    """Tests for DEFAULT_SUMMARIZATION_PROMPT and _get_summarization_prompt()."""

    def test_prompt_is_nonempty_string(self):
        """DEFAULT_SUMMARIZATION_PROMPT is a non-empty string."""
        from amplifier_module_context_managed import DEFAULT_SUMMARIZATION_PROMPT

        assert isinstance(DEFAULT_SUMMARIZATION_PROMPT, str)
        assert len(DEFAULT_SUMMARIZATION_PROMPT) > 0

    def test_prompt_mentions_key_sections(self):
        """DEFAULT_SUMMARIZATION_PROMPT mentions all required sections."""
        from amplifier_module_context_managed import DEFAULT_SUMMARIZATION_PROMPT

        assert "User Requests" in DEFAULT_SUMMARIZATION_PROMPT
        assert "Files Examined" in DEFAULT_SUMMARIZATION_PROMPT
        assert "Errors Encountered" in DEFAULT_SUMMARIZATION_PROMPT
        assert "Current Task State" in DEFAULT_SUMMARIZATION_PROMPT
        assert "Key Technical Details" in DEFAULT_SUMMARIZATION_PROMPT

    def test_get_summarization_prompt_returns_default(self):
        """_get_summarization_prompt() returns default when no path is configured."""
        from amplifier_module_context_managed import (
            DEFAULT_SUMMARIZATION_PROMPT,
            ManagedContextManager,
        )

        mgr = ManagedContextManager()
        assert mgr._get_summarization_prompt() == DEFAULT_SUMMARIZATION_PROMPT

    def test_get_summarization_prompt_loads_from_file(self, tmp_path):
        """_get_summarization_prompt() loads prompt from file when path is configured."""
        from amplifier_module_context_managed import ManagedContextManager

        custom_prompt = "Custom summarization prompt content."
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text(custom_prompt)

        mgr = ManagedContextManager(summarization_prompt_path=str(prompt_file))
        assert mgr._get_summarization_prompt() == custom_prompt

    def test_get_summarization_prompt_falls_back_on_missing_file(self, tmp_path):
        """_get_summarization_prompt() falls back to default when file is missing."""
        from amplifier_module_context_managed import (
            DEFAULT_SUMMARIZATION_PROMPT,
            ManagedContextManager,
        )

        missing_path = tmp_path / "nonexistent_prompt.txt"
        mgr = ManagedContextManager(summarization_prompt_path=str(missing_path))
        assert mgr._get_summarization_prompt() == DEFAULT_SUMMARIZATION_PROMPT


class TestSegmentBoundary:
    """Tests for _calculate_segment_boundary()."""

    def test_no_boundary_when_empty(self):
        """Returns None when _messages is empty."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(verbatim_window_tokens=1000)
        assert mgr._calculate_segment_boundary() is None

    def test_no_boundary_when_under_verbatim_limit(self):
        """Returns None when _running_token_estimate <= verbatim_window_tokens."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(verbatim_window_tokens=1000)
        mgr._messages = [{"role": "user", "content": "hello"}]
        mgr._running_token_estimate = 500  # under limit
        assert mgr._calculate_segment_boundary() is None

    def test_boundary_returns_range_when_over_limit(self):
        """Returns (start, end) tuple when running estimate exceeds verbatim window."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(verbatim_window_tokens=10)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "more content"},
        ]
        mgr._messages = messages
        mgr._running_token_estimate = 200  # much higher than verbatim_window_tokens=10

        result = mgr._calculate_segment_boundary()

        assert result is not None
        start, end = result
        assert start == 0
        assert 0 < end <= len(messages)

    def test_boundary_covers_excess_tokens(self):
        """Accumulated tokens at boundary are >= excess_tokens."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(verbatim_window_tokens=20)
        messages = [
            {"role": "user", "content": "first message content here"},
            {"role": "assistant", "content": "second message response here"},
            {"role": "user", "content": "third message goes here"},
        ]

        # Compute actual token estimates the same way the method does
        tok1 = len(str(messages[0])) // 4
        tok2 = len(str(messages[1])) // 4
        tok3 = len(str(messages[2])) // 4
        total = tok1 + tok2 + tok3

        mgr._messages = messages
        mgr._running_token_estimate = total

        # Verify precondition: total > verbatim_window_tokens
        assert total > 20, (
            f"Test setup: total={total} must exceed verbatim_window_tokens=20"
        )

        excess = total - 20
        result = mgr._calculate_segment_boundary()

        assert result is not None
        start, end = result
        assert start == 0

        # Verify accumulated tokens up to boundary >= excess_tokens
        accumulated = sum(len(str(messages[i])) // 4 for i in range(end))
        assert accumulated >= excess


class TestToolPairSnapping:
    """Tests for _snap_to_tool_pair_boundary()."""

    def test_extends_past_assistant_with_tool_calls(self):
        """When last included message is assistant with tool_calls, extends past tool results."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "test"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result2"},
            {"role": "user", "content": "next question"},
        ]
        mgr._messages = messages

        # end_idx=1: last included is messages[0] (assistant with tool_calls)
        result = mgr._snap_to_tool_pair_boundary(1)

        # Should extend past both tool results (index 1 and 2), stopping at index 3
        assert result == 3

    def test_extends_past_orphaned_tool_result(self):
        """When first excluded message is a tool result, extends past consecutive tool results."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "thinking about it"},
            {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result2"},
            {"role": "user", "content": "follow up"},
        ]
        mgr._messages = messages

        # end_idx=2: first excluded is messages[2] (a tool result)
        result = mgr._snap_to_tool_pair_boundary(2)

        # Should extend past both tool results (index 2 and 3), stopping at index 4
        assert result == 4

    def test_no_snap_needed_when_boundary_clean(self):
        """When boundary is between user/assistant messages, end_idx is unchanged."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        messages = [
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "question 2"},
            {"role": "assistant", "content": "answer 2"},
        ]
        mgr._messages = messages

        # end_idx=2: boundary between two clean turns (no tool calls or results)
        result = mgr._snap_to_tool_pair_boundary(2)

        assert result == 2  # Unchanged


class TestTriggerSummarization:
    """Tests for _trigger_summarization() guards and behavior."""

    @pytest.mark.asyncio
    async def test_trigger_sets_is_summarizing_flag(self):
        """_trigger_summarization() sets _is_summarizing=True and creates a task."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(verbatim_window_tokens=10)
        mgr._cached_provider = object()  # non-None provider
        mgr._messages = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "response here"},
        ]
        mgr._running_token_estimate = 1000  # exceeds verbatim_window_tokens=10

        await mgr._trigger_summarization()

        # Immediately after call: flag should be True and task should exist
        assert mgr._is_summarizing is True
        assert mgr._summarization_task is not None

        task = mgr._summarization_task
        await task  # Task catches NotImplementedError internally, completes normally

        # After task runs: flag reset, failure counted (NotImplementedError)
        assert mgr._is_summarizing is False
        assert mgr._summarization_task is None
        assert mgr._summarization_failures == 1

    @pytest.mark.asyncio
    async def test_trigger_skips_when_already_summarizing(self):
        """_trigger_summarization() is a no-op when _is_summarizing is True."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(verbatim_window_tokens=10)
        mgr._cached_provider = object()
        mgr._messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "response"},
        ]
        mgr._running_token_estimate = 1000
        mgr._is_summarizing = True  # Already summarizing

        await mgr._trigger_summarization()

        # Guard fires: no task created, no state change
        assert mgr._summarization_task is None
        assert mgr._summarization_failures == 0

    @pytest.mark.asyncio
    async def test_trigger_skips_when_no_provider(self):
        """_trigger_summarization() is a no-op when _cached_provider is None."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(verbatim_window_tokens=10)
        mgr._cached_provider = None  # No provider available
        mgr._messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "response"},
        ]
        mgr._running_token_estimate = 1000

        await mgr._trigger_summarization()

        assert mgr._is_summarizing is False
        assert mgr._summarization_task is None

    @pytest.mark.asyncio
    async def test_trigger_skips_when_no_boundary(self):
        """_trigger_summarization() is a no-op when _calculate_segment_boundary() is None."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(verbatim_window_tokens=10_000)
        mgr._cached_provider = object()
        # Running estimate below verbatim_window_tokens → boundary returns None
        mgr._messages = [{"role": "user", "content": "hi"}]
        mgr._running_token_estimate = 5

        await mgr._trigger_summarization()

        assert mgr._is_summarizing is False
        assert mgr._summarization_task is None


class TestThresholdWiring:
    """Tests for threshold-based trigger wiring in add_message()."""

    @pytest.mark.asyncio
    async def test_add_message_increments_turn_on_user_message(self):
        """add_message() increments _current_turn only for user messages."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        assert mgr._current_turn == 0

        await mgr.add_message({"role": "user", "content": "hello"})
        assert mgr._current_turn == 1

        await mgr.add_message({"role": "assistant", "content": "world"})
        assert mgr._current_turn == 1  # Not incremented for non-user messages

        await mgr.add_message({"role": "user", "content": "follow up"})
        assert mgr._current_turn == 2

    @pytest.mark.asyncio
    async def test_add_message_triggers_at_080_threshold(self):
        """add_message() triggers summarization when usage fraction >= summarize_trigger."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(
            max_tokens=1000,
            summarize_trigger=0.80,
            verbatim_window_tokens=10,
        )
        mgr._cached_provider = object()  # non-None so trigger guard passes
        # Pre-seed messages so _calculate_segment_boundary() returns non-None
        mgr._messages = [
            {"role": "user", "content": "earlier message"},
            {"role": "assistant", "content": "earlier response"},
        ]
        # 90% of 1000 → well above 0.80 trigger
        mgr._running_token_estimate = 900

        await mgr.add_message({"role": "user", "content": "new message"})

        # Give the event loop a tick to let the background task run
        await asyncio.sleep(0)

        # Summarization was triggered: task ran, hit NotImplementedError, failure counted
        assert mgr._summarization_failures == 1

    @pytest.mark.asyncio
    async def test_add_message_does_not_trigger_below_threshold(self):
        """add_message() does not trigger summarization when usage fraction < summarize_trigger."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(
            max_tokens=1000,
            summarize_trigger=0.80,
        )
        mgr._cached_provider = object()
        # 10% of 1000 → well below 0.80 trigger
        mgr._running_token_estimate = 100

        await mgr.add_message({"role": "user", "content": "short message"})

        await asyncio.sleep(0)

        assert mgr._summarization_failures == 0
        assert mgr._is_summarizing is False
        assert mgr._summarization_task is None


class TestPerformSummarization:
    """Tests for _perform_summarization() with real provider call."""

    @pytest.mark.asyncio
    async def test_returns_summary_result(self):
        """_perform_summarization() returns a SummaryResult."""
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryResult,
        )

        mgr = ManagedContextManager()
        mgr._messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

        mock_block = MagicMock()
        mock_block.text = "Summary text"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        mgr._cached_provider = mock_provider

        result = await mgr._perform_summarization((0, 2))

        assert isinstance(result, SummaryResult)
        assert result.summary_text == "Summary text"

    @pytest.mark.asyncio
    async def test_calls_provider_complete(self):
        """_perform_summarization() calls self._cached_provider.complete() with a ChatRequest."""
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_core import ChatRequest

        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        mgr._messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

        mock_block = MagicMock()
        mock_block.text = "Summary"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        mgr._cached_provider = mock_provider

        await mgr._perform_summarization((0, 2))

        mock_provider.complete.assert_called_once()
        # Verify the argument is a ChatRequest
        call_args = mock_provider.complete.call_args
        assert isinstance(call_args[0][0], ChatRequest)

    @pytest.mark.asyncio
    async def test_uses_absolute_source_message_range(self):
        """_perform_summarization() uses _transcript_message_offset for source_message_range."""
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        mgr._transcript_message_offset = 10  # Previous messages already summarized
        mgr._messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
            {"role": "user", "content": "More"},
        ]

        mock_block = MagicMock()
        mock_block.text = "Summary"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        mgr._cached_provider = mock_provider

        result = await mgr._perform_summarization((0, 2))

        # source_message_range should be absolute (offset + boundary)
        assert result.source_message_range == (10, 12)  # offset=10, start=0, end=2

    @pytest.mark.asyncio
    async def test_calculates_turn_range(self):
        """_perform_summarization() calculates turn_range using _summarized_through_turn."""
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        mgr._summarized_through_turn = 5  # Already summarized through turn 5
        mgr._messages = [
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Resp 2"},
        ]

        mock_block = MagicMock()
        mock_block.text = "Summary"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        mgr._cached_provider = mock_provider

        result = await mgr._perform_summarization((0, 4))

        # turn_start = _summarized_through_turn + 1 = 6
        # turn_end = _summarized_through_turn + user_count_in_segment = 5 + 2 = 7
        assert result.turn_range == (6, 7)


class TestFormatMessagesForSummarization:
    """Tests for _format_messages_for_summarization()."""

    def test_formats_string_content(self):
        """_format_messages_for_summarization() formats messages with string content as '[role]: content'."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

        result = mgr._format_messages_for_summarization(messages)

        assert "[user]: Hello" in result
        assert "[assistant]: World" in result

    def test_formats_list_content_blocks(self):
        """_format_messages_for_summarization() handles list content by joining text blocks."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()

        class _TextBlock:
            def __init__(self, text):
                self.text = text

        messages = [
            {
                "role": "user",
                "content": [_TextBlock("First part"), _TextBlock(" second part")],
            },
        ]

        result = mgr._format_messages_for_summarization(messages)

        assert "[user]:" in result
        assert "First part" in result
        assert "second part" in result


class TestExtractTextFromResponse:
    """Tests for _extract_text_from_response()."""

    def test_extracts_text_from_content_blocks(self):
        """_extract_text_from_response() extracts and joins text from blocks with .text attribute."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()

        class _TextBlock:
            def __init__(self, text):
                self.text = text

        class _MockResponse:
            def __init__(self, blocks):
                self.content = blocks

        response = _MockResponse([_TextBlock("Hello "), _TextBlock("World")])

        result = mgr._extract_text_from_response(response)

        assert result == "Hello World"

    def test_skips_non_text_blocks(self):
        """_extract_text_from_response() skips blocks that lack a .text attribute."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()

        class _TextBlock:
            def __init__(self, text):
                self.text = text

        class _NonTextBlock:
            pass  # No .text attribute

        class _MockResponse:
            def __init__(self, blocks):
                self.content = blocks

        response = _MockResponse(
            [_TextBlock("Hello"), _NonTextBlock(), _TextBlock(" World")]
        )

        result = mgr._extract_text_from_response(response)

        assert result == "Hello World"


class TestPendingSummarySwap:
    """Tests for Phase 2 pending summary swap in get_messages_for_request()."""

    @pytest.mark.asyncio
    async def test_swap_removes_verbatim_creates_tier(self):
        """get_messages_for_request() removes summarized messages and creates a SummaryTier."""
        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryResult,
        )

        mgr = ManagedContextManager()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
            {"role": "user", "content": "More"},
            {"role": "assistant", "content": "Content"},
        ]
        mgr._messages = list(messages)
        mgr._running_token_estimate = mgr._estimate_tokens(messages)
        mgr._transcript_message_offset = 0

        # Pending summary covers messages 0-2 (local = absolute since offset=0)
        mgr._pending_summary = SummaryResult(
            summary_text="Summary of first two messages",
            turn_range=(1, 1),
            source_message_range=(0, 2),
            compression_passes=1,
        )

        await mgr.get_messages_for_request()

        # Messages 0 and 1 should be removed, only messages 2 and 3 remain
        assert len(mgr._messages) == 2
        assert mgr._messages[0]["content"] == "More"
        assert mgr._messages[1]["content"] == "Content"

        # A SummaryTier should have been created
        assert len(mgr._summary_tiers) == 1
        assert mgr._summary_tiers[0].content == "Summary of first two messages"
        assert mgr._summary_tiers[0].turn_range == (1, 1)
        assert mgr._summary_tiers[0].source_message_range == (0, 2)
        assert mgr._summary_tiers[0].compression_passes == 1

    @pytest.mark.asyncio
    async def test_swap_updates_tracking_fields(self):
        """get_messages_for_request() updates tracking fields after a successful swap."""
        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryResult,
        )

        mgr = ManagedContextManager()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
            {"role": "user", "content": "More"},
            {"role": "assistant", "content": "Content"},
        ]
        mgr._messages = list(messages)
        initial_estimate = mgr._estimate_tokens(messages)
        mgr._running_token_estimate = initial_estimate
        mgr._transcript_message_offset = 0
        mgr._summarization_failures = 2  # Start with some failures to verify reset

        old_tokens = mgr._estimate_tokens(messages[0:2])
        new_tier_tokens = mgr._estimate_tokens_single(
            {"role": "system", "content": "Summary of first two messages"}
        )

        mgr._pending_summary = SummaryResult(
            summary_text="Summary of first two messages",
            turn_range=(1, 1),
            source_message_range=(0, 2),
            compression_passes=1,
        )

        await mgr.get_messages_for_request()

        # _pending_summary should be cleared
        assert mgr._pending_summary is None
        # _transcript_message_offset should be advanced by the number of removed messages (2)
        assert mgr._transcript_message_offset == 2
        # _summarized_through_turn should be set to turn_range[1]
        assert mgr._summarized_through_turn == 1
        # _summarization_failures should be reset to 0
        assert mgr._summarization_failures == 0
        # _running_token_estimate should be updated (subtract old, add new tier)
        expected_estimate = initial_estimate - old_tokens + new_tier_tokens
        assert mgr._running_token_estimate == expected_estimate

    @pytest.mark.asyncio
    async def test_swap_clears_pending_on_invalid_boundary(self):
        """get_messages_for_request() discards pending summary and clears it on invalid boundary."""
        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryResult,
        )

        mgr = ManagedContextManager()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        mgr._messages = list(messages)
        initial_estimate = mgr._estimate_tokens(messages)
        mgr._running_token_estimate = initial_estimate
        mgr._transcript_message_offset = 0

        # Invalid: end_local = 10 - 0 = 10, but len(messages) = 2
        mgr._pending_summary = SummaryResult(
            summary_text="Summary",
            turn_range=(1, 1),
            source_message_range=(0, 10),
            compression_passes=1,
        )

        await mgr.get_messages_for_request()

        # _pending_summary should be cleared even on invalid boundary
        assert mgr._pending_summary is None
        # No tier should be created
        assert len(mgr._summary_tiers) == 0
        # Messages should be unchanged
        assert len(mgr._messages) == 2
        # Token estimate should be unchanged
        assert mgr._running_token_estimate == initial_estimate

    @pytest.mark.asyncio
    async def test_no_swap_when_no_pending(self):
        """get_messages_for_request() does nothing when _pending_summary is None."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        mgr._messages = list(messages)
        initial_estimate = mgr._estimate_tokens(messages)
        mgr._running_token_estimate = initial_estimate
        mgr._pending_summary = None

        await mgr.get_messages_for_request()

        # Messages should be unchanged
        assert len(mgr._messages) == 2
        # No tier should be created
        assert len(mgr._summary_tiers) == 0
        # Estimate should be unchanged
        assert mgr._running_token_estimate == initial_estimate


class TestTiersInAssembly:
    """Tests for summary tier insertion in get_messages_for_request() (task-7)."""

    @pytest.mark.asyncio
    async def test_tiers_inserted_between_system_and_verbatim(self):
        """Summary tiers appear between system message and verbatim conversation messages."""
        from amplifier_module_context_managed import ManagedContextManager, SummaryTier

        mgr = ManagedContextManager()
        # Set up a stored system message at index 0
        mgr._messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        # Add two summary tiers
        tier1 = SummaryTier(
            content="Tier 1 summary",
            turn_range=(1, 2),
            source_message_range=(0, 4),
            compression_passes=1,
            token_estimate=100,
        )
        tier2 = SummaryTier(
            content="Tier 2 summary",
            turn_range=(3, 4),
            source_message_range=(4, 8),
            compression_passes=1,
            token_estimate=150,
        )
        mgr._summary_tiers = [tier1, tier2]
        mgr._running_token_estimate = mgr._estimate_tokens(mgr._messages)

        result = await mgr.get_messages_for_request()

        # Expected order: system msg, tier1, tier2, user msg, assistant msg
        assert len(result) == 5
        # First message is the system prompt
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System prompt"
        # Next two are tiers
        assert result[1]["role"] == "system"
        assert result[1]["content"] == "Tier 1 summary"
        assert result[2]["role"] == "system"
        assert result[2]["content"] == "Tier 2 summary"
        # Last two are the verbatim conversation messages
        assert result[3]["role"] == "user"
        assert result[4]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_last_tier_gets_cache_hint(self):
        """Only the LAST summary tier gets cache_hint='breakpoint' in metadata."""
        from amplifier_module_context_managed import ManagedContextManager, SummaryTier

        mgr = ManagedContextManager()
        mgr._messages = [
            {"role": "user", "content": "Hello"},
        ]
        tier1 = SummaryTier(
            content="First tier summary",
            turn_range=(1, 2),
            source_message_range=(0, 4),
            compression_passes=1,
            token_estimate=100,
        )
        tier2 = SummaryTier(
            content="Second tier summary",
            turn_range=(3, 5),
            source_message_range=(4, 10),
            compression_passes=2,
            token_estimate=150,
        )
        mgr._summary_tiers = [tier1, tier2]
        mgr._running_token_estimate = mgr._estimate_tokens(mgr._messages)

        result = await mgr.get_messages_for_request()

        # Find the tier messages (role=system with type=context_managed_summary)
        tier_messages = [
            m
            for m in result
            if m.get("role") == "system"
            and (m.get("metadata") or {}).get("type") == "context_managed_summary"
        ]
        assert len(tier_messages) == 2

        # The FIRST tier should NOT have cache_hint
        first_tier_meta = tier_messages[0].get("metadata") or {}
        assert "cache_hint" not in first_tier_meta, (
            f"First tier should NOT have cache_hint, but got: {first_tier_meta}"
        )

        # The LAST tier should have cache_hint='breakpoint'
        last_tier_meta = tier_messages[1].get("metadata") or {}
        assert last_tier_meta.get("cache_hint") == "breakpoint", (
            f"Last tier should have cache_hint='breakpoint', but got: {last_tier_meta}"
        )

    @pytest.mark.asyncio
    async def test_tier_metadata_has_required_fields(self):
        """Each tier message metadata contains type, turn_range (as list), and compression_passes."""
        from amplifier_module_context_managed import ManagedContextManager, SummaryTier

        mgr = ManagedContextManager()
        mgr._messages = [
            {"role": "user", "content": "Hello"},
        ]
        tier = SummaryTier(
            content="Summary content",
            turn_range=(3, 7),
            source_message_range=(0, 10),
            compression_passes=3,
            token_estimate=200,
        )
        mgr._summary_tiers = [tier]
        mgr._running_token_estimate = mgr._estimate_tokens(mgr._messages)

        result = await mgr.get_messages_for_request()

        # Find the tier message
        tier_messages = [
            m
            for m in result
            if m.get("role") == "system"
            and (m.get("metadata") or {}).get("type") == "context_managed_summary"
        ]
        assert len(tier_messages) == 1

        meta = tier_messages[0].get("metadata") or {}

        # Must have type field
        assert meta.get("type") == "context_managed_summary"

        # Must have turn_range as a list
        assert "turn_range" in meta, f"turn_range missing from metadata: {meta}"
        assert meta["turn_range"] == [3, 7], (
            f"turn_range should be [3, 7] (list), got: {meta['turn_range']}"
        )
        assert isinstance(meta["turn_range"], list), (
            f"turn_range must be a list, got: {type(meta['turn_range'])}"
        )

        # Must have compression_passes
        assert "compression_passes" in meta, (
            f"compression_passes missing from metadata: {meta}"
        )
        assert meta["compression_passes"] == 3

    @pytest.mark.asyncio
    async def test_empty_tiers_no_change(self):
        """When _summary_tiers is empty, no tier messages are inserted."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        mgr._messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        mgr._summary_tiers = []  # No tiers
        mgr._running_token_estimate = mgr._estimate_tokens(mgr._messages)

        result = await mgr.get_messages_for_request()

        # Should only have the two conversation messages
        assert len(result) == 2
        # Verify no tier messages were inserted
        tier_messages = [
            m
            for m in result
            if (m.get("metadata") or {}).get("type") == "context_managed_summary"
        ]
        assert len(tier_messages) == 0


class TestTierMerging:
    """Tests for _merge_oldest_tiers() (task-8)."""

    @pytest.mark.asyncio
    async def test_merge_oldest_two_tiers(self):
        """_merge_oldest_tiers() replaces the first two tiers with a single merged tier."""
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryTier,
        )

        mgr = ManagedContextManager(max_summary_tiers=1)

        tier_a = SummaryTier(
            content="Summary A covering turns 1-3",
            turn_range=(1, 3),
            source_message_range=(0, 6),
            compression_passes=1,
            token_estimate=100,
        )
        tier_b = SummaryTier(
            content="Summary B covering turns 4-6",
            turn_range=(4, 6),
            source_message_range=(6, 12),
            compression_passes=1,
            token_estimate=100,
        )
        tier_c = SummaryTier(
            content="Summary C covering turns 7-9",
            turn_range=(7, 9),
            source_message_range=(12, 18),
            compression_passes=1,
            token_estimate=100,
        )
        mgr._summary_tiers = [tier_a, tier_b, tier_c]

        mock_block = MagicMock()
        mock_block.text = "Merged summary of A and B"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        mgr._cached_provider = mock_provider

        await mgr._merge_oldest_tiers()

        # Should have 2 tiers: merged + tier_c
        assert len(mgr._summary_tiers) == 2

        # First tier is the merged one
        merged = mgr._summary_tiers[0]
        assert merged.content == "Merged summary of A and B"
        # Combined turn_range: tier_a start to tier_b end
        assert merged.turn_range == (1, 6)
        # Combined source_message_range: tier_a start to tier_b end
        assert merged.source_message_range == (0, 12)

        # Third tier unchanged
        assert mgr._summary_tiers[1] is tier_c

    @pytest.mark.asyncio
    async def test_merge_calls_provider(self):
        """_merge_oldest_tiers() calls provider.complete() with a ChatRequest containing a single user message."""
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_core import ChatRequest

        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryTier,
        )

        mgr = ManagedContextManager(max_summary_tiers=1)

        tier_a = SummaryTier(
            content="Summary A",
            turn_range=(1, 3),
            source_message_range=(0, 6),
            compression_passes=1,
            token_estimate=100,
        )
        tier_b = SummaryTier(
            content="Summary B",
            turn_range=(4, 6),
            source_message_range=(6, 12),
            compression_passes=1,
            token_estimate=100,
        )
        mgr._summary_tiers = [tier_a, tier_b]

        mock_block = MagicMock()
        mock_block.text = "Merged"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        mgr._cached_provider = mock_provider

        await mgr._merge_oldest_tiers()

        # provider.complete should be called once
        mock_provider.complete.assert_called_once()
        # The argument should be a ChatRequest
        call_args = mock_provider.complete.call_args
        request = call_args[0][0]
        assert isinstance(request, ChatRequest)
        # Should have exactly one message (single user message)
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_merge_increments_compression_passes(self):
        """_merge_oldest_tiers() sets merged tier compression_passes = max(a, b) + 1."""
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryTier,
        )

        mgr = ManagedContextManager(max_summary_tiers=1)

        # tier_a has 2 passes, tier_b has 3 passes -> max = 3, merged = 4
        tier_a = SummaryTier(
            content="Summary A",
            turn_range=(1, 3),
            source_message_range=(0, 6),
            compression_passes=2,
            token_estimate=100,
        )
        tier_b = SummaryTier(
            content="Summary B",
            turn_range=(4, 6),
            source_message_range=(6, 12),
            compression_passes=3,
            token_estimate=100,
        )
        mgr._summary_tiers = [tier_a, tier_b]

        mock_block = MagicMock()
        mock_block.text = "Merged"
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        mgr._cached_provider = mock_provider

        await mgr._merge_oldest_tiers()

        assert len(mgr._summary_tiers) == 1
        merged = mgr._summary_tiers[0]
        assert merged.compression_passes == 4  # max(2, 3) + 1

    @pytest.mark.asyncio
    async def test_no_merge_when_under_limit(self):
        """_merge_oldest_tiers() returns early without calling provider when count <= max_summary_tiers."""
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryTier,
        )

        # max_summary_tiers=3, tiers=2 -> no merge needed (2 <= 3)
        mgr = ManagedContextManager(max_summary_tiers=3)

        tier_a = SummaryTier(
            content="Summary A",
            turn_range=(1, 3),
            source_message_range=(0, 6),
            compression_passes=1,
            token_estimate=100,
        )
        tier_b = SummaryTier(
            content="Summary B",
            turn_range=(4, 6),
            source_message_range=(6, 12),
            compression_passes=1,
            token_estimate=100,
        )
        mgr._summary_tiers = [tier_a, tier_b]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock()
        mgr._cached_provider = mock_provider

        await mgr._merge_oldest_tiers()

        # Provider should NOT be called - returned early
        mock_provider.complete.assert_not_called()
        # Tiers should be unchanged
        assert len(mgr._summary_tiers) == 2
        assert mgr._summary_tiers[0] is tier_a
        assert mgr._summary_tiers[1] is tier_b


class TestSummarizationEvents:
    """Tests for event emission in _run_summarization() (task-9)."""

    @pytest.mark.asyncio
    async def test_pre_summarize_event_emitted(self):
        """_run_summarization() emits 'context:pre_summarize' with {boundary, message_count} before calling _perform_summarization."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        mgr._messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

        mock_hooks = MagicMock()
        mock_hooks.emit = AsyncMock()
        mgr._hooks = mock_hooks

        # Make _perform_summarization raise to keep test simple
        with patch.object(mgr, "_perform_summarization", side_effect=Exception("fail")):
            await mgr._run_summarization((0, 2))

        # Verify pre_summarize was emitted
        emitted_events = [call.args[0] for call in mock_hooks.emit.call_args_list]
        assert "context:pre_summarize" in emitted_events

        # Verify the data payload for pre_summarize
        pre_call = next(
            call
            for call in mock_hooks.emit.call_args_list
            if call.args[0] == "context:pre_summarize"
        )
        data = pre_call.args[1]
        assert data["boundary"] == (0, 2)
        assert data["message_count"] == 2  # boundary[1] - boundary[0]

    @pytest.mark.asyncio
    async def test_post_summarize_event_emitted_with_stats(self):
        """_run_summarization() emits 'context:post_summarize' with stats after successful summarization."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryResult,
        )

        mgr = ManagedContextManager()
        mgr._messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

        mock_hooks = MagicMock()
        mock_hooks.emit = AsyncMock()
        mgr._hooks = mock_hooks

        mock_result = SummaryResult(
            summary_text="This is a summary.",
            turn_range=(1, 3),
            source_message_range=(0, 2),
            compression_passes=2,
        )

        with patch.object(mgr, "_perform_summarization", return_value=mock_result):
            await mgr._run_summarization((0, 2))

        # Verify post_summarize was emitted
        emitted_events = [call.args[0] for call in mock_hooks.emit.call_args_list]
        assert "context:post_summarize" in emitted_events

        # Verify the data payload for post_summarize
        post_call = next(
            call
            for call in mock_hooks.emit.call_args_list
            if call.args[0] == "context:post_summarize"
        )
        data = post_call.args[1]
        assert data["turn_range"] == [1, 3]  # turn_range as list
        assert data["summary_length"] == len("This is a summary.")
        assert data["compression_passes"] == 2

    @pytest.mark.asyncio
    async def test_no_post_event_on_failure(self):
        """_run_summarization() does NOT emit 'context:post_summarize' when _perform_summarization raises."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()
        mgr._messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

        mock_hooks = MagicMock()
        mock_hooks.emit = AsyncMock()
        mgr._hooks = mock_hooks

        with patch.object(mgr, "_perform_summarization", side_effect=Exception("fail")):
            await mgr._run_summarization((0, 2))

        # Verify post_summarize was NOT emitted
        emitted_events = [call.args[0] for call in mock_hooks.emit.call_args_list]
        assert "context:post_summarize" not in emitted_events
        # Failure counter should still be incremented
        assert mgr._summarization_failures == 1


class TestEmergencyFallback:
    """Tests for _emergency_mechanical_fallback() and its trigger in _check_summarization_trigger()."""

    @pytest.mark.asyncio
    async def test_fallback_fires_when_failures_and_high_usage(self):
        """_check_summarization_trigger() calls _emergency_mechanical_fallback() when
        _summarization_failures >= summarization_retries_before_fallback AND
        usage_fraction >= emergency_fallback.
        """
        from unittest.mock import AsyncMock, patch

        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(
            max_tokens=1000,
            summarize_trigger=0.80,
            emergency_fallback=0.92,
            summarization_retries_before_fallback=3,
        )
        # Set failures at the threshold
        mgr._summarization_failures = 3
        # Usage fraction = 950 / 1000 = 0.95 >= 0.92 (emergency_fallback)
        mgr._running_token_estimate = 950

        with patch.object(
            mgr, "_emergency_mechanical_fallback", new_callable=AsyncMock
        ) as mock_fallback:
            await mgr._check_summarization_trigger()

        mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_reduces_token_estimate(self):
        """_emergency_mechanical_fallback() reduces _running_token_estimate
        by removing non-protected messages.
        """
        from amplifier_module_context_managed import ManagedContextManager

        # max_tokens=1000, emergency_target_usage=0.10 → target_tokens=100
        # Messages are ~173 tokens total, which exceeds 100 → removal triggered
        mgr = ManagedContextManager(
            max_tokens=1000,
            emergency_target_usage=0.10,
        )
        # 3 removable messages (no system/hook messages = all removable)
        mgr._messages = [
            {"role": "user", "content": "A" * 200},
            {"role": "assistant", "content": "B" * 200},
            {"role": "user", "content": "C" * 200},
        ]
        initial_estimate = mgr._estimate_tokens(mgr._messages)
        mgr._running_token_estimate = initial_estimate
        mgr._summarization_failures = 1

        # Verify the test precondition: initial estimate > target
        target_tokens = int(1000 * 0.10)
        assert initial_estimate > target_tokens, (
            f"Test precondition failed: initial_estimate={initial_estimate} "
            f"must exceed target_tokens={target_tokens}"
        )

        await mgr._emergency_mechanical_fallback()

        assert mgr._running_token_estimate < initial_estimate

    @pytest.mark.asyncio
    async def test_fallback_does_not_fire_below_failure_threshold(self):
        """_check_summarization_trigger() does NOT call _emergency_mechanical_fallback()
        when _summarization_failures < summarization_retries_before_fallback.
        """
        from unittest.mock import AsyncMock, patch

        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(
            max_tokens=1000,
            summarize_trigger=0.80,
            emergency_fallback=0.92,
            summarization_retries_before_fallback=3,
        )
        # Only 2 failures — below the threshold of 3
        mgr._summarization_failures = 2
        # Usage fraction = 950 / 1000 = 0.95 >= 0.92 (emergency_fallback) -- high usage
        mgr._running_token_estimate = 950

        with patch.object(
            mgr, "_emergency_mechanical_fallback", new_callable=AsyncMock
        ) as mock_fallback:
            await mgr._check_summarization_trigger()

        mock_fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_preserves_hook_messages(self):
        """_emergency_mechanical_fallback() does not remove messages with metadata.source=='hook'."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(
            max_tokens=1000,
            emergency_target_usage=0.10,  # Very low target — force aggressive removal
        )
        hook_msg = {
            "role": "user",
            "content": "Hook-injected content",
            "metadata": {"source": "hook"},
        }
        system_msg = {"role": "system", "content": "System prompt"}
        removable_msg = {"role": "user", "content": "Regular user message"}

        mgr._messages = [system_msg, hook_msg, removable_msg]
        mgr._running_token_estimate = mgr._estimate_tokens(mgr._messages)
        mgr._summarization_failures = 1

        await mgr._emergency_mechanical_fallback()

        # Hook message and system message must still be present
        remaining_contents = [m["content"] for m in mgr._messages]
        assert "Hook-injected content" in remaining_contents
        assert "System prompt" in remaining_contents
        # Removable message may or may not be present depending on whether we hit target
        # But hook and system messages must NEVER be removed

    @pytest.mark.asyncio
    async def test_fallback_truncates_large_tool_results_first(self):
        """_emergency_mechanical_fallback() truncates tool messages with string content > 1000 chars
        to first 500 chars + '\\n\\n[truncated by emergency compaction]' before removing messages.
        """
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(
            max_tokens=1000,
            emergency_target_usage=0.90,  # High target — only truncation needed
        )
        large_content = "X" * 2000  # Well over 1000 chars
        tool_msg = {
            "role": "tool",
            "content": large_content,
            "tool_call_id": "call_abc",
        }
        mgr._messages = [tool_msg]
        mgr._running_token_estimate = mgr._estimate_tokens(mgr._messages)
        mgr._summarization_failures = 1

        await mgr._emergency_mechanical_fallback()

        # The tool message content should be truncated
        updated_content = mgr._messages[0]["content"]
        assert updated_content == "X" * 500 + "\n\n[truncated by emergency compaction]"


class TestSummaryPersistence:
    """Tests for summary marker persistence to transcript (task-11)."""

    def test_persist_summary_marker_writes_to_transcript(self, tmp_path):
        """_persist_summary_marker() writes a marker dict to the transcript JSONL file."""
        from amplifier_module_context_managed import ManagedContextManager, SummaryTier

        mgr = ManagedContextManager(session_dir=tmp_path)

        tier = SummaryTier(
            content="Summary of first conversation segment",
            turn_range=(1, 5),
            source_message_range=(0, 10),
            compression_passes=2,
            token_estimate=350,
        )

        mgr._persist_summary_marker(tier)

        # Transcript file should now exist
        assert mgr.transcript_path is not None
        assert mgr.transcript_path.exists()

        # Read the transcript and find the summary marker
        records = []
        with open(mgr.transcript_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        # Find the summary marker record (not the header)
        markers = [
            r
            for r in records
            if (r.get("metadata") or {}).get("type") == "context_managed_summary"
        ]
        assert len(markers) == 1, f"Expected 1 summary marker, found: {len(markers)}"

        marker = markers[0]
        # role must be 'system'
        assert marker["role"] == "system"
        # content must be the tier content
        assert marker["content"] == "Summary of first conversation segment"
        # metadata must have all required fields
        meta = marker["metadata"]
        assert meta["type"] == "context_managed_summary"
        assert meta["turn_range"] == [1, 5]  # list, not tuple
        assert isinstance(meta["turn_range"], list)
        assert meta["source_message_range"] == [0, 10]  # list, not tuple
        assert isinstance(meta["source_message_range"], list)
        assert meta["compression_passes"] == 2
        assert meta["token_estimate"] == 350

    @pytest.mark.asyncio
    async def test_get_messages_excludes_persisted_summary_markers(self, tmp_path):
        """get_messages() excludes persisted summary markers from the returned list."""
        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryTier,
        )

        mgr = ManagedContextManager(session_dir=tmp_path)

        # Add some regular conversation messages
        await mgr.add_message({"role": "user", "content": "Hello"})
        await mgr.add_message({"role": "assistant", "content": "World"})

        # Persist a summary marker directly
        tier = SummaryTier(
            content="Summary of the conversation",
            turn_range=(1, 1),
            source_message_range=(0, 2),
            compression_passes=1,
            token_estimate=200,
        )
        mgr._persist_summary_marker(tier)

        # get_messages() reads from transcript — should exclude the summary marker
        messages = await mgr.get_messages()

        # Only the two conversation messages should be returned
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "World"

        # No summary markers should appear in get_messages() output
        for msg in messages:
            meta = msg.get("metadata") or {}
            assert meta.get("type") != "context_managed_summary", (
                f"Summary marker leaked into get_messages() output: {msg}"
            )


# ── Helper for tier reconstruction tests ─────────────────────────────────────


def _write_transcript_with_markers(session_dir, records):
    """Write a transcript.jsonl file with header + the given records (messages and markers)."""
    from amplifier_module_context_managed import TRANSCRIPT_FORMAT_VERSION

    transcript_path = session_dir / "transcript.jsonl"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with open(transcript_path, "w") as f:
        header = {
            "type": "transcript_header",
            "format_version": TRANSCRIPT_FORMAT_VERSION,
            "created_at": "2024-01-01T00:00:00.000+00:00",
        }
        f.write(json.dumps(header) + "\n")
        for record in records:
            f.write(json.dumps(record) + "\n")


class TestTierReconstructionOnResume:
    """Tests for Phase 2 tier reconstruction on session resume (task-12)."""

    @pytest.mark.asyncio
    async def test_resume_reconstructs_tiers(self, tmp_path):
        """_load_from_transcript() reconstructs SummaryTier objects from summary markers."""
        from amplifier_module_context_managed import ManagedContextManager, SummaryTier

        records = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {
                "role": "system",
                "content": "Summary of messages 0-1",
                "metadata": {
                    "type": "context_managed_summary",
                    "turn_range": [1, 1],
                    "source_message_range": [0, 2],
                    "compression_passes": 1,
                    "token_estimate": 100,
                },
            },
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Second response"},
        ]
        _write_transcript_with_markers(tmp_path, records)

        ctx = ManagedContextManager(session_dir=tmp_path)
        await ctx._load_from_transcript()

        # Should have reconstructed 1 SummaryTier
        assert len(ctx._summary_tiers) == 1

        tier = ctx._summary_tiers[0]
        assert isinstance(tier, SummaryTier)
        assert tier.content == "Summary of messages 0-1"
        # turn_range and source_message_range must be tuples (not lists)
        assert tier.turn_range == (1, 1)
        assert isinstance(tier.turn_range, tuple)
        assert tier.source_message_range == (0, 2)
        assert isinstance(tier.source_message_range, tuple)
        assert tier.compression_passes == 1
        assert tier.token_estimate == 100

        assert ctx._loaded_from_transcript is True

    @pytest.mark.asyncio
    async def test_resume_sets_verbatim_window_correctly(self, tmp_path):
        """_load_from_transcript() sets _messages to only the verbatim window after last summary."""
        from amplifier_module_context_managed import ManagedContextManager

        records = [
            {"role": "user", "content": "Msg 0"},
            {"role": "assistant", "content": "Msg 1"},
            {
                "role": "system",
                "content": "Summary covering msgs 0-1",
                "metadata": {
                    "type": "context_managed_summary",
                    "turn_range": [1, 1],
                    "source_message_range": [0, 2],
                    "compression_passes": 1,
                    "token_estimate": 80,
                },
            },
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Msg 3"},
        ]
        _write_transcript_with_markers(tmp_path, records)

        ctx = ManagedContextManager(session_dir=tmp_path)
        await ctx._load_from_transcript()

        # _messages should only contain msgs after the summary boundary (offset 2)
        assert len(ctx._messages) == 2
        assert ctx._messages[0]["content"] == "Msg 2"
        assert ctx._messages[1]["content"] == "Msg 3"

        # _transcript_message_offset should be 2 (last summarized end index)
        assert ctx._transcript_message_offset == 2

        # _summarized_through_turn should be 1 (from turn_range[1])
        assert ctx._summarized_through_turn == 1

        # _current_turn = _summarized_through_turn + user messages in verbatim window
        # verbatim has 1 user message ("Msg 2") -> _current_turn = 1 + 1 = 2
        assert ctx._current_turn == 2

        assert ctx._loaded_from_transcript is True

    @pytest.mark.asyncio
    async def test_resume_without_tiers_loads_all_messages(self, tmp_path):
        """_load_from_transcript() loads all messages with offset=0 when no summary markers exist."""
        from amplifier_module_context_managed import ManagedContextManager

        records = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        _write_transcript_with_markers(tmp_path, records)

        ctx = ManagedContextManager(session_dir=tmp_path)
        await ctx._load_from_transcript()

        # All 3 messages should be in _messages
        assert len(ctx._messages) == 3
        assert ctx._messages[0]["content"] == "First"
        assert ctx._messages[1]["content"] == "Second"
        assert ctx._messages[2]["content"] == "Third"

        # _transcript_message_offset should be 0
        assert ctx._transcript_message_offset == 0

        # No summary tiers
        assert len(ctx._summary_tiers) == 0

        assert ctx._loaded_from_transcript is True

    @pytest.mark.asyncio
    async def test_resume_reconstructs_multiple_tiers(self, tmp_path):
        """_load_from_transcript() reconstructs multiple SummaryTier objects correctly."""
        from amplifier_module_context_managed import ManagedContextManager

        records = [
            {"role": "user", "content": "Msg 0"},
            {"role": "assistant", "content": "Msg 1"},
            {
                "role": "system",
                "content": "Summary of msgs 0-1",
                "metadata": {
                    "type": "context_managed_summary",
                    "turn_range": [1, 1],
                    "source_message_range": [0, 2],
                    "compression_passes": 1,
                    "token_estimate": 80,
                },
            },
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Msg 3"},
            {
                "role": "system",
                "content": "Summary of msgs 2-3",
                "metadata": {
                    "type": "context_managed_summary",
                    "turn_range": [2, 2],
                    "source_message_range": [2, 4],
                    "compression_passes": 1,
                    "token_estimate": 90,
                },
            },
            {"role": "user", "content": "Msg 4"},
            {"role": "assistant", "content": "Msg 5"},
        ]
        _write_transcript_with_markers(tmp_path, records)

        ctx = ManagedContextManager(session_dir=tmp_path)
        await ctx._load_from_transcript()

        # Should have 2 tiers reconstructed
        assert len(ctx._summary_tiers) == 2

        # First tier covers messages [0, 2)
        assert ctx._summary_tiers[0].source_message_range == (0, 2)
        assert ctx._summary_tiers[0].turn_range == (1, 1)

        # Second tier covers messages [2, 4)
        assert ctx._summary_tiers[1].source_message_range == (2, 4)
        assert ctx._summary_tiers[1].turn_range == (2, 2)

        # _messages should be [Msg 4, Msg 5] (after max source end = 4)
        assert len(ctx._messages) == 2
        assert ctx._messages[0]["content"] == "Msg 4"
        assert ctx._messages[1]["content"] == "Msg 5"

        # _transcript_message_offset = max source_message_range[1] across all tiers = 4
        assert ctx._transcript_message_offset == 4

        # _summarized_through_turn = max turn_range[1] across all tiers = 2
        assert ctx._summarized_through_turn == 2


class TestClearDuringSummarization:
    """Tests for clear() handling of active summarization (task-13)."""

    @pytest.mark.asyncio
    async def test_clear_resets_phase2_fields(self):
        """clear() resets all Phase 2 fields to their zero/empty defaults."""
        from amplifier_module_context_managed import (
            ManagedContextManager,
            SummaryResult,
            SummaryTier,
        )

        mgr = ManagedContextManager()

        # Set all Phase 2 fields to non-zero/non-default values
        mgr._current_turn = 5
        mgr._summarized_through_turn = 3
        mgr._transcript_message_offset = 10
        mgr._is_summarizing = True
        mgr._pending_summary = SummaryResult(
            summary_text="Some pending summary",
            turn_range=(1, 3),
            source_message_range=(0, 6),
        )
        mgr._summarization_failures = 4
        mgr._summary_tiers = [
            SummaryTier(
                content="Tier content",
                turn_range=(1, 2),
                source_message_range=(0, 4),
                compression_passes=1,
                token_estimate=100,
            )
        ]
        # _summarization_task is None (cancel handled in separate test)

        await mgr.clear()

        # Verify all Phase 2 fields are reset
        assert mgr._current_turn == 0
        assert mgr._summarized_through_turn == 0
        assert mgr._transcript_message_offset == 0
        assert mgr._is_summarizing is False
        assert mgr._pending_summary is None
        assert mgr._summarization_failures == 0
        assert mgr._summary_tiers == []
        assert mgr._summarization_task is None

    @pytest.mark.asyncio
    async def test_clear_cancels_active_task(self):
        """clear() cancels and clears an in-flight _summarization_task."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager()

        # Create a long-running asyncio task
        long_task = asyncio.create_task(asyncio.sleep(100))
        mgr._summarization_task = long_task

        await mgr.clear()

        # Give the event loop a tick to process the cancellation
        await asyncio.sleep(0)

        # Task should be cancelled
        assert long_task.cancelled()
        # _summarization_task should be None after clear
        assert mgr._summarization_task is None


class TestNoProviderGuard:
    """Tests for no-provider guard in summarization trigger (task-13)."""

    @pytest.mark.asyncio
    async def test_trigger_deferred_without_provider(self):
        """Summarization trigger is deferred (no task created) when _cached_provider is None."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(
            max_tokens=1000,
            summarize_trigger=0.80,
            verbatim_window_tokens=10,
        )

        # No provider set - verify initial state
        assert mgr._cached_provider is None

        # Pre-seed messages so _calculate_segment_boundary() returns non-None
        mgr._messages = [
            {"role": "user", "content": "earlier message"},
            {"role": "assistant", "content": "earlier response"},
        ]
        # 90% of 1000 - well above the 0.80 trigger threshold
        mgr._running_token_estimate = 900

        # Add a message - trigger should be deferred (no provider)
        await mgr.add_message({"role": "user", "content": "new message"})

        # Give event loop a tick
        await asyncio.sleep(0)

        # No summarization should have started (guard blocked it)
        assert mgr._is_summarizing is False
        assert mgr._summarization_task is None
        assert mgr._summarization_failures == 0

    @pytest.mark.asyncio
    async def test_trigger_fires_after_provider_available(self):
        """Summarization trigger fires on the next add_message() once provider is available."""
        from amplifier_module_context_managed import ManagedContextManager

        mgr = ManagedContextManager(
            max_tokens=1000,
            summarize_trigger=0.80,
            verbatim_window_tokens=10,
        )

        # No provider initially
        assert mgr._cached_provider is None

        # Pre-seed messages and high usage
        mgr._messages = [
            {"role": "user", "content": "earlier message"},
            {"role": "assistant", "content": "earlier response"},
        ]
        mgr._running_token_estimate = 900  # 90% > 0.80 threshold

        # First add_message without provider - trigger deferred
        await mgr.add_message({"role": "user", "content": "first message"})
        await asyncio.sleep(0)

        # Verify still deferred
        assert mgr._summarization_task is None
        assert mgr._summarization_failures == 0

        # Provider becomes available via get_messages_for_request
        mock_provider = object()  # Non-None provider - will fail on .complete() call
        await mgr.get_messages_for_request(provider=mock_provider)

        # Provider is now cached
        assert mgr._cached_provider is mock_provider

        # Next add_message - trigger should fire now that provider is available
        await mgr.add_message({"role": "user", "content": "second message"})

        # Wait for background task to run and complete
        await asyncio.sleep(0)
        # The task may still be running; wait for it to finish
        if mgr._summarization_task is not None:
            task = mgr._summarization_task
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass

        await asyncio.sleep(0)

        # Summarization was triggered: task ran and failed (object() has no .complete())
        # This proves the trigger fired (not deferred)
        assert mgr._summarization_failures == 1
