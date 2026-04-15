"""
Tests for Phase 2 summarization dataclasses: SummaryResult and SummaryTier.

Task 1: Add SummaryResult and SummaryTier dataclasses.
Task 2: Default summarization prompt and _get_summarization_prompt().
Task 4: Trigger summarization with guard and threshold wiring.
"""

import asyncio

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
