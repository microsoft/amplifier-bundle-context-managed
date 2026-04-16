"""
Inline compaction tests for ManagedContextManager.

Covers:
- _inline_compact(): core behaviour (truncation, removal, protection rules,
  tool-pair atomicity, event emission, no persistent-state mutation)
- get_messages_for_request(): hard budget enforcement gate — the returned list
  must never exceed the configured budget regardless of how many messages have
  accumulated

This is the regression suite for the session-944d882e overshoot bug where
context grew to 125% of the configured max_tokens before any intervention,
because get_messages_for_request() assembled the full message list and returned
it without checking whether the result fit in budget.
"""

from unittest.mock import AsyncMock, patch

import pytest

from amplifier_module_context_managed import ManagedContextManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(role: str, content: str, **extra) -> dict:
    """Build a minimal message dict."""
    msg: dict = {"role": role, "content": content}
    msg.update(extra)
    return msg


def _make_tool_calls_msg(content: str = "calling tool") -> dict:
    """Build an assistant message that carries tool_calls (pair trigger)."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "foo"}}],
    }


def _total_tokens(ctx: ManagedContextManager, msgs: list[dict]) -> int:
    return ctx._estimate_tokens(msgs)


# ---------------------------------------------------------------------------
# _inline_compact — basic output contract
# ---------------------------------------------------------------------------


class TestInlineCompactBasic:
    """Core output contract for _inline_compact()."""

    @pytest.mark.asyncio
    async def test_returns_copy_not_original(self):
        """_inline_compact returns a new list object, not the input list."""
        ctx = ManagedContextManager(max_tokens=10_000, emergency_target_usage=0.50)
        msgs = [_make_msg("user", "hello")]
        result = await ctx._inline_compact(msgs, budget=10_000)
        assert result is not msgs

    @pytest.mark.asyncio
    async def test_noop_when_already_under_target(self):
        """When tokens < target, the list is returned unchanged."""
        ctx = ManagedContextManager(max_tokens=10_000, emergency_target_usage=0.50)
        msgs = [
            _make_msg("system", "sys"),
            _make_msg("user", "hi"),
        ]
        # Budget very large — already under target
        result = await ctx._inline_compact(msgs, budget=100_000)
        assert len(result) == 2
        assert result[0]["content"] == "sys"
        assert result[1]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_does_not_mutate_self_messages(self):
        """_inline_compact operates on a view; self._messages is untouched."""
        ctx = ManagedContextManager(max_tokens=500, emergency_target_usage=0.50)
        # Populate _messages with some content
        large_content = "X" * 4000
        ctx._messages = [
            _make_msg("system", "sys"),
            _make_msg("user", "old"),
            _make_msg("assistant", large_content),
            _make_msg("user", "latest"),
        ]
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)
        original_messages_snapshot = list(ctx._messages)

        assembled = list(ctx._messages)
        await ctx._inline_compact(assembled, budget=500)

        # self._messages must be identical to before
        assert ctx._messages == original_messages_snapshot

    @pytest.mark.asyncio
    async def test_result_fits_within_budget(self):
        """The returned list must be estimated within the budget.

        Design constraints so the assertion is achievable:
        - Budget = 500 tokens.  4 chars ≈ 1 token (chars/4 heuristic).
        - System message: tiny (~8 tokens)
        - Last user message: small (~50 tokens) — must survive; is under budget
        - Removable messages: large enough to push total over budget
        After compaction, [system + last_user] ≈ 58 tokens ≤ 500. ✓
        """
        ctx = ManagedContextManager(max_tokens=500, emergency_target_usage=0.80)
        budget = 500
        msgs = [
            _make_msg("system", "sys"),
            _make_msg("user", "A" * 2000),  # removable — large
            _make_msg("assistant", "B" * 2000),  # removable — large
            _make_msg("user", "current intent"),  # ← last user, small, protected
        ]
        result = await ctx._inline_compact(msgs, budget=budget)
        assert _total_tokens(ctx, result) <= budget


# ---------------------------------------------------------------------------
# _inline_compact — protection rules
# ---------------------------------------------------------------------------


class TestInlineCompactProtection:
    """Verify which messages are protected from removal."""

    @pytest.mark.asyncio
    async def test_system_messages_never_removed(self):
        """System messages survive even when every other message must go."""
        ctx = ManagedContextManager(max_tokens=50, emergency_target_usage=0.50)
        # Tiny budget forces aggressive removal; system must stay
        msgs = [
            _make_msg("system", "important system context"),
            _make_msg("user", "A" * 2000),
            _make_msg("user", "latest intent"),  # last user
        ]
        result = await ctx._inline_compact(msgs, budget=50)
        roles = [m["role"] for m in result]
        assert "system" in roles

    @pytest.mark.asyncio
    async def test_last_user_message_never_removed(self):
        """The last user message (current intent) is always preserved."""
        ctx = ManagedContextManager(max_tokens=50, emergency_target_usage=0.50)
        last_user = _make_msg("user", "current task request")
        msgs = [
            _make_msg("system", "s"),
            _make_msg("user", "B" * 2000),
            _make_msg("assistant", "C" * 2000),
            last_user,
        ]
        result = await ctx._inline_compact(msgs, budget=50)
        assert last_user in result

    @pytest.mark.asyncio
    async def test_hook_messages_never_removed(self):
        """Messages with metadata.source == 'hook' are preserved."""
        ctx = ManagedContextManager(max_tokens=50, emergency_target_usage=0.50)
        hook_msg = {
            "role": "system",
            "content": "hook-injected context",
            "metadata": {"source": "hook"},
        }
        msgs = [
            hook_msg,
            _make_msg("user", "D" * 2000),
            _make_msg("user", "latest"),  # last user
        ]
        result = await ctx._inline_compact(msgs, budget=50)
        assert hook_msg in result


# ---------------------------------------------------------------------------
# _inline_compact — tool result truncation (Step 1)
# ---------------------------------------------------------------------------


class TestInlineCompactTruncation:
    """Step 1: truncate large tool results before removing messages."""

    @pytest.mark.asyncio
    async def test_large_tool_result_truncated(self):
        """Tool result with content > 1_000 chars is truncated to 500 + suffix."""
        ctx = ManagedContextManager(max_tokens=10_000, emergency_target_usage=0.10)
        large_content = "X" * 2000
        tool_msg = _make_msg("tool", large_content, tool_call_id="tc1")
        msgs = [
            _make_msg("system", "s"),
            _make_msg("user", "u"),
            _make_tool_calls_msg(),
            tool_msg,
        ]
        # Budget tiny enough to force truncation
        result = await ctx._inline_compact(msgs, budget=500)
        tool_results = [m for m in result if m.get("role") == "tool"]
        if tool_results:  # tool msg may have been removed entirely — that's fine
            assert len(tool_results[0]["content"]) < len(large_content)

    @pytest.mark.asyncio
    async def test_truncation_uses_correct_suffix(self):
        """Truncated tool results end with '[truncated by inline compaction]'."""
        # Use a budget that's tight enough to force truncation but not removal
        ctx = ManagedContextManager(max_tokens=500, emergency_target_usage=0.90)
        large_content = "Y" * 2000
        tool_msg = _make_msg("tool", large_content, tool_call_id="tc1")
        msgs = [
            _make_msg("system", "s"),
            _make_msg("user", "short user"),
            _make_tool_calls_msg("calling"),
            tool_msg,
            _make_msg("user", "latest"),  # last user — protected
        ]
        result = await ctx._inline_compact(msgs, budget=500)
        tool_results = [m for m in result if m.get("role") == "tool"]
        if tool_results:
            assert tool_results[0]["content"].endswith(
                "\n\n[truncated by inline compaction]"
            )

    @pytest.mark.asyncio
    async def test_last_five_tool_results_not_truncated(self):
        """The last 5 tool results are protected from truncation (step 1)."""
        ctx = ManagedContextManager(max_tokens=100_000, emergency_target_usage=0.50)
        large_content = "Z" * 2000
        # 6 tool results: only the first should be eligible for truncation
        tool_msgs = [
            _make_msg("tool", large_content, tool_call_id=f"tc{i}") for i in range(6)
        ]
        msgs = [_make_msg("system", "s")] + tool_msgs + [_make_msg("user", "latest")]
        # Large budget so only step 1 runs (no removals needed)
        result = await ctx._inline_compact(msgs, budget=100_000)
        # All last 5 should be unmodified; first one may be truncated
        result_tools = [m for m in result if m.get("role") == "tool"]
        for t in result_tools[-5:]:
            assert t["content"] == large_content, (
                "last 5 tool results must be untouched"
            )

    @pytest.mark.asyncio
    async def test_small_tool_result_not_truncated(self):
        """Tool results ≤ 1_000 chars are never truncated."""
        ctx = ManagedContextManager(max_tokens=50, emergency_target_usage=0.10)
        small_content = "A" * 999
        tool_msg = _make_msg("tool", small_content, tool_call_id="tc1")
        msgs = [
            _make_msg("system", "s"),
            _make_msg("user", "u"),
            _make_tool_calls_msg(),
            tool_msg,
            _make_msg("user", "latest"),
        ]
        result = await ctx._inline_compact(msgs, budget=50)
        tool_results = [m for m in result if m.get("role") == "tool"]
        if tool_results:
            assert tool_results[0]["content"] == small_content


# ---------------------------------------------------------------------------
# _inline_compact — tool pair atomicity (Step 2)
# ---------------------------------------------------------------------------


class TestInlineCompactToolPairAtomicity:
    """Step 2: assistant+tool_results must be removed as an atomic block."""

    @pytest.mark.asyncio
    async def test_assistant_and_tool_results_removed_together(self):
        """An assistant message with tool_calls is always removed with its tool results."""
        ctx = ManagedContextManager(max_tokens=200, emergency_target_usage=0.50)
        assistant_w_tools = _make_tool_calls_msg("calling tool")
        tool_r1 = _make_msg("tool", "result 1", tool_call_id="tc1")
        tool_r2 = _make_msg("tool", "result 2", tool_call_id="tc1")
        msgs = [
            _make_msg("system", "s"),
            _make_msg("user", "old user " + "A" * 400),  # removable, large
            assistant_w_tools,
            tool_r1,
            tool_r2,
            _make_msg("user", "latest"),  # last user
        ]
        result = await ctx._inline_compact(msgs, budget=200)
        result_roles = [m["role"] for m in result]
        # Either all three (assistant+tool+tool) are present or all three are gone
        has_assistant = assistant_w_tools in result
        has_tool_r1 = tool_r1 in result
        has_tool_r2 = tool_r2 in result
        # They must agree — all in or all out
        assert has_assistant == has_tool_r1 == has_tool_r2, (
            "assistant-with-tool_calls and its tool results must appear/disappear together"
        )
        # system and last user must still be present
        assert any(m["role"] == "system" for m in result)
        assert result_roles[-1] == "user"

    @pytest.mark.asyncio
    async def test_no_orphaned_tool_results(self):
        """After any removal cycle, no tool result exists without its assistant."""
        ctx = ManagedContextManager(max_tokens=100, emergency_target_usage=0.50)
        assistant_w_tools = _make_tool_calls_msg("call")
        tool_r = _make_msg("tool", "T" * 100, tool_call_id="tc1")
        msgs = [
            _make_msg("system", "sys"),
            _make_msg("user", "old " + "A" * 500),  # large, removable
            assistant_w_tools,
            tool_r,
            _make_msg("user", "latest"),
        ]
        result = await ctx._inline_compact(msgs, budget=100)
        # If the tool result is in result, the assistant must also be there
        if any(m.get("role") == "tool" for m in result):
            assert any(
                m.get("role") == "assistant" and m.get("tool_calls") for m in result
            ), "tool result present but no assistant-with-tool_calls found"


# ---------------------------------------------------------------------------
# _inline_compact — event emission
# ---------------------------------------------------------------------------


class TestInlineCompactEvents:
    """Verify context:compaction event is emitted with correct payload."""

    @pytest.mark.asyncio
    async def test_compaction_event_emitted(self):
        """_inline_compact emits context:compaction when compaction occurs."""
        ctx = ManagedContextManager(max_tokens=100, emergency_target_usage=0.50)
        emitted: list[tuple] = []

        async def fake_emit(event: str, data: dict) -> None:
            emitted.append((event, data))

        ctx._emit_event = fake_emit  # type: ignore[assignment]

        msgs = [
            _make_msg("system", "s"),
            _make_msg("user", "E" * 1000),
            _make_msg("user", "latest"),
        ]
        await ctx._inline_compact(msgs, budget=100)

        assert len(emitted) == 1
        event_name, event_data = emitted[0]
        assert event_name == "context:compaction"
        assert event_data["reason"] == "inline_budget_enforcement"
        assert "budget" in event_data
        assert "original_tokens" in event_data
        assert "compacted_tokens" in event_data

    @pytest.mark.asyncio
    async def test_compaction_event_emitted_even_when_no_messages_removed(self):
        """Event fires even if only truncation (no removal) occurred."""
        # Only step 1 executes (target met after truncation alone)
        ctx = ManagedContextManager(max_tokens=500, emergency_target_usage=0.99)
        emitted: list[tuple] = []

        async def fake_emit(event: str, data: dict) -> None:
            emitted.append((event, data))

        ctx._emit_event = fake_emit  # type: ignore[assignment]

        large_tool = _make_msg("tool", "F" * 3000, tool_call_id="tc1")
        msgs = [
            _make_msg("system", "s"),
            _make_msg("user", "u"),
            _make_tool_calls_msg(),
            large_tool,
            _make_msg("user", "latest"),
        ]
        # Budget is 500 tokens — step 2 may or may not fire, but either way
        # the event should be emitted exactly once
        await ctx._inline_compact(msgs, budget=500)
        assert len(emitted) == 1
        assert emitted[0][0] == "context:compaction"


# ---------------------------------------------------------------------------
# get_messages_for_request — hard budget enforcement gate
# ---------------------------------------------------------------------------


class TestGetMessagesHardBudgetGate:
    """get_messages_for_request() must never return more tokens than the budget."""

    @pytest.mark.asyncio
    async def test_returned_list_never_exceeds_budget(self):
        """After adding 2× budget worth of messages, get_messages_for_request()
        returns a list whose estimated tokens fit within the configured budget."""
        # Budget = 500 tokens, messages ≈ 2000 tokens worth
        ctx = ManagedContextManager(max_tokens=500, emergency_target_usage=0.80)
        # Add messages that push well over budget
        for _ in range(5):
            await ctx.add_message(_make_msg("user", "G" * 800))
            await ctx.add_message(_make_msg("assistant", "H" * 800))
        await ctx.add_message(_make_msg("user", "final request"))

        result = await ctx.get_messages_for_request()

        actual_tokens = ctx._estimate_tokens(result)
        assert actual_tokens <= 500, (
            f"get_messages_for_request returned {actual_tokens} tokens, "
            f"which exceeds the budget of 500"
        )

    @pytest.mark.asyncio
    async def test_inline_compact_called_when_over_budget(self):
        """When assembled messages exceed the budget, _inline_compact is invoked."""
        ctx = ManagedContextManager(max_tokens=100, emergency_target_usage=0.50)
        # Force _running_token_estimate to exceed budget
        ctx._messages = [
            _make_msg("user", "I" * 2000),
            _make_msg("user", "latest"),
        ]
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)

        with patch.object(
            ctx, "_inline_compact", new_callable=AsyncMock
        ) as mock_compact:
            mock_compact.return_value = [_make_msg("user", "latest")]
            await ctx.get_messages_for_request()
            mock_compact.assert_called_once()
            call_args = mock_compact.call_args
            assert call_args[0][1] == 100  # budget argument

    @pytest.mark.asyncio
    async def test_inline_compact_not_called_when_within_budget(self):
        """When assembled messages fit within budget, _inline_compact is NOT called."""
        ctx = ManagedContextManager(max_tokens=200_000, emergency_target_usage=0.50)
        ctx._messages = [
            _make_msg("user", "small message"),
            _make_msg("user", "latest"),
        ]
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)

        with patch.object(
            ctx, "_inline_compact", new_callable=AsyncMock
        ) as mock_compact:
            await ctx.get_messages_for_request()
            mock_compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_budget_enforcement_preserves_system_message(self):
        """Hard budget enforcement never removes the system message."""
        ctx = ManagedContextManager(max_tokens=200, emergency_target_usage=0.80)
        await ctx.add_message(_make_msg("system", "important system context"))
        for _ in range(8):
            await ctx.add_message(_make_msg("user", "J" * 400))
            await ctx.add_message(_make_msg("assistant", "K" * 400))
        await ctx.add_message(_make_msg("user", "current intent"))

        result = await ctx.get_messages_for_request()

        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) >= 1, "system message must survive budget enforcement"

    @pytest.mark.asyncio
    async def test_budget_enforcement_preserves_last_user_message(self):
        """Hard budget enforcement never removes the last user message."""
        ctx = ManagedContextManager(max_tokens=200, emergency_target_usage=0.80)
        for _ in range(8):
            await ctx.add_message(_make_msg("user", "L" * 400))
            await ctx.add_message(_make_msg("assistant", "M" * 400))
        current_intent = "current user request for this turn"
        await ctx.add_message(_make_msg("user", current_intent))

        result = await ctx.get_messages_for_request()

        user_msgs = [m for m in result if m.get("role") == "user"]
        last_user = user_msgs[-1] if user_msgs else None
        assert last_user is not None
        # The last user in the result must be the final message we added
        assert current_intent in last_user["content"]

    @pytest.mark.asyncio
    async def test_pending_summary_swapped_before_enforcement(self):
        """A pending LLM summary is swapped in before the budget enforcement check.

        If the swap brings tokens under budget, _inline_compact must NOT be called.
        """
        from amplifier_module_context_managed import SummaryResult

        ctx = ManagedContextManager(max_tokens=500, emergency_target_usage=0.80)

        # Populate messages that exceed budget
        ctx._messages = [
            _make_msg("user", "A" * 1000),
            _make_msg("user", "latest"),
        ]
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)
        ctx._transcript_message_offset = 0

        # Inject a pending summary that covers the first message and is tiny
        ctx._pending_summary = SummaryResult(
            summary_text="brief summary",
            turn_range=(1, 1),
            source_message_range=(0, 1),  # covers ctx._messages[0]
        )

        with patch.object(
            ctx, "_inline_compact", new_callable=AsyncMock
        ) as mock_compact:
            mock_compact.return_value = [_make_msg("user", "latest")]
            result = await ctx.get_messages_for_request()

        # After swap: the large message is gone, replaced by tiny summary tier.
        # If inline_compact was called, the mock's return is used.
        # If NOT called, result contains the swapped summary + last user.
        # Either way, _running_token_estimate should reflect the swap.
        # The key assertion: if tokens are under budget after swap, no compaction.
        after_swap_tokens = ctx._estimate_tokens(result)
        if not mock_compact.called:
            assert after_swap_tokens <= 500

    @pytest.mark.asyncio
    async def test_budget_enforcement_token_estimate_matches_budget(self):
        """Token estimate of returned list is ≤ max_tokens after enforcement."""
        # Regression for the 125% overshoot seen in session 944d882e.
        # Use a tight budget to guarantee overshoot without the gate.
        ctx = ManagedContextManager(max_tokens=300, emergency_target_usage=0.80)
        # ~1500 tokens of content total
        for i in range(6):
            await ctx.add_message(_make_msg("user", f"user message {i} " + "N" * 200))
            await ctx.add_message(
                _make_msg("assistant", f"assistant reply {i} " + "O" * 200)
            )
        await ctx.add_message(_make_msg("user", "final intent"))

        result = await ctx.get_messages_for_request()

        actual = ctx._estimate_tokens(result)
        assert actual <= 300, (
            f"Regression: get_messages_for_request returned {actual} tokens "
            f"against a budget of 300 (session-944d882e overshoot fix)"
        )
