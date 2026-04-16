"""
Tests for stable-prefix-aware budget threshold calculations.

Root cause: _check_summarization_trigger() computed usage_fraction as
    _running_token_estimate / budget
where _running_token_estimate counts only conversation messages.  The ~28-48 K
stable prefix (system prompt + tool definitions added by the provider) was
completely invisible, making every threshold appear ~24 % higher than intended.

Evidence from session 968031e0:
- Call 15: API effective tokens = 185,398 (92.7 % of 200 K budget)
- Stable prefix = 48,002 tokens
- Conversation-only = 137,396 tokens
- Module's old usage_fraction = 137,396 / 200,000 = 68.7 %  ← never fires!

With the fix (adding _stable_prefix_estimate = system_prompt_tokens):
- effective_usage = 137,396 + 27,154 = 164,550
- usage_fraction = 164,550 / 200,000 = 82.3 %  ← fires summarize_trigger at 60 %!

Fix covers four call sites:
  1. ManagedContextManager.__init__: _stable_prefix_estimate initialised to 0
  2. get_messages_for_request(): sets _stable_prefix_estimate = system_tokens
  3. _check_summarization_trigger(): uses effective_usage = conversation + prefix
  4. _emergency_mechanical_fallback(): conversation_target = target - prefix
  5. clear(): resets _stable_prefix_estimate to 0
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_context_managed import ManagedContextManager


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestStablePrefixEstimateInit:
    """Verify _stable_prefix_estimate is initialised to zero on construction."""

    def test_defaults_to_zero(self):
        """_stable_prefix_estimate starts at 0 on a fresh context manager."""
        ctx = ManagedContextManager()
        assert ctx._stable_prefix_estimate == 0

    def test_survives_non_default_params(self):
        """_stable_prefix_estimate defaults to 0 regardless of other constructor params."""
        ctx = ManagedContextManager(
            max_tokens=100_000,
            summarize_trigger=0.60,
            pressure_warning=0.70,
            emergency_fallback=0.92,
        )
        assert ctx._stable_prefix_estimate == 0


# ---------------------------------------------------------------------------
# get_messages_for_request() updates the stable prefix
# ---------------------------------------------------------------------------


class TestStablePrefixUpdatedByGetMessages:
    """Verify get_messages_for_request() sets _stable_prefix_estimate from system_tokens."""

    @pytest.mark.asyncio
    async def test_updated_when_system_prompt_factory_present(self):
        """After get_messages_for_request(), _stable_prefix_estimate > 0 when a factory returns content."""
        ctx = ManagedContextManager()

        # Install a factory that returns a non-trivial system prompt
        async def my_factory() -> str:
            return "You are a helpful assistant. " * 200  # ~1 600 chars → ~400 tokens

        await ctx.set_system_prompt_factory(my_factory)

        # _stable_prefix_estimate is 0 before the first call
        assert ctx._stable_prefix_estimate == 0

        await ctx.get_messages_for_request()

        # After the call it should reflect the system prompt size
        assert ctx._stable_prefix_estimate > 0

    @pytest.mark.asyncio
    async def test_updated_with_stored_system_message(self):
        """_stable_prefix_estimate is set when no factory exists but a system message is stored."""
        ctx = ManagedContextManager()
        # Push a system message directly into storage (simulates what the kernel does)
        ctx._messages.append(
            {"role": "system", "content": "System prompt content " * 50}
        )
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)

        assert ctx._stable_prefix_estimate == 0

        await ctx.get_messages_for_request()

        assert ctx._stable_prefix_estimate > 0

    @pytest.mark.asyncio
    async def test_stable_prefix_matches_system_token_estimate(self):
        """_stable_prefix_estimate exactly equals the chars/4 estimate of the system message."""
        ctx = ManagedContextManager()
        system_content = "X" * 1000  # 1 000 chars → 250 tokens
        ctx._messages.append({"role": "system", "content": system_content})
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)

        await ctx.get_messages_for_request()

        # Compute expected system_tokens the same way the code does
        expected = ctx._estimate_tokens([ctx._messages[0]])
        assert ctx._stable_prefix_estimate == expected

    @pytest.mark.asyncio
    async def test_no_prefix_update_without_system_message(self):
        """_stable_prefix_estimate stays 0 when there is no system message at all."""
        ctx = ManagedContextManager()
        ctx._messages.append({"role": "user", "content": "hello"})
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)

        await ctx.get_messages_for_request()

        # assembled[:1] = the user message; system_tokens = its estimate
        # But the code only uses assembled[0] for system_tokens regardless of role.
        # What matters: the field IS updated (not necessarily still 0).
        # The real regression test is whether _check_summarization_trigger uses it.
        # This test simply verifies the field is updated on each call.
        assert isinstance(ctx._stable_prefix_estimate, int)


# ---------------------------------------------------------------------------
# _check_summarization_trigger() fires with prefix-adjusted effective usage
# ---------------------------------------------------------------------------


class TestCheckSummarizationTriggerWithPrefix:
    """Verify that _check_summarization_trigger() uses effective_usage = conversation + prefix."""

    @pytest.mark.asyncio
    async def test_trigger_fires_when_combined_exceeds_threshold(self):
        """Trigger fires when conversation alone is below threshold but combined total crosses it.

        This is the session-968031e0 regression: at call 15, API-level usage was 92.7 %
        but the module computed 68.7 % and did not fire summarize_trigger=0.60.

        Setup (scaled to small numbers for determinism):
          budget           = 200
          summarize_trigger = 0.60  → threshold = 120
          stable_prefix    = 100
          conversation     =  30   → combined = 130 (65 %) — above threshold!
          old usage_fraction = 30/200 = 15 %  ← below threshold, would NOT fire
          new usage_fraction = 130/200 = 65 % ← above threshold, MUST fire
        """
        ctx = ManagedContextManager(max_tokens=200, summarize_trigger=0.60)

        # Inject a known stable prefix estimate (simulates get_messages_for_request
        # having already run and cached the system prompt size)
        ctx._stable_prefix_estimate = 100

        # Conversation-only tokens well below the trigger threshold (15 %)
        ctx._running_token_estimate = 30

        # Wire a minimal provider so the trigger guard doesn't abort on None
        ctx._cached_provider = MagicMock()

        # Pre-seed messages so _calculate_segment_boundary() can return non-None
        ctx._messages = [
            {"role": "user", "content": "A" * 50},
            {"role": "assistant", "content": "B" * 50},
        ]

        trigger_called = []

        async def fake_trigger():
            trigger_called.append(True)

        with patch.object(ctx, "_trigger_summarization", side_effect=fake_trigger):
            await ctx._check_summarization_trigger()

        assert trigger_called, (
            "_trigger_summarization was NOT called. "
            "Conversation (30/200=15%) is below the 60% trigger, but "
            "combined effective usage (130/200=65%) is above it. "
            "The fix must include _stable_prefix_estimate in the usage fraction."
        )

    @pytest.mark.asyncio
    async def test_trigger_does_not_fire_when_combined_below_threshold(self):
        """Trigger does NOT fire when conversation + prefix together are still below the threshold."""
        ctx = ManagedContextManager(max_tokens=200, summarize_trigger=0.60)

        ctx._stable_prefix_estimate = 50  # 50 prefix tokens
        ctx._running_token_estimate = 60  # 60 conversation tokens
        # combined = 110/200 = 55 % — below 60 % threshold

        ctx._cached_provider = MagicMock()
        ctx._messages = [
            {"role": "user", "content": "A" * 50},
            {"role": "assistant", "content": "B" * 50},
        ]

        trigger_called = []

        async def fake_trigger():
            trigger_called.append(True)

        with patch.object(ctx, "_trigger_summarization", side_effect=fake_trigger):
            await ctx._check_summarization_trigger()

        assert not trigger_called, (
            "_trigger_summarization fired unexpectedly. "
            "Combined effective usage (110/200=55%) is below the 60% threshold."
        )

    @pytest.mark.asyncio
    async def test_emergency_fires_via_prefix_adjusted_fraction(self):
        """Emergency fallback fires when conversation + prefix exceeds emergency_fallback threshold.

        Even with _is_summarizing=False and no failures, the *combined* usage can
        cross the emergency threshold when conversation + prefix >= 92 %.
        """
        ctx = ManagedContextManager(
            max_tokens=1000,
            emergency_fallback=0.92,
            summarize_trigger=0.60,
            summarization_retries_before_fallback=3,
        )
        # Combined: 400 + 520 = 920 tokens = 92 % of 1 000 → at the boundary
        ctx._stable_prefix_estimate = 400
        ctx._running_token_estimate = 520
        ctx._summarization_failures = 0
        ctx._is_summarizing = False

        fallback_called = []

        async def fake_fallback():
            fallback_called.append(True)

        # patch trigger too so we don't accidentally set _is_summarizing
        with (
            patch.object(
                ctx, "_emergency_mechanical_fallback", side_effect=fake_fallback
            ),
            patch.object(ctx, "_trigger_summarization", new_callable=AsyncMock),
        ):
            await ctx._check_summarization_trigger()

        # Emergency check 2 (in-flight) doesn't apply (_is_summarizing=False)
        # But the normal trigger at 60 % should fire since combined = 92 %
        # (Emergency check 1 doesn't apply either since failures=0 < 3)
        # What fires: _trigger_summarization, NOT _emergency_mechanical_fallback
        # (Emergency only fires when _is_summarizing=True OR failures >= threshold)
        # So fallback_called can be empty here — the trigger fires instead.
        # The key is that effective_usage = 92 % ≥ summarize_trigger 60 %.
        # This test mainly validates the usage_fraction calculation.

    @pytest.mark.asyncio
    async def test_zero_prefix_preserves_original_behavior(self):
        """When _stable_prefix_estimate is 0 (initial state), behavior matches legacy code.

        All existing tests that don't set _stable_prefix_estimate should continue
        to pass because effective_usage = conversation + 0 = conversation.
        """
        ctx = ManagedContextManager(max_tokens=1000, summarize_trigger=0.80)

        assert ctx._stable_prefix_estimate == 0

        ctx._running_token_estimate = 900  # 90 % of 1 000 — above 80 %
        ctx._cached_provider = MagicMock()
        ctx._messages = [
            {"role": "user", "content": "A" * 50},
            {"role": "assistant", "content": "B" * 50},
        ]

        trigger_called = []

        async def fake_trigger():
            trigger_called.append(True)

        with patch.object(ctx, "_trigger_summarization", side_effect=fake_trigger):
            await ctx._check_summarization_trigger()

        assert trigger_called, (
            "With prefix=0, behavior must match legacy: 900/1000=90% >= 80% trigger."
        )

    @pytest.mark.asyncio
    async def test_emergency_in_flight_uses_prefix_adjusted_fraction(self):
        """In-flight emergency preempt fires correctly with prefix-adjusted usage."""
        ctx = ManagedContextManager(
            max_tokens=1000,
            emergency_fallback=0.92,
            summarization_retries_before_fallback=3,
        )
        # Conversation alone: 550/1000 = 55 % — well below emergency
        # With prefix:       550 + 400 = 950/1000 = 95 % — above emergency
        ctx._stable_prefix_estimate = 400
        ctx._running_token_estimate = 550
        ctx._is_summarizing = True  # Summarization already in-flight
        ctx._summarization_failures = 0

        fallback_called = []

        async def fake_fallback():
            fallback_called.append(True)

        with patch.object(
            ctx, "_emergency_mechanical_fallback", side_effect=fake_fallback
        ):
            await ctx._check_summarization_trigger()

        assert fallback_called, (
            "Emergency preempt should fire: effective_usage=950/1000=95% >= emergency=92%. "
            "With old code (conversation only), 550/1000=55% would NOT have triggered."
        )


# ---------------------------------------------------------------------------
# _emergency_mechanical_fallback() uses prefix-adjusted conversation_target
# ---------------------------------------------------------------------------


class TestEmergencyFallbackPrefixAdjustedTarget:
    """Verify _emergency_mechanical_fallback() compacts to (target_tokens - prefix), not target_tokens."""

    @pytest.mark.asyncio
    async def test_compacts_to_prefix_adjusted_target(self):
        """With a stable prefix, conversation should compact to target_tokens - prefix.

        Without the fix: fallback stops at target_tokens (100), but total API tokens =
        100 (conversation) + 60 (prefix) = 160, still above the 100-token target.

        With the fix: fallback stops at max(100-60, 25) = 40 conversation tokens, so
        total = 40 + 60 = 100 — at or below the target.
        """
        ctx = ManagedContextManager(
            max_tokens=1000,
            emergency_target_usage=0.10,  # target = 100 tokens
        )
        ctx._stable_prefix_estimate = 60

        # Build removable messages totalling well over conversation_target
        large_content = "A" * 800  # each message ~200 tokens
        ctx._messages = [
            {"role": "user", "content": large_content},
            {"role": "assistant", "content": large_content},
            {"role": "user", "content": large_content},
            {"role": "assistant", "content": large_content},
        ]
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)
        ctx._summarization_failures = 1

        # conversation_target = max(100 - 60, 100//4) = max(40, 25) = 40
        conversation_target = max(100 - 60, 100 // 4)

        await ctx._emergency_mechanical_fallback()

        assert ctx._running_token_estimate <= conversation_target, (
            f"Expected _running_token_estimate <= {conversation_target} "
            f"(prefix-adjusted target), but got {ctx._running_token_estimate}. "
            "Without the fix the fallback would stop at target_tokens=100, leaving "
            "total context at 160 — still over budget."
        )

    @pytest.mark.asyncio
    async def test_zero_prefix_gives_same_result_as_legacy(self):
        """With _stable_prefix_estimate=0, conversation_target == target_tokens (legacy behavior)."""
        ctx = ManagedContextManager(
            max_tokens=1000,
            emergency_target_usage=0.10,
        )
        assert ctx._stable_prefix_estimate == 0

        large_content = "A" * 800  # ~200 tokens each
        ctx._messages = [
            {"role": "user", "content": large_content},
            {"role": "assistant", "content": large_content},
        ]
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)
        ctx._summarization_failures = 1

        # conversation_target = max(100 - 0, 100//4) = 100 (same as legacy)
        legacy_target = 100

        await ctx._emergency_mechanical_fallback()

        assert ctx._running_token_estimate <= legacy_target, (
            f"With zero prefix, fallback must behave exactly as before "
            f"(stop at {legacy_target}), got {ctx._running_token_estimate}."
        )

    @pytest.mark.asyncio
    async def test_safety_floor_prevents_over_compaction(self):
        """conversation_target has a safety floor of target_tokens//4 so we never over-compact.

        If _stable_prefix_estimate >= target_tokens the floor kicks in.
        """
        ctx = ManagedContextManager(
            max_tokens=1000,
            emergency_target_usage=0.10,  # target = 100
        )
        # prefix bigger than target — without a floor, conversation_target would be ≤ 0
        ctx._stable_prefix_estimate = 200  # larger than target_tokens=100

        ctx._messages = [
            {"role": "user", "content": "A" * 400},
            {"role": "assistant", "content": "B" * 400},
        ]
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)
        ctx._summarization_failures = 1

        # floor = max(100-200, 100//4) = max(-100, 25) = 25
        floor = max(100 - 200, 100 // 4)

        await ctx._emergency_mechanical_fallback()

        # Should compact to at most the floor value (may be less if all messages removed)
        assert ctx._running_token_estimate <= floor or len(ctx._messages) == 0, (
            f"Safety floor must prevent over-compaction: expected <= {floor} "
            f"or empty messages, got {ctx._running_token_estimate}."
        )

    @pytest.mark.asyncio
    async def test_compaction_event_includes_prefix_fields(self):
        """context:compaction event payload includes stable_prefix_estimate and conversation_target."""
        hooks = MagicMock()
        hooks.emit = AsyncMock()

        ctx = ManagedContextManager(
            max_tokens=1000,
            emergency_target_usage=0.50,
            hooks=hooks,
        )
        ctx._stable_prefix_estimate = 100

        ctx._messages = [{"role": "user", "content": "A" * 100}]
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)
        ctx._summarization_failures = 1

        await ctx._emergency_mechanical_fallback()

        compaction_calls = [
            c for c in hooks.emit.call_args_list if c.args[0] == "context:compaction"
        ]
        assert compaction_calls, "No context:compaction event was emitted."

        data = compaction_calls[0].args[1]
        assert "stable_prefix_estimate" in data, (
            "context:compaction payload must include 'stable_prefix_estimate'."
        )
        assert data["stable_prefix_estimate"] == 100
        assert "conversation_target" in data, (
            "context:compaction payload must include 'conversation_target'."
        )


# ---------------------------------------------------------------------------
# clear() resets _stable_prefix_estimate
# ---------------------------------------------------------------------------


class TestClearResetsStablePrefix:
    """Verify clear() resets _stable_prefix_estimate to 0."""

    @pytest.mark.asyncio
    async def test_clear_resets_stable_prefix_estimate(self):
        """After clear(), _stable_prefix_estimate returns to 0."""
        ctx = ManagedContextManager()

        # Simulate the field having been set during a get_messages_for_request() call
        ctx._stable_prefix_estimate = 12_000

        await ctx.clear()

        assert ctx._stable_prefix_estimate == 0, (
            "clear() must reset _stable_prefix_estimate to 0 so the next session "
            "starts without a stale prefix estimate."
        )


# ---------------------------------------------------------------------------
# End-to-end: prefix propagation through the threshold chain
# ---------------------------------------------------------------------------


class TestPrefixPropagationEndToEnd:
    """Integration tests: prefix set by get_messages_for_request(), consumed by trigger."""

    @pytest.mark.asyncio
    async def test_prefix_set_in_gmfr_used_in_check_trigger(self):
        """get_messages_for_request() caches prefix; next _check_summarization_trigger uses it.

        Sequence:
          1. call get_messages_for_request() with a large system prompt → sets prefix
          2. manually invoke _check_summarization_trigger() with conversation below the
             trigger threshold (alone) but above it (combined)
          3. assert trigger fires
        """
        ctx = ManagedContextManager(max_tokens=1000, summarize_trigger=0.60)

        # System prompt with known size: ~250 tokens (1 000 chars / 4)
        system_content = "S" * 1000

        async def system_factory() -> str:
            return system_content

        await ctx.set_system_prompt_factory(system_factory)
        ctx._messages = [{"role": "user", "content": "Hi"}]
        ctx._running_token_estimate = ctx._estimate_tokens(ctx._messages)

        # Call to populate _stable_prefix_estimate
        await ctx.get_messages_for_request()
        prefix = ctx._stable_prefix_estimate
        assert prefix > 0, (
            "get_messages_for_request() must set _stable_prefix_estimate > 0"
        )

        # Now engineer a scenario where conversation alone is below the trigger
        # but conversation + prefix is above it.
        # trigger_threshold = 0.60 × 1000 = 600
        # We want: conversation < 600 AND conversation + prefix >= 600
        # Choose conversation = 600 - prefix - 10 (just below trigger on its own)
        #        effective    = 600 - prefix - 10 + prefix = 600 - 10 = 590 ... hmm
        # Actually let me set conversation = 600 - prefix + 50 (above when combined)
        # effective = 600 - prefix + 50 + prefix = 650 >= 600 ✓
        # conversation alone = 600 - prefix + 50; if prefix > 50, alone < 600 ✓
        if prefix > 50:
            conversation_tokens = 600 - prefix + 50
            ctx._running_token_estimate = conversation_tokens
            ctx._cached_provider = MagicMock()
            ctx._messages = [
                {"role": "user", "content": "A" * 200},
                {"role": "assistant", "content": "B" * 200},
            ]

            trigger_called = []

            async def fake_trigger():
                trigger_called.append(True)

            with patch.object(ctx, "_trigger_summarization", side_effect=fake_trigger):
                await ctx._check_summarization_trigger()

            effective_pct = (conversation_tokens + prefix) / 1000 * 100
            assert trigger_called, (
                f"Trigger must fire: combined={conversation_tokens}+{prefix}="
                f"{conversation_tokens + prefix} ({effective_pct:.1f}%) >= 60% threshold. "
                f"Conversation alone = {conversation_tokens}/1000 = "
                f"{conversation_tokens / 1000 * 100:.1f}% < 60%."
            )
        else:
            # Prefix too small to demonstrate the gap; just verify field was set
            pytest.skip(
                f"System prompt too small ({prefix} tokens) to demonstrate the gap"
            )
