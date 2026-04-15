"""
Tests for Phase 2 summarization dataclasses: SummaryResult and SummaryTier.

Task 1: Add SummaryResult and SummaryTier dataclasses.
"""


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
