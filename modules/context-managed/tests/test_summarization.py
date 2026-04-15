"""
Tests for Phase 2 summarization dataclasses: SummaryResult and SummaryTier.

Task 1: Add SummaryResult and SummaryTier dataclasses.
Task 2: Default summarization prompt and _get_summarization_prompt().
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
