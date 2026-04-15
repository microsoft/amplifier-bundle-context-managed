"""Tests for context/summary-instructions.md content requirements.

Acceptance criteria:
1. context/summary-instructions.md reviewed
2. All 5 principles confirmed present
3. No internal architecture details exposed (tiers, JSONL, summarization mechanics)
4. search parameter mentioned in principle 2 (added if missing)
"""

from pathlib import Path

SUMMARY_INSTRUCTIONS = Path(__file__).parent.parent / "context" / "summary-instructions.md"


def read_content() -> str:
    """Read the summary-instructions.md file."""
    assert SUMMARY_INSTRUCTIONS.exists(), f"File not found: {SUMMARY_INSTRUCTIONS}"
    return SUMMARY_INSTRUCTIONS.read_text()


class TestAllFivePrinciplesPresent:
    def test_principle_1_summaries_are_hints(self):
        """Principle 1: Summaries are hints, not truth."""
        content = read_content()
        assert "hints" in content.lower() and "truth" in content.lower(), (
            "Principle 1 must state that summaries are hints, not truth"
        )

    def test_principle_2_how_to_use_transcript_tool(self):
        """Principle 2: How to use the transcript tool with read_transcript(start_turn=N, end_turn=M)."""
        content = read_content()
        assert "read_transcript" in content, (
            "Principle 2 must mention the read_transcript tool"
        )
        assert "start_turn" in content and "end_turn" in content, (
            "Principle 2 must show read_transcript(start_turn=N, end_turn=M) usage"
        )

    def test_principle_3_gradient_is_intentional(self):
        """Principle 3: The gradient is intentional."""
        content = read_content()
        assert "gradient" in content.lower(), (
            "Principle 3 must mention 'the gradient is intentional'"
        )

    def test_principle_4_memory_is_not_the_codebase(self):
        """Principle 4: Memory is not the codebase."""
        content = read_content()
        assert "memory" in content.lower() and "codebase" in content.lower(), (
            "Principle 4 must state that memory is not the codebase"
        )

    def test_principle_5_budget_awareness(self):
        """Principle 5: Budget awareness."""
        content = read_content()
        assert "budget" in content.lower(), (
            "Principle 5 must mention budget awareness"
        )


class TestNoInternalArchitecture:
    def test_no_tiers_mentioned(self):
        """File must not expose internal tier architecture."""
        content = read_content()
        # "tiers" as an internal architecture concept (e.g., "compression tiers")
        # We check for "tiers" only in a context suggesting internal architecture
        # Simple presence check — the word "tiers" should not appear
        assert "tier" not in content.lower(), (
            "File must not expose internal tier architecture"
        )

    def test_no_jsonl_mentioned(self):
        """File must not expose JSONL internal storage details."""
        content = read_content()
        assert "jsonl" not in content.lower(), (
            "File must not expose JSONL internal storage details"
        )

    def test_no_summarization_mechanics_mentioned(self):
        """File must not expose internal summarization mechanics."""
        content = read_content()
        # Should not describe the internal mechanism of how summarization works
        assert "summarization mechanic" not in content.lower(), (
            "File must not expose internal summarization mechanics"
        )


class TestSearchParameterInPrinciple2:
    def test_search_parameter_mentioned(self):
        """Principle 2 must mention the search parameter for targeted content retrieval."""
        content = read_content()
        assert "search" in content, (
            "Principle 2 must mention the `search` parameter to find specific content "
            "without reading entire ranges"
        )

    def test_search_parameter_in_principle_2_section(self):
        """The search parameter mention should be in or near principle 2."""
        content = read_content()
        lines = content.splitlines()
        # Find the line with principle 2
        p2_idx = None
        for i, line in enumerate(lines):
            if "How to use the transcript tool" in line:
                p2_idx = i
                break
        assert p2_idx is not None, "Principle 2 heading not found"

        # The search parameter should appear somewhere in the document
        # (near principle 2, but within a few lines)
        nearby_text = "\n".join(lines[p2_idx : p2_idx + 5])
        assert "search" in nearby_text, (
            "The `search` parameter should be mentioned in principle 2 "
            "(within its text block)"
        )
