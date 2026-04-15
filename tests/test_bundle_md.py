"""Tests for bundle.md frontmatter and content requirements.

Acceptance criteria:
1. bundle.md contains YAML frontmatter with both modules declared
2. Module ordering note present (context-managed before tool-transcript)
3. All three component subsections present
"""

from pathlib import Path

import yaml

BUNDLE_MD = Path(__file__).parent.parent / "bundle.md"


def parse_bundle_md():
    """Parse bundle.md into frontmatter dict and body string."""
    content = BUNDLE_MD.read_text()
    assert content.startswith("---"), "bundle.md must start with YAML frontmatter"
    # Split out frontmatter
    parts = content.split("---", 2)
    # parts[0] is empty, parts[1] is YAML, parts[2] is body
    assert len(parts) == 3, "bundle.md must have opening and closing --- for frontmatter"
    frontmatter = yaml.safe_load(parts[1])
    body = parts[2]
    return frontmatter, body


class TestFrontmatter:
    def test_bundle_name(self):
        fm, _ = parse_bundle_md()
        assert fm["bundle"]["name"] == "context-managed"

    def test_bundle_version(self):
        fm, _ = parse_bundle_md()
        assert fm["bundle"]["version"] == "0.1.0"

    def test_bundle_description_mentions_rolling_context(self):
        fm, _ = parse_bundle_md()
        desc = fm["bundle"]["description"].lower()
        assert "rolling" in desc or "summarization" in desc or "context" in desc

    def test_modules_array_present(self):
        """Acceptance criterion 1: modules array declared in frontmatter."""
        fm, _ = parse_bundle_md()
        assert "modules" in fm, "frontmatter must contain a 'modules' array"
        assert isinstance(fm["modules"], list), "'modules' must be a list"
        assert len(fm["modules"]) == 2, "must declare exactly 2 modules"

    def test_context_managed_module_declared(self):
        """Acceptance criterion 1: context-managed module declared."""
        fm, _ = parse_bundle_md()
        modules = fm["modules"]
        names = [m.get("name") for m in modules]
        assert "context-managed" in names, "modules must include 'context-managed'"

    def test_context_managed_module_type_and_path(self):
        """context-managed module has correct type and path."""
        fm, _ = parse_bundle_md()
        cm_module = next(m for m in fm["modules"] if m.get("name") == "context-managed")
        assert cm_module["path"] == "modules/context-managed"
        assert cm_module["type"] == "context"

    def test_tool_transcript_module_declared(self):
        """Acceptance criterion 1: tool-transcript module declared."""
        fm, _ = parse_bundle_md()
        modules = fm["modules"]
        names = [m.get("name") for m in modules]
        assert "tool-transcript" in names, "modules must include 'tool-transcript'"

    def test_tool_transcript_module_type_and_path(self):
        """tool-transcript module has correct type and path."""
        fm, _ = parse_bundle_md()
        tt_module = next(m for m in fm["modules"] if m.get("name") == "tool-transcript")
        assert tt_module["path"] == "modules/tool-transcript"
        assert tt_module["type"] == "tool"

    def test_tool_transcript_rate_limit_config(self):
        """tool-transcript module has rate_limit_per_turn: 3 in config."""
        fm, _ = parse_bundle_md()
        tt_module = next(m for m in fm["modules"] if m.get("name") == "tool-transcript")
        assert "config" in tt_module, "tool-transcript must have a 'config' section"
        assert tt_module["config"].get("rate_limit_per_turn") == 3

    def test_modules_order_context_managed_first(self):
        """Acceptance criterion 2: context-managed declared before tool-transcript."""
        fm, _ = parse_bundle_md()
        modules = fm["modules"]
        names = [m.get("name") for m in modules]
        assert names.index("context-managed") < names.index("tool-transcript"), (
            "context-managed must appear before tool-transcript in modules array"
        )

    def test_empty_includes_array(self):
        fm, _ = parse_bundle_md()
        assert "includes" in fm
        assert fm["includes"] == [] or fm["includes"] is None


class TestBodyContent:
    def test_heading_present(self):
        _, body = parse_bundle_md()
        assert "# Context Managed Bundle" in body

    def test_context_manager_module_section(self):
        """Acceptance criterion 3: Context Manager Module section present."""
        _, body = parse_bundle_md()
        assert "Context Manager Module" in body

    def test_context_manager_implements_protocol(self):
        """Context Manager Module mentions ContextManager protocol."""
        _, body = parse_bundle_md()
        assert "ContextManager" in body

    def test_context_manager_mounting_order_note(self):
        """Acceptance criterion 2: module ordering note present."""
        _, body = parse_bundle_md()
        # Must mention that context-managed must be mounted before transcript tool
        body_lower = body.lower()
        assert "before" in body_lower and ("transcript" in body_lower or "tool" in body_lower), (
            "body must mention mounting order (context-managed before transcript tool)"
        )

    def test_context_manager_registers_context_transcript_path(self):
        """Context Manager Module section mentions context_transcript_path."""
        _, body = parse_bundle_md()
        assert "context_transcript_path" in body

    def test_transcript_tool_section(self):
        """Acceptance criterion 3: Transcript Tool section present."""
        _, body = parse_bundle_md()
        assert "Transcript Tool" in body

    def test_transcript_tool_full_fidelity(self):
        """Transcript Tool mentions full-fidelity past messages."""
        _, body = parse_bundle_md()
        body_lower = body.lower()
        assert "full-fidelity" in body_lower or "full fidelity" in body_lower

    def test_context_instructions_section(self):
        """Acceptance criterion 3: Context Instructions section present."""
        _, body = parse_bundle_md()
        assert "Context Instructions" in body

    def test_context_instructions_reference(self):
        """Context Instructions mentions the @context-managed:context/summary-instructions.md reference."""
        _, body = parse_bundle_md()
        assert "@context-managed:context/summary-instructions.md" in body
