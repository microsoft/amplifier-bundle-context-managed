"""
Structural scaffold tests for Task 1: Repository Scaffold.

These tests verify the scaffold was created correctly:
- Required files exist
- Module is importable
- Key classes/attributes are present
- Configuration constants are correct
"""

from pathlib import Path

import pytest

BUNDLE_ROOT = Path(__file__).parent.parent.parent.parent
MODULE_ROOT = BUNDLE_ROOT / "modules" / "context-managed"
TOOL_MODULE_ROOT = BUNDLE_ROOT / "modules" / "tool-transcript"


class TestDirectoryStructure:
    """Verify required directories exist."""

    def test_bundle_root_exists(self):
        assert BUNDLE_ROOT.exists(), f"Bundle root missing: {BUNDLE_ROOT}"

    def test_context_managed_module_dir(self):
        assert (MODULE_ROOT / "amplifier_module_context_managed").exists()

    def test_tool_transcript_module_dir(self):
        assert (TOOL_MODULE_ROOT / "amplifier_module_tool_transcript").exists()

    def test_context_dir(self):
        assert (BUNDLE_ROOT / "context").exists()

    def test_docs_dir(self):
        assert (BUNDLE_ROOT / "docs").exists()

    def test_tests_dir(self):
        assert (MODULE_ROOT / "tests").exists()


class TestRequiredFilesExist:
    """Verify all required files are present."""

    def test_bundle_md(self):
        assert (BUNDLE_ROOT / "bundle.md").exists()

    def test_context_managed_pyproject(self):
        assert (MODULE_ROOT / "pyproject.toml").exists()

    def test_context_managed_init(self):
        assert (
            MODULE_ROOT / "amplifier_module_context_managed" / "__init__.py"
        ).exists()

    def test_tool_transcript_pyproject(self):
        assert (TOOL_MODULE_ROOT / "pyproject.toml").exists()

    def test_tool_transcript_init(self):
        assert (
            TOOL_MODULE_ROOT / "amplifier_module_tool_transcript" / "__init__.py"
        ).exists()

    def test_summary_instructions(self):
        assert (BUNDLE_ROOT / "context" / "summary-instructions.md").exists()

    def test_configuration_docs(self):
        assert (BUNDLE_ROOT / "docs" / "CONFIGURATION.md").exists()


class TestBundleMdContent:
    """Verify bundle.md has required content."""

    @pytest.fixture
    def bundle_md(self):
        return (BUNDLE_ROOT / "bundle.md").read_text()

    def test_has_bundle_name(self, bundle_md):
        assert "context-managed" in bundle_md

    def test_has_version(self, bundle_md):
        assert "0.1.0" in bundle_md

    def test_references_context_module(self, bundle_md):
        assert "context-managed" in bundle_md

    def test_references_transcript_tool(self, bundle_md):
        assert "tool-transcript" in bundle_md or "transcript" in bundle_md

    def test_references_context_instructions(self, bundle_md):
        assert "summary-instructions.md" in bundle_md


class TestPyprojectContent:
    """Verify pyproject.toml files have required content."""

    @pytest.fixture
    def context_pyproject(self):
        return (MODULE_ROOT / "pyproject.toml").read_text()

    @pytest.fixture
    def transcript_pyproject(self):
        return (TOOL_MODULE_ROOT / "pyproject.toml").read_text()

    def test_context_package_name(self, context_pyproject):
        assert "amplifier-module-context-managed" in context_pyproject

    def test_context_entry_point(self, context_pyproject):
        assert (
            'context-managed = "amplifier_module_context_managed:mount"'
            in context_pyproject
        )

    def test_context_hatchling(self, context_pyproject):
        assert "hatchling" in context_pyproject

    def test_context_amplifier_core_source(self, context_pyproject):
        assert "amplifier-core" in context_pyproject

    def test_transcript_package_name(self, transcript_pyproject):
        assert "amplifier-module-tool-transcript" in transcript_pyproject

    def test_transcript_entry_point(self, transcript_pyproject):
        assert (
            'tool-transcript = "amplifier_module_tool_transcript:mount"'
            in transcript_pyproject
        )


class TestContextManagedModule:
    """Verify the context-managed __init__.py has correct content."""

    def test_module_importable(self):
        """Module can be imported."""
        import amplifier_module_context_managed  # noqa: F401

    def test_module_type_attribute(self):
        """Has __amplifier_module_type__ = 'context'."""
        import amplifier_module_context_managed as m

        assert m.__amplifier_module_type__ == "context"

    def test_no_legacy_writer_exports(self):
        import amplifier_module_context_managed as module

        for name in ("TRANSCRIPT_FORMAT_VERSION", "ManagedContextManager", "SummaryTier", "SummaryResult"):
            assert not hasattr(module, name)

    def test_mount_function_exists(self):
        """Has async mount() function."""
        import amplifier_module_context_managed as m

        assert callable(m.mount)

    def test_managed_context_manager_class(self):
        """The concrete manager lives in the Boundary module, not a Legacy alias."""
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        assert BoundaryContextManager is not None

    def test_class_has_add_message(self):
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        assert hasattr(BoundaryContextManager, "add_message")

    def test_class_has_get_messages_for_request(self):
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        assert hasattr(BoundaryContextManager, "get_messages_for_request")

    def test_class_has_get_messages(self):
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        assert hasattr(BoundaryContextManager, "get_messages")

    def test_class_has_set_messages(self):
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        assert hasattr(BoundaryContextManager, "set_messages")

    def test_class_has_set_system_prompt_factory(self):
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        assert hasattr(BoundaryContextManager, "set_system_prompt_factory")

    def test_class_has_clear(self):
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        assert hasattr(BoundaryContextManager, "clear")

    def test_class_has_no_private_transcript_path(self):
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        assert not hasattr(BoundaryContextManager(), "transcript_path")

    def test_class_has_no_private_tool_results_dir(self):
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        assert not hasattr(BoundaryContextManager(), "tool_results_dir")

    def test_default_config_values(self):
        """Boundary uses provider counting by default, with no Legacy policies."""
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        ctx = BoundaryContextManager()
        assert ctx.config == {"token_meter": "actual"}
        assert ctx.summary is None
        assert ctx.messages == []


class TestToolTranscriptModule:
    """Verify the tool-transcript __init__.py has correct content."""

    def test_tool_init_content(self):
        """Tool module declares its type and mount entry point."""
        content = (
            TOOL_MODULE_ROOT / "amplifier_module_tool_transcript" / "__init__.py"
        ).read_text()
        assert "__amplifier_module_type__" in content
        assert "tool" in content
        assert "mount" in content

    def test_mount_is_async(self):
        """mount() is defined as async."""
        content = (
            TOOL_MODULE_ROOT / "amplifier_module_tool_transcript" / "__init__.py"
        ).read_text()
        assert "async def mount" in content


class TestContextInstructions:
    """Verify summary-instructions.md has 5 principles."""

    @pytest.fixture
    def instructions(self):
        return (BUNDLE_ROOT / "context" / "summary-instructions.md").read_text()

    def test_has_content(self, instructions):
        assert len(instructions) > 100

    def test_has_five_numbered_principles(self, instructions):
        """Has at least 5 numbered items."""
        import re

        # Match numbered items like "1.", "2.", etc.
        numbered = re.findall(r"^\s*\d+\.", instructions, re.MULTILINE)
        assert len(numbered) >= 5, (
            f"Expected 5 numbered principles, found {len(numbered)}"
        )

    def test_mentions_summaries(self, instructions):
        assert "summar" in instructions.lower()

    def test_mentions_transcript_tool(self, instructions):
        assert "read_transcript" in instructions or "transcript" in instructions.lower()

    def test_mentions_verify(self, instructions):
        assert "verify" in instructions.lower() or "verif" in instructions.lower()


class TestConfigurationDocs:
    """Verify CONFIGURATION.md has required parameter table."""

    @pytest.fixture
    def config_doc(self):
        return (BUNDLE_ROOT / "docs" / "CONFIGURATION.md").read_text()

    def test_has_max_tokens(self, config_doc):
        assert "max_tokens" in config_doc
        assert "200" in config_doc  # 200,000

    def test_has_verbatim_window_tokens(self, config_doc):
        assert "verbatim_window_tokens" in config_doc

    def test_has_max_summary_tiers(self, config_doc):
        assert "max_summary_tiers" in config_doc

    def test_has_summary_target_tokens(self, config_doc):
        assert "summary_target_tokens" in config_doc

    def test_has_summarization_model(self, config_doc):
        assert "summarization_model" in config_doc

    def test_has_summarization_prompt_path(self, config_doc):
        assert "summarization_prompt_path" in config_doc

    def test_has_summarization_retries_before_fallback(self, config_doc):
        assert "summarization_retries_before_fallback" in config_doc

    def test_has_emergency_target_usage(self, config_doc):
        assert "emergency_target_usage" in config_doc

    def test_has_large_result_threshold(self, config_doc):
        assert "large_result_threshold" in config_doc

    def test_has_pressure_warning(self, config_doc):
        assert "pressure_warning" in config_doc

    def test_has_summarize_trigger(self, config_doc):
        assert "summarize_trigger" in config_doc

    def test_has_emergency_fallback(self, config_doc):
        assert "emergency_fallback" in config_doc

    def test_has_rate_limit_per_turn(self, config_doc):
        assert "rate_limit_per_turn" in config_doc
