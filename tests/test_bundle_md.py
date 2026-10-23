"""Guard the shipped host composition and its Boundary-only context settings."""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tomllib
import yaml

BUNDLE_ROOT = Path(__file__).parent.parent
BUNDLE_MD = BUNDLE_ROOT / "bundle.md"


def parse_bundle_md():
    content = BUNDLE_MD.read_text()
    assert content.startswith("---"), "bundle.md must start with YAML frontmatter"
    parts = content.split("---", 2)
    assert len(parts) == 3, "bundle.md must have opening and closing frontmatter delimiters"
    return yaml.safe_load(parts[1]), parts[2]


class TestFrontmatter:
    def test_bundle_name(self):
        fm, _ = parse_bundle_md()
        assert fm["bundle"]["name"] == "context-managed"

    def test_bundle_version(self):
        fm, _ = parse_bundle_md()
        assert fm["bundle"]["version"] == "0.1.0"

    def test_bundle_description_mentions_boundary(self):
        fm, _ = parse_bundle_md()
        description = fm["bundle"]["description"].lower()
        assert "boundary" in description and "host-owned" in description
        assert "persistent transcript" not in description

    def test_production_session_structure(self):
        fm, _ = parse_bundle_md()
        assert "modules" not in fm  # The shipped bundle uses session + tools.
        assert fm["session"]["raw"] is True
        assert fm["session"]["orchestrator"] == {
            "module": "loop-streaming",
            "source": "git+https://github.com/microsoft/amplifier-module-loop-streaming@main",
            "config": {"extended_thinking": True},
        }

    def test_context_managed_module_declared(self):
        fm, _ = parse_bundle_md()
        assert fm["session"]["context"]["module"] == "context-managed"

    def test_context_managed_package_entrypoint_and_path(self):
        path = BUNDLE_ROOT / "modules/context-managed"
        project = tomllib.loads((path / "pyproject.toml").read_text())
        assert project["project"]["entry-points"]["amplifier.modules"]["context-managed"] == (
            "amplifier_module_context_managed:mount")
        assert (path / "amplifier_module_context_managed/__init__.py").is_file()

    def test_tool_transcript_module_declared(self):
        fm, _ = parse_bundle_md()
        assert [tool["module"] for tool in fm["tools"]] == [
            "tool-filesystem", "tool-bash", "tool-web", "tool-search", "tool-transcript"]

    def test_tool_transcript_package_entrypoint_and_path(self):
        path = BUNDLE_ROOT / "modules/tool-transcript"
        project = tomllib.loads((path / "pyproject.toml").read_text())
        assert project["project"]["entry-points"]["amplifier.modules"]["tool-transcript"] == (
            "amplifier_module_tool_transcript:mount")
        assert (path / "amplifier_module_tool_transcript/__init__.py").is_file()

    def test_tool_transcript_rate_limit_config(self):
        fm, _ = parse_bundle_md()
        transcript = next(tool for tool in fm["tools"] if tool["module"] == "tool-transcript")
        assert transcript["config"] == {"rate_limit_per_turn": 3}

    def test_boundary_configuration(self):
        from amplifier_module_context_managed import LEGACY_ONLY_OPTIONS

        fm, _ = parse_bundle_md()
        config = fm["session"]["context"]["config"]
        assert config == {"engine": "boundary", "max_tokens": 200000,
                          "summarize_trigger": .6, "summary_target_tokens": 1500}
        assert not LEGACY_ONLY_OPTIONS.intersection(config)

    def test_host_includes_preserved(self):
        fm, _ = parse_bundle_md()
        assert [item["bundle"] for item in fm["includes"]] == [
            "git+https://github.com/microsoft/amplifier@main#subdirectory=behaviors/amplifier-expert.yaml",
            "git+https://github.com/microsoft/amplifier-core@main#subdirectory=behaviors/core-expert.yaml",
            "foundation:behaviors/foundation-expert", "foundation:behaviors/sessions",
            "foundation:behaviors/status-context", "foundation:behaviors/redaction",
            "foundation:behaviors/todo-reminder", "foundation:behaviors/streaming-ui",
            "foundation:behaviors/agents",
            "git+https://github.com/microsoft/amplifier-bundle-recipes@main#subdirectory=behaviors/recipes.yaml",
            "git+https://github.com/microsoft/amplifier-bundle-design-intelligence@main#subdirectory=behaviors/design-intelligence.yaml",
            "git+https://github.com/microsoft/amplifier-bundle-python-dev@main",
            "git+https://github.com/microsoft/amplifier-bundle-amplifier-tester@main",
            "git+https://github.com/microsoft/amplifier-bundle-skills@main#subdirectory=behaviors/skills.yaml",
            "git+https://github.com/microsoft/amplifier-bundle-browser-tester@main#subdirectory=behaviors/browser-tester.yaml",
            "git+https://github.com/microsoft/amplifier-bundle-superpowers@main#subdirectory=behaviors/superpowers-methodology.yaml",
            "git+https://github.com/microsoft/amplifier-module-hook-shell@main#subdirectory=behaviors/hook-shell.yaml",
            "git+https://github.com/microsoft/amplifier-module-tool-mcp@main#subdirectory=behaviors/mcp.yaml",
            "git+https://github.com/microsoft/amplifier-bundle-filesystem@main#subdirectory=behaviors/apply-patch.yaml",
            "git+https://github.com/microsoft/amplifier-bundle-routing-matrix@main",
        ]
        assert fm["agents"]["include"] == [
            "foundation:bug-hunter", "foundation:explorer", "foundation:file-ops",
            "foundation:git-ops", "foundation:integration-specialist", "foundation:modular-builder",
            "foundation:post-task-cleanup", "foundation:security-guardian", "foundation:test-coverage",
            "foundation:web-research", "foundation:zen-architect"]

    @pytest.mark.asyncio
    async def test_loaded_bundle_configuration_mounts_boundary(self):
        from amplifier_module_context_managed import mount
        from amplifier_module_context_managed.boundary import BoundaryContextManager

        fm, _ = parse_bundle_md()
        config = fm["session"]["context"]["config"]
        original = dict(config)
        capabilities = {}
        coordinator = SimpleNamespace(
            mount=AsyncMock(), hooks=None, get_capability=capabilities.get,
            register_capability=capabilities.__setitem__, register_contributor=Mock())
        await mount(coordinator, config)
        context = coordinator.mount.call_args.args[1]
        assert isinstance(context, BoundaryContextManager)
        assert context.config == {**original, "token_meter": "actual"}
        assert config == original
        assert capabilities["context.history_authority"] == "host"


class TestBodyContent:
    def test_heading_present(self):
        _, body = parse_bundle_md()
        assert "# Context Managed Bundle" in body

    def test_context_manager_module_section(self):
        _, body = parse_bundle_md()
        assert "Context Manager Module" in body

    def test_context_manager_implements_protocol(self):
        _, body = parse_bundle_md()
        assert "ContextManager" in body

    def test_context_manager_mounting_order_note(self):
        _, body = parse_bundle_md()
        assert "Mount it before" in body and "`tool-transcript`" in body

    def test_context_manager_advertises_host_history_not_private_writer(self):
        _, body = parse_bundle_md()
        assert "context.history" in body and "context.history_authority: host" in body
        assert "not a second" in body and "transcript writer" in body

    def test_transcript_tool_section(self):
        _, body = parse_bundle_md()
        assert "Transcript Tool" in body

    def test_transcript_tool_distinguishes_formatted_history_and_bounded_raw_archives(self):
        _, body = parse_bundle_md()
        retrieval = body.split("## Transcript Tool", 1)[1].split("## Context Instructions", 1)[0]
        assert "formatted host canonical history" in retrieval
        assert "arguments can be truncated" in retrieval
        assert "does not promise full-fidelity ordinary retrieval" in retrieval
        assert "Only explicit archives return complete raw records" in retrieval
        assert "within page limits" in retrieval and "oversized single records fail visibly" in retrieval
        assert "retrieves full-fidelity messages" not in retrieval
        assert "read-only" in body and "host-approved archive IDs" in body
        assert "does not provide universal durability" in body

    def test_context_instructions_section(self):
        _, body = parse_bundle_md()
        assert "Context Instructions" in body

    def test_context_instructions_reference(self):
        _, body = parse_bundle_md()
        assert "@context-managed:context/summary-instructions.md" in body
        assert "@foundation:context/shared/common-system-base.md" in body


class TestCurrentDocumentation:
    def test_archive_contract_links_target_existing_configuration_section(self):
        target = BUNDLE_ROOT / "docs/CONFIGURATION.md"
        assert "### Legacy archive usage" in target.read_text()
        for filename in ("README.md", "bundle.md", "docs/BOUNDARY_ENGINE.md",
                         "modules/context-managed/README.md"):
            document = BUNDLE_ROOT / filename
            links = re.findall(r"\]\(([^)]+)\)", document.read_text())
            archive_links = [link for link in links if link.endswith("#legacy-archive-usage")]
            assert archive_links, filename
            for link in archive_links:
                assert (document.parent / link.split("#", 1)[0]).resolve() == target.resolve()
            assert "tool-transcript/README.md" not in document.read_text()

    def test_archive_example_and_bounds_match_current_reader(self):
        from amplifier_module_tool_transcript.legacy import (
            DEFAULT_MAX_BYTES,
            DEFAULT_MAX_RECORDS,
            MAX_BYTES,
            MAX_RECORDS,
            ArchiveSpec,
        )

        content = (BUNDLE_ROOT / "docs/CONFIGURATION.md").read_text()
        example = yaml.safe_load(content.split("```yaml\n", 1)[1].split("```", 1)[0])
        config = example["tools"][0]["config"]
        assert example["tools"][0]["module"] == "tool-transcript"
        paths = config["legacy_transcript_paths"]
        assert isinstance(paths["saved"], str)
        assert set(paths["relocated"]) == set(ArchiveSpec.__annotations__)
        assert all(isinstance(value, str) for value in paths["relocated"].values())
        assert f"Default\n`max_records` is **{DEFAULT_MAX_RECORDS}**" in content
        assert DEFAULT_MAX_RECORDS == MAX_RECORDS == 100
        assert "also the hard maximum per page" in content
        assert f"**{DEFAULT_MAX_BYTES:,} bytes (64 KiB)**" in content
        assert f"**{MAX_BYTES:,} bytes (1 MiB)**" in content
        assert "including metadata, JSON escaping, and base64 overhead" in content
        assert "Records that still cannot fit" in content
        assert "are not retrievable through this API" in content
        assert "not a promise of\nunlimited full-fidelity retrieval" in content

    def test_archive_generation_and_safety_contract_is_explicit(self):
        content = (BUNDLE_ROOT / "docs/CONFIGURATION.md").read_text()
        for field in ("archive_revision", "result_revision", "result_record_offset",
                      "next_offset_bytes", "archive_changed", "unsupported_safe_open"):
            assert f"`{field}`" in content
        assert "Revision arguments are optional on the first\nread" in content
        assert "Echo both\nvalues on each sidecar continuation" in content
        assert "does not provide an automatic cross-page snapshot" in content
        assert "Tokens fingerprint\nfile stat attributes, not full content" in content
        assert "POSIX descriptor-relative no-follow opens" in content
        assert "explicit destination roots are preflighted before tool\nmounting" in content
        assert "implicit default `tool_results` directory may be absent" in content
        assert "It does not search by basename" in content

    def test_current_summary_call_docs_distinguish_pass_and_cumulative_progress(self):
        for filename in ("docs/CONFIGURATION.md", "modules/context-managed/README.md"):
            content = (BUNDLE_ROOT / filename).read_text()
            assert "per preparation pass" in content
            assert "Cumulative calls across resumed progress are tracked separately" in content
            assert "not a lifetime allowance for an unchanged prefix" in content