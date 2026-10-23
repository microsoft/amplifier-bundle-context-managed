"""Public mount validation and host-owned Boundary integration."""

import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import amplifier_module_context_managed as module
import pytest
from amplifier_module_context_managed.boundary import BoundaryContextManager


def coordinator():
    capabilities = {}
    return SimpleNamespace(
        mount=AsyncMock(), hooks=None, session_id="fixture-session",
        get_capability=Mock(side_effect=capabilities.get),
        register_capability=Mock(side_effect=capabilities.__setitem__),
        register_contributor=Mock(),
    ), capabilities


def test_legacy_only_option_set_matches_retired_contract():
    assert module.LEGACY_ONLY_OPTIONS == {
        "verbatim_window_tokens", "max_summary_tiers", "summarization_prompt_path",
        "summarization_retries_before_fallback", "emergency_target_usage",
        "large_result_threshold", "pressure_warning", "emergency_fallback",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [None, {}, {"engine": "boundary"}])
async def test_default_and_explicit_mount_are_boundary(config):
    host, capabilities = coordinator()
    await module.mount(host, config)
    host.mount.assert_awaited_once()
    name, context = host.mount.call_args.args
    assert name == "context" and isinstance(context, BoundaryContextManager)
    assert context.config == {"engine": "boundary", "token_meter": "actual"}
    assert capabilities["context.history_authority"] == "host"
    assert await capabilities["context.history"]() == []
    assert capabilities["context.request_retention"] == context.get_messages_for_request_retaining
    assert capabilities["context.measured_request_view"] == context.get_measured_request_view
    assert "context-managed.transcript_path" not in capabilities


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [None, {}, {"engine": "boundary"}])
async def test_default_configuration_hash_matches_old_explicit_boundary(config):
    from amplifier_module_context_managed.boundary import SUMMARY_PROMPT
    from amplifier_module_context_managed.checkpoint import digest

    host, _ = coordinator()
    await module.mount(host, config)
    mounted = host.mount.call_args.args[1]
    # Pre-retirement explicit mounts passed engine through unchanged. token_meter
    # is inserted by Boundary and excluded by its existing v2 compatibility rule.
    assert mounted.checkpoint_configuration() == digest({
        "config": {"engine": "boundary"}, "prompt": SUMMARY_PROMPT})


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["legacy", "unknown", "Boundary", "", None, False, 1, [], {}])
async def test_invalid_engine_fails_before_coordinator_access_or_mutation(engine):
    host, capabilities = coordinator()
    config = {"engine": engine}
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="only engine: boundary") as caught:
        await module.mount(host, config)
    assert "Remove" in str(caught.value) and "archive" in str(caught.value)
    host.mount.assert_not_called()
    host.get_capability.assert_not_called()
    host.register_capability.assert_not_called()
    host.register_contributor.assert_not_called()
    assert capabilities == {} and config == before


@pytest.mark.asyncio
async def test_engine_requires_exact_string_not_subclass():
    class Engine(str):
        pass
    host, _ = coordinator()
    with pytest.raises(ValueError, match="only engine: boundary"):
        await module.mount(host, {"engine": Engine("boundary")})
    host.mount.assert_not_called()
    host.register_capability.assert_not_called()
    host.register_contributor.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("option", sorted(module.LEGACY_ONLY_OPTIONS))
@pytest.mark.parametrize("value", [None, False, 0, "configured"])
async def test_legacy_only_option_rejected_even_when_false_or_null(option, value):
    host, capabilities = coordinator()
    with pytest.raises(ValueError, match="Remove Legacy-only") as caught:
        await module.mount(host, {option: value})
    assert option in str(caught.value) and "CONFIGURATION.md" in str(caught.value)
    host.mount.assert_not_called()
    host.get_capability.assert_not_called()
    host.register_capability.assert_not_called()
    host.register_contributor.assert_not_called()
    assert capabilities == {}


@pytest.mark.asyncio
async def test_rejection_lists_all_legacy_options_without_echoing_values():
    host, _ = coordinator()
    with pytest.raises(ValueError) as caught:
        await module.mount(host, dict.fromkeys(module.LEGACY_ONLY_OPTIONS, "private-value"))
    assert all(option in str(caught.value) for option in module.LEGACY_ONLY_OPTIONS)
    assert "private-value" not in str(caught.value)


@pytest.mark.asyncio
async def test_supported_config_is_copied_and_passed_without_reinterpretation():
    host, _ = coordinator()
    config = {
        "max_tokens": 100000, "summarize_trigger": .6, "summary_target_tokens": 37,
        "summary_max_output_tokens": 9000, "native_compaction": False,
        "summary_max_source_chars": 10000, "summary_max_calls": 8,
        "summary_reasoning_effort": "low", "summarization_model": "fixture",
        "token_meter": "estimate", "durable_checkpoints": True,
        "summary_timeout": .001, "summary_total_work_timeout": .001,
        "native_compaction_timeout": .001,
    }
    before = copy.deepcopy(config)
    await module.mount(host, config)
    context = host.mount.call_args.args[1]
    assert config == before and context.config is not config
    ignored = {"summary_timeout", "summary_total_work_timeout", "native_compaction_timeout"}
    assert context.config == {**{k: v for k, v in config.items() if k not in ignored}, "engine": "boundary"}
    assert context.config["summary_target_tokens"] == 37
    assert context.config["summary_max_output_tokens"] == 9000


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [None, 9000])
async def test_summary_target_tokens_is_not_remapped_to_portable_output(output):
    from amplifier_module_context_managed.summary import summary_request

    host, _ = coordinator()
    config = {"summary_target_tokens": 37}
    if output is not None:
        config["summary_max_output_tokens"] = output
    await module.mount(host, config)
    context = host.mount.call_args.args[1]
    request = summary_request("Instructions", "Conversation data", "", context.config)
    assert request.max_output_tokens == (8192 if output is None else output)


@pytest.mark.asyncio
async def test_mount_contributes_boundary_events_without_replacing_other_events():
    host, capabilities = coordinator()
    existing = capabilities["observability.events"] = ["kernel:turn_start"]
    await module.mount(host)
    assert capabilities["observability.events"] is existing
    name, owner, events = host.register_contributor.call_args.args
    assert (name, owner) == ("observability.events", "context-managed")
    assert events() == ["context:budget_exceeded", "context:compaction_started",
                        "context:compaction_finished", "context:compaction_progress",
                        "context:checkpoint_progress_rejected"]


@pytest.mark.asyncio
async def test_no_private_transcript_io_on_mount_ingress_request_or_clear(monkeypatch):
    host, capabilities = coordinator()
    # Trap path discovery as well as reads/writes, including an existing session ID.
    with monkeypatch.context() as guard:
        def forbidden(*args, **kwargs):
            raise AssertionError("Boundary must not discover, read or write a private transcript")
        guard.setattr(Path, "home", forbidden)
        guard.setattr(Path, "cwd", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(Path, "mkdir", forbidden)
        guard.setattr("builtins.open", forbidden)
        await module.mount(host)
        context = host.mount.call_args.args[1]
        original = [{"role": "user", "content": "Host restored history", "metadata": {"custom": [1]}}]
        await context.set_messages(original)
        await context.add_message({"role": "assistant", "content": "Acknowledged"})
        assert (await capabilities["context.history"]())[0] == original[0]
        assert await context.get_messages_for_request()
        await context.clear()
        assert await context.get_messages() == []


def test_package_import_has_no_legacy_executor_or_estimator_import():
    result = subprocess.run([sys.executable, "-c", """
import sys
import amplifier_module_context_managed as module
assert not hasattr(module, 'ManagedContextManager')
assert not hasattr(module, 'SummaryTier')
assert not hasattr(module, 'SummaryResult')
assert 'amplifier_module_context_managed._text_estimate' not in sys.modules
assert 'amplifier_module_context_managed.boundary' not in sys.modules
"""], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr