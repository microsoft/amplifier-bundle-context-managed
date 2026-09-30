"""Semantic triggers follow the complete request, not transcript byte size."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from amplifier_module_context_managed.boundary import BoundaryContextManager


def history():
    return [
        {"role": "developer", "content": "Preserve local artifacts."},
        {"role": "user", "content": "Investigate the CPU problem."},
        {"role": "assistant", "content": "Earlier research is saved. " * 30},
        {"role": "user", "content": "Check the measurement harness."},
        {"role": "assistant", "content": "Harness evidence is saved."},
        {"role": "user", "content": "Keep working from the saved evidence."},
    ]


def summarizer():
    return SimpleNamespace(name="fixture", default_model="unchanged", complete=AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text=
            "Investigate CPU usage. Preserve local artifacts. Harness evidence is saved.")])) )


def envelope(count, *, measured=True, limit=20000):
    decision = {"estimated_input_tokens": count, "input_limit_tokens": limit}
    if measured:
        decision["measurement"] = {"kind": "provider_count", "source": "fixture.full_request", "input_tokens": count}
    return {"dispatch": object(), "budget_decision": decision}


@pytest.mark.asyncio
async def test_low_actual_input_does_not_summarize_inflated_canonical_rows():
    context = BoundaryContextManager({"max_tokens": 20000, "compaction_notice_enabled": False})
    original = history()
    original[2]["thinking"] = "opaque private transport " * 10000
    await context.set_messages(original)
    provider = summarizer()
    attempts = []
    async def count_view(view):
        attempt = envelope(12000)
        attempts.append(attempt)
        return attempt
    result = await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    provider.complete.assert_not_awaited()
    assert context.summary is None
    assert result["final_attempt"] is attempts[-1]
    assert len(attempts) == result["count_calls"] == 1
    assert context.last_budget["semantic_measurement_kind"] == "provider_count"
    assert context.last_budget["semantic_input_tokens"] == 12000
    assert await context.get_messages() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("measured,kind", [(True, "provider_count"), (False, "provider_estimate")])
async def test_large_complete_request_triggers_even_when_history_text_is_small(measured, kind):
    context = BoundaryContextManager({"max_tokens": 20000, "compaction_notice_enabled": False})
    original = history()
    await context.set_messages(original)
    factory = AsyncMock(return_value="One stable system prompt.")
    await context.set_system_prompt_factory(factory)
    context.active_operations = lambda: [{"job_id": "still-pending", "status": "queued"}]
    provider = summarizer()
    attempts = []
    async def count_view(view):
        # This callback stands for loop assembly: tool schemas and request
        # overlays can be large even when canonical text is very short.
        assert view[0]["content"] == "One stable system prompt."
        assert "still-pending" in str(view)
        count = 2000 if "Continuation note" in str(view) else 16000
        attempt = envelope(count, measured=measured)
        attempts.append(attempt)
        return attempt
    result = await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    provider.complete.assert_awaited_once()
    factory.assert_awaited_once()
    assert context.summary is not None
    assert "Keep working from the saved evidence." in str(result["base_view"])
    assert "Preserve local artifacts." in str(result["base_view"])
    assert result["final_attempt"] is attempts[-1]
    assert len(attempts) == result["count_calls"] == 2
    assert context.last_budget["semantic_measurement_kind"] == kind
    assert context.last_budget["semantic_input_tokens"] == 16000
    assert await context.get_messages() == original


@pytest.mark.asyncio
async def test_fallback_estimate_does_not_serialize_private_duplicate_fields():
    context = BoundaryContextManager({"max_tokens": 20000, "compaction_notice_enabled": False})
    original = history()
    original[2]["thinking"] = "private duplicate " * 10000
    original[2]["metadata"] = {"transport_diagnostics": "opaque " * 10000}
    await context.set_messages(original)
    provider = summarizer()
    await context.get_messages_for_request(provider=provider)
    provider.complete.assert_not_awaited()
    assert context.summary is None
    assert context.last_budget["semantic_measurement_kind"] == "public_estimate"
    assert context.last_budget["semantic_input_tokens"] < 1000
    assert await context.get_messages() == original


@pytest.mark.asyncio
async def test_changed_history_during_measurement_never_commits_summary():
    context = BoundaryContextManager({"max_tokens": 20000})
    await context.set_messages(history())
    provider = summarizer()
    replacement = [{"role": "user", "content": "New authoritative task"}]
    async def count_view(view):
        await context.set_messages(replacement)
        return envelope(16000)
    with pytest.raises(RuntimeError, match="History changed"):
        await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    provider.complete.assert_not_awaited()
    assert context.summary is None
    assert await context.get_messages() == replacement


@pytest.mark.asyncio
async def test_malformed_full_request_budget_is_not_treated_as_zero():
    context = BoundaryContextManager({"max_tokens": 20000})
    await context.set_messages(history())
    provider = summarizer()
    async def count_view(view):
        return {"dispatch": object(), "budget_decision": {"estimated_input_tokens": False, "input_limit_tokens": 20000}}
    with pytest.raises(TypeError, match="nonnegative integer"):
        await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    provider.complete.assert_not_awaited()
    assert context.summary is None


@pytest.mark.asyncio
async def test_note_that_does_not_reduce_measured_input_is_not_committed():
    context = BoundaryContextManager({"max_tokens": 20000, "compaction_notice_enabled": False})
    original = history()
    await context.set_messages(original)
    provider = summarizer()
    async def count_view(view):
        return envelope(16000)
    await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    provider.complete.assert_awaited_once()
    assert context.summary is None
    assert context._summary_progress is None
    assert not context.summary_failure["retryable"]
    assert await context.get_messages() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["configuration", "checkpoint_identity", "provider_model", "summary_provider"])
async def test_contract_change_during_final_measurement_cannot_commit_or_dispatch(change):
    context = BoundaryContextManager({"max_tokens": 20000, "compaction_notice_enabled": False})
    original = history()
    await context.set_messages(original)
    provider = summarizer()
    identity = {"connection": "original", "model": "unchanged"}
    context.checkpoint_identity = lambda: identity
    summary_connection = [provider]
    context.summary_provider = lambda: summary_connection[0]

    async def count_view(view):
        if "Continuation note" not in str(view):
            return envelope(16000)
        # Final preflight can await a remote count while a settings save changes
        # the intended contract. The old note/request must remain unpublished.
        if change == "configuration":
            context.config["summary_target_tokens"] = 2000
        elif change == "checkpoint_identity":
            identity["connection"] = "replacement"
        elif change == "provider_model":
            provider.default_model = "replacement"
        else:
            summary_connection[0] = summarizer()
            summary_connection[0].default_model = "replacement"
        return envelope(2000)

    with pytest.raises(RuntimeError, match="contract changed"):
        await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    provider.complete.assert_awaited_once()
    assert context.summary is None
    assert context._summary_progress is None
    assert not context.is_compacting
    assert await context.get_messages() == original
