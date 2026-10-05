"""Work managed-boundary uses the shared projection before summaries and counts."""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from amplifier_module_context_managed import ManagedContextManager
from amplifier_module_context_managed._text_estimate import estimate_messages
from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.summary import public_messages, source_fragments


ORIGINAL = "PUBLIC_ORIGINAL_SENTINEL" * 10000
HISTORY_ONLY = "HISTORY_ONLY_EXECUTION_SENTINEL" * 10000


def rows(*, pressure=False):
    return [
        {"role": "user", "content": "Investigate execution", "metadata": {
            "amplifier_public_message": {"blocks": [ORIGINAL]}}},
        {"role": "assistant", "content": "Research evidence. " * (1000 if pressure else 1)},
        {"role": "user", "content": "Continue from evidence"},
        {"role": "assistant", "content": "Checkpointed progress"},
        {"role": "user", "content": "Keep the latest correction", "metadata": {
            "amplifier_public_message": {"blocks": [ORIGINAL]},
            "amplifier_public_copy_source": {"source": ORIGINAL},
            "legacy": "unchanged metadata"}},
        {"role": "user", "content": HISTORY_ONLY, "metadata": {
            "amplifier_public_reference_only": True,
            "amplifier_public_message": {"blocks": [ORIGINAL]}}},
    ]


def assert_clean(value):
    text = str(value)
    assert ORIGINAL not in text
    assert HISTORY_ONLY not in text
    assert "amplifier_public_" not in text


def provider():
    async def complete(request):
        assert_clean(request)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=
            "Investigation evidence saved; preserve latest correction.")])
    async def budget(request, **kwargs):
        assert_clean(request)
        return {"estimated_input_tokens": 100, "input_limit_tokens": 6000}
    return SimpleNamespace(name="fixture", default_model="fixture", complete=AsyncMock(side_effect=complete),
                           request_budget=AsyncMock(side_effect=budget))


def test_summary_and_text_estimate_share_projection_without_changing_canonical_rows():
    original = rows()
    before = copy.deepcopy(original)
    projected = public_messages(original)
    assert_clean(projected)
    assert_clean(list(source_fragments(projected, None)))
    expected = copy.deepcopy(original[:-1])
    for row in expected:
        metadata = row.get("metadata", {})
        for key in ("amplifier_public_message", "amplifier_public_copy_source"):
            metadata.pop(key, None)
    assert estimate_messages(original) == estimate_messages(expected)
    assert original == before


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["ordinary", "retaining", "measured"])
@pytest.mark.parametrize("pressure", [False, True])
async def test_work_boundary_all_consumers_exclude_public_history(route, pressure):
    context = BoundaryContextManager({"max_tokens": 6000, "summarize_trigger": 0.2,
        "compaction_notice_enabled": False})
    original = rows(pressure=pressure)
    await context.set_messages(original)
    canonical = copy.deepcopy(await context.get_messages())
    model = provider()
    calls = []
    async def count(view):
        assert_clean(view)
        calls.append(copy.deepcopy(view))
        count = 3000 if pressure and "Continuation note" not in str(view) else 100
        return {"dispatch": object(), "budget_decision": {
            "estimated_input_tokens": count, "input_limit_tokens": 6000,
            "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": count}}}
    if route == "ordinary":
        view = await context.get_messages_for_request(provider=model)
    elif route == "retaining":
        view = await context.get_messages_for_request_retaining(retain_contents=[], provider=model)
    else:
        result = await context.get_measured_request_view(provider=model, retain_contents=[], count_view=count)
        view = result["base_view"]
        assert calls
        if result.get("transaction"):
            await result["transaction"].commit()
    assert_clean(view)
    assert any(row.get("content") == "Keep the latest correction" and
               row.get("metadata", {}).get("legacy") == "unchanged metadata" for row in view)
    assert await context.get_messages() == canonical
    assert not context._human(original[-1])
    if pressure:
        model.complete.assert_awaited()
        assert "Continuation note" in str(view)
    else:
        model.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_existing_legacy_engine_also_preserves_raw_history_and_cleans_summary():
    context = ManagedContextManager(session_dir=None)
    original = rows()
    await context.set_messages(original)
    before = copy.deepcopy(await context.get_messages())
    view = await context.get_messages_for_request()
    assert_clean(view)
    assert_clean(context._format_messages_for_summarization(original))
    assert await context.get_messages() == before


@pytest.mark.asyncio
async def test_native_compaction_request_uses_the_shared_execution_projection():
    context = BoundaryContextManager({"max_tokens": 6000, "summarize_trigger": 0.2,
                                      "compaction_notice_enabled": False})
    original = rows(pressure=True)
    await context.set_messages(original)
    native = {"role": "user", "content": "Native checkpoint", "metadata": {
        "source": "context-managed", "ephemeral": True, "persisted": True, "opaque": "fixture"}}
    calls = []
    async def compact(request):
        assert_clean(request)
        calls.append(request)
        return {"kind": "native", "message": native}
    def budget(request, **kwargs):
        assert_clean(request)
        count = 100 if any((row.metadata or {}).get("opaque") for row in request.messages) else 9000
        return {"measurement": {"kind": "provider_count", "input_tokens": count}}
    model = SimpleNamespace(supports_native_compaction=lambda: True,
        validate_compacted_context=lambda value: True, compact_context=compact,
        complete=AsyncMock(), request_budget=budget)
    await context.get_messages_for_request(provider=model)
    assert calls
    model.complete.assert_not_awaited()
    assert await context.get_messages() == original
