"""Failures must stop execution, never turn into silent history loss."""
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import ContextLengthError

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.errors import CompactionError
from amplifier_module_context_managed.request_context import RequestContext
from amplifier_module_context_managed.summary import public_messages, request_fits, source_fragments, summary_request


@pytest.mark.asyncio
@pytest.mark.parametrize("measured", [False, True])
async def test_large_current_tool_result_reaches_request_unchanged(measured):
    context = BoundaryContextManager({"max_tokens": 100000})
    original = [
        {"role": "user", "content": "Inspect the complete result."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "large", "name": "read_file", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "large", "content": "Evidence " * 20000},
    ]
    await context.set_messages(original)
    if measured:
        async def count(view):
            assert view[-1]["content"] == original[-1]["content"]
            return {"dispatch": object(), "budget_decision": {
                "estimated_input_tokens": 50000, "input_limit_tokens": 100000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 50000}}}
        result = await context.get_measured_request_view(provider=None, retain_contents=[], count_view=count)
        view = result["base_view"]
        assert result["count_calls"] == 1
    else:
        view = await context.get_messages_for_request()
    assert view[-1]["content"] == original[-1]["content"]
    assert await context.get_messages() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("measured", [False, True])
async def test_oversized_current_turn_stops_without_trimming_or_extra_counting(measured):
    context = BoundaryContextManager({"max_tokens": 1000})
    original = [{"role": "user", "content": "Current task " * 10000}]
    await context.set_messages(original)
    model = SimpleNamespace(complete=AsyncMock())
    views = []
    async def count(view):
        views.append(copy.deepcopy(view))
        return {"dispatch": object(), "budget_decision": {
            "estimated_input_tokens": 30000, "input_limit_tokens": 1000,
            "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 30000}}}
    with pytest.raises(ContextLengthError, match="No emergency history trimming"):
        if measured:
            await context.get_measured_request_view(provider=model, retain_contents=[], count_view=count)
        else:
            await context.get_messages_for_request(provider=model)
    assert len(views) == int(measured)
    assert await context.get_messages() == original
    model.complete.assert_not_awaited()
    assert context.summary is None


@pytest.mark.parametrize("config, expected", [({}, 8192), ({"summary_target_tokens": 1500}, 8192),
                                            ({"summary_max_output_tokens": 12000}, 12000)])
def test_summary_output_allowance_is_separate_from_legacy_note_target(config, expected):
    request = summary_request("Summarize", "Evidence", "", config)
    assert request.max_output_tokens == expected


MEDIA = "experimental.media.v1"


def media_history():
    def ref(digest):
        return {"v": 1, "session_id": "fixture-session", "sha256": digest * 64,
                "mime": "image/png", "bytes": 101, "width": 20, "height": 10,
                "provenance": {"kind": "tool", "session_id": "fixture-session",
                               "id": "inspect", "created_at": "2026-01-01T00:00:00Z",
                               "expires_at": None}}

    return [
        {"role": "user", "content": "Inspect these observations."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "inspect", "name": "read_file", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "inspect", "content": "Image observations.",
         "metadata": {"source": "tool", MEDIA: {"refs": [ref("b"), ref("a"), ref("b"), ref("c")]}}},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, {}, [], {"estimated_input_tokens": False, "input_limit_tokens": 1000},
                                   {"estimated_input_tokens": 10, "input_limit_tokens": 0}])
async def test_ref_request_cannot_fit_on_absent_or_invalid_provider_budget(budget):
    context = BoundaryContextManager({"max_tokens": 10000})
    original = media_history()
    await context.set_messages(original)
    provider = SimpleNamespace(request_budget=AsyncMock(return_value=budget), complete=AsyncMock())
    with pytest.raises(CompactionError) as caught:
        await context.get_messages_for_request(provider=provider)
    assert caught.value.code == "media_budget_unavailable"
    assert await context.get_messages() == original
    provider.complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("measured", [False, True])
async def test_ref_request_without_counter_cannot_use_text_estimate(measured):
    context = BoundaryContextManager({"max_tokens": 10000})
    original = media_history()
    await context.set_messages(original)

    async def count_view(view):
        assert view[-1]["metadata"][MEDIA] == original[-1]["metadata"][MEDIA]
        return {"dispatch": object(), "budget_decision": None}

    with pytest.raises(CompactionError) as caught:
        if measured:
            await context.get_measured_request_view(provider=None, retain_contents=[], count_view=count_view)
        else:
            await context.get_messages_for_request()
    assert caught.value.code == "media_budget_unavailable"
    assert await context.get_messages() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("decision", [
    [], {}, {"estimated_input_tokens": False, "input_limit_tokens": 10000},
    {"estimated_input_tokens": -1, "input_limit_tokens": 10000},
])
async def test_malformed_measured_contract_keeps_type_error_before_media_guard(preflight, decision):
    from amplifier_module_context_managed.measurement import RequestMeasurement

    fitter = RequestContext(max_tokens=10000)
    original = media_history()
    await fitter.set_messages(original)

    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]), "budget_decision": decision}

    with pytest.raises(TypeError):
        if preflight:
            await RequestMeasurement(count_view).measure(fitter, await fitter._view())
        else:
            await fitter.get_measured_request_view(provider=None, retain_contents=[], count_view=count_view)
    assert await fitter.get_messages() == original


def media_budget(count=3000, limit=10000):
    return {"estimated_input_tokens": count, "input_limit_tokens": limit,
            "measurement": {"kind": "provider_count", "source": "fixture.media", "input_tokens": count}}


@pytest.mark.asyncio
@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("evidence", [
    None, {}, {"v": True, "validated": True, "projection_digest": "a" * 64},
    {"v": 1, "validated": False, "projection_digest": "a" * 64},
    {"v": 1, "validated": True, "projection_digest": "NOT-A-DIGEST"},
    {"v": 1, "validated": True, "projection_digest": "A" * 64},
])
async def test_even_a_numeric_count_requires_media_projection_evidence(measured, evidence):
    context = BoundaryContextManager({"max_tokens": 10000})
    original = media_history()
    await context.set_messages(original)
    provider = SimpleNamespace(request_budget=AsyncMock(return_value=media_budget()))
    if evidence is not None:
        provider.media_budget_witness = lambda request: evidence

    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]),
                "budget_decision": media_budget()}

    with pytest.raises(CompactionError) as caught:
        if measured:
            await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
        else:
            await context.get_messages_for_request(provider=provider)
    assert caught.value.code == "media_budget_unavailable"
    assert await context.get_messages() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("count", [3000, 10001])
async def test_media_aware_count_and_exact_witness_gate_unchanged_ref_view(measured, count):
    context = BoundaryContextManager({"max_tokens": 10000})
    original = media_history()
    await context.set_messages(original)
    counted = []
    witnessed = []

    async def budget(request, **kwargs):
        assert [ref["sha256"][0] for ref in request.messages[-1].metadata[MEDIA]["refs"]] == ["b", "a", "b", "c"]
        counted.append(request.model_dump())
        return media_budget(count)

    def witness(request):
        assert request.model_dump() == counted[-1]
        witnessed.append(request.model_dump())
        return {"v": 1, "validated": True, "projection_digest": "a" * 64}

    provider = SimpleNamespace(request_budget=budget, media_budget_witness=witness, complete=AsyncMock())

    async def count_view(view):
        dispatch = ChatRequest(messages=[Message(**row) for row in view])
        return {"dispatch": dispatch, "budget_decision": await budget(dispatch)}

    async def prepare():
        if measured:
            result = await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
            assert result["measured_after"] == count
            return result["base_view"]
        return await context.get_messages_for_request(provider=provider)

    if count > 10000:
        with pytest.raises(ContextLengthError):
            await prepare()
    else:
        view = await prepare()
        assert view[-1]["metadata"] == original[-1]["metadata"]
        view[-1]["metadata"][MEDIA]["refs"].clear()
    assert counted and witnessed
    assert await context.get_messages() == original
    provider.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_measured_dispatch_refs_cannot_hide_behind_text_only_base_view():
    context = BoundaryContextManager({"max_tokens": 10000})
    await context.set_messages([{"role": "user", "content": "Text view."}])

    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in media_history()]),
                "budget_decision": None}

    with pytest.raises(CompactionError) as caught:
        await context.get_measured_request_view(provider=None, retain_contents=[], count_view=count_view)
    assert caught.value.code == "media_budget_unavailable"


@pytest.mark.asyncio
async def test_output_fit_must_revalidate_media_budget_for_replacement_dispatch():
    fitter = RequestContext(max_tokens=10000)
    original = media_history()
    await fitter.set_messages(original)
    dispatch = ChatRequest(messages=[Message(**row) for row in original])
    witness = SimpleNamespace(media_budget_witness=lambda request: (
        {"v": 1, "validated": True, "projection_digest": "a" * 64} if request.max_output_tokens == 8000 else None))
    dispatch.max_output_tokens = 8000

    async def count_view(view):
        return {"dispatch": dispatch, "budget_decision": media_budget(12000)}

    async def fit_output(view, attempt):
        return {"dispatch": dispatch.model_copy(update={"max_output_tokens": 1000}),
                "budget_decision": media_budget(3000), "count_calls": 1}

    with pytest.raises(CompactionError) as caught:
        await fitter.get_measured_request_view(provider=witness, retain_contents=[],
                                               count_view=count_view, fit_output=fit_output)
    assert caught.value.code == "media_budget_unavailable"


@pytest.mark.asyncio
async def test_media_preflight_refuses_before_utility_summary_on_unqualified_count():
    context = BoundaryContextManager({"max_tokens": 10000, "summarize_trigger": 0})
    await context.set_messages(media_history() + [{"role": "user", "content": "Later turn."},
                                                 {"role": "assistant", "content": "Completed."},
                                                 {"role": "user", "content": "Current turn."}])
    provider = SimpleNamespace(complete=AsyncMock())

    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]),
                "budget_decision": media_budget(9000)}

    with pytest.raises(CompactionError) as caught:
        await context.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    assert caught.value.code == "media_budget_unavailable"
    provider.complete.assert_not_awaited()
    assert context.summary is None


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, {}, [], {"estimated_input_tokens": False, "input_limit_tokens": 1000}])
async def test_plain_text_keeps_existing_unavailable_budget_semantics(budget):
    provider = SimpleNamespace(request_budget=AsyncMock(return_value=budget))
    request = summary_request("Summarize", "Text evidence only.", "", {})
    assert await request_fits(provider, request) == (True, None)
    context = BoundaryContextManager({"max_tokens": 10000})
    original = [{"role": "user", "content": "Plain text."}]
    await context.set_messages(original)
    view = await context.get_messages_for_request(provider=provider)
    assert view[0]["content"] == original[0]["content"]
    assert await context.get_messages() == original


@pytest.mark.asyncio
async def test_plain_text_measured_no_budget_still_uses_local_estimate():
    context = BoundaryContextManager({"max_tokens": 10000})
    original = [{"role": "user", "content": "Plain text."}]
    await context.set_messages(original)

    async def count_view(view):
        return {"dispatch": object(), "budget_decision": None}

    result = await context.get_measured_request_view(provider=None, retain_contents=[], count_view=count_view)
    assert result["base_view"][0]["content"] == original[0]["content"]
    assert result["count_calls"] == 1
    assert result["measured_after"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("measured", [False, True])
async def test_explicit_unavailable_text_can_use_fallback_without_resurrecting_refs(measured):
    # The host, not the context, freezes this text-only candidate before count.
    # Each observation remains at the original tool/user position and multiplicity.
    rows = media_history()
    rows[-1]["content"] += "\n" + "\n".join(
        f"[Image {index + 1} unavailable: this request is text-only.]" for index in range(4))
    del rows[-1]["metadata"][MEDIA]
    context = BoundaryContextManager({"max_tokens": 10000})
    await context.set_messages(rows)

    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]), "budget_decision": None}

    if measured:
        result = await context.get_measured_request_view(provider=None, retain_contents=[], count_view=count_view)
        view = result["base_view"]
    else:
        view = await context.get_messages_for_request()
    assert view[-1]["tool_call_id"] == "inspect"
    assert view[-1]["content"] == rows[-1]["content"]
    assert MEDIA not in str(view)
    assert await context.get_messages() == rows


def test_text_summary_source_never_carries_hydratable_media_metadata():
    original = media_history()
    source = public_messages(original)
    assert source[-1] == {"role": "tool", "tool_call_id": "inspect", "content": "Image observations.",
                          "metadata": {"source": "tool"}}
    assert MEDIA not in str(source)
    # Quoting source as data must not turn reference metadata into image input.
    request = summary_request("Summarize", next(source_fragments(source, None)), "", {})
    assert all(MEDIA not in (message.metadata or {}) for message in request.messages)
    assert original[-1]["metadata"][MEDIA]["refs"][0]["sha256"] == "b" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize("measured", [False, True])
async def test_empty_ref_envelope_and_quoted_lookalike_are_ordinary_text(measured):
    original = [{"role": "user", "content": 'Quoted data: {"experimental.media.v1":{"refs":["not authority"]}}',
                 "metadata": {MEDIA: {"refs": []}}}]
    context = BoundaryContextManager({"max_tokens": 10000})
    await context.set_messages(original)

    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]), "budget_decision": None}

    if measured:
        result = await context.get_measured_request_view(provider=None, retain_contents=[], count_view=count_view)
        view = result["base_view"]
    else:
        view = await context.get_messages_for_request()
    assert view[0]["content"] == original[0]["content"]
    assert view[0]["metadata"] == original[0]["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize("envelope", [None, {}, {"refs": None}, {"refs": [], "unknown": True}])
async def test_malformed_ref_metadata_does_not_sneak_through_no_budget_fallback(envelope):
    context = BoundaryContextManager({"max_tokens": 10000})
    await context.set_messages([{"role": "user", "content": "Text.", "metadata": {MEDIA: envelope}}])
    with pytest.raises(CompactionError) as caught:
        await context.get_messages_for_request()
    assert caught.value.code == "media_budget_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_rows", [
    [{"role": "tool", "content": "warning", "metadata": {MEDIA: {"refs": ["hidden"]}}}],
    [{"role": "user", "content": [{"type": "image", "source": {"data": "private"}}]}],
    [{"role": "user", "content": "warning", "opaque": "uncounted"}],
    [{"role": "user", "content": [{"text": "missing type"}]}],
    [],
])
async def test_private_fallback_witness_cannot_return_hidden_nontext_or_opaque_rows(invalid_rows):
    fitter = RequestContext(max_tokens=10000)
    await fitter.set_messages(media_history())

    async def fallback(request, *, request_options=None):
        return {"v": 1, "validated": True, "projection_digest": "a" * 64, "messages": invalid_rows}

    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]), "budget_decision": None}

    with pytest.raises(CompactionError, match="media_budget_unavailable"):
        await fitter.get_measured_request_view(provider=SimpleNamespace(media_text_fallback_witness=fallback),
                                               retain_contents=[], count_view=count_view)


@pytest.mark.asyncio
async def test_failed_numeric_witness_does_not_retry_warning_only_fallback():
    fitter = RequestContext(max_tokens=10000)
    await fitter.set_messages(media_history())
    provider = SimpleNamespace(media_budget_witness=lambda request: None,
                               media_text_fallback_witness=AsyncMock())

    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]), "budget_decision": media_budget()}

    with pytest.raises(CompactionError):
        await fitter.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    provider.media_text_fallback_witness.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("preflight", [False, True])
async def test_zero_limit_decision_cannot_bypass_warning_text_estimation(preflight):
    from amplifier_module_context_managed.measurement import RequestMeasurement

    fitter = RequestContext(max_tokens=100)
    await fitter.set_messages(media_history())
    fallback = AsyncMock(return_value={
        "v": 1, "validated": True, "projection_digest": "a" * 64,
        "messages": [{"role": "user", "content": "Warning text " * 10000}]})
    provider = SimpleNamespace(media_text_fallback_witness=fallback)

    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]),
                "budget_decision": {"estimated_input_tokens": 0, "input_limit_tokens": 0}}

    with pytest.raises(CompactionError, match="media_budget_unavailable"):
        if preflight:
            await RequestMeasurement(count_view, provider).measure(fitter, await fitter._view())
        else:
            await fitter.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_canonical_and_request_view_deep_copies_preserve_ref_order_and_multiplicity():
    original = media_history()
    context = BoundaryContextManager()
    await context.set_messages(original[:-1])
    await context.add_message(original[-1])
    original[-1]["metadata"][MEDIA]["refs"].clear()
    saved = await context.get_messages()
    expected = media_history()
    assert saved == expected
    saved[-1]["metadata"][MEDIA]["refs"].reverse()
    assert await context.get_messages() == expected

    fitter = RequestContext()
    await fitter.set_messages(await context.get_messages())
    view = await fitter._view()
    assert view[-1]["metadata"][MEDIA] == expected[-1]["metadata"][MEDIA]
    view[-1]["metadata"][MEDIA]["refs"].clear()
    assert await fitter.get_messages() != view
    assert await context.get_messages() == expected


def test_summarized_rows_and_interrupted_stubs_do_not_inherit_removed_refs():
    from amplifier_module_context_managed.interrupted import project

    context = BoundaryContextManager()
    original = media_history()
    removed = context._assembled(original, (len(original), "Text continuation note."), [], [])
    assert MEDIA not in str(removed)
    # A missing sibling result gets an absence receipt, never another result's refs.
    original[1]["tool_calls"].append({"id": "missing", "name": "read_file", "arguments": {}})
    projected = project(original, [{"call_id": "missing", "tool": "read_file",
                                    "source_message": 1, "continued_at": 3}])
    assert next(row for row in projected if row.get("tool_call_id") == "inspect")["metadata"][MEDIA] == original[-1]["metadata"][MEDIA]
    assert MEDIA not in next(row for row in projected if row.get("tool_call_id") == "missing").get("metadata", {})
