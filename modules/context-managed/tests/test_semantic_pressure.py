"""Semantic triggers follow the complete request, not transcript byte size."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_module_context_managed.errors import CompactionError

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.measurement import RequestMeasurement, measurement_diagnostics
from amplifier_module_context_managed.request_context import RequestContext


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


def diagnostic_decision():
    decision = envelope(12000)["budget_decision"]
    decision["measurement"].update({
        "attribution": {
            "version": 1, "scope_id": "a" * 32, "count_id": "b" * 32,
            "request": {
                "payload_tag": "c" * 64, "serialized_bytes": 1200,
                "input_items": 2, "input_types": {"message": 1, "other": 1},
                "input_bytes": 800, "instructions_bytes": 100, "tools": 1, "tools_bytes": 200,
            },
            "compact_input": {
                "payload_tag": "d" * 64, "serialized_bytes": 1000,
                "input_items": 2, "input_types": {"message": 1, "other": 1},
                "input_bytes": 800, "instructions_bytes": 100, "tools": 0, "tools_bytes": 2,
            },
        },
        "implementation": {
            "provider_source_sha256": "e" * 64, "count_source_sha256": None,
            "sdk_version": "2.14.0", "imported_bytes_verified": False,
        },
    })
    return decision


def test_measurement_diagnostics_exact_projection_ignores_private_unknown_values():
    class PrivateValue:
        def __deepcopy__(self, memo):
            raise AssertionError("Private values must not be copied")

        def __iter__(self):
            raise AssertionError("Private values must not be traversed")

    decision = diagnostic_decision()
    expected = diagnostic_decision()["measurement"]
    expected = {field: expected[field] for field in ("attribution", "implementation")}
    measurement = decision["measurement"]
    for mapping in (decision, measurement, measurement["attribution"], measurement["implementation"],
                    measurement["attribution"]["request"], measurement["attribution"]["compact_input"],
                    measurement["attribution"]["request"]["input_types"],
                    measurement["attribution"]["compact_input"]["input_types"]):
        mapping["private/content/path/error"] = PrivateValue()

    result = measurement_diagnostics(decision)
    assert result == expected
    # The nested receipt and implementation are detached, not aliases.
    measurement["attribution"]["request"]["input_types"]["message"] = 99
    measurement["attribution"]["compact_input"]["payload_tag"] = "f" * 64
    measurement["implementation"]["imported_bytes_verified"] = True
    assert result == expected


@pytest.mark.parametrize("part", ["attribution", "implementation"])
@pytest.mark.parametrize("invalid", [None, False, [], "private error"])
def test_measurement_diagnostics_projects_optional_blocks_independently(part, invalid):
    decision = diagnostic_decision()
    other = "implementation" if part == "attribution" else "attribution"
    expected_other = diagnostic_decision()["measurement"][other]
    decision["measurement"][part] = invalid
    assert measurement_diagnostics(decision) == {part: None, other: expected_other}
    decision["measurement"][other] = None
    assert measurement_diagnostics(decision) is None


@pytest.mark.parametrize("description", ["request", "compact_input"])
@pytest.mark.parametrize("field", [
    "serialized_bytes", "input_items", "input_bytes", "instructions_bytes", "tools", "tools_bytes",
    "message", "other",
])
@pytest.mark.parametrize("invalid", [True, -1, 2**53, 1.0, "1", None])
def test_measurement_diagnostics_rejects_invalid_description_counts(description, field, invalid):
    decision = diagnostic_decision()
    target = decision["measurement"]["attribution"][description]
    if field in ("message", "other"):
        target = target["input_types"]
    target[field] = invalid
    result = measurement_diagnostics(decision)
    assert result["attribution"] is None
    assert result["implementation"] == decision["measurement"]["implementation"]


@pytest.mark.parametrize("field,invalid", [
    ("version", True), ("version", -1), ("version", 2), ("version", 1.0), ("version", "1"),
    ("scope_id", "a" * 31), ("scope_id", "A" * 32), ("scope_id", None),
    ("count_id", "b" * 33), ("count_id", "g" * 32), ("count_id", False),
    ("request", None), ("compact_input", []),
])
def test_measurement_diagnostics_rejects_invalid_receipt_identity(field, invalid):
    decision = diagnostic_decision()
    decision["measurement"]["attribution"][field] = invalid
    assert measurement_diagnostics(decision)["attribution"] is None


@pytest.mark.parametrize("description", ["request", "compact_input"])
@pytest.mark.parametrize("field,invalid", [
    ("payload_tag", "c" * 63), ("payload_tag", "C" * 64), ("payload_tag", 123),
    ("input_types", []), ("input_types", None),
])
def test_measurement_diagnostics_rejects_invalid_description_shape(description, field, invalid):
    decision = diagnostic_decision()
    decision["measurement"]["attribution"][description][field] = invalid
    assert measurement_diagnostics(decision)["attribution"] is None


@pytest.mark.parametrize("field,invalid", [
    ("provider_source_sha256", "e" * 63), ("count_source_sha256", "E" * 64),
    ("provider_source_sha256", False), ("count_source_sha256", {}),
    ("sdk_version", "/private/sdk"), ("sdk_version", "2.14.0\nprivate error"),
    ("sdk_version", "2.14.0+private"), ("sdk_version", "9" * 1000),
    ("sdk_version", True), ("sdk_version", None), ("sdk_version", "02.14.0"),
    ("imported_bytes_verified", 0), ("imported_bytes_verified", 1),
    ("imported_bytes_verified", "false"), ("imported_bytes_verified", None),
])
def test_measurement_diagnostics_rejects_invalid_implementation(field, invalid):
    decision = diagnostic_decision()
    decision["measurement"]["implementation"][field] = invalid
    result = measurement_diagnostics(decision)
    assert result["implementation"] is None
    assert result["attribution"] == decision["measurement"]["attribution"]


@pytest.mark.parametrize("verified", [False, True])
def test_measurement_diagnostics_preserves_bounded_counts_hashes_and_boolean(verified):
    decision = diagnostic_decision()
    measurement = decision["measurement"]
    measurement["input_tokens"] = 2**53 - 1
    for description in ("request", "compact_input"):
        measurement["attribution"][description]["serialized_bytes"] = 2**53 - 1
        measurement["attribution"][description]["tools"] = 0
        measurement["attribution"][description]["input_types"] = {
            kind: 0 for kind in (
                "message", "function_call", "function_call_output", "compaction", "reasoning",
                "computer_call", "computer_call_output", "apply_patch_call",
                "apply_patch_call_output", "item_reference", "other",
            )
        }
    measurement["implementation"].update({
        "provider_source_sha256": None, "count_source_sha256": "f" * 64,
        "imported_bytes_verified": verified,
    })
    result = measurement_diagnostics(decision)
    assert result == {field: measurement[field] for field in ("attribution", "implementation")}
    assert result["implementation"]["imported_bytes_verified"] is verified


@pytest.mark.parametrize("field,invalid", [
    ("kind", None), ("kind", "provider_estimate"), ("kind", []),
    ("source", ""), ("source", None), ("source", []),
    ("input_tokens", True), ("input_tokens", -1), ("input_tokens", 2**53),
    ("input_tokens", 1.0),
])
def test_measurement_diagnostics_requires_provider_measurement(field, invalid):
    decision = diagnostic_decision()
    decision["measurement"][field] = invalid
    assert measurement_diagnostics(decision) is None


@pytest.mark.parametrize("decision", [None, [], {}, {"measurement": []}, envelope(12000)["budget_decision"]])
def test_measurement_diagnostics_missing_metadata_is_unavailable(decision):
    assert measurement_diagnostics(decision) is None


@pytest.mark.asyncio
async def test_request_measurement_keeps_tuple_preflight_and_cache_contract():
    attempt = {"dispatch": object(), "budget_decision": diagnostic_decision()}
    callback = AsyncMock(return_value=attempt)
    measurement = RequestMeasurement(callback)
    assert measurement.diagnostics is None
    fitter = RequestContext({"max_tokens": 20000})
    view = history()
    assert await measurement.measure(fitter, view) == (12000, 20000, "provider_count")
    assert measurement.diagnostics == measurement_diagnostics(attempt["budget_decision"])
    assert await measurement.count(view) is attempt
    assert measurement.preflight_calls == measurement.reused_calls == 1
    callback.assert_awaited_once()
    # A later provider without this metadata remains supported and clears it.
    callback.return_value = envelope(12000)
    assert await measurement.measure(fitter, view) == (12000, 20000, "provider_count")
    assert measurement.diagnostics is None
    callback.return_value = envelope(12000, measured=False)
    assert await measurement.measure(fitter, view) == (12000, 20000, "provider_estimate")
    assert measurement.diagnostics is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["measure", "count"])
async def test_request_measurement_count_failure_clears_old_diagnostics_before_callback(operation):
    attempt = {"dispatch": object(), "budget_decision": diagnostic_decision()}
    callback = AsyncMock(return_value=attempt)
    measurement = RequestMeasurement(callback)
    fitter = RequestContext({"max_tokens": 20000})
    await measurement.measure(fitter, history())
    assert measurement.diagnostics is not None

    async def fail(view):
        assert measurement.diagnostics is None
        raise RuntimeError("count failed")

    callback.side_effect = fail
    changed_view = [{"role": "user", "content": "A different request"}]
    with pytest.raises(RuntimeError, match="count failed"):
        if operation == "measure":
            await measurement.measure(fitter, changed_view)
        else:
            await measurement.count(changed_view)
    assert measurement.diagnostics is None


@pytest.mark.asyncio
async def test_request_measurement_malformed_optional_metadata_does_not_change_count():
    decision = diagnostic_decision()
    decision["measurement"]["attribution"]["version"] = True
    decision["measurement"]["implementation"]["sdk_version"] = "/private/error"
    callback = AsyncMock(return_value={"dispatch": object(), "budget_decision": decision})
    measurement = RequestMeasurement(callback)
    fitter = RequestContext({"max_tokens": 20000})
    assert await measurement.measure(fitter, history()) == (12000, 20000, "provider_count")
    assert measurement.diagnostics is None


@pytest.mark.asyncio
async def test_request_measurement_validates_numeric_budget_before_recording_diagnostics():
    attempt = {"dispatch": object(), "budget_decision": diagnostic_decision()}
    callback = AsyncMock(return_value=attempt)
    measurement = RequestMeasurement(callback)
    fitter = RequestContext({"max_tokens": 20000})
    await measurement.measure(fitter, history())
    assert measurement.diagnostics is not None
    attempt["budget_decision"]["estimated_input_tokens"] = False
    with pytest.raises(TypeError, match="nonnegative integer"):
        await measurement.measure(fitter, history())
    assert measurement.diagnostics is None


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
    with pytest.raises(CompactionError):
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
