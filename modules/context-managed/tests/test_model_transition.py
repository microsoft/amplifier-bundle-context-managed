"""Offline model-switch contracts; no foreground inference or tools execute."""
import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import ContextLengthError

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.checkpoint import digest


SOURCE = {"provider": "fixture-connection", "model": "source-model"}
TARGET = {**SOURCE, "model": "target-model"}


def rows():
    return [
        {"role": "system", "content": "Preserve original evidence and never replay tools."},
        {"role": "user", "content": "Build the report; keep unfinished work."},
        {"role": "assistant", "content": "Verified historical details. " * 500},
        {"role": "user", "content": "Correction: leave the source files unchanged."},
        {"role": "assistant", "content": "Verification remains unfinished."},
        {"role": "user", "content": "Continue with the explicitly selected model."},
    ]


def native_message(model):
    return {"role": "user", "content": "Native continuation", "metadata": {
        "source": "context-managed", "ephemeral": True, "persisted": True,
        "fixture:opaque": {"model": model, "payload": "source-private-state"},
    }}


def context(*, native):
    ctx = BoundaryContextManager({
        "durable_checkpoints": True, "summarize_trigger": 100, "max_tokens": 20000,
    })
    ctx.messages = rows()
    ctx.summary = (3, native_message(SOURCE["model"]) if native
                   else "Verified report evidence; source files unchanged; verification unfinished.")
    ctx.summary_identity = copy.deepcopy(SOURCE)
    ctx.summary_omitted_indices = [0] if native else []
    ctx.evidence_refs = [{"kind": "transcript", "id": "original-evidence"}]
    ctx.checkpoint_identity = lambda: copy.deepcopy(TARGET)
    return ctx


def provider(*, native, identity=TARGET):
    return SimpleNamespace(
        name=identity["provider"], default_model=identity["model"],
        supports_native_compaction=lambda: native,
        validate_compacted_context=lambda row: (
            row.get("metadata", {}).get("fixture:opaque", {}).get("model")
            == identity["model"]),
        compact_context=AsyncMock(side_effect=AssertionError("No new compaction expected")),
        complete=AsyncMock(side_effect=AssertionError("No completion or tool execution")),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "opaque"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["count-error", "cancelled"])
async def test_failed_target_preparation_keeps_source_checkpoint(native, cancelled):
    ctx, model = context(native=native), provider(native=native)
    original = copy.deepcopy(ctx.messages)
    source_record = ctx.export_checkpoint(SOURCE)
    source_state = copy.deepcopy((
        ctx.summary, ctx.summary_identity, ctx.summary_omitted_indices,
        ctx.evidence_refs, ctx.recovery,
    ))
    calls = []

    async def count_view(view):
        calls.append(copy.deepcopy(view))
        # Source-model opaque state must not reach even the target count endpoint.
        assert "source-private-state" not in str(view)
        if cancelled:
            raise asyncio.CancelledError()
        raise RuntimeError("Synthetic target count failed")

    with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
        await ctx.get_measured_request_view(
            provider=model, retain_contents=[], count_view=count_view)
    assert len(calls) == 1
    assert (ctx.summary, ctx.summary_identity, ctx.summary_omitted_indices,
            ctx.evidence_refs, ctx.recovery) == source_state
    assert ctx.export_checkpoint(SOURCE) == source_record
    assert await ctx.get_messages() == original
    model.compact_context.assert_not_awaited()
    model.complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "opaque"])
async def test_smaller_target_final_fit_refusal_keeps_source_checkpoint(native):
    ctx, model = context(native=native), provider(native=native)
    source_record = ctx.export_checkpoint(SOURCE)
    views = []

    async def count_view(view):
        assert "source-private-state" not in str(view)
        views.append(copy.deepcopy(view))
        return {
            "dispatch": ChatRequest(
                messages=[Message(**row) for row in view], model=TARGET["model"]),
            "budget_decision": {
                "estimated_input_tokens": 25000, "input_limit_tokens": 20000,
                "measurement": {"kind": "provider_count", "source": "fixture",
                                "input_tokens": 25000},
            },
        }

    with pytest.raises(ContextLengthError):
        await ctx.get_measured_request_view(
            provider=model, retain_contents=[], count_view=count_view)
    assert views
    assert ctx.export_checkpoint(SOURCE) == source_record
    assert ctx.messages == rows()
    model.compact_context.assert_not_awaited()
    model.complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "opaque"])
async def test_explicit_transition_fitting_originals_never_pays_or_installs_none(native):
    ctx, model = context(native=native), provider(native=native)
    source_record = ctx.export_checkpoint(SOURCE)
    source_bytes = copy.deepcopy(source_record)
    save, install, admit = AsyncMock(), AsyncMock(), AsyncMock()
    ctx.preserve_evidence = AsyncMock(return_value=copy.deepcopy(ctx.evidence_refs))
    guard = SimpleNamespace(calls=0)

    def check():
        guard.calls += 1

    async def count_view(view):
        assert ctx.summary_identity == SOURCE  # No early target activation.
        assert "source-private-state" not in str(view)
        return {
            "dispatch": ChatRequest(
                messages=[Message(**row) for row in view], model=TARGET["model"],
                reasoning_effort="high", max_output_tokens=73),
            "budget_decision": {
                "estimated_input_tokens": 1000, "input_limit_tokens": 20000,
                "max_output_tokens": 73,
                "measurement": {"kind": "provider_count", "source": "fixture",
                                "input_tokens": 1000},
            },
        }

    result = await ctx.recover_request(
        provider=model, count_view=count_view, identity=TARGET,
        source_checkpoint=source_record, admit_call=admit, max_calls=16,
        save_progress=save, install=install, guard=check)
    assert result["outcome"] == "transition_fitted"
    assert result["final_attempt"]["dispatch"].model == TARGET["model"]
    assert result["final_attempt"]["dispatch"].reasoning_effort == "high"
    assert result["final_attempt"]["dispatch"].max_output_tokens == 73
    assert "source-private-state" not in str(result["base_view"])
    assert guard.calls >= 2
    assert source_record == source_bytes
    assert ctx.messages == rows()
    assert ctx.summary is None  # Host retains quarantined source_record, not target-active.
    admit.assert_not_awaited()
    save.assert_not_awaited()
    install.assert_not_awaited()
    model.complete.assert_not_awaited()
    model.compact_context.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "opaque"])
async def test_same_identity_reuses_checkpoint_without_generation(native):
    ctx, model = context(native=native), provider(native=native, identity=SOURCE)
    ctx.checkpoint_identity = lambda: copy.deepcopy(SOURCE)
    source_record = ctx.export_checkpoint(SOURCE)

    async def count_view(view):
        return {
            "dispatch": ChatRequest(
                messages=[Message(**row) for row in view], model=SOURCE["model"]),
            "budget_decision": {
                "estimated_input_tokens": 100, "input_limit_tokens": 20000,
                "measurement": {"kind": "provider_count", "source": "fixture",
                                "input_tokens": 100},
            },
        }

    result = await ctx.get_measured_request_view(
        provider=model, retain_contents=[], count_view=count_view)
    assert result["final_attempt"]["dispatch"].model == SOURCE["model"]
    assert ctx.export_checkpoint(SOURCE) == source_record
    model.compact_context.assert_not_awaited()
    model.complete.assert_not_awaited()


@pytest.mark.parametrize("native", [False, True], ids=["portable", "opaque"])
@pytest.mark.parametrize("invalid", ["identity", "config-and-identity", "digest-and-identity"])
def test_restart_does_not_admit_foreign_checkpoint_as_target_state(native, invalid):
    ctx = context(native=native)
    record = ctx.export_checkpoint(SOURCE)
    restored = BoundaryContextManager(copy.deepcopy(ctx.config))
    restored.messages = copy.deepcopy(ctx.messages)
    if invalid == "config-and-identity":
        restored.config["summary_target_tokens"] = 99
    if invalid == "digest-and-identity":
        record["evidenceRefs"].append({"kind": "tampered"})
    status = restored.restore_checkpoint(record, TARGET)
    assert status["status"] == "rejected"
    assert status["originalsAvailable"] is True
    assert restored.summary is None
    assert restored.messages == ctx.messages


@pytest.mark.parametrize("native", [False, True], ids=["portable", "opaque"])
def test_passive_transition_classification_validates_source_without_installing_it(native):
    ctx = context(native=native)
    record = ctx.export_checkpoint(SOURCE)
    before = copy.deepcopy((ctx.messages, ctx.summary, ctx.summary_identity))
    classify = getattr(ctx, "classify_checkpoint_transition", None)
    assert callable(classify), "Host needs structured identity-only transition provenance"
    status = classify(record, TARGET)
    assert status == {
        "status": "transition_required", "sourceIdentity": SOURCE,
        "targetIdentity": TARGET, "sourceRevision": record["sourceRevision"],
        "configuration": ctx.checkpoint_configuration(), "sha256": record["sha256"],
    }
    assert (ctx.messages, ctx.summary, ctx.summary_identity) == before
    assert ctx.export_checkpoint(SOURCE) == record


@pytest.mark.parametrize("invalid", [
    "digest", "canonical-prefix", "configuration", "forged-configuration",
    "opaque-envelope", "target-identity",
])
def test_transition_classifier_never_upgrades_invalid_checkpoint_to_admission(invalid):
    ctx = context(native=True)
    record = ctx.export_checkpoint(SOURCE)
    target = copy.deepcopy(TARGET)
    if invalid == "digest":
        record["summary"]["message"]["content"] = "Tampered evidence"
    elif invalid == "canonical-prefix":
        ctx.messages[1]["content"] = "Different canonical objective"
    elif invalid == "configuration":
        ctx.config["summary_target_tokens"] = 999
    elif invalid == "forged-configuration":
        # A self-consistent foreign record is not proof of the trusted contract.
        record["configuration"] = digest({"config": "attacker-selected"})
    elif invalid == "opaque-envelope":
        record["summary"]["message"]["metadata"]["persisted"] = False
    else:
        target["model"] = None
    if invalid in {"forged-configuration", "opaque-envelope"}:
        record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    before = copy.deepcopy((ctx.messages, ctx.summary, ctx.summary_identity))
    classify = getattr(ctx, "classify_checkpoint_transition", None)
    assert callable(classify), "Host needs structured identity-only transition provenance"
    status = classify(record, target)
    assert status["status"] == "rejected"
    assert status["originalsAvailable"] is True
    assert status["code"] != "model_transition_required"
    assert (ctx.messages, ctx.summary, ctx.summary_identity) == before