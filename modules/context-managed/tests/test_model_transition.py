"""Offline model-switch contracts; no foreground inference or tools execute."""
import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message, ToolSpec

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.checkpoint import digest
from amplifier_module_context_managed.errors import CompactionError


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

    # Without an admitted host transaction, refuse before target callbacks.
    # Exception/cancellation recovery itself is exercised by explicit entry tests.
    with pytest.raises(CompactionError, match="model_transition_required"):
        await ctx.get_measured_request_view(
            provider=model, retain_contents=[], count_view=count_view)
    assert len(calls) == 0
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

    with pytest.raises(CompactionError, match="model_transition_required"):
        await ctx.get_measured_request_view(
            provider=model, retain_contents=[], count_view=count_view)
    assert not views
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
        "validation": {"record": "verified", "canonicalPrefix": "verified",
                       "configuration": "verified", "sourceProviderEnvelope": "unverified"},
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


def transition_fixture(native, *, archive=False):
    ctx = context(native=native)
    ctx.config.update(summarize_trigger=.1, summary_max_output_tokens=73,
                      summary_max_source_chars=3000, archive_recovery=archive)
    ctx.messages = [
        {"role": "system", "content": "Current safety: never replay tools."},
        *[{"role": role, "content": f"entry-{i} " + ("settled evidence " * 70 if role == "assistant"
                                                 else "Continue report.")}
          for i in range(8) for role in ("user", "assistant")],
        {"role": "user", "content": "Keep pending question q1; outcome op1 UNKNOWN."},
        {"role": "assistant", "content": "No action has been repeated."},
        {"role": "user", "content": "Use the selected model and retain q1/op1."},
    ]
    ctx.summary = (3, native_message(SOURCE["model"]) if native else "Prior source note.")
    ctx.summary_omitted_indices = [0] if native else []
    old = ctx.export_checkpoint(SOURCE)
    ctx.preserve_evidence = AsyncMock(return_value=[{"kind": "fixture", "id": "full-originals"}])
    model = provider(native=native)
    state = SimpleNamespace(ctx=ctx, model=model, old=old, saved=[], admitted=[], installed=[],
                            fail=None, limit=6000, counts=[], request_options=[],
                            initial=copy.deepcopy({key: getattr(ctx, key) for key in (
                                "summary", "summary_identity", "summary_omitted_indices",
                                "_summary_progress", "_restored_progress", "recovery", "evidence_refs")}))

    def unchanged():
        assert {key: getattr(ctx, key) for key in state.initial} == state.initial

    def size(request):
        return len(json.dumps(request.model_dump(mode="json"), sort_keys=True))

    async def budget(request, **kwargs):
        assert "source-private-state" not in str(request)
        count = size(request)
        return {"estimated_input_tokens": count, "input_limit_tokens": state.limit,
                "measurement": {"kind": "provider_count", "source": "fixture",
                                "input_tokens": count}}

    async def count_view(view):
        unchanged()
        if state.fail == "count":
            raise RuntimeError("synthetic count outage")
        request = ChatRequest(
            messages=[Message(**row) for row in view],
            model=TARGET["model"], reasoning_effort="high", max_output_tokens=73,
            tools=[ToolSpec(name="fixture", description="Read evidence; do not replay writes",
                            parameters={"type": "object"})],
            metadata={"channel": "exact-account-selection"})
        state.counts.append(request.model_copy(deep=True))
        return {"dispatch": request, "budget_decision": await budget(request),
                "fixture": "unchanged"}

    async def admit(report):
        unchanged()
        state.admitted.append(copy.deepcopy(report))
        if state.fail == "admit":
            return False
        return True

    async def compact(request):
        unchanged()
        assert state.admitted[-1]["requestSha256"] == digest(request.model_dump(mode="json"))
        assert request.model == TARGET["model"]
        assert request.reasoning_effort == "high"
        if state.fail == "compact":
            raise asyncio.CancelledError()
        result = native_message(TARGET["model"])
        result["metadata"]["fixture:opaque"]["payload"] = "target-private-state"
        return {"kind": "native", "message": result}

    async def complete(request, **kwargs):
        unchanged()
        assert state.admitted[-1]["requestSha256"] == digest(request.model_dump(mode="json"))
        if state.fail == "compact":
            raise asyncio.CancelledError()
        return SimpleNamespace(content=[SimpleNamespace(type="text",
            text="Settled report evidence; keep originals; question q1 and op1 UNKNOWN.")])

    async def save(record, report):
        unchanged()
        assert report["call"] == state.admitted[-1]
        assert report["disposition"] == "completed"
        if state.fail == "save":
            raise OSError("synthetic atomic storage rejection")
        state.saved.append((copy.deepcopy(record), copy.deepcopy(report)))

    async def install(record, verification):
        unchanged()
        assert verification["sha256"] == record["sha256"]
        state.installed.append((copy.deepcopy(record), copy.deepcopy(verification)))
        if state.fail == "install":
            return False
        return {"sha256": record["sha256"]}

    model.request_budget = AsyncMock(side_effect=budget)
    model.compact_context = AsyncMock(side_effect=compact)
    model.complete = AsyncMock(side_effect=complete)
    state.count_view, state.admit, state.save, state.install = count_view, admit, save, install
    state.unchanged = unchanged
    state.kwargs = dict(provider=model, count_view=count_view, identity=TARGET,
                        source_checkpoint=old, admit_call=admit, max_calls=16,
                        save_progress=save, install=install, guard=lambda: None,
                        target_bytes=2500)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
@pytest.mark.parametrize("archive", [False, True], ids=["full-history", "archive-policy"])
async def test_transition_batches_only_target_source_and_installs_exact_final_fit(native, archive):
    state = transition_fixture(native, archive=archive)
    original = copy.deepcopy(state.ctx.messages)
    result = await state.ctx.recover_request(**state.kwargs)
    assert result["outcome"] == "transitioned"
    assert result["recovery"]["status"] == "installed"
    assert state.ctx.summary_identity == TARGET
    assert state.ctx.messages == original
    assert state.old["identity"] == SOURCE
    assert len(state.admitted) == len(state.saved) > 1
    assert len(state.installed) == 1
    assert result["measured_after"] < result["measured_before"]
    assert "q1/op1" in str(result["base_view"])
    assert all(row["sourceCheckpointSha256"] == state.old["sha256"] for row in state.admitted)
    assert all(row["targetIdentity"] == TARGET for row in state.admitted)
    assert state.ctx.export_checkpoint(TARGET) == state.installed[0][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
@pytest.mark.parametrize("failure", ["count", "admit", "compact", "save", "install"])
async def test_explicit_transition_failure_preserves_entire_source_unit(native, failure):
    state = transition_fixture(native)
    state.fail = failure
    with pytest.raises((RuntimeError, CompactionError, OSError, asyncio.CancelledError)):
        await state.ctx.recover_request(**state.kwargs)
    state.unchanged()
    assert state.ctx.export_checkpoint(SOURCE) == state.old
    if failure in {"count", "admit"}:
        state.model.compact_context.assert_not_awaited()
        state.model.complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
async def test_irreducible_target_tail_refuses_before_any_paid_work(native):
    state = transition_fixture(native)
    state.ctx.messages[-1]["content"] += "required current context " * 1000
    with pytest.raises(CompactionError, match="native_input_oversized"):
        await state.ctx.recover_request(**state.kwargs)
    assert state.admitted == state.saved == state.installed == []
    state.unchanged()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
async def test_transition_call_cap_preserves_progress_without_activating_partial(native):
    state = transition_fixture(native)
    state.kwargs["max_calls"] = 1
    with pytest.raises(CompactionError, match="summary_work_paused"):
        await state.ctx.recover_request(**state.kwargs)
    assert len(state.admitted) == len(state.saved) == 1
    assert not state.installed
    state.unchanged()


@pytest.mark.asyncio
async def test_checkpointless_originals_fork_never_fabricates_source_checkpoint():
    state = transition_fixture(native=True)
    ctx = state.ctx
    ctx.summary = ctx.summary_identity = None
    ctx.summary_omitted_indices = []
    state.initial = copy.deepcopy({key: getattr(ctx, key) for key in state.initial})
    binding = {"messages": len(ctx.messages), "sha256": digest(ctx.messages)}
    state.kwargs.pop("source_checkpoint")
    state.kwargs["transition_source"] = {"kind": "canonical-originals", "sourceRevision": binding}
    result = await ctx.recover_request(**state.kwargs)
    assert result["recovery"]["sourceRevision"] == binding
    assert all(row["sourceCheckpointSha256"] is None for row in state.admitted)