"""Offline model-switch contracts; no foreground inference or tools execute."""
import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message, ToolSpec
from amplifier_core.llm_errors import ContextLengthError

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
                            target=copy.deepcopy(TARGET),
                            request_model=TARGET["model"],
                            effective_identity=copy.deepcopy(TARGET),
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
            model=state.request_model, reasoning_effort="high", max_output_tokens=73,
            tools=[ToolSpec(name="fixture", description="Read evidence; do not replay writes",
                            parameters={"type": "object"})],
            metadata={"channel": "exact-account-selection"})
        state.counts.append(request.model_copy(deep=True))
        return {"dispatch": request, "budget_decision": await budget(request),
                "effective_identity": copy.deepcopy(state.effective_identity),
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
        assert (request.model or model.default_model) == state.target["model"]
        assert request.reasoning_effort == "high"
        if state.fail == "compact":
            raise asyncio.CancelledError()
        result = native_message(state.target["model"])
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
    model.validate_compacted_context = lambda row: (
        row.get("metadata", {}).get("fixture:opaque", {}).get("model") == state.target["model"])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
async def test_confirmed_target_progress_resumes_without_repeating_completed_calls(native):
    state = transition_fixture(native)
    state.kwargs["max_calls"] = 1
    with pytest.raises(CompactionError, match="summary_work_paused"):
        await state.ctx.recover_request(**state.kwargs)
    accepted = copy.deepcopy(state.saved[-1][0])
    prior_requests = {row["requestSha256"] for row in state.admitted}
    state.kwargs.update(progress=accepted, max_calls=15)
    state.admitted.clear()
    result = await state.ctx.recover_request(**state.kwargs)
    assert result["outcome"] == "transitioned"
    assert not (prior_requests & {row["requestSha256"] for row in state.admitted})
    assert state.ctx.summary_identity == TARGET


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
@pytest.mark.parametrize("remaining", [0, 16])
async def test_installed_candidate_readback_reconciles_without_new_paid_calls(native, remaining):
    state = transition_fixture(native)
    state.fail = "install"
    with pytest.raises(CompactionError, match="recovery_install_unconfirmed"):
        await state.ctx.recover_request(**state.kwargs)
    candidate = copy.deepcopy(state.installed[-1][0])
    state.unchanged()
    state.fail = None
    binding = {"messages": len(state.ctx.messages), "sha256": digest(state.ctx.messages)}
    state.ctx.messages.extend([
        {"role": "assistant", "content": "Short post-install update."},
        {"role": "user", "content": "New foreground readback only."},
    ])
    assert state.ctx._boundary(state.ctx.messages, []) > candidate["summary"]["throughMessage"]
    state.kwargs.update(progress=candidate, max_calls=remaining, source_revision=binding,
                        installed_candidate_sha256=candidate["sha256"])
    state.ctx.preserve_evidence.return_value = [
        {"kind": "fixture", "id": "new-preservation-receipt-not-a-new-install"}]
    state.admitted.clear()
    state.saved.clear()
    result = await state.ctx.recover_request(**state.kwargs)
    assert result["recovery"]["sha256"] == candidate["sha256"]
    assert result["outcome"] == "transitioned"
    assert state.admitted == state.saved == []
    assert state.ctx.export_checkpoint(TARGET) == candidate


@pytest.mark.asyncio
async def test_portable_prepay_floor_includes_all_canonical_unknown_observations():
    state = transition_fixture(False)
    inserted = []
    for i in range(40):
        inserted.extend([
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"unknown-op-{i}", "name": "write_file", "arguments": {"path": f"evidence-{i}"}}]},
            {"role": "user", "content": "Later ordinary input does not settle that operation."},
        ])
    state.ctx.messages[4:4] = inserted
    state.kwargs["source_checkpoint"] = state.ctx.export_checkpoint(SOURCE)
    before = copy.deepcopy(state.ctx.messages)
    floor_unknown_counts = []
    async def count_view(view):
        count = sum(f'"call_id": "unknown-op-{i}"' in str(view) for i in range(40))
        floor_unknown_counts.append(count)
        # The source and irreducible notice both exceed the real target limit;
        # stripping the notice would wrongly make a paid transition look useful.
        tokens = 25000 if count else 1000
        return {
            "dispatch": ChatRequest(messages=[Message(**row) for row in view], model=TARGET["model"]),
            "budget_decision": {
                "estimated_input_tokens": tokens, "input_limit_tokens": 20000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": tokens}},
        }
    state.kwargs["count_view"] = count_view
    with pytest.raises(CompactionError, match="native_input_oversized"):
        await state.ctx.recover_request(**state.kwargs)
    assert floor_unknown_counts == [40, 40]
    assert state.admitted == state.saved == state.installed == []
    assert state.ctx.messages == before


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
async def test_wrong_installed_candidate_digest_never_admits_generation(native):
    state = transition_fixture(native)
    state.kwargs["max_calls"] = 1
    with pytest.raises(CompactionError, match="summary_work_paused"):
        await state.ctx.recover_request(**state.kwargs)
    partial = copy.deepcopy(state.saved[-1][0])
    state.admitted.clear()
    state.saved.clear()
    state.kwargs.update(progress=partial, max_calls=16, installed_candidate_sha256="0" * 64)
    with pytest.raises(CompactionError, match="transition_candidate_invalid"):
        await state.ctx.recover_request(**state.kwargs)
    assert state.admitted == state.saved == state.installed == []
    state.unchanged()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
@pytest.mark.parametrize("mismatch", [None, "different-record", "paid-partial", "no-pin"])
@pytest.mark.parametrize("checkpointless", [False, True], ids=["checkpoint-backed", "canonical-originals"])
async def test_final_journal_failure_reconciles_only_exact_active_pinned_candidate(native, mismatch, checkpointless):
    state = transition_fixture(native)
    binding = {"messages": len(state.ctx.messages), "sha256": digest(state.ctx.messages)}
    if checkpointless:
        state.ctx.summary = state.ctx.summary_identity = None
        state.ctx.summary_omitted_indices = []
        state.initial = copy.deepcopy({key: getattr(state.ctx, key) for key in state.initial})
        state.kwargs.pop("source_checkpoint")
        state.kwargs["transition_source"] = {"kind": "canonical-originals", "sourceRevision": binding}
    await state.ctx.recover_request(**state.kwargs)
    candidate = copy.deepcopy(state.installed[-1][0])
    # Context activated B, but the host's subsequent journal COMMITTED write
    # failed. The host still owns A + the exact installed B readback receipt.
    if mismatch == "different-record":
        state.ctx.evidence_refs.append({"kind": "different-active-record"})
    elif mismatch == "paid-partial":
        state.ctx._summary_progress = {
            "completed": 1, "identity": copy.deepcopy(TARGET),
            "configuration": state.ctx.checkpoint_configuration(),
            "source_revision": {"messages": len(state.ctx.messages), "sha256": digest(state.ctx.messages)},
        }
    state.initial = copy.deepcopy({key: getattr(state.ctx, key) for key in state.initial})
    state.ctx.messages.extend([
        {"role": "assistant", "content": "Short post-activation update."},
        {"role": "user", "content": "Reconcile existing installation, not generation."},
    ])
    state.kwargs.update(progress=candidate, max_calls=0, source_revision=binding,
                        installed_candidate_sha256=None if mismatch == "no-pin" else candidate["sha256"])
    state.admitted.clear()
    state.saved.clear()
    state.installed.clear()
    if mismatch:
        with pytest.raises(CompactionError):
            await state.ctx.recover_request(**state.kwargs)
        state.unchanged()
        assert state.installed == []
    else:
        result = await state.ctx.recover_request(**state.kwargs)
        assert result["recovery"]["sha256"] == candidate["sha256"]
        assert len(state.installed) == 1  # Host callback acknowledges readback only.
        assert state.ctx.export_checkpoint(TARGET) == candidate
    assert state.admitted == state.saved == []


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
@pytest.mark.parametrize("default_model", [TARGET["model"], "different-target"])
async def test_selected_provider_effective_model_without_mutating_original_dispatch(native, default_model):
    state = transition_fixture(native)
    state.request_model = None  # The real loop leaves model application to its selected provider.
    state.model.default_model = default_model
    if default_model != TARGET["model"]:
        with pytest.raises(CompactionError, match="recovery_request_changed"):
            await state.ctx.recover_request(**state.kwargs)
        assert state.admitted == state.saved == state.installed == []
        state.unchanged()
    else:
        result = await state.ctx.recover_request(**state.kwargs)
        assert result["outcome"] == "transitioned"
        # Surface/provider prepared-request caches depend on preserving the loop
        # envelope and dispatch identity, not replacing it with a selected copy.
        assert result["final_attempt"]["dispatch"].model is None
        assert result["final_attempt"]["fixture"] == "unchanged"
        assert state.ctx.summary_identity == TARGET


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("binding", [None, {"provider": "different-account", "model": TARGET["model"]},
                                      {**TARGET, "model": "stale-model"}])
async def test_absent_request_model_without_exact_host_binding_is_rejected(native, binding):
    state = transition_fixture(native)
    state.request_model = None
    state.effective_identity = binding
    with pytest.raises(CompactionError, match="recovery_request_changed"):
        await state.ctx.recover_request(**state.kwargs)
    assert state.admitted == state.saved == state.installed == []
    state.unchanged()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
@pytest.mark.parametrize("phase", ["count", "admit", "save", "install"])
async def test_guard_rejection_at_each_async_boundary_prevents_activation(native, phase):
    state = transition_fixture(native)
    invalidated = False
    original = {"count": state.count_view, "admit": state.admit,
                "save": state.save, "install": state.install}[phase]

    def guard():
        if invalidated:
            raise RuntimeError("Synthetic selection or Stop revision changed")

    async def mutate(*args):
        nonlocal invalidated
        result = await original(*args)
        invalidated = True
        return result

    argument = {"count": "count_view", "admit": "admit_call",
                "save": "save_progress", "install": "install"}[phase]
    state.kwargs.update({argument: mutate, "guard": guard})
    # Portable persistence preserves its public safe storage-error wrapper;
    # both outcomes must leave the accepted source unit unchanged.
    with pytest.raises(RuntimeError, match="selection or Stop|checkpoint_persist_failed"):
        await state.ctx.recover_request(**state.kwargs)
    state.unchanged()
    if phase in {"count", "admit"}:
        state.model.complete.assert_not_awaited()
        state.model.compact_context.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
async def test_source_paid_partial_is_parked_not_rebound_or_discarded(native):
    state = transition_fixture(native)
    # Reuse a real accepted portable partial; its entire unit is the source.
    portable = transition_fixture(False)
    portable.kwargs["max_calls"] = 1
    with pytest.raises(CompactionError, match="summary_work_paused"):
        await portable.ctx.recover_request(**portable.kwargs)
    paid_record = portable.saved[-1][0]
    # The selected target changes again; the old paid partial remains quarantined.
    source_identity = copy.deepcopy(paid_record["identity"])
    target_identity = {**TARGET, "model": "third-model"}
    state.ctx.messages = copy.deepcopy(portable.ctx.messages)
    state.ctx.summary = None
    state.ctx.summary_identity = None
    assert state.ctx.restore_checkpoint(paid_record, source_identity)["status"] == "resuming"
    assert state.ctx.export_checkpoint(source_identity) == paid_record
    classify = state.ctx.classify_checkpoint_transition(paid_record, target_identity)
    assert classify["status"] == "transition_required"
    assert state.ctx.export_checkpoint(source_identity) == paid_record
    state.initial = copy.deepcopy({key: getattr(state.ctx, key) for key in state.initial})
    state.target = target_identity
    state.request_model = target_identity["model"]
    state.model.default_model = target_identity["model"]
    state.ctx.checkpoint_identity = lambda: copy.deepcopy(target_identity)
    state.kwargs.update(identity=target_identity, source_checkpoint=copy.deepcopy(paid_record))
    result = await state.ctx.recover_request(**state.kwargs)
    assert result["outcome"] == "transitioned"
    assert state.ctx.summary_identity == target_identity
    assert paid_record["progress"]["completed"] == 1
    assert paid_record["identity"] == source_identity


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True], ids=["unanswered", "queued"])
async def test_transition_never_infers_unresolved_or_queued_tool_completion(queued):
    state = transition_fixture(True)
    ctx = state.ctx
    insertion = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "unresolved-operation", "name": "write_file", "arguments": {"path": "proof"}}]},
    ]
    if queued:
        insertion.append({"role": "tool", "tool_call_id": "unresolved-operation",
                          "content": '{"status":"queued","job_id":"still-running"}'})
    ctx.messages[4:4] = insertion
    state.old = ctx.export_checkpoint(SOURCE)
    state.kwargs["source_checkpoint"] = state.old
    # Preserve full originals; no method executes tools or writes a fake result.
    before = copy.deepcopy(ctx.messages)
    with pytest.raises(CompactionError):
        await ctx.recover_request(**state.kwargs)
    assert ctx.messages == before
    assert state.admitted == []


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
async def test_fitting_originals_does_not_resolve_unused_utility_provider(native):
    ctx, model = context(native=native), provider(native=native)
    old = ctx.export_checkpoint(SOURCE)
    ctx.preserve_evidence = AsyncMock(return_value=copy.deepcopy(ctx.evidence_refs))
    def unavailable():
        raise RuntimeError("Unused utility provider is unavailable")
    ctx.summary_provider = unavailable
    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view],
                                        model=TARGET["model"]),
                "budget_decision": {"estimated_input_tokens": 1000, "input_limit_tokens": 20000,
                    "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 1000}}}
    result = await ctx.recover_request(provider=model, count_view=count_view, identity=TARGET,
        source_checkpoint=old, admit_call=AsyncMock(), max_calls=0,
        save_progress=AsyncMock(), install=AsyncMock(), guard=lambda: None)
    assert result["outcome"] == "transition_fitted"


@pytest.mark.asyncio
async def test_native_regeneration_does_not_resolve_unused_utility_provider():
    state = transition_fixture(True)
    def unavailable():
        raise RuntimeError("Unused utility provider is unavailable")
    state.ctx.summary_provider = unavailable
    result = await state.ctx.recover_request(**state.kwargs)
    assert result["outcome"] == "transitioned"


@pytest.mark.asyncio
async def test_live_provider_portable_transition_requires_isolated_utility():
    state = transition_fixture(False)
    state.model.native_bundle_live = True
    with pytest.raises(RuntimeError, match="separate context.summary_provider"):
        await state.ctx.recover_request(**state.kwargs)
    assert state.admitted == state.saved == state.installed == []
    state.unchanged()


@pytest.mark.asyncio
async def test_portable_paid_prefix_resumes_after_eligible_boundary_advances():
    state = transition_fixture(False)
    state.kwargs["max_calls"] = 1
    admitted_prefix = {"messages": len(state.ctx.messages), "sha256": digest(state.ctx.messages)}
    state.kwargs["source_revision"] = admitted_prefix
    with pytest.raises(CompactionError, match="summary_work_paused"):
        await state.ctx.recover_request(**state.kwargs)
    accepted = copy.deepcopy(state.saved[-1][0])
    saved_end = accepted["sourceRevision"]["messages"]
    # This appended normal input advances the eligible boundary within the
    # original admitted ceiling; finish only the old paid job first.
    state.ctx.messages.extend([
        {"role": "assistant", "content": "Short later update."},
        {"role": "user", "content": "Continue with new evidence still pending."},
    ])
    assert state.ctx._boundary(state.ctx.messages, []) > saved_end
    state.kwargs.update(progress=accepted, max_calls=15)
    result = await state.ctx.recover_request(**state.kwargs)
    assert result["outcome"] == "transitioned"
    assert state.ctx.summary[0] == saved_end
    assert "new evidence still pending" in str(result["base_view"])


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_cold_preservation_may_initialize_identity_and_restore_checkpoint(native):
    ctx = context(native=native)
    old = ctx.export_checkpoint(SOURCE)
    cold = BoundaryContextManager(copy.deepcopy(ctx.config))
    cold.messages = copy.deepcopy(ctx.messages)
    selected = [None]
    cold.checkpoint_identity = lambda: selected[0]
    async def preserve(messages):
        selected[0] = copy.deepcopy(SOURCE)
        assert cold.restore_checkpoint(old, SOURCE)["status"] == "restored"
        return copy.deepcopy(ctx.evidence_refs)
    cold.preserve_evidence = preserve
    model = provider(native=native, identity=SOURCE)
    model.request_budget = AsyncMock(return_value={
        "estimated_input_tokens": 1000, "input_limit_tokens": 20000,
        "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 1000}})
    await cold.get_messages_for_request(provider=model)
    assert cold.summary == ctx.summary
    assert cold.summary_identity == SOURCE
    model.complete.assert_not_awaited()
    model.compact_context.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("measured", [False, True])
async def test_ordinary_no_transition_host_returns_fitting_originals_without_paid_work(native, measured):
    ctx, model = context(native=native), provider(native=native)
    old = ctx.export_checkpoint(SOURCE)
    model.request_budget = AsyncMock(return_value={
        "estimated_input_tokens": 1000, "input_limit_tokens": 20000,
        "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 1000}})
    async def count_view(view):
        assert "source-private-state" not in str(view)
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view],
                                        model=TARGET["model"]),
                "budget_decision": await model.request_budget(None)}
    if measured:
        result = await ctx.get_measured_request_view(
            provider=model, retain_contents=[], count_view=count_view)
        view = result["base_view"]
    else:
        view = await ctx.get_messages_for_request(provider=model)
    assert "Verified historical details" in str(view)
    assert "source-private-state" not in str(view)
    assert ctx.export_checkpoint(SOURCE) == old
    model.compact_context.assert_not_awaited()
    model.complete.assert_not_awaited()