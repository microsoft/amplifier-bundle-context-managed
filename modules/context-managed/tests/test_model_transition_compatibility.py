"""Ordinary nonparticipating-host model switches must retain bounded compaction.

These use no web foreground/journal/recovery admission callback. Source-target
identity change is representative of a delegate/default-model change as well as
a direct caller deliberately selecting another continuation provider.
"""
import copy
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message, ToolSpec
from amplifier_core.llm_errors import LLMError

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.errors import CompactionError
from amplifier_module_context_managed.request_context import RequestContext


SOURCE = {"provider": "fixture", "model": "old-model"}
TARGET = {"provider": "fixture", "model": "new-model"}


def native_note(model):
    return {"role": "user", "content": "Derived native note", "metadata": {
        "source": "context-managed", "ephemeral": True, "persisted": True,
        "fixture:compaction": {"model": model, "payload": f"opaque-{model}"},
    }}


async def ordinary_case(native, measured=True):
    config = {"max_tokens": 10000, "summarize_trigger": .1,
              "summary_max_output_tokens": 73, "summary_max_calls": 2}
    ctx = BoundaryContextManager(config)
    original = [
        {"role": "system", "content": "Do not replay completed tools or invent approvals."},
        {"role": "user", "content": "Keep the original objective and records."},
        {"role": "assistant", "content": "Settled earlier evidence. " * 500},
        {"role": "user", "content": "Correction: keep all source files unchanged."},
        {"role": "assistant", "content": "Protected recent evidence is saved."},
        {"role": "user", "content": "Continue unfinished work using the selected model."},
    ]
    await ctx.set_messages(original)
    ctx.summary = (3, native_note(SOURCE["model"]) if native else "Old portable note.")
    ctx.summary_identity = copy.deepcopy(SOURCE)
    ctx.summary_omitted_indices = [0] if native else []
    model = SimpleNamespace(
        name=TARGET["provider"], default_model=TARGET["model"],
        supports_native_compaction=lambda: native,
        native_compaction_requires_request_context=measured,
        validate_compacted_context=lambda row: (
            row.get("metadata", {}).get("fixture:compaction", {}).get("model") == TARGET["model"]),
    )
    state = SimpleNamespace(ctx=ctx, model=model, original=original, phase=None,
                            derived_counts=0, compact_count=4000,
                            compact_preflights=0,
                            source=copy.deepcopy((ctx.summary, ctx.summary_identity,
                                                  ctx.summary_omitted_indices)))

    def count(request):
        assert f"opaque-{SOURCE['model']}" not in str(request)
        compact = (request.metadata or {}).get("purpose") == "context-compaction"
        if compact:
            state.compact_preflights += 1
            if state.phase == "second-preflight-cancel" and state.compact_preflights == 2:
                raise asyncio.CancelledError()
        derived = any((row.metadata or {}).get("fixture:compaction")
                      or "Continuation note (reference data" in str(row.content)
                      for row in request.messages)
        if derived and not compact:
            state.derived_counts += 1
            if state.phase == "candidate-count-error" and state.derived_counts == 1:
                raise RuntimeError("Synthetic candidate count outage")
            if state.phase in {"count-error", "count-cancel"} and state.derived_counts == 2:
                raise (asyncio.CancelledError() if state.phase == "count-cancel"
                       else RuntimeError("Synthetic final count outage"))
        # Deterministic count contract: the complete foreground originals exceed
        # the allowance; the eligible prefix fits ONE existing compact request;
        # the reduced full target request fits. No native batching is needed.
        value = 1000 if derived else state.compact_count if compact else 12000
        return {"estimated_input_tokens": value, "input_limit_tokens": 10000,
                "measurement": {"kind": "provider_count", "source": "fixture",
                                "input_tokens": value}}

    async def compact(request):
        assert (request.model or model.default_model) == TARGET["model"]
        assert f"opaque-{SOURCE['model']}" not in str(request)
        if state.phase == "paid-cancel":
            raise asyncio.CancelledError()
        return {"kind": "native", "message": native_note(TARGET["model"])}

    async def complete(request, **kwargs):
        assert (request.model or model.default_model) == TARGET["model"]
        assert f"opaque-{SOURCE['model']}" not in str(request)
        if state.phase == "paid-cancel":
            raise asyncio.CancelledError()
        return SimpleNamespace(content=[SimpleNamespace(type="text",
            text="Original objective and records retained; source files unchanged; unfinished work remains.")])

    model.request_budget = AsyncMock(side_effect=lambda request, **kwargs: count(request))
    model.compact_context = AsyncMock(side_effect=compact)
    model.complete = AsyncMock(side_effect=complete)
    assert ctx.run_pending_recovery is None

    async def count_view(view):
        request = ChatRequest(messages=[Message(**row) for row in view], model=TARGET["model"],
            max_output_tokens=73, tools=[ToolSpec(name="fixture", description="Read only",
                                                  parameters={"type": "object"})])
        return {"dispatch": request, "budget_decision": count(request)}

    async def run():
        if measured:
            return await ctx.get_measured_request_view(
                provider=model, retain_contents=[], count_view=count_view)
        return await ctx.get_messages_for_request(provider=model)
    state.run = run
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
@pytest.mark.parametrize("measured", [False, True], ids=["ordinary", "measured"])
async def test_nonparticipating_host_compacts_changed_model_with_existing_ordinary_policy(native, measured):
    state = await ordinary_case(native, measured)
    ctx, model, original = state.ctx, state.model, state.original
    result = await state.run()
    if measured:
        view = result["base_view"]
        assert result["final_attempt"]["budget_decision"]["estimated_input_tokens"] <= 10000
    else:
        view = result
    assert ctx.summary_identity == TARGET
    assert ctx.summary is not None
    assert f"opaque-{SOURCE['model']}" not in json.dumps(view)
    assert await ctx.get_messages() == original
    assert model.compact_context.await_count == int(native)
    assert model.complete.await_count == int(not native)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("phase", ["count-error", "count-cancel"])
async def test_final_fit_failure_retains_target_candidate_and_reuses_without_new_generation(native, phase, monkeypatch):
    state = await ordinary_case(native)
    final = RequestContext.get_measured_request_view
    async def fail_final(fitter, **kwargs):
        derived = "Continuation note (reference data" in str(fitter.messages) or "fixture:compaction" in str(fitter.messages)
        if derived and state.phase:
            raise asyncio.CancelledError() if state.phase == "final-cancel" else RuntimeError("Final fit outage")
        return await final(fitter, **kwargs)
    monkeypatch.setattr(RequestContext, "get_measured_request_view", fail_final)
    state.phase = "final-cancel" if phase == "count-cancel" else "final-error"
    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        await state.run()
    assert (state.ctx.summary, state.ctx.summary_identity, state.ctx.summary_omitted_indices) == state.source
    job = state.ctx._ordinary_transition
    assert job is not None and job["clone"].summary is not None
    assert job["prepared"] is not None
    assert state.model.complete.await_count + state.model.compact_context.await_count == 1
    state.phase = None
    await state.run()
    assert state.ctx.summary_identity == TARGET
    assert state.ctx._ordinary_transition is None
    assert state.model.complete.await_count + state.model.compact_context.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_uncertain_paid_call_is_retained_and_never_automatically_rebought(native):
    state = await ordinary_case(native)
    state.phase = "paid-cancel"
    with pytest.raises(asyncio.CancelledError):
        await state.run()
    assert state.ctx._ordinary_transition["uncertain"] is True
    state.phase = None
    with pytest.raises(CompactionError, match="transition_operation_unconfirmed"):
        await state.run()
    assert state.model.complete.await_count + state.model.compact_context.await_count == 1
    assert state.ctx.summary_identity == SOURCE


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("measured", [False, True])
async def test_completed_paid_result_survives_candidate_count_outage_without_rebuy(native, measured):
    state = await ordinary_case(native, measured)
    state.phase = "candidate-count-error"
    with pytest.raises((RuntimeError, CompactionError)):
        await state.run()
    assert state.ctx.summary_identity == SOURCE
    assert state.ctx._ordinary_transition is not None
    assert state.model.complete.await_count + state.model.compact_context.await_count == 1
    state.phase = None
    await state.run()
    assert state.ctx.summary_identity == TARGET
    assert state.model.complete.await_count + state.model.compact_context.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["durable", "zero-arg", "host-managed"])
@pytest.mark.parametrize("native", [False, True])
async def test_durable_legacy_or_host_managed_requires_explicit_transition_before_generation(native, writer):
    state = await ordinary_case(native)
    if writer == "durable":
        state.ctx.config["durable_checkpoints"] = True
    elif writer == "zero-arg":
        state.ctx.persist_checkpoint = AsyncMock()
    else:
        state.ctx.transition_policy = lambda: "host-managed"
    code = "model_transition_required" if writer == "host-managed" else "transition_persistence_required"
    with pytest.raises(CompactionError, match=code):
        await state.run()
    state.model.complete.assert_not_awaited()
    state.model.compact_context.assert_not_awaited()
    if state.ctx.persist_checkpoint is not None:
        state.ctx.persist_checkpoint.assert_not_awaited()
    assert state.ctx._ordinary_transition is None
    assert state.ctx.summary_identity == SOURCE


@pytest.mark.asyncio
async def test_native_oversized_single_window_refuses_without_batching_or_fallback():
    state = await ordinary_case(True)
    state.compact_count = 12000
    with pytest.raises(CompactionError, match="native_input_oversized"):
        await state.run()
    state.model.complete.assert_not_awaited()
    state.model.compact_context.assert_not_awaited()
    assert state.ctx.summary_identity == SOURCE


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source", "model", "config", "provider-instance"])
async def test_single_provisional_job_is_quarantined_instead_of_rebound(change, monkeypatch):
    state = await ordinary_case(False)
    final = RequestContext.get_measured_request_view
    async def fail_final(fitter, **kwargs):
        if "Continuation note (reference data" in str(fitter.messages):
            raise RuntimeError("Final fit outage")
        return await final(fitter, **kwargs)
    monkeypatch.setattr(RequestContext, "get_measured_request_view", fail_final)
    with pytest.raises(RuntimeError):
        await state.run()
    job = state.ctx._ordinary_transition
    if change == "source":
        state.ctx.messages[2]["content"] += " changed original"
    elif change == "model":
        state.model.default_model = "third-model"
    elif change == "provider-instance":
        # Same labels/defaults are not evidence of the same account adapter.
        state.model = copy.copy(state.model)
        async def changed_run():
            return await state.ctx.get_messages_for_request(provider=state.model)
        state.run = changed_run
    else:
        state.ctx.config["summary_max_output_tokens"] = 74
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        await state.run()
    assert state.ctx._ordinary_transition is job
    state.model.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_unpaid_native_preflight_job_can_be_replaced_after_configuration_repair():
    state = await ordinary_case(True)
    state.compact_count = 12000
    with pytest.raises(CompactionError, match="native_input_oversized"):
        await state.run()
    assert state.ctx._ordinary_transition["paid_calls"] == 0
    state.ctx.config["native_compaction_max_output_tokens"] = 73
    state.compact_count = 4000
    await state.run()
    assert state.ctx.summary_identity == TARGET
    state.model.compact_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_participation_policy_refuses_before_generation():
    state = await ordinary_case(False)
    state.ctx.transition_policy = lambda: "mistyped-host-managed"
    with pytest.raises(ValueError, match="transition_policy"):
        await state.run()
    state.model.complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
async def test_mounted_pending_wrapper_is_not_itself_host_participation(managed):
    state = await ordinary_case(False)
    state.ctx.run_pending_recovery = AsyncMock(return_value=None)
    state.ctx.transition_policy = lambda: "host-managed" if managed else None
    if managed:
        with pytest.raises(CompactionError, match="model_transition_required"):
            await state.run()
        state.model.complete.assert_not_awaited()
    else:
        await state.run()
        state.model.complete.assert_awaited_once()
        assert state.ctx.summary_identity == TARGET
    state.ctx.run_pending_recovery.assert_awaited_once()


def test_prepared_originals_attestation_is_pure_and_rejects_derived_or_provisional_state():
    ctx = BoundaryContextManager({"durable_checkpoints": True})
    ctx.checkpoint_identity = lambda: copy.deepcopy(TARGET)
    before = copy.deepcopy((ctx.messages, ctx.checkpoint_status, ctx.summary_identity))
    assert ctx.prepared_originals(TARGET) is True
    assert ctx.prepared_originals(SOURCE) is False
    assert ctx.prepared_originals({}) is False
    for field, value in (("summary", (3, "foreign summary")),
                         ("summary_identity", SOURCE),
                         ("summary_omitted_indices", [0]),
                         ("recovery", {"source": "foreign"}),
                         ("_summary_progress", {"completed": 1}),
                         ("_restored_progress", {"completed": 1}),
                         ("_ordinary_transition", {"uncertain": True})):
        setattr(ctx, field, value)
        assert ctx.prepared_originals(TARGET) is False
        setattr(ctx, field, [] if field == "summary_omitted_indices" else None)
    assert (ctx.messages, ctx.checkpoint_status, ctx.summary_identity) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
async def test_prepared_originals_preserves_ordinary_failure_latch_and_cooldown(native):
    state = await ordinary_case(native)
    ctx = state.ctx
    # This is ordinary target preparation after a settled originals-only view,
    # not another model transition or a provisional paid job.
    ctx.summary = ctx.summary_identity = None
    ctx.summary_omitted_indices = []
    ctx.checkpoint_identity = lambda: copy.deepcopy(TARGET)
    failing = state.model.compact_context if native else state.model.complete
    failing.side_effect = LLMError("Synthetic ordinary compaction failure", retryable=True)
    with pytest.raises(CompactionError):
        await state.run()
    assert ctx.summary_failure is not None
    assert ctx.summary is ctx._summary_progress is ctx._restored_progress is None
    assert ctx._ordinary_transition is None
    assert ctx.summary_identity is None and ctx.summary_omitted_indices == []
    assert ctx.recovery is None
    fields = ("messages", "summary", "summary_identity", "summary_omitted_indices",
              "recovery", "_summary_progress", "_restored_progress", "summary_failure",
              "checkpoint_status", "evidence_refs", "_ordinary_transition")
    before = copy.deepcopy({key: getattr(ctx, key) for key in fields})
    assert ctx.prepared_originals(TARGET) is True
    assert {key: getattr(ctx, key) for key in fields} == before
    with pytest.raises(CompactionError, match="previous_failure"):
        await state.run()
    failing.assert_awaited_once()
    assert ctx.summary_failure == before["summary_failure"]
    assert ctx.prepared_originals(TARGET) is True
    assert ctx.messages == state.original


@pytest.mark.asyncio
async def test_covered_history_replacement_cannot_bypass_uncertain_provisional_job():
    state = await ordinary_case(False)
    state.phase = "paid-cancel"
    with pytest.raises(asyncio.CancelledError):
        await state.run()
    job = state.ctx._ordinary_transition
    replacement = copy.deepcopy(state.ctx.messages)
    replacement[2]["content"] += " replaced canonical evidence"
    await state.ctx.set_messages(replacement)
    assert state.ctx.summary is None
    state.phase = None
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        await state.run()
    assert state.ctx._ordinary_transition is job
    state.model.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_host_recovery_entry_cannot_bypass_uncertain_ordinary_job():
    state = await ordinary_case(False)
    state.phase = "paid-cancel"
    with pytest.raises(asyncio.CancelledError):
        await state.run()
    callback = AsyncMock(side_effect=AssertionError("Host intent must not bypass retained UNKNOWN"))
    state.ctx.run_pending_recovery = callback
    with pytest.raises(CompactionError, match="transition_operation_unconfirmed"):
        await state.run()
    callback.assert_not_awaited()
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        await state.ctx.recover_request(provider=state.model, count_view=callback,
            identity=TARGET, save_progress=callback, install=callback, guard=lambda: None)
    callback.assert_not_awaited()
    state.model.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_portable_completed_piece_survives_later_preflight_cancellation_without_rebuy():
    state = await ordinary_case(False)
    state.ctx.config.update(summary_max_source_chars=3000, summary_max_calls=16)
    state.phase = "second-preflight-cancel"
    with pytest.raises(asyncio.CancelledError):
        await state.run()
    job = state.ctx._ordinary_transition
    paid = copy.deepcopy(job["clone"]._summary_progress)
    assert paid["completed"] == paid["calls"] == 1
    assert job["uncertain"] is False  # Cancelled counting, not a paid operation.
    assert state.ctx.summary_identity == SOURCE
    first_request = state.model.complete.await_args_list[0].args[0].model_dump()
    state.phase = None
    await state.run()
    assert state.ctx.summary_identity == TARGET
    assert state.ctx._ordinary_transition is None
    assert all(call.args[0].model_dump() != first_request
               for call in state.model.complete.await_args_list[1:])


@pytest.mark.asyncio
@pytest.mark.parametrize("guard_change", ["identity", "config", "operations"])
async def test_changed_contract_after_paid_generation_never_publishes_target(guard_change):
    state = await ordinary_case(False)
    original = state.model.complete.side_effect
    selected = [copy.deepcopy(TARGET)]
    state.ctx.checkpoint_identity = lambda: copy.deepcopy(selected[0])
    ops = [{"id": "running-op", "status": "unknown"}]
    state.ctx.active_operations = lambda: copy.deepcopy(ops)
    async def mutate(*args, **kwargs):
        response = await original(*args, **kwargs)
        if guard_change == "identity":
            selected[0] = {**TARGET, "model": "third-model"}
        elif guard_change == "config":
            state.ctx.config["summary_max_output_tokens"] = 74
        else:
            ops.append({"id": "new-op", "status": "queued"})
        return response
    state.model.complete.side_effect = mutate
    with pytest.raises((RuntimeError, CompactionError)):
        await state.run()
    assert state.ctx.summary_identity == SOURCE
    job = state.ctx._ordinary_transition
    assert job["uncertain"] is True
    assert job["last_response"] is not None
    state.model.complete.assert_awaited_once()