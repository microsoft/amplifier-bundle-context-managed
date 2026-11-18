"""Main-graph ordinary fitting must not activate or rebuy parked transitions."""
import copy
from unittest.mock import AsyncMock

import pytest

from amplifier_module_context_managed import ordinary_transition
from amplifier_module_context_managed.checkpoint import canonical_bytes
from .test_ordinary_settled_fitting import Provider, history, manager, measured


SOURCE = {"provider": "fixture", "model": "previous"}
TARGET = {"provider": "fixture", "model": "bounded"}
UNIT = ("summary", "summary_identity", "summary_omitted_indices", "recovery",
        "_summary_progress", "_restored_progress", "summary_failure", "evidence_refs",
        "_ordinary_transition")


def park(context, state):
    context.summary = (2, "Source portable note")
    context.summary_identity = copy.deepcopy(SOURCE)
    context.evidence_refs = [{"kind": "transcript", "id": "source-original"}]
    context.summary_failure = {"fingerprint": "retained-original-failure"}
    if state == "foreign-native":
        context.summary = (2, {"role": "user", "content": "Source opaque state",
            "metadata": {"source": "context-managed", "ephemeral": True, "persisted": True,
                         "opaque": {"model": SOURCE["model"], "data": "source-only"}}})
        context.summary_omitted_indices = [0]
        context.recovery = {"kind": "native-recovery", "source": "retained"}
    if state in {"foreign-progress", "foreign-restored"}:
        context.summary = None
        context.summary_identity = None
        key = "_summary_progress" if state == "foreign-progress" else "_restored_progress"
        setattr(context, key, {"identity": copy.deepcopy(SOURCE), "completed": 1,
            "text": "Paid source piece", "pending": ["Unfinished exact source"],
            "calls": 1, "configuration": context.checkpoint_configuration(),
            "source_revision": {"messages": 2, "sha256": "source-only"}})
    if state.startswith("job-"):
        context.summary_identity = copy.deepcopy(TARGET)
        context._ordinary_transition = {"binding": ["original-binding"], "paid_calls": 1,
            "uncertain": state == "job-unconfirmed",
            "prepared": {"note": "Paid target not activated"} if state == "job-completed" else None,
            "last_response": {"note": "Returned exact paid result"},
            "clone": {"summary": "Provisional isolated target"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [
    "foreign-portable", "foreign-native", "foreign-progress", "foreign-restored",
    "job-pending", "job-completed", "job-unconfirmed"])
@pytest.mark.parametrize("hard_fit", [False, True], ids=["measured", "hard-fit"])
@pytest.mark.parametrize("cold_identity", [False, True], ids=["bound-host", "cold-host"])
async def test_legacy_durable_ordinary_answers_without_changing_parked_unit(
        state, hard_fit, cold_identity, monkeypatch):
    rows = history()
    context = await manager(rows, durable_checkpoints=True)
    park(context, state)
    before = canonical_bytes({key: getattr(context, key) for key in UNIT})
    job = context._ordinary_transition
    transition = AsyncMock(side_effect=AssertionError("Ordinary fitting must not enter transition machinery"))
    monkeypatch.setattr(ordinary_transition, "prepare", transition)
    current = {"identity": None if cold_identity else copy.deepcopy(TARGET)}
    context.checkpoint_identity = lambda: copy.deepcopy(current["identity"])
    async def preserve(messages):
        assert messages == rows
        current["identity"] = copy.deepcopy(TARGET)
        return [{"kind": "transcript", "id": "fresh-evidence-not-source-checkpoint"}]
    context.preserve_evidence = AsyncMock(side_effect=preserve)
    context.persist_checkpoint = AsyncMock(side_effect=AssertionError("Do not write source checkpoint"))
    context.run_pending_recovery = AsyncMock(return_value=None)
    provider = Provider()
    for _ in range(2):
        if hard_fit:
            from amplifier_core import ChatRequest, Message
            view = await context.get_messages_for_request_retaining(
                provider=provider, retain_contents=[], hard_fit=True)
            request = ChatRequest(messages=[Message(**row) for row in view])
        else:
            result = await measured(context, provider)
            view, request = result["base_view"], result["final_attempt"]["dispatch"]
        await provider.complete(request)
        assert "Current objective" in str(view)
        assert "Source opaque state" not in str(view)
        assert "Source portable note" not in str(view)
        assert canonical_bytes({key: getattr(context, key) for key in UNIT}) == before
        assert context._ordinary_transition is job
    transition.assert_not_awaited()
    context.persist_checkpoint.assert_not_awaited()
    assert context.preserve_evidence.await_count == 2
    assert context.run_pending_recovery.await_count == (0 if state.startswith("job-") or hard_fit else 2)
    assert len(provider.completed) == 2 and provider.summaries == 0
    assert context._foreground_paid_input is None and not context._foreground_unconfirmed
    assert await context.get_messages() == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["portable", "native"])
@pytest.mark.parametrize("phase", ["paid-cancel", "candidate-count-error"])
async def test_real_retained_transition_job_is_not_touched_by_opted_in_dispatch(native, phase):
    import asyncio
    from .test_model_transition_compatibility import ordinary_case, TARGET as JOB_TARGET
    from amplifier_module_context_managed.errors import CompactionError
    state = await ordinary_case(native)
    state.phase = phase
    with pytest.raises((asyncio.CancelledError, RuntimeError, CompactionError)):
        await state.run()
    context = state.ctx
    job = context._ordinary_transition
    assert job is not None
    clone = job["clone"]
    job_repr = repr(job)
    clone_before = canonical_bytes({
        "messages": clone.messages,
        "unit": {key: getattr(clone, key) for key in UNIT if key != "_ordinary_transition"}})
    paid_before = state.model.complete.await_count + state.model.compact_context.await_count
    context.config.update(foreground_continuation=True, durable_checkpoints=True)
    context.checkpoint_identity = lambda: copy.deepcopy(JOB_TARGET)
    async def preserve(messages):
        return [{"kind": "transcript", "id": "new-preservation"}]
    context.preserve_evidence = preserve
    before = canonical_bytes({key: getattr(context, key) for key in UNIT if key != "_ordinary_transition"})
    provider = Provider()
    provider.default_model = JOB_TARGET["model"]
    for _ in range(2):
        result = await measured(context, provider)
        await provider.complete(result["final_attempt"]["dispatch"])
        assert context._ordinary_transition is job
        assert repr(job) == job_repr
        assert canonical_bytes({key: getattr(context, key) for key in UNIT
                                if key != "_ordinary_transition"}) == before
        assert canonical_bytes({"messages": clone.messages,
            "unit": {key: getattr(clone, key) for key in UNIT if key != "_ordinary_transition"}}) == clone_before
    assert len(provider.completed) == 2 and provider.summaries == 0
    assert state.model.complete.await_count + state.model.compact_context.await_count == paid_before == 1