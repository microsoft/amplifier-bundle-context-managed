"""Bounded combined thinking/text allowance; no inference or tool execution."""
import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import ContextLengthError
from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.checkpoint import digest
from amplifier_module_context_managed.errors import CompactionError
from amplifier_module_context_managed.summary import request_fits, summary_config, summary_request


IDENTITY = {"provider": "fixture", "model": "large"}


def rows():
    return [
        {"role": "user", "content": "Preserve ORBIT and all completed receipts."},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "", "signature": "opaque-signed-original"},
            {"type": "tool_call", "id": "done", "name": "record", "input": {}}]},
        {"role": "tool", "tool_call_id": "done", "content": "RECEIPT-completed " + "x" * 4200},
        {"role": "user", "content": "Accepted correction."},
        {"role": "assistant", "content": "Correction acknowledged."},
        {"role": "user", "content": "Continue without replay."},
    ]


def provider(window=1000000, output=128000, *, finish=None, reject=False):
    calls, counts = [], []

    async def count(request, **kwargs):
        counts.append(request.model_dump())
        if reject:
            raise ContextLengthError("Output allowance exceeds the selected model ceiling")
        # Simulates a provider-owned assembly clamp, including its reservation.
        cap = output if request.max_output_tokens is None else min(request.max_output_tokens, output)
        tokens = sum(len(str(row.content)) for row in request.messages) // 4
        return {"estimated_input_tokens": tokens, "input_limit_tokens": window - cap - 4096,
                "context_token_budget": kwargs["context_estimate"], "max_output_tokens": cap,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": tokens}}

    async def complete(request, **kwargs):
        assert request.model_dump() == counts[-1], "Count/dispatch request changed"
        calls.append(request.model_dump())
        # The real incident's needed output is censored above8192. This finite
        # fixture requires12000 combined thinking/text tokens, not12000 text.
        reason = finish or ("max_tokens" if request.max_output_tokens is not None and request.max_output_tokens < 12000 else "end_turn")
        return SimpleNamespace(finish_reason=reason, content=[
            SimpleNamespace(type="thinking", text="not-summary-evidence"),
            SimpleNamespace(type="text", text="ORBIT; RECEIPT-completed; originals retained.")],
            usage={"input_tokens": 100, "output_tokens": min(12000, request.max_output_tokens or output)})

    obj = SimpleNamespace(name="fixture", default_model="large", complete=complete,
        request_budget=count, get_info=lambda: SimpleNamespace(defaults={
            "model": "large", "context_window": window, "max_output_tokens": output}, capabilities=[]))
    obj.calls, obj.counts = calls, counts
    return obj


async def manager(**config):
    value = BoundaryContextManager({"max_tokens": 10000, "summarize_trigger": .01,
        "summary_max_source_chars": 1200, "durable_checkpoints": True,
        "summary_retry_delay": 0, **config}, SimpleNamespace(emit=AsyncMock()))
    value.checkpoint_identity = lambda: copy.deepcopy(IDENTITY)
    await value.set_messages(rows())
    return value


@pytest.mark.asyncio
async def test_large_output_profile_completes_multiple_fragments_without_touching_originals():
    context, model = await manager(), provider()
    original = await context.get_messages()
    hashes = [digest(row) for row in original]
    await context._prepare(model, None, [])
    assert len(model.calls) >= 4
    assert {row["max_output_tokens"] for row in model.calls} == {32768}
    assert {row["reasoning_effort"] for row in model.calls} == {"low"}
    assert context.summary is not None and context._summary_progress is None
    assert [digest(row) for row in await context.get_messages()] == hashes
    assert context.export_checkpoint(IDENTITY)["summary"]["text"].startswith("ORBIT")
    # The context only consumes saved receipts; no tool executor is supplied.
    assert original[2]["content"].startswith("RECEIPT-completed")


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [4096, 8192, 16384])
async def test_explicit_cost_allowance_wins_over_profile(cap):
    context, model = await manager(summary_max_output_tokens=cap), provider(finish="end_turn")
    await context._prepare(model, None, [])
    assert {row["max_output_tokens"] for row in model.calls} == {cap}


@pytest.mark.asyncio
@pytest.mark.parametrize("window,output,expected", [(128000, 16000, 8192), (1000000, 4096, 4096)])
async def test_smaller_advertised_profiles_keep_bounded_allowance(window, output, expected):
    context, model = await manager(), provider(window, output, finish="end_turn")
    await context._prepare(model, None, [])
    assert {row["max_output_tokens"] for row in model.calls} == {expected}


@pytest.mark.asyncio
async def test_unknown_profile_does_not_gain_larger_default():
    context, model = await manager(), provider(finish="end_turn")
    model.get_info = lambda: SimpleNamespace(defaults={}, capabilities=[])
    await context._prepare(model, None, [])
    assert {row["max_output_tokens"] for row in model.calls} == {8192}


def service_limited_provider():
    model = provider()
    info = model.get_info()
    info.capabilities = ["output_limit:service_default_only"]
    model.get_info = lambda: info
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize("config,wire_limit", [({}, None), ({"summary_max_output_tokens": 16384}, 16384)])
async def test_service_managed_default_applies_only_to_an_implicit_limit(config, wire_limit):
    context, model = await manager(**config), service_limited_provider()
    await context._prepare(model, None, [])
    assert model.calls and {req["max_output_tokens"] for req in model.calls} == {wire_limit}


@pytest.mark.asyncio
async def test_service_default_keeps_room_for_output_without_adding_a_wire_limit():
    model = service_limited_provider()
    config, _ = await summary_config(model, {})
    request = summary_request("system", "source", "", config)
    async def count(counted, **kwargs):
        assert counted is request and counted.max_output_tokens is None
        return {"estimated_input_tokens": 30000, "input_limit_tokens": 60000, "max_output_tokens": 4096}
    model.request_budget = count
    fits, budget = await request_fits(model, request, output_reserve=config["summary_output_reserve"])
    assert not fits and budget["output_reserve"] == 32768


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [{}, {"summary_target_tokens": 1500}])
async def test_resume_with_service_default_requires_same_generation_contract(config):
    context = await manager(summary_max_calls=1, **config)
    with pytest.raises(CompactionError):
        await context._prepare(service_limited_provider(), None, [])
    record = context.export_checkpoint(IDENTITY)
    assert record["configurationVersion"] == 6
    contract = record["progress"]["generation_contract"]
    assert contract["max_output_tokens"] is None
    assert ("prompt_sha256" in contract) == bool(config)

    cold = await manager(**config)
    assert cold.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
    replacement = provider()
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        await cold._prepare(replacement, None, [])
    assert not replacement.counts and not replacement.calls

    assert cold.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
    await cold._prepare(service_limited_provider(), None, [])
    assert cold.summary is not None
    downgraded = copy.deepcopy(record)
    downgraded["configurationVersion"] = 2
    downgraded["sha256"] = digest({k: v for k, v in downgraded.items() if k != "sha256"})
    assert cold.restore_checkpoint(downgraded, IDENTITY)["status"] != "resuming"

    contract["max_output_tokens"] = 8192
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    assert cold.restore_checkpoint(record, IDENTITY)["status"] != "resuming"


@pytest.mark.asyncio
async def test_provider_output_rejection_does_not_split_or_resubmit_source():
    context, model = await manager(), provider(reject=True)
    await context.set_messages([{"role": "user", "content": "SOURCE" * 300}] + rows()[1:])
    original = await context.get_messages()
    with pytest.raises(CompactionError):
        await context._prepare(model, None, [])
    assert len(model.counts) == 1
    assert model.calls == []
    assert context.summary is None
    assert await context.get_messages() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["max_tokens", "length", "incomplete"])
async def test_partial_note_remains_rejected_at_larger_allowance(finish):
    context, model = await manager(), provider(finish=finish)
    original = await context.get_messages()
    with pytest.raises(CompactionError, match="output allowance"):
        await context._prepare(model, None, [])
    assert len(model.calls) == 1
    assert context.summary is None and context._summary_progress is None
    assert context.export_checkpoint(IDENTITY) is None
    assert await context.get_messages() == original


@pytest.mark.asyncio
async def test_checkpoint_contract_does_not_reuse_implicit_8k_progress():
    context, model = await manager(summary_max_calls=1), provider(finish="end_turn")
    with pytest.raises(CompactionError):
        await context._prepare(model, None, [])
    record = context.export_checkpoint(IDENTITY)
    cold = await manager()
    assert cold.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
    # Changing only the advertised profile changes selected output work. Old
    # paid progress must not silently reappear under that different contract.
    small = provider(128000, 16000, finish="end_turn")
    before = copy.deepcopy(cold._restored_progress)
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        await cold._prepare(small, None, [])
    assert cold.summary is None and cold._restored_progress == before
    assert small.calls == [] and small.counts == []


@pytest.mark.asyncio
async def test_same_contract_checkpoint_resumes_completed_fragments():
    context, model = await manager(summary_max_calls=1), provider()
    with pytest.raises(CompactionError):
        await context._prepare(model, None, [])
    record = context.export_checkpoint(IDENTITY)
    assert record["progress"]["completed"] == 1
    assert record["configurationVersion"] == 4
    cold = await manager()
    assert cold.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
    resumed = provider()
    await cold._prepare(resumed, None, [])
    assert cold.summary is not None
    finished = next(call.args[1] for call in cold.hooks.emit.call_args_list
                    if call.args[0] == "context:compaction_finished")
    assert finished["resumed_fragments"] == 1
    assert not any(call.args[0] == "context:checkpoint_progress_rejected"
                   for call in cold.hooks.emit.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("defaults", [
    {"model": "different", "context_window": 1000000, "max_output_tokens": 128000},
    {"model": "large", "context_window": 1000000, "max_output_tokens": "128000"},
    {"model": "large", "context_window": 1000000, "max_output_tokens": 8192},
])
async def test_unknown_mismatched_or_foreground_limited_profile_stays_conservative(defaults):
    context, model = await manager(), provider(finish="end_turn")
    model.get_info = lambda: SimpleNamespace(defaults=defaults, capabilities=[])
    await context._prepare(model, None, [])
    assert {row["max_output_tokens"] for row in model.calls} == {8192}


@pytest.mark.asyncio
async def test_explicit_allowance_is_not_lowered_by_foreground_default():
    context, model = await manager(summary_max_output_tokens=32768), provider(finish="end_turn")
    model.get_info = lambda: SimpleNamespace(defaults={
        "model": "large", "context_window": 1000000, "max_output_tokens": 8192}, capabilities=[])
    await context._prepare(model, None, [])
    assert {row["max_output_tokens"] for row in model.calls} == {32768}


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [None, True, 0, -1, "32768", 32768.0])
async def test_invalid_explicit_allowance_fails_before_provider_dispatch(cap):
    context, model = await manager(summary_max_output_tokens=cap), provider()
    with pytest.raises(CompactionError):
        await context._prepare(model, None, [])
    assert model.calls == [] and model.counts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cost", "model", "history", "cancel"])
async def test_metadata_await_cannot_dispatch_under_changed_contract(change, monkeypatch):
    # The legacy foreground fitter has a synchronous metadata API. Exercise
    # this new asynchronous boundary without asking that unrelated fitter to
    # consume a coroutine.
    from amplifier_module_context_managed.boundary import RequestContext
    monkeypatch.setattr(RequestContext, "_calculate_budget", lambda *args: 10000)
    context, model = await manager(), provider()
    entered, release = asyncio.Event(), asyncio.Event()
    original_info = model.get_info()
    original = await context.get_messages()

    async def get_info():
        entered.set()
        await release.wait()
        return original_info

    model.get_info = get_info
    task = asyncio.create_task(context._summarize(model, original, context.revision,
        {"calls": 0}, fingerprint="fixture",
        source_revision={"messages": len(original), "sha256": digest(original)}, identity=IDENTITY))
    await entered.wait()
    if change == "cost":
        context.config["summary_max_output_tokens"] = 4096
    elif change == "model":
        model.default_model = "other"
    elif change == "history":
        await context.add_message({"role": "user", "content": "new original"})
    else:
        task.cancel()
    release.set()
    with pytest.raises((CompactionError, RuntimeError, asyncio.CancelledError)):
        await task
    assert not model.calls and not model.counts
    assert context.export_checkpoint(IDENTITY) is None
    assert (await context.get_messages())[:len(original)] == original


@pytest.mark.asyncio
async def test_dispatch_output_refusal_is_not_repaired_by_splitting():
    context, model = await manager(), provider()
    await context.set_messages([{"role": "user", "content": "SOURCE" * 300}] + rows()[1:])
    counts_at_dispatch = []

    async def refusal(request, **kwargs):
        counts_at_dispatch.append(len(model.counts))
        model.calls.append(request.model_dump())
        raise ContextLengthError("Output ceiling is smaller than requested")

    model.complete = refusal
    with pytest.raises(CompactionError):
        await context._prepare(model, None, [])
    assert len(model.calls) == 1
    assert len(model.counts) == counts_at_dispatch[0]  # no fitting/retry after refusal
    assert context.summary is None


@pytest.mark.asyncio
async def test_larger_output_reservation_can_refuse_a_previously_fitting_request():
    model = provider(37000, 128000, finish="end_turn")
    # Advisory metadata selects the workload, not the actual assembled limit.
    model.get_info = lambda: SimpleNamespace(defaults={
        "model": "large", "context_window": 1000000, "max_output_tokens": 128000})
    resolved, _ = await summary_config(model, {})
    request = summary_request("required instructions " * 40, "source " * 1000, "previous note", resolved)
    fits, budget = await request_fits(model, request)
    assert not fits and budget["output_reserve"] == 32768
    small = request.model_copy(update={"max_output_tokens": 8192})
    assert (await request_fits(model, small))[0]
    assert model.calls == []


@pytest.mark.asyncio
async def test_legacy_unbound_partial_progress_is_refused_without_automatic_recomputation():
    context, model = await manager(summary_max_calls=1), provider()
    with pytest.raises(CompactionError):
        await context._prepare(model, None, [])
    record = context.export_checkpoint(IDENTITY)
    record["configurationVersion"] = 2
    record["progress"].pop("generation_contract")
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    cold = await manager()
    assert cold.restore_checkpoint(record, IDENTITY)["status"] == "resuming"
    replacement = provider()
    before = copy.deepcopy(cold._restored_progress)
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        await cold._prepare(replacement, None, [])
    assert replacement.calls == [] and replacement.counts == []
    assert cold.summary is None and cold._restored_progress == before
