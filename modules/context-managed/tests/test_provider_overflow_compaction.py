"""A rejected wire view can be huge while its public semantic view is small."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import ContextLengthError

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.checkpoint import digest
from amplifier_module_context_managed.errors import CompactionError


def history():
    return [
        {"role": "user", "content": "Build the report. Preserve all source evidence."},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "",
             "signature": "SIGNED-SYNTHETIC-STATE" * 200_000},
            {"type": "text", "text": "Public verified research. " * 500},
        ], "tool_calls": [{"id": "old-tool", "name": "research", "arguments": {"topic": "report"}}]},
        {"role": "tool", "tool_call_id": "old-tool", "content": "Original tool receipt. " * 100},
        {"role": "user", "content": "Keep all originals."},
        {"role": "assistant", "content": "Second turn completed."},
        {"role": "user", "content": "Continue the current task."},
    ]


def model():
    return SimpleNamespace(name="fixture", default_model="fixture-model",
        complete=AsyncMock(return_value=SimpleNamespace(content=[
            SimpleNamespace(type="text", text="Objective: report. Research complete. Preserve originals.")
        ])))


def counter(calls, *, unavailable=False):
    async def count(view):
        request = ChatRequest(messages=[Message(**row) for row in view])
        calls.append(request)
        opaque = "SIGNED-SYNTHETIC-STATE" in str(view)
        decision = None if unavailable or opaque else {
            "estimated_input_tokens": 2_000, "input_limit_tokens": 1_000_000,
            "context_token_budget": 0,
            "measurement": {"kind": "provider_count", "source": "fixture.count", "input_tokens": 2_000},
        }
        return {"dispatch": request, "budget_decision": decision}
    return count


@pytest.mark.asyncio
async def test_overflow_reason_compacts_low_public_pressure_and_keeps_exact_originals():
    config = {"max_tokens": 1_000_000, "summarize_trigger": .7, "durable_checkpoints": True}
    context = BoundaryContextManager(config)
    identity = {"provider": "fixture", "model": "fixture-model"}
    context.checkpoint_identity = lambda: copy.deepcopy(identity)
    original = history()
    original_hash = digest(original)
    receipt_hash = digest(original[2])
    await context.set_messages(original)
    provider = model()
    calls = []

    # This is the missing case: public pressure is below the normal trigger,
    # and the full initial wire view cannot currently be authoritatively counted.
    first = await context.get_measured_request_view(
        provider=provider, retain_contents=[], count_view=counter(calls))
    assert first["final_attempt"]["budget_decision"] is None
    assert provider.complete.await_count == 0

    result = await context.get_measured_request_view(
        provider=provider, retain_contents=[], count_view=counter(calls),
        reason="provider_overflow")

    assert provider.complete.await_count >= 1
    summary_requests = str(provider.complete.call_args_list)
    assert "Public verified research." in summary_requests
    assert "PRIVATE-SYNTHETIC-STATE" not in summary_requests
    assert "SIGNED-SYNTHETIC-STATE" not in summary_requests
    assert result["final_attempt"]["dispatch"] is calls[-1]
    assert result["final_attempt"]["budget_decision"]["measurement"]["input_tokens"] == 2_000
    assert "Continuation note" in str(result["base_view"])
    assert "SIGNED-SYNTHETIC-STATE" not in str(result["base_view"])
    assert digest(await context.get_messages()) == original_hash
    assert digest((await context.get_messages())[2]) == receipt_hash

    checkpoint = context.export_checkpoint(identity)
    restored = BoundaryContextManager(config)
    restored.checkpoint_identity = context.checkpoint_identity
    await restored.set_messages(original)
    assert restored.restore_checkpoint(checkpoint, identity)["status"] == "restored"
    next_view = await restored.get_measured_request_view(
        provider=provider, retain_contents=[], count_view=counter([]))
    assert next_view["base_view"] == result["base_view"]
    assert digest(await restored.get_messages()) == original_hash


@pytest.mark.asyncio
async def test_overflow_reason_does_not_trim_protected_tail():
    context = BoundaryContextManager({"max_tokens": 1_000_000})
    original = [{"role": "user", "content": "Only current protected turn."}]
    await context.set_messages(original)

    async def count(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]),
            "budget_decision": {"estimated_input_tokens": 1_000_001, "input_limit_tokens": 1_000_000,
                "measurement": {"kind": "provider_count", "source": "fixture.count", "input_tokens": 1_000_001}}}

    with pytest.raises(ContextLengthError):
        await context.get_measured_request_view(
            provider=model(), retain_contents=[], count_view=count, reason="provider_overflow")
    assert await context.get_messages() == original
    assert context.summary is None


@pytest.mark.asyncio
async def test_unknown_compaction_reason_is_rejected_before_work():
    context = BoundaryContextManager()
    provider = model()
    with pytest.raises(ValueError, match="reason"):
        await context.get_measured_request_view(
            provider=provider, retain_contents=[], count_view=counter([]), reason="generic_retry")
    provider.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_overflow_reason_keeps_native_count_requirement_without_portable_fallback():
    context = BoundaryContextManager({"max_tokens": 1_000_000})
    original = history()
    await context.set_messages(original)
    provider = model()
    provider.supports_native_compaction = lambda: True
    provider.native_compaction_requires_request_context = True
    provider.validate_compacted_context = lambda row: True
    provider.compact_context = AsyncMock()
    with pytest.raises(CompactionError, match="Native compaction"):
        await context.get_measured_request_view(
            provider=provider, retain_contents=[], count_view=counter([], unavailable=True),
            reason="provider_overflow")
    provider.complete.assert_not_awaited()
    provider.compact_context.assert_not_awaited()
    assert await context.get_messages() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["count", "summary"])
async def test_cancellation_in_forced_preparation_keeps_canonical_and_prior_checkpoint(phase):
    context = BoundaryContextManager({"max_tokens": 1_000_000})
    original = history()
    await context.set_messages(original)
    entered = asyncio.Event()
    provider = model()

    async def pending(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    count = counter([])
    if phase == "count":
        count = pending
    else:
        provider.complete.side_effect = pending
    task = asyncio.create_task(context.get_measured_request_view(
        provider=provider, retain_contents=[], count_view=count, reason="provider_overflow"))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await context.get_messages() == original
    assert context.summary is None
    assert not context.is_compacting


@pytest.mark.asyncio
async def test_completed_semantic_checkpoint_survives_unavailable_final_count_and_restore():
    config = {"max_tokens": 1_000_000, "durable_checkpoints": True}
    identity = {"provider": "fixture", "model": "fixture-model"}
    context = BoundaryContextManager(config)
    context.checkpoint_identity = lambda: copy.deepcopy(identity)
    original = history()
    await context.set_messages(original)
    provider = model()
    uncounted = await context.get_measured_request_view(
        provider=provider, retain_contents=[], count_view=counter([], unavailable=True),
        reason="provider_overflow")
    # Context prepared a source-validated note; this is NOT retry admission.
    # The Loop must reject this final None decision without committing a request.
    assert uncounted["final_attempt"]["budget_decision"] is None
    assert uncounted["transaction"] is None
    assert context.summary is not None
    calls = provider.complete.await_count
    record = context.export_checkpoint(identity)
    restored = BoundaryContextManager(config)
    restored.checkpoint_identity = context.checkpoint_identity
    await restored.set_messages(original)
    assert restored.restore_checkpoint(record, identity)["status"] == "restored"
    counted = await restored.get_measured_request_view(
        provider=provider, retain_contents=[], count_view=counter([]))
    assert counted["final_attempt"]["budget_decision"]["measurement"]["kind"] == "provider_count"
    assert provider.complete.await_count == calls
    assert await restored.get_messages() == original


@pytest.mark.asyncio
async def test_forced_native_compaction_does_not_skip_small_new_eligible_prefix():
    context = BoundaryContextManager({"max_tokens": 1_000_000, "native_min_new_tokens": 500})
    original = history()
    await context.set_messages(original)
    provider = model()
    provider.supports_native_compaction = lambda: True
    provider.native_compaction_requires_request_context = True
    provider.validate_compacted_context = lambda row: bool((row.get("metadata") or {}).get("native"))
    rounds = []

    async def compact(request):
        rounds.append(request)
        return {"kind": "native", "message": {
            "role": "user", "content": "Native checkpoint",
            "metadata": {"source": "context-managed", "native": len(rounds)}}}

    provider.compact_context = compact

    async def count(view):
        native = next(((row.get("metadata") or {}).get("native") for row in view
                       if (row.get("metadata") or {}).get("native")), 0)
        tokens = 10_000 if not native else 1_000 if native == 1 else 900
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]),
            "budget_decision": {"estimated_input_tokens": tokens, "input_limit_tokens": 1_000_000,
                "measurement": {"kind": "provider_count", "source": "fixture.count", "input_tokens": tokens}}}

    await context.get_measured_request_view(
        provider=provider, retain_contents=[], count_view=count, reason="provider_overflow")
    assert len(rounds) == 1
    # The next eligible prefix is tiny and below both ordinary native triggers,
    # but a consumed provider rejection explicitly requests legal compaction.
    await context.add_message({"role": "assistant", "content": "Recent result."})
    await context.add_message({"role": "user", "content": "Next human turn."})
    result = await context.get_measured_request_view(
        provider=provider, retain_contents=[], count_view=count, reason="provider_overflow")
    assert len(rounds) == 2
    provider.complete.assert_not_awaited()
    assert result["final_attempt"]["budget_decision"]["measurement"]["input_tokens"] == 900
    assert (await context.get_messages())[:len(original)] == original