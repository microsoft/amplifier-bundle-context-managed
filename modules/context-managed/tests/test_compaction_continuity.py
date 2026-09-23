import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import ContextLengthError, LLMError
from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.checkpoint import digest
from amplifier_module_context_managed.summary import add_usage, public_messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [{}, {"source": "context-managed", "ephemeral": True, "persisted": True}],
)
async def test_recomputed_checkpoint_digest_cannot_hide_missing_native_transport(
    metadata,
):
    manager = context(durable_checkpoints=True, summarize_trigger=100)
    identity = {"provider": "openai", "model": "same-model"}
    manager.checkpoint_identity = lambda: identity
    original = history()
    await manager.set_messages(original)
    manager.summary_identity = identity
    manager.summary = (
        4,
        {"role": "user", "content": "Placeholder", "metadata": metadata},
    )
    checkpoint = manager.export_checkpoint(identity)
    checkpoint["sha256"] = digest(
        {key: value for key, value in checkpoint.items() if key != "sha256"}
    )
    manager.restore_checkpoint(checkpoint, identity)
    model = SimpleNamespace(
        supports_native_compaction=lambda: True,
        validate_compacted_context=lambda message: bool(
            message.get("metadata", {}).get("openai:compaction")
        ),
        request_budget=lambda request, **kw: {
            "measurement": {"kind": "provider_count", "input_tokens": 100}
        },
        compact_context=AsyncMock(),
    )
    fitter, _ = await manager._prepare(model, None, [])
    assert manager.summary is None
    assert manager.checkpoint_status["status"] == "rejected"
    assert without_fitter_sequence(await fitter.get_messages()) == original
    assert await manager.get_messages() == original
    model.compact_context.assert_not_awaited()


def test_usage_preserves_cached_buckets_and_partial_cost_coverage():
    stats = {}
    add_usage(
        stats,
        {
            "input_tokens": 3,
            "output_tokens": 20,
            "cache_write_tokens": 100,
            "cost_usd": "0.001",
        },
    )
    add_usage(stats, {"input_tokens": 50, "output_tokens": 30, "cache_read_tokens": 40})
    assert stats == {
        "usage_calls": 2,
        "priced_calls": 1,
        "usage": {
            "input_tokens": 53,
            "output_tokens": 50,
            "cache_write_tokens": 100,
            "cache_read_tokens": 40,
            "cost_usd": "0.001",
        },
    }


def without_fitter_sequence(rows):
    rows = copy.deepcopy(rows)
    for row in rows:
        if "metadata" in row:
            row["metadata"].pop("_seq", None)
            if not row["metadata"]:
                del row["metadata"]
    return rows


def response(text):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage={"input_tokens": 100, "output_tokens": 10},
    )


def history():
    return [
        {"role": "system", "content": "Never replay completed tools."},
        {"role": "developer", "content": "Keep unverified work separate."},
        {"role": "user", "content": "Deliver project ORBIT; budget $40."},
        {"role": "assistant", "content": "Historical verified material. " * 2000},
        {
            "role": "user",
            "content": "Correction: budget $25; leave originals untouched.",
        },
        {"role": "assistant", "content": "Continuing verified research. " * 100},
        {"role": "user", "content": "Next, finish the report."},
    ]


def context(**config):
    return BoundaryContextManager(
        {"max_tokens": 10000, "summarize_trigger": 0.1, **config}
    )


def test_public_projection_omits_private_state_and_duplicate_calls_without_mutation():
    call = {
        "type": "tool_call",
        "id": "c1",
        "name": "read",
        "arguments": {"path": "proof.txt"},
    }
    original = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "Verified file."},
                call,
            ],
            "tool_calls": [call],
            "thinking_block": "private",
            "metadata": {
                "openai:reasoning_items": [{"encrypted_content": "opaque"}],
                "amplifier_input": {"kind": "service", "version": 1},
            },
        }
    ]
    before = copy.deepcopy(original)
    result = public_messages(original)
    serialized = json.dumps(result)
    assert "private" not in serialized and "opaque" not in serialized
    assert serialized.count('"id": "c1"') == 1
    assert result[0]["metadata"]["amplifier_input"]["kind"] == "service"
    assert original == before


@pytest.mark.asyncio
async def test_huge_tool_evidence_splits_without_budget_capability_and_commits_once():
    original = history()
    payload = (
        "EDGE_START=alpha "
        + "padding " * 2000
        + " EDGE_END=omega; LATEST_CORRECTION=25"
    )
    original[3:4] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "read1", "name": "read", "arguments": {}}],
        },
        {"role": "tool", "tool_call_id": "read1", "content": payload},
    ]
    requests = []

    async def complete(request):
        requests.append(request)
        # An older provider offers no request_budget capability, and rejects
        # the first oversized fragment. The engine must adapt within this run.
        if len(request.messages[-1].content) > 2000:
            raise ContextLengthError("input exceeds fixture allowance")
        source = request.messages[-1].content
        facts = [
            fact
            for fact in ("EDGE_START=alpha", "EDGE_END=omega", "LATEST_CORRECTION=25")
            if fact in source
        ]
        return response("Objective ORBIT. " + "; ".join(facts))

    model = SimpleNamespace(complete=complete)
    manager = context(summary_max_source_chars=4000, summary_max_calls=64)
    await manager.set_messages(original)
    await manager.get_messages_for_request(provider=model)
    assert manager.summary is not None
    assert "EDGE_START=alpha" in manager.summary[1]
    assert "EDGE_END=omega" in manager.summary[1]
    assert "LATEST_CORRECTION=25" in manager.summary[1]
    assert len(requests) > 2
    assert all(
        request.metadata["stream"] is False and request.reasoning_effort == "low"
        for request in requests
    )
    assert await manager.get_messages() == original


@pytest.mark.asyncio
async def test_permanent_failure_on_unchanged_prefix_is_not_retried_for_each_tool_result():
    complete = AsyncMock(
        side_effect=LLMError("secret and prompt must not escape", retryable=False)
    )
    hooks = SimpleNamespace(emit=AsyncMock())
    manager = context()
    manager.hooks = hooks
    original = history()
    await manager.set_messages(original)
    model = SimpleNamespace(complete=complete)
    for number in range(10):
        await manager.get_messages_for_request(provider=model)
        await manager.add_message(
            {"role": "assistant", "content": f"Observed progress {number}"}
        )
    assert complete.await_count == 1
    event = next(
        call.args[1]
        for call in hooks.emit.call_args_list
        if call.args[0] == "context:compaction_finished"
    )
    assert event["failure"] == {
        "type": "LLMError",
        "retryable": False,
        "attempt": 1,
        "retry_suppressed": True,
    }
    assert "secret" not in json.dumps(event)
    assert manager.summary is None
    await manager.add_message(
        {"role": "user", "content": "New boundary and explicit correction"}
    )
    await manager.get_messages_for_request(provider=model)
    assert complete.await_count == 2


@pytest.mark.asyncio
async def test_transient_failure_has_cooldown_and_three_attempt_ceiling(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        "amplifier_module_context_managed.boundary.time.monotonic", lambda: now[0]
    )
    complete = AsyncMock(side_effect=LLMError("temporary", retryable=True))
    model = SimpleNamespace(complete=complete)
    manager = context(summary_retry_delay=10)
    await manager.set_messages(history())
    for stamp, expected in [(0, 1), (1, 1), (10, 2), (20, 2), (30, 3), (10000, 3)]:
        now[0] = stamp
        await manager.get_messages_for_request(provider=model)
        assert complete.await_count == expected


@pytest.mark.asyncio
async def test_partial_note_never_commits_when_later_fragment_fails():
    complete = AsyncMock(
        side_effect=[
            response("Partial, not authoritative"),
            LLMError("permanent", retryable=False),
        ]
    )
    manager = context(summary_max_source_chars=1000)
    original = history()
    await manager.set_messages(original)
    await manager.get_messages_for_request(provider=SimpleNamespace(complete=complete))
    assert complete.await_count == 2
    assert manager.summary is None
    assert await manager.get_messages() == original


@pytest.mark.asyncio
async def test_native_window_is_preserved_across_repeat_checkpoint_and_resume():
    manager = context(durable_checkpoints=True)
    identity = {"provider": "openai", "model": "gpt-6-astra"}
    manager.checkpoint_identity = lambda: identity
    originals = history()
    await manager.set_messages(originals)
    windows = []

    async def compact(request):
        window = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "original"}],
            },
            {"type": "compaction", "encrypted_content": f"opaque-{len(windows)}"},
        ]
        windows.append(copy.deepcopy(window))
        return {
            "kind": "native",
            "message": {
                "role": "user",
                "content": "Native checkpoint",
                "metadata": {
                    "source": "context-managed",
                    "ephemeral": True,
                    "persisted": True,
                    "openai:compaction": {
                        "version": 1,
                        "model": identity["model"],
                        "output": window,
                    },
                },
            },
            "usage": {"input_tokens": 500, "output_tokens": 50},
        }

    model = SimpleNamespace(
        supports_native_compaction=lambda: True,
        validate_compacted_context=lambda message: True,
        compact_context=compact,
        complete=AsyncMock(
            side_effect=AssertionError("semantic fallback was not requested")
        ),
    )

    def count(request, **kwargs):
        count = sum(len(str(row.model_dump())) // 4 for row in request.messages)
        return {"measurement": {"kind": "provider_count", "input_tokens": count}}

    model.request_budget = count
    first = await manager.get_messages_for_request(provider=model)
    assert "Correction: budget $25" in str(first)
    for row in [
        {"role": "assistant", "content": "Additional results " * 1000},
        {"role": "user", "content": "Latest correction: ship on Tuesday, not Monday."},
    ]:
        await manager.add_message(row)
        originals.append(row)
    second = await manager.get_messages_for_request(provider=model)
    assert len(windows) == 2
    assert "ship on Tuesday" in str(second)
    checkpoint = manager.export_checkpoint(identity)
    restored = context(durable_checkpoints=True)
    restored.checkpoint_identity = lambda: identity
    await restored.set_messages(originals)
    assert restored.restore_checkpoint(checkpoint, identity)["status"] == "restored"
    assert restored.summary == manager.summary
    assert await restored.get_messages_for_request(provider=model) == second
    assert await manager.get_messages() == originals == await restored.get_messages()
    assert model.complete.await_count == 0


@pytest.mark.asyncio
async def test_native_uses_continuation_provider_not_different_utility_model():
    manager = context(summarization_model="utility-model")
    await manager.set_messages(history())
    utility = SimpleNamespace(
        default_model="utility-model", complete=AsyncMock(), compact_context=AsyncMock()
    )
    manager.summary_provider = lambda: utility
    native_message = {
        "role": "user",
        "content": "Native state",
        "metadata": {
            "source": "context-managed",
            "ephemeral": True,
            "persisted": True,
            "opaque": "retained complete canonical window",
        },
    }
    main = SimpleNamespace(
        default_model="main-model",
        supports_native_compaction=lambda: True,
        validate_compacted_context=lambda message: True,
        compact_context=AsyncMock(
            return_value={"kind": "native", "message": native_message}
        ),
        request_budget=lambda request, **kw: {
            "measurement": {
                "kind": "provider_count",
                "input_tokens": 100
                if any((row.metadata or {}).get("opaque") for row in request.messages)
                else 9000,
            }
        },
    )
    await manager.get_messages_for_request(provider=main)
    main.compact_context.assert_awaited_once()
    assert main.compact_context.call_args.args[0].model is None
    utility.compact_context.assert_not_awaited()
    utility.complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native_finishes", [True, False])
async def test_native_and_semantic_compaction_have_independent_deadlines(native_finishes):
    # Model the production full-window native call exceeding the portable
    # summary timeout, without sending production history or waiting minutes.
    manager = context(summary_timeout=0.01 if native_finishes else 1,
                      native_compaction_timeout=1 if native_finishes else 0.01)
    original = history()
    await manager.set_messages(original)
    manager.hooks = SimpleNamespace(emit=AsyncMock())
    message = {"role":"user", "content":"Native continuation", "metadata":{
        "source":"context-managed", "ephemeral":True, "persisted":True, "opaque":"fixture"}}
    async def compact(request):
        await asyncio.sleep(0.03)
        return {"kind":"native", "message":message}
    async def summarize(request):
        await asyncio.sleep(0.03)
        return response("ORBIT; budget $25; originals preserved; report pending.")
    model = SimpleNamespace(
        supports_native_compaction=lambda:True,
        validate_compacted_context=lambda value:True,
        compact_context=AsyncMock(side_effect=compact),
        complete=AsyncMock(side_effect=summarize),
        request_budget=lambda request, **kw:{"measurement":{"kind":"provider_count", "input_tokens":
            100 if any((row.metadata or {}).get("opaque") for row in request.messages) else 9000}})
    await manager.get_messages_for_request(provider=model)
    assert await manager.get_messages() == original
    assert manager.summary is not None
    finished = next(call.args[1] for call in manager.hooks.emit.call_args_list
                    if call.args[0] == "context:compaction_finished")
    assert finished["outcome"] == "completed"
    if native_finishes:
        assert manager.summary[1] == message
        model.complete.assert_not_awaited()
        assert finished["method"] == "native"
    else:
        assert isinstance(manager.summary[1], str)
        assert model.complete.await_count >= 1
        assert finished["method"] == "semantic"
        assert finished["native_failure"] == {"type":"TimeoutError", "stage":"native", "timeout_seconds":0.01}


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False])
async def test_compaction_waits_for_provider_without_a_default_deadline(monkeypatch, native):
    # Compress any accidentally restored finite deadline, so the old 120-second
    # default fails this regression without spending two minutes in the test.
    actual_wait_for = asyncio.wait_for
    deadlines = []
    async def accelerated_wait_for(awaitable, timeout):
        deadlines.append(timeout)
        return await actual_wait_for(awaitable, 0.001 if timeout is not None else None)
    monkeypatch.setattr(asyncio, "wait_for", accelerated_wait_for)
    manager = context()
    original = history()
    await manager.set_messages(original)
    message = {"role":"user", "content":"Native continuation", "metadata":{
        "source":"context-managed", "ephemeral":True, "persisted":True, "opaque":"fixture"}}
    async def compact(request):
        await asyncio.sleep(0.01)
        return {"kind":"native", "message":message}
    async def summarize(request):
        await asyncio.sleep(0.01)
        return response("ORBIT; budget $25; originals preserved; report pending.")
    model = SimpleNamespace(
        supports_native_compaction=lambda:native,
        validate_compacted_context=lambda value:True,
        compact_context=AsyncMock(side_effect=compact),
        complete=AsyncMock(side_effect=summarize),
        request_budget=lambda request, **kw:{"measurement":{"kind":"provider_count", "input_tokens":
            100 if any((row.metadata or {}).get("opaque") for row in request.messages) else 9000}})
    await manager.get_messages_for_request(provider=model)
    assert deadlines and all(value is None for value in deadlines)
    assert manager.summary is not None
    assert await manager.get_messages() == original
    if native:
        model.complete.assert_not_awaited()
        assert manager.summary[1] == message
    else:
        model.compact_context.assert_not_awaited()
        assert isinstance(manager.summary[1], str)


@pytest.mark.asyncio
@pytest.mark.parametrize("summary_fails", [False, True])
async def test_native_failure_attempts_text_summary_before_last_resort_fitting(summary_fails):
    calls = []
    async def compact(request):
        calls.append("native")
        raise ConnectionError("provider connection failed")
    async def summarize(request):
        calls.append("summary")
        if summary_fails:
            raise LLMError("provider rejected summary", retryable=False)
        return response("ORBIT; budget $25; originals preserved; report pending.")
    model = SimpleNamespace(
        supports_native_compaction=lambda:True,
        validate_compacted_context=lambda value:True,
        compact_context=compact, complete=summarize,
        request_budget=lambda request, **kw:{"measurement":{"kind":"provider_count", "input_tokens":9000}})
    manager = context(summary_max_calls=1)
    manager.hooks = SimpleNamespace(emit=AsyncMock())
    original = history()
    await manager.set_messages(original)
    result = await manager.get_messages_for_request(provider=model)
    assert calls == ["native", "summary"]
    assert await manager.get_messages() == original
    finished = next(call.args[1] for call in manager.hooks.emit.call_args_list
                    if call.args[0] == "context:compaction_finished")
    assert finished["native_failure"]["type"] == "ConnectionError"
    assert finished["method"] == "semantic"
    if summary_fails:
        assert manager.summary is None
        assert finished["outcome"] == "fallback"
        assert len(json.dumps(result)) < len(json.dumps(original))
    else:
        assert manager.summary is not None
        assert finished["outcome"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False])
async def test_user_cancellation_during_compaction_does_not_start_fallback(native):
    entered = asyncio.Event()
    async def compact(request):
        entered.set()
        await asyncio.Event().wait()
    model = SimpleNamespace(supports_native_compaction=lambda:native,
        validate_compacted_context=lambda value:True, compact_context=AsyncMock(side_effect=compact),
        complete=AsyncMock(side_effect=compact),
        request_budget=lambda request, **kw:{"measurement":{"kind":"provider_count", "input_tokens":9000}})
    manager = context()
    original = history()
    await manager.set_messages(original)
    task = asyncio.create_task(manager.get_messages_for_request(provider=model))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.complete.await_count == (0 if native else 1)
    assert model.compact_context.await_count == (1 if native else 0)
    assert manager.summary is None
    assert await manager.get_messages() == original


@pytest.mark.asyncio
async def test_large_retained_native_window_is_counted_not_its_small_label():
    manager = context(native_min_new_tokens=0)
    await manager.set_messages(history())
    main = SimpleNamespace(
        default_model="main-model",
        supports_native_compaction=lambda: True,
        validate_compacted_context=lambda message: True,
        complete=AsyncMock(return_value=response("Portable factual note.")),
    )
    native_message = {
        "role": "user",
        "content": "Short label",
        "metadata": {
            "source": "context-managed",
            "ephemeral": True,
            "persisted": True,
            "opaque": "large-window",
        },
    }
    main.compact_context = AsyncMock(
        return_value={"kind": "native", "message": native_message}
    )
    # The returned window is bigger, despite its short carrier label. Reject it
    # and use semantic fallback rather than committing a false reduction.
    main.request_budget = lambda request, **kw: {
        "measurement": {
            "kind": "provider_count",
            "input_tokens": 20000
            if any((row.metadata or {}).get("opaque") for row in request.messages)
            else 9000,
        }
    }
    await manager.get_messages_for_request(provider=main)
    assert isinstance(manager.summary[1], str)
    assert main.complete.await_count >= 1
    # A previously valid opaque checkpoint grows above the next trigger when
    # new content arrives; the complete measured window must trigger again.
    manager.summary = (4, native_message)
    for row in [
        {"role": "assistant", "content": "More work"},
        {"role": "user", "content": "Continue"},
    ]:
        await manager.add_message(row)
    await manager.get_messages_for_request(provider=main)
    assert main.compact_context.await_count == 2


@pytest.mark.asyncio
async def test_old_provider_after_restart_rebuilds_opaque_checkpoint_from_originals():
    config = {"durable_checkpoints": True, "summarize_trigger": 100}
    manager = context(**config)
    identity = {"provider": "openai", "model": "same-model"}
    manager.checkpoint_identity = lambda: identity
    original = history()
    await manager.set_messages(original)
    manager.summary = (
        4,
        {
            "role": "user",
            "content": "Short opaque placeholder",
            "metadata": {
                "source": "context-managed",
                "persisted": True,
                "ephemeral": True,
                "openai:compaction": {
                    "version": 1,
                    "model": "same-model",
                    "output": [],
                },
            },
        },
    )
    manager.summary_identity = identity
    saved = manager.export_checkpoint(identity)
    restored = context(**config)
    restored.checkpoint_identity = lambda: identity
    await restored.set_messages(original)
    assert restored.restore_checkpoint(saved, identity)["status"] == "restored"
    # Same configured provider/model, older plugin: no native transport method.
    older = SimpleNamespace(
        name="openai", default_model="same-model", complete=AsyncMock()
    )
    fitter, _ = await restored._prepare(older, None, [])
    assert restored.summary is None
    assert without_fitter_sequence(await fitter.get_messages()) == original
    assert await manager.get_messages() == original
    older.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_raising_native_measurement_rebuilds_originals_but_cancellation_propagates():
    manager = context(summarize_trigger=100)
    original = history()
    await manager.set_messages(original)
    identity = {"provider": "openai", "model": "same-model"}
    manager.checkpoint_identity = lambda: identity
    carrier = {
        "role": "user",
        "content": "Opaque checkpoint",
        "metadata": {"source": "context-managed"},
    }
    manager.summary, manager.summary_identity = (4, carrier), identity
    model = SimpleNamespace(
        supports_native_compaction=lambda: True,
        validate_compacted_context=lambda message: True,
        request_budget=AsyncMock(side_effect=ValueError("unusable encrypted state")),
        compact_context=AsyncMock(),
    )
    fitter, _ = await manager._prepare(model, None, [])
    assert manager.summary is None
    assert manager.checkpoint_status["status"] == "rejected"
    assert without_fitter_sequence(await fitter.get_messages()) == original
    assert await manager.get_messages() == original
    model.compact_context.assert_not_awaited()
    manager.summary, manager.summary_identity = (4, carrier), identity
    model.request_budget.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await manager._prepare(model, None, [])
    assert manager.summary == (4, carrier)
    assert await manager.get_messages() == original


@pytest.mark.asyncio
async def test_tiny_native_prefix_waits_unless_full_request_requires_fitting():
    manager = context()
    original = history()
    await manager.set_messages(original)
    carrier = {
        "role": "user",
        "content": "Opaque checkpoint",
        "metadata": {"source": "context-managed"},
    }
    model = SimpleNamespace(
        name="openai",
        default_model="main",
        supports_native_compaction=lambda: True,
        validate_compacted_context=lambda message: True,
        compact_context=AsyncMock(return_value={"kind": "native", "message": carrier}),
    )
    manager.summary_identity = {"provider": "openai", "model": "main"}
    manager.summary = (4, carrier)
    # The next eligible turn is tiny while a recent protected turn is large.
    manager.messages[5]["content"] = "OK"
    await manager.add_message(
        {"role": "assistant", "content": "Recent material " * 3000}
    )
    await manager.add_message({"role": "user", "content": "Continue"})
    counts = [5000]
    model.request_budget = lambda request, **kw: {
        "measurement": {"kind": "provider_count", "input_tokens": counts[0]}
    }
    await manager._prepare(model, None, [])
    model.compact_context.assert_not_awaited()
    # An actually over-budget request may compact even a small eligible prefix.
    counts[0] = 20000

    async def compact(request):
        counts[0] = 100
        return {"kind": "native", "message": carrier}

    model.compact_context.side_effect = compact
    await manager._prepare(model, None, [])
    assert model.compact_context.await_count == 1
    # Advancing a substantial safe prefix permits the next boundary as well.
    await manager.add_message(
        {"role": "assistant", "content": "Later material " * 2000}
    )
    await manager.add_message({"role": "user", "content": "Next"})
    counts[0] = 5000
    await manager._prepare(model, None, [])
    assert model.compact_context.await_count == 2


@pytest.mark.asyncio
async def test_required_reminder_occurs_once_outside_native_window():
    manager = context()
    original = history()
    reminder = {
        "role": "user",
        "content": "Required pending work reminder",
        "metadata": {"ephemeral": True, "persisted": True},
    }
    original.insert(3, reminder)
    await manager.set_messages(original)
    carrier = {
        "role": "user",
        "content": "Native checkpoint",
        "metadata": {"source": "context-managed"},
    }
    model = SimpleNamespace(
        supports_native_compaction=lambda: True,
        validate_compacted_context=lambda message: True,
        compact_context=AsyncMock(return_value={"kind": "native", "message": carrier}),
        request_budget=lambda request, **kw: {
            "measurement": {
                "kind": "provider_count",
                "input_tokens": 100
                if any(
                    (row.metadata or {}).get("source") == "context-managed"
                    for row in request.messages
                )
                else 9000,
            }
        },
    )
    fitter, _ = await manager._prepare(model, None, [reminder["content"]])
    request = model.compact_context.call_args.args[0]
    assert all(row.content != reminder["content"] for row in request.messages)
    assert without_fitter_sequence(await fitter.get_messages()).count(reminder) == 1
    assert await manager.get_messages() == original
