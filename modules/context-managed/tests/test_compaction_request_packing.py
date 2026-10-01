"""Native windows stay whole; portable notes split only when their budget requires it."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message, ToolSpec
from amplifier_module_context_managed.boundary import BoundaryContextManager


def originals(size=600_000):
    return [
        {"role": "system", "content": "Preserve original evidence."},
        {"role": "user", "content": "FIRST objective: preserve ORBIT."},
        {"role": "assistant", "content": "FIRST evidence " + "x" * size + " LAST evidence"},
        {"role": "user", "content": "Latest correction: budget 25."},
        {"role": "assistant", "content": "Recent verified work."},
        {"role": "user", "content": "Current request: finish the report."},
    ]


def decision(count, limit=1_000_000, kind="provider_count"):
    return {"estimated_input_tokens": count, "input_limit_tokens": limit,
            "measurement": {"kind": kind, "source": "fixture.full_request", "input_tokens": count}}


def reply():
    return SimpleNamespace(content=[SimpleNamespace(type="text", text="ORBIT, evidence preserved; report pending.")])


async def context(**config):
    manager = BoundaryContextManager({"max_tokens": 1_000_000, "summarize_trigger": .001,
                                      "compaction_notice_enabled": False, **config},
                                     SimpleNamespace(emit=AsyncMock()))
    await manager.set_messages(originals())
    return manager


def finished(manager):
    return [call.args[1] for call in manager.hooks.emit.call_args_list
            if call.args[0] == "context:compaction_finished"][-1]


@pytest.mark.asyncio
async def test_counted_fitting_prefix_is_one_portable_request_above_old_character_cap():
    manager = await context()
    source = await manager.get_messages()
    requests = []
    budgets = []

    async def budget(request, **kwargs):
        budgets.append(request.model_copy(deep=True))
        # The actual request includes summary instructions and output reserve.
        assert request.max_output_tokens == 1500
        return decision(sum(len(str(row.content)) for row in request.messages) // 4)

    async def complete(request):
        requests.append(request)
        assert request == budgets[-1]
        return reply()

    model = SimpleNamespace(request_budget=budget, complete=complete)
    await manager._prepare(model, None, [])
    assert len(requests) == len(budgets) == 1
    assert len(requests[0].messages[-1].content) > 512_000
    assert "FIRST evidence" in requests[0].messages[-1].content
    assert "LAST evidence" in requests[0].messages[-1].content
    assert "Current request" not in requests[0].messages[-1].content
    assert manager.summary is not None
    assert finished(manager)["completed_fragments"] == 1
    assert finished(manager)["last_request_budget"]["measurement_kind"] == "provider_count"
    assert finished(manager)["native_selection"] == "unsupported_by_provider"
    assert await manager.get_messages() == source


@pytest.mark.asyncio
async def test_oversized_portable_prefix_is_split_only_after_complete_request_preflight():
    manager = await context()
    requests, budgets = [], []

    async def budget(request, **kwargs):
        count = sum(len(str(row.content)) for row in request.messages) // 4
        budgets.append(count)
        return decision(count, limit=100_000)

    async def complete(request):
        assert sum(len(str(row.content)) for row in request.messages) // 4 <= 100_000
        requests.append(request)
        return reply()

    await manager._prepare(SimpleNamespace(request_budget=budget, complete=complete), None, [])
    assert budgets[0] > 100_000  # Rejected preflight never dispatches a completion.
    assert len(requests) == 2 and len(budgets) == 3
    assert "Previous continuation note" in requests[1].messages[-1].content
    assert manager.summary is not None
    assert finished(manager)["summary_calls_total"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["provider_estimate", "missing", "mismatched", "negative", "missing_source"])
async def test_untrusted_budget_cannot_lift_conservative_portable_limit(kind):
    manager = await context()
    requests = []

    def budget(request, **kwargs):
        result = decision(100, kind=kind)
        if kind == "missing":
            return {}
        if kind == "mismatched":
            result["measurement"]["kind"] = "provider_count"
            result["measurement"]["input_tokens"] = 50
        if kind == "negative":
            result = decision(-1)
        if kind == "missing_source":
            result["measurement"] = {"kind": "provider_count", "input_tokens": 100}
        return result

    async def complete(request):
        requests.append(request)
        return reply()

    await manager._prepare(SimpleNamespace(request_budget=budget, complete=complete), None, [])
    assert len(requests) > 1
    assert manager.summary is not None
    assert all(len(request.messages[-1].content) < 512_000 for request in requests)


@pytest.mark.asyncio
async def test_explicit_portable_source_cap_is_honored_even_with_authoritative_count():
    manager = await context(summary_max_source_chars=100_000)
    model = SimpleNamespace(request_budget=lambda request, **kw: decision(100),
                            complete=AsyncMock(return_value=reply()))
    await manager._prepare(model, None, [])
    assert model.complete.await_count > 1 and manager.summary is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel", "history", "model"])
async def test_whole_prefix_preflight_obeys_cancellation_and_stale_request_guards(stop):
    manager = await context()
    entered, release = asyncio.Event(), asyncio.Event()

    async def budget(request, **kwargs):
        entered.set()
        await release.wait()
        return decision(100)

    model = SimpleNamespace(name="fixture", default_model="initial", request_budget=budget,
                            complete=AsyncMock(return_value=reply()))
    task = asyncio.create_task(manager.get_messages_for_request(provider=model))
    await entered.wait()
    if stop == "cancel":
        task.cancel()
        expected = asyncio.CancelledError
    else:
        if stop == "history":
            replacement = await manager.get_messages()
            replacement[2]["content"] += " Corrected evidence."
            await manager.set_messages(replacement)
        else:
            model.default_model = "replacement"
        release.set()
        expected = RuntimeError
    with pytest.raises(expected):
        await task
    assert manager.summary is None and manager._summary_progress is None
    model.complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_compaction_is_one_complete_window_with_actual_request_envelope():
    manager = await context(summary_max_source_chars=256, summary_max_calls=1)
    source = await manager.get_messages()
    reminder = {"role": "user", "content": "Keep the active operation private.",
                "metadata": {"source": "hook", "ephemeral": True, "persisted": True}}
    source.insert(3, reminder)
    await manager.set_messages(source)
    factory = AsyncMock(return_value="Current dynamic instructions.")
    await manager.set_system_prompt_factory(factory)
    manager.active_operations = lambda: [{"job_id": "queued-operation", "status": "queued"}]
    tool = ToolSpec(name="fixture", description="A synthetic tool.", parameters={"type": "object"})
    native = {"role": "user", "content": "Native checkpoint", "metadata": {
        "source": "context-managed", "ephemeral": True, "persisted": True,
        "opaque": [{"type": "message", "role": "user", "text": "Objective ORBIT"},
                   {"type": "compaction", "encrypted_content": "synthetic"}]}}
    model = SimpleNamespace(name="fixture", default_model="continuation-model",
        native_compaction_requires_request_context=True,
        supports_native_compaction=lambda: True, validate_compacted_context=lambda row: bool((row.get("metadata") or {}).get("opaque")),
        compact_context=AsyncMock(return_value={"kind": "native", "message": native}),
        complete=AsyncMock())

    async def count_view(view):
        request = ChatRequest(messages=[Message(**row) for row in view], model="chosen-model",
                              tools=[tool], tool_choice="auto", reasoning_effort="high",
                              metadata={"owner": "loop"})
        count = 500 if any((row.get("metadata") or {}).get("opaque") for row in view) else 200_000
        return {"dispatch": request, "budget_decision": decision(count)}

    result = await manager.get_measured_request_view(provider=model, retain_contents=[reminder["content"]], count_view=count_view)
    model.compact_context.assert_awaited_once()
    model.complete.assert_not_awaited()
    request = model.compact_context.call_args.args[0]
    assert request.model == "chosen-model" and request.tools == [tool]
    assert request.tool_choice == "auto" and request.reasoning_effort == "high"
    assert request.metadata["owner"] == "loop"
    assert request.metadata["native_compaction_request_context"] is True
    assert request.max_output_tokens == 4096
    assert request.messages[0].content == "Current dynamic instructions."
    assert any(row.content == source[2]["content"] for row in request.messages)
    assert len(source[2]["content"]) > 512_000
    assert not any(row.content == reminder["content"] for row in request.messages)
    assert "queued-operation" not in str(request.messages)
    assert "Current request" not in str(request.messages)
    assert manager.summary[1] == native
    assert finished(manager)["method"] == "native" and finished(manager)["calls"] == 1
    assert "completed_fragments" not in finished(manager)
    rows = result["base_view"]
    carrier = next(i for i, row in enumerate(rows) if (row.get("metadata") or {}).get("opaque"))
    assert rows[carrier + 1]["content"] == reminder["content"]
    factory.assert_awaited_once()
    assert await manager.get_messages() == source


@pytest.mark.asyncio
async def test_missing_native_request_context_falls_back_without_creating_opaque_state():
    manager = await context()
    model = SimpleNamespace(native_compaction_requires_request_context=True,
        supports_native_compaction=lambda: True, validate_compacted_context=lambda row: True,
        request_budget=lambda request, **kw: decision(100), compact_context=AsyncMock(),
        complete=AsyncMock(return_value=reply()))
    await manager.get_messages_for_request(provider=model)
    model.compact_context.assert_not_awaited()
    model.complete.assert_awaited_once()
    assert finished(manager)["native_selection"] == "request_context_unavailable"
