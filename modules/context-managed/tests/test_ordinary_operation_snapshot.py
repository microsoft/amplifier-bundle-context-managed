"""Settled host jobs must not invalidate an otherwise unchanged counted request."""
import asyncio
import copy
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message

from .test_ordinary_settled_fitting import Provider, exchange, history, manager, measured


def pending(identity):
    # Loop Live's context.active_operations wire contract.
    return {"job_id": identity, "call_id": "call-" + identity, "status": "pending", "tool": "delegate"}


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prompt", "preserve", "pressure", "failed-fit", "final-fit"])
async def test_settled_operation_keeps_conservative_snapshot_and_completes_twice(phase):
    rows = history()
    context = await manager(rows, durable_checkpoints=True)
    operations = [pending("returned"), pending("still-pending")]
    before = copy.deepcopy(operations)
    context.active_operations = lambda: copy.deepcopy(operations)
    context.summary_provider = lambda: (_ for _ in ()).throw(RuntimeError("No optional utility"))
    model = Provider()
    counts = 0

    def settle():
        if operations and operations[0]["job_id"] == "returned":
            operations.pop(0)

    async def prompt():
        if phase == "prompt":
            settle()
        return "Required host prompt"

    async def preserve(messages):
        if phase == "preserve":
            settle()
        return []

    context.factory = prompt
    context.preserve_evidence = preserve

    async def count(view):
        nonlocal counts
        counts += 1
        request = ChatRequest(messages=[Message(**row) for row in view])
        decision = await model.request_budget(request)
        oversized = decision["estimated_input_tokens"] > decision["input_limit_tokens"]
        if (phase == "pressure" and counts == 1
                or phase == "failed-fit" and counts == 2 and oversized
                or phase == "final-fit" and not oversized):
            # An inbox consumer can settle a job while the request awaits count.
            await asyncio.sleep(0)
            settle()
        return {"dispatch": request, "budget_decision": decision}

    first = await context.get_measured_request_view(provider=model, retain_contents=[], count_view=count)
    await model.complete(first["final_attempt"]["dispatch"])
    assert operations == before[1:]
    # Keep the original conservative carrier: no uncounted overlay after fitting,
    # no inference that disappearance settles side effects, and no action replay.
    assert '"job_id": "returned"' in str(first["base_view"])
    assert '"job_id": "still-pending"' in str(first["base_view"])
    second = await measured(context, model)
    await model.complete(second["final_attempt"]["dispatch"])
    assert '"job_id": "returned"' not in str(second["base_view"])
    assert len(model.completed) == 2 and model.summaries == 0
    assert await context.get_messages() == rows


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["new", "call-id", "tool", "status", "history",
                                      "model", "config", "identity", "checkpoint", "stop"])
async def test_count_cannot_hide_unsafe_change_behind_operation_completion(mutation):
    rows = history()
    context = await manager(rows)
    operations = [pending("returned"), pending("still-pending")]
    context.active_operations = lambda: copy.deepcopy(operations)
    model = Provider()
    identity = {"provider": model.name, "model": model.default_model}
    context.checkpoint_identity = lambda: copy.deepcopy(identity)

    async def count(view):
        operations.pop(0)
        if mutation == "new":
            operations.append(pending("new-job"))
        elif mutation == "call-id":
            operations[0]["call_id"] = "different-call"
        elif mutation == "tool":
            operations[0]["tool"] = "write"
        elif mutation == "status":
            operations[0]["status"] = "cancel_requested"
        elif mutation == "history":
            await context.add_message({"role": "user", "content": "New input"})
        elif mutation == "model":
            model.default_model = "different"
        elif mutation == "config":
            context.config["max_tokens"] = 1000
        elif mutation == "identity":
            identity["model"] = "different"
        elif mutation == "checkpoint":
            context.summary = (1, "Different checkpoint")
        else:
            context._is_cancelled = lambda: True
        request = ChatRequest(messages=[Message(**row) for row in view])
        return {"dispatch": request, "budget_decision": await model.request_budget(request)}

    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        await context.get_measured_request_view(provider=model, retain_contents=[], count_view=count)
    assert len(model.counted) == 1 and not model.completed and model.summaries == 0


@pytest.mark.asyncio
async def test_actual_loop_job_return_and_selected_provider_count():
    """Optional installed-pair regression: use the actual capability and consumer."""
    import ast
    import inspect
    import textwrap
    from types import SimpleNamespace

    loop_module = pytest.importorskip("amplifier_module_loop_live.orchestrator")
    session_module = pytest.importorskip("amplifier_web.host.session")
    loop_type = loop_module.BundleLiveOrchestrator
    # Extract the actual registered lambda, not a parallel host implementation.
    tree = ast.parse(textwrap.dedent(inspect.getsource(loop_type._execute_live)))
    register = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and len(node.args) == 2 and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "context.active_operations")
    loop = SimpleNamespace(jobs={"job": {"call_id": "call-job", "status": "pending",
        "tool_call": {"name": "delegate"}}}, runtime=SimpleNamespace(emit=AsyncMock()))
    operations = eval(compile(ast.Expression(register.args[1]), "<actual-loop-capability>", "eval"),
                      {"self": loop})
    rows = history() + exchange("call-job", size=10)
    context = await manager(rows)
    context.active_operations = operations
    raw = Provider()
    original = raw.request_budget

    async def budget(request, **kwargs):
        if loop.jobs["job"].get("result") is None:
            await loop_type._job_returned(loop, "job", {"report": "Read-only analysis returned"},
                                          "returned", deliver=False)
        return await original(request, **kwargs)

    raw.request_budget = budget
    model = session_module.SelectedProvider(raw, {"model": raw.default_model})
    result = await measured(context, model)
    await model.complete(result["final_attempt"]["dispatch"])
    second = await measured(context, model)
    await model.complete(second["final_attempt"]["dispatch"])
    assert operations() == []
    assert loop.jobs["job"]["status"] == "returned"
    assert "job" in str(result["base_view"])
    assert await context.get_messages() == rows
    assert len(raw.completed) == 2 and raw.summaries <= 1


def test_operation_guard_rejects_observed_aba_and_ambiguous_or_unknown_schemas():
    from amplifier_module_context_managed.boundary import BoundaryContextManager
    context = BoundaryContextManager()
    operations = [pending("one"), pending("two")]
    context.active_operations = lambda: operations
    guard = context._operations_guard(operations)
    assert guard()
    removed = operations.pop(0)
    assert guard()
    operations.append(removed)
    assert not guard()
    for rows in ([pending("one"), pending("one")], [{"job_id": "unknown-schema"}]):
        operations = copy.deepcopy(rows)
        guard = context._operations_guard(operations)
        operations.pop()
        assert not guard()


@pytest.mark.asyncio
async def test_removed_during_pressure_cannot_reappear_during_final_fit():
    context = await manager(history())
    operations = [pending("one")]
    context.active_operations = lambda: copy.deepcopy(operations)
    model = Provider()
    counts = 0

    async def count(view):
        nonlocal counts
        counts += 1
        operations[:] = [] if counts == 1 else [pending("one")]
        request = ChatRequest(messages=[Message(**row) for row in view])
        return {"dispatch": request, "budget_decision": await model.request_budget(request)}

    with pytest.raises(RuntimeError, match="contract changed"):
        await context.get_measured_request_view(provider=model, retain_contents=[], count_view=count)
    assert counts == 2 and not model.completed