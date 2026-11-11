"""Immediate host persistence can inspect history without a detached snapshot."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from amplifier_module_context_managed.boundary import BoundaryContextManager, mount_boundary


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("meter", ["actual", "estimate"])
async def test_mounted_visit_reads_complete_current_rows_without_changing_detached_api(durable, meter):
    capabilities, mounted = {}, {}

    async def mount(name, value):
        mounted[name] = value

    coordinator = SimpleNamespace(mount=mount, hooks=None, get_capability=capabilities.get,
        register_capability=capabilities.__setitem__, register_contributor=lambda *args: None)
    await mount_boundary(coordinator, {"durable_checkpoints": durable, "token_meter": meter})
    context = mounted["context"]
    rows = [
        {"role": "user", "content": "Original request", "metadata": {"nested": ["keep"]}},
        {"role": "assistant", "content": [{"type": "image", "source": {"type": "base64", "data": "AQ=="}}]},
        {"role": "tool", "tool_call_id": "recorded", "content": "Recorded result"},
    ]
    await context.set_messages(rows)
    saved = []
    visit = capabilities["context.history.visit"]
    assert visit(lambda current: saved.append(json.dumps(current))) is None
    assert json.loads(saved.pop()) == rows
    # A fresh visit sees new content, not a retained snapshot or revision cache.
    await context.add_message({"role": "assistant", "content": "New response"})
    assert visit(lambda current: saved.append(json.dumps(current))) is None
    expected = rows + [{"role": "assistant", "content": "New response"}]
    assert json.loads(saved.pop()) == expected
    # The callback cannot return borrowed rows through the visitor's result.
    assert visit(lambda current: current) is None
    detached = await capabilities["context.history"]()
    detached[0]["metadata"]["nested"].append("reader edit")
    rows[0]["metadata"]["nested"].append("input edit")
    assert (await context.get_messages())[0]["metadata"]["nested"] == ["keep"]
    assert capabilities["context.history_authority"] == "host"


@pytest.mark.asyncio
async def test_visit_does_not_copy_nested_provider_fields_or_change_checkpoint_state():
    copies = []

    class Witness(dict):
        def __deepcopy__(self, memo):
            copies.append(True)
            result = Witness(copy.deepcopy(dict(self), memo))
            memo[id(self)] = result
            return result

    context = BoundaryContextManager()
    await context.set_messages([{"role": "assistant", "content": "Kept", "provider": Witness(nested=["exact"])}])
    copies.clear()
    before = (context.revision, context.summary, context.recovery, context.checkpoint_status.copy())
    observed = []
    context.visit_history(lambda rows: observed.append(json.dumps(rows)))
    assert copies == []
    assert json.loads(observed[0])[0]["provider"] == {"nested": ["exact"]}
    assert (context.revision, context.summary, context.recovery, context.checkpoint_status) == before
    await context.get_messages()
    assert copies == [True]  # The ordinary API still detaches custom fields.


def test_async_consumer_is_rejected_before_calling_it():
    context = BoundaryContextManager()
    called = []

    async def consumer(rows):
        called.append(True)

    with pytest.raises(TypeError, match="must be synchronous"):
        context.visit_history(consumer)
    assert called == []


def test_accidental_coroutine_result_is_closed_without_running_or_retrying():
    context = BoundaryContextManager()
    calls, executed, pending = [], [], []

    async def later():
        executed.append(True)

    def consumer(rows):
        calls.append(True)
        pending.append(later())
        return pending[-1]

    with pytest.raises(TypeError, match="must not return an awaitable"):
        context.visit_history(consumer)
    assert calls == [True] and executed == []
    assert pending[0].cr_frame is None


def test_other_awaitable_results_are_refused_without_running_them():
    context = BoundaryContextManager()
    executed = []

    class Pending:
        def __await__(self):
            executed.append(True)
            yield

    with pytest.raises(TypeError, match="must not return an awaitable"):
        context.visit_history(lambda rows: Pending())
    assert executed == []


@pytest.mark.parametrize("error", [ValueError("consumer failed"), asyncio.CancelledError()])
def test_consumer_failure_is_preserved_without_retry(error):
    context = BoundaryContextManager()
    calls = []

    def consumer(rows):
        calls.append(True)
        raise error

    with pytest.raises(type(error)) as caught:
        context.visit_history(consumer)
    assert caught.value is error
    assert calls == [True]
