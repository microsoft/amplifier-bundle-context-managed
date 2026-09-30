"""Background result synchronization cannot reset an unchanged-prefix failure cap."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import LLMError
from amplifier_module_context_managed.boundary import BoundaryContextManager


def history():
    return [
        {"role": "user", "content": "Preserve the objective.", "metadata": {"flag": True}},
        {"role": "assistant", "content": "Verified historical evidence. " * 1000},
        {"role": "user", "content": "Preserve all originals."},
        {"role": "assistant", "content": "Verified recent results."},
        {"role": "user", "content": "Finish the current task."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "current-call", "name": "delegate", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "current-call", "content": '{"status":"pending","job_id":"job"}'},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable,expected", [(False, 1), (True, 3)])
async def test_replacing_job_receipts_keeps_suppression_for_exact_failed_prefix(retryable, expected):
    manager = BoundaryContextManager({"max_tokens": 10000, "summarize_trigger": .01, "summary_retry_delay": 0})
    model = SimpleNamespace(complete=AsyncMock(side_effect=LLMError("fixture failure", retryable=retryable)))
    await manager.set_messages(history())
    for number in range(10):
        await manager._prepare(model, None, [])
        replacement = await manager.get_messages()
        replacement[-1]["content"] = f"Verified background result {number}"
        await manager.set_messages(replacement)
    assert model.complete.await_count == expected
    assert manager.summary_failure["attempts"] == expected
    assert manager.summary is None
    assert await manager.get_messages() == replacement
    replacement[0]["metadata"]["flag"] = 1  # True and 1 differ in canonical identity.
    await manager.set_messages(replacement)
    assert manager.summary_failure is None
    await manager._prepare(model, None, [])
    assert model.complete.await_count == expected + 1
