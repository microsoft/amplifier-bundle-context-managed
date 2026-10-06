"""Failures must stop execution, never turn into silent history loss."""
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import ContextLengthError

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.summary import summary_request


@pytest.mark.asyncio
@pytest.mark.parametrize("measured", [False, True])
async def test_large_current_tool_result_reaches_request_unchanged(measured):
    context = BoundaryContextManager({"max_tokens": 100000})
    original = [
        {"role": "user", "content": "Inspect the complete result."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "large", "name": "read_file", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "large", "content": "Evidence " * 20000},
    ]
    await context.set_messages(original)
    if measured:
        async def count(view):
            assert view[-1]["content"] == original[-1]["content"]
            return {"dispatch": object(), "budget_decision": {
                "estimated_input_tokens": 50000, "input_limit_tokens": 100000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 50000}}}
        result = await context.get_measured_request_view(provider=None, retain_contents=[], count_view=count)
        view = result["base_view"]
        assert result["count_calls"] == 1
    else:
        view = await context.get_messages_for_request()
    assert view[-1]["content"] == original[-1]["content"]
    assert await context.get_messages() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("measured", [False, True])
async def test_oversized_current_turn_stops_without_trimming_or_extra_counting(measured):
    context = BoundaryContextManager({"max_tokens": 1000})
    original = [{"role": "user", "content": "Current task " * 10000}]
    await context.set_messages(original)
    model = SimpleNamespace(complete=AsyncMock())
    views = []
    async def count(view):
        views.append(copy.deepcopy(view))
        return {"dispatch": object(), "budget_decision": {
            "estimated_input_tokens": 30000, "input_limit_tokens": 1000,
            "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 30000}}}
    with pytest.raises(ContextLengthError, match="No emergency history trimming"):
        if measured:
            await context.get_measured_request_view(provider=model, retain_contents=[], count_view=count)
        else:
            await context.get_messages_for_request(provider=model)
    assert len(views) == int(measured)
    assert await context.get_messages() == original
    model.complete.assert_not_awaited()
    assert context.summary is None


@pytest.mark.parametrize("config, expected", [({}, 8192), ({"summary_target_tokens": 1500}, 8192),
                                            ({"summary_max_output_tokens": 12000}, 12000)])
def test_summary_output_allowance_is_separate_from_legacy_note_target(config, expected):
    request = summary_request("Summarize", "Evidence", "", config)
    assert request.max_output_tokens == expected
