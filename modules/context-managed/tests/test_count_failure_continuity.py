"""A count outage never poisons a valid checkpoint or replays history."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_module_context_managed.boundary import BoundaryContextManager


def checkpoint(label):
    return {'role': 'user', 'content': 'Native checkpoint', 'metadata': {
        'source': 'context-managed', 'ephemeral': True, 'persisted': True, 'opaque': label}}


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['initial_count', 'compact_count', 'result_count'])
async def test_count_failure_preserves_checkpoint_then_continues(phase):
    error_type = type('TokenCountError', (RuntimeError,), {
        '__module__': 'amplifier_module_provider_openai._token_count'})
    failure = error_type('Could not check conversation size')
    ctx = BoundaryContextManager({'durable_checkpoints': True, 'max_tokens': 5000,
        'summarize_trigger': .2, 'native_min_new_tokens': 0})
    identity = {'provider': 'fixture', 'model': 'native'}
    ctx.checkpoint_identity = lambda: identity
    rows = [{'role': 'user', 'content': 'Produce a report'},
            {'role': 'assistant', 'content': 'Earlier report saved'},
            {'role': 'user', 'content': 'Check the remaining evidence'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'done', 'name': 'read_file', 'arguments': {}}]},
            {'role': 'tool', 'tool_call_id': 'done', 'content': 'Already completed evidence'},
            {'role': 'user', 'content': 'Continue'},
            {'role': 'assistant', 'content': 'Progress saved'},
            {'role': 'user', 'content': 'Keep going'}]
    await ctx.set_messages(rows)
    ctx.summary = (2, checkpoint('old'))
    ctx.summary_identity = identity
    before = json.dumps(ctx.export_checkpoint(identity), sort_keys=True)
    recovered = False
    compacted = False

    async def budget(request, **kwargs):
        if not recovered and phase == 'compact_count':
            raise failure
        return {'estimated_input_tokens': 2000, 'input_limit_tokens': 5000,
                'measurement': {'kind': 'provider_count', 'source': 'fixture', 'input_tokens': 2000}}

    async def compact(request):
        nonlocal compacted
        compacted = True
        return {'kind': 'native', 'message': checkpoint('new')}

    provider = SimpleNamespace(name='fixture', default_model='native',
        supports_native_compaction=lambda: True,
        validate_compacted_context=lambda row: bool(row.get('metadata', {}).get('opaque')),
        request_budget=budget, compact_context=AsyncMock(side_effect=compact), complete=AsyncMock())

    async def count_view(view):
        if not recovered and ((phase == 'initial_count' and not compacted) or
                              (phase == 'result_count' and compacted)):
            raise failure
        tokens = 100 if compacted else 4500
        return {'dispatch': ChatRequest(messages=[Message(**row) for row in view]),
                'budget_decision': {'estimated_input_tokens': tokens, 'input_limit_tokens': 5000,
                    'measurement': {'kind': 'provider_count', 'source': 'fixture', 'input_tokens': tokens}}}

    with pytest.raises(error_type) as caught:
        await ctx.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    assert caught.value is failure
    assert json.dumps(ctx.export_checkpoint(identity), sort_keys=True) == before
    assert ctx.summary_failure is None
    assert ctx.messages == rows
    provider.complete.assert_not_called()
    recovered, compacted = True, False
    result = await ctx.get_measured_request_view(provider=provider, retain_contents=[], count_view=count_view)
    assert result['final_attempt']['budget_decision']['measurement']['input_tokens'] == 100
    assert ctx.summary[0] > 2
    assert ctx.messages == rows
    assert sum(row.get('tool_call_id') == 'done' for row in ctx.messages) == 1
    provider.complete.assert_not_called()
    restored = BoundaryContextManager(copy.deepcopy(ctx.config))
    await restored.set_messages(rows)
    assert restored.restore_checkpoint(ctx.export_checkpoint(identity), identity)['status'] == 'restored'
