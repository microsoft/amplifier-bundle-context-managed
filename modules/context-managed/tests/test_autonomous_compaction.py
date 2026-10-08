"""Long autonomous runs must compact without waiting for another human turn."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import ContextLengthError
from amplifier_module_context_managed.boundary import BoundaryContextManager


def exchange(i, result='evidence ' * 120):
    return [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': str(i), 'name': 'read_file', 'arguments': {}}]},
            {'role': 'tool', 'tool_call_id': str(i), 'content': result}]


def snapshot(text, placement='tail'):
    return {'role': 'user', 'content': text, 'metadata': {'ephemeral': True, 'persisted': True, 'reminder_placement': placement}}


def provider():
    model = SimpleNamespace(name='fixture', default_model='native')
    model.supports_native_compaction = lambda: True
    model.validate_compacted_context = lambda row: bool((row.get('metadata') or {}).get('opaque'))
    model.request_budget = lambda request, **kw: {'estimated_input_tokens': sum(len(str(row.content)) // 4 + 10 for row in request.messages),
        'input_limit_tokens': 5000, 'measurement': {'kind': 'provider_count',
        'input_tokens': sum(len(str(row.content)) // 4 + 10 for row in request.messages)}}
    model.compact_context = AsyncMock(return_value={'kind': 'native', 'message': {'role': 'user', 'content': 'opaque',
        'metadata': {'source': 'context-managed', 'ephemeral': True, 'persisted': True, 'opaque': 'fixture'}}})
    return model


@pytest.mark.asyncio
async def test_single_human_run_compacts_repeatedly_and_restores_without_replay():
    model = provider()
    ctx = BoundaryContextManager({'max_tokens': 5000, 'summarize_trigger': .35, 'durable_checkpoints': True, 'native_min_new_tokens': 0})
    identity = {'provider': 'fixture', 'model': 'native'}
    ctx.checkpoint_identity = lambda: identity
    original = [{'role': 'user', 'content': 'Build this, preserving existing files.'}]
    await ctx.set_messages(original)
    checkpoints = []
    for i in range(40):
        for row in exchange(i):
            original.append(row)
            await ctx.add_message(row)
        view = await ctx.get_messages_for_request(provider=model)
        assert view[-1]['content'] == original[-1]['content']
        assert sum(row.get('content') == original[0]['content'] for row in view) == 1
        if ctx.summary:
            checkpoints.append(ctx.summary[0])
    assert model.compact_context.await_count > 2
    assert checkpoints[-1] > checkpoints[0]
    assert ctx.messages == original
    # Current instructions are preserved outside the provider's opaque checkpoint.
    for call in model.compact_context.await_args_list:
        assert original[0]['content'] not in [row.content for row in call.args[0].messages]
    record = ctx.export_checkpoint(identity)
    restored = BoundaryContextManager(ctx.config)
    await restored.set_messages(original)
    assert restored.restore_checkpoint(record, identity)['status'] == 'restored'
    assert restored.messages == original
    assert (await restored.get_messages_for_request(provider=model))[-1]['content'] == original[-1]['content']


def test_complete_parallel_batches_and_pending_jobs_are_never_split():
    ctx = BoundaryContextManager()
    rows = [{'role': 'user', 'content': 'Continue'}]
    for i in range(10): rows += exchange(i)
    pending = len(rows)
    rows += [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'a'}, {'id': 'b'}]},
             {'role': 'tool', 'tool_call_id': 'a', 'content': '{"status":"queued","job_id":"a-job"}'},
             {'role': 'tool', 'tool_call_id': 'b', 'content': 'done'}]
    for i in range(10, 20): rows += exchange(i)
    assert ctx._boundary(rows, []) <= pending
    assert ctx._boundary(rows, ['Continue']) == 0
    # A durable completed receipt permits advancing on the next request.
    rows[pending + 1]['content'] = 'completed'
    assert ctx._boundary(rows, []) > pending + 2


def test_reminder_projection_retains_current_and_explicitly_required_snapshots():
    ctx = BoundaryContextManager()
    old, current = snapshot('old', 'pre_user'), snapshot('current')
    ordinary = {'role': 'user', 'content': 'unrelated hook data', 'metadata': {'ephemeral': True, 'persisted': True}}
    rows = [{'role': 'user', 'content': 'task'}, old, ordinary, current]
    before = copy.deepcopy(rows)
    assert ctx._assembled(rows, None) == [rows[0], ordinary, current]
    assert ctx._assembled(rows, None, ['old']) == rows
    assert rows == before


@pytest.mark.asyncio
async def test_safe_budget_failure_records_counts_and_boundary_without_content():
    hooks = SimpleNamespace(emit=AsyncMock())
    ctx = BoundaryContextManager({'max_tokens': 100}, hooks)
    await ctx.add_message({'role': 'user', 'content': 'private requirement ' * 1000})
    with pytest.raises(ContextLengthError):
        await ctx.get_messages_for_request(provider=None)
    event = next(call.args[1] for call in hooks.emit.call_args_list if call.args[0] == 'context:budget_exceeded')
    assert event['input_tokens'] > event['input_limit_tokens']
    assert event['eligible_through_message'] == 0
    assert 'private requirement' not in json.dumps(event)
