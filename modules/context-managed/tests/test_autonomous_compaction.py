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


@pytest.mark.parametrize('method', ['none', 'semantic', 'native'])
@pytest.mark.parametrize('placement', ['tail', 'pre_user'])
@pytest.mark.parametrize('retain', [False, True])
def test_recurring_snapshot_uses_latest_canonical_row_not_equal_historical_rows(method, placement, retain):
    ctx = BoundaryContextManager()
    original = snapshot('Recurring selected context A.', placement)
    intermediate = snapshot('Different selected context B.', placement)
    latest = copy.deepcopy(original)
    rows = [{'role': 'user', 'content': 'task'}, original,
            {'role': 'assistant', 'content': 'completed'}, intermediate, latest]
    summary = None if method == 'none' else (3, 'portable note' if method == 'semantic' else {
        'role': 'user', 'content': 'opaque checkpoint', 'metadata': {'opaque': 'fixture'}})
    before = copy.deepcopy(rows)
    view = ctx._assembled(rows, summary, [original['content']] if retain else [], omitted_indices=[1])
    assert view.count(latest) == (2 if retain else 1)
    assert intermediate not in view
    assert view[-1] == latest
    assert rows == before


def signed_exchange():
    """Opaque provider state belongs to the provider, not the history projector."""
    rows = exchange('signed', result='The tool already ran; do not replay it.')
    rows[0]['content'] = [
        {'type': 'thinking', 'thinking': 'fixture reasoning', 'signature': 'fixture-signature'},
        {'type': 'text', 'text': 'Inspect the recorded result.'},
    ]
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize('placement', ['tail', 'pre_user'])
@pytest.mark.parametrize('changed', [False, True])
@pytest.mark.parametrize('measured', [False, True])
async def test_reminder_refresh_can_change_prefix_before_unchanged_provider_reasoning(placement, changed, measured):
    ctx = BoundaryContextManager({'max_tokens': 5000})
    old = snapshot('Current capabilities: selected A; live surface revision 1.', placement)
    rows = [{'role': 'user', 'content': 'Complete the authorized task.'}, old] + signed_exchange()
    await ctx.set_messages(rows)
    counted = []

    async def request_view():
        if not measured:
            return await ctx.get_messages_for_request()

        async def count(view):
            counted.append(copy.deepcopy(view))
            return {'dispatch': object(), 'budget_decision': {
                'estimated_input_tokens': 100, 'input_limit_tokens': 5000,
                'measurement': {'kind': 'provider_count', 'input_tokens': 100}}}

        result = await ctx.get_measured_request_view(provider=None, retain_contents=[], count_view=count)
        assert result['count_calls'] == 1
        assert result['base_view'] == counted[-1]
        return result['base_view']

    before = await request_view()
    refreshed = snapshot('Current capabilities: A revoked; live surface unavailable.', placement)
    if changed:
        # The loop deduplicates unchanged snapshots but appends changed bodies.
        await ctx.add_message(refreshed)
    canonical = await ctx.get_messages()
    after = await request_view()

    assert ctx.summary is None  # A legal projection change, not just compaction.
    assert old in canonical
    assert (old in after) is not changed
    assert after.count(refreshed) == int(changed)
    thinking_before = next(i for i, row in enumerate(before) if row.get('tool_calls'))
    thinking_after = next(i for i, row in enumerate(after) if row.get('tool_calls'))
    assert (before[:thinking_before] == after[:thinking_after]) is not changed
    assert before[thinking_before:] == after[thinking_after:thinking_after + 2]
    assert after[thinking_after:thinking_after + 2] == signed_exchange()
    assert await ctx.get_messages() == canonical
    # Providers receive detached views; their binding/invalidation must not edit
    # canonical thinking, receipts, reminders or trigger tool execution.
    after[thinking_after]['content'][0]['signature'] = 'request-only change'
    assert await ctx.get_messages() == canonical


@pytest.mark.asyncio
async def test_superseded_snapshots_do_not_consume_request_window_to_preserve_reasoning():
    ctx = BoundaryContextManager({'max_tokens': 1000})
    rows = [{'role': 'user', 'content': 'Keep current context bounded.'}]
    for i in range(40):
        rows.append(snapshot(f'Obsolete revision {i}: ' + 'old surface and capability data ' * 200))
    current = snapshot('Only current selected context is available.')
    rows += [current] + signed_exchange()
    await ctx.set_messages(rows)
    view = await ctx.get_messages_for_request()
    assert view == [rows[0], current] + signed_exchange()
    assert ctx.summary is None
    assert await ctx.get_messages() == rows


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['native', 'semantic'])
@pytest.mark.parametrize('placement', ['tail', 'pre_user'])
async def test_replaced_snapshot_stays_current_only_across_compaction_and_restore(method, placement):
    model = provider()
    if method == 'semantic':
        model.supports_native_compaction = lambda: False
        model.complete = AsyncMock(return_value=SimpleNamespace(content=[
            SimpleNamespace(type='text', text='Historical exchanges completed; preserve their recorded evidence.')]))
    ctx = BoundaryContextManager({'max_tokens': 5000, 'summarize_trigger': .35,
                                  'durable_checkpoints': True, 'native_min_new_tokens': 0})
    identity = {'provider': model.name, 'model': model.default_model}
    ctx.checkpoint_identity = lambda: identity
    stale = snapshot('Stale selected capability and surface revision.', placement)
    current = snapshot('Current selected capability; previous surface unavailable.', placement)
    rows = [{'role': 'user', 'content': 'Complete the authorized task.'}, stale]
    for i in range(12):
        rows += exchange(i)
    rows += [current] + signed_exchange()
    await ctx.set_messages(rows)
    view = await ctx.get_messages_for_request(provider=model)
    assert ctx.summary and ctx.summary[0] > 2
    assert stale not in view and view.count(current) == 1
    assert view[-2:] == signed_exchange()
    # The superseded aggregate must not be baked into a new native window or
    # portable summary source. Provider binding sees this projected source too.
    operation = model.compact_context if method == 'native' else model.complete
    assert operation.await_count == 1
    assert stale['content'] not in str(operation.await_args)

    ctx.config['summarize_trigger'] = 100  # Reuse the unchanged accepted prefix.
    record = ctx.export_checkpoint(identity)
    refreshed = snapshot('Current capability revoked; no live surface.', placement)
    await ctx.add_message(refreshed)
    canonical = await ctx.get_messages()
    updated = await ctx.get_messages_for_request(provider=model)
    restored = BoundaryContextManager(ctx.config)
    await restored.set_messages(canonical)
    assert restored.restore_checkpoint(record, identity)['status'] == 'restored'
    resumed = await restored.get_messages_for_request(provider=model)
    assert resumed == updated
    assert stale not in resumed and current not in resumed
    assert resumed.count(refreshed) == 1
    assert [row for row in resumed if row.get('tool_call_id') == 'signed'] == [signed_exchange()[1]]
    assert next(row for row in resumed if row.get('tool_calls') == signed_exchange()[0]['tool_calls']) == signed_exchange()[0]
    assert operation.await_count == 1
    assert await ctx.get_messages() == canonical == await restored.get_messages()


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
