"""Strict compaction through real Core/loop/provider with in-memory HTTP only."""
import asyncio
import json
import socket

import httpx
import openai
from amplifier_core import AmplifierSession
from amplifier_module_loop_live.runtime import Input, Runtime
from amplifier_module_provider_openai import OpenAIProvider

from validate_compaction_flow_offline import Server, MODEL, response
from validate_compaction_wait_offline import history


async def run_case(mode):
    server = Server('fallback' if mode == 'native-error' else mode)
    runtime = Runtime()
    session = AmplifierSession({'session': {
        'orchestrator': {'module': 'loop-live', 'config': {'configured_bundle': True,
            'background_delegate': False, 'min_delay_between_calls_ms': 0}},
        'context': {'module': 'context-managed', 'config': {'engine': 'boundary',
            'max_tokens': 200000, 'summarize_trigger': .01}}},
        'providers': [], 'tools': [], 'hooks': []}, session_id=runtime.session_id)
    await session.initialize()
    coordinator = session.coordinator
    coordinator.register_capability('live.runtime', runtime)
    sdk = openai.AsyncOpenAI(api_key='fixture', max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
    provider = OpenAIProvider(client=sdk, config={'default_model': MODEL,
        'enable_long_context': True, 'use_streaming': False, 'max_retries': 0})
    provider.coordinator = coordinator
    if mode == 'semantic':
        # Explicitly simulate an adapter without a native capability, not a
        # failed native operation being silently downgraded to a summary.
        provider.supports_native_compaction = lambda: False
    await coordinator.mount('providers', provider, name='openai')
    context = coordinator.get('context')
    original = history(10)
    await context.set_messages(original)
    task = asyncio.create_task(session.execute(''))
    try:
        await runtime.wait_for(lambda event: event['type'] == 'session.ready', 10)
        before = runtime.sequence
        await runtime.submit(Input('user', 'Status only; perform no tools.', id='strict-'+mode))
        if mode == 'native-error':
            event = await runtime.wait_for(lambda event: event['sequence'] > before and
                event['type'] in {'generation.finished', 'generation.failed'}, 15)
            assert event['type'] == 'generation.failed', event
            assert server.compactions == 1 and server.summaries == 0
            assert not server.normal_wires, 'No foreground inference after compaction failure'
        else:
            await asyncio.wait_for(server.requests.get(), 15)
            await server.replies.put(response('Status: preserved; no tools executed.'))
            event = await runtime.wait_for(lambda event: event['sequence'] > before and
                event['type'] in {'generation.finished', 'generation.failed'}, 15)
            assert event['type'] == 'generation.finished', event
            assert context.summary
            assert server.compactions > 0 if mode == 'native' else server.summaries > 0
        current = await context.get_messages()
        assert current[:len(original)] == original
        return {'mode': mode, 'passed': True, 'native_calls': server.compactions,
            'summary_calls': server.summaries, 'foreground_calls': len(server.normal_wires),
            'original_messages_preserved': len(original)}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await session.cleanup()
        await provider.close()


async def main():
    # No accidental network escape; the real SDK only reaches MockTransport.
    def offline(*args, **kwargs):
        raise AssertionError('External network is forbidden in this test')
    socket.create_connection = offline
    reports = [await run_case(mode) for mode in ('native', 'semantic', 'native-error')]
    print(json.dumps({'scope': 'Scripted HTTP; no model-quality claim', 'cases': reports}, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
