import asyncio, json, socket
from collections import Counter
import httpx, openai
from amplifier_core import AmplifierSession
import amplifier_module_loop_live.orchestrator as loop_module
import traceback
original_failure=loop_module.turn_failure
def failure(exc,stage):
 traceback.print_exception(exc)
 return original_failure(exc,stage)
loop_module.turn_failure=failure
from amplifier_module_loop_live.runtime import Input, Runtime
from amplifier_module_provider_openai import OpenAIProvider
from validate_compaction_flow_offline import Server, Tool, MODEL, IDENTITY, response

async def main():
 def offline(*a,**kw):raise AssertionError('No network in synthetic validation')
 socket.create_connection=offline
 server=Server('native'); counts=Counter(); runtime=Runtime()
 session=AmplifierSession({'session':{'orchestrator':{'module':'loop-live','config':{'configured_bundle':True,'background_delegate':False,'min_delay_between_calls_ms':0,'max_iterations':40}},'context':{'module':'context-managed','config':{'engine':'boundary','durable_checkpoints':True,'max_tokens':200000,'summarize_trigger':.01,'native_min_new_tokens':0}}},'providers':[],'tools':[],'hooks':[]},session_id=runtime.session_id)
 await session.initialize();co=session.coordinator
 co.register_capability('live.runtime',runtime);co.register_capability('context.checkpoint_identity',lambda:IDENTITY)
 async def preserve(rows):return [{'kind':'synthetic-history','messages':len(rows)}]
 co.register_capability('context.preserve_evidence',preserve)
 sdk=openai.AsyncOpenAI(api_key='fixture',max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
 provider=OpenAIProvider(client=sdk,config={'default_model':MODEL,'enable_long_context':True,'use_streaming':False,'max_retries':0});provider.coordinator=co
 await co.mount('providers',provider,name='openai');await co.mount('tools',Tool(counts),name='fixture')
 context=co.get('context')
 async def prompt():return 'Synthetic protocol test; do not replay completed tools.'
 await context.set_system_prompt_factory(prompt)
 task=asyncio.create_task(session.execute(''))
 try:
  await runtime.wait_for(lambda e:e['type']=='session.ready',10)
  await runtime.submit(Input('user','Run the synthetic operations once; never replay a completed operation.',id='autonomous'))
  for n in range(24):
   body=await asyncio.wait_for(server.requests.get(),15)
   assert 'never replay a completed operation' in json.dumps(body)
   await server.replies.put(response(calls=[(f'call-{n}',n)]))
  await asyncio.wait_for(server.requests.get(),15)
  await server.replies.put(response('All synthetic operations completed exactly once.'))
  end=await runtime.wait_for(lambda e:e['type'] in {'generation.finished','generation.failed'},15)
  assert end['type']=='generation.finished',end
  assert server.compactions>=3,server.compactions
  assert len(counts)==24 and all(v==1 for v in counts.values())
  rows=await context.get_messages(); checkpoint=context.export_checkpoint(IDENTITY)
  assert context.restore_checkpoint(checkpoint,IDENTITY)['status']=='restored'
  print(json.dumps({'passed':True,'humanTurns':1,'operationsExactlyOnce':24,'nativeCompactions':server.compactions,'canonicalMessages':len(rows),'checkpointThrough':checkpoint['summary']['throughMessage'],'checkpointRestored':True,'scope':'Real Core/loop/provider SDK with synthetic HTTP and tool'}))
 except Exception:
  print(json.dumps({'counts':dict(counts),'compactions':server.compactions,'events':runtime.events[-8:]},default=str));raise
 finally:
  task.cancel();await asyncio.gather(task,return_exceptions=True);await session.cleanup();await provider.close()
asyncio.run(main())
