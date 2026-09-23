"""Long Core/loop-live conversation using the real SDK and scripted HTTP.

Fixture replies are deliberately deterministic. This proves protocol, history,
tool execution and checkpoint continuity, not a model's summarization quality.
All tool operations are synthetic and stay in process; no user data is loaded.
"""

import argparse
import asyncio
import copy
import json
import re
import socket
from collections import Counter
from pathlib import Path
from typing import ClassVar

import httpx
import openai
from amplifier_core import AmplifierSession, HookResult, ToolResult
from amplifier_module_loop_live.runtime import Input, Runtime
from amplifier_module_provider_openai import OpenAIProvider
from validate_compaction_wait_offline import checksum

MODEL = "gpt-6-astra"
IDENTITY = {"provider": "openai", "model": MODEL}


def response(text="", calls=()):
    output = [{"id": "message", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": text, "annotations": []}]}] if text else []
    output.extend({"type": "function_call", "id": "item-"+call, "call_id": call,
        "name": "fixture", "arguments": json.dumps({"operation": operation}), "status": "completed"}
        for call, operation in calls)
    return {"id": "response", "object": "response", "created_at": 1,
        "model": MODEL, "status": "completed", "output": output,
        "usage": {"input_tokens": 1000, "output_tokens": 20, "total_tokens": 1020}}


class Server:
    def __init__(self, mode):
        self.mode = mode
        self.requests, self.replies = asyncio.Queue(), asyncio.Queue()
        self.compactions, self.summaries, self.counts = 0, 0, 0
        self.windows = []
        self.normal_wires = []

    async def __call__(self, request):
        body = json.loads(request.content)
        wire = json.dumps(body.get("input", []))
        if request.url.path.endswith("input_tokens"):
            self.counts += 1
            return httpx.Response(200, json={"input_tokens": len(wire)//4})
        assert request.extensions["timeout"]["read"] is None
        if request.url.path.endswith("compact"):
            self.compactions += 1
            if self.mode == "fallback":
                return httpx.Response(400, json={"error": {"message": "synthetic native rejection", "type": "invalid_request_error"}})
            # Preserve returned canonical items on later calls: actual provider
            # expansion must contain the previous opaque item unchanged.
            if self.windows:
                for item in self.windows[-1]:
                    assert item in body["input"], "Previous canonical native window changed"
            retained = [copy.deepcopy(item) for item in body["input"]
                        if item.get("type") == "message" and item.get("role") == "user"]
            window = retained + [{"type": "compaction", "encrypted_content": f"fixture-window-{self.compactions}"}]
            self.windows.append(copy.deepcopy(window))
            return httpx.Response(200, json={"id": "cmp_fixture", "object": "response.compaction", "created_at": 1,
                "output": window, "usage": {"input_tokens": len(wire)//4, "output_tokens": 100, "total_tokens": len(wire)//4+100}})
        if "Prepare a factual continuation note" in body.get("instructions", ""):
            self.summaries += 1
            receipts = sorted(set(re.findall(r"receipt=R-\d+", wire)))
            # This oracle is not an LLM quality claim. It makes transport loss
            # observable when repeated summarization drops earlier receipts.
            note = "ORBIT; budget=25; publication=pending; never replay completed tools. " + "; ".join(receipts)
            return httpx.Response(200, json=response(note))
        self.normal_wires.append(copy.deepcopy(body))
        await self.requests.put(body)
        return httpx.Response(200, json=await self.replies.get())


class Tool:
    name = "fixture"
    description = "Execute a synthetic operation exactly once, with no outside effects."
    input_schema: ClassVar[dict] = {"type": "object", "properties": {"operation": {"type": "integer"}}, "required": ["operation"]}

    def __init__(self, counts):
        self.counts = counts

    async def execute(self, args):
        operation = args["operation"]
        self.counts[operation] += 1
        assert self.counts[operation] == 1, "Tool operation replayed"
        return ToolResult(success=True, output={"receipt": f"receipt=R-{operation}",
            "completed": True, "evidence": "Synthetic historical evidence. " * 500})


async def run_case(mode, turns):
    server, counts, events = Server(mode), Counter(), []
    config = {"engine": "boundary", "durable_checkpoints": True,
        "max_tokens": 30000, "summarize_trigger": 0.1, "native_min_new_tokens": 0,
        "native_compaction": mode != "semantic", "summary_max_source_chars": 24000}

    async def start(saved=None):
        runtime = Runtime()
        session = AmplifierSession({"session": {
            "orchestrator": {"module": "loop-live", "config": {"configured_bundle": True,
                "background_delegate": False, "min_delay_between_calls_ms": 0, "max_iterations": 6}},
            "context": {"module": "context-managed", "config": config}},
            "providers": [], "tools": [], "hooks": []}, session_id=runtime.session_id)
        await session.initialize()
        coordinator = session.coordinator
        coordinator.register_capability("live.runtime", runtime)
        coordinator.register_capability("context.checkpoint_identity", lambda: IDENTITY)
        async def preserve(rows):
            return [{"kind": "synthetic-history", "sha256": checksum(rows)}]
        coordinator.register_capability("context.preserve_evidence", preserve)
        async def observe(kind, data):
            events.append((kind, copy.deepcopy(data)))
            return HookResult()
        coordinator.hooks.register("context:compaction_finished", observe)
        sdk = openai.AsyncOpenAI(api_key="fixture", timeout=0.001, max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
        provider = OpenAIProvider(client=sdk, config={"default_model": MODEL,
            "enable_long_context": True, "use_streaming": False, "max_retries": 0})
        provider.coordinator = coordinator
        await coordinator.mount("providers", provider, name="openai")
        await coordinator.mount("tools", Tool(counts), name="fixture")
        context = coordinator.get("context")
        async def prompt():
            return "Synthetic protocol evaluation. Original objective ORBIT; never replay completed tools."
        await context.set_system_prompt_factory(prompt)
        if saved:
            await context.set_messages(saved[0])
            assert context.restore_checkpoint(saved[1], IDENTITY)["status"] == "restored"
        task = asyncio.create_task(session.execute(""))
        await runtime.wait_for(lambda event: event["type"] == "session.ready", 10)
        return session, runtime, context, provider, task

    async def close(active):
        session, _, _, provider, task = active
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await session.cleanup()
        await provider.close()

    active = await start()
    checkpoints, evidence = [], []
    try:
        for turn in range(turns):
            _, runtime, context, _, task = active
            assert not task.done()
            original = await context.get_messages()
            sequence = runtime.sequence
            await runtime.submit(Input("user", f"Run synthetic operation {turn} once. Budget=25. Latest correction=turn-{turn}. Publication=pending.", id=f"input-{turn}"))
            body = await asyncio.wait_for(server.requests.get(), 15)
            assert f"Latest correction=turn-{turn}" in json.dumps(body)
            if context.summary and mode != "native":
                # Every covered completed receipt must still be available in
                # the summary request view after repeated compaction/resume.
                canonical = await context.get_messages()
                covered = json.dumps(canonical[:context.summary[0]])
                receipts = set(re.findall(r"receipt=R-\d+", covered))
                assert receipts <= set(re.findall(r"receipt=R-\d+", json.dumps(body)))
            await server.replies.put(response(calls=[(f"call-{turn}", turn)]))
            body = await asyncio.wait_for(server.requests.get(), 15)
            assert f"receipt=R-{turn}" in json.dumps(body)
            await server.replies.put(response(f"Verified operation {turn}; publication remains pending."))
            event = await runtime.wait_for(lambda event, after=sequence: event["sequence"] > after and
                event["type"] in {"generation.finished", "generation.failed"}, 15)
            assert event["type"] == "generation.finished", event
            messages = await context.get_messages()
            assert messages[:len(original)] == original, "Canonical earlier messages changed"
            assert set(counts) == set(range(turn+1)) and all(n == 1 for n in counts.values())
            evidence.append({"turn": turn+1, "canonical_messages": len(messages), "canonical_prefix_unchanged": True,
                "canonical_sha256": checksum(messages), "completed_once": turn+1})
            if turn == turns//2:
                checkpoint = context.export_checkpoint(IDENTITY)
                assert checkpoint and context.summary
                checkpoints.append({"turn": turn+1, "sourceRevision": checkpoint["sourceRevision"]})
                await close(active)
                active = await start((messages, json.loads(json.dumps(checkpoint))))
        completed = [data for kind, data in events if kind == "context:compaction_finished" and data["outcome"] == "completed"]
        assert len(completed) >= 3
        expected = "native" if mode == "native" else "semantic"
        assert all(data["method"] == expected for data in completed)
        if mode == "fallback":
            assert all(data["native_failure"]["type"] == "BadRequestError" for data in completed)
        assert all(data["outcome"] == "completed" for kind, data in events if kind == "context:compaction_finished")
        return {"mode": mode, "passed": True, "turns": evidence, "checkpoints": checkpoints,
            "compaction_count": len(completed), "native_http_calls": server.compactions,
            "summary_http_calls": server.summaries, "provider_counts": server.counts,
            "model_requests": len(server.normal_wires), "tool_operations": dict(counts),
            "scope": "Real Core + loop-live + context + provider + SDK; scripted in-memory HTTP and synthetic tool only."}
    finally:
        await close(active)


async def main(args):
    denials = []
    def refuse(*unused, **kwargs):
        denials.append("real_network_attempt")
        raise AssertionError("No real network permitted")
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.getaddrinfo = refuse
    reports = []
    for mode in args.mode:
        report = await run_case(mode, args.turns)
        reports.append(report)
        assert not denials
        Path(args.output).write_text(json.dumps({"cases": reports, "actual_network_attempts": len(denials)}, indent=2)+"\n")
        print(json.dumps({k: v for k, v in report.items() if k not in {"turns", "tool_operations"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument("--mode", nargs="+", choices=["native", "semantic", "fallback"], default=["native", "semantic", "fallback"])
    asyncio.run(main(parser.parse_args()))
