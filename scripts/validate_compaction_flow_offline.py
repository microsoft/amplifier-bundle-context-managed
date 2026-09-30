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
import hashlib
import inspect
import sys
import time
from collections import Counter
from pathlib import Path
from typing import ClassVar
from types import SimpleNamespace

import httpx
import openai
from amplifier_core import AmplifierSession, HookResult, ToolResult
from amplifier_module_loop_live.runtime import Input, Runtime
from amplifier_module_loop_live.orchestrator import BundleLiveOrchestrator
from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_context_managed.boundary import BoundaryContextManager
from validate_compaction_wait_offline import checksum

MODEL = "gpt-6.1-sol"
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
        sdk = openai.AsyncOpenAI(api_key="fixture", base_url="https://api.openai.com/v1", timeout=0.001, max_retries=0,
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


async def run_work_budget_cases():
    """Exercise installed Core mounts and logical SDK calls on an unchanged prefix."""
    reports = []
    for mode in ("slow-fragments", "auto-continuations", "resume-work", "stalled-replacements"):
        events, requests, cancellations = [], [], []
        config = {"engine": "boundary", "durable_checkpoints": True,
            "max_tokens": 10000, "summarize_trigger": .01, "native_compaction": False,
            "summary_max_source_chars": 6000, "summary_retry_delay": 0,
            "summary_timeout": .12, "summary_total_work_timeout": .5}
        if mode == "resume-work":
            config.update(summary_timeout=.5, summary_total_work_timeout=.06)
        session = AmplifierSession({"session": {
            "orchestrator": {"module": "loop-live", "config": {"configured_bundle": True}},
            "context": {"module": "context-managed", "config": config}},
            "providers": [], "tools": [], "hooks": []})
        await session.initialize()
        coordinator = session.coordinator
        coordinator.register_capability("context.checkpoint_identity", lambda: IDENTITY)
        async def preserve(rows):
            return [{"kind": "synthetic-history", "sha256": checksum(rows)}]
        coordinator.register_capability("context.preserve_evidence", preserve)
        async def observe(kind, data):
            events.append(copy.deepcopy(data))
            return HookResult()
        coordinator.hooks.register("context:compaction_finished", observe)

        async def server(request):
            body = json.loads(request.content)
            wire = json.dumps(body.get("input", []))
            if request.url.path.endswith("input_tokens"):
                return httpx.Response(200, json={"input_tokens": len(wire)//4})
            assert request.url.path == "/v1/responses"
            assert "Prepare a factual continuation note" in body.get("instructions", "")
            requests.append(copy.deepcopy(body))
            if mode == "stalled-replacements" or (mode == "resume-work" and len(requests) == 2):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancellations.append(len(requests))
                    raise
            if mode in {"slow-fragments", "auto-continuations"}:
                await asyncio.sleep(.03)
            if mode == "auto-continuations" and len(requests) % 2:
                partial = response()
                partial.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
                return httpx.Response(200, json=partial)
            return httpx.Response(200, json=response(
                "ORBIT; budget=25; publication=pending; FIRST and LAST evidence verified; preserve originals."))

        sdk = openai.AsyncOpenAI(api_key="fixture", base_url="https://api.openai.com/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
        provider = OpenAIProvider(client=sdk, config={"default_model": MODEL,
            "enable_long_context": True, "use_streaming": False, "max_retries": 0})
        provider.coordinator = coordinator
        await coordinator.mount("providers", provider, name="openai")
        manager = coordinator.get("context")
        receipt = json.dumps({"status": "pending", "job_id": "fixture-job"})
        original = [{"role": "user", "content": "ORBIT; budget=25; preserve originals."},
            {"role": "assistant", "content": "FIRST verified evidence. " + "x"*22000 + " LAST verified evidence."},
            {"role": "user", "content": "Preserve all original tools and instructions."},
            {"role": "assistant", "content": "Recent completed work."},
            {"role": "user", "content": "Finish the report; publication=pending."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call", "name": "delegate", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "call", "content": receipt}]
        await manager.set_messages(original)
        started = time.monotonic()
        try:
            await manager._prepare(provider, None, [])
            first_calls = len(requests)
            if mode == "resume-work":
                assert manager.summary is None and manager.export_checkpoint(IDENTITY) is None
                assert manager._summary_progress["completed"] >= 1
                staged = manager._summary_progress["completed"]
                job = {"receipt": receipt, "result": "Verified completed background result."}
                await BundleLiveOrchestrator._synchronize_job_results(
                    SimpleNamespace(native_job=lambda call: job), manager)
                assert manager._summary_progress["completed"] == staged
                await manager._prepare(provider, None, [])
                assert requests[first_calls]["input"] == requests[first_calls-1]["input"]
                assert events[0]["timeout"]["stage"] == "semantic_work_pass"
                assert events[-1]["resumed_fragments"] == staged
            elif mode == "stalled-replacements":
                for number in range(8):
                    replacement = await manager.get_messages()
                    replacement[-1]["content"] = receipt
                    await manager.set_messages(replacement)
                    job = {"receipt": receipt, "result": f"Verified completed result {number}"}
                    await BundleLiveOrchestrator._synchronize_job_results(
                        SimpleNamespace(native_job=lambda call: job), manager)
                    await manager._prepare(provider, None, [])
                assert len(requests) == 3 and len(cancellations) == 3
                assert manager.summary is None and manager.summary_failure["attempts"] == 3
            if mode != "stalled-replacements":
                assert manager.summary and manager._summary_progress is None
                checkpoint = manager.export_checkpoint(IDENTITY)
                assert checkpoint and checkpoint["summary"]["throughMessage"] == 2
                restored = BoundaryContextManager(manager.config)
                restored.checkpoint_identity = lambda: IDENTITY
                await restored.set_messages(await manager.get_messages())
                assert restored.restore_checkpoint(checkpoint, IDENTITY)["status"] == "restored"
                assert restored.summary == manager.summary
            if mode == "auto-continuations":
                assert len(requests) == 2 * events[-1]["calls"]
                assert all(body["reasoning"]["effort"] == "low" for body in requests)
            canonical = await manager.get_messages()
            assert canonical[:2] == original[:2]
            reports.append({"mode": mode, "passed": True,
                "summary_http_requests": len(requests),
                "logical_summary_calls": sum(event["calls"] for event in events),
                "cancelled_requests": len(cancellations),
                "elapsed_seconds": round(time.monotonic()-started, 3),
                "canonical_prefix_sha256": checksum(canonical[:2]), "covered_prefix_unchanged": True,
                "checkpoint_round_trip": mode != "stalled-replacements", "events": events})
        finally:
            await session.cleanup()
            await provider.close()
    return {"mode": "work-budget", "passed": True, "cases": reports,
        "scope": "Installed real Core mount + loop-live replacement + OpenAI provider/SDK, synthetic in-memory HTTP only."}


def loaded_sources():
    from importlib.metadata import version
    result = {"python": sys.executable, "core_version": version("amplifier-core"),
              "sdk_version": version("openai"), "modules": {}}
    for name in ("amplifier_core", "amplifier_module_context_managed.boundary",
                 "amplifier_module_context_simple", "amplifier_module_loop_live.orchestrator",
                 "amplifier_module_loop_streaming", "amplifier_module_provider_openai"):
        module = __import__(name, fromlist=["__name__"])
        path = Path(inspect.getfile(module))
        result["modules"][name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


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
        report = await run_work_budget_cases() if mode == "work-budget" else await run_case(mode, args.turns)
        reports.append(report)
        assert not denials
        Path(args.output).write_text(json.dumps({"cases": reports, "actual_network_attempts": len(denials),
            "loaded_sources": loaded_sources()}, indent=2)+"\n")
        print(json.dumps({k: v for k, v in report.items() if k not in {"turns", "tool_operations", "cases"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument("--mode", nargs="+", choices=["native", "semantic", "fallback", "work-budget"], default=["native", "semantic", "fallback", "work-budget"])
    asyncio.run(main(parser.parse_args()))
