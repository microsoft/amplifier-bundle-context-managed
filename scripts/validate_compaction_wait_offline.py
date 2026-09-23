"""Actual context + OpenAI provider + SDK, synthetic in-memory HTTP only.

This deliberately waits longer than the old 120-second context deadline. It
validates transport/orchestration, not model summary quality or live deployment.
Run with the candidate context and provider checkouts on PYTHONPATH.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_provider_openai import OpenAIProvider


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def history(turns=80):
    rows = [{"role": "system", "content": "Preserve originals; never replay tools."}]
    for turn in range(turns):
        rows.extend([
            {"role": "user", "content": f"ORBIT turn {turn}; budget=25; artifact=/reports/orbit.txt; publication=pending."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"call-{turn}", "name": "fixture", "arguments": {"operation": f"op-{turn}"}}]},
            {"role": "tool", "tool_call_id": f"call-{turn}",
             "content": f"receipt=R-{turn}; completed=true; " + "Historical evidence. " * 200},
            {"role": "assistant", "content": f"Verified R-{turn}; do not execute again."},
        ])
    rows.append({"role": "user", "content": "Latest correction: deadline=Thursday. Continue the report."})
    return rows


class FixtureHTTP:
    """A synthetic server; only this callable receives HTTP requests."""

    def __init__(self, mode, delay):
        self.mode, self.delay = mode, delay
        self.calls, self.timeouts, self.events = [], [], []
        self.entered = asyncio.Event()
        self.cancelled = False

    async def __call__(self, request):
        body = json.loads(request.content)
        path = request.url.path
        self.calls.append(path)
        if path.endswith("input_tokens"):
            # Match a provider-count contract, with a compacted opaque window
            # deliberately much smaller than the canonical synthetic history.
            count = 500 if any(item.get("type") == "compaction" for item in body["input"]) else 100_000
            return httpx.Response(200, json={"input_tokens": count})
        self.timeouts.append(dict(request.extensions.get("timeout", {})))
        if (self.mode == "native-error" and path.endswith("compact")) or self.mode == "both-error":
            return httpx.Response(400, json={"error": {"message": "synthetic native rejection", "type": "invalid_request_error"}})
        self.entered.set()
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if path.endswith("compact"):
            return httpx.Response(200, json={
                "id": "cmp_fixture", "object": "response.compaction", "created_at": 1,
                "output": [
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ORBIT; budget=25; publication=pending."}]},
                    {"type": "compaction", "encrypted_content": "synthetic-opaque-state"},
                ],
                "usage": {"input_tokens": 100_000, "output_tokens": 500, "total_tokens": 100_500},
            })
        return httpx.Response(200, json={
            "id": "resp_fixture", "object": "response", "created_at": 1,
            "model": "gpt-6-astra", "status": "completed",
            "output": [{"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ORBIT; budget=25; artifact=/reports/orbit.txt; publication=pending; completed receipts are evidence, never replay them.", "annotations": []}]}],
            "usage": {"input_tokens": 1000, "output_tokens": 40, "total_tokens": 1040},
        })


async def run_case(mode, delay, cancel=False):
    fixture = FixtureHTTP(mode, delay)
    sdk = openai.AsyncOpenAI(api_key="fixture", timeout=0.001, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fixture)))
    provider = OpenAIProvider(client=sdk, config={"default_model": "gpt-6-astra", "enable_long_context": True,
        "use_streaming": False, "max_retries": 0})
    events = []
    async def emit(kind, data):
        events.append((kind, copy.deepcopy(data)))
    manager = BoundaryContextManager({"max_tokens": 200000, "summarize_trigger": 0.1,
        "summary_max_source_chars": 1_000_000, "native_compaction": mode != "summary"}, SimpleNamespace(emit=emit))
    original = history()
    await manager.set_messages(original)
    before = checksum(original)
    start = time.monotonic()
    task = asyncio.create_task(manager.get_messages_for_request(
        provider=provider, token_budget=4000 if mode == "both-error" else None))
    try:
        if cancel:
            await fixture.entered.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("Cancellation did not propagate")
            assert manager.summary is None
            assert fixture.cancelled
            assert len(fixture.timeouts) == 1
        else:
            result = await task
            if mode == "both-error":
                assert manager.summary is None
                assert 0 < len(result) < len(original)
            else:
                assert manager.summary is not None
            assert "deadline=Thursday" in str(result)
        assert checksum(await manager.get_messages()) == before
        assert all(timeout["read"] is None and timeout["write"] is None for timeout in fixture.timeouts)
        assert all(timeout["connect"] == 5 and timeout["pool"] == 5 for timeout in fixture.timeouts)
        event = [data for kind, data in events if kind == "context:compaction_finished"][-1]
        assert event["outcome"] == ("cancelled" if cancel else "fallback" if mode == "both-error" else "completed")
        if mode in {"native-error", "both-error"}:
            assert event["method"] == "semantic" and event["native_failure"]["type"] == "BadRequestError"
            assert fixture.calls.index("/v1/responses/compact") < fixture.calls.index("/v1/responses")
        if mode == "both-error":
            assert event["calls"] == 2
            assert event["failure"]["type"] == "InvalidRequestError"
        result = {"mode": mode, "cancel": cancel, "elapsed_seconds": round(time.monotonic()-start, 3),
            "canonical_messages": len(original), "canonical_sha256": before, "originals_unchanged": True,
            "sdk_timeout_initially": 0.001, "actual_model_timeouts": fixture.timeouts,
            "http_paths": fixture.calls, "compaction": event}
        print(json.dumps(result), flush=True)
        return result
    finally:
        await provider.close()


async def main(args):
    denied = []
    def refuse(*unused, **kwargs):
        denied.append("network_attempt")
        raise AssertionError("Real network is forbidden in this offline qualification")
    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.getaddrinfo = refuse
    started = time.time()
    cases = await asyncio.gather(run_case("native", args.delay), run_case("summary", args.delay),
        run_case("native-error", 0), run_case("both-error", 0),
        run_case("native", 60, cancel=True), run_case("summary", 60, cancel=True))
    assert not denied
    Path(args.output).write_text(json.dumps({"started_at": started, "cases": cases,
        "actual_network_attempts": len(denied), "scope": "Actual candidate context and provider with real OpenAI SDK; in-memory HTTP fixture, no real model or tool execution."}, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=125)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
