"""Exercise actual loop-live job replacement with synthetic providers, no network.

Run in an environment containing context-managed and loop-live. Set PYTHONPATH
only to the candidate context-managed module to compare it with an installed
baseline. No app, saved session or real provider is instantiated.
"""
import argparse
import asyncio
import copy
import hashlib
import inspect
import json
import socket
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock


def deny(*args, **kwargs):
    raise AssertionError("Network is forbidden in this offline validation")


socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.getaddrinfo = deny

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_loop_live.orchestrator import BundleLiveOrchestrator


def source(cls):
    path = Path(inspect.getfile(cls))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


async def scenario(method, replacements, chars):
    identity = {"provider": "fixture", "model": "fixture"}
    manager = BoundaryContextManager({"max_tokens": 10000, "summarize_trigger": .2,
        "durable_checkpoints": True})
    manager.checkpoint_identity = lambda: identity
    native = {"role": "user", "content": "Native fixture", "metadata": {
        "source": "context-managed", "ephemeral": True, "persisted": True, "opaque": "fixture"}}
    model = SimpleNamespace(name="fixture", default_model="fixture",
        complete=AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(type="text", text="Original objective and constraints.")])))
    if method == "native":
        model.supports_native_compaction = lambda: True
        model.validate_compacted_context = lambda row: bool((row.get("metadata") or {}).get("opaque"))
        model.compact_context = AsyncMock(return_value={"kind": "native", "message": native})
        model.request_budget = lambda request, **kw: {"measurement": {"kind": "provider_count",
            "input_tokens": 100 if any((row.metadata or {}).get("opaque") for row in request.messages) else 9000}}
    receipt = json.dumps({"status": "pending", "job_id": "job-0"})
    originals = [
        {"role": "user", "content": "Original objective and constraints."},
        {"role": "assistant", "content": ("Historical evidence. " * (chars // 21 + 1))[:chars]},
        {"role": "user", "content": "Preserve the original files."},
        {"role": "assistant", "content": "Second human turn complete."},
        {"role": "user", "content": "Finish current work."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-0", "name": "delegate", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "call-0", "content": receipt}]
    await manager.set_messages(originals)
    await manager._prepare(model, None, [])
    assert manager.summary and manager.summary[0] == 2
    initial_calls = model.compact_context.await_count if method == "native" else model.complete.await_count
    timings = []
    for i in range(replacements):
        call_id = f"call-{i}"
        if i:
            receipt = json.dumps({"status": "pending", "job_id": f"job-{i}"})
            await manager.add_message({"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "name": "delegate", "arguments": {}}]})
            await manager.add_message({"role": "tool", "tool_call_id": call_id, "content": receipt})
        result = f"Exact completed result {i}"
        job = {"call_id": call_id, "receipt": receipt, "result": result}
        loop = SimpleNamespace(native_job=lambda value: job if value == call_id else None)
        started = time.perf_counter()
        await BundleLiveOrchestrator._synchronize_job_results(loop, manager)
        timings.append((time.perf_counter() - started) * 1000)
        await manager._prepare(model, None, [])
        assert (await manager.get_messages())[-1]["content"] == result
    canonical = await manager.get_messages()
    assert canonical[:2] == originals[:2]
    calls_after_updates = model.compact_context.await_count if method == "native" else model.complete.await_count
    checkpoint = manager.export_checkpoint(identity)
    restored = BoundaryContextManager(manager.config)
    restored.checkpoint_identity = manager.checkpoint_identity
    await restored.set_messages(canonical)
    assert restored.restore_checkpoint(checkpoint, identity)["status"] == "restored"
    replacement = copy.deepcopy(canonical)
    replacement[-1]["content"] = "Restored suffix correction"
    await restored.set_messages(replacement)
    await restored._prepare(model, None, [])
    after_restore = model.compact_context.await_count if method == "native" else model.complete.await_count
    replacement[0]["content"] = "Changed authoritative objective"
    await restored.set_messages(replacement)
    assert restored.summary is None
    return {"method": method, "sourceChars": chars, "replacements": replacements,
        "initialCalls": initial_calls, "callsAfterReplacements": calls_after_updates,
        "callsAfterRestoredReplacement": after_restore,
        "preservedWithoutRecompaction": initial_calls == calls_after_updates == after_restore,
        "changedCoveredPrefixInvalidates": True, "allExactResultsRetained": True,
        "replacementMedianMs": round(statistics.median(timings), 3),
        "replacementMaxMs": round(max(timings), 3)}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-baseline-failure", action="store_true")
    args = parser.parse_args()
    rows = [await scenario("native", 50, 3600000), await scenario("semantic", 50, 30000)]
    result = {"contextSource": source(BoundaryContextManager), "loopSource": source(BundleLiveOrchestrator),
        "networkRequests": 0, "modelRequests": 0, "appAccessed": False, "cases": rows,
        "passed": all(row["preservedWithoutRecompaction"] for row in rows)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not args.allow_baseline_failure:
        assert result["passed"], "Unchanged covered prefix was compacted again after a job result replacement"


if __name__ == "__main__":
    asyncio.run(main())
