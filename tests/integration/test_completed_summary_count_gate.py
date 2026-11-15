"""Offline accepted Loop/Boundary/Anthropic count-outage retention boundary.

A synthetic HTTP400 is NOT a diagnosis of the Mac endpoint's unknown400 body.
It only proves that a completed paid note survives the strict no-uncounted-retry
boundary, and can be reused when the same provider later returns a valid count.
"""
import copy
import json
from pathlib import Path

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.checkpoint import digest


@pytest.mark.asyncio
async def test_completed_summary_survives_sdk_count400_and_strict_loop_gate():
    foundation = pytest.importorskip("amplifier_foundation.bundle")
    streaming = pytest.importorskip("amplifier_module_loop_streaming")
    provider_module = pytest.importorskip("amplifier_module_provider_anthropic")
    managed = pytest.importorskip("amplifier_module_context_managed")
    SelectedProvider = pytest.importorskip("amplifier_web.host.session").SelectedProvider
    from anthropic import _base_client
    http = getattr(_base_client, "httpx2", None) or _base_client.httpx
    identity = {"provider": "anthropic", "model": "claude-fable-5-2"}
    foreground, summaries, counts = [], [], []
    count_available = False

    async def transport(request):
        if request.method == "GET":
            return http.Response(404, json={"type": "error", "error": {
                "type": "not_found_error", "message": "No synthetic model metadata"}})
        body = json.loads(request.content)
        if request.url.path.endswith("/count_tokens"):
            counts.append(body)
            if not count_available:
                return http.Response(400, headers={"request-id": "synthetic-count400"},
                    json={"type": "error", "error": {"type": "invalid_request_error",
                          "message": "Synthetic unavailable counter; not the historical Mac cause"}})
            return http.Response(200, json={"input_tokens": 2000})
        assert request.url.path.endswith("/messages") and body["model"] == identity["model"]
        if "Next ordered conversation data. Incorporate this evidence" in json.dumps(body):
            summaries.append(body)
            text = "ORBIT completed evidence; RECEIPT-old remains completed. Keep originals."
        else:
            foreground.append(body)
            return http.Response(400, headers={"request-id": "synthetic-foreground-overflow"},
                json={"type": "error", "error": {"type": "invalid_request_error",
                      "message": "prompt is too long: 1000507 tokens > 1000000 maximum"}})
        return http.Response(200, json={"id": "synthetic-note", "type": "message",
            "role": "assistant", "model": identity["model"], "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 100}})

    plan = {"session": {
        "orchestrator": {"module": "loop-streaming", "config": {
            "use_streaming": False, "min_delay_between_calls_ms": 0,
            "ephemeral_injection_mode": "tail"}},
        "context": {"module": "context-managed", "config": {
            "max_tokens": 1000000, "summarize_trigger": .7,
            "durable_checkpoints": True, "compaction_notice_enabled": False}}}}
    session = await foundation.PreparedBundle(plan, foundation.BundleModuleResolver({
        "context-managed": Path(managed.__file__).parent,
        "loop-streaming": Path(streaming.__file__).parent}),
        foundation.Bundle(name="count-outage-retention", session=plan["session"])).create_session(
            session_id="count-outage-retention")
    coordinator = session.coordinator
    context = coordinator.get("context")

    async def preserve(messages):
        return [{"kind": "synthetic-existing-originals", "sha256": digest(messages)}]

    coordinator.register_capability("context.checkpoint_identity", lambda: copy.deepcopy(identity))
    coordinator.register_capability("context.preserve_evidence", preserve)
    sdk = provider_module.AsyncAnthropic(api_key="synthetic-not-live", max_retries=0,
        http_client=http.AsyncClient(transport=http.MockTransport(transport)))
    raw = provider_module.AnthropicProvider(api_key="synthetic-not-live", coordinator=coordinator,
        config={"default_model": identity["model"], "use_streaming": False,
                "enable_1m_context": True, "enable_prompt_caching": False, "max_retries": 0})
    raw._client = sdk
    selected = SelectedProvider(raw, {"model": identity["model"], "max_output_tokens": 128000})
    await coordinator.mount("providers", selected, name="anthropic")
    coordinator.get("orchestrator").root_provider = selected
    seed = [
        {"role": "user", "content": "ORBIT preserve completed evidence."},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "",
            "signature": "synthetic-signed-original" * 200000},
            {"type": "text", "text": "Verified evidence " * 500}],
            "tool_calls": [{"id": "old", "name": "record", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "old", "content": "RECEIPT-old completed " * 100},
        {"role": "user", "content": "Accepted earlier correction."},
        {"role": "assistant", "content": "Correction acknowledged."},
        {"role": "user", "content": "Latest retained correction."},
        {"role": "assistant", "content": "Acknowledged."},
    ]
    await context.set_messages(seed)
    original_hashes = [digest(row) for row in seed]
    try:
        with pytest.raises(Exception, match="requires a provider count"):
            await session.execute("Continue ORBIT; never replay completed tools.")
        assert len(foreground) == 1 and summaries and counts
        assert all(row["max_tokens"] == 32768 for row in summaries)
        assert context.summary is not None
        canonical = await context.get_messages()
        assert [digest(row) for row in canonical[:len(seed)]] == original_hashes
        assert digest(canonical[2]) == digest(seed[2])
        record = context.export_checkpoint(identity)
        assert record["summary"] is not None and record["progress"] is None
        completed_calls = len(summaries)
        restored = BoundaryContextManager(copy.deepcopy(context.config))
        restored.checkpoint_identity = lambda: copy.deepcopy(identity)
        await restored.set_messages(canonical)
        assert restored.restore_checkpoint(record, identity)["status"] == "restored"
        count_available = True

        async def count(view):
            request = ChatRequest(messages=[Message(**row) for row in view], max_output_tokens=128000)
            decision = await selected.request_budget(request, context_estimate=len(str(view)))
            return {"dispatch": request, "budget_decision": decision}

        result = await restored.get_measured_request_view(provider=selected, retain_contents=[], count_view=count)
        budget = result["final_attempt"]["budget_decision"]
        assert budget["measurement"]["input_tokens"] == 2000
        assert budget["estimated_input_tokens"] == 6096
        assert len(summaries) == completed_calls and len(foreground) == 1
        assert await restored.get_messages() == canonical
    finally:
        await sdk.close()
        await session.cleanup()
