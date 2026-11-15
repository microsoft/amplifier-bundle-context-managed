"""Offline integration with the installed Unified wrapper; no live provider calls.

Run explicitly with the qualified public Unified wheel installed. This is not a
model-capability test: the stub's selected-model assembler owns its actual clamp
or refusal independently of potentially inherited provider defaults.
"""
import copy
import json
from types import SimpleNamespace

import pytest
from amplifier_core import ProviderInfo
from amplifier_core.llm_errors import ContextLengthError
from amplifier_module_context_managed.summary import request_fits, summary_config, summary_request


@pytest.mark.asyncio
@pytest.mark.parametrize("window,foreground,explicit,ceiling,policy,mode", [
    (1000000, 128000, None, 128000, 32768, "clamp"),
    (200000, 128000, None, 128000, 32768, "clamp"),
    # Raw defaults can describe another model: final selection still clamps.
    (1000000, 128000, None, 4096, 32768, "clamp"),
    (64000, 128000, None, 128000, 8192, "clamp"),
    (1000000, 4096, None, 128000, 4096, "clamp"),
    (1000000, 4096, 32768, 128000, 32768, "clamp"),
    (1000000, 128000, 4096, 128000, 4096, "clamp"),
    (1000000, 128000, None, 4096, 32768, "reject"),
])
async def test_selected_wrapper_uses_same_summary_contract_for_count_and_dispatch(
        window, foreground, explicit, ceiling, policy, mode):
    SelectedProvider = pytest.importorskip("amplifier_web.host.session").SelectedProvider
    counts, dispatched = [], []

    def assembled(request, kwargs):
        assert request.model == "selected" and kwargs["model"] == "selected"
        assert request.reasoning_effort == "low" and kwargs["reasoning_effort"] == "low"
        if mode == "reject" and request.max_output_tokens > ceiling:
            raise ContextLengthError("Selected model output ceiling")
        return {"model": request.model, "max_tokens": min(request.max_output_tokens, ceiling),
                "messages": [row.model_dump() for row in request.messages]}

    async def count(request, **kwargs):
        wire = assembled(request, kwargs)
        counts.append((request, copy.deepcopy(wire)))
        tokens = sum(len(str(row.content)) for row in request.messages) // 4
        return {"estimated_input_tokens": tokens,
                "input_limit_tokens": 1000000 - wire["max_tokens"] - 4096,
                "measurement": {"kind": "provider_count", "input_tokens": tokens}}

    async def complete(request, **kwargs):
        wire = assembled(request, kwargs)
        assert counts[-1][0] is request
        assert counts[-1][1] == wire
        dispatched.append(wire)
        return SimpleNamespace(finish_reason="end_turn", content=[])

    raw = SimpleNamespace(name="fixture", default_model="base", request_budget=count,
        complete=complete, get_info=lambda: ProviderInfo(id="fixture", display_name="Fixture",
            defaults={"model": "base", "context_window": window, "max_output_tokens": 128000}))
    selected = SelectedProvider(raw, {"model": "selected", "max_output_tokens": foreground,
                                       "effort": "high"})
    configured = {} if explicit is None else {"summary_max_output_tokens": explicit}
    resolved, _ = await summary_config(selected, configured)
    assert configured == ({} if explicit is None else {"summary_max_output_tokens": explicit})
    assert resolved["summary_max_output_tokens"] == policy
    request = summary_request("instructions", "source", "previous note", resolved)
    if mode == "reject":
        with pytest.raises(ContextLengthError):
            await request_fits(selected, request)
        assert not counts and not dispatched
    else:
        fits, budget = await request_fits(selected, request)
        assert fits and budget["output_reserve"] == policy
        await selected.complete(request)
        assert dispatched[0]["max_tokens"] == min(policy, ceiling)
        assert selected._selected_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit,expected", [(None, 32768), (8192, 8192), (200000, 128000)])
async def test_actual_anthropic_assembler_keeps_summary_allowance_and_model_ceiling(explicit, expected):
    module = pytest.importorskip("amplifier_module_provider_anthropic")
    SelectedProvider = pytest.importorskip("amplifier_web.host.session").SelectedProvider
    from anthropic import _base_client
    http = getattr(_base_client, "httpx2", None) or _base_client.httpx
    wires, counts = [], []

    async def transport(request):
        if request.method == "GET":
            return http.Response(404, json={"type": "error", "error": {
                "type": "not_found_error", "message": "No fixture metadata"}})
        body = json.loads(request.content)
        if request.url.path.endswith("/count_tokens"):
            counts.append(body)
            return http.Response(200, json={"input_tokens": 1000})
        wires.append(body)
        assert body["model"] == "claude-fable-5-2" and body["max_tokens"] == expected
        assert body["output_config"]["effort"] == "low"
        return http.Response(200, json={
            "id": "synthetic-summary", "type": "message", "role": "assistant",
            "model": body["model"], "content": [{"type": "text", "text": "Completed note."}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 12000 if expected > 8192 else 7208}})

    sdk = module.AsyncAnthropic(api_key="synthetic-not-live", max_retries=0,
        http_client=http.AsyncClient(transport=http.MockTransport(transport)))
    raw = module.AnthropicProvider(api_key="synthetic-not-live", config={
        "default_model": "claude-fable-5-2", "use_streaming": False,
        "max_retries": 0, "enable_prompt_caching": False})
    raw._client = sdk
    selected = SelectedProvider(raw, {"model": "claude-fable-5-2", "max_output_tokens": 128000})
    try:
        config = {} if explicit is None else {"summary_max_output_tokens": explicit}
        resolved, _ = await summary_config(selected, config)
        request = summary_request("instructions", "ORBIT source", "", resolved)
        fits, budget = await request_fits(selected, request)
        assert fits and budget["measurement_kind"] == "provider_count"
        assert budget["input_tokens"] == 1000 + 4096  # reserve is not dropped
        response = await selected.complete(request, request_options={"auto_continue": False})
        assert response.finish_reason == "end_turn"
        assert counts and len(wires) == 1
        assert all(row["model"] == "claude-fable-5-2" for row in counts)
        assert request.max_output_tokens == (32768 if explicit is None else explicit)
    finally:
        await sdk.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected_fit", [(950000, True), (980000, False), (1000001, False)])
async def test_padded_count_summary_admission_preserves_full_output_headroom(raw, expected_fit):
    """Conservative portable-summary admission, even with input-only provider limits."""
    module = pytest.importorskip("amplifier_module_provider_anthropic")
    SelectedProvider = pytest.importorskip("amplifier_web.host.session").SelectedProvider
    from anthropic import _base_client
    http = getattr(_base_client, "httpx2", None) or _base_client.httpx
    sends = []

    async def transport(request):
        if request.method == "GET":
            return http.Response(404, json={"type": "error", "error": {
                "type": "not_found_error", "message": "No fixture metadata"}})
        if request.url.path.endswith("/count_tokens"):
            return http.Response(200, json={"input_tokens": raw})
        sends.append(request)
        raise AssertionError("Near-window count tests never dispatch completion")

    sdk = module.AsyncAnthropic(api_key="synthetic-not-live", max_retries=0,
        http_client=http.AsyncClient(transport=http.MockTransport(transport)))
    provider = module.AnthropicProvider(api_key="synthetic-not-live", config={
        "default_model": "claude-fable-5-2", "enable_1m_context": True,
        "use_streaming": False, "max_retries": 0})
    provider._client = sdk
    selected = SelectedProvider(provider, {"model": "claude-fable-5-2", "max_output_tokens": 128000})
    try:
        request = summary_request("instructions", "Source", "", {"summary_max_output_tokens": 32768})
        fits, budget = await request_fits(selected, request)
        assert fits is expected_fit
        assert budget["input_tokens"] == raw + 4096
        assert budget["output_reserve"] == 32768
        assert not sends
    finally:
        await sdk.close()
