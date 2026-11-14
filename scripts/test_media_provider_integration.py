"""Explicit offline owner-wrapper integration; run with MEDIA_PROVIDER_SOURCE.

The source path must be an exported, exact committed media_provider.py from
Foundation. It is loaded only by this test, never imported by the context module.
A structural prepared-capability fixture substitutes for assets/authorization;
no pixels, image decoding, provider SDK, network or inference is involved.
"""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import pytest
from amplifier_core import ChatRequest, ImageBlock, Message, TextBlock
from amplifier_core.llm_errors import ContextLengthError

from amplifier_module_context_managed.errors import CompactionError
from amplifier_module_context_managed.measurement import RequestMeasurement
from amplifier_module_context_managed.request_context import RequestContext

MEDIA = "experimental.media.v1"


def rows():
    return [{"role": "user", "content": "Observe."},
            {"role": "assistant", "content": "Inspecting observations."},
            {"role": "tool", "tool_call_id": "inspect", "content": "Observations.",
             "metadata": {MEDIA: {"refs": [{"fixture_id": "b"}, {"fixture_id": "a"}, {"fixture_id": "b"}]}}}]


@pytest.fixture(scope="module")
def wrapper_type():
    path = Path(os.environ["MEDIA_PROVIDER_SOURCE"])
    expected = os.environ["MEDIA_PROVIDER_SHA256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location("_owner_media_provider_integration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MediaPreparedProvider


class Prepared:
    def __init__(self, capability, canonical, binding):
        self.capability = capability
        self.canonical = canonical
        self.original = canonical.model_dump()
        self.binding = copy.deepcopy(binding)
        self.released = False
        self.release_calls = 0
        self.request = canonical.model_copy(deep=True)
        self.request.messages[-1].metadata.pop(MEDIA, None)
        self.request.messages[-1].content = [
            TextBlock(type="text", text=self.request.messages[-1].content),
            *[TextBlock(type="text", text="[Image unavailable: provider or execution scope does not support this image position; no pixels supplied]")
              for _ in range(3)],
        ]
        if capability.warning_chars:
            self.request.messages[-1].content.append(TextBlock(type="text", text="warning details " * capability.warning_chars))
        if capability.hydrated:
            self.request.messages[-1].content.append(ImageBlock(type="image", source={"type": "base64", "media_type": "image/png", "data": "SENTINEL"}))
        self.projected = self.request.model_dump()
        self.projection_digest = hashlib.sha256(repr(self.projected).encode()).hexdigest()

    def validate(self, *, request_options=None):
        if (self.released or not self.capability.valid
                or self.canonical.model_dump() != self.original
                or self.request.model_dump() != self.projected
                or (request_options or {}) != self.binding["request_options"]):
            raise ValueError("Fixture media validation refused")
        return {"valid": True}

    def release(self):
        self.released = True
        self.release_calls += 1


class Capability:
    def __init__(self):
        self.prepared = []
        self.valid = True
        self.warning_chars = 0
        self.hydrated = False

    async def prepare_request(self, canonical, **binding):
        if not self.valid or binding["request_options"].get("mandatory_vision"):
            raise ValueError("Fixture unowned/malformed/mandatory media refusal")
        prepared = Prepared(self, canonical, binding)
        self.prepared.append(prepared)
        return prepared


class Provider:
    name = "fixture-provider"
    config = {}
    default_model = "fixture-model"

    def __init__(self, capability, *, budget=True):
        self.capability = capability
        self.counted = []
        self.sent = []
        self.has_budget = budget

    async def request_budget(self, request, *, request_options=None, **kwargs):
        self.counted.append((request, copy.deepcopy(request_options)))
        if not self.has_budget:
            return None
        count = 12000 if request.max_output_tokens == 8000 else 3000
        return {"estimated_input_tokens": count, "input_limit_tokens": 10000,
                "context_token_budget": 10000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": count}}

    async def complete(self, request, *, request_options=None):
        self.sent.append((request, copy.deepcopy(request_options)))
        return "fixture complete"


def setup(wrapper_type, *, budget=True):
    capability = Capability()
    provider = Provider(capability, budget=budget)
    wrapper = wrapper_type(provider, capability, provider_identity={"provider": "fixture", "account": "owner"})
    return wrapper, provider, capability


@pytest.mark.asyncio
@pytest.mark.parametrize("preflight", [False, True])
async def test_actual_wrapper_exact_owner_options_survive_preflight_and_final(wrapper_type, preflight):
    wrapper, provider, capability = setup(wrapper_type)
    fitter = RequestContext(max_tokens=10000)
    original = rows()
    await fitter.set_messages(original)
    options = {"extended_thinking": True, "nested": {"mode": "exact"}}

    async def count_view(view):
        request = ChatRequest(messages=[Message(**row) for row in view])
        decision = await wrapper.request_budget(request, request_options=options, context_estimate=1)
        return {"dispatch": request, "budget_decision": decision, "request_options": copy.deepcopy(options)}

    try:
        callback = count_view
        if preflight:
            measurement = RequestMeasurement(count_view, wrapper)
            assert (await measurement.measure(fitter, await fitter._view()))[0] == 3000
            callback = measurement.count
        result = await fitter.get_measured_request_view(provider=wrapper, retain_contents=[], count_view=callback)
        attempt = result["final_attempt"]
        assert await wrapper.complete(attempt["dispatch"], request_options=attempt["request_options"]) == "fixture complete"
        assert provider.counted[-1][0] is provider.sent[-1][0]
        assert provider.counted[-1][1] == provider.sent[-1][1] == options
        assert len(capability.prepared) == 1
        assert capability.prepared[0].release_calls == 1
        assert await fitter.get_messages() == original
    finally:
        wrapper.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", [None, {"extended_thinking": False}])
async def test_actual_wrapper_does_not_guess_missing_or_wrong_options(wrapper_type, mismatch):
    wrapper, provider, capability = setup(wrapper_type)
    fitter = RequestContext(max_tokens=10000)
    await fitter.set_messages(rows())

    async def count_view(view):
        request = ChatRequest(messages=[Message(**row) for row in view])
        decision = await wrapper.request_budget(request, request_options={"extended_thinking": True}, context_estimate=1)
        return {"dispatch": request, "budget_decision": decision, "request_options": mismatch}

    try:
        with pytest.raises(CompactionError, match="media_budget_unavailable"):
            await fitter.get_measured_request_view(provider=wrapper, retain_contents=[], count_view=count_view)
        assert not provider.sent
        assert capability.prepared[0].released
    finally:
        wrapper.close()


@pytest.mark.asyncio
async def test_actual_wrapper_output_fit_preserves_replacement_options(wrapper_type):
    wrapper, provider, capability = setup(wrapper_type)
    fitter = RequestContext(max_tokens=10000)
    original = rows()
    await fitter.set_messages(original)

    async def counted(request, options):
        return {"dispatch": request, "request_options": copy.deepcopy(options),
                "budget_decision": await wrapper.request_budget(request, request_options=options, context_estimate=1)}

    async def count_view(view):
        return await counted(ChatRequest(messages=[Message(**row) for row in view], max_output_tokens=8000),
                             {"extended_thinking": True})

    async def fit_output(view, attempt):
        fitted = await counted(attempt["dispatch"].model_copy(update={"max_output_tokens": 1000}),
                               {"extended_thinking": False, "fitted": True})
        return {**fitted, "count_calls": 1}

    try:
        result = await fitter.get_measured_request_view(provider=wrapper, retain_contents=[],
                                                       count_view=count_view, fit_output=fit_output)
        attempt = result["final_attempt"]
        await wrapper.complete(attempt["dispatch"], request_options=attempt["request_options"])
        assert result["outcome"] == "reduced_output"
        assert result["count_calls"] == 2
        assert provider.sent[0][1] == {"extended_thinking": False, "fitted": True}
        assert capability.prepared[0].released and capability.prepared[1].released
        assert provider.counted[1][0] is provider.sent[0][0]
        assert await fitter.get_messages() == original
    finally:
        wrapper.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["nested_options", "equal_dispatch"])
async def test_actual_wrapper_cached_binding_changes_refuse_before_transport(wrapper_type, change):
    wrapper, provider, capability = setup(wrapper_type)
    fitter = RequestContext(max_tokens=10000)
    await fitter.set_messages(rows())
    envelope = None

    async def count_view(view):
        nonlocal envelope
        request = ChatRequest(messages=[Message(**row) for row in view])
        options = {"extended_thinking": True, "nested": {"mode": "counted"}}
        envelope = {"dispatch": request, "request_options": options,
                    "budget_decision": await wrapper.request_budget(request, request_options=options, context_estimate=1)}
        return envelope

    try:
        measurement = RequestMeasurement(count_view, wrapper)
        await measurement.measure(fitter, await fitter._view())
        if change == "nested_options":
            envelope["request_options"]["nested"]["mode"] = "uncounted"
        else:
            envelope["dispatch"] = envelope["dispatch"].model_copy(deep=True)
        with pytest.raises(CompactionError, match="media_budget_unavailable"):
            await fitter.get_measured_request_view(provider=wrapper, retain_contents=[], count_view=measurement.count)
        assert not provider.sent
    finally:
        wrapper.close()
    assert capability.prepared[0].released


@pytest.mark.asyncio
@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("counter", [False, True])
async def test_actual_wrapper_private_warning_only_no_budget_keeps_refs_and_exact_dispatch(wrapper_type, preflight, counter):
    wrapper, provider, capability = setup(wrapper_type, budget=False)
    if not counter:
        del wrapper.request_budget
    fitter = RequestContext(max_tokens=10000)
    original = rows()
    await fitter.set_messages(original)
    options = {"extended_thinking": True}

    async def count_view(view):
        request = ChatRequest(messages=[Message(**row) for row in view])
        decision = await wrapper.request_budget(request, request_options=options, context_estimate=1) if counter else None
        return {"dispatch": request, "budget_decision": decision, "request_options": copy.deepcopy(options)}

    try:
        callback = count_view
        if preflight:
            measurement = RequestMeasurement(count_view, wrapper)
            count, limit, kind = await measurement.measure(fitter, await fitter._view())
            safe = await wrapper.media_text_fallback_witness(measurement.cached[1]["dispatch"], request_options=options)
            assert count == fitter._estimate_tokens(safe["messages"])
            assert count > fitter._estimate_tokens([{"role": row["role"], "content": row["content"]} for row in original])
            assert limit is None and kind == "public_estimate"
            callback = measurement.count
        result = await fitter.get_measured_request_view(provider=wrapper, retain_contents=[], count_view=callback)
        assert result["measured_before"] is result["measured_after"] is None
        assert result["final_attempt"]["budget_decision"] is None
        assert result["base_view"][-1]["metadata"] == original[-1]["metadata"]
        attempt = result["final_attempt"]
        await wrapper.complete(attempt["dispatch"], request_options=attempt["request_options"])
        assert len(capability.prepared) == 1
        assert provider.sent[0][0] is capability.prepared[0].request
        assert [message.role for message in provider.sent[0][0].messages] == [row["role"] for row in original]
        assert provider.sent[0][0].messages[-1].tool_call_id == "inspect"
        assert len(provider.sent[0][0].messages[-1].content) == 4
        assert capability.prepared[0].release_calls == 1
        assert await fitter.get_messages() == original
    finally:
        wrapper.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["oversized_warnings", "hydrated", "unowned", "mandatory"])
async def test_actual_wrapper_private_fallback_never_certifies_unsafe_projection(wrapper_type, failure):
    wrapper, provider, capability = setup(wrapper_type, budget=False)
    fitter = RequestContext(max_tokens=1000)
    original = rows()
    await fitter.set_messages(original)
    if failure == "oversized_warnings":
        capability.warning_chars = 10000
    if failure == "hydrated":
        capability.hydrated = True
    if failure == "unowned":
        capability.valid = False
    options = {"mandatory_vision": True} if failure == "mandatory" else {}

    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view]),
                "budget_decision": None, "request_options": options}

    try:
        with pytest.raises(ContextLengthError if failure == "oversized_warnings" else CompactionError):
            await fitter.get_measured_request_view(provider=wrapper, retain_contents=[], count_view=count_view)
        assert not provider.sent
        assert await fitter.get_messages() == original
    finally:
        wrapper.close()


@pytest.mark.asyncio
async def test_actual_wrapper_direct_warning_only_no_counter_uses_projected_warning_estimate(wrapper_type):
    wrapper, provider, capability = setup(wrapper_type, budget=False)
    del wrapper.request_budget
    fitter = RequestContext(max_tokens=10000)
    original = rows()
    await fitter.set_messages(original)
    try:
        result = await fitter.get_messages_for_request_retaining(provider=wrapper, retain_contents=[])
        assert result[-1]["metadata"] == original[-1]["metadata"]
        assert len(capability.prepared) == 1
        assert not provider.counted and not provider.sent
        assert await fitter.get_messages() == original
    finally:
        wrapper.close()
