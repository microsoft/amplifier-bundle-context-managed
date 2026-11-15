"""Offline integration with the installed Unified wrapper; no live provider calls.

Run explicitly with the qualified public Unified wheel installed. This is not a
model-capability test: the stub's selected-model assembler owns its actual clamp
or refusal independently of potentially inherited provider defaults.
"""
import copy
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
