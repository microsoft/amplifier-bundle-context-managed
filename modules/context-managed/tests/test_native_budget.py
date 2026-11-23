from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_context_managed.native_budget import native_output_limit
from amplifier_module_context_managed.single_pass import PartialSource
from amplifier_module_context_managed.summary import request_fits


@pytest.mark.asyncio
async def test_model_bound_native_output_and_explicit_override():
    provider = SimpleNamespace(
        default_model="small",
        get_info=lambda: {"defaults": {"model": "small", "max_output_tokens": 8000, "context_window": 32000}},
        list_models=AsyncMock(return_value=[SimpleNamespace(id="large", max_output_tokens=128000, context_window=1000000)]),
    )
    independent = {"input_limit_independent_of_output": True}
    assert await native_output_limit(provider, "small", {}, independent) == 8000
    provider.list_models.assert_not_awaited()
    assert await native_output_limit(provider, "large", {}, independent) == 128000
    assert await native_output_limit(provider, "large", {}, {}) == 50000
    assert await native_output_limit(provider, "unlisted", {}, independent) == 4096
    assert await native_output_limit(provider, "large", {"native_compaction_max_output_tokens": 50}, {}) == 50
    with pytest.raises(ValueError):
        await native_output_limit(provider, "large", {"native_compaction_max_output_tokens": False}, {})


@pytest.mark.asyncio
async def test_partial_note_sizes_exact_envelope_while_preserving_history_and_note():
    note = "paid note\nUnprocessed transcript evidence (reference data):\ncontent"
    source = PartialSource(note, ["START" + "q" * 20000 + "END"])
    protected = Message(role="system", content="x" * 601)
    notice = Message(role="user", content="unresolved")
    template = ChatRequest(messages=[protected, Message(role="user", content=""), notice], max_output_tokens=200)
    counts = []
    def count(req, **kwargs):
        raw = 500 + sum(len(row.content) for row in req.messages)
        counts.append(raw)
        return {"estimated_input_tokens": raw + 50, "input_limit_tokens": 10000,
                "measurement": {"kind": "provider_count", "input_tokens": raw, "source": "fixture"}}
    provider = SimpleNamespace(request_budget=count)
    sized = await source.fit(template, provider, 5000, lambda: None)
    fits, receipt = await request_fits(provider, sized)
    assert fits and 9900 < receipt["input_tokens"] <= 10000 and len(counts) < 12
    assert sized.messages[0] == protected and sized.messages[2] == notice
    assert note in sized.messages[1].content and "START" in sized.messages[1].content and "END" in sized.messages[1].content
    assert template.messages[1].content == ""
