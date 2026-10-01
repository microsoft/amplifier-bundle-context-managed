"""Request-scoped semantic pressure, separate from canonical evidence size."""

import copy

from amplifier_core import ChatRequest

from .checkpoint import digest
from .summary import public_messages


def estimate_pressure(fitter, messages):
    # This is only the semantic trigger's fallback. The request fitter still
    # owns hard input safety, including media and provider-owned opaque state.
    return fitter._estimate_tokens(public_messages(messages))


class RequestMeasurement:
    """Reuse one preflight only when the final provider-facing view is exact.

    The loop callback adds tools, instructions and request options. Calling a
    provider directly with history alone cannot measure that complete request.
    No model completion, history fit or tool execution occurs here.
    """

    def __init__(self, count_view):
        self.callback = count_view
        self.cached = None
        self.preflight_calls = 0
        self.reused_calls = 0
        self.dispatch = None

    @staticmethod
    def key(view):
        # A fitter stamps and removes its internal sequence metadata, leaving
        # an empty metadata object where the original row had no metadata.
        rows = copy.deepcopy(view)
        for row in rows:
            if row.get("metadata") == {}:
                row.pop("metadata", None)
        return digest(rows)

    async def measure(self, fitter, view):
        public_view = fitter._strip_internal_metadata(view)
        envelope = await self.callback(public_view)
        self.preflight_calls += 1
        if not isinstance(envelope, dict) or "dispatch" not in envelope:
            raise TypeError("count_view must return {dispatch, budget_decision}")
        if self.dispatch is None and isinstance(envelope["dispatch"], ChatRequest):
            self.dispatch = envelope["dispatch"].model_copy(deep=True)
        measurement = fitter._measured_budget_decision(envelope)
        self.cached = (self.key(public_view), envelope)
        decision = envelope.get("budget_decision")
        if measurement is not None:
            count, _estimate, limit, _source = measurement
            return count, limit, "provider_count"
        if isinstance(decision, dict):
            # The validator above checks these original hard-safety fields
            # even when a provider-native measurement is unavailable.
            return decision["estimated_input_tokens"], decision["input_limit_tokens"], "provider_estimate"
        return None, None, "public_estimate"

    async def count(self, view):
        cached, self.cached = self.cached, None
        if cached is not None and cached[0] == self.key(view):
            self.reused_calls += 1
            return cached[1]
        return await self.callback(view)
