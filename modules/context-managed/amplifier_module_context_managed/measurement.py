"""Request-scoped semantic pressure, separate from canonical evidence size."""

import copy
import re

from amplifier_core import ChatRequest

from .checkpoint import digest
from .summary import public_messages


def estimate_pressure(fitter, messages):
    # This is only the semantic trigger's fallback. The request fitter still
    # owns hard input safety, including media and provider-owned opaque state.
    return fitter._estimate_tokens(public_messages(messages))


def _diagnostic_count(value):
    return type(value) is int and 0 <= value <= 2**53 - 1


def _diagnostic_hex(value, size):
    return type(value) is str and len(value) == size and re.fullmatch(r"[0-9a-f]+", value) is not None


def _measurement_description(value):
    if type(value) is not dict or not _diagnostic_hex(value.get("payload_tag"), 64):
        return None
    result = {"payload_tag": value["payload_tag"]}
    for field in ("serialized_bytes", "input_items", "input_bytes", "instructions_bytes", "tools", "tools_bytes"):
        count = value.get(field)
        if not _diagnostic_count(count):
            return None
        result[field] = count
    types = value.get("input_types")
    if type(types) is not dict:
        return None
    result["input_types"] = {}
    # Iterate only schema keys, never provider-supplied names or unknown values.
    for kind in ("message", "function_call", "function_call_output", "compaction",
                 "reasoning", "computer_call", "computer_call_output",
                 "apply_patch_call", "apply_patch_call_output", "item_reference", "other"):
        if kind in types:
            if not _diagnostic_count(types[kind]):
                return None
            result["input_types"][kind] = types[kind]
    return result


def _measurement_attribution(value):
    if type(value) is not dict or type(value.get("version")) is not int or value["version"] != 1:
        return None
    if not all(_diagnostic_hex(value.get(field), 32) for field in ("scope_id", "count_id")):
        return None
    request = _measurement_description(value.get("request"))
    compact_input = _measurement_description(value.get("compact_input"))
    if request is None or compact_input is None:
        return None
    return {"version": 1, "scope_id": value["scope_id"], "count_id": value["count_id"],
            "request": request, "compact_input": compact_input}


def _measurement_implementation(value):
    if type(value) is not dict or type(value.get("imported_bytes_verified")) is not bool:
        return None
    version = value.get("sdk_version")
    # SDK metadata is not a free-form string channel for paths or error bodies.
    if (type(version) is not str or len(version) > 29
            or re.fullmatch(r"(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){2}", version) is None):
        return None
    result = {"sdk_version": version, "imported_bytes_verified": value["imported_bytes_verified"]}
    for field in ("provider_source_sha256", "count_source_sha256"):
        if field not in value or (value[field] is not None and not _diagnostic_hex(value[field], 64)):
            return None
        result[field] = value[field]
    return result


def measurement_diagnostics(decision):
    """Detach only known content-free fields; optional metadata never gates fit."""
    if type(decision) is not dict:
        return None
    measurement = decision.get("measurement")
    if (type(measurement) is not dict or type(measurement.get("kind")) is not str
            or measurement["kind"] != "provider_count"
            or type(measurement.get("source")) is not str or not measurement["source"]
            or not _diagnostic_count(measurement.get("input_tokens"))):
        return None
    attribution = _measurement_attribution(measurement.get("attribution"))
    implementation = _measurement_implementation(measurement.get("implementation"))
    if attribution is None and implementation is None:
        return None
    return {"attribution": attribution, "implementation": implementation}


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
        self.diagnostics = None

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
        self.diagnostics = None
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
            self.diagnostics = measurement_diagnostics(decision)
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
        self.diagnostics = None
        return await self.callback(view)
