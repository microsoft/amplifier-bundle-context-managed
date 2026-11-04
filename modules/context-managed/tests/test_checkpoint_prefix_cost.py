"""Exact digest reuse must not mistake mutable canonical history for a cache key."""
import copy
import json
import sys
from enum import IntEnum
from unittest.mock import Mock

import pytest

from amplifier_module_context_managed import checkpoint
from amplifier_module_context_managed.boundary import BoundaryContextManager

IDENTITY = {"provider": "fixture", "model": "model-a"}


def context():
    result = BoundaryContextManager({"durable_checkpoints": True})
    result.messages = [
        {"role": "user", "content": "Keep the original objective."},
        {"role": "assistant", "content": "Large saved evidence. " * 500,
         "metadata": {"result": {"values": [1, {"ok": True}], "unknown": None}}},
    ]
    result.summary = (2, "Derived note; original evidence is retained.")
    result.summary_identity = copy.deepcopy(IDENTITY)
    return result


@pytest.mark.asyncio
async def test_unchanged_covered_prefix_is_hashed_once_across_tool_result_appends(monkeypatch):
    ctx = context()
    canonical = checkpoint.digest
    prefix_calls = []

    def measured(value):
        if isinstance(value, list):
            prefix_calls.append(len(value))
        return canonical(value)

    monkeypatch.setattr(checkpoint, "digest", measured)
    expected = canonical(ctx.messages)
    for i in range(6):
        await ctx.add_message({"role": "tool", "tool_call_id": f"call-{i}", "content": "Saved."})
        record = ctx.export_checkpoint(IDENTITY)
        assert record["sourceRevision"] == {"messages": 2, "sha256": expected}
        assert record["sha256"] == canonical({k: v for k, v in record.items() if k != "sha256"})
    assert prefix_calls == [2]


@pytest.mark.parametrize("value", [False, 0, 0.0, -0.0, True, 1.0, "1", None,
                                        [1, 2], (1, 2), {"a": [1, {"b": "λ\n"}]}])
def test_nested_in_place_edit_always_uses_exact_current_json_digest(value):
    ctx = context()
    ctx.export_checkpoint(IDENTITY)
    ctx.messages[1]["metadata"]["result"]["values"][0] = value
    record = ctx.export_checkpoint(IDENTITY)
    assert record["sourceRevision"]["sha256"] == checkpoint.digest(ctx.messages)


@pytest.mark.parametrize("mutation", ["rotate", "truncate", "boundary", "append-covered", "replace"])
def test_boundary_and_order_changes_do_not_reuse_old_prefix(mutation):
    ctx = context()
    ctx.export_checkpoint(IDENTITY)
    if mutation == "rotate":
        ctx.messages.append(ctx.messages.pop(0))
    elif mutation == "truncate":
        ctx.messages.pop()
    elif mutation == "boundary":
        ctx.summary = (1, ctx.summary[1])
    elif mutation == "append-covered":
        ctx.messages.append({"role": "user", "content": "New objective"})
        ctx.summary = (3, ctx.summary[1])
    else:
        ctx.messages[0] = {"role": "user", "content": "Changed objective"}
    end = ctx.summary[0]
    assert ctx.export_checkpoint(IDENTITY)["sourceRevision"]["sha256"] == checkpoint.digest(ctx.messages[:end])


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["same", "tail", "changed", "empty"])
async def test_set_messages_retires_witness_even_when_summary_can_survive(replacement):
    ctx = context()
    record = ctx.export_checkpoint(IDENTITY)
    rows = copy.deepcopy(ctx.messages)
    if replacement == "tail": rows.append({"role": "tool", "content": "finished"})
    if replacement == "changed": rows[0]["content"] = "Other objective"
    if replacement == "empty": rows = []
    await ctx.set_messages(rows)
    assert ctx._checkpoint_prefix_digest.witness is None
    assert ctx._checkpoint_prefix_digest.charge == 0
    assert ctx.messages == rows
    if replacement in {"same", "tail"}:
        assert ctx.export_checkpoint(IDENTITY)["sourceRevision"] == record["sourceRevision"]
    else:
        assert ctx.export_checkpoint(IDENTITY) is None


@pytest.mark.parametrize("rejection", [False, True])
def test_restore_retires_old_witness_and_validates_canonical_history(rejection):
    ctx = context()
    record = ctx.export_checkpoint(IDENTITY)
    if rejection: ctx.messages[0]["content"] = "Changed outside the context setter"
    original = copy.deepcopy(ctx.messages)
    status = ctx.restore_checkpoint(record, IDENTITY)
    assert status["status"] == ("rejected" if rejection else "restored")
    assert ctx._checkpoint_prefix_digest.witness is None
    assert ctx.messages == original
    if not rejection:
        assert ctx.export_checkpoint(IDENTITY) == record


def test_native_omission_recovery_provenance_is_unchanged_after_warm_exports_and_restore():
    ctx = context()
    ctx.summary = (2, {"role": "user", "content": "opaque checkpoint", "metadata": {
        "source": "context-managed", "ephemeral": True, "persisted": True,
        "canonical_window": [{"type": "compaction", "encrypted_content": "fixture"}]}})
    ctx.summary_omitted_indices = [0]
    ctx.recovery = {"kind": "native-recovery", "version": 1, "retentionSha256": checkpoint.digest([0])}
    record = ctx.export_checkpoint(IDENTITY)
    assert ctx.export_checkpoint(IDENTITY) == record
    assert record["configurationVersion"] == 3
    assert record["summary"]["omittedIndices"] == [0]
    assert ctx.restore_checkpoint(json.loads(json.dumps(record)), IDENTITY)["status"] == "restored"
    assert ctx._checkpoint_prefix_digest.witness is None
    assert ctx.export_checkpoint(IDENTITY) == record
    bad = copy.deepcopy(record)
    bad["summary"]["omittedIndices"] = []
    bad["sha256"] = checkpoint.digest({k: v for k, v in bad.items() if k != "sha256"})
    assert ctx.restore_checkpoint(bad, IDENTITY)["status"] == "rejected"
    assert ctx._checkpoint_prefix_digest.witness is None


@pytest.mark.parametrize("state", ["disabled", "empty", "identity"])
def test_ineligible_export_releases_retained_source_references(state):
    ctx = context()
    ctx.export_checkpoint(IDENTITY)
    identity = IDENTITY
    if state == "disabled": ctx.config["durable_checkpoints"] = False
    if state == "empty": ctx.summary = None
    if state == "identity": identity = {**IDENTITY, "model": "changed"}
    assert ctx.export_checkpoint(identity) is None
    assert ctx._checkpoint_prefix_digest.witness is None
    assert ctx._checkpoint_prefix_digest.sha256 is None
    assert ctx._checkpoint_prefix_digest.charge == 0


def cache():
    from amplifier_module_context_managed.prefix_digest import PrefixDigest
    return PrefixDigest()


def test_scalar_distinctions_and_dict_order_follow_canonical_json():
    saved = cache()
    encode = Mock(side_effect=checkpoint.digest)
    for value in [1, True, 1.0, 0.0, -0.0, False]:
        assert saved.digest({"value": value}, encode) == checkpoint.digest({"value": value})
    assert encode.call_count == 6
    saved.digest({"a": "alpha", "z": [2]}, encode)
    count = encode.call_count
    saved.digest({"z": [2], "a": "alpha"}, encode)
    assert encode.call_count == count


def test_shared_mutable_aliases_are_observed_after_each_edit():
    shared = {"content": ["before"]}
    rows = [shared, shared]
    saved = cache()
    saved.digest(rows, checkpoint.digest)
    shared["content"].append("after")
    assert saved.digest(rows, checkpoint.digest) == checkpoint.digest(rows)


class Number(IntEnum):
    ONE = 1


class CustomDict(dict):
    def items(self):
        self["visits"] = self.get("visits", 0) + 1
        return super().items()


@pytest.mark.parametrize("value", [{1: "integer-key"}, {False: "boolean-key"}, {None: "none-key"},
                                   {"enum": Number.ONE},
                                   {"set": {1}}, {"value": float("nan")}, {"value": float("inf")},
                                   {"mixed": {1: "a", "a": "b"}}, {"unicode": "\ud800"}])
def test_unsupported_values_keep_serializer_results_errors_and_side_effects(value):
    expected = copy.deepcopy(value)
    actual = copy.deepcopy(value)

    def outcome(fn):
        try: return ("ok", fn())
        except Exception as exc: return (type(exc), str(exc))

    saved = cache()
    saved.digest({"previous": "safe"}, checkpoint.digest)
    assert outcome(lambda: saved.digest(actual, checkpoint.digest)) == outcome(lambda: checkpoint.digest(expected))
    assert actual == expected or "nan" in str(value)
    assert saved.witness is None


def test_custom_container_is_serialized_once_without_witness_side_effects():
    expected, actual = {"dict": CustomDict(x=1)}, {"dict": CustomDict(x=1)}
    saved = cache()
    for i in range(1, 4):
        assert saved.digest(actual, checkpoint.digest) == checkpoint.digest(expected)
        assert actual == expected == {"dict": {"x": 1, "visits": i}}
        assert saved.witness is None


def test_large_integer_conversion_failure_leaves_no_digest_reuse():
    saved = cache()
    saved.digest([1], checkpoint.digest)
    previous_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(4300)
        value = {"large": 10 ** 5000}
        with pytest.raises(ValueError) as expected:
            checkpoint.digest(value)
        with pytest.raises(ValueError) as actual:
            saved.digest(value, checkpoint.digest)
        assert str(actual.value) == str(expected.value)
        assert saved.witness is None
    finally:
        sys.set_int_max_str_digits(previous_limit)


@pytest.mark.parametrize("depth", [400, 1500])
def test_deeply_nested_values_use_original_serializer_if_witness_recurses_too_far(depth):
    value = "leaf"
    for _ in range(depth): value = [value]
    saved = cache()
    try:
        expected = checkpoint.digest(value)
    except RecursionError:
        with pytest.raises(RecursionError):
            saved.digest(value, checkpoint.digest)
    else:
        assert saved.digest(value, checkpoint.digest) == expected


def test_cycles_preserve_original_failure_and_retire_previous_witness():
    saved = cache()
    saved.digest([1], checkpoint.digest)
    cyclic = []; cyclic.append(cyclic)
    with pytest.raises(ValueError, match="Circular reference detected"):
        saved.digest(cyclic, checkpoint.digest)
    assert saved.witness is None


def test_witness_budget_is_only_an_optimization_budget():
    saved = cache()
    saved.LIMIT = 1024
    saved.digest([1], checkpoint.digest)
    original = [{"value": i} for i in range(500)]
    encode = Mock(side_effect=checkpoint.digest)
    for _ in range(2):
        assert saved.digest(original, encode) == checkpoint.digest(original)
    assert encode.call_count == 2
    assert saved.witness is None and saved.charge == 0


def test_retained_structure_is_charged_and_equal_hits_refresh_string_ownership():
    saved = cache()
    old = "original " * 1000
    fresh = old.encode().decode()
    assert old is not fresh and old == fresh
    saved.digest({"value": [old, 3, 3.0]}, checkpoint.digest)
    first_charge = saved.charge

    def contains(value, target):
        return value is target or (type(value) is tuple and any(contains(v, target) for v in value))

    assert contains(saved.witness, old)
    saved.digest({"value": [fresh, 3, 3.0]}, checkpoint.digest)
    assert not contains(saved.witness, old) and contains(saved.witness, fresh)
    assert saved.charge == first_charge

    def charge(value):
        if type(value) is tuple: return sys.getsizeof(value) + sum(charge(v) for v in value)
        if type(value) is bytes: return sys.getsizeof(value)
        return 0

    assert saved.charge == charge(saved.witness)
    saved.clear()
    assert saved.witness is None and saved.sha256 is None and saved.charge == 0
