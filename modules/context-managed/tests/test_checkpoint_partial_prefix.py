"""The digest stays canonical when only a bounded leading witness fits."""
import copy
import hashlib
import json
import sys
from enum import IntEnum
from unittest.mock import Mock

import pytest

from amplifier_module_context_managed import checkpoint
from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.prefix_digest import PrefixDigest


def oracle(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def rows():
    # Payload bytes are intentionally concentrated in the front, as in retained
    # large tool results. Many small later records overflow witness accounting.
    return [{"role": "tool", "content": "long\nλ" * 10000, "metadata": {"values": [1, True]}}] + [
        {"role": "tool", "content": f"small-{i}", "metadata": {"values": [i, True]}}
        for i in range(40)]


def partial():
    result = PrefixDigest()
    result.LIMIT = 2048
    return result


def run(saved, value, encoded=None):
    return saved.digest(value, checkpoint.digest, encoded or checkpoint.canonical_bytes)


def outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:
        return (type(exc), str(exc))


def test_oversize_witness_reuses_complete_leading_rows_and_still_encodes_current_tail():
    saved, value = partial(), rows()
    encoded = Mock(side_effect=checkpoint.canonical_bytes)
    assert run(saved, value, encoded) == oracle(value)
    assert 0 < saved.charge <= saved.LIMIT
    assert saved._prefix_hash is not None
    count = len(saved.witness[1])
    assert 0 < count < len(value)
    encoded.reset_mock()
    for _ in range(3):
        assert run(saved, value, encoded) == oracle(value)
    assert encoded.call_count == 3
    assert all(call.args == (value[count:],) for call in encoded.call_args_list)
    assert value == rows()


@pytest.mark.parametrize("where", [0, 20, -1])
@pytest.mark.parametrize("changed", [False, 1.0, -0.0, "1", [1, 2], (1, 2), {"λ": "\n"}])
def test_mutable_alias_edits_and_scalar_container_types_change_exact_digest(where, changed):
    saved, value = partial(), rows()
    run(saved, value)
    value[where]["metadata"]["values"][0] = changed
    assert run(saved, value) == oracle(value)
    value[-1] = value[0]  # a repeated reference is not a cycle
    run(saved, value)
    value[0]["metadata"]["values"].append("borrowed mutation")
    assert run(saved, value) == oracle(value)


@pytest.mark.parametrize("change", ["rotate", "truncate", "grow", "replace", "empty"])
def test_order_length_and_boundary_changes_remain_canonical(change):
    saved, value = partial(), rows()
    run(saved, value)
    if change == "rotate": value.append(value.pop(0))
    elif change == "truncate": value[:] = value[:1]
    elif change == "grow": value += rows()
    elif change == "replace": value[:] = [{"other": "messages"}] * 25
    else: value.clear()
    assert run(saved, value) == oracle(value)


class CustomDict(dict):
    def items(self):
        self["visits"] = self.get("visits", 0) + 1
        return super().items()


class Number(IntEnum):
    ONE = 1


@pytest.mark.parametrize("where", [0, -1])
@pytest.mark.parametrize("bad", [CustomDict(x=1), {1: "key"}, {None: "key"},
                                   {"mixed": {"a": 1, 1: 2}}, Number.ONE, {1},
                                   float("nan"), float("inf"), "\ud800", 10 ** 5000],
                         ids=["custom", "int-key", "none-key", "mixed-keys", "enum",
                              "set", "nan", "inf", "surrogate", "huge-int"])
def test_unsupported_tail_or_front_preserves_whole_serializer_errors_and_custom_effects(where, bad):
    saved, value = partial(), rows()
    run(saved, value)
    if isinstance(bad, CustomDict):
        value[where] = CustomDict(x=1)
        expected = rows()
        expected[where] = CustomDict(x=1)
    else:
        value[where] = copy.deepcopy(bad)
        expected = copy.deepcopy(value)
    assert outcome(lambda: run(saved, value)) == outcome(lambda: oracle(expected))
    if isinstance(bad, CustomDict):
        assert value[where] == expected[where] == {"x": 1, "visits": 1}
    assert saved.witness is None and saved._prefix_hash is None and saved.charge == 0


@pytest.mark.parametrize("kind", ["root", "cross-rows", "local"])
def test_cycles_in_uncached_tail_use_original_whole_serializer(kind):
    saved, value = partial(), rows()
    run(saved, value)
    if kind == "root":
        value[-1]["cycle"] = value
    elif kind == "cross-rows":
        value[0]["other"] = value[-1]
        value[-1]["other"] = value[0]
    else:
        value[-1]["cycle"] = value[-1]
    assert outcome(lambda: run(saved, value)) == outcome(lambda: oracle(value))
    assert saved.witness is None


@pytest.mark.parametrize("depth", [200, 600, 1200])
@pytest.mark.parametrize("where", [0, -1])
def test_deep_shapes_preserve_serializer_recursion_semantics(depth, where):
    saved, value = partial(), rows()
    run(saved, value)
    nested = "leaf"
    for _ in range(depth): nested = [nested]
    value[where] = nested
    actual, expected = outcome(lambda: run(saved, value)), outcome(lambda: oracle(value))
    assert actual[0] == expected[0]
    if actual[0] == "ok": assert actual == expected


@pytest.mark.parametrize("where", [0, -1])
def test_integer_conversion_setting_is_part_of_partial_witness(where):
    saved, value = partial(), rows()
    old = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(4300)
        value[where]["large"] = 10 ** 1000
        assert run(saved, value) == oracle(value)
        assert saved._prefix_hash is not None
        sys.set_int_max_str_digits(640)
        assert outcome(lambda: run(saved, value)) == outcome(lambda: oracle(value))
        assert saved.witness is None
    finally:
        sys.set_int_max_str_digits(old)


def test_memory_accounting_and_string_reference_refresh_stay_bounded():
    saved, value = partial(), rows()
    old = value[0]["content"]
    fresh = old.encode().decode()
    run(saved, value)
    value[0]["content"] = fresh
    run(saved, value)

    def contains(node, target):
        return node is target or (type(node) is tuple and any(contains(v, target) for v in node))

    def charged(node):
        if type(node) is tuple: return sys.getsizeof(node) + sum(charged(v) for v in node)
        if type(node) is bytes: return sys.getsizeof(node)
        return 0

    assert old == fresh and old is not fresh
    assert contains(saved.witness, fresh) and not contains(saved.witness, old)
    assert saved.charge == charged(saved.witness) + sys.getsizeof(saved.witness[0]) + sys.getsizeof(saved._prefix_hash)
    assert saved.charge <= saved.LIMIT
    saved.clear()
    assert saved.witness is None and saved._prefix_hash is None and saved.charge == 0


def test_flat_scalar_rows_cannot_overrun_budget_via_unallocated_strings():
    saved, value = partial(), ["value"] * 1000
    assert run(saved, value) == oracle(value)
    assert 0 < saved.charge <= saved.LIMIT
    assert len(saved.witness[1]) < len(value)


def test_huge_first_row_structure_falls_back_without_keeping_a_useless_cache():
    saved, value = partial(), [[str(i) for i in range(5000)], {"tail": "small"}]
    assert run(saved, value) == oracle(value)
    assert saved.witness is None and saved.charge == 0


@pytest.mark.asyncio
async def test_real_checkpoint_export_restore_and_context_replacement_keep_exact_history():
    ctx = BoundaryContextManager({"durable_checkpoints": True})
    ctx.messages = rows()
    original = copy.deepcopy(ctx.messages)
    ctx.summary = (len(ctx.messages), "Derived notes")
    identity = {"provider": "fixture", "model": "fixture"}
    ctx.summary_identity = identity
    ctx._checkpoint_prefix_digest.LIMIT = 2048
    first = ctx.export_checkpoint(identity)
    assert ctx._checkpoint_prefix_digest._prefix_hash is not None
    assert ctx.export_checkpoint(identity) == first
    assert first["sourceRevision"]["sha256"] == oracle(original)
    assert ctx.messages == original
    assert ctx.restore_checkpoint(first, identity)["status"] == "restored"
    assert ctx._checkpoint_prefix_digest._prefix_hash is None
    ctx.export_checkpoint(identity)
    await ctx.set_messages(copy.deepcopy(original))
    assert ctx._checkpoint_prefix_digest._prefix_hash is None
    assert ctx.export_checkpoint(identity) == first


def test_36k_rows_with_heavy_leading_content_does_not_rehash_whole_prefix(monkeypatch):
    ctx = BoundaryContextManager({"durable_checkpoints": True})
    ctx.messages = [{"role": "tool", "content": "saved evidence " * 20000,
                     "metadata": {"values": [0, True]}}] + [
        {"role": "tool", "content": f"small-{i}", "metadata": {"values": [i, True]}}
        for i in range(35999)]
    ctx.summary = (36000, "Derived notes")
    identity = {"provider": "fixture", "model": "fixture"}
    ctx.summary_identity = identity
    canonical = checkpoint.digest
    full_hashes = []

    def counted(value):
        if type(value) is list and len(value) == 36000:
            full_hashes.append(len(value))
        return canonical(value)

    monkeypatch.setattr(checkpoint, "digest", counted)
    expected = oracle(ctx.messages)
    for _ in range(3):
        assert ctx.export_checkpoint(identity)["sourceRevision"]["sha256"] == expected
    assert len(full_hashes) <= 1
    assert ctx._checkpoint_prefix_digest.charge <= 8 * 1024 * 1024
    assert 0 < len(ctx._checkpoint_prefix_digest.witness[1]) < 36000
    ctx.messages[-1]["content"] = "current tail mutation"
    assert ctx.export_checkpoint(identity)["sourceRevision"]["sha256"] == oracle(ctx.messages)
