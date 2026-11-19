"""Foreground fitting policy must not invalidate prior summary provenance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.checkpoint import digest
from amplifier_module_context_managed.errors import CompactionError


SOURCE = {"provider": "fixture-instance", "model": "model-a"}
TARGET = {**SOURCE, "model": "model-b"}
CONFIG = {"durable_checkpoints": True, "max_tokens": 10000,
          "summary_target_tokens": 1500, "summary_max_output_tokens": 8192}
# Frozen from accepted 31fc364's checkpoint_configuration() under CONFIG.
# Do not obtain this oracle from the production helper being tested.
LEGACY_CONFIGURATION = "a67783e667b2b71385969b0723dda6182357b86a6b05a3f4cbbc5c5e5ccbf891"
ROWS = [
    {"role": "user", "content": "Preserve originals; keep pending work and UNKNOWN effects."},
    {"role": "assistant", "content": "Verified evidence remains in canonical history."},
    {"role": "user", "content": "New correction: do not replay tools."},
    {"role": "assistant", "content": "Further work is unfinished."},
    {"role": "user", "content": "Continue under the selected provider contract."},
]


def legacy_completed(native):
    summary = {"throughMessage": 2, "text": "Preserve originals and unfinished work."}
    if native:
        summary = {"throughMessage": 2, "kind": "native", "version": 2,
                   "omittedIndices": [], "message": {
                       "role": "user", "content": "Source native checkpoint",
                       "metadata": {"source": "context-managed", "ephemeral": True,
                                    "persisted": True,
                                    "fixture:opaque": {"model": SOURCE["model"],
                                                       "state": "opaque-source-only"}}}}
    record = {
        "format": "context-managed-boundary-v1", "identity": copy.deepcopy(SOURCE),
        "configuration": LEGACY_CONFIGURATION, "configurationVersion": 3 if native else 2,
        "sourceRevision": {"messages": 2, "sha256": digest(ROWS[:2])},
        "summary": summary, "progress": None,
        "evidenceRefs": [{"kind": "fixture-originals", "id": "canonical"}], "recovery": None,
    }
    record["sha256"] = digest(record)
    return record


def manager(config=None, rows=None):
    ctx = BoundaryContextManager({**CONFIG, **(config or {})})
    ctx.messages = copy.deepcopy(ROWS if rows is None else rows)
    return ctx


@pytest.mark.parametrize("native", [False, True], ids=["portable-v2", "native-v3"])
@pytest.mark.parametrize("foreground", [None, False, True], ids=["absent", "disabled", "enabled"])
def test_legacy_same_model_checkpoint_survives_foreground_policy(native, foreground):
    config = {"compaction_schedule": "after_turn"}
    if foreground is not None:
        config["foreground_continuation"] = foreground
    ctx, record = manager(config), legacy_completed(native)
    before = copy.deepcopy(record)
    assert ctx.restore_checkpoint(record, SOURCE)["status"] == "restored"
    assert ctx.export_checkpoint(SOURCE) == before
    assert ctx.messages == ROWS
    assert record == before


@pytest.mark.parametrize("native", [False, True], ids=["portable-v2", "native-v3"])
def test_legacy_cross_model_classification_does_not_install_or_relabel_source(native):
    ctx = manager({"foreground_continuation": True, "compaction_schedule": "after_turn"})
    record, before = legacy_completed(native), copy.deepcopy(ctx.messages)
    assert ctx.restore_checkpoint(record, TARGET)["status"] == "rejected"
    assert ctx.summary is None
    status = ctx.classify_checkpoint_transition(record, TARGET)
    assert status["status"] == "transition_required"
    assert status["sha256"] == record["sha256"]
    assert status["sourceIdentity"] == SOURCE and status["targetIdentity"] == TARGET
    assert status["validation"]["sourceProviderEnvelope"] == "unverified"
    assert ctx.summary is None and ctx.messages == before
    assert ctx.classify_checkpoint_transition(record, SOURCE)["status"] == "compatible"


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("change", ["summary-output", "summary-target", "source", "record-digest"])
def test_foreground_policy_never_launders_real_checkpoint_incompatibility(native, change):
    ctx = manager({"foreground_continuation": True, "compaction_schedule": "after_turn"})
    record = legacy_completed(native)
    if change == "summary-output":
        ctx.config["summary_max_output_tokens"] = 4096
    elif change == "summary-target":
        ctx.config["summary_target_tokens"] = 1000
    elif change == "source":
        ctx.messages[0]["content"] = "Different canonical objective"
    else:
        record["summary"]["throughMessage"] = 1  # Deliberately keep the stale digest.
    assert ctx.restore_checkpoint(record, SOURCE)["status"] == "rejected"
    assert ctx.classify_checkpoint_transition(record, TARGET)["status"] == "rejected"
    assert ctx.summary is None


@pytest.mark.parametrize("version", [4, 5])
def test_paid_partial_is_preserved_but_cannot_be_exported_under_foreign_identity(version):
    packet = json.loads(Path(__file__).with_name("summary_work_pause_baseline.json").read_text())
    fixture = packet["fixtures"][f"v{version}_completed_with_partial"]
    record = copy.deepcopy(fixture["checkpoint"])
    identity = record["identity"]
    ctx = BoundaryContextManager({**fixture["config"], "foreground_continuation": True,
                                  "compaction_schedule": "after_turn"})
    ctx.messages = copy.deepcopy(fixture["messages"])
    assert ctx.restore_checkpoint(record, identity)["status"] == "resuming"
    before = copy.deepcopy(ctx._restored_progress)
    assert ctx.export_checkpoint(identity) == record
    assert ctx.restore_checkpoint(record, {**identity, "model": "other-model"})["status"] == "rejected"
    assert ctx._restored_progress == before  # Rejected import is transactional.
    assert ctx.classify_checkpoint_transition(
        record, {**identity, "model": "other-model"})["status"] == "transition_required"
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        ctx.export_checkpoint({**identity, "model": "other-model"})
    assert ctx._restored_progress == before
    assert ctx.export_checkpoint(identity) == record


@pytest.mark.parametrize("version", [4, 5])
@pytest.mark.parametrize("change", ["summary-policy", "canonical-prefix"])
def test_foreground_policy_keeps_paid_partial_source_and_generation_rejection(version, change):
    packet = json.loads(Path(__file__).with_name("summary_work_pause_baseline.json").read_text())
    fixture = packet["fixtures"][f"v{version}_completed_with_partial"]
    ctx = BoundaryContextManager({**fixture["config"], "foreground_continuation": True})
    ctx.messages = copy.deepcopy(fixture["messages"])
    if change == "summary-policy":
        ctx.config["summary_reasoning_effort"] = "high"
    else:
        ctx.messages[0]["content"] += " changed source"
    record = fixture["checkpoint"]
    assert ctx.restore_checkpoint(record, record["identity"])["status"] == "rejected"
    assert ctx.classify_checkpoint_transition(
        record, {**record["identity"], "model": "other-model"})["status"] == "rejected"
    assert ctx._restored_progress is None


@pytest.mark.asyncio
async def test_foreground_compatibility_does_not_rebind_inflight_summary_policy_change():
    ctx = manager({"summarize_trigger": .01, "summary_max_source_chars": 3000,
                   "foreground_continuation": True})
    ctx.messages[1]["content"] *= 100
    ctx.checkpoint_identity = lambda: copy.deepcopy(SOURCE)
    async def complete(request, **kwargs):
        ctx.config["summary_max_output_tokens"] = 4096
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Retained paid result.")])
    provider = SimpleNamespace(name="fixture-instance", default_model="model-a",
                               complete=AsyncMock(side_effect=complete))
    with pytest.raises(RuntimeError, match="contract changed"):
        await ctx._prepare(provider, None, [])
    assert ctx.summary is None
    provider.complete.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_legacy_restored_checkpoint_reaches_foreground_count_without_generation(native):
    ctx = manager({"foreground_continuation": True, "compaction_schedule": "after_turn"})
    ctx.checkpoint_identity = lambda: copy.deepcopy(SOURCE)
    record = legacy_completed(native)
    assert ctx.restore_checkpoint(record, SOURCE)["status"] == "restored"
    provider = SimpleNamespace(name=SOURCE["provider"], default_model=SOURCE["model"],
        supports_native_compaction=lambda: native,
        validate_compacted_context=lambda row: (
            row.get("metadata", {}).get("fixture:opaque", {}).get("model") == SOURCE["model"]),
        complete=AsyncMock(), compact_context=AsyncMock())
    async def count_view(view):
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view], model=SOURCE["model"]),
                "budget_decision": {"estimated_input_tokens": 1000, "input_limit_tokens": 10000,
                    "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 1000}}}
    result = await ctx.get_measured_request_view(
        provider=provider, retain_contents=[], count_view=count_view)
    assert result["final_attempt"]["dispatch"].model == SOURCE["model"]
    assert ctx.export_checkpoint(SOURCE) == record
    provider.complete.assert_not_awaited()
    provider.compact_context.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_legacy_cross_model_checkpoint_reaches_existing_guarded_transition(native):
    from .test_model_transition import transition_fixture
    state = transition_fixture(native)
    source_record = copy.deepcopy(state.old)
    state.ctx.config.update(foreground_continuation=True, compaction_schedule="after_turn")
    result = await state.ctx.recover_request(**state.kwargs)
    assert result["outcome"] == "transitioned"
    assert state.ctx.summary_identity == state.target
    assert state.old == source_record
    assert state.installed[-1][0] == state.ctx.export_checkpoint(state.target)


def boolean_fixture(kind, flag):
    packet = json.loads(Path(__file__).with_name("foreground_boolean_baseline.json").read_text())
    fixture = packet["fixtures"][f"{kind}-{str(flag).lower()}"]
    return packet, copy.deepcopy(fixture)


@pytest.mark.parametrize("kind", ["portable", "native", "partial-v4", "partial-v5",
                                 "completed-partial-v5"])
@pytest.mark.parametrize("flag", [False, True])
def test_real_boolean_writer_roundtrip_and_cross_identity_classification(kind, flag):
    packet, fixture = boolean_fixture(kind, flag)
    ctx = BoundaryContextManager({**fixture["config"], "foreground_continuation": not flag})
    ctx.messages = copy.deepcopy(packet["messages"])
    record, identity = fixture["checkpoint"], packet["identity"]
    untouched = copy.deepcopy(record)
    expected = "resuming" if record["progress"] else "restored"
    assert ctx.restore_checkpoint(record, identity)["status"] == expected
    assert ctx.export_checkpoint(identity) == untouched  # Also keeps original fingerprint/digest.
    other = {**identity, "model": "other"}
    assert ctx.restore_checkpoint(record, other)["status"] == "rejected"
    assert ctx.export_checkpoint(identity) == untouched
    assert ctx.classify_checkpoint_transition(record, other)["status"] == "transition_required"
    assert ctx.export_checkpoint(identity) == untouched
    if record["progress"]:
        with pytest.raises(CompactionError, match="summary_progress_incompatible"):
            ctx.export_checkpoint(other)
    assert record == untouched


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["partial-v4", "partial-v5", "completed-partial-v5"])
@pytest.mark.parametrize("flag", [False, True])
async def test_real_boolean_paid_prefix_resumes_without_repurchase_or_rebinding(kind, flag):
    packet, fixture = boolean_fixture(kind, flag)
    config = {**fixture["config"], "foreground_continuation": not flag}
    ctx = BoundaryContextManager(config)
    ctx.messages = copy.deepcopy(packet["messages"])
    identity, record = packet["identity"], fixture["checkpoint"]
    ctx.checkpoint_identity = lambda: copy.deepcopy(identity)
    assert ctx.restore_checkpoint(record, identity)["status"] == "resuming"
    old = copy.deepcopy(ctx._restored_progress)
    provider = SimpleNamespace(name="fixture", default_model="fixture-model",
        complete=AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Finished note preserving paid prefix.")],
            usage={"input_tokens": 10, "output_tokens": 5})))
    with pytest.raises(CompactionError, match="summary_work_paused"):
        await ctx._prepare(provider, None, [])
    assert old["text"] in str(provider.complete.call_args_list[0])
    assert ctx._restored_progress is None
    assert ctx.messages == packet["messages"]
    exported = ctx.export_checkpoint(identity)
    assert exported["configuration"] == record["configuration"]
    # Only the first pending piece ran, against the original bound prefix.
    provider.complete.assert_awaited_once()
    assert exported["progress"]["fingerprint"] == old["fingerprint"]
    assert exported["progress"]["source_revision"] == old["source_revision"]
    assert exported["progress"]["completed"] == old["completed"] + 1
    assert exported["progress"]["calls"] == old["calls"] + 1
    remaining = exported["progress"]["segments"]
    assert remaining == old["segments"][1:] or (
        remaining[1:] == old["segments"][1:]
        and remaining[0][0] == old["segments"][0][0]
        and old["segments"][0][1] < remaining[0][1] < remaining[0][2] == old["segments"][0][2])


@pytest.mark.parametrize("change", ["summary-target", "summary-output", "summary-effort",
    "summary-prompt", "sibling-key", "prefix", "digest", "partial-provider"])
def test_boolean_digest_proof_still_rejects_other_changes_transactionally(change):
    packet, fixture = boolean_fixture("partial-v5", True)
    config = {**fixture["config"], "foreground_continuation": False}
    record = fixture["checkpoint"]
    if change == "summary-target": config["summary_target_tokens"] = 1100
    if change == "summary-output": config["summary_max_output_tokens"] = 4096
    if change == "summary-effort": config["summary_reasoning_effort"] = "high"
    if change == "summary-prompt": config["summary_prompt"] = "Changed instructions"
    if change == "sibling-key": config["compaction_schedule_new_meaning"] = True
    ctx = BoundaryContextManager(config)
    ctx.messages = copy.deepcopy(packet["messages"])
    if change == "prefix": ctx.messages[0]["content"] += " changed"
    if change == "digest": record["progress"]["text"] += " corrupt"
    if change == "partial-provider": record["progress"]["identity"]["model"] = "other"
    if change == "partial-provider":
        record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    assert ctx.restore_checkpoint(record, packet["identity"])["status"] == "rejected"
    assert ctx.classify_checkpoint_transition(
        record, {**packet["identity"], "model": "other"})["status"] == "rejected"
    assert ctx.summary is None and ctx._restored_progress is None


@pytest.mark.parametrize("bad_flag", [0, 1, "true", None, {}, []])
def test_nonboolean_legacy_flag_hash_does_not_match_boolean_proof(bad_flag):
    from amplifier_module_context_managed.boundary import SUMMARY_PROMPT
    packet, fixture = boolean_fixture("portable", True)
    # Historical writer projection with a non-boolean value: bool coercion would
    # wrongly accept these distinct canonical JSON hashes.
    projected = {k: v for k, v in fixture["config"].items() if k not in {
        "durable_checkpoints", "summarize_trigger", "summary_retry_delay",
        "summary_max_calls", "token_meter", "archive_recovery", "compaction_schedule"}}
    projected["foreground_continuation"] = bad_flag
    record = fixture["checkpoint"]
    record["configuration"] = digest({"config": projected, "prompt": SUMMARY_PROMPT})
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    ctx = BoundaryContextManager(fixture["config"])
    ctx.messages = copy.deepcopy(packet["messages"])
    assert ctx.restore_checkpoint(record, packet["identity"])["status"] == "rejected"


def test_boolean_proof_never_applies_to_v1_full_configuration_contract():
    packet, fixture = boolean_fixture("portable", True)
    record = fixture["checkpoint"]
    record["configurationVersion"] = 1  # Re-signed, but not a v1 full-config digest.
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    ctx = BoundaryContextManager(fixture["config"])
    ctx.messages = copy.deepcopy(packet["messages"])
    assert ctx.restore_checkpoint(record, packet["identity"])["status"] == "rejected"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["portable", "native", "partial-v5"])
async def test_legacy_hash_provenance_moves_only_with_its_derived_unit(kind):
    packet, fixture = boolean_fixture(kind, True)
    ctx = BoundaryContextManager({**fixture["config"], "foreground_continuation": False})
    ctx.messages = copy.deepcopy(packet["messages"])
    assert ctx.restore_checkpoint(fixture["checkpoint"], packet["identity"])["status"] in {
        "restored", "resuming"}
    original = copy.deepcopy(ctx.export_checkpoint(packet["identity"]))
    corrupt = copy.deepcopy(original); corrupt["configuration"] = "wrong"
    corrupt["sha256"] = digest({k: v for k, v in corrupt.items() if k != "sha256"})
    assert ctx.restore_checkpoint(corrupt, packet["identity"])["status"] == "rejected"
    assert ctx.export_checkpoint(packet["identity"]) == original
    await ctx.clear()
    assert ctx._checkpoint_legacy_foreground is None
    assert ctx.checkpoint_configuration() == ctx._checkpoint_configuration()


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_malformed_current_flag_cannot_admit_historical_boolean_digest(flag):
    packet, fixture = boolean_fixture("portable", True)
    ctx = BoundaryContextManager({**fixture["config"], "foreground_continuation": flag})
    ctx.messages = copy.deepcopy(packet["messages"])
    assert ctx.restore_checkpoint(fixture["checkpoint"], packet["identity"])["status"] == "rejected"


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_historical_boolean_source_transition_installs_fresh_canonical_target(native):
    from .test_model_transition import transition_fixture
    state = transition_fixture(native)
    state.ctx._checkpoint_legacy_foreground = True  # Exactly proof-selected historical hash.
    state.old = state.ctx.export_checkpoint(state.old["identity"])
    state.kwargs["source_checkpoint"] = state.old
    state.ctx.config["foreground_continuation"] = False
    result = await state.ctx.recover_request(**state.kwargs)
    assert result["outcome"] == "transitioned"
    assert state.ctx._checkpoint_legacy_foreground is None
    assert state.installed[-1][0]["configuration"] == state.ctx._checkpoint_configuration()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,request_sha", [
    ("partial-v4", "14223048915900e4badad9264f6d248a0022b8b699c97f65bdd2f04917dd2de2"),
    ("partial-v5", "b4951f13b7778a91d34c95755d04db974773315dd921ac52cb090bde49a8aa93")])
@pytest.mark.parametrize("flag", [False, True])
async def test_flag_is_inert_in_actual_summary_request_against_frozen_writer(kind, request_sha, flag):
    packet, fixture = boolean_fixture(kind, flag)
    ctx = BoundaryContextManager(fixture["config"])
    ctx.checkpoint_identity = lambda: copy.deepcopy(packet["identity"])
    ctx.messages = copy.deepcopy(packet["messages"])
    requests = []
    async def complete(request, **kwargs):
        requests.append({"request": request.model_dump(mode="json"), "kwargs": kwargs})
        return SimpleNamespace(content=[SimpleNamespace(type="text",
            text="PAID piece; originals retained; UNKNOWN stays unresolved.")],
            usage={"input_tokens": 100, "output_tokens": 10})
    provider = SimpleNamespace(name="fixture", default_model="fixture-model", complete=complete)
    with pytest.raises(CompactionError, match="summary_work_paused"):
        await ctx._prepare(provider, None, [])
    assert digest(requests) == request_sha  # Exact a446 writer request/options, False == True.


@pytest.mark.asyncio
async def test_replaced_completed_unit_cannot_keep_legacy_binding():
    packet, fixture = boolean_fixture("portable", True)
    ctx = BoundaryContextManager(fixture["config"])
    ctx.messages = copy.deepcopy(packet["messages"])
    assert ctx.restore_checkpoint(fixture["checkpoint"], packet["identity"])["status"] == "restored"
    replacement = copy.deepcopy(packet["messages"])
    replacement[0]["content"] += " replaced prefix"
    await ctx.set_messages(replacement)
    assert ctx.summary is None and ctx._checkpoint_legacy_foreground is None
    assert ctx.checkpoint_configuration() == ctx._checkpoint_configuration()


def test_fresh_restore_does_not_borrow_prior_unit_proof():
    packet, fixture = boolean_fixture("portable", True)
    ctx = BoundaryContextManager(fixture["config"])
    ctx.messages = copy.deepcopy(packet["messages"])
    assert ctx.restore_checkpoint(fixture["checkpoint"], packet["identity"])["status"] == "restored"
    canonical = copy.deepcopy(fixture["checkpoint"])
    canonical["configuration"] = ctx._checkpoint_configuration()
    canonical["sha256"] = digest({k: v for k, v in canonical.items() if k != "sha256"})
    assert ctx.restore_checkpoint(canonical, packet["identity"])["status"] == "restored"
    assert ctx._checkpoint_legacy_foreground is None
    assert ctx.export_checkpoint(packet["identity"]) == canonical


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["identity", "source", "generation", "utility"])
async def test_boolean_paid_binding_mismatch_refuses_before_any_new_call_or_write(change):
    packet, fixture = boolean_fixture("partial-v5", True)
    ctx = BoundaryContextManager({**fixture["config"], "foreground_continuation": False})
    ctx.messages = copy.deepcopy(packet["messages"])
    identity = copy.deepcopy(packet["identity"])
    ctx.checkpoint_identity = lambda: copy.deepcopy(identity)
    assert ctx.restore_checkpoint(fixture["checkpoint"], identity)["status"] == "resuming"
    paid = copy.deepcopy(ctx._restored_progress)
    if change == "identity": identity["model"] = "other"
    if change == "source": ctx.messages[0]["content"] += " changed"
    if change == "generation": ctx.config["summary_reasoning_effort"] = "high"
    provider = SimpleNamespace(name="other-utility" if change == "utility" else "fixture",
        default_model="fixture-model", complete=AsyncMock(), compact_context=AsyncMock())
    ctx.persist_checkpoint = AsyncMock()
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        await ctx._prepare(provider, None, [])
    provider.complete.assert_not_awaited()
    provider.compact_context.assert_not_awaited()
    ctx.persist_checkpoint.assert_not_awaited()
    assert ctx._restored_progress == paid and ctx._summary_progress is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["portable", "native"])
@pytest.mark.parametrize("flag", [False, True])
async def test_retention_discard_without_paid_work_starts_canonical_fingerprint(kind, flag):
    packet, fixture = boolean_fixture(kind, flag)
    ctx = BoundaryContextManager(fixture["config"])
    ctx.messages = copy.deepcopy(packet["messages"])
    identity = packet["identity"]
    ctx.checkpoint_identity = lambda: copy.deepcopy(identity)
    record = copy.deepcopy(fixture["checkpoint"])
    assert ctx.restore_checkpoint(record, identity)["status"] == "restored"
    await ctx.get_messages_for_request_retaining(
        provider=None, retain_contents=[packet["messages"][0]["content"]])
    assert ctx.summary is None and ctx._checkpoint_legacy_foreground is None
    assert ctx.checkpoint_configuration() == ctx._checkpoint_configuration()
    fresh = BoundaryContextManager(fixture["config"])
    provider = SimpleNamespace(name="fixture", default_model="fixture-model")
    assert ctx._summary_fingerprint(ctx.messages, identity, provider) == fresh._summary_fingerprint(
        ctx.messages, identity, provider)
    assert record == fixture["checkpoint"]  # No host-file rewrite/rollback claim.


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [False, True])
async def test_retention_discard_cannot_rebind_completed_paid_prefix(flag):
    packet, fixture = boolean_fixture("completed-partial-v5", flag)
    ctx = BoundaryContextManager(fixture["config"])
    ctx.messages = copy.deepcopy(packet["messages"])
    identity = packet["identity"]
    ctx.checkpoint_identity = lambda: copy.deepcopy(identity)
    assert ctx.restore_checkpoint(fixture["checkpoint"], identity)["status"] == "resuming"
    before = ctx.export_checkpoint(identity)
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        await ctx.get_messages_for_request_retaining(
            provider=None, retain_contents=[packet["messages"][0]["content"]])
    assert ctx.export_checkpoint(identity) == before


@pytest.mark.parametrize("kind", ["portable", "native", "partial-v4", "partial-v5"])
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("caller", ["cold", "warm-source", "opposite", "canonical"])
def test_classification_configuration_is_verified_source_not_caller_state(kind, flag, caller):
    packet, fixture = boolean_fixture(kind, flag)
    ctx = BoundaryContextManager(fixture["config"])
    ctx.messages = copy.deepcopy(packet["messages"])
    identity, record = packet["identity"], fixture["checkpoint"]
    if caller != "cold":
        previous = copy.deepcopy(record)
        if caller == "opposite":
            previous = boolean_fixture(kind, not flag)[1]["checkpoint"]
        elif caller == "canonical":
            previous = legacy_completed(False)
            previous["identity"] = copy.deepcopy(identity)
            previous["sourceRevision"] = {"messages": 2, "sha256": digest(ctx.messages[:2])}
            previous["configuration"] = ctx._checkpoint_configuration()
            previous["sha256"] = digest({k: v for k, v in previous.items() if k != "sha256"})
        assert ctx.restore_checkpoint(previous, identity)["status"] in {"restored", "resuming"}
    state = copy.deepcopy(ctx.__dict__)
    result = ctx.classify_checkpoint_transition(record, {**identity, "model": "target"})
    assert result["status"] == "transition_required"
    assert result["configuration"] == record["configuration"]
    assert result["sha256"] == record["sha256"]
    # Classifier may not clear caller provenance/paid state while verifying a foreign unit.
    for field in ("summary", "_summary_progress", "_restored_progress", "_checkpoint_legacy_foreground"):
        assert getattr(ctx, field) == state[field]


@pytest.mark.parametrize("bad_flag", [0, 1, "true", None, {}, []])
def test_canonical_digest_restore_rejects_malformed_flag_not_just_legacy_fallback(bad_flag):
    packet, fixture = boolean_fixture("portable", True)
    ctx = BoundaryContextManager({**fixture["config"], "foreground_continuation": bad_flag})
    ctx.messages = copy.deepcopy(packet["messages"])
    record = copy.deepcopy(fixture["checkpoint"])
    record["configuration"] = ctx._checkpoint_configuration()
    record["sha256"] = digest({k: v for k, v in record.items() if k != "sha256"})
    result = ctx.restore_checkpoint(record, packet["identity"])
    assert result["status"] == "rejected"
    assert "boolean" in result["reason"]
    assert ctx.classify_checkpoint_transition(
        record, {**packet["identity"], "model": "target"})["status"] == "rejected"


def same_model_recovery_case(kind, flag):
    packet, fixture = boolean_fixture(kind, flag)
    ctx = BoundaryContextManager(fixture["config"])
    ctx.messages = copy.deepcopy(packet["messages"]) + [
        {"role": "assistant", "content": "New verified evidence since source note."},
        {"role": "user", "content": "Continue; no completed effects may be repeated."}]
    identity = copy.deepcopy(packet["identity"])
    ctx.checkpoint_identity = lambda: copy.deepcopy(identity)
    assert ctx.restore_checkpoint(fixture["checkpoint"], identity)["status"] in {"restored", "resuming"}
    before = ctx.export_checkpoint(identity)
    def unchanged():
        assert ctx.export_checkpoint(identity) == before
    async def budget(request, **kwargs):
        return {"estimated_input_tokens": 100, "input_limit_tokens": 10000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 100}}
    async def compact(request):
        unchanged()
        return {"kind": "native", "message": {"role": "user", "content": "New derived note.",
            "metadata": {"source": "context-managed", "ephemeral": True, "persisted": True,
                         "fixture:opaque": {"model": identity["model"], "state": "replacement"}}}}
    model = SimpleNamespace(name=identity["provider"], default_model=identity["model"],
        supports_native_compaction=lambda: True, validate_compacted_context=lambda row: (
            row.get("metadata", {}).get("fixture:opaque", {}).get("model") == identity["model"]),
        request_budget=AsyncMock(side_effect=budget), compact_context=AsyncMock(side_effect=compact),
        complete=AsyncMock())
    async def count(view):
        unchanged()
        replaced = "replacement" in str(view)
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view], model=identity["model"]),
            "budget_decision": {"estimated_input_tokens": 500 if replaced else 7000,
                "input_limit_tokens": 10000, "measurement": {"kind": "provider_count",
                    "source": "fixture", "input_tokens": 500 if replaced else 7000}}}
    async def save(*args):
        unchanged()
    async def install(record, report):
        unchanged()
        return record["sha256"]
    ctx.preserve_evidence = AsyncMock(return_value=copy.deepcopy(ctx.evidence_refs))
    return SimpleNamespace(ctx=ctx, model=model, before=before, identity=identity,
        kwargs={"provider": model, "identity": identity, "count_view": AsyncMock(side_effect=count),
            "guard": lambda: None, "save_progress": AsyncMock(side_effect=save),
            "install": AsyncMock(side_effect=install), "target_bytes": 10000})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["partial-v4", "partial-v5", "completed-partial-v5"])
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("location", ["restored", "in-memory"])
async def test_same_model_native_recovery_cannot_drop_real_paid_partial(kind, flag, location):
    state = same_model_recovery_case(kind, flag)
    if location == "in-memory":
        utility = SimpleNamespace(name="fixture", default_model="fixture-model",
            complete=AsyncMock(return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Next exact pending prefix piece.")],
                usage={"input_tokens": 10, "output_tokens": 5})))
        with pytest.raises(CompactionError, match="summary_work_paused"):
            await state.ctx._prepare(utility, None, [])
        utility.complete.assert_awaited_once()
        assert state.ctx._summary_progress["completed"] == 2
        assert state.ctx._restored_progress is None
        state.before = state.ctx.export_checkpoint(state.identity)
        state.ctx.preserve_evidence.assert_awaited_once()  # Ordinary prefix resume above.
        state.ctx.preserve_evidence.reset_mock()
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        await state.ctx.recover_request(**state.kwargs)
    assert state.ctx.export_checkpoint(state.identity) == state.before
    state.model.compact_context.assert_not_awaited()
    state.model.complete.assert_not_awaited()
    state.ctx.preserve_evidence.assert_not_awaited()
    for key in ("count_view", "save_progress", "install"):
        state.kwargs[key].assert_not_awaited()
    assert not state.ctx.lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("unpaid_draft", [False, True])
async def test_completed_native_progress_is_not_permission_to_drop_active_paid_unit(flag, unpaid_draft):
    state = same_model_recovery_case("partial-v5", flag)
    if unpaid_draft:
        state.ctx._summary_progress = {"completed": 0}  # Must not mask restored paid work.
    before = copy.deepcopy(state.ctx._restored_progress)
    native = boolean_fixture("native", flag)[1]["checkpoint"]
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        await state.ctx.recover_request(**state.kwargs, progress=native)
    assert state.ctx._restored_progress == before
    state.model.compact_context.assert_not_awaited()
    state.ctx.preserve_evidence.assert_not_awaited()
    for key in ("count_view", "save_progress", "install"):
        state.kwargs[key].assert_not_awaited()
    assert not state.ctx.lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("phase", ["count", "save", "install", "success"])
async def test_same_model_historical_native_recovery_retains_source_until_exact_replace(flag, phase):
    state = same_model_recovery_case("native", flag)
    if phase != "success":
        key = {"count": "count_view", "save": "save_progress", "install": "install"}[phase]
        state.kwargs[key].side_effect = OSError("Synthetic storage/count boundary")
        with pytest.raises(OSError):
            await state.ctx.recover_request(**state.kwargs)
        assert state.ctx.export_checkpoint(state.identity) == state.before
    else:
        result = await state.ctx.recover_request(**state.kwargs)
        assert result["outcome"] == "recovered"
        installed = state.kwargs["install"].call_args.args[0]
        assert state.ctx.export_checkpoint(state.identity) == installed
        assert installed["sha256"] != state.before["sha256"]
        assert installed["configuration"] == state.before["configuration"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["portable", "native"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("phase", ["count", "install", "success"])
async def test_real_frozen_source_transition_keeps_exact_readback_and_rollback_scope(kind, warm, phase):
    packet, fixture = boolean_fixture(kind, True)
    ctx = BoundaryContextManager(fixture["config"])
    ctx.messages = copy.deepcopy(packet["messages"])
    source_identity, record = packet["identity"], copy.deepcopy(fixture["checkpoint"])
    source_bytes = copy.deepcopy(record)
    if warm:
        assert ctx.restore_checkpoint(record, source_identity)["status"] == "restored"
    selected = {**source_identity, "model": "model-b"}
    ctx.checkpoint_identity = lambda: copy.deepcopy(selected)
    active_before = copy.deepcopy(ctx._checkpoint_state())
    def unchanged():
        assert ctx._checkpoint_state() == active_before
        assert record == source_bytes
        if warm:
            assert ctx.export_checkpoint(source_identity) == source_bytes
    async def budget(request, **kwargs):
        assert "source-only" not in str(request)
        return {"estimated_input_tokens": 100, "input_limit_tokens": 10000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": 100}}
    async def count(view):
        unchanged()
        assert "source-only" not in str(view)
        if phase == "count":
            raise OSError("Synthetic target count error")
        tokens = 12000 if any(len(str(row.get("content", ""))) > 3000 for row in view) else 500
        return {"dispatch": ChatRequest(messages=[Message(**row) for row in view], model=selected["model"]),
            "budget_decision": {"estimated_input_tokens": tokens, "input_limit_tokens": 10000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": tokens}}}
    async def compact(request):
        unchanged()
        assert request.model == selected["model"]
        return {"kind": "native", "message": {"role": "user", "content": "Target note.",
            "metadata": {"source": "context-managed", "ephemeral": True, "persisted": True,
                         "fixture:opaque": {"model": selected["model"], "state": "target-only"}}}}
    async def install(candidate, verification):
        unchanged()
        if phase == "install":
            raise OSError("Synthetic installation failure")
        return candidate["sha256"]
    model = SimpleNamespace(name=selected["provider"], default_model=selected["model"],
        supports_native_compaction=lambda: True, validate_compacted_context=lambda row: (
            row.get("metadata", {}).get("fixture:opaque", {}).get("model") == selected["model"]),
        compact_context=AsyncMock(side_effect=compact), request_budget=AsyncMock(side_effect=budget),
        complete=AsyncMock())
    ctx.preserve_evidence = AsyncMock(return_value=copy.deepcopy(record["evidenceRefs"]))
    write, admit = AsyncMock(), AsyncMock(return_value=True)
    kwargs = dict(provider=model, count_view=count, identity=selected, save_progress=write,
        install=install, guard=lambda: None, source_checkpoint=record, admit_call=admit,
        max_calls=16, target_bytes=10000)
    if phase != "success":
        with pytest.raises(OSError):
            await ctx.recover_request(**kwargs)
        unchanged()
        if phase == "count":
            model.compact_context.assert_not_awaited()
            admit.assert_not_awaited()
            write.assert_not_awaited()
    else:
        result = await ctx.recover_request(**kwargs)
        assert result["outcome"] == "transitioned"
        target_record = ctx.export_checkpoint(selected)
        readback = BoundaryContextManager(fixture["config"])
        readback.messages = copy.deepcopy(packet["messages"])
        assert readback.restore_checkpoint(target_record, selected)["status"] == "restored"
        assert readback.export_checkpoint(selected) == target_record
        assert target_record["configuration"] == ctx._checkpoint_configuration()
        assert ctx._checkpoint_legacy_foreground is None
        assert record == source_bytes