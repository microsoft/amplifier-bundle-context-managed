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