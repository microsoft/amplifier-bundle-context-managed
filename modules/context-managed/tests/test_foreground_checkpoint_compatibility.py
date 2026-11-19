"""Foreground fitting policy must not invalidate prior summary provenance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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
    with pytest.raises(CompactionError, match="summary_progress_incompatible"):
        ctx.export_checkpoint({**identity, "model": "other-model"})
    assert ctx._restored_progress == before
    assert ctx.export_checkpoint(identity) == record


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