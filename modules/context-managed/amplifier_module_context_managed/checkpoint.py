"""Portable, derived context checkpoints. Canonical history is host-owned."""
import copy
import hashlib
import json

FORMAT = "context-managed-boundary-v1"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def native_handoff_eligible(context, provider, target, source_identity, through, omitted, retain, ceiling):
    from .measurement import supports_native_for

    if (source_identity["provider"] != target["provider"] or through > ceiling
            or context._boundary(context.messages, retain, ceiling=through) != through
            or not context.config.get("native_compaction", True)
            or not supports_native_for(provider, target["model"])):
        return False
    required = {id(row) for row in context._carry(context.messages)}
    return all(i in omitted for i, row in enumerate(context.messages[:through])
               if id(row) in required or context._retained_reminder(row, retain))


def handoff_summary(context, provider, target, *, source_identity, summary, omitted, retain, ceiling):
    """Qualify a known source prefix locally. Only the provider interprets opaque payloads."""
    from .errors import CompactionError

    if not summary or not source_identity:
        return None
    through, value = summary
    if through > ceiling or context._boundary(context.messages, retain, ceiling=through) != through:
        return None
    if isinstance(value, str):
        return through, value
    rebind = getattr(provider, "rebind_compacted_context", None)
    if callable(rebind) and native_handoff_eligible(
            context, provider, target, source_identity, through, omitted, retain, ceiling):
        rebound = rebind(copy.deepcopy(value), source_model=source_identity["model"], model=target["model"])
        if rebound is not None:
            if not provider.validate_compacted_context(rebound):
                raise CompactionError("native_checkpoint_invalid", "Reused native window did not verify.")
            return through, rebound

    # A source adapter may explicitly identify its validated, complete native
    # block text as portable. A label on an encrypted carrier isn't a summary.
    content = value.get("content")
    if ((value.get("metadata") or {}).get("context-managed:portable-summary") is not True
            or not isinstance(content, str) or not content.strip()):
        return None
    if omitted:
        # Text checkpoints do not store an omission map. Only convert if their
        # normal projection carries every omitted source row outside the note.
        projected = context._assembly_rows(context.messages, (through, content), retain, context._carry(context.messages), ())
        carried = {i for i, _ in projected if i is not None and i < through}
        if not set(omitted).issubset(carried):
            return None
    return through, content


def classify_transition(context, record, identity):
    """Validate against host originals/config, without installing foreign state.

    The host must additionally bind the actual saved bytes and explicit selection
    to its admission journal. A record's self-declared identity is not authority.
    No provider is contacted, including to validate source-native transport.
    """
    from .boundary import BoundaryContextManager

    def rejected(code, reason):
        return {"status": "rejected", "code": code, "reason": reason,
                "originalsAvailable": True}

    valid_identity = lambda value: (isinstance(value, dict)
        and all(isinstance(value.get(key), str) and value[key].strip()
                for key in ("provider", "model")))
    if not valid_identity(identity):
        return rejected("transition_identity_invalid", "Target identity is incomplete.")
    source_identity = record.get("identity") if isinstance(record, dict) else None
    if not valid_identity(source_identity):
        return rejected("transition_source_invalid", "Source identity is incomplete.")
    scratch = BoundaryContextManager(copy.deepcopy(context.config))
    scratch.messages = copy.deepcopy(context.messages)
    status = scratch.restore_checkpoint(copy.deepcopy(record), source_identity)
    if status["status"] not in {"restored", "resuming"}:
        return rejected("transition_source_invalid", status["reason"])
    if source_identity == identity:
        return {"status": "compatible", "sha256": record["sha256"]}
    return {"status": "transition_required",
            "sourceIdentity": copy.deepcopy(source_identity),
            "targetIdentity": copy.deepcopy(identity),
            "sourceRevision": copy.deepcopy(record["sourceRevision"]),
            "configuration": record["configuration"],
            "sha256": record["sha256"],
            "validation": {"record": "verified", "canonicalPrefix": "verified",
                           "configuration": "verified",
                           "sourceProviderEnvelope": "unverified"}}


def export_checkpoint(context, identity):
    if not context.config.get("durable_checkpoints"):
        context._checkpoint_prefix_digest.clear()
        return None
    progress = context._summary_progress or context._restored_progress
    if progress and progress["completed"] and (progress.get("configuration") != context.checkpoint_configuration()
            or progress["identity"] != identity or not context._same_prefix(progress, context.messages)):
        # Incompatible paid work stays quarantined in memory/its old host
        # checkpoint. Never return a completed-only replacement that silently
        # deletes the partial, or stamp it with changed configuration/originals.
        context._refuse_progress()
    if not context.summary and not (progress and progress["completed"]):
        context._checkpoint_prefix_digest.clear()
        return None
    if not isinstance(identity, dict) or not identity.get("provider") or not identity.get("model"):
        raise ValueError("Checkpoint identity needs an explicit provider and model")
    if context.summary and context.summary_identity != identity:
        context._checkpoint_prefix_digest.clear()
        context.checkpoint_status = {"status": "stale", "reason": "The summary belongs to a different provider/model identity."}
        return None
    through, text = context.summary if context.summary else (progress["source_revision"]["messages"], None)
    partial = None
    if (progress and progress["completed"] and progress["identity"] == identity
            and context._same_prefix(progress, context.messages)):
        # Only derived notes and piece boundaries are stored. Pending source
        # bytes are regenerated from canonical history, never copied to a second
        # output archive and never treated as executable work to replay.
        partial = {key: copy.deepcopy(progress[key]) for key in
                   ("fingerprint", "source_revision", "identity", "provider", "parts_digest",
                    "segments", "text", "calls", "completed")}
        partial["provider"] = list(partial["provider"])
        if "generation_contract" in progress:
            partial["generation_contract"] = copy.deepcopy(progress["generation_contract"])
    version = 3 if isinstance(text, dict) else 2
    if partial and "generation_contract" in partial:
        contract = partial["generation_contract"]
        version = 5 if "prompt_sha256" in contract else 4
        if contract["max_output_tokens"] is None:
            version = 6  # The service chose the output limit for these fragments.
    record = {"format": FORMAT, "identity": copy.deepcopy(identity),
              # Version 3 fences older readers that would silently discard the
              # native summary's omission offsets. The configuration hash is
              # unchanged; portable notes retain the version 2 contract.
              "configuration": context.checkpoint_configuration(),
              # Older readers must not resume new output-contract progress as
              # though it had been generated under their implicit 8192 default.
              "configurationVersion": version,
              "sourceRevision": {"messages": through, "sha256": context._checkpoint_prefix_digest.digest(context.messages[:through], digest, canonical_bytes)},
              "summary": ({"throughMessage": through, **({"message": copy.deepcopy(text), "kind": "native",
                          "version": 2, "omittedIndices": copy.deepcopy(context.summary_omitted_indices)}
                          if isinstance(text, dict) else {"text": text})} if context.summary else None),
              "progress": partial,
              "evidenceRefs": copy.deepcopy(context.evidence_refs),
              "recovery": copy.deepcopy(context.recovery)}
    record["sha256"] = digest(record)
    return record


def restore_checkpoint(context, record, identity):
    context._checkpoint_prefix_digest.clear()
    context.summary = None
    context.summary_omitted_indices = []
    context.summary_identity = None
    context.evidence_refs = []
    context.recovery = None
    reason = None
    try:
        if not context.config.get("durable_checkpoints"):
            raise ValueError("Durable checkpoints are not enabled")
        if not isinstance(record, dict) or record.get("format") != FORMAT:
            raise ValueError("Checkpoint format is unsupported")
        if record.get("sha256") != digest({k: v for k, v in record.items() if k != "sha256"}):
            raise ValueError("Checkpoint digest does not match its contents")
        if record.get("identity") != identity or not identity.get("provider") or not identity.get("model"):
            raise ValueError("Provider or model identity changed")
        # Restore admission always starts with the current canonical projection,
        # never the active unit's historical provenance.
        configuration = context._checkpoint_configuration()
        legacy_foreground = None
        if record.get("configurationVersion", 1) == 1:
            from .boundary import SUMMARY_PROMPT
            configuration = digest({"config": context.config, "prompt": SUMMARY_PROMPT})
        elif record.get("configurationVersion") not in {2, 3, 4, 5, 6}:
            raise ValueError("Checkpoint configuration contract is unsupported")
        elif ("foreground_continuation" in context.config
                and type(context.config["foreground_continuation"]) is not bool):
            raise ValueError("Foreground continuation policy must be boolean")
        if record.get("configuration") != configuration:
            if record.get("configurationVersion") not in {2, 3, 4, 5, 6}:
                raise ValueError("Summary configuration or prompt changed")
            for flag in (False, True):
                if record.get("configuration") == context._checkpoint_configuration(flag):
                    legacy_foreground = flag
                    break
            else:
                raise ValueError("Summary configuration or prompt changed")
        source, summary = record["sourceRevision"], record.get("summary")
        if record.get("configurationVersion") == 3 and (
                not isinstance(summary, dict) or summary.get("kind") != "native"
                or summary.get("version") != 2):
            raise ValueError("Checkpoint requires the native omission contract")
        partial = record.get("progress")
        version = record.get("configurationVersion")
        if version not in {4, 5, 6} and isinstance(partial, dict) and "generation_contract" in partial:
            raise ValueError("Saved summary generation contract requires checkpoint version 4 or newer")
        if version in {4, 5, 6}:
            contract = partial.get("generation_contract") if isinstance(partial, dict) else None
            keys = {"model", "max_output_tokens", "reasoning_effort"}
            if not isinstance(contract, dict):
                raise ValueError("Saved summary generation contract is invalid")
            if version == 5 or (version == 6 and "prompt_sha256" in contract):
                keys.add("prompt_sha256")
            cap = contract.get("max_output_tokens")
            valid_cap = (cap is None) if version == 6 else (type(cap) is int and cap > 0)
            if (set(contract) != keys or not valid_cap
                    or contract["model"] is not None and not isinstance(contract["model"], str)
                    or contract["reasoning_effort"] is not None and not isinstance(contract["reasoning_effort"], str)):
                raise ValueError("Saved summary generation contract is invalid")
            if "prompt_sha256" in keys:
                prompt_digest = contract["prompt_sha256"]
                if (not isinstance(prompt_digest, str) or len(prompt_digest) != 64
                        or any(c not in "0123456789abcdef" for c in prompt_digest)):
                    raise ValueError("Saved summary prompt contract is invalid")
        if summary is None:
            if not partial:
                raise ValueError("Checkpoint has no completed note or saved progress")
            end = source["messages"]
        else:
            end = summary["throughMessage"]
        if type(end) is not int or not 0 < end <= len(context.messages) or source["messages"] != end:
            raise ValueError("Checkpoint source boundary is incompatible")
        if source["sha256"] != digest(context.messages[:end]):
            raise ValueError("Canonical source prefix changed")
        omitted = []
        if summary is None:
            text = None
        elif summary.get("kind") == "native":
            version = summary.get("version", 1)
            if type(version) is not int or version not in {1, 2}:
                raise ValueError("Native summary contract is unsupported")
            recovery = record.get("recovery")
            allowed = {"throughMessage", "message", "kind", "version"}
            if version == 2:
                if record.get("configurationVersion") not in {3, 4, 5, 6}:
                    raise ValueError("Native omission contract requires checkpoint version 3")
                allowed.add("omittedIndices")
                # The only reconstructible hash-only provenance is known empty.
                empty_binding = {"kind": "native-recovery", "version": 1,
                                 "retentionSha256": digest([])}
                omitted = summary.get("omittedIndices", [] if recovery == empty_binding else None)
                if (not isinstance(omitted, list)
                        or any(type(i) is not int or not 0 <= i < end for i in omitted)
                        or omitted != sorted(set(omitted))):
                    raise ValueError("Native summary omission indices are invalid")
            if set(summary) - allowed:
                raise ValueError("Native summary contains unsupported metadata")
            if isinstance(recovery, dict) and recovery.get("kind") == "native-recovery":
                # Hash-only legacy records cannot reconstruct retained rows.
                # Only the known-empty hash is safe without explicit offsets.
                if recovery != {"kind": "native-recovery", "version": 1,
                                "retentionSha256": digest(omitted)}:
                    raise ValueError("Native recovery omission provenance is incompatible")
            text = summary.get("message")
            if (not isinstance(text, dict) or text.get("role") != "user" or not text.get("content")
                    or not isinstance(text.get("metadata"), dict)
                    or text["metadata"].get("source") != "context-managed"
                    or text["metadata"].get("ephemeral") is not True
                    or text["metadata"].get("persisted") is not True):
                raise ValueError("Native checkpoint message is invalid")
        else:
            if "omittedIndices" in summary or "version" in summary:
                raise ValueError("Omission metadata requires a native summary")
            text = summary.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Checkpoint summary is empty")
        if not isinstance(record.get("evidenceRefs", []), list):
            raise ValueError("Checkpoint evidence references are invalid")
        if partial:
            if (not isinstance(partial, dict) or partial.get("identity") != identity
                    or not context._same_prefix(partial, context.messages)
                    or type(partial.get("completed")) is not int or partial["completed"] < 1
                    or type(partial.get("calls")) is not int or partial["calls"] < partial["completed"]
                    or not isinstance(partial.get("text"), str) or not partial["text"].strip()
                    or not isinstance(partial.get("provider"), list) or len(partial["provider"]) != 2
                    or not isinstance(partial.get("fingerprint"), str)
                    or not isinstance(partial.get("parts_digest"), str)
                    or not isinstance(partial.get("segments"), list)
                    or any(not isinstance(v, list) or len(v) != 3 or any(type(n) is not int for n in v)
                           for v in partial["segments"])):
                raise ValueError("Saved summary progress is incompatible")
        # Set only after the full record/identity/source/shape checks. This keeps
        # exports and paid fingerprints exact, without rewriting saved records.
        context._checkpoint_legacy_foreground = legacy_foreground
        context._restored_progress = copy.deepcopy(partial)
        if context._restored_progress:
            # In-memory provenance for export before the first resumed call.
            # Not a new serialized partial field or a relaxed import contract.
            context._restored_progress["configuration"] = context.checkpoint_configuration()
        context.summary = (end, copy.deepcopy(text)) if summary is not None else None
        context.summary_omitted_indices = copy.deepcopy(omitted)
        context.summary_identity = copy.deepcopy(identity) if summary is not None else None
        context.evidence_refs = copy.deepcopy(record.get("evidenceRefs", []))
        context.recovery = copy.deepcopy(record.get("recovery"))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        reason = str(exc)
    context.checkpoint_status = ({"status": "rejected", "reason": reason, "originalsAvailable": True} if reason else
        {"status": "resuming" if context._restored_progress else "restored",
         "throughMessage": context.summary[0] if context.summary else None,
         "completedParts": context._restored_progress["completed"] if context._restored_progress else 0,
         "originalsAvailable": True})
    return copy.deepcopy(context.checkpoint_status)
