"""Portable, derived context checkpoints. Canonical history is host-owned."""
import copy
import hashlib
import json

FORMAT = "context-managed-boundary-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def export_checkpoint(context, identity):
    if not context.config.get("durable_checkpoints"):
        return None
    progress = context._summary_progress
    if not context.summary and not (progress and progress["completed"]):
        return None
    if not isinstance(identity, dict) or not identity.get("provider") or not identity.get("model"):
        raise ValueError("Checkpoint identity needs an explicit provider and model")
    if context.summary and context.summary_identity != identity:
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
    record = {"format": FORMAT, "identity": copy.deepcopy(identity),
              "configuration": context.checkpoint_configuration(), "configurationVersion": 2,
              "sourceRevision": {"messages": through, "sha256": digest(context.messages[:through])},
              "summary": ({"throughMessage": through, **({"message": copy.deepcopy(text), "kind": "native"}
                          if isinstance(text, dict) else {"text": text})} if context.summary else None),
              "progress": partial,
              "evidenceRefs": copy.deepcopy(context.evidence_refs)}
    record["sha256"] = digest(record)
    return record


def restore_checkpoint(context, record, identity):
    context.summary = None
    context.summary_identity = None
    context.evidence_refs = []
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
        configuration = context.checkpoint_configuration()
        if record.get("configurationVersion", 1) == 1:
            from .boundary import SUMMARY_PROMPT
            configuration = digest({"config": context.config, "prompt": SUMMARY_PROMPT})
        elif record.get("configurationVersion") != 2:
            raise ValueError("Checkpoint configuration contract is unsupported")
        if record.get("configuration") != configuration:
            raise ValueError("Summary configuration or prompt changed")
        source, summary = record["sourceRevision"], record.get("summary")
        partial = record.get("progress")
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
        if summary is None:
            text = None
        elif summary.get("kind") == "native":
            text = summary.get("message")
            if (not isinstance(text, dict) or text.get("role") != "user" or not text.get("content")
                    or not isinstance(text.get("metadata"), dict)
                    or text["metadata"].get("source") != "context-managed"
                    or text["metadata"].get("ephemeral") is not True
                    or text["metadata"].get("persisted") is not True):
                raise ValueError("Native checkpoint message is invalid")
        else:
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
        context._restored_progress = copy.deepcopy(partial)
        context.summary = (end, copy.deepcopy(text)) if summary is not None else None
        context.summary_identity = copy.deepcopy(identity) if summary is not None else None
        context.evidence_refs = copy.deepcopy(record.get("evidenceRefs", []))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        reason = str(exc)
    context.checkpoint_status = ({"status": "rejected", "reason": reason, "originalsAvailable": True} if reason else
        {"status": "resuming" if context._restored_progress else "restored",
         "throughMessage": context.summary[0] if context.summary else None,
         "completedParts": context._restored_progress["completed"] if context._restored_progress else 0,
         "originalsAvailable": True})
    return copy.deepcopy(context.checkpoint_status)
