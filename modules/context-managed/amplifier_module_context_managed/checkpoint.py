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
    if not context.summary:
        return None
    if not isinstance(identity, dict) or not identity.get("provider") or not identity.get("model"):
        raise ValueError("Checkpoint identity needs an explicit provider and model")
    if context.summary_identity != identity:
        context.checkpoint_status = {"status": "stale", "reason": "The summary belongs to a different provider/model identity."}
        return None
    through, text = context.summary
    record = {"format": FORMAT, "identity": copy.deepcopy(identity),
              "configuration": context.checkpoint_configuration(),
              "sourceRevision": {"messages": through, "sha256": digest(context.messages[:through])},
              "summary": {"throughMessage": through, **({"message": copy.deepcopy(text), "kind": "native"}
                          if isinstance(text, dict) else {"text": text})},
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
        if record.get("configuration") != context.checkpoint_configuration():
            raise ValueError("Summary configuration or prompt changed")
        source, summary = record["sourceRevision"], record["summary"]
        end = summary["throughMessage"]
        if type(end) is not int or not 0 < end <= len(context.messages) or source["messages"] != end:
            raise ValueError("Checkpoint source boundary is incompatible")
        if source["sha256"] != digest(context.messages[:end]):
            raise ValueError("Canonical source prefix changed")
        if summary.get("kind") == "native":
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
        context.summary = (end, copy.deepcopy(text))
        context.summary_identity = copy.deepcopy(identity)
        context.evidence_refs = copy.deepcopy(record.get("evidenceRefs", []))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        reason = str(exc)
    context.checkpoint_status = ({"status": "rejected", "reason": reason, "originalsAvailable": True} if reason else
        {"status": "restored", "throughMessage": context.summary[0], "originalsAvailable": True})
    return copy.deepcopy(context.checkpoint_status)
