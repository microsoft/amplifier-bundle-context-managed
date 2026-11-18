"""One non-durable ordinary transition job, never a host persistence writer."""
import copy
import inspect

from amplifier_core.llm_errors import ContextLengthError

from .checkpoint import digest
from .errors import CompactionError


UNIT = ("summary", "summary_identity", "summary_omitted_indices", "recovery",
        "_summary_progress", "_restored_progress", "summary_failure", "evidence_refs")


async def prepare(context, *, provider, token_budget, retain, measurement, reason,
                  messages, revision, identity, system_prompt, operations):
    """Retain source through final fit; reuse only one exactly bound target job.

    There is no durable progress claim here. A legacy zero-argument writer cannot
    atomically preserve both units, so durable hosts refuse before generation.
    Participating hosts keep their explicit journal admission policy instead.
    """
    from .boundary import BoundaryContextManager

    selected = lambda: (copy.deepcopy(context.checkpoint_identity())
        if context.checkpoint_identity else {
            "provider": getattr(provider, "name", type(provider).__name__),
            "model": getattr(provider, "default_model", None)})
    def policy():
        value = context.transition_policy() if context.transition_policy else None
        if value not in (None, "host-managed"):
            raise ValueError("context.transition_policy must return None or host-managed")
        return value
    callbacks = (context.factory, context.active_operations, context.checkpoint_identity,
                 context.summary_provider, context.preserve_evidence,
                 context.persist_checkpoint, context.transition_policy)
    source_unit = copy.deepcopy({key: getattr(context, key) for key in UNIT})
    binding = (digest(messages), copy.deepcopy(context.config), copy.deepcopy(identity),
               id(provider),
               getattr(provider, "name", type(provider).__name__),
               getattr(provider, "default_model", None), callbacks,
               copy.deepcopy(source_unit), tuple(retain), system_prompt,
               copy.deepcopy(operations), policy())

    def guard():
        current = (digest(context.messages), context.config, selected(),
                   id(provider),
                   getattr(provider, "name", type(provider).__name__),
                   getattr(provider, "default_model", None),
                   (context.factory, context.active_operations, context.checkpoint_identity,
                    context.summary_provider, context.preserve_evidence,
                    context.persist_checkpoint, context.transition_policy),
                   {key: getattr(context, key) for key in UNIT}, tuple(retain),
                   system_prompt, operations, policy())
        if revision != context.revision or current != binding:
            raise RuntimeError("Request contract changed during ordinary model transition")
        if context.active_operations and context.active_operations() != operations:
            raise RuntimeError("Operation observations changed during ordinary model transition")
        pending = context._ordinary_transition
        if pending is not None and pending.get("utility") is not None:
            utility, key = pending["utility"]
            current_utility = context.summary_provider() if context.summary_provider else provider
            if (current_utility is not utility
                    or key != (getattr(utility, "name", type(utility).__name__),
                               getattr(utility, "default_model", None))):
                raise RuntimeError("Summary provider changed during ordinary model transition")

    guard()
    job = context._ordinary_transition
    if job is not None and job["binding"] != binding:
        if job["paid_calls"]:
            raise CompactionError("summary_progress_incompatible",
                "A provisional model-transition job belongs to different source or selection. "
                "Its paid work is retained in memory; do not replace or automatically retry it.")
        context._ordinary_transition = job = None  # No paid result/progress to discard.
    if job is not None and job["uncertain"]:
        raise CompactionError("transition_operation_unconfirmed",
            "The previous transition model operation has an unconfirmed outcome. "
            "Its source and provisional work remain in memory; reconcile before new generation.")

    def fresh():
        clone = BoundaryContextManager(copy.deepcopy(context.config), context.hooks)
        clone.messages, clone.revision = copy.deepcopy(messages), revision
        # Archive omission is not whole-prefix transition compatibility.
        clone.config["archive_recovery"] = False
        clone.checkpoint_identity = lambda: copy.deepcopy(identity)
        if context.factory:
            async def prompt():
                guard()
                return system_prompt
            clone.factory = prompt
        clone.active_operations = lambda: copy.deepcopy(operations)
        return clone

    if job is None:
        clone = fresh()
        fitted, protected = await clone._prepare(None, token_budget, retain)
        guard()
        # Preserve zero-paid fitting for all hosts, without passing source-model
        # opaque state to count or generation.
        try:
            if measurement is None:
                await fitted.get_messages_for_request_retaining(
                    retain_contents=protected, provider=provider, token_budget=token_budget)
            else:
                await fitted.get_measured_request_view(
                    provider=provider, retain_contents=protected, count_view=measurement.count)
        except ContextLengthError:
            guard()
        else:
            guard()
            fitted._transition_guard = guard
            return fitted, protected
        if policy() == "host-managed":
            raise CompactionError("model_transition_required",
                "The participating host must admit this model transition through its existing recovery transaction.")
        if context.config.get("durable_checkpoints") or context.persist_checkpoint is not None:
            raise CompactionError("transition_persistence_required",
                "Ordinary model-switch regeneration cannot safely stage target progress with a legacy "
                "checkpoint writer. Keep the existing checkpoint and integrate the host transition "
                "source_checkpoint/save_progress/install callbacks before buying replacement work.")
        job = {"binding": binding, "clone": fresh(), "uncertain": False,
               "last_response": None, "prepared": None, "paid_calls": 0}
        context._ordinary_transition = job
    clone = job["clone"]

    class GuardedProvider:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, key):
            value = getattr(self.original, key)
            if not callable(value) or key in {"get_info", "supports_native_compaction",
                                              "validate_compacted_context"}:
                return value
            async def call(*args, **kwargs):
                guard()
                paid = key in {"complete", "compact_context"}
                request_key = digest(args[0].model_dump(mode="json")) if paid else None
                if (paid and job.get("returned_request") == (key, request_key)
                        and job["last_response"] is not None):
                    # Reuse a completed returned value after a counting outage;
                    # this is not another transport operation or paid retry.
                    return copy.deepcopy(job["last_response"])
                if paid:
                    job["paid_calls"] += 1
                    job["uncertain"] = True
                try:
                    result = value(*args, **kwargs)
                    result = await result if inspect.isawaitable(result) else result
                except BaseException:
                    if key == "request_budget":
                        job["count_failed"] = True
                    raise
                if paid:
                    # Keep a returned paid result even if a late guard prevents
                    # its normal staging; never claim that it is free to rebuy.
                    job["last_response"] = result
                    job["returned_request"] = key, request_key
                guard()
                if paid:
                    job["uncertain"] = False
                return result
            return call

    class GuardedMeasurement:
        def __getattr__(self, key):
            return getattr(measurement, key)

        async def measure(self, *args):
            guard()
            job["count_progress"] = copy.deepcopy(clone._summary_progress)
            try:
                result = await measurement.measure(*args)
            except BaseException:
                job["count_failed"] = True
                raise
            guard()
            return result

    if job["prepared"] is None:
        guarded = GuardedProvider(provider)
        if context.summary_provider:
            utility = context.summary_provider()
            job["utility"] = (utility, (getattr(utility, "name", type(utility).__name__),
                                       getattr(utility, "default_model", None)))
            clone.summary_provider = lambda: GuardedProvider(context.summary_provider())
        job["count_failed"] = False
        try:
            fitted, protected = await clone._prepare(
                guarded, token_budget, retain,
                GuardedMeasurement() if measurement is not None else None, reason=reason)
        except BaseException:
            if job["count_failed"] and job["last_response"] is not None:
                # A measurement outage is not a failed generation. Retain the
                # exact completed portable progress even when _prepare's generic
                # error cleanup removed its fully covered provisional note.
                paid_progress = job.get("count_progress")
                if paid_progress and paid_progress.get("completed"):
                    clone._summary_progress = paid_progress
                clone.summary_failure = None
            raise
        guard()
        job["prepared"] = fitted, protected
    else:
        fitted, protected = job["prepared"]

    def commit():
        activation = copy.deepcopy({key: getattr(clone, key) for key in UNIT})
        guard()
        for key, value in activation.items():
            setattr(context, key, value)
        context.checkpoint_status = copy.deepcopy(clone.checkpoint_status)
        context._ordinary_transition = None

    fitted._transition_guard = guard
    fitted._transition_commit = commit
    return fitted, protected