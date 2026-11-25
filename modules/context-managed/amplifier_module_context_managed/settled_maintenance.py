"""Optional, session-owned checkpoint maintenance on a settled turn."""
import asyncio
import copy
import inspect
import logging

from .errors import CompactionError
from .measurement import RequestMeasurement, supports_native_for

# These are the derived pieces exported or restored together. Canonical history,
# operation state and the host's authority never move between contexts.
_FIELDS = ("summary", "summary_identity", "summary_omitted_indices", "evidence_refs",
           "_summary_progress", "_restored_progress", "summary_failure", "recovery",
           "checkpoint_status", "_foreground_native_refusal", "_checkpoint_legacy_foreground")


def _writer_guard(context):
    # The durable mount deliberately resolves this capability lazily. Its
    # forwarding function stays the same when the host changes actual writers.
    callback = context.persist_checkpoint
    resolve = context._host_capability
    selected = resolve("context.persist_checkpoint") if resolve else None

    def current():
        return (context.persist_checkpoint is callback and context._host_capability is resolve
                and (resolve is None or callable(selected) and resolve("context.persist_checkpoint") is selected))

    return current


def after_response(context, provider, retain_contents, input_tokens, output_tokens, prompt_characters):
    """Plan one checkpoint from an already confirmed reply; do not start it here."""
    if (any(type(n) is not int or n < 0 for n in (input_tokens, output_tokens, prompt_characters))
            or input_tokens == 0 or prompt_characters == 0 or not context.requires_measured_foreground
            or callable(getattr(provider, "request_budget", None)) or callable(getattr(provider, "stream", None))
            or not callable(getattr(provider, "complete", None)) or context.summary_provider is not None
            or supports_native_for(provider, getattr(provider, "default_model", None))
            or not context.messages or context.messages[-1].get("role") != "assistant"):
        return None
    fitter = context._fitter()
    budget = fitter._calculate_budget(None, provider)
    window = fitter._budget_provenance.get("context_window", budget)
    if input_tokens + output_tokens > window:
        return None
    # The preceding dispatch reports its own cost, including tools and instructions.
    # One more similarly sized prompt is a forecast for scheduling, not admission.
    anticipated_prompt = (prompt_characters + 3) // 4
    forecast = input_tokens + output_tokens + anticipated_prompt
    if forecast < budget * context.config.get("summarize_trigger", 0.70):
        return None

    def identity():
        return (context.checkpoint_identity() if context.checkpoint_identity else None,
                getattr(provider, "name", None), getattr(provider, "default_model", None))

    selected = copy.deepcopy(identity())
    checkpoint_identity = selected[0] if selected[0] is not None else {"provider": selected[1], "model": selected[2]}

    def available():
        return (not context._ordinary_transition and not context.recovery
                and not context._summary_progress and not context._restored_progress
                and not context._foreground_unconfirmed and not context.summary_failure
                and context.checkpoint_status.get("status") != "rejected"
                and (context.summary is None or isinstance(context.summary[1], str)
                     and context.summary_identity == checkpoint_identity))

    writer_current = _writer_guard(context)
    if not available() or not writer_current():
        return None
    revision, config, checkpoint = context.revision, copy.deepcopy(context.config), context._checkpoint_state()
    operations_current = context._operations_guard(context.active_operations() if context.active_operations else None)
    retained, consumed = tuple(retain_contents), False

    def settle():
        # Execution-end hooks may have stopped or superseded this exact reply.
        nonlocal consumed
        if consumed:
            return False
        consumed = True
        if (context.revision != revision or context.config != config or identity() != selected
                or context._checkpoint_state() != checkpoint or not operations_current() or not available() or not writer_current()):
            return False
        return schedule(context, provider, None, retained, anticipated_prompt_tokens=anticipated_prompt,
                        writer_current=writer_current)

    return settle


def schedule(context, provider, count_view, retain_contents, *, anticipated_prompt_tokens=0, writer_current=None):
    if (not context.requires_measured_foreground or not (context._maintenance_requested or anticipated_prompt_tokens)
            or context._maintenance_task is not None or context.lock.locked()
            or context._maintenance_revision == context.revision
            or context._ordinary_transition is not None or context._foreground_unconfirmed
            or context._is_cancelled and context._is_cancelled()):
        return False
    revision = context.revision
    config = copy.deepcopy(context.config)
    identity = copy.deepcopy(context.checkpoint_identity()) if context.checkpoint_identity else None
    operations = copy.deepcopy(context.active_operations()) if context.active_operations else None
    operations_current = context._operations_guard(operations)
    retain = tuple(retain_contents)
    context._maintenance_requested = False
    context._maintenance_revision = revision

    history = context.messages
    checkpoint = copy.deepcopy(context._checkpoint_state())
    selected = getattr(provider, "default_model", None)
    paid = None

    def current():
        source_current = context.revision == revision
        if paid is not None and paid.allow_append:
            source_current = (context.messages is history and context._checkpoint_state() == checkpoint
                              and getattr(provider, "default_model", None) == selected)
        return (source_current and (writer_current is None or writer_current()) and context.config == config
                and context._ordinary_transition is None and not task.cancelling()
                and context._maintenance_task is task and operations_current()
                and not (context._is_cancelled and context._is_cancelled())
                and (context.checkpoint_identity is None or context.checkpoint_identity() == identity))

    async def maintain():
        if not current():
            return
        # Generation works on private state. A provider may finish even after
        # cancellation. Only a still-current snapshot can publish its result.
        worker = copy.copy(context)
        worker.messages = list(context.messages)
        worker.config = config
        worker._settled_maintenance = True
        if context.hooks is not None:
            class MaintenanceHooks:
                async def emit(self, event, data):
                    return await context.hooks.emit(event, {**data, "context_work": "settled_maintenance"})
            worker.hooks = MaintenanceHooks()
        worker._is_cancelled = lambda: not current()
        for name in _FIELDS:
            setattr(worker, name, copy.deepcopy(getattr(context, name)))

        async def pay(invoke, input_key):
            nonlocal paid
            if anticipated_prompt_tokens:
                from .pending_foreground import PaidPrefix
                paid = PaidPrefix(task, provider, worker._summary_progress["source_revision"], current, writer_current)
                context._paid_settled_prefix = paid
            response = await context._paid_compaction(invoke, input_key)
            if paid is not None and paid.allow_append and not context._same_prefix(
                    {"source_revision": paid.source_revision}, context.messages):
                raise asyncio.CancelledError()
            return response

        worker._paid_compaction = pay

        async def publish():
            async with context.lock:
                if not current():
                    raise asyncio.CancelledError()
                fields = _FIELDS
                if paid is not None and paid.allow_append:
                    # The already-paid note must still cover the exact original
                    # bytes. Whole-history evidence belongs to the current host.
                    if (context._checkpoint_state() != checkpoint
                            or not context._same_prefix({"source_revision": paid.source_revision}, context.messages)):
                        raise asyncio.CancelledError()
                    if context.revision != revision:
                        fields = tuple(name for name in _FIELDS if name != "evidence_refs")
                previous = {name: getattr(context, name) for name in fields}
                for name in fields:
                    setattr(context, name, copy.deepcopy(getattr(worker, name)))
                try:
                    if context.persist_checkpoint:
                        saved = context.persist_checkpoint()
                        if inspect.isawaitable(saved):
                            saved = await saved
                        if saved is False:
                            raise RuntimeError("Checkpoint storage declined the candidate")
                    if paid is not None and paid.allow_append:
                        paid.committed_checkpoint = copy.deepcopy(context._checkpoint_state())
                except (Exception, asyncio.CancelledError):
                    # A suffix replacement may already have invalidated parts
                    # of the derived state; never restore a stale pointer.
                    if context.revision == revision or (paid is not None and paid.allow_append
                            and context.messages is history and context.summary == worker.summary):
                        for name, value in previous.items():
                            setattr(context, name, value)
                    raise
            return True

        worker.persist_checkpoint = publish
        try:
            await worker._prepare(provider, None, retain, RequestMeasurement(count_view) if count_view else None,
                                  ordinary=True, operations_guard=current,
                                  anticipated_prompt_tokens=anticipated_prompt_tokens)
        except CompactionError as error:
            if error.code == "native_input_oversized" and current():
                context._foreground_native_refusal = worker._foreground_native_refusal
                # The foreground may have its own fit failure. Keep the last
                # confirmed native refusal visible alongside the active checkpoint.
                context.checkpoint_status = {**context.checkpoint_status,
                    "lastNativeFailure": copy.deepcopy(worker.checkpoint_status["failure"])}
            raise
        if worker.summary != context.summary:
            await publish()

    def finished(done):
        if paid is not None and context._paid_settled_prefix is paid:
            context._paid_settled_prefix = None
        if context._maintenance_task is done:
            context._maintenance_task = None
        if not done.cancelled() and (error := done.exception()) is not None:
            # Exceptions can contain original inputs. Report only a safe type.
            logging.getLogger(__name__).info("Settled compaction deferred (%s)", type(error).__name__)

    task = asyncio.create_task(maintain(), name="context-managed-settled-compaction")
    context._maintenance_task = task
    task.add_done_callback(finished)
    return True
