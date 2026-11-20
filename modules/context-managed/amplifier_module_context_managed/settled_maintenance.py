"""Optional, session-owned checkpoint maintenance on a settled turn."""
import asyncio
import copy
import inspect
import logging

from .errors import CompactionError
from .measurement import RequestMeasurement

# These are the derived pieces exported or restored together. Canonical history,
# operation state and the host's authority never move between contexts.
_FIELDS = ("summary", "summary_identity", "summary_omitted_indices", "evidence_refs",
           "_summary_progress", "_restored_progress", "summary_failure", "recovery",
           "checkpoint_status", "_foreground_native_refusal", "_checkpoint_legacy_foreground")


def schedule(context, provider, count_view, retain_contents):
    if (not context.requires_measured_foreground or not context._maintenance_requested
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

    def current():
        return (context.revision == revision and context.config == config
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
        worker._paid_compaction = context._paid_compaction

        async def publish():
            async with context.lock:
                if not current():
                    raise asyncio.CancelledError()
                previous = {name: getattr(context, name) for name in _FIELDS}
                for name in _FIELDS:
                    setattr(context, name, copy.deepcopy(getattr(worker, name)))
                try:
                    if context.persist_checkpoint:
                        saved = context.persist_checkpoint()
                        if inspect.isawaitable(saved):
                            saved = await saved
                        if saved is False:
                            raise RuntimeError("Checkpoint storage declined the candidate")
                except (Exception, asyncio.CancelledError):
                    # A suffix replacement may already have invalidated parts
                    # of the derived state; never restore a stale pointer.
                    if context.revision == revision:
                        for name, value in previous.items():
                            setattr(context, name, value)
                    raise
            return True

        worker.persist_checkpoint = publish
        try:
            await worker._prepare(provider, None, retain, RequestMeasurement(count_view), ordinary=True,
                                  operations_guard=current)
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
        if context._maintenance_task is done:
            context._maintenance_task = None
        if not done.cancelled() and (error := done.exception()) is not None:
            # Exceptions can contain original inputs. Report only a safe type.
            logging.getLogger(__name__).info("Settled compaction deferred (%s)", type(error).__name__)

    task = asyncio.create_task(maintain(), name="context-managed-settled-compaction")
    context._maintenance_task = task
    task.add_done_callback(finished)
    return True
