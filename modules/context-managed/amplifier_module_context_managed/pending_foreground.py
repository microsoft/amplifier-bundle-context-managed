"""An ordinary text turn can use the unchanged source of one already-paid prefix."""
import asyncio
import copy
from contextlib import asynccontextmanager

from amplifier_core import ChatRequest
from amplifier_core.llm_errors import ContextLengthError


def text_request_fits(request, fitter, provider):
    if not isinstance(request, ChatRequest):
        return False
    for msg in request.messages:
        if (msg.metadata or {}).get("experimental.media.v1"):
            return False
        if not isinstance(msg.content, str):
            if not isinstance(msg.content, list) or any(
                    part.type not in {"text", "tool_call"} and not (
                        part.type == "tool_result" and isinstance(part.output, str))
                    for part in msg.content):
                return False
    # Same local planning convention as ordinary uncounted text. Include all
    # projected tools/options, and reserve the actual requested/selected maximum.
    budget = fitter._calculate_budget(None, provider)
    source = fitter._budget_provenance
    selected = source.get("max_output_tokens", 0)
    info = provider.get_info() if callable(getattr(provider, "get_info", None)) else None
    defaults = info.get("defaults", {}) if isinstance(info, dict) else getattr(info, "defaults", {})
    output = defaults.get("max_output_tokens", selected)
    if type(output) is not int or output < 0:
        return False
    reserve = max(request.max_output_tokens or 0, output, selected)
    limit = min(budget, source["derived_budget"] - max(0, reserve - source.get("reserved_output", 0)))
    return (len(request.model_dump_json(exclude_none=True).encode("utf-8")) + 3) // 4 <= limit


class PaidPrefix:
    def __init__(self, task, provider, source_revision, current, writer_current):
        self.task = task
        self.provider = provider
        self.source_revision = source_revision
        self.current = current
        self.writer_current = writer_current
        self.allow_append = False
        self.committed_checkpoint = None

    def begin(self, context, provider):
        prepare = getattr(provider, "prepared_foreground", None)
        if (provider is not self.provider or not callable(prepare) or not self.current()
                or callable(getattr(provider, "request_budget", None)) or callable(getattr(provider, "stream", None))):
            return None
        self.allow_append = True
        if not self.current():
            return None

        @asynccontextmanager
        async def fit(*, retain_contents, project_view, request_options):
            eligible = (self.current() and not request_options.get("stream") and not request_options.get("extended_thinking"))
            if eligible:
                try:
                    async with context.lock:
                        current_revision = context.revision
                        config = copy.deepcopy(context.config)
                        identity = copy.deepcopy(context.checkpoint_identity()) if context.checkpoint_identity else None
                        checkpoint = copy.deepcopy(context._checkpoint_state())
                        operations = context._operations_guard(context.active_operations() if context.active_operations else None)

                        def validate():
                            context._check_stop()
                            observed = context._checkpoint_state()
                            if (context.revision != current_revision or context.config != config
                                    or not self.writer_current() or not operations()
                                    or context.checkpoint_identity and context.checkpoint_identity() != identity
                                    or observed != checkpoint and observed != self.committed_checkpoint
                                    or self.task.cancelling()):
                                raise RuntimeError("The prepared foreground conversation changed before dispatch")

                        # Skip only speculative semantic generation: this exact
                        # prefix is already being summarized on the original provider.
                        fitter, retain = await context._prepare(provider, None, retain_contents,
                                                                ordinary=True, skip_optional_summary=True)
                        rows = await fitter.get_messages_for_request_retaining(retain_contents=retain, provider=provider)
                        logical = project_view(rows)
                except ContextLengthError:
                    logical = None
                if logical is not None and self.current() and current_revision == context.revision:
                    async with prepare(logical, request_options=request_options) as projected:
                        try:
                            fits = text_request_fits(projected, fitter, provider)
                        except (AttributeError, TypeError, ValueError):
                            fits = False  # Unknown envelope or output option stays on the original path.
                        if self.current() and current_revision == context.revision and fits:
                            validate()
                            yield {"base_view": rows, "dispatch": logical, "validate": validate}
                            return
            # Release the trial and context lock before joining a needed save.
            if not self.task.done():
                await asyncio.wait({self.task})
            context._check_stop()
            yield None

        return fit
