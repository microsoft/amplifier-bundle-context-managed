"""Validate a compacted request without mechanically deleting or clipping history."""
import asyncio
import copy
import inspect
import logging

from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import ContextLengthError
from amplifier_module_context_simple import SimpleContextManager

from .summary import request_fits, public_messages, require_media_budget


logger = logging.getLogger(__name__)


class RequestContext(SimpleContextManager):
    """Reuse budget/measurement contracts, never the emergency reduction ladder."""

    async def set_messages(self, messages):
        # The parent ingress path can truncate tool results. Only semantic or
        # native compaction may transform the request owned by this module.
        self.messages = copy.deepcopy(messages)

    async def _view(self):
        rows = copy.deepcopy(self.messages)
        if self._system_prompt_factory:
            prompt = await self._system_prompt_factory()
            rows = [{"role": "system", "content": prompt}] + [row for row in rows
                if row.get("role") != "system" or (row.get("metadata") or {}).get("source") == "hook"]
        return self._strip_internal_metadata(rows)

    @staticmethod
    def _oversize(count, limit):
        error = ContextLengthError(
            f"Context preparation stopped: input {count} tokens exceeds allowance {limit}. "
            "No emergency history trimming was applied. Restore a compatible compaction "
            "checkpoint or explicitly recover the oversized history before continuing.")
        error.context_input_tokens = count
        error.context_input_limit = limit
        raise error

    async def get_messages_for_request_retaining(self, *, retain_contents, provider=None, token_budget=None, hard_fit=False):
        view = await self._view()
        budget = self._calculate_budget(token_budget, provider)
        count = self._estimate_tokens(public_messages(view))
        request = ChatRequest(messages=[Message(**row) for row in view])
        failed = False
        try:
            fits, decision = await request_fits(provider, request,
                                                estimate_tokens=self._estimate_tokens, fallback_budget=budget)
            if decision:
                count, budget = decision["input_tokens"], min(budget, decision["input_limit_tokens"])
            if not fits or count > budget:
                self._oversize(count, budget)
            return view
        except BaseException:
            failed = True
            raise
        finally:
            # Only rows escape this API, so nobody else can abandon this exact
            # preliminary request. The measured-view API instead hands dispatch
            # ownership to its caller and must not release it here.
            try:
                release = getattr(provider, "release_media_request", None)
                if callable(release):
                    released = release(request, request_options=None)
                    if inspect.isawaitable(released):
                        released = await released
                    if released is False:
                        logger.warning("Preliminary media request release was not confirmed")
            except asyncio.CancelledError:
                logger.warning("Preliminary media request release was not confirmed")
                if not failed:
                    raise
            except Exception:
                # Cleanup diagnostics must not expose private request transport
                # or replace the original count/refusal/cancellation.
                logger.warning("Preliminary media request release was not confirmed")

    async def get_measured_request_view(self, *, provider, retain_contents, count_view, fit_output=None):
        view = await self._view()
        attempt = await count_view(view)
        if not isinstance(attempt, dict) or "dispatch" not in attempt:
            raise TypeError("count_view must return {dispatch, budget_decision}")
        measurement = self._measured_budget_decision(attempt)
        fallback_rows = await require_media_budget(provider, attempt["dispatch"], attempt.get("budget_decision"),
                                                   messages=view, request_options=attempt.get("request_options"))
        decision = attempt.get("budget_decision")
        count_calls = 1
        outcome = "unchanged"
        if decision and decision["estimated_input_tokens"] > decision["input_limit_tokens"]:
            # Output reserve relief may change generation options, never input.
            fitted = await self._fit_measured_output(fit_output, view, attempt) if fit_output else None
            if fitted:
                attempt, measurement, extra = fitted
                decision = attempt["budget_decision"]
                fallback_rows = await require_media_budget(provider, attempt["dispatch"], decision,
                                                           messages=view, request_options=attempt.get("request_options"))
                count_calls += extra
                outcome = "reduced_output"
            if decision["estimated_input_tokens"] > decision["input_limit_tokens"]:
                self._oversize(decision["estimated_input_tokens"], decision["input_limit_tokens"])
        if decision is None:
            budget = self._calculate_budget(None, provider)
            count = self._estimate_tokens(fallback_rows if fallback_rows is not None else public_messages(view))
            if count > budget:
                self._oversize(count, budget)
        else:
            budget = decision["input_limit_tokens"]
        return {"base_view": view, "final_attempt": attempt, "outcome": outcome,
                "measured_before": measurement[0] if measurement else None,
                "measured_after": measurement[0] if measurement else None,
                "policy_budget": budget, "trigger": budget, "target": budget,
                "count_calls": count_calls, "transaction": None}
