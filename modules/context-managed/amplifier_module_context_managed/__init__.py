"""
Context manager with LLM-powered rolling summaries and persistent transcript.

Implements the ContextManager protocol with:
  - Persistent JSONL transcript (every message written to disk)
  - Rolling LLM summaries (Phase 2)
  - Budget-aware tracking with three thresholds
  - Cache-friendly message ordering
  - Session resume from transcript

Phase 1: Core infrastructure (persistence, budget, protocol compliance)
Phase 2: Summarization engine (async LLM summaries, tier management)
Phase 3: Transcript tool + bundle integration
"""

__amplifier_module_type__ = "context"

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._text_estimate import estimate_messages

logger = logging.getLogger(__name__)

# Format version for transcript.jsonl header
TRANSCRIPT_FORMAT_VERSION = "1.0.0"


@dataclass
class SummaryResult:
    """Result of a single LLM summarization call (Phase 2).

    Captures the output text along with the turn range and message range
    that were summarized, plus how many compression passes were applied.

    ``offset_at_creation`` records ``_transcript_message_offset`` at the
    moment ``_perform_summarization()`` computed the absolute boundary.
    The swap logic in ``get_messages_for_request()`` compares this against
    the *current* offset to detect stale boundaries: if the offset has
    grown (a prior swap or compaction ran between trigger and swap), the
    summary is discarded gracefully rather than producing a negative local
    index.
    """

    summary_text: str
    turn_range: tuple[int, int]
    source_message_range: tuple[int, int]
    compression_passes: int = 1
    offset_at_creation: int = 0  # _transcript_message_offset when boundary was computed


@dataclass
class SummaryTier:
    """A finalized summary tier stored in the context window (Phase 2).

    Represents a committed summary block inserted into the assembled message
    list, including a token estimate for budget tracking.
    """

    content: str
    turn_range: tuple[int, int]
    source_message_range: tuple[int, int]
    compression_passes: int
    token_estimate: int


DEFAULT_SUMMARIZATION_PROMPT = """\
Produce a compact summary of the conversation so far. Use the following sections:

## User Requests & Decisions
List the key requests made by the user and any important decisions reached.

## Files Examined or Modified
List files that were read, analyzed, or modified during the conversation.

## Errors Encountered & Resolutions
Describe any errors, failures, or unexpected behavior encountered, and how they were resolved.

## Current Task State
Describe the current state of work — what has been completed, what is in progress, and what remains.

## Key Technical Details
Note any important technical constraints, patterns, configurations, or implementation details
discovered during the conversation.

## Guidelines
- Be factual and concise. Do not speculate beyond what the conversation contains.
- Preserve numeric values, file paths, error messages, and command outputs exactly.
- Each section may be omitted if there is nothing to report for it.

Note: Use the read_transcript tool to retrieve verbatim content from a specific turn range
when exact wording, full error output, or code details are needed.
"""


async def mount(coordinator: Any, config: dict[str, Any] | None = None):
    """
    Mount the context-managed context manager.

    Args:
        coordinator: Module coordinator
        config: Optional configuration (see docs/CONFIGURATION.md)

    Returns:
        Optional cleanup function
    """
    config = config or {}
    if config.get("engine", "legacy") == "boundary":
        from .boundary import mount_boundary
        return await mount_boundary(coordinator, config)

    # Resolve session directory using coordinator.session_id + CLI slug algorithm.
    # AmplifierSession has no session_dir attribute — the correct approach is to
    # reconstruct the path the same way the CLI does (path-based slug, not hash).
    session_dir = None
    session_id = getattr(coordinator, "session_id", None)
    if session_id and isinstance(session_id, str):
        # Get working_dir: registered capability for child sessions, CWD for root
        working_dir = coordinator.get_capability("session.working_dir")
        cwd = Path(working_dir).resolve() if working_dir else Path.cwd().resolve()

        # CLI's slug algorithm (from amplifier_app_cli/project_utils.py)
        slug = str(cwd).replace("/", "-").replace("\\", "-").replace(":", "")
        if not slug.startswith("-"):
            slug = "-" + slug

        session_dir = (
            Path.home() / ".amplifier" / "projects" / slug / "sessions" / session_id
        )
        logger.debug(f"Resolved session_dir: {session_dir}")

    context = ManagedContextManager(
        max_tokens=config.get("max_tokens", 200_000),
        verbatim_window_tokens=config.get("verbatim_window_tokens", 40_000),
        max_summary_tiers=config.get("max_summary_tiers", 3),
        summary_target_tokens=config.get("summary_target_tokens", 1_500),
        summarization_model=config.get("summarization_model"),
        summarization_prompt_path=config.get("summarization_prompt_path"),
        summarization_retries_before_fallback=config.get(
            "summarization_retries_before_fallback", 3
        ),
        emergency_target_usage=config.get("emergency_target_usage", 0.50),
        large_result_threshold=config.get("large_result_threshold", 50_000),
        pressure_warning=config.get("pressure_warning", 0.70),
        summarize_trigger=config.get("summarize_trigger", 0.60),
        emergency_fallback=config.get("emergency_fallback", 0.92),
        hooks=getattr(coordinator, "hooks", None),
        session_dir=session_dir,
    )

    # Register custom events for auto-discovery by hooks-logging.
    # hooks-logging checks "observability.events" during its own mount() and
    # registers handlers for every event listed there.  Our module mounts
    # before hooks (module order: context → providers → tools → hooks), so
    # registering here ensures the events are visible when hooks-logging
    # mounts.  Without this, our context:* events fire into the void because
    # no handler is ever subscribed for them.
    existing_events: list[str] = list(
        coordinator.get_capability("observability.events") or []
    )
    our_events = [
        "context:budget_pressure",
        "context:pre_summarize",
        "context:post_summarize",
        "context:compaction",
    ]
    coordinator.register_capability(
        "observability.events",
        existing_events + our_events,
    )

    # Register transcript path (namespaced key) for the transcript tool to discover
    transcript_path = context.transcript_path
    if transcript_path is not None:
        coordinator.register_capability(
            "context-managed.transcript_path", str(transcript_path)
        )

    # Attempt to load existing transcript for session resume
    await context._load_from_transcript()

    await coordinator.mount("context", context)
    logger.info("Mounted ManagedContextManager")
    return


class ManagedContextManager:
    """
    Context manager with persistent transcript and budget-aware tracking.

    Implements the full ContextManager protocol. Every message is persisted
    to a JSONL transcript file. Budget is tracked continuously with three
    thresholds: summarize_trigger (0.60), pressure_warning (0.70), and
    emergency_fallback (0.92).

    Threshold cascade (in firing order):
      summarize_trigger (0.60) — start async LLM summarization with ample
          headroom for the ~50 s completion window during which the orchestrator
          continues adding messages.
      pressure_warning (0.70) — emit context:budget_pressure event; by this
          point summarization should already be in-flight.
      emergency_fallback (0.92) — mechanical fallback if LLM summarization
          is still in-flight or has failed.
      inline compact (1.00) — hard cap enforced in get_messages_for_request();
          the returned list will never exceed the token budget.

    Phase 1: Persistence, budget tracking, protocol compliance.
    Phase 2 adds: async LLM summarization, tier management, swap mechanic.
    """

    def __init__(
        self,
        max_tokens: int = 200_000,
        verbatim_window_tokens: int = 40_000,
        max_summary_tiers: int = 3,
        summary_target_tokens: int = 1_500,
        summarization_model: str | None = None,
        summarization_prompt_path: str | None = None,
        summarization_retries_before_fallback: int = 3,
        emergency_target_usage: float = 0.50,
        large_result_threshold: int = 50_000,
        pressure_warning: float = 0.70,
        summarize_trigger: float = 0.60,
        emergency_fallback: float = 0.92,
        hooks: Any = None,
        session_dir: str | Path | None = None,
    ):
        # Config
        self.max_tokens = max_tokens
        self.verbatim_window_tokens = verbatim_window_tokens
        self.max_summary_tiers = max_summary_tiers
        self.summary_target_tokens = summary_target_tokens
        self.summarization_model = summarization_model
        self.summarization_prompt_path = summarization_prompt_path
        self.summarization_retries_before_fallback = (
            summarization_retries_before_fallback
        )
        self.emergency_target_usage = emergency_target_usage
        self.large_result_threshold = large_result_threshold
        self.pressure_warning = pressure_warning
        self.summarize_trigger = summarize_trigger
        self.emergency_fallback = emergency_fallback
        self._hooks = hooks

        # Storage
        self._session_dir = Path(session_dir) if session_dir else None
        self._messages: list[dict[str, Any]] = []
        self._message_index: int = 0  # Running index for large result filenames
        self._running_token_estimate: int = 0
        # Stable-prefix estimate (system prompt tokens) updated on every
        # get_messages_for_request() call and used by _check_summarization_trigger()
        # to compute the *effective* API-level usage fraction.  Without this,
        # the ~48 K stable prefix (system prompt + tool defs) is invisible to
        # threshold checks, making every threshold appear ~24 % higher than
        # intended.
        self._stable_prefix_estimate: int = 0
        self._loaded_from_transcript: bool = False

        # System prompt factory
        self._system_prompt_factory: Callable[[], Awaitable[str]] | None = None

        # Budget tracking
        self._pressure_emitted: bool = (
            False  # Prevent duplicate pressure events per growth
        )

        # Phase 2 placeholders
        self._summary_tiers: list[SummaryTier] = []
        self._is_summarizing: bool = False
        self._pending_summary: SummaryResult | None = None
        self._summarization_failures: int = 0
        self._cached_provider: Any = None

        # Phase 2 tracking fields
        self._current_turn: int = 0
        self._summarized_through_turn: int = 0
        self._transcript_message_offset: int = 0
        self._summarization_task: asyncio.Task[None] | None = None

    @property
    def transcript_path(self) -> Path | None:
        """Path to the transcript JSONL file, or None if no session dir.

        Files are stored under a ``context-managed/`` subdirectory within the
        session directory to avoid collisions with the CLI's own
        ``transcript.jsonl`` (written by SessionStore) and with other modules.
        Follows the ``context-intelligence/`` precedent for module-owned storage.
        """
        if self._session_dir is None:
            return None
        return self._session_dir / "context-managed" / "transcript.jsonl"

    @property
    def tool_results_dir(self) -> Path | None:
        """Path to the large tool results directory, or None if no session dir.

        Stored under ``context-managed/tool_results/`` alongside the transcript.
        """
        if self._session_dir is None:
            return None
        return self._session_dir / "context-managed" / "tool_results"

    # ── Protocol Methods ──────────────────────────────────────────────────────

    async def add_message(self, message: dict[str, Any]) -> None:
        """Add a message to the context.

        Persists to transcript.jsonl, injects timestamp, handles large tool
        results, updates running token estimate, and checks thresholds.
        """
        # Increment turn counter for user messages (Phase 2)
        if message.get("role") == "user":
            self._current_turn += 1

        # Inject timestamp if not present (same pattern as context-simple)
        existing_meta = message.get("metadata") or {}
        if "timestamp" not in existing_meta:
            message = {
                **message,
                "metadata": {
                    **existing_meta,
                    "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                },
            }

        # Handle large tool results
        message = self._handle_large_result(message)

        # Store in memory
        self._messages.append(message)
        self._message_index += 1

        # Update running token estimate
        msg_tokens = self._estimate_tokens_single(message)
        self._running_token_estimate += msg_tokens

        # Persist to disk
        self._append_to_transcript(message)

        logger.debug(
            f"Added message: {message.get('role', 'unknown')} - "
            f"{len(self._messages)} total messages, "
            f"{self._running_token_estimate:,} tokens"
        )

        # Check summarization threshold (Phase 2)
        await self._check_summarization_trigger()

    async def get_messages_for_request(
        self,
        token_budget: int | None = None,
        provider: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Assemble messages for an LLM request.

        Budget resolution: explicit token_budget > provider.get_model_info() >
        max_tokens fallback.

        Assembly order (cache-friendly):
        1. System prompt (stable) — cache hint
        2. Summary tiers (Phase 2, change infrequently) — cache hint
        3. Verbatim conversation (append-only)
        """
        # Cache provider reference for Phase 2 summarization
        if provider is not None:
            self._cached_provider = provider

        budget = self._calculate_budget(token_budget, provider)

        # Build system prompt
        system_message = None
        if self._system_prompt_factory:
            system_content = await self._system_prompt_factory()
            system_message = {
                "role": "system",
                "content": system_content,
                "metadata": {"cache_hint": "breakpoint"},
            }

        # Filter conversation messages (exclude stored system messages when factory is set)
        if self._system_prompt_factory:
            conversation_messages = [
                msg
                for msg in self._messages
                if msg.get("role") != "system"
                or (msg.get("metadata") or {}).get("source") == "hook"
            ]
        else:
            conversation_messages = list(self._messages)

        # Assemble: system + summaries (Phase 2) + verbatim
        assembled: list[dict[str, Any]] = []

        if system_message:
            assembled.append(system_message)
        elif self._messages and self._messages[0].get("role") == "system":
            # Use stored system message if no factory
            assembled.append(self._messages[0])
            conversation_messages = conversation_messages[1:]

        # Phase 2: perform pending summary swap if available
        if self._pending_summary is not None:
            pending = self._pending_summary

            # Check for offset drift: if _transcript_message_offset grew between
            # when _perform_summarization() computed the boundary and now, a prior
            # swap or compaction removed the messages this summary was targeting.
            # Attempting the swap would produce a negative start_local (the original
            # "start_local=-38" bug).  Discard gracefully; _summarization_failures
            # is NOT incremented — the summary itself was valid, just stale.
            offset_drift = self._transcript_message_offset - pending.offset_at_creation
            if offset_drift > 0:
                logger.info(
                    "Discarding stale summary (offset drifted by %d): "
                    "boundary was %s, current offset is %d",
                    offset_drift,
                    pending.source_message_range,
                    self._transcript_message_offset,
                )
            else:
                abs_start, abs_end = pending.source_message_range
                start_local = abs_start - self._transcript_message_offset
                end_local = abs_end - self._transcript_message_offset

                if start_local >= 0 and end_local <= len(self._messages):
                    # Valid boundary: perform the swap
                    old_tokens = self._estimate_tokens(
                        self._messages[start_local:end_local]
                    )
                    token_estimate = self._estimate_tokens_single(
                        {"role": "system", "content": pending.summary_text}
                    )
                    tier = SummaryTier(
                        content=pending.summary_text,
                        turn_range=pending.turn_range,
                        source_message_range=pending.source_message_range,
                        compression_passes=pending.compression_passes,
                        token_estimate=token_estimate,
                    )
                    self._summary_tiers.append(tier)
                    self._messages = (
                        self._messages[:start_local] + self._messages[end_local:]
                    )
                    self._transcript_message_offset += end_local - start_local
                    self._running_token_estimate = (
                        self._running_token_estimate - old_tokens + token_estimate
                    )
                    self._summarized_through_turn = pending.turn_range[1]
                    self._summarization_failures = 0

                    # Persist summary marker to transcript
                    self._persist_summary_marker(tier)

                    # Phase 2: merge oldest tiers if count exceeds limit
                    if len(self._summary_tiers) > self.max_summary_tiers:
                        try:
                            await self._merge_oldest_tiers()
                        except Exception as e:
                            logger.warning(f"Tier merge failed: {e}")

                    # Rebuild conversation_messages after swap
                    if self._system_prompt_factory:
                        conversation_messages = [
                            msg
                            for msg in self._messages
                            if msg.get("role") != "system"
                            or (msg.get("metadata") or {}).get("source") == "hook"
                        ]
                    else:
                        conversation_messages = list(self._messages)
                        # Re-apply system message slice if stored system already in assembled
                        if (
                            not system_message
                            and self._messages
                            and self._messages[0].get("role") == "system"
                        ):
                            conversation_messages = conversation_messages[1:]
                else:
                    logger.warning(
                        f"Discarding pending summary with invalid boundary: "
                        f"start_local={start_local}, end_local={end_local}, "
                        f"messages_len={len(self._messages)}"
                    )

            self._pending_summary = None

        # Insert summary tiers (Phase 2) — only the last tier gets cache hint
        for i, tier in enumerate(self._summary_tiers):
            is_last = i == len(self._summary_tiers) - 1
            metadata: dict[str, Any] = {
                "type": "context_managed_summary",
                "turn_range": list(tier.turn_range),
                "compression_passes": tier.compression_passes,
            }
            if is_last:
                metadata["cache_hint"] = "breakpoint"
            assembled.append(
                {
                    "role": "system",
                    "content": tier.content,
                    "metadata": metadata,
                }
            )

        assembled.extend(conversation_messages)

        # Check budget and emit pressure event
        system_tokens = self._estimate_tokens(assembled[:1]) if assembled else 0
        # Cache the system-prompt token count so _check_summarization_trigger() can
        # include it in the effective-usage fraction.  This is the only place in the
        # code that computes the rendered system prompt size, so we update the shared
        # field here and use it across all threshold checks.
        self._stable_prefix_estimate = system_tokens
        available = budget - system_tokens
        conversation_tokens = self._running_token_estimate
        if available > 0:
            usage_fraction = conversation_tokens / available
            if usage_fraction >= self.pressure_warning and not self._pressure_emitted:
                self._pressure_emitted = True
                await self._emit_event(
                    "context:budget_pressure",
                    {
                        "usage_fraction": usage_fraction,
                        "token_count": conversation_tokens,
                        "estimate_scope": (
                            "text_only"
                            if estimate_messages(assembled).has_unmeasured_images
                            else "all_text"
                        ),
                        "budget": available,
                    },
                )

        # Text-budget enforcement; unmeasured images require provider validation.
        # The async LLM summarization is the preferred path (produces proper summaries),
        # but this is the last gate before the provider call.  It operates on the
        # assembled view so self._messages and self._running_token_estimate are left
        # untouched; the async summary continues and will swap in on the next turn.
        total_tokens = self._estimate_tokens(assembled)
        if total_tokens > budget:
            logger.warning(
                "Assembled messages (%s tokens) exceed budget (%s). "
                "Applying inline compaction.",
                f"{total_tokens:,}",
                f"{budget:,}",
            )
            assembled = await self._inline_compact(assembled, budget)

        return assembled

    async def get_messages(self) -> list[dict[str, Any]]:
        """Return full raw conversation history.

        Returns every original message as passed to add_message(), unmodified.
        Summary markers are excluded (they are implementation detail).
        This is the lossless accessor for SessionStore and external tools.

        If a transcript file exists, reads from it for completeness.
        Otherwise returns in-memory messages.
        """
        if self.transcript_path and self.transcript_path.exists():
            return self._read_transcript_messages()
        return list(self._messages)

    async def set_messages(self, messages: list[dict[str, Any]]) -> None:
        """Set messages from external source (kernel SessionStore).

        Two paths:
        - If already loaded from own transcript: NO-OP (file-based ignore pattern)
        - If fresh session (no transcript): accept and initialize from messages
        """
        if self._loaded_from_transcript:
            logger.info(
                "set_messages: NO-OP — already loaded from own transcript "
                f"({len(self._messages)} messages)"
            )
            return

        self._messages = list(messages)
        self._message_index = len(messages)
        self._running_token_estimate = self._estimate_tokens(messages)
        self._pressure_emitted = False
        logger.info(f"set_messages: initialized from {len(messages)} external messages")

    async def set_system_prompt_factory(
        self, factory: Callable[[], Awaitable[str]]
    ) -> None:
        """Set a factory function that produces fresh system prompt content.

        The factory is called on EVERY get_messages_for_request() invocation,
        enabling dynamic content like @mentions to be re-processed each turn.
        """
        self._system_prompt_factory = factory
        logger.info("System prompt factory registered — will refresh on each request")

    async def clear(self) -> None:
        """Archive current transcript and reset all state.

        Archives transcript.jsonl to transcript.{timestamp}.archived.jsonl,
        archives tool_results/ directory, then resets all in-memory state.
        """
        await self._archive_transcript()

        # Cancel any in-flight summarization task
        if self._summarization_task is not None and not self._summarization_task.done():
            self._summarization_task.cancel()
        self._summarization_task = None

        # Reset all in-memory state
        self._messages = []
        self._message_index = 0
        self._running_token_estimate = 0
        self._stable_prefix_estimate = 0
        self._loaded_from_transcript = False
        self._pressure_emitted = False
        self._summary_tiers = []
        self._is_summarizing = False
        self._pending_summary = None
        self._summarization_failures = 0

        # Reset Phase 2 tracking fields
        self._current_turn = 0
        self._summarized_through_turn = 0
        self._transcript_message_offset = 0

        # Write fresh transcript header
        self._write_transcript_header()

        logger.info("Context cleared — transcript archived, state reset")

    # ── Budget Calculation ────────────────────────────────────────────────────

    def _calculate_budget(self, token_budget: int | None, provider: Any | None) -> int:
        """Calculate effective token budget.

        Priority: explicit token_budget > provider.get_model_info() >
        provider.get_info().defaults > self.max_tokens fallback.

        self.max_tokens acts as a **ceiling** on any provider-derived budget.
        Even if the provider reports a very large context window (e.g. 1 M tokens
        with the Claude extended-context beta), the returned budget will never
        exceed self.max_tokens.  This ensures that the summarization thresholds
        (pressure_warning / summarize_trigger / emergency_fallback) fire at the
        token count the operator configured, not at some fraction of the provider's
        physical limit.

        Same logic as context-simple's _calculate_budget(), extended with the
        max_tokens ceiling.
        """
        if token_budget is not None:
            logger.debug(f"Using explicit token_budget: {token_budget}")
            return token_budget

        safety_margin = 4096
        output_reserve_fraction = 0.5

        if provider is not None:
            try:
                if hasattr(provider, "get_model_info"):
                    model_info = provider.get_model_info()
                    if model_info:
                        context_window = getattr(model_info, "context_window", None)
                        max_output = getattr(model_info, "max_output_tokens", None)
                        if context_window and max_output:
                            reserved_output = int(max_output * output_reserve_fraction)
                            budget = context_window - reserved_output - safety_margin
                            capped = min(budget, self.max_tokens)
                            logger.info(
                                f"Budget from provider model info: {budget:,} "
                                f"(context={context_window:,}, "
                                f"reserved_output={reserved_output:,}) "
                                f"→ capped at max_tokens={capped:,}"
                            )
                            return capped

                info = provider.get_info()
                defaults = info.defaults or {}
                context_window = defaults.get("context_window")
                max_output_tokens = defaults.get("max_output_tokens")

                if context_window and max_output_tokens:
                    reserved_output = int(max_output_tokens * output_reserve_fraction)
                    budget = context_window - reserved_output - safety_margin
                    capped = min(budget, self.max_tokens)
                    logger.info(
                        f"Budget from provider defaults: {budget:,} "
                        f"→ capped at max_tokens={capped:,}"
                    )
                    return capped
            except Exception as e:
                logger.debug(f"Could not get budget from provider: {e}")

        logger.info(f"Using fallback max_tokens budget: {self.max_tokens:,}")
        return self.max_tokens

    # ── Token Estimation ──────────────────────────────────────────────────────

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Text estimate; typed image cost remains unknown to this heuristic."""
        return estimate_messages(messages).tokens

    def _estimate_tokens_single(self, message: dict[str, Any]) -> int:
        """Rough token estimation for a single message."""
        return estimate_messages([message]).tokens

    # ── Persistence ───────────────────────────────────────────────────────────

    def _write_transcript_header(self) -> None:
        """Write the format version header to a fresh transcript file."""
        if self.transcript_path is None:
            return
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "type": "transcript_header",
            "format_version": TRANSCRIPT_FORMAT_VERSION,
            "created_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        }
        with open(self.transcript_path, "w") as f:
            f.write(json.dumps(header) + "\n")

    def _append_to_transcript(self, message: dict[str, Any]) -> None:
        """Append a single message to the transcript JSONL file."""
        if self.transcript_path is None:
            return

        # Ensure file exists with header
        if not self.transcript_path.exists():
            self._write_transcript_header()

        with open(self.transcript_path, "a") as f:
            f.write(json.dumps(message) + "\n")

    def _read_transcript_messages(self) -> list[dict[str, Any]]:
        """Read all conversation messages from transcript.jsonl.

        Skips the header and any summary markers (metadata.type ==
        'context_managed_summary'). Returns raw conversation messages only.
        This is the lossless accessor for SessionStore and external tools.
        """
        return [
            r
            for r in self._read_all_transcript_records()
            if (r.get("metadata") or {}).get("type") != "context_managed_summary"
        ]

    def _read_all_transcript_records(self) -> list[dict[str, Any]]:
        """Read all records from transcript.jsonl, skipping only the header.

        Unlike _read_transcript_messages(), this includes summary markers.
        Returns all non-header records as a list of dicts.
        """
        if self.transcript_path is None or not self.transcript_path.exists():
            return []

        records = []
        with open(self.transcript_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed transcript line: {line[:100]}")
                    continue

                # Skip only the header
                if record.get("type") == "transcript_header":
                    continue

                records.append(record)

        return records

    async def _load_from_transcript(self) -> None:
        """Load state from existing transcript file (session resume).

        Reads all records, separates summary markers from conversation messages,
        reconstructs SummaryTier objects, and restores the verbatim window.
        Sets _loaded_from_transcript = True so set_messages() becomes a NO-OP.
        """
        if self.transcript_path is None or not self.transcript_path.exists():
            return

        try:
            all_records = self._read_all_transcript_records()
            if not all_records:
                return

            # Separate summary markers from conversation messages
            summary_markers = [
                r
                for r in all_records
                if (r.get("metadata") or {}).get("type") == "context_managed_summary"
            ]
            conversation_messages = [
                r
                for r in all_records
                if (r.get("metadata") or {}).get("type") != "context_managed_summary"
            ]

            # Reconstruct SummaryTier objects from markers with valid fields
            tiers: list[SummaryTier] = []
            for marker in summary_markers:
                meta = marker.get("metadata") or {}
                try:
                    turn_range_raw = meta["turn_range"]
                    source_range_raw = meta["source_message_range"]
                    tier = SummaryTier(
                        content=marker["content"],
                        turn_range=tuple(turn_range_raw),
                        source_message_range=tuple(source_range_raw),
                        compression_passes=meta.get("compression_passes", 1),
                        token_estimate=meta.get("token_estimate", 0),
                    )
                    tiers.append(tier)
                except (KeyError, TypeError) as e:
                    logger.warning(
                        f"Skipping malformed summary marker during resume: {e}"
                    )
                    continue

            if tiers:
                # Set verbatim window to messages after the last summarized end
                last_summarized_end = max(
                    tier.source_message_range[1] for tier in tiers
                )
                self._messages = conversation_messages[last_summarized_end:]
                self._transcript_message_offset = last_summarized_end
                self._summarized_through_turn = max(
                    tier.turn_range[1] for tier in tiers
                )
            else:
                self._messages = conversation_messages
                self._transcript_message_offset = 0
                self._summarized_through_turn = 0

            self._summary_tiers = tiers
            self._message_index = len(conversation_messages)
            self._running_token_estimate = sum(
                tier.token_estimate for tier in tiers
            ) + self._estimate_tokens(self._messages)
            user_in_verbatim = sum(
                1 for msg in self._messages if msg.get("role") == "user"
            )
            self._current_turn = self._summarized_through_turn + user_in_verbatim
            self._loaded_from_transcript = True
            self._pressure_emitted = False

            logger.info(
                f"Resumed from transcript: {len(self._messages)} verbatim messages, "
                f"{len(tiers)} summary tier(s), "
                f"~{self._running_token_estimate:,} tokens"
            )
        except Exception as e:
            logger.warning(f"Failed to load transcript, starting fresh: {e}")
            self._messages = []
            self._message_index = 0
            self._running_token_estimate = 0
            self._summary_tiers = []
            self._transcript_message_offset = 0
            self._summarized_through_turn = 0

    def _persist_summary_marker(self, tier: "SummaryTier") -> None:
        """Write a summary marker to the transcript JSONL file.

        Builds a marker dict with role='system', content=tier.content,
        and metadata containing all tier fields (type, turn_range as list,
        source_message_range as list, compression_passes, token_estimate).
        Appends the marker to the transcript via _append_to_transcript().
        """
        marker: dict[str, Any] = {
            "role": "system",
            "content": tier.content,
            "metadata": {
                "type": "context_managed_summary",
                "turn_range": list(tier.turn_range),
                "source_message_range": list(tier.source_message_range),
                "compression_passes": tier.compression_passes,
                "token_estimate": tier.token_estimate,
            },
        }
        self._append_to_transcript(marker)

    async def _archive_transcript(self) -> None:
        """Archive current transcript and tool results."""
        if self.transcript_path is None or not self.transcript_path.exists():
            return

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

        # Archive transcript
        archive_name = f"transcript.{timestamp}.archived.jsonl"
        archive_path = self.transcript_path.parent / archive_name
        self.transcript_path.rename(archive_path)
        logger.info(f"Archived transcript to {archive_name}")

        # Archive tool_results/ if it exists
        if self.tool_results_dir and self.tool_results_dir.exists():
            archive_dir = (
                self.tool_results_dir.parent / f"tool_results.{timestamp}.archived"
            )
            self.tool_results_dir.rename(archive_dir)
            logger.info(f"Archived tool_results to tool_results.{timestamp}.archived")

    # ── Large Result Handling ─────────────────────────────────────────────────

    def _handle_large_result(self, message: dict[str, Any]) -> dict[str, Any]:
        """Store large tool results to pointer files, truncate in transcript.

        If a tool message's content exceeds large_result_threshold characters,
        the full result is written to a separate file and the message content
        is truncated with a pointer to the full file.
        """
        if message.get("role") != "tool":
            return message

        content = message.get("content", "")
        if not isinstance(content, str) or len(content) <= self.large_result_threshold:
            return message

        if self.tool_results_dir is None:
            return message

        # Write full result to pointer file
        tool_call_id = message.get("tool_call_id", "unknown")
        filename = f"{self._message_index}_{tool_call_id}.txt"
        self.tool_results_dir.mkdir(parents=True, exist_ok=True)
        full_path = self.tool_results_dir / filename

        with open(full_path, "w") as f:
            f.write(content)

        # Truncate content and add pointer
        truncated_content = content[: self.large_result_threshold]
        truncated_content += (
            f"\n\n[truncated: {len(content):,} chars total — full result at {filename}]"
        )

        existing_meta = message.get("metadata") or {}
        return {
            **message,
            "content": truncated_content,
            "metadata": {
                **existing_meta,
                "full_result_path": str(full_path),
                "original_length": len(content),
            },
        }

    # ── Event Emission ────────────────────────────────────────────────────────

    async def _emit_event(self, event: str, data: dict[str, Any]) -> None:
        """Emit an event via hooks if available."""
        if self._hooks is not None:
            try:
                await self._hooks.emit(event, data)
            except Exception as e:
                logger.warning(f"Could not emit {event}: {e}")

    # ── Summarization Helpers ─────────────────────────────────────────────────

    def _get_summarization_prompt(self) -> str:
        """Return the summarization prompt.

        Returns the prompt read from file if summarization_prompt_path is
        configured and the file exists. Falls back to DEFAULT_SUMMARIZATION_PROMPT
        on OSError/FileNotFoundError, logging a warning.
        """
        if self.summarization_prompt_path:
            try:
                return Path(self.summarization_prompt_path).read_text()
            except (OSError, FileNotFoundError) as e:
                logger.warning(
                    f"Could not read summarization prompt from "
                    f"{self.summarization_prompt_path}: {e}. "
                    "Falling back to DEFAULT_SUMMARIZATION_PROMPT."
                )
        return DEFAULT_SUMMARIZATION_PROMPT

    def _calculate_segment_boundary(self) -> tuple[int, int] | None:
        """Calculate the boundary for message segmentation.

        Returns None if _messages is empty or _running_token_estimate <=
        verbatim_window_tokens.

        Otherwise calculates excess_tokens = _running_token_estimate -
        verbatim_window_tokens and accumulates message tokens from index 0
        until accumulated >= excess_tokens. Calls _snap_to_tool_pair_boundary()
        before returning.

        Returns:
            Tuple (0, end_idx) as indices into self._messages (end_idx
            exclusive), or None.
        """
        if not self._messages:
            return None

        if self._running_token_estimate <= self.verbatim_window_tokens:
            return None

        excess_tokens = self._running_token_estimate - self.verbatim_window_tokens

        accumulated = 0
        end_idx = 0
        for i, msg in enumerate(self._messages):
            accumulated += self._estimate_tokens_single(msg)
            end_idx = i + 1
            if accumulated >= excess_tokens:
                break

        end_idx = self._snap_to_tool_pair_boundary(end_idx)

        return (0, end_idx)

    async def _perform_summarization(self, boundary: tuple[int, int]) -> SummaryResult:
        """Perform LLM summarization over the given message boundary.

        Slices messages, builds a ChatRequest with the summarization prompt and
        formatted conversation, calls the cached provider, and returns a
        SummaryResult with absolute source_message_range (offset by
        _transcript_message_offset) and a turn_range derived from
        _summarized_through_turn.
        """
        from amplifier_core import ChatRequest, Message

        start, end = boundary
        messages_to_summarize = self._messages[start:end]

        prompt = self._get_summarization_prompt()
        formatted_conversation = self._format_messages_for_summarization(
            messages_to_summarize
        )

        # Calculate turn range
        turn_start = self._summarized_through_turn + 1
        user_count = sum(
            1 for msg in messages_to_summarize if msg.get("role") == "user"
        )
        turn_end = self._summarized_through_turn + user_count

        # Calculate absolute source_message_range
        abs_start = self._transcript_message_offset + start
        abs_end = self._transcript_message_offset + end
        source_message_range = (abs_start, abs_end)

        # Build and send the ChatRequest
        request = ChatRequest(
            messages=[
                Message(role="system", content=prompt),
                Message(role="user", content=formatted_conversation),
            ],
            model=self.summarization_model,
        )

        response = await self._cached_provider.complete(request)

        summary_text = self._extract_text_from_response(response)

        return SummaryResult(
            summary_text=summary_text,
            turn_range=(turn_start, turn_end),
            source_message_range=source_message_range,
            offset_at_creation=self._transcript_message_offset,
        )

    def _format_messages_for_summarization(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for the summarization prompt.

        Each message is formatted as '[role]: content'.  When content is a
        list of content blocks, text blocks and tool_call blocks are both
        included.  Thinking blocks are silently skipped.  Tool call inputs
        are compacted to at most 500 chars.  Tool-role messages are formatted
        as '[tool_result for {id}]: {content}' so the summarizer can link
        results back to the calls that requested them.

        The ``tool_calls`` field (separate from content blocks) is also
        included for any calls not already shown via content blocks.
        """
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # IDs of tool calls already shown via content blocks (avoid duplication)
            shown_tc_ids: set[str] = set()

            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "text":
                            text = block.get("text", "")
                            if text:
                                parts.append(text)
                        elif block_type in ("tool_call", "tool_use"):
                            name = block.get("name", "unknown_tool")
                            inp = block.get("input", {})
                            tc_id = block.get("id", "")
                            if tc_id:
                                shown_tc_ids.add(tc_id)
                            inp_str = json.dumps(inp, separators=(",", ":"))
                            if len(inp_str) > 500:
                                inp_str = inp_str[:500] + "..."
                            parts.append(f"[tool_call: {name}({inp_str})]")
                        elif block_type == "thinking":
                            pass  # Skip internal reasoning blocks
                        elif "text" in block:
                            # Fallback: dict has a text key but unrecognised type
                            text = block.get("text", "")
                            if text:
                                parts.append(text)
                    elif hasattr(block, "text"):
                        parts.append(block.text)
                content = "\n".join(parts)

            # For tool result messages: include the tool_call_id for linkage
            if role == "tool":
                tc_id = msg.get("tool_call_id", "")
                msg_line = f"[tool_result for {tc_id}]: {content}"
            else:
                msg_line = f"[{role}]: {content}"

            # Supplement with tool_calls field for any calls not shown via content blocks
            extra_lines: list[str] = []
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id", "")
                if tc_id and tc_id in shown_tc_ids:
                    continue  # already emitted via content block
                # Support both {function: {name, arguments}} and {name/tool, input/arguments}
                if "function" in tc:
                    name = tc["function"].get("name", "unknown_tool")
                    raw_args = tc["function"].get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            inp_str = json.dumps(
                                json.loads(raw_args), separators=(",", ":")
                            )
                        except (json.JSONDecodeError, ValueError):
                            inp_str = raw_args
                    else:
                        inp_str = json.dumps(raw_args, separators=(",", ":"))
                else:
                    name = tc.get("name") or tc.get("tool", "unknown_tool")
                    inp = tc.get("input") or tc.get("arguments") or {}
                    if isinstance(inp, str):
                        inp_str = inp
                    else:
                        inp_str = json.dumps(inp, separators=(",", ":"))
                if len(inp_str) > 500:
                    inp_str = inp_str[:500] + "..."
                extra_lines.append(f"  [tool_call: {name}({inp_str})]")

            if extra_lines:
                lines.append(msg_line + "\n" + "\n".join(extra_lines))
            else:
                lines.append(msg_line)

        return "\n\n".join(lines)

    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text content from a ChatResponse.

        Iterates response.content and joins the .text attribute of every
        block that has one (TextBlock instances).  Blocks without .text are
        silently skipped.
        """
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)

    async def _trigger_summarization(self) -> None:
        """Trigger summarization with guards.

        Guards (in order):
          1. Skip if _is_summarizing is True.
          2. Skip if _cached_provider is None (log debug).
          3. Skip if _calculate_segment_boundary() returns None.

        On pass: sets _is_summarizing=True and creates an asyncio.Task for
        _run_summarization(boundary).
        """
        if self._is_summarizing:
            return
        if self._cached_provider is None:
            logger.debug("Skipping summarization: no cached provider")
            return
        boundary = self._calculate_segment_boundary()
        if boundary is None:
            return
        self._is_summarizing = True
        self._summarization_task = asyncio.create_task(
            self._run_summarization(boundary)
        )

    async def _run_summarization(self, boundary: tuple[int, int]) -> None:
        """Wrapper that runs summarization and handles errors.

        Emits 'context:pre_summarize' before calling _perform_summarization.
        On success, stores result in _pending_summary and emits
        'context:post_summarize' with stats.  On exception: increments
        _summarization_failures and logs a warning (no post_summarize emitted).
        In the finally block: resets _is_summarizing=False and
        _summarization_task=None.
        """
        start, end = boundary
        message_count = end - start
        await self._emit_event(
            "context:pre_summarize",
            {"boundary": boundary, "message_count": message_count},
        )
        try:
            result = await self._perform_summarization(boundary)
            self._pending_summary = result
            await self._emit_event(
                "context:post_summarize",
                {
                    "turn_range": list(result.turn_range),
                    "summary_length": len(result.summary_text),
                    "compression_passes": result.compression_passes,
                },
            )
        except Exception as e:
            self._summarization_failures += 1
            logger.warning(f"Summarization failed: {e}")
        finally:
            self._is_summarizing = False
            self._summarization_task = None

    async def _check_summarization_trigger(self) -> None:
        """Check if summarization should be triggered based on usage fraction.

        Budget and usage fraction are always computed first.

        Emergency checks run **even when summarization is already in-flight**
        to ensure the safety net fires regardless of async timing:

        1. Failure-based emergency: if _summarization_failures >=
           summarization_retries_before_fallback AND usage_fraction >=
           emergency_fallback → call _emergency_mechanical_fallback() and return.

        2. In-flight overshoot emergency: if _is_summarizing is True AND
           usage_fraction >= emergency_fallback → the async LLM call is too
           slow to rescue us.  Cancel the in-flight task, reset _is_summarizing,
           and call _emergency_mechanical_fallback() and return.

        Otherwise: skip if _is_summarizing is True (prevent duplicate triggers).
        Call _trigger_summarization() if usage_fraction >= summarize_trigger.
        """
        budget = self._calculate_budget(None, self._cached_provider)
        if budget <= 0:
            return
        # Include the stable prefix (system prompt tokens, tracked from the most
        # recent get_messages_for_request() call) in the effective usage total.
        # Without this, the ~28-48 K stable prefix is invisible here, making
        # every threshold appear ~24 % higher than intended — e.g. a real 93 %
        # API-level context load computed as only 69 % inside this method.
        effective_usage = self._running_token_estimate + self._stable_prefix_estimate
        usage_fraction = effective_usage / budget

        # Emergency check 1: repeated LLM failures — do mechanical fallback
        if (
            self._summarization_failures >= self.summarization_retries_before_fallback
            and usage_fraction >= self.emergency_fallback
        ):
            await self._emergency_mechanical_fallback()
            return

        # Emergency check 2: summarization is in-flight but we've overshot the
        # emergency threshold.  Cancel the slow LLM call and use the mechanical
        # fallback as a safety net right now.
        if self._is_summarizing and usage_fraction >= self.emergency_fallback:
            logger.warning(
                "Summarization in-flight but usage at %.1f%% "
                "(emergency threshold %.0f%%). Running emergency mechanical fallback.",
                usage_fraction * 100,
                self.emergency_fallback * 100,
            )
            if (
                self._summarization_task is not None
                and not self._summarization_task.done()
            ):
                self._summarization_task.cancel()
            self._is_summarizing = False
            self._summarization_task = None
            await self._emergency_mechanical_fallback()
            return

        # Normal trigger: block duplicates, fire when above summarize_trigger
        if self._is_summarizing:
            return

        if usage_fraction >= self.summarize_trigger:
            await self._trigger_summarization()

    async def _emergency_mechanical_fallback(self) -> None:
        """Emergency mechanical compaction when LLM summarization keeps failing.

        Calculates target_tokens = budget * emergency_target_usage.
        Emits 'context:compaction' event with reason, failures, target_usage,
        current_tokens, and target_tokens.

        Step 1: Truncate large tool results — for each tool message whose string
        content exceeds 1000 chars, truncate to first 500 chars +
        '\\n\\n[truncated by emergency compaction]' and update
        _running_token_estimate.

        Step 2: Remove oldest non-protected messages — while
        _running_token_estimate > target_tokens, find the first removable
        message (not a system message, not a hook-sourced message), pop it and
        subtract its token estimate.  Break if only protected messages remain.
        """
        budget = self._calculate_budget(None, self._cached_provider)
        target_tokens = int(budget * self.emergency_target_usage)
        # Conversation-only target: the stable prefix (system prompt) occupies part of
        # the budget, so the *conversation* portion must fit in (target_tokens - prefix).
        # This prevents the fallback from stopping at a conversation size that, when
        # added to the prefix, still exceeds the total target.
        # Safety floor: always leave at least 25% of target_tokens for conversation so
        # we never over-compact when the prefix estimate is large or stale.
        conversation_target = max(
            target_tokens - self._stable_prefix_estimate,
            target_tokens // 4,
        )

        await self._emit_event(
            "context:compaction",
            {
                "reason": "emergency_mechanical_fallback",
                "failures": self._summarization_failures,
                "target_usage": self.emergency_target_usage,
                "current_tokens": self._running_token_estimate,
                "target_tokens": target_tokens,
                "stable_prefix_estimate": self._stable_prefix_estimate,
                "conversation_target": conversation_target,
            },
        )

        # Step 1: Truncate large tool results
        for i, msg in enumerate(self._messages):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) <= 1000:
                continue
            old_tokens = self._estimate_tokens_single(msg)
            truncated = content[:500] + "\n\n[truncated by emergency compaction]"
            self._messages[i] = {**msg, "content": truncated}
            new_tokens = self._estimate_tokens_single(self._messages[i])
            self._running_token_estimate -= old_tokens - new_tokens

        # Step 2: Remove oldest non-protected messages
        original_count = len(self._messages)
        removed_messages_info: list[dict[str, str]] = []
        while self._running_token_estimate > conversation_target:
            removable_idx = None
            for i, msg in enumerate(self._messages):
                if not self._is_protected_message(msg):
                    removable_idx = i
                    break
            if removable_idx is None:
                break  # Only protected messages remain
            removed = self._messages.pop(removable_idx)
            self._running_token_estimate -= self._estimate_tokens_single(removed)
            removed_messages_info.append(
                {
                    "role": removed.get("role", "unknown"),
                    "content_preview": str(removed.get("content", ""))[:100],
                }
            )

        # Update transcript offset by the number of messages removed.  Without
        # this, any pending summary that was computed before the fallback will
        # have abs_start = (old_offset + local_start).  When the swap later
        # attempts start_local = abs_start - current_offset it gets 0 or a
        # negative number even though offset_drift == 0, because the offset
        # was never incremented to reflect the messages that were popped here.
        removed_count = original_count - len(self._messages)
        if removed_count > 0:
            self._transcript_message_offset += removed_count

        # Step 3: Insert a mechanical summary marker so the model knows what
        # happened.  compression_passes=0 distinguishes mechanical summaries
        # (no LLM call) from LLM-generated summaries (compression_passes >= 1).
        # The marker is appended as a SummaryTier so it is handled consistently
        # by get_messages_for_request() assembly and persisted to the transcript.
        if removed_count > 0:
            role_counts: dict[str, int] = {}
            user_previews: list[str] = []
            for info in removed_messages_info:
                role = info["role"]
                role_counts[role] = role_counts.get(role, 0) + 1
                if role == "user":
                    user_previews.append(info["content_preview"])

            role_summary = ", ".join(
                f"{count} {role}" for role, count in sorted(role_counts.items())
            )
            turn_start = self._summarized_through_turn + 1
            user_count = role_counts.get("user", 0)
            turn_end = self._summarized_through_turn + max(user_count, 1)

            marker_lines: list[str] = [
                f"[Emergency context compaction — {removed_count} messages removed ({role_summary})]",
                "",
                "This is a mechanical summary, not LLM-generated. Details may be incomplete.",
                "",
            ]
            if user_previews:
                marker_lines.append("User topics covered:")
                for preview in user_previews:
                    marker_lines.append(f"  - {preview}")
                marker_lines.append("")
            marker_lines.extend(
                [
                    f"For full details of turns {turn_start}-{turn_end}, use: read_transcript(start_turn={turn_start}, end_turn={turn_end})",
                    "",
                    "Treat this summary as approximate. Verify against the actual codebase before acting on any specifics.",
                ]
            )
            marker_content = "\n".join(marker_lines)

            marker_token_estimate = self._estimate_tokens_single(
                {"role": "system", "content": marker_content}
            )
            tier = SummaryTier(
                content=marker_content,
                turn_range=(turn_start, turn_end),
                source_message_range=(
                    self._transcript_message_offset - removed_count,
                    self._transcript_message_offset,
                ),
                compression_passes=0,  # 0 = mechanical, not LLM-generated
                token_estimate=marker_token_estimate,
            )
            self._summary_tiers.append(tier)
            self._persist_summary_marker(tier)
            self._summarized_through_turn = turn_end
            self._running_token_estimate += marker_token_estimate

        # Invalidate any pending summary: the messages it was summarizing may
        # have been truncated (Step 1) or removed (Step 2) by this fallback,
        # making its stored boundary stale regardless of whether offset_drift
        # is zero.  A fresh summarization cycle will be triggered on the next
        # _check_summarization_trigger() call.
        if self._pending_summary is not None:
            logger.info(
                "Invalidating pending summary after emergency fallback: "
                "messages were modified or removed (removed_count=%d)",
                removed_count,
            )
            self._pending_summary = None

    def _is_protected_message(self, msg: dict[str, Any]) -> bool:
        """Return True if a message should not be removed during emergency compaction.

        Protected messages:
        - System messages (any system message, with or without hook source)
        - Any message with metadata.source == 'hook'
        """
        meta = msg.get("metadata") or {}
        if msg.get("role") == "system":
            return True
        if meta.get("source") == "hook":
            return True
        return False

    async def _inline_compact(
        self, messages: list[dict[str, Any]], budget: int
    ) -> list[dict[str, Any]]:
        """Compact an assembled message list to fit within the budget.

        This is a per-request safety net that operates on a copy of the
        assembled view.  It does **not** modify ``self._messages`` or
        ``self._running_token_estimate`` (the persistent state).  The async
        LLM summarization continues running and will produce a proper summary
        for the next turn; this ensures the *current* request never sees more
        tokens than the budget allows.

        Step 1 — Truncate large tool results: for each tool message whose
        string content exceeds 1 000 chars, truncate to the first 500 chars
        plus ``"\\n\\n[truncated by inline compaction]"``.  The last five tool
        results in the list are protected and not truncated.

        Step 2 — Remove oldest non-protected messages: while the estimated
        token count of ``compacted`` exceeds ``target_tokens``, scan forward
        and remove the first removable block.  Protected = system messages,
        ``metadata.source == "hook"`` messages, or the last user message
        (current intent).  An assistant message that carries ``tool_calls`` is
        removed atomically together with all immediately following ``tool``
        messages so we never leave orphaned tool results.

        Emits a ``context:compaction`` event with
        ``reason="inline_budget_enforcement"``.

        Args:
            messages: The assembled message list (will be shallow-copied).
            budget: The effective token budget for this request.

        Returns:
            A compacted copy of ``messages`` that fits within ``budget``.
        """
        target_tokens = int(budget * self.emergency_target_usage)
        compacted: list[dict[str, Any]] = list(messages)
        original_count = len(compacted)
        original_tokens = self._estimate_tokens(compacted)

        # ── Step 1: truncate large tool results (protect last 5) ──────────
        tool_indices = [i for i, m in enumerate(compacted) if m.get("role") == "tool"]
        protected_from_truncation: set[int] = set(tool_indices[-5:])

        for idx in range(len(compacted)):
            if idx in protected_from_truncation:
                continue
            msg = compacted[idx]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) <= 1000:
                continue
            truncated_content = content[:500] + "\n\n[truncated by inline compaction]"
            compacted[idx] = {**msg, "content": truncated_content}

        # ── Step 2: remove oldest non-protected messages ──────────────────
        while self._estimate_tokens(compacted) > target_tokens:
            # Recompute last user index each iteration as the list shrinks.
            last_user_idx: int | None = next(
                (
                    i
                    for i in range(len(compacted) - 1, -1, -1)
                    if compacted[i].get("role") == "user"
                ),
                None,
            )

            removed = False
            i = 0
            while i < len(compacted):
                msg = compacted[i]
                if self._is_protected_message(msg) or i == last_user_idx:
                    i += 1
                    continue

                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # Remove this assistant message + all consecutive tool
                    # results atomically to avoid orphaned tool results.
                    block_end = i + 1
                    while (
                        block_end < len(compacted)
                        and compacted[block_end].get("role") == "tool"
                    ):
                        block_end += 1
                    del compacted[i:block_end]
                else:
                    # Single non-protected message (user, assistant without
                    # tool_calls, or an orphaned tool result).
                    del compacted[i]

                removed = True
                break

            if not removed:
                # Only protected messages remain; cannot compact further.
                break

        # ── Step 3: insert an ephemeral compaction notice when messages were ──
        # removed so the model knows a gap exists and where to look for details.
        # This notice is NOT persisted and NOT a SummaryTier — it lives only in
        # this assembled view, so it doesn't affect _messages or _summary_tiers.
        messages_removed = original_count - len(compacted)
        if messages_removed > 0:
            notice: dict[str, Any] = {
                "role": "system",
                "content": (
                    f"[Context compacted: {messages_removed} older messages were removed "
                    f"to fit within budget. "
                    f"Use read_transcript to retrieve details from earlier in the conversation. "
                    f"Verify any specifics against the actual codebase.]"
                ),
                "metadata": {"source": "inline_compact", "type": "compaction_notice"},
            }
            # Insert after any leading system messages but before the conversation body
            insert_idx = next(
                (i for i, m in enumerate(compacted) if m.get("role") != "system"),
                len(compacted),
            )
            compacted.insert(insert_idx, notice)

        compacted_tokens = self._estimate_tokens(compacted)
        await self._emit_event(
            "context:compaction",
            {
                "reason": "inline_budget_enforcement",
                "budget": budget,
                "target_tokens": target_tokens,
                "original_tokens": original_tokens,
                "compacted_tokens": compacted_tokens,
                "original_message_count": original_count,
                "compacted_message_count": len(compacted),
            },
        )
        logger.info(
            "Inline compaction: %s → %s tokens (%d → %d messages)",
            f"{original_tokens:,}",
            f"{compacted_tokens:,}",
            original_count,
            len(compacted),
        )
        return compacted

    async def _merge_oldest_tiers(self) -> None:
        """Merge the two oldest summary tiers into one when tier count exceeds limit.

        Returns early if len(_summary_tiers) <= max_summary_tiers.

        Takes tier_a=_summary_tiers[0] and tier_b=_summary_tiers[1], builds
        a merge prompt requesting cohesion of two summaries with their turn
        ranges and a read_transcript advisory note, then calls the cached
        provider to produce a merged summary.

        The merged SummaryTier has:
          - turn_range: (tier_a.turn_range[0], tier_b.turn_range[1])
          - source_message_range: (tier_a.source_message_range[0], tier_b.source_message_range[1])
          - compression_passes: max(tier_a.compression_passes, tier_b.compression_passes) + 1
          - token_estimate: estimated from merged content

        Replaces first two tiers: self._summary_tiers = [merged_tier] + self._summary_tiers[2:]
        """
        if len(self._summary_tiers) <= self.max_summary_tiers:
            return

        from amplifier_core import ChatRequest, Message

        tier_a = self._summary_tiers[0]
        tier_b = self._summary_tiers[1]

        merge_prompt = (
            f"Merge the following two summaries into a single cohesive summary. "
            f"The first summary covers turns {tier_a.turn_range[0]}-{tier_a.turn_range[1]} "
            f"and the second covers turns {tier_b.turn_range[0]}-{tier_b.turn_range[1]}. "
            f"Produce a unified summary that preserves all important information from both.\n\n"
            f"Note: Use the read_transcript tool to retrieve verbatim content from a specific "
            f"turn range when exact wording, full error output, or code details are needed.\n\n"
            f"SUMMARY 1 (turns {tier_a.turn_range[0]}-{tier_a.turn_range[1]}):\n"
            f"{tier_a.content}\n\n"
            f"SUMMARY 2 (turns {tier_b.turn_range[0]}-{tier_b.turn_range[1]}):\n"
            f"{tier_b.content}"
        )

        request = ChatRequest(
            messages=[
                Message(role="user", content=merge_prompt),
            ],
            model=self.summarization_model,
        )

        response = await self._cached_provider.complete(request)
        merged_text = self._extract_text_from_response(response)

        merged_turn_range = (tier_a.turn_range[0], tier_b.turn_range[1])
        merged_source_range = (
            tier_a.source_message_range[0],
            tier_b.source_message_range[1],
        )
        merged_compression_passes = (
            max(tier_a.compression_passes, tier_b.compression_passes) + 1
        )
        merged_token_estimate = self._estimate_tokens_single(
            {"role": "system", "content": merged_text}
        )

        merged_tier = SummaryTier(
            content=merged_text,
            turn_range=merged_turn_range,
            source_message_range=merged_source_range,
            compression_passes=merged_compression_passes,
            token_estimate=merged_token_estimate,
        )

        self._summary_tiers = [merged_tier] + self._summary_tiers[2:]
        logger.debug(
            f"Merged oldest two tiers: turns {merged_turn_range[0]}-{merged_turn_range[1]}, "
            f"compression_passes={merged_compression_passes}"
        )

    def _snap_to_tool_pair_boundary(self, end_idx: int) -> int:
        """Adjust end_idx to avoid splitting tool call/result pairs.

        Case 1: If the last included message (at end_idx - 1) is an assistant
        message with tool_calls, extend end_idx to include all following tool
        result messages.

        Case 2: If the first excluded message (at end_idx) is a tool result,
        extend end_idx past all consecutive tool result messages.

        Args:
            end_idx: Exclusive end index into self._messages.

        Returns:
            Adjusted end_idx.
        """
        messages = self._messages
        n = len(messages)

        # Case 1: last included message is an assistant message with tool_calls
        if (
            end_idx > 0
            and end_idx <= n
            and messages[end_idx - 1].get("role") == "assistant"
            and messages[end_idx - 1].get("tool_calls")
        ):
            while end_idx < n and messages[end_idx].get("role") == "tool":
                end_idx += 1
            return end_idx

        # Case 2: first excluded message is a tool result
        if end_idx < n and messages[end_idx].get("role") == "tool":
            while end_idx < n and messages[end_idx].get("role") == "tool":
                end_idx += 1

        return end_idx
