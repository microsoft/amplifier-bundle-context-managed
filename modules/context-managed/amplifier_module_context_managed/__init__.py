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

logger = logging.getLogger(__name__)

# Format version for transcript.jsonl header
TRANSCRIPT_FORMAT_VERSION = "1.0.0"


@dataclass
class SummaryResult:
    """Result of a single LLM summarization call (Phase 2).

    Captures the output text along with the turn range and message range
    that were summarized, plus how many compression passes were applied.
    """

    summary_text: str
    turn_range: tuple[int, int]
    source_message_range: tuple[int, int]
    compression_passes: int = 1


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

    # Resolve session directory for transcript storage
    session = getattr(coordinator, "session", None)
    session_dir = None
    if session is not None:
        session_dir = getattr(session, "session_dir", None)

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
        summarize_trigger=config.get("summarize_trigger", 0.80),
        emergency_fallback=config.get("emergency_fallback", 0.92),
        hooks=getattr(coordinator, "hooks", None),
        session_dir=session_dir,
    )

    # Register transcript path for the transcript tool to discover
    transcript_path = context.transcript_path
    if transcript_path is not None:
        coordinator.register_capability("context_transcript_path", str(transcript_path))

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
    thresholds: pressure_warning (0.70), summarize_trigger (0.80), and
    emergency_fallback (0.92).

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
        summarize_trigger: float = 0.80,
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
        """Path to the transcript JSONL file, or None if no session dir."""
        if self._session_dir is None:
            return None
        return self._session_dir / "transcript.jsonl"

    @property
    def tool_results_dir(self) -> Path | None:
        """Path to the large tool results directory, or None if no session dir."""
        if self._session_dir is None:
            return None
        return self._session_dir / "tool_results"

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

        # Phase 2: summary tiers would be inserted here with cache hints
        # For Phase 1, all conversation is verbatim

        assembled.extend(conversation_messages)

        # Check budget and emit pressure event
        system_tokens = self._estimate_tokens(assembled[:1]) if assembled else 0
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
                        "budget": available,
                    },
                )

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

        Same logic as context-simple's _calculate_budget().
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
                            logger.info(
                                f"Budget from provider model info: {budget:,} "
                                f"(context={context_window:,}, "
                                f"reserved_output={reserved_output:,})"
                            )
                            return budget

                info = provider.get_info()
                defaults = info.defaults or {}
                context_window = defaults.get("context_window")
                max_output_tokens = defaults.get("max_output_tokens")

                if context_window and max_output_tokens:
                    reserved_output = int(max_output_tokens * output_reserve_fraction)
                    budget = context_window - reserved_output - safety_margin
                    logger.info(f"Budget from provider defaults: {budget:,}")
                    return budget
            except Exception as e:
                logger.debug(f"Could not get budget from provider: {e}")

        logger.info(f"Using fallback max_tokens budget: {self.max_tokens:,}")
        return self.max_tokens

    # ── Token Estimation ──────────────────────────────────────────────────────

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Rough token estimation (chars / 4) for a list of messages."""
        return sum(len(str(msg)) // 4 for msg in messages)

    def _estimate_tokens_single(self, message: dict[str, Any]) -> int:
        """Rough token estimation for a single message."""
        return len(str(message)) // 4

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

        Skips the header line and any summary markers (metadata.type ==
        'context_managed_summary'). Returns raw conversation messages only.
        """
        if self.transcript_path is None or not self.transcript_path.exists():
            return []

        messages = []
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

                # Skip header
                if record.get("type") == "transcript_header":
                    continue

                # Skip summary markers (implementation detail, not conversation)
                meta = record.get("metadata") or {}
                if meta.get("type") == "context_managed_summary":
                    continue

                messages.append(record)

        return messages

    async def _load_from_transcript(self) -> None:
        """Load state from existing transcript file (session resume).

        Reconstructs in-memory state from the JSONL file. Sets
        _loaded_from_transcript = True so set_messages() becomes a NO-OP.
        """
        if self.transcript_path is None or not self.transcript_path.exists():
            return

        try:
            messages = self._read_transcript_messages()
            if not messages:
                return

            self._messages = messages
            self._message_index = len(messages)
            self._running_token_estimate = self._estimate_tokens(messages)
            self._loaded_from_transcript = True
            self._pressure_emitted = False

            # Phase 2: reconstruct summary tiers from marked messages

            logger.info(
                f"Resumed from transcript: {len(messages)} messages, "
                f"~{self._running_token_estimate:,} tokens"
            )
        except Exception as e:
            logger.warning(f"Failed to load transcript, starting fresh: {e}")
            self._messages = []
            self._message_index = 0
            self._running_token_estimate = 0

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
        )

    def _format_messages_for_summarization(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for the summarization prompt.

        Each message is formatted as '[role]: content'.  When content is a
        list of content blocks, text is extracted from blocks that have a
        .text attribute and joined together.
        """
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Join text from all blocks that carry a .text attribute
                text_parts = []
                for block in content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                content = "".join(text_parts)
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

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

        Calls _perform_summarization(boundary) and stores result in
        _pending_summary.  On exception: increments _summarization_failures
        and logs a warning.  In the finally block: resets _is_summarizing=False
        and _summarization_task=None.
        """
        try:
            result = await self._perform_summarization(boundary)
            self._pending_summary = result
        except Exception as e:
            self._summarization_failures += 1
            logger.warning(f"Summarization failed: {e}")
        finally:
            self._is_summarizing = False
            self._summarization_task = None

    async def _check_summarization_trigger(self) -> None:
        """Check if summarization should be triggered based on usage fraction.

        Skips if _is_summarizing is True.  Calculates budget via
        _calculate_budget(None, _cached_provider) and computes
        usage_fraction = _running_token_estimate / budget.  Calls
        _trigger_summarization() if usage_fraction >= summarize_trigger.
        """
        if self._is_summarizing:
            return
        budget = self._calculate_budget(None, self._cached_provider)
        usage_fraction = self._running_token_estimate / budget
        if usage_fraction >= self.summarize_trigger:
            await self._trigger_summarization()

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
