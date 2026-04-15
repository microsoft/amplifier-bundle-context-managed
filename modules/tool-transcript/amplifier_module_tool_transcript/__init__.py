"""
Transcript retrieval tool for the context-managed bundle.

Gives the LLM on-demand access to full-fidelity past messages that have been
compressed into summaries. Reads from the context module's transcript.jsonl.

Implementation is Phase 3. This is a skeleton for bundle composition.
"""

__amplifier_module_type__ = "tool"

import json
import logging
import os
import re
from typing import Any

from amplifier_core import ToolResult

logger = logging.getLogger(__name__)

TOOL_DESCRIPTION = (
    "Read verbatim messages from the session transcript. Use this when a summary "
    "references an earlier turn and you need the exact wording, full error output, "
    "or code details. Returns the raw transcript entries for the specified turn range "
    "or search query. Helps recover precise context that was summarized away."
)


async def mount(coordinator: Any, config: dict[str, Any] | None = None) -> None:
    """Mount the transcript tool into the coordinator."""
    rate_limit = (config or {}).get("rate_limit_per_turn", 3)
    tool = ReadTranscriptTool(coordinator, rate_limit_per_turn=rate_limit)
    await coordinator.mount("tools", tool, name=tool.name)
    logger.info("tool-transcript mounted: registered 'read_transcript'")


class ReadTranscriptTool:
    """Tool for reading verbatim transcript entries on demand.

    Provides the LLM with direct access to full-fidelity past messages that
    may have been compressed by the context manager's summarization engine.
    """

    _MAX_CONTENT_LENGTH: int = 2000

    def __init__(self, coordinator: Any, rate_limit_per_turn: int = 3) -> None:
        self._coordinator = coordinator
        self._rate_limit_per_turn = rate_limit_per_turn
        self._calls_this_turn: int = 0

    @property
    def name(self) -> str:
        return "read_transcript"

    @property
    def description(self) -> str:
        return TOOL_DESCRIPTION

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "start_turn": {
                    "type": "integer",
                    "description": "First turn number to retrieve (inclusive).",
                },
                "end_turn": {
                    "type": "integer",
                    "description": "Last turn number to retrieve (inclusive).",
                },
                "search": {
                    "type": "string",
                    "description": "Text to search for across all transcript entries.",
                },
            },
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute the transcript read operation."""
        # Rate limit check
        if self._calls_this_turn >= self._rate_limit_per_turn:
            return ToolResult(
                success=False,
                output="Rate limit exceeded: too many transcript reads this turn.",
                error={"message": "rate_limit_exceeded"},
            )

        # Increment call counter
        self._calls_this_turn += 1

        # Discover transcript path via coordinator capability registry.
        # The context-managed module registers this at mount time under the
        # namespaced key "context-managed.transcript_path".
        transcript_path = self._coordinator.get_capability(
            "context-managed.transcript_path"
        )
        if transcript_path is None:
            return ToolResult(
                success=False,
                output="No transcript available.",
                error={"message": "no_transcript_path"},
            )

        # Parse transcript into turns
        turns = self._parse_transcript(transcript_path)

        # Empty transcript — return success with empty output
        if not turns:
            return ToolResult(success=True, output="")

        # Turn range (1-indexed): clamp start and end
        total = len(turns)
        start = max(1, input.get("start_turn", 1))
        end = min(total, input.get("end_turn", total))

        # start beyond total: return empty with turn count
        if start > total:
            return ToolResult(
                success=True,
                output=f"No turns found. Transcript has {total} turn(s).",
            )

        # Slice to requested range (convert to 0-indexed)
        range_turns = turns[start - 1 : end]
        range_indices = list(range(start, end + 1))  # 1-indexed turn numbers

        # Search filter (optional)
        search = input.get("search")
        if search:
            # Compile as regex (IGNORECASE); fall back to literal match on invalid regex
            try:
                pattern = re.compile(search, re.IGNORECASE)
            except re.error:
                pattern = re.compile(re.escape(search), re.IGNORECASE)

            filtered_turns: list[list[dict]] = []
            filtered_indices: list[int] = []
            for idx, turn_msgs in zip(range_indices, range_turns):
                combined = " ".join(
                    msg.get("content", "")
                    for msg in turn_msgs
                    if isinstance(msg.get("content"), str)
                )
                if pattern.search(combined):
                    filtered_turns.append(turn_msgs)
                    filtered_indices.append(idx)

            if not filtered_turns:
                return ToolResult(success=True, output="No matches found.")
        else:
            filtered_turns = range_turns
            filtered_indices = range_indices

        # Format turns with correct turn numbers using zip of filtered_turns and filtered_indices
        sections: list[str] = []
        for idx, turn_msgs in zip(filtered_indices, filtered_turns):
            section = self._format_turns([turn_msgs], start_turn=idx)
            sections.append(section)

        return ToolResult(success=True, output="\n\n".join(sections))

    def _parse_transcript(self, transcript_path: str) -> list[list[dict]]:
        """Parse a transcript file into turns.

        A turn starts with a user message and includes all subsequent messages
        until the next user message.  The following lines are silently skipped:

        * Lines whose top-level ``type`` field equals ``"transcript_header"``.
        * Lines whose ``metadata.type`` field equals
          ``"context_managed_summary"`` (context-manager summary markers).
        * Lines with malformed JSON (a warning is logged for each).

        Args:
            transcript_path: Absolute or relative path to a ``transcript.jsonl``
                file.

        Returns:
            A list of turns.  Each turn is a list of message dicts.  Returns
            an empty list when the file does not exist or contains no
            conversation messages.
        """
        if not os.path.exists(transcript_path):
            return []

        turns: list[list[dict]] = []
        current_turn: list[dict] = []

        try:
            with open(transcript_path) as fh:
                for line_num, raw in enumerate(fh, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Malformed JSON on line %d of %s – skipping",
                            line_num,
                            transcript_path,
                        )
                        continue

                    # Skip header lines.
                    if msg.get("type") == "transcript_header":
                        continue

                    # Skip context-manager summary markers.
                    metadata = msg.get("metadata")
                    if (
                        isinstance(metadata, dict)
                        and metadata.get("type") == "context_managed_summary"
                    ):
                        continue

                    # A user message starts a new turn.
                    if msg.get("role") == "user":
                        if current_turn:
                            turns.append(current_turn)
                        current_turn = [msg]
                    else:
                        # Non-user messages belong to the current turn.
                        # Discard any that arrive before the first user message.
                        if current_turn:
                            current_turn.append(msg)

        except OSError:
            logger.warning("Could not read transcript file %s", transcript_path)
            return []

        # Flush the last in-progress turn.
        if current_turn:
            turns.append(current_turn)

        return turns

    def reset_rate_limit(self) -> None:
        """Reset the per-turn call counter to zero."""
        self._calls_this_turn = 0

    def _format_turns(self, turns: list[list[dict]], start_turn: int) -> str:
        """Format a list of turns into human-readable text.

        Args:
            turns: A list of turns, each being a list of message dicts.
            start_turn: The turn number to assign to the first turn in the list.

        Returns:
            A formatted string with one section per turn, or a message
            indicating no content when ``turns`` is empty.
        """
        if not turns:
            return "No transcript content found for the requested range."

        sections: list[str] = []

        for turn_offset, turn_messages in enumerate(turns):
            turn_num = start_turn + turn_offset
            lines: list[str] = [f"--- Turn {turn_num} ---"]

            for msg in turn_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "tool":
                    # Tool result: show the tool_call_id
                    tc_id = msg.get("tool_call_id", "")
                    lines.append(f"[tool result: {tc_id}]")
                elif role == "assistant":
                    # Show role label + content
                    tool_calls = msg.get("tool_calls")
                    if content:
                        truncated = self._truncate_content(content)
                        lines.append(f"[assistant] {truncated}")
                    if tool_calls:
                        names = ", ".join(
                            tc.get("function", {}).get("name", "") for tc in tool_calls
                        )
                        lines.append(f"(calls: {names})")
                else:
                    # user and any other roles
                    truncated = self._truncate_content(content)
                    lines.append(f"[{role}] {truncated}")

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _truncate_content(self, content: str) -> str:
        """Truncate content that exceeds _MAX_CONTENT_LENGTH.

        Args:
            content: The text to potentially truncate.

        Returns:
            The original content if it fits within the limit, or a truncated
            version with a note showing the original character count.
        """
        if len(content) <= self._MAX_CONTENT_LENGTH:
            return content
        total = len(content)
        truncated = content[: self._MAX_CONTENT_LENGTH]
        return f"{truncated}... [truncated \u2014 {total:,} chars total]"
