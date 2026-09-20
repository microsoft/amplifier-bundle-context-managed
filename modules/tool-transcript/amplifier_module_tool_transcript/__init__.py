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
    async def reset(event, data):
        from amplifier_core import HookResult
        tool.reset_rate_limit()
        return HookResult()
    if getattr(coordinator, "hooks", None):
        coordinator.hooks.register("prompt:submit", reset, name="transcript-rate-limit")
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
        history = self._coordinator.get_capability("context.history")
        if self._coordinator.get_capability("context.history_authority") == "host" and callable(history):
            turns = []
            for message in await history():
                if message.get("role") == "user" and not (message.get("metadata") or {}).get("ephemeral"):
                    turns.append([])
                if turns:
                    turns[-1].append(message)
        elif transcript_path is None:
            return ToolResult(
                success=False,
                output="No transcript available.",
                error={"message": "no_transcript_path"},
            )

        # Parse transcript into turns
        else:
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

        output = "\n\n".join(sections)

        # If the formatted output is very large, return a leading subset of turns
        # with a note so the caller can use a narrower range.
        _MAX_OUTPUT_CHARS = 100_000
        if len(output) > _MAX_OUTPUT_CHARS:
            truncated: list[str] = []
            total = 0
            last_idx = filtered_indices[0]
            for section, idx in zip(sections, filtered_indices):
                entry_len = len(section) + 2  # +2 for the "\n\n" separator
                if total + entry_len > _MAX_OUTPUT_CHARS and truncated:
                    break
                truncated.append(section)
                total += entry_len
                last_idx = idx
            note = (
                f"\n\n[Showing turns {filtered_indices[0]}–{last_idx} of requested "
                f"{filtered_indices[0]}–{filtered_indices[-1]}. "
                f"Use a narrower range to retrieve the remaining turns.]"
            )
            output = "\n\n".join(truncated) + note

        return ToolResult(success=True, output=output)

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

        Each turn is introduced with a ``--- Turn N ---`` header.  Messages
        are formatted as follows:

        * **user / other roles** — ``[role] <text>`` with content truncated to
          ``_MAX_CONTENT_LENGTH``.
        * **assistant** — ``[assistant] <text>`` (truncated) plus one line per
          tool call: ``[tool_call: name({compact_input})]``.  Tool calls are
          gathered from both the ``tool_calls`` field (supporting OpenAI
          ``function.name`` and direct ``name``/``tool`` formats) and any
          ``tool_call`` / ``tool_use`` content blocks.
        * **tool results** — ``[tool result: {id}]`` followed immediately by
          the full content on the next line.  Tool result content is not
          truncated here because it is already bounded upstream; the overall
          output-size guard in ``execute()`` handles extreme cases.

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
                    # Tool result: show id + full content (already bounded upstream)
                    tc_id = msg.get("tool_call_id", "")
                    # Content is normally a string; handle list as a fallback
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        )
                    lines.append(f"[tool result: {tc_id}]")
                    if content:
                        lines.append(content)

                elif role == "assistant":
                    # Extract text and tool_call blocks from list content
                    shown_tc_ids: set[str] = set()
                    tc_block_lines: list[str] = []

                    if isinstance(content, list):
                        text_parts: list[str] = []
                        for block in content:
                            if isinstance(block, dict):
                                btype = block.get("type", "")
                                if btype == "text":
                                    text_val = block.get("text", "")
                                    if text_val:
                                        text_parts.append(text_val)
                                elif btype in ("tool_call", "tool_use"):
                                    name = block.get("name", "unknown_tool")
                                    inp = block.get("input", {})
                                    tc_id = block.get("id", "")
                                    if tc_id:
                                        shown_tc_ids.add(tc_id)
                                    inp_str = json.dumps(inp, separators=(",", ":"))
                                    if len(inp_str) > 500:
                                        inp_str = inp_str[:500] + "..."
                                    tc_block_lines.append(
                                        f"[tool_call: {name}({inp_str})]"
                                    )
                                elif "text" in block:
                                    text_val = block.get("text", "")
                                    if text_val:
                                        text_parts.append(text_val)
                            elif hasattr(block, "text"):
                                text_parts.append(block.text)
                        content = "\n".join(text_parts)

                    if content:
                        truncated = self._truncate_content(content)
                        lines.append(f"[assistant] {truncated}")

                    # Emit tool_call blocks found in content
                    lines.extend(tc_block_lines)

                    # Also emit tool_calls field entries not already shown
                    tool_calls = msg.get("tool_calls") or []
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        tc_id = tc.get("id", "")
                        if tc_id and tc_id in shown_tc_ids:
                            continue
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
                        lines.append(f"[tool_call: {name}({inp_str})]")

                else:
                    # user and any other roles
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        )
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
