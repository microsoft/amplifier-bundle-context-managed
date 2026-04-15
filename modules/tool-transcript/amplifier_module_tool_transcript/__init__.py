"""
Transcript retrieval tool for the context-managed bundle.

Gives the LLM on-demand access to full-fidelity past messages that have been
compressed into summaries. Reads from the context module's transcript.jsonl.

Implementation is Phase 3. This is a skeleton for bundle composition.
"""

__amplifier_module_type__ = "tool"

import logging
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
    tool = ReadTranscriptTool(coordinator)
    await coordinator.mount("tools", tool, name=tool.name)
    logger.info("tool-transcript mounted: registered 'read_transcript'")


class ReadTranscriptTool:
    """Tool for reading verbatim transcript entries on demand.

    Provides the LLM with direct access to full-fidelity past messages that
    may have been compressed by the context manager's summarization engine.
    """

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
        """Execute the transcript read operation.

        Phase 3 implementation pending. Returns a placeholder response.
        """
        return ToolResult(success=True, output="(not yet implemented)")

    def reset_rate_limit(self) -> None:
        """Reset the per-turn call counter to zero."""
        self._calls_this_turn = 0
