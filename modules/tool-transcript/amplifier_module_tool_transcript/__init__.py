"""
Transcript retrieval tool for the context-managed bundle.

Gives the LLM on-demand access to full-fidelity past messages that have been
compressed into summaries. Reads from the context module's transcript.jsonl.

Implementation is Phase 3. This is a mount stub for bundle composition.
"""

__amplifier_module_type__ = "tool"

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def mount(coordinator: Any, config: dict[str, Any] | None = None):
    """Mount the transcript tool. (Phase 3 — stub only.)"""
    logger.info("tool-transcript: stub mounted (implementation pending Phase 3)")
    return
