"""Boundary-only context management; canonical history and storage are host-owned."""

from typing import Any

__amplifier_module_type__ = "context"

LEGACY_ONLY_OPTIONS = frozenset({
    "verbatim_window_tokens",
    "max_summary_tiers",
    "summarization_prompt_path",
    "summarization_retries_before_fallback",
    "emergency_target_usage",
    "large_result_threshold",
    "pressure_warning",
    "emergency_fallback",
})


async def mount(coordinator: Any, config: dict[str, Any] | None = None):
    """Validate configuration before delegating to the request-boundary engine."""
    config = dict(config) if config is not None else {}
    engine = config.setdefault("engine", "boundary")
    if type(engine) is not str or engine != "boundary":
        raise ValueError(
            "context-managed supports only engine: boundary. Remove the engine setting "
            "or set it to boundary; the Legacy executor has been removed. "
            "Old saved data remains accessible through read_transcript archive IDs."
        )
    obsolete = sorted(LEGACY_ONLY_OPTIONS.intersection(config))
    if obsolete:
        raise ValueError(
            "Remove Legacy-only context-managed options: " + ", ".join(obsolete)
            + ". Boundary does not implement these policies; see docs/CONFIGURATION.md."
        )

    from .boundary import mount_boundary

    return await mount_boundary(coordinator, config)