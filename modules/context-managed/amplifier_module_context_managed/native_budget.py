"""Select a native output allowance using public, model-bound provider metadata."""

import inspect
from collections.abc import Mapping


def _field(record, key):
    return record.get(key) if isinstance(record, Mapping) else getattr(record, key, None)


def _positive(value):
    return type(value) is int and value > 0


async def native_output_limit(provider, model, config, decision):
    explicit = config.get("native_compaction_max_output_tokens")
    if "native_compaction_max_output_tokens" in config:
        if not _positive(explicit):
            raise ValueError("native_compaction_max_output_tokens must be a positive integer")
        return explicit

    model = model or getattr(provider, "default_model", None)
    info = getattr(provider, "get_info", lambda: None)()
    if inspect.isawaitable(info):
        info = await info
    record = _field(info, "defaults")
    if not model or _field(record, "model") != model:
        # A chat can override its connection's default model. Only use an
        # exact catalog match; a different model's limits are unsafe here.
        record = None
        catalog = getattr(provider, "list_models", None)
        if model and callable(catalog):
            try:
                models = catalog()
                if inspect.isawaitable(models):
                    models = await models
                record = next((item for item in models if _field(item, "id") == model), None)
            except Exception:
                # Optional discovery is not needed to continue with the legacy allowance.
                record = None
    output = _field(record, "max_output_tokens")
    window = _field(record, "context_window")
    if not _positive(output):
        return 4096
    if (decision or {}).get("input_limit_independent_of_output") is True or not _positive(window):
        return output
    # Shared or unspecified windows must leave capacity for the source.
    # The provider still admits the exact assembled request including output.
    return min(output, max(4096, window // 20))
