# amplifier-bundle-context-managed

An [Amplifier](https://github.com/microsoft/amplifier-core) bundle providing intelligent, budget-aware context management with persistent JSONL transcripts, LLM-powered rolling summaries, and session resume support.

## Status

### Phase 1 — Complete ✅

Core infrastructure is fully implemented and tested:

- **Protocol compliance** — implements the `ContextManager` protocol from `amplifier-core`
- **Persistent JSONL transcript with format versioning** — every message written to disk with a versioned header; archives on clear
- **Budget calculation and pressure events** — tracks token usage across three configurable thresholds (`pressure_warning`, `summarize_trigger`, `emergency_fallback`) and emits `context:budget_pressure` events
- **Session resume** — reloads transcript from disk on startup so long-running sessions survive process restarts
- **Large tool result handling** — tool results over the threshold are pointer-filed to disk; only a truncated reference is kept in memory
- **System prompt factory support** — accepts a callable that produces the system prompt on every request, enabling dynamic prompts

### Phase 2 — Planned 🔜

LLM-powered rolling summarization engine: async summary generation, multi-tier summary management, cache-friendly ordering.

### Phase 3 — Planned 🔜

Transcript tool + full bundle integration: on-demand full-fidelity message retrieval, bundle wiring, and end-to-end testing.

---

## Usage

Reference the bundle from your Amplifier configuration:

```yaml
bundle:
  modules:
    - context-managed
    - tool-transcript
```

The `context-managed` module registers itself as the session context provider.  The `tool-transcript` module gives the model on-demand access to past messages that have been compressed into summaries.

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for all configuration parameters.

---

## Development

Install dependencies:

```bash
cd modules/context-managed
uv sync
```

Run the test suite:

```bash
uv run pytest tests/ -v
```
