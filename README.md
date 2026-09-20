# amplifier-bundle-context-managed

An [Amplifier](https://github.com/microsoft/amplifier) bundle providing intelligent, budget-aware context management with persistent JSONL transcripts, LLM-powered rolling summaries, and session resume support.

## Overview

Drop-in replacement for `context-simple`. Instead of mechanically chopping old messages when the context window fills up, this bundle uses the LLM itself to build rolling summaries of earlier conversation. The model always sees compressed older context + full-detail recent context, and can pull back any detail on demand.

### What You Get

- **Sessions survive past the context limit.** Decisions, file paths, errors, task state -- all captured in structured summaries instead of silently dropped.
- **A bundled `read_transcript` tool** lets the model pull verbatim history (including full tool inputs and results) whenever it needs earlier details. No silent gaps, no hallucinated fill-ins.
- **Budget-aware context tracking** starts summarizing at 60%, warns at 70%, emergency fallback at 92%. The model never gets close enough to the limit to start behaving erratically.
- **Everything persists to disk.** Sessions can be resumed, transcripts inspected, and full history is always available even after summarization compresses the live context.

## Modules

### `context-managed`

Core context management module implementing the `ContextManager` protocol from `amplifier-core`:

- **Persistent JSONL transcript with format versioning** -- every message written to disk with a versioned header; archives on clear
- **Budget calculation and pressure events** -- tracks token usage across configurable thresholds (`summarize_trigger`, `pressure_warning`, `emergency_fallback`) and emits `context:budget_pressure` events
- **LLM-powered rolling summarization** -- async summary generation with multi-tier summary management and cache-friendly ordering
- **Session resume** -- reloads transcript from disk on startup so long-running sessions survive process restarts
- **Large tool result handling** -- tool results over the threshold are pointer-filed to disk; only a truncated reference is kept in memory
- **System prompt factory support** -- accepts a callable that produces the system prompt on every request, enabling dynamic prompts

### `tool-transcript`

Provides the `read_transcript` tool for on-demand full-fidelity message retrieval:

- **Turn-based access** -- read specific turn ranges from the persistent transcript
- **Search** -- find specific content across the full transcript history
- **Rate-limited** -- configurable per-turn call limit to prevent excessive retrieval

## Usage

Reference the bundle from your Amplifier configuration:

```yaml
bundle:
  modules:
    - context-managed
    - tool-transcript
```

The `context-managed` module registers itself as the session context provider. The `tool-transcript` module gives the model on-demand access to past messages that have been compressed into summaries.

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for all configuration parameters.

## Development

Install dependencies (per module):

```bash
cd modules/context-managed
uv sync
```

Run the test suite:

```bash
uv run pytest tests/ -v
```

The same pattern applies for `modules/tool-transcript`.

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

Boundary-engine hosts can opt into [durable compaction checkpoints](modules/context-managed/README.md).
The host retains original transcripts and provides storage; the module validates
compatible derived summaries without replaying work.
