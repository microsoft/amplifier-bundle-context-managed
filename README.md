# amplifier-bundle-context-managed

An [Amplifier](https://github.com/microsoft/amplifier) bundle providing
[Boundary-only context management](docs/BOUNDARY_ENGINE.md), with host-owned
canonical history, request-budget validation, and optional durable compaction checkpoints.

## Overview

The context manager implements Core's `ContextManager` protocol. It compacts
eligible settled history before a foreground request. Providers advertising native
compaction must satisfy that contract; providers without native support use
portable continuation notes. A failed compaction or still-oversized request stops
preparation instead of deleting messages or clipping tool output.

### What You Get

- **Explicit request safety.** The module preserves canonical originals and reports preparation failures. Compaction is not a guarantee of lossless model recall or unlimited context.
- **Evidence retrieval.** The bundled `read_transcript` tool reads host history. Host-approved old archives and sidecar results are separately accessible read-only.
- **Host-controlled durability.** A host can preserve originals and persist compatible derived checkpoints for resume. The module does not create its own transcript writer or provide storage by itself.
- **No context-owned elapsed deadlines.** Source, output, and logical-call limits remain; caller cancellation and real provider errors still stop preparation.

## Modules

### `context-managed`

Core context management module implementing the `ContextManager` protocol from `amplifier-core`:

- **Boundary-only mount** -- omitted `engine` selects `boundary`; Legacy, unknown and null values fail before coordinator mutation
- **Request validation and compaction events** -- the engine trigger defaults to 70%; this bundle explicitly retains its 60% trigger
- **Native checkpoints or portable notes** -- strict native handling, with no native-error fallback to text summaries
- **Host-owned resume** -- the host restores canonical history and optionally validates a derived checkpoint
- **Whole-message preservation** -- large results and metadata stay in canonical history; no private pointer filing
- **System prompt factory support** -- accepts a callable that produces the system prompt on every request, enabling dynamic prompts

### `tool-transcript`

Provides the `read_transcript` tool for formatted host-history retrieval and
separate bounded raw archive access:

- **Turn-based access and search** -- retrieve formatted host canonical history; message text and tool arguments can be truncated
- **Legacy archive access** -- explicit approved IDs select bounded raw record pages and referenced sidecar byte pages, without rewriting files or importing them into canonical history
- **Rate-limited** -- configurable per-turn call limit to prevent excessive retrieval

Only explicit archives return complete raw records within page limits; an
oversized single record fails visibly. See the
[archive usage and limits](docs/CONFIGURATION.md#legacy-archive-usage).

## Usage

Include the bundle from your Amplifier configuration:

```yaml
includes:
  - bundle: git+https://github.com/microsoft/amplifier-bundle-context-managed@main
```

The bundle keeps its existing host composition and explicitly sets
`session.context.config.engine: boundary`. The `ManagedContextManager` Python
class and Legacy execution path have been removed; direct callers must use
`amplifier_module_context_managed.boundary.BoundaryContextManager` and supply
host persistence as needed. Remove Legacy-only options before mounting.

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

The same pattern applies for `modules/tool-transcript`. Run each module and the
root `tests/` in separate pytest processes to avoid their `tests.conftest`
namespace collision. A controlled environment can disable plugin autoload and
load `pytest_asyncio.plugin` and `amplifier_core.pytest_plugin` explicitly.

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
