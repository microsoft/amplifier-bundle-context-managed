---
bundle:
  name: context-managed
  display_name: Context Managed
  version: 0.1.0
  description: |
    Foundation bundle with Boundary-only request preparation and compaction.

    Features:
    - Two-parameter context inheritance (context_depth + context_scope)
    - Session resume (use full session_id from delegate calls)
    - Fixed tool inheritance (explicit declarations always honored)
    - Multi-agent collaboration patterns
    - Host-owned canonical history and optional durable checkpoints
    - Provider-native compaction or portable continuation notes
    - Explicit preparation errors instead of emergency history trimming

includes:
  # Ecosystem expert behaviors (provides @amplifier: and @core: namespaces)
  - bundle: git+https://github.com/microsoft/amplifier@main#subdirectory=behaviors/amplifier-expert.yaml
  - bundle: git+https://github.com/microsoft/amplifier-core@main#subdirectory=behaviors/core-expert.yaml
  # Foundation expert behavior
  - bundle: foundation:behaviors/foundation-expert
  # Foundation behaviors
  - bundle: foundation:behaviors/sessions
  - bundle: foundation:behaviors/status-context
  - bundle: foundation:behaviors/redaction
  - bundle: foundation:behaviors/todo-reminder
  - bundle: foundation:behaviors/streaming-ui
  # Agent orchestration with delegate tool
  - bundle: foundation:behaviors/agents
  # External bundles
  - bundle: git+https://github.com/microsoft/amplifier-bundle-recipes@main#subdirectory=behaviors/recipes.yaml
  - bundle: git+https://github.com/microsoft/amplifier-bundle-design-intelligence@main#subdirectory=behaviors/design-intelligence.yaml
  - bundle: git+https://github.com/microsoft/amplifier-bundle-python-dev@main
  - bundle: git+https://github.com/microsoft/amplifier-bundle-amplifier-tester@main
  - bundle: git+https://github.com/microsoft/amplifier-bundle-skills@main#subdirectory=behaviors/skills.yaml
  - bundle: git+https://github.com/microsoft/amplifier-bundle-browser-tester@main#subdirectory=behaviors/browser-tester.yaml
  - bundle: git+https://github.com/microsoft/amplifier-bundle-superpowers@main#subdirectory=behaviors/superpowers-methodology.yaml
  - bundle: git+https://github.com/microsoft/amplifier-module-hook-shell@main#subdirectory=behaviors/hook-shell.yaml
  - bundle: git+https://github.com/microsoft/amplifier-module-tool-mcp@main#subdirectory=behaviors/mcp.yaml
  - bundle: git+https://github.com/microsoft/amplifier-bundle-filesystem@main#subdirectory=behaviors/apply-patch.yaml
  - bundle: git+https://github.com/microsoft/amplifier-bundle-routing-matrix@main


session:
  raw: true
  orchestrator:
    module: loop-streaming
    source: git+https://github.com/microsoft/amplifier-module-loop-streaming@main
    config:
      extended_thinking: true
  context:
    module: context-managed
    config:
      engine: boundary
      max_tokens: 200000
      summarize_trigger: 0.60
      summary_target_tokens: 1500

tools:
  - module: tool-filesystem
    source: git+https://github.com/microsoft/amplifier-module-tool-filesystem@main
  - module: tool-bash
    source: git+https://github.com/microsoft/amplifier-module-tool-bash@main
  - module: tool-web
    source: git+https://github.com/microsoft/amplifier-module-tool-web@main
  - module: tool-search
    source: git+https://github.com/microsoft/amplifier-module-tool-search@main
  # NOTE: delegate tool comes from agents behavior
  - module: tool-transcript
    config:
      rate_limit_per_turn: 3

agents:
  include:
    # Note: amplifier-expert, core-expert, and foundation-expert come via included behaviors above
    - foundation:bug-hunter
    - foundation:explorer
    - foundation:file-ops
    - foundation:git-ops
    - foundation:integration-specialist
    - foundation:modular-builder
    - foundation:post-task-cleanup
    - foundation:security-guardian
    - foundation:test-coverage
    - foundation:web-research
    - foundation:zen-architect
---

# Context Managed Bundle v0.1.0

This bundle retains its Amplifier foundation composition and uses the Boundary
context engine. Canonical history and persistence belong to the host, not a second
transcript writer inside the context manager.

## Key Features

| Feature | Description |
|---------|-------------|
| **Delegate tool** | Two-parameter context control (depth + scope) |
| **Session resume** | Continue agent sessions with full session_id |
| **Tool inheritance** | Explicit declarations always honored |
| **MCP support** | Model Context Protocol integration (configure via mcp.json) |
| **Boundary compaction** | Native compaction when advertised; portable notes for providers without native support |
| **Canonical history** | Full-fidelity originals retained by the host; optional validated derived checkpoints |
| **Request safety** | Budget validation; no emergency message deletion or tool-output clipping |

## Context Manager Module

`context-managed` implements the Core `ContextManager` protocol. Mount it before
`tool-transcript` so the tool can discover `context.history` with
`context.history_authority: host`. This bundle explicitly selects `engine: boundary`;
omitting the setting now selects the same engine. Legacy engine selection and
Legacy-only options fail with removal guidance. See [configuration](docs/CONFIGURATION.md).

The bundled `summary_target_tokens` value is accepted for compatibility, but is
inert in Boundary. Portable generation uses `summary_max_output_tokens` (8192 by
default); native generation has a separate allowance.

## Transcript Tool

`read_transcript` returns formatted host canonical history; message text and tool
arguments can be truncated. It does not promise full-fidelity ordinary retrieval.
Old saved transcripts and large-result sidecars are available separately through
host-approved archive IDs. Only explicit archives return complete raw records
within page limits; oversized single records fail visibly. See
[archive usage and limits](docs/CONFIGURATION.md#legacy-archive-usage).
Archive access is read-only and does not import or rewrite history.
A host must preserve originals and supply storage; this bundle
alone does not provide universal durability or automatic Legacy migration.

## Context Instructions

The included summary instructions treat continuation notes as reference data,
never new authorization. Retrieve evidence when available; report missing history
rather than inventing details.

## Delegate Tool

```python
# Context depth: HOW MUCH to inherit
context_depth: "none" | "recent" | "all"

# Context scope: WHICH content to include
context_scope: "conversation" | "agents" | "full"
```

## MCP Configuration

To use MCP servers, create `.amplifier/mcp.json`:

```json
{
  "mcpServers": {
    "your-server": {
      "url": "https://example.com/mcp"
    }
  }
}
```

@foundation:context/shared/common-system-base.md

@context-managed:context/summary-instructions.md
