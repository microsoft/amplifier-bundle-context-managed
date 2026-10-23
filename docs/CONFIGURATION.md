# Configuration Reference

## Context manager (`context-managed`)

Boundary is the only engine. Omitted `engine` normalizes to `boundary`.
Only the exact string `boundary` is accepted: `legacy`, unknown values, and
explicit null fail before mounting or registering coordinator capabilities.
Configuration is copied; caller-owned settings are not changed.

The Legacy executor and its Python exports (`ManagedContextManager`, `SummaryTier`,
`SummaryResult`, and `TRANSCRIPT_FORMAT_VERSION`) are removed, not aliases.
Direct manager users can import
`amplifier_module_context_managed.boundary.BoundaryContextManager`.
The host must restore canonical history with `set_messages`; mounting never loads
or writes a private transcript. See [Boundary policy](BOUNDARY_ENGINE.md) and
[host checkpoint integration](../modules/context-managed/README.md).

| Parameter | Engine default | Description |
|---|---|---|
| `engine` | `boundary` | The sole supported engine. |
| `max_tokens` | unset | Optional cap on the shared helper's derived request budget. This bundle explicitly sets 200,000; provider/measured allowances may be smaller. |
| `max_tokens_fallback` | 200,000 | Shared helper's fallback when no provider budget is available. |
| `summarize_trigger` | 0.70 | Trigger fraction before a foreground request. This bundle explicitly sets 0.60. |
| `token_meter` | `actual` | Advertise the measured-request capability. `estimate` does not advertise it. Estimates do not validate opaque native state. |
| `native_compaction` | `true` | Use the continuation provider's native contract when advertised. Disabling an advertised native contract stops preparation; it does not select portable fallback. |
| `native_min_new_tokens` | 500 | Avoid tiny native recompactions unless fitting is required. |
| `native_compaction_max_output_tokens` | 4096 | Separate native operation output allowance. |
| `summary_max_output_tokens` | 8192 | Portable note output allowance, including reasoning when the provider counts it there. |
| `summary_target_tokens` | unset | Accepted compatibility setting, inert in Boundary. The bundle retains 1500; it is not remapped to `summary_max_output_tokens`. |
| `summarization_model` | provider default | Model for portable summaries, not opaque native state. |
| `summary_reasoning_effort` | `low` | Portable summary reasoning setting. |
| `summary_max_source_chars` | automatic | Explicit cap for portable source fragments; no authoritative count uses a conservative 512000-character cap, also bounded by provider context. Does not fragment native requests. |
| `summary_max_calls` | 16 | Logical portable-call allowance per preparation pass, including failed calls. Cumulative calls across resumed progress are tracked separately; this is not a lifetime allowance for an unchanged prefix. Completed pieces can resume. |
| `summary_retry_delay` | 60 seconds | Initial transient-failure cooldown with exponential backoff and bounded attempts. Not a generation deadline. |
| `separate_summary_provider` | `false` | Require the host's `context.summary_provider` for portable notes. |
| `durable_checkpoints` | `false` | Opt into host evidence preservation and checkpoint capabilities. Storage is host-owned. |
| `archive_recovery` | `false` | Opt-in bounded working-view selection from host-retained originals, with durable checkpoints; not automatic Legacy import. |

Shared request settings such as `max_tokens_fallback`, `output_reserve_fraction`,
and compaction-notice settings still pass through to the existing budget helpers.
They do not enable context-simple's emergency deletion/truncation ladder.
The mount does not add a new general schema or reinterpret other Boundary settings.

`summary_timeout`, `summary_total_work_timeout`, and `native_compaction_timeout`
remain accepted and ignored. Neither method has a context-owned elapsed-time
deadline. Healthy generation can take many minutes; completion, real provider
errors, or caller cancellation end it. Provider transport deadlines are separate.

### Remove Legacy-only options

Remove all of the following from old bundle or saved configuration before mounting.
Even an explicit null or false value is rejected. There is no silent replacement:

- `verbatim_window_tokens`, `max_summary_tiers`
- `summarization_prompt_path`, `summarization_retries_before_fallback`
- `emergency_target_usage`, `large_result_threshold`
- `pressure_warning`, `emergency_fallback`

Boundary does not recreate Legacy tiers, background summarization, emergency
trimming, pointer filing, or private transcript resume. Old explicit Boundary
configurations retain their checkpoint identity; selecting Boundary by omission
now normalizes to the same explicit configuration. Checkpoints still validate
source, provider/model, generation configuration, prompt, and record integrity.
This does not convert Legacy summary markers into Boundary checkpoints.

## Transcript tool (`tool-transcript`)

| Parameter | Default | Description |
|---|---|---|
| `rate_limit_per_turn` | 3 | Maximum transcript calls per iteration. |
| `legacy_transcript_paths` | none | Host-approved archive ID map, separate from host canonical history. Each entry is a transcript path string or `{transcript, original_tool_results?, tool_results?}`. |

Ordinary turn-range/search retrieval formats host canonical history for display.
Message text and tool arguments can be truncated; this interface does not promise
complete raw messages. Explicit archive reads use the separate contract below.

### Legacy archive usage

The host configures approved archive IDs on `tool-transcript`. Paths are host
configuration, never tool-call arguments. For example:

```yaml
tools:
  - module: tool-transcript
    config:
      rate_limit_per_turn: 3
      legacy_transcript_paths:
        saved: /srv/archives/session/context-managed/transcript.jsonl
        relocated:
          transcript: /srv/archives/moved/transcript.jsonl
          original_tool_results: /srv/original/session/context-managed/tool_results
          tool_results: /srv/archives/moved/tool_results
```

A string entry uses `tool_results` beside the transcript as its implicit sidecar
root. An object entry requires `transcript`; `original_tool_results` and
`tool_results` are optional. To relocate sidecars, specify the original approved
root and its one destination root. The reader keeps the exact relative suffix.
It does not search by basename, discover alternate roots, or modify pointer
records. Absolute pointers must fall under the original root. Relative pointers
are resolved under that root, with an exact configured relative-root prefix
removed when present. Traversal and symlinks are rejected.

Configured transcripts and explicit destination roots are preflighted before tool
mounting. The implicit default `tool_results` directory may be absent when an
archive has no sidecars; reading a missing result still fails visibly. Safe access
requires POSIX descriptor-relative no-follow opens. Unsupported platforms fail
closed with `unsupported_safe_open` rather than following paths unsafely.

**Read raw records.** Call `read_transcript` with:

```json
{"archive_id": "saved", "offset_bytes": 0, "max_records": 100, "max_bytes": 65536}
```

Explicit `archive_id` takes precedence over active host history. An unknown ID
fails; it never silently substitutes host history. Do not mix archive arguments
with `start_turn`, `end_turn`, or `search`. Archive access validates the
`transcript_header` with `format_version: "1.0.0"` and returns the header plus
complete raw JSON objects in `records`. Unknown fields, summary markers and
`metadata.full_result_path` pointers remain unchanged, not hydrated or executed.

Each record entry has its own `offset_bytes`, `next_offset_bytes`, and `record`.
Continue with the page's `next_offset_bytes` until `eof` is true; offsets are byte
positions in the archive, not turn numbers or character positions. Default
`max_records` is **100**, also the hard maximum per page.

**Read a referenced sidecar.** Use the selected tool record's `offset_bytes` as
`result_record_offset`, not its turn number:

```json
{"archive_id": "saved", "result_record_offset": 128, "offset_bytes": 0, "max_bytes": 65536}
```

Here `128` is illustrative: use an offset returned by the record page. In sidecar
mode, `offset_bytes` starts at zero within that result file. The response supplies
`encoding: "base64"`, `data`, `total_bytes`, `next_offset_bytes`, and `eof`.
Decode each `data` page as bytes and continue using its `next_offset_bytes`.
This leaves the original pointer record and all original files untouched.

**Pin pagination to a generation.** Revision arguments are optional on the first
read. Every successful record page returns `archive_revision`. Echo that exact
value in the next call's `archive_revision`, together with `next_offset_bytes`.
Pass it when switching from a record page to a sidecar read as well, so the
selected pointer still belongs to the same transcript generation.

A sidecar page returns both `archive_revision` and `result_revision`. Echo both
values on each sidecar continuation, retaining the same `archive_id` and
`result_record_offset` and advancing `offset_bytes` to `next_offset_bytes`.
`result_revision` is valid only with `result_record_offset`. Revision values are
opaque 64-character lowercase hexadecimal strings; copy them, do not invent them.

The reader checks file generation before reading and again before returning.
A detected in-read mutation or a revision mismatch fails safely with
`archive_changed`, not a successful mixed-generation page. Do not continue
combining earlier pages after that error; start a fresh read or have the host
provide stable originals. Omitting a revision token starts a fresh view of that
file; it does not provide an automatic cross-page snapshot. Tokens fingerprint
file stat attributes, not full content: they detect ordinary changes and
replacement, not authenticity or malicious owner forgery. Preflight does not
guarantee that later opens see unchanged files.

**Response limits.** `max_bytes` defaults to **65,536 bytes (64 KiB)** and has a
hard maximum of **1,048,576 bytes (1 MiB)**. It bounds the JSON-encoded response,
including metadata, JSON escaping, and base64 overhead; it is not the source-byte
allowance. Complete raw records are returned only when they fit within this
bound. A single oversized record fails visibly with `archive_record_too_large`;
retry with a larger page budget only within 1 MiB. Records that still cannot fit
are not retrievable through this API. A budget too small for sidecar-page metadata
and data fails with `archive_page_too_small`. Bounded pages are not a promise of
unlimited full-fidelity retrieval.

The tool does not rewrite, migrate, delete, execute or replay saved data, and
does not automatically import it into canonical history. Missing/corrupt files
and unsafe sidecar references fail visibly rather than skipping records.