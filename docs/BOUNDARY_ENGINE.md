# Request-boundary context engine

Select `session.context.config.engine: boundary` to use this engine. Legacy
rolling-summary behavior remains the default for callers that have not opted
into the Work profile; its settings and emergency policies are separate.

The host owns persistence. `get_messages()` returns detached, complete originals.
`set_messages()` invalidates derived state when its covered prefix changes; an
unchanged prefix can retain its checkpoint while the recent tail changes. No
private transcript is loaded or written here. The transcript tool uses the host's
canonical history through `context.history` and `context.history_authority: host`.

## Compaction policy

Before a foreground request, the engine can compact settled old turns at the
`summarize_trigger` fraction of the request budget (default 0.70). Measurement
includes the loop's tools, instructions and options when its callback is available.
An estimate is labelled as such and cannot validate opaque native state.

- When the continuation provider advertises native compaction, use that provider's
  native contract. Missing configuration, request context or counting support,
  invalid output, an oversized native input, and actual provider errors stop
  preparation. They **do not** start a portable summary or emergency trimming.
- Providers without native support use an LLM continuation note. The same provider
  can summarize, or the host can supply a separate `context.summary_provider`.
  A failed, empty, truncated or non-reducing note stops preparation as well.
- Request validation does not clip tool results or drop messages. A still-oversized
  request fails with `ContextLengthError`. The optional `fit_output` callback may
  reduce the output reserve, but must preserve the input.

Compaction errors preserve canonical messages and the last valid checkpoint.
`CompactionError` exposes a safe code and retryability, without putting raw SDK
errors into diagnostics. Failed unchanged prefixes retain bounded retry/cooldown
tracking, so another request cannot silently proceed or start a retry storm.
Fix the underlying provider/configuration issue or explicitly recover incompatible
old history before continuing. This engine does not automatically migrate an
already oversized legacy history by changing compaction strategies.

Hosts retaining the authoritative archive can opt into `archive_recovery: true`
along with `durable_checkpoints`. If local pressure exceeds the window, recovery
preserves any compatible native checkpoint and selects required instructions, hook context,
the original objective, and the newest complete exchanges before remote counting.
The current tail is reserved first. Earlier messages remain in the archive: the
provider receives an explicit notice that those messages are **not summarized**
in this working view, and the checkpoint records exact selected ranges.
This is bounded archive retrieval followed by native compaction, not chronological
replay of the entire archive. Normal/native-error paths never fall back to portable
summaries. Exact assembled-request counting still gates dispatch, required context
is never truncated, and a successful native result is reused on subsequent turns
and restarts. Unified enables this policy on each boundary-engine mount.

The first objective, recent turns, system/developer instructions, retained reminders
and active-operation identities have explicit retention paths. Unknown or queued
tool calls restrict the boundary; a partial tool batch is never split. A portable
continuation note is reference data, not a new instruction or authorization.

## Native window and portable notes

Native compaction receives the previous complete provider checkpoint plus the
newly covered suffix, rather than replaying the covered original transcript. On
first compaction it receives the eligible original window. The covered-message
boundary and source identity are persisted with the checkpoint. Native input must
already fit the provider's compaction allowance; triggering early is essential.

The measured path reuses the loop's selected model, tools, options and current
system instructions. Current-turn input and operation overlays remain outside the
checkpoint. Required reminders excluded from native input follow the returned
window. Returned provider items remain intact through repeat compaction and restore.

Portable summaries preserve objective, corrections, decisions, evidence and remaining
work. They preflight the whole eligible prefix when authoritative counting exists;
only an oversized input, explicit source cap or unavailable count requires ordered
fragments. Completed fragment progress can be persisted, but partial notes never
enter a foreground request.

`summary_max_output_tokens` supplies the portable summary generation allowance,
default **8192**. It is separate from the legacy `summary_target_tokens: 1500`
setting and includes reasoning where the provider accounts for reasoning there.
An output-limit finish raises `summary_output_limit`, rather than accepting an
incomplete note. Configure the allowance for the provider/model; 8192 is not a
claim of summary quality or a universal model limit. Native requests retain their
separate `native_compaction_max_output_tokens` setting (default 4096).

Neither compaction method has a context-owned elapsed-time deadline. Healthy calls
can complete; caller cancellation stops preparation. History or contract changes
reject in-flight results. Compaction emits started/finished events with completed,
failed, cancelled or superseded outcomes; an unavailable native contract can fail
before an operation starts. Hosts should surface the request error too.

## Persistence and validation

With `durable_checkpoints: true`, the host preserves and restores validated derived
state alongside canonical history. Invalid native transport or unavailable native
counting stops use of a saved native checkpoint. Provider/model or source changes
invalidate incompatible state, and the normal strict preparation rules apply.

Budget and measurement helpers come from context-simple's tracked `main`; its
emergency reduction ladder is not used by this engine. Record the resolved
revision and validate compatibility when updating dependencies. Offline scripted
checks establish transport and state contracts, not live model retention quality
or performance on production histories.

Provider contracts: [OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction)
and [Anthropic compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand).
A provider adapter must implement its own native capability; provider documentation
alone does not mean a particular installed adapter supports it.
