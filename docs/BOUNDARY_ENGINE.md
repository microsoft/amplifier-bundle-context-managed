# Request-boundary context engine

Select `session.context.config.engine: boundary` to use the new engine. Legacy
rolling-summary behavior remains available as the existing default for callers
that have not opted into the experimental Work profile.

The host owns persistence. `get_messages()` returns detached, complete original
messages; `set_messages()` is authoritative and invalidates derived summaries.
No private transcript is loaded or written by the boundary engine. The existing
read_transcript tool discovers `context.history` with
`context.history_authority: host`, so retrieval uses that same canonical history.

Before the next foreground request, the engine may pause to summarize settled old
turns. It emits `context:compaction_started` and `context:compaction_finished`
with completed, fallback, cancelled, or superseded outcomes. Hosts can display
that pause while their input inbox continues accepting messages. The next safe
request boundary consumes those messages; this is not universal mid-inference
interruption. Observability failures cannot strand compaction state.

The first human objective, current turns, system/developer instructions, requested
reminders, and active-operation identities have explicit retention paths. A
continuation note is reference data at user-message priority, never a new system
instruction. Unknown or queued tool calls restrict semantic summary boundaries.
Original tool outputs are retained even when the request fitter clips its view.

Request fitting delegates to the pinned context-simple implementation. Its
provider-count callback accounts for the complete assembled request, including
tools and injected context. The semantic-summary trigger itself uses an estimate;
final request validation uses provider measurement when available. Oversized
protected content can still fail rather than silently vanish.

The summarizer is awaited at a request boundary and uses no tools. Conventional
providers can supply the same instance while no foreground inference runs. A
native live provider requires a separate host-supplied `context.summary_provider`
and `separate_summary_provider: true`. Summary traffic is marked with
`metadata.purpose: context-compaction`; `context.compacting` lets hosts exclude it
from public text streams. Summary failure falls back to the existing request
fitter. History changes invalidate in-flight summaries and request views.

Current limits: continuation notes are derived in-memory state and are recomputed
after restore; there is no provider-opaque compaction backend or durable summary
checkpoint. The engine reuses pinned context-simple budget helpers as well as its
public optional capabilities; upstream changes need contract validation. It is an
experimental portable implementation, not a claim of ChatGPT quality parity.
