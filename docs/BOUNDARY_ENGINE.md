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

Request fitting delegates to context-simple's tracked `main` branch. Its
provider-count callback accounts for the complete assembled request, including
tools and injected context. The compaction trigger uses that full request count
when available, with a labelled estimate otherwise. Oversized
protected content can still fail rather than silently vanish.
The optional request-owner `fit_output` callback is forwarded unchanged, allowing
the shared fitter to try a smaller output reserve after context reduction while
retaining the counted dispatch and canonical history.

The summarizer is awaited at a request boundary and uses no tools. Conventional
providers can supply the same instance while no foreground inference runs. A
native live provider requires a separate host-supplied `context.summary_provider`
and `separate_summary_provider: true`. Summary traffic is marked with
`metadata.purpose: context-compaction`; `context.compacting` lets hosts exclude it
from public text streams. Native compaction is preferred when the continuation
provider exposes a validated and measured native contract. A native failure tries
the portable text summary; only a failed summary or a still-oversized request
falls back to fitting. Neither model phase has a default elapsed-time deadline.
Explicit cancellation stops preparation without starting fallback work. History
changes invalidate in-flight summaries and request views.

Native compaction is a single operation on the complete eligible history window,
not a chain of fragment summaries. On the measured path it reuses the loop's
actual request envelope, including selected model, tools, options and current
system instructions. Current-turn input and pending-operation overlays remain
outside the checkpoint; required reminders follow the returned canonical window.

For portable summaries, providers with authoritative request counts preflight
the whole eligible prefix before splitting. A fitting request is summarized once.
Only an oversized request, an explicit source cap, or unavailable authoritative
counting requires the conservative fragment path. Finished events include
`native_selection` so unsupported capabilities, unavailable measurement, real
native failure and a resumed portable draft are distinguishable.

Continuation notes remain derived state. With `durable_checkpoints: true`, a host
can preserve and restore a validated summary checkpoint alongside its canonical
history. Otherwise, summaries are recomputed after restore. Provider-owned native
checkpoints must pass transport and measurement validation before use; an invalid
checkpoint falls back to originals. The engine reuses context-simple budget
helpers as well as its public optional capabilities; record the resolved revision
and validate the contract when updating the tracked branch. It is an experimental
portable implementation, not a claim of ChatGPT quality parity.
