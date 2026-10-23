# Context-managed: Boundary only

The public `mount` validates configuration and delegates to
`boundary.BoundaryContextManager`. Omitted `engine` becomes `boundary`; only that
exact string is accepted. Legacy, unknown, and null selections fail before
coordinator mutation. `ManagedContextManager` and Legacy summary/writer exports
are removed, not compatibility aliases. Direct Python callers must use the
Boundary class and host-owned history/persistence.

Remove `verbatim_window_tokens`, `max_summary_tiers`, `summarization_prompt_path`,
`summarization_retries_before_fallback`, `emergency_target_usage`,
`large_result_threshold`, `pressure_warning`, and `emergency_fallback`. They fail
even when false or null. `summary_target_tokens` is still accepted but inert;
portable notes use `summary_max_output_tokens` (default 8192). Existing timeout
settings remain ignored. See [configuration](../../docs/CONFIGURATION.md).

No private transcript is loaded, written, pointer-filed, or archived by this
manager. `set_messages` accepts the host's complete canonical history; `get_messages`
returns detached originals with metadata unchanged. Old transcripts and sidecars
remain separately accessible through host-approved `read_transcript` archive IDs;
this is read-only access, not automatic canonical import. See the
[archive usage and limits](../../docs/CONFIGURATION.md#legacy-archive-usage).

### Durable boundary checkpoints (opt in)

With `engine: boundary` and `durable_checkpoints: true`, the host must register
`context.preserve_evidence(messages)` (async) and `context.checkpoint_identity()`.
The former commits the complete original transcript before request preparation
and returns durable reference records from the host's
existing evidence store. Do not create a competing transcript writer. The latter
returns the exact `{provider, model}` identity; hosts should include the selected
provider instance and model, including any temporary pin.

The module exposes `context.checkpoint.export(identity)`,
`context.checkpoint.restore(record, identity)` and `context.checkpoint.status()`.
The JSON format is `context-managed-boundary-v1`: a derived summary and evidence
references, exact covered-prefix count/hash (`sourceRevision`), provider/model,
configuration/prompt digest, and record digest. New unsummarized suffix messages
are compatible only when the entire covered prefix still matches. Replaced,
shortened, corrupt, incompatibly configured, or differently identified history
rejects the derived summary with a visible reason and `originalsAvailable: true`.
It never restores/replays a tool or changes original messages. `set_messages`
retains completed derived notes only when their entire covered prefix is unchanged.
Trigger, retry, observer, and call-allowance changes do not invalidate
an otherwise compatible note; summary-generation settings still do. Older
checkpoints retain their original, stricter configuration validation.
Normalized omitted-engine mounts keep the configuration identity of old explicit
`engine: boundary` mounts. Legacy summary markers are not Boundary checkpoints.

Hosts can optionally register `context.persist_checkpoint()` to save each completed
portable-summary piece into the same checkpoint file. Sync or async callbacks
returning `None` (legacy acknowledgement) or `True` accept the piece. Explicit
`False` or an ordinary callback exception stops preparation with the safe code
`checkpoint_persist_failed`; an existing `CompactionError` retains its code.
Cancellation rolls back the staged piece and propagates unchanged. Failed or
cancelled saves do not advance accepted progress or emit a success progress event;
prior accepted pieces, the last valid summary, and canonical messages remain intact.
Actual model calls and billed usage are not rolled back. Hosts must make storage
writes atomic: the module cannot undo writes a callback already performed.
Without a callback, existing in-memory compaction remains available, as does a
compatible restored request view; no checkpoint writer is required.
A `progress` record holds
only the derived note, exact source identity, and remaining piece boundaries;
pending source is regenerated from the canonical transcript. After restart,
compatible progress resumes without repeating completed pieces. Partial notes
never enter a model's conversation view before covering the full chosen prefix.
Restore status and any later progress rejection provide a safe reason while
leaving originals available.
The continuation note remains labelled untrusted reference history, not fresh
instructions or authorization. This module does not own task lifecycle or storage.

Required persisted ephemeral user reminders supplied through `retain_contents`
remain verbatim, including metadata, in each assembled request even when their
canonical position falls inside a summarized prefix. They do not permanently
freeze later compaction boundaries. Ordinary required messages and unresolved
tool calls still prevent crossing their boundary. Summaries never replace the
required reminder text or grant authority from historical content.

### Bounded and native compaction

The boundary engine requires an advertised continuation provider's native
compaction contract: `supports_native_compaction()`,
`validate_compacted_context(message)`, `compact_context(request)`, and a
`request_budget` result with an authoritative full-window count. Native state
travels in message metadata and is validated before use, including after resume.
An unavailable or invalid native contract stops preparation without using a
portable fallback or deleting canonical history.
The utility summary provider/model is used only for portable notes.

Native compaction sends one complete eligible window, using the loop's selected
model, current instructions and request configuration. It never uses the text
summary's fragment limits. Retain every returned canonical item unchanged.
An oversized-input preflight stops preparation; native compaction is never split
into pieces or replaced with a portable summary after a native error.

Portable notes first preflight the complete eligible prefix when the provider
offers authoritative counting. A fitting prefix uses one summary request;
oversized requests split into bounded fragments. Providers without authoritative
counting keep conservative limits, and an explicit `summary_max_source_chars`
still bounds portable requests. Providers advertising
`completion:auto_continue:v1` receive a per-call `auto_continue: false` override
for auxiliary summaries; normal chat keeps its configured behavior. A response
ending at its output limit is billed in the evidence but cannot commit a note.
`summary_max_calls` bounds logical portable calls per preparation pass, including
failed calls. Cumulative calls across resumed progress are tracked separately;
this setting is not a lifetime allowance for an unchanged prefix.
Completed pieces remain available for compatible continuation, including durable
progress after restart. It is never an elapsed or read deadline:
healthy generation can take many minutes until completion or caller cancellation.
`context:compaction_progress` reports completed and remaining pieces without
copying source or notes into events. Notes commit atomically and suppress repeated
unchanged-prefix failures. Explicit
`context:compaction_finished` evidence identifies the method, safe failure
category, elapsed time, call count and usage, including cache buckets. Raw
provider exception bodies are not copied to these events. `native_selection`
records why native compaction was selected or why a non-native provider uses
portable summaries.

See [Boundary policy](../../docs/BOUNDARY_ENGINE.md) for current behavior. The
[historical acceptance report](../../docs/COMPACTION_ACCEPTANCE.md) records the
evaluated revisions and live limits, not universal production quality guarantees.

### Long autonomous runs

The boundary engine can checkpoint completed tool exchanges within a human turn.
It leaves four recent exchanges verbatim and never splits an unresolved tool batch
or queued background job. Current human instructions and the latest persisted
reminder snapshot remain outside newly compacted input. Older replacement reminder
snapshots stay in canonical history but leave the working request.

An unanswered tool call before later ordinary input no longer pins all
subsequent completed work. Request preparation adds a truthful unknown-outcome
observation without editing history or running tools. Exact unresolved identities
and the instruction not to replay them remain outside checkpoints; current calls
and recorded queued work remain protected. See the boundary-engine policy for
the distinction between unknown evidence and actual operation status. Ordinary
input can include host-authorized peer messages; recovery does not establish
human authorship or grant permission to repeat actions. Conflicting duplicate
call declarations remain barriers.

`context:budget_exceeded` records numeric input/allowance and boundary positions
without conversation content when required context still cannot fit.

Oversized existing histories can be recovered explicitly with `recovery.recover_native`.
This offline helper uses measured, complete exchange batches and persists resumable
progress; it never runs foreground inference or tools. The caller must preserve the
original transcript, hold exclusive session ownership, verify the finished request,
and install the derived checkpoint atomically. Ordinary request preparation never
invokes this recovery helper automatically.

The required `save_progress(record, report)` callback may be synchronous or
asynchronous. Its return value is ignored; raise an exception to reject a save.
If exporting or saving a piece raises or is cancelled, the helper restores the
last accepted summary and identity and propagates the original exception.
Canonical messages and accepted earlier pieces remain intact. Completed provider
calls are not undone, and callbacks must make their own storage writes atomic.
