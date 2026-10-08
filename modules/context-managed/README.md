
### Durable boundary checkpoints (opt in)

With `engine: boundary` and `durable_checkpoints: true`, the host must register
`context.preserve_evidence(messages)` (async) and `context.checkpoint_identity()`.
The former commits the complete original transcript before any request fitting
can clip tool output, and returns durable reference records from the host's
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
Trigger, retry, observer, and per-pass call-allowance changes do not invalidate
an otherwise compatible note; summary-generation settings still do. Older
checkpoints retain their original, stricter configuration validation.

Hosts can register `context.persist_checkpoint()` to save each completed
portable-summary piece into the same checkpoint file. A `progress` record holds
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

The boundary engine now prefers a continuation provider's optional native
compaction contract: `supports_native_compaction()`,
`validate_compacted_context(message)`, `compact_context(request)`, and a
`request_budget` result with an authoritative full-window count. Native state
travels in message metadata and is validated before use, including after resume.
An unavailable or invalid transport rebuilds the request from original history.
The utility summary provider/model is used only for portable notes.

Native compaction sends one complete eligible window, using the loop's selected
model, current instructions and request configuration. It never uses the text
summary's fragment limits. Retain every returned canonical item unchanged.
An oversized-input preflight skips the native request and uses portable recovery
directly; native compaction is never split into pieces.

Portable notes first preflight the complete eligible prefix when the provider
offers authoritative counting. A fitting prefix uses one summary request;
oversized requests split into bounded fragments. Providers without authoritative
counting keep conservative limits, and an explicit `summary_max_source_chars`
still bounds portable requests. Providers advertising
`completion:auto_continue:v1` receive a per-call `auto_continue: false` override
for auxiliary summaries; normal chat keeps its configured behavior. A response
ending at its output limit is billed in the evidence but cannot commit a note.
`summary_max_calls` bounds one preparation pass, preserves completed pieces, and
allows continuation in a later pass. It is never an elapsed or read deadline:
healthy generation can take many minutes until completion or caller cancellation.
`context:compaction_progress` reports completed and remaining pieces without
copying source or notes into events. Notes commit atomically and suppress repeated
unchanged-prefix failures. Explicit
`context:compaction_finished` evidence identifies the method, safe failure
category, elapsed time, call count and usage, including cache buckets. Raw
provider exception bodies are not copied to these events. `native_selection`
records why native compaction was selected or why a portable fallback was used.

See the [research, acceptance evidence and configuration](../../docs/COMPACTION_ACCEPTANCE.md)
for actual live test limits and dependency integration order.

### Long autonomous runs

The boundary engine can checkpoint completed tool exchanges within a human turn.
It leaves four recent exchanges verbatim and never splits an unresolved tool batch
or queued background job. Current human instructions and the latest persisted
reminder snapshot remain outside newly compacted input. Older replacement reminder
snapshots stay in canonical history but leave the working request.

`context:budget_exceeded` records numeric input/allowance and boundary positions
without conversation content when required context still cannot fit.

Oversized existing histories can be recovered explicitly with `recovery.recover_native`.
This offline helper uses measured, complete exchange batches and persists resumable
progress; it never runs foreground inference or tools. The caller must preserve the
original transcript, hold exclusive session ownership, verify the finished request,
and install the derived checkpoint atomically. Ordinary request preparation never
invokes this recovery helper automatically.
