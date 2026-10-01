
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
shortened, corrupt, differently configured, or differently identified history
rejects the derived summary with a visible reason and `originalsAvailable: true`.
It never restores/replays a tool or changes original messages. `set_messages`
always invalidates derived state; an explicit compatible restore can follow it.
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

Portable notes first preflight the complete eligible prefix when the provider
offers authoritative counting. A fitting prefix uses one summary request;
oversized requests split into bounded fragments. Providers without authoritative
counting keep conservative limits, and an explicit `summary_max_source_chars`
still bounds portable requests. Notes commit atomically and suppress repeated
unchanged-prefix failures. Explicit
`context:compaction_finished` evidence identifies the method, safe failure
category, elapsed time, call count and usage, including cache buckets. Raw
provider exception bodies are not copied to these events. `native_selection`
records why native compaction was selected or why a portable fallback was used.

See the [research, acceptance evidence and configuration](../../docs/COMPACTION_ACCEPTANCE.md)
for actual live test limits and dependency integration order.
