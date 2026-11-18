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
even when false or null. A configured positive `summary_target_tokens` supplies
soft guidance to rewrite the whole cumulative note near that visible-token target,
not append that many tokens per fragment. Required facts take priority; no text is
truncated and no target adherence or semantic-quality guarantee is made.
Portable notes use `summary_max_output_tokens` (an absent-only advertised-profile
policy: 32768 for context >=131072/output >=32768, otherwise 8192 bounded by
smaller advertised output). Explicit positive allowances are preserved. Provider
metadata is advisory, not selected-model capability proof; the final provider
still owns actual ceilings and full-request admission. Existing timeout
settings remain ignored. See [configuration](../../docs/CONFIGURATION.md).

Portable whole-prefix preflight recognizes a raw provider count below its
safety-padded estimate; admission still compares the padded estimate, not the
smaller raw count. It also reserves the request's output allowance (or the
validated effective allowance returned for that exact request). Providers with
an already output-subtracted input limit can therefore be conservatively
double-reserved and fragment earlier. A model with insufficient room even for
instructions, the prior note and that reserve stops safely; reducing input alone
cannot repair an impossible output allowance. Explicit smaller allowances remain
available. This does not change native/foreground accounting or permit an
uncounted reactive retry after provider overflow.

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

Export may reuse a bounded, exact witness of unchanged covered history. If the
whole witness exceeds its 8 MiB budget, only complete leading rows are retained;
the remaining rows are still checked and canonically hashed on every export.
Borrowed nested edits, row order, scalar/container types and serializer settings
must invalidate reuse. The digest and restore validation remain the canonical
whole-prefix JSON digest, including ordinary serializer errors for unsupported
values. This is a private optimization, not a second history store or a weaker
checkpoint boundary. Performance checks must include many small rows after large
leading results, so exceeding the witness budget cannot silently restore repeated
whole-history serialization. Host transcript writes and partial-summary progress
validation are separate work and are not reduced by this cache.

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
On every later admitted preparation, including an unchanged eligible boundary,
an unfinished portable job continues **only** when canonical prefix identity,
current legal retention/protocol boundary, reconstructed projected-source
fingerprint, provider/model, generation/prompt contract, parts digest and pending
segment chain all match. Its fingerprint is compared, never rebound to new source.
The entire new tail remains outside that job and verbatim in the assembled request;
final fitting may still refuse it. No second compaction batch runs automatically.
Changed retention/reminder projection or an unreconstructible old job stops with
`summary_progress_incompatible` before buying replacement work. Keep the original
paid checkpoint and explicitly decide whether recomputation is appropriate.
This conservative reuse needs no record migration and does not promise compatibility
for any particular private checkpoint. Existing v4/v5 import validation stays unchanged:
a rejected import is not continuation permission. Incompatible already-admitted
paid work is quarantined rather than replaced, including after covered history,
configuration or identity changes. In-flight stale responses are never committed.
Export preserves admitted restored partials before the first resumed call. It
raises `summary_progress_incompatible` for quarantined paid work instead of
returning an empty or completed-only replacement. Hosts must retain the previously
acknowledged checkpoint after refusal. Invalid imports are transactional: they do
not discard previously admitted progress or its compatible completed note.

`summary_max_calls` still bounds each preparation (default 16). Exhaustion raises
nonretryable `summary_work_paused`, emits `context:compaction_finished` with
`outcome: paused`, and retains acknowledged completed pieces without installing a
provider-failure latch. It does not mean ready or failed SDK work. No foreground
request is returned at the pause. Hosts must not automatically retry it or use
timers, observer events or restart alone to buy the next batch. Another preparation
requires the host's ordinary explicit continuation/cost admission. Context does not
authenticate that admission: the bound is per preparation, not a lifetime spending
limit. Pause status itself is observational; only pieces acknowledged by the
existing persistence callback are claimed durable. No new durable authorization
token, pause record or checkpoint schema is introduced.

Semantic progress and finished events additionally report `source_chars_total`,
`source_chars_completed`, `source_chars_pending`, `pending_segments`,
`largest_pending_segment_chars`, `partition_final`, `source_character_basis`,
`calls_this_pass`, `call_allowance`, `summary_calls_total` and
`summary_work_disposition`. Characters are Python-string offsets over the initial
serialized summary-source parts (including initial part labels), excluding repeated
prompts and cumulative-note carry. Splits conserve total = completed + pending.
Pending segments can subdivide: completed calls / (completed + pending segments)
is **not** a completion fraction, ETA or future call budget. `partition_final` is
false while source remains; `source_covered` still precedes reduction/admission and
checkpoint installation. Only `compaction_finished/outcome: completed` establishes
completed compaction, not a progress event with zero pending segments.

Target-guided partial records use configuration version 5, additionally binding
the rendered prompt digest; older readers reject them. Without a configured target,
partial records keep version 4 and the exact legacy prompt. Version 4's existing
model/output/effort contract remains readable but cannot silently resume target-guided
work. A prompt-policy correction does not invalidate otherwise compatible completed
portable v2/native v3 notes or discard the valid completed note beside old partial
work. It does reject incompatible partial work: preserve that paid record, and
expect recomputation from originals under the new policy rather than free reuse.
No retry, migration or recomputation is automatic. Explicitly choosing to replace
legacy unbound or changed generation contracts can incur generation cost again.
Completed portable v2/native v3 notes retain their prior contract.
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

The optional `context.measured_request_view` capability accepts
`reason="provider_overflow"` from an orchestrator that has consumed valid,
provider-owned rejection feedback. This requests compaction of the next legal
settled prefix even when public semantic pressure is below the normal trigger:
opaque signed state can dominate the rejected wire request. It never extends the
eligible boundary, changes canonical originals, includes private reasoning in a
portable note, or bypasses native compaction requirements. Other reasons fail
before preparation. Ordinary measured requests keep their existing trigger.

The reason is not proof of a fit. The orchestrator must require an authoritative
provider count of its complete exact replacement request, a changed input view,
and a reduction from the rejected input before any bounded foreground retry.
An unavailable or estimate-only count cannot authorize that retry.

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

Native checkpoints record canonical `omittedIndices` for covered messages kept
outside the native window. An earlier carried instruction stays in subsequent
requests even after newer inputs replace it in the latest-two carry set. Its
offset is removed only when a later accepted native compaction includes that
message; failed or cancelled candidates preserve the previous offsets. Canonical
history and the provider's opaque native payload are not rewritten.

These checkpoints use native-summary version 2 and checkpoint configuration
contract 3. Older readers reject that contract instead of ignoring omission
provenance. Compatible legacy native checkpoints without recovery exclusions
remain readable. Legacy explicit recovery with a nonempty exclusion hash but no
offsets is rejected because its retained messages cannot be reconstructed safely;
original history remains available.

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

With durable checkpoints, `context.recovery.recover` exposes the explicit
request-level entry. It uses a detached context and the caller's count-only
`count_view` to verify the complete ordinary request, not just each compact input.
Native counts must strictly decrease; the conservative hard-safety estimate must
fit the unchanged allowance. Current instructions, retained reminders, recent
history and operation observations remain in the final request. An indivisible
oversized exchange, unresolved exchange or still-oversized required suffix can
legitimately refuse recovery without changing the accepted checkpoint.

The host supplies exclusive custody, a synchronous `guard`, `save_progress`,
and `install(record, verification)`. Installation must acknowledge the exact
record digest (a string or `{"sha256": digest}`); only then does the module
activate prevalidated state. Cancellation, a failed acknowledgement, or changed
authority leaves active state untouched. Storage may nevertheless have committed:
the host must reconcile its receipt before another attempt, not replay the write.
Progress records bind retained canonical indices and are not accepted checkpoints.

A host can register `context.recovery.run_pending` to consume an **explicit
one-shot intent** at the beginning of a later independently authorized measured
preparation. It receives the real `provider`, `retain_contents`, and `count_view`.
No intent returns `None` and leaves ordinary preparation unchanged. Success
returns the measured-view result; failure stops that preparation. An optional
`source_revision` bounds recovery to the authorized canonical prefix while
appended input stays outside it. The host owns intent persistence, source/model/
configuration/Stop checks, and uncertain-outcome reconciliation. Overflow must
not create or re-arm an intent, submit input, resume a task, or replay a tool.

### Deliberate foreground model transitions

Selecting another model alone does not run a model or migrate a checkpoint.
Ordinary preparation can count and return fitting canonical originals, keeping
the incompatible completed source unit inactive and intact without generating
or persisting a replacement. An oversized originals-only request still refuses;
it never starts an automatic migration campaign. Incompatible
paid portable progress retains the existing `summary_progress_incompatible`
refusal. A host may admit a bounded transition on a genuine foreground input
through the same `context.recovery.recover` entry, before ordinary preparation.
This is not an automatic overflow fallback or permission to retry UNKNOWN work.

`context.checkpoint.transition(record, identity)` is passive. It checks record
integrity, summary shape, current canonical prefix and generation configuration
on a detached context. It returns structured `transition_required` provenance,
`compatible` for the same identity, or `rejected`. Its `validation` coverage
explicitly reports `sourceProviderEnvelope: unverified`: a self-declared source
identity is not source-provider attestation. The host must bind exact previously
accepted source bytes/provenance and the deliberate target selection. No source
opaque checkpoint is sent to the target, even for counting.

For a transition, supply `source_checkpoint=<exact prior record>`, or the distinct
checkpointless fork contract
`transition_source={"kind": "canonical-originals", "sourceRevision": {"messages": N, "sha256": digest}}`.
Never supply both or substitute canonical-originals for a rejected saved record.
Optional `source_revision` binds the full journal-admitted prefix, which can be
longer than source-checkpoint coverage; later appended input stays outside it.
The host retains the source summary, identity, omission/recovery metadata and paid
partials together in its existing recovery transaction, not a per-model bank or
a second transcript store.

The existing `guard`, `save_progress(record, report)` and exact-digest `install`
callbacks still apply. A mandatory `admit_call(report)` additionally reserves each
logical generation operation durably **before** transport and must return exactly
`True`. Reports contain `method` (`native` or `semantic`),
`sourceCheckpointSha256` (null for canonical-originals), `targetIdentity`,
one-based local `sequence`, `requestSha256` and `maxOutputTokens`. Progress saves
include that exact report in `call` and `disposition: completed`; only an
acknowledged save establishes resumable progress. False/exception/cancellation
does not authorize another call. Hosts enforce a finite cumulative allowance
across restarts and repeated preparations using `summary_max_calls` (default 16).
`max_calls` supplies this invocation's remaining allowance, from zero through the
configured limit; zero permits only no-generation fitting/reconciliation. Existing
bounded provider transport retries are not additional logical calls.

Transitions preserve originals and derive fresh target state on a detached
context. Before generation they count the full irreducible current
instructions/tools/options and required recent tail. Native targets reuse
chronological whole-exchange batching; archive-recent selection does not stand in
for full eligible-prefix coverage. Portable targets reuse the existing public
projection, summary prompt/output policy and exact partial-progress validation.
Native errors never fall back to portable summaries. The final full target request
must fit the unchanged conservative allowance and authoritatively reduce its
originals-only baseline before checkpoint installation. Guards fence callbacks,
paid operations and final activation; source state survives failures/cancellation.

When originals already fit, the result is `outcome: transition_fitted` with
`recovery.status: prepared-originals`: zero paid calls, no `install(None)` or fake
digest acknowledgement, and no checkpoint-format change. The host keeps the exact
source record quarantined and marks that transaction prepared, so later ordinary
target requests cannot re-arm the paid campaign. A completed target checkpoint
instead returns `outcome: transitioned` / `recovery.status: installed`.
Both results report exact `sourceRevision`, `targetIdentity`, `requestSha256`,
`before`, `after`, and `limit`; installed results also report checkpoint `sha256`.
The loop's `dispatch` object is not rewritten: an absent request model is accepted
only when the guarded selected provider's default matches the exact target model.
An explicit different request model is rejected.

Confirmed target progress may be supplied through `progress` only under the same
host journal binding. For a final installed candidate, additionally pass
`installed_candidate_sha256=<exact journal-pinned digest>`. This distinct readback
contract preserves the supplied record's exact coverage/digest even when new
foreground input advances the eligible boundary. It never extends generation,
even with a positive remaining allowance. An intermediate native checkpoint or
portable partial is not a final installed candidate. `install` must acknowledge
the existing digest readback without repeating its write. A running reservation,
lost response or uncertain install must first be
reconciled by the host. Saved progress alone never settles UNKNOWN or authorizes
replaying a model operation, checkpoint write, tool, task resume or Stop clearance.
If the final host journal write failed after Context had already activated the
target, the same pinned readback can reconcile only when the complete active
export exactly equals that candidate and no active paid partial is present.
Other active target records do not bypass source-unit validation.
