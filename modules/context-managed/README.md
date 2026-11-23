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

### Ordinary request fitting (host opt in)

Hosts enabling both `foreground_continuation: true` and
`compaction_schedule: after_turn` select whole exchanges locally before
counting a foreground candidate. The modeled request uses the validated
checkpoint when it fits. A transient, explicitly classified counter outage
can select a smaller candidate. A fresh authoritative measurement then admits
that exact request. The latest completed tool batch stays paired with its
receipts. If its saved result alone prevents a measured fit, its result body
can carry the canonical message index and digest for retrieval.

After a successful turn the optional synchronous capability
`context.schedule_settled_compaction(provider=..., count_view=..., retain_contents=...)`
queues one session-owned task when the last turn needed less history or crossed
the configured pressure threshold. The host supplies a count callback that
remains valid for background work. The capability returns whether it queued a
task. Call `context.cancel_settled_compaction()` at Stop. The mounted context
cancels and drains its task on coordinator cleanup. New foreground work
supersedes the task. Only validated results for the same source revision and
selection can advance the checkpoint. Observers can distinguish these events
using `context_work: settled_maintenance`.

When the full foreground history no longer fits, the selector leaves room for
new steps and keeps the same earlier exchanges while the growing request fits.
Settled maintenance uses at most one model generation. It prefers the selected
provider's qualified native operation and measures the request before dispatch.
An unsupported provider can produce one portable continuation note. Token counts
are separate from model generations.

With archive recovery enabled, an oversized history can use an excerpt if the
host has preserved the originals. A compatible paid partial note can enter the
native request alongside still-unprocessed evidence. Its source and fragment
digests must match. The prior checkpoint remains available until the replacement
is validated and stored. Large new inputs or required instructions can consume
the available room sooner.

`foreground_continuation: true` makes measured ordinary requests (and explicit
`hard_fit` requests) independent of optional compaction success. It reuses the
existing whole-exchange selector and exact request fitter; it does not install a
recovery checkpoint or change canonical originals. Settled exchanges after the
latest input may be omitted, including large call arguments, results, nested
delegation observations and ephemeral recovery-reference rows. A result-only
structured user row is not ordinary input. Mixed result/text rows remain current
input and retain their complete protocol group.

The original objective, latest two actual input rows, instructions, hooks,
explicit retention, current reminder snapshot, live/queued/uncertain call groups
and current operation observations remain required. Historical unanswered calls
retain their existing explicit unknown/no-replay observation. If required context
still exceeds the model allowance, preparation fails visibly without clipping it.

Every omission-bearing request includes a developer warning with metadata
`source: context-managed-continuation` and `ephemeral: true` **before** counting.
The warning says omitted evidence is not summarized and must not be invented or
used as a reason to replay actions. The returned dispatch includes that warning;
preparation itself does not publish a host notice. The host may show its notice
only when the request is actually dispatched.

Stop/cancellation, stale request contracts, evidence-preservation and checkpoint
storage failures still stop preparation. Optional paid compaction is limited to
one invocation per actual input in this context instance; tool/service appends do
not reset it. An unconfirmed invocation suppresses all further automatic paid
compaction in that instance. This is not a durable operation journal or a
restart-safe settlement claim; hosts still own uncertain-operation recovery.
Default strict behavior is unchanged when this option is absent.

The operation carrier is captured once per preparation and counted unchanged.
For Loop Live's uniquely identified job/call rows, removal of returned jobs may
advance the observed active set without invalidating that conservative carrier.
New or changed surviving rows, ambiguous identities, and observed reappearance
still stop preparation. Disappearance is not proof of successful effects; this
is an observation guard, not a durable operation-generation fence.

A typed transient failure of optional full-history pressure counting may skip
that full view and enter bounded canonical selection. Every selected candidate
then requires an authoritative provider count; a final count error, missing count
or estimate-only answer stops preparation. This does not retry a failed final
count or authorize a model/tool call. An unchanged optional native input already
refused as oversized is not preflighted again for the same input, provider,
identity, configuration, policy allowance and exact native request. That private
negative memo never admits a foreground request or replaces final counting.

Ordinary fitting is not a model-transition transaction. After mandatory host
preservation and initial identity binding, a foreign checkpoint, paid progress
unit or retained provisional transition defers optional compaction before the
transition generator is entered. The request is fitted from canonical originals;
the source checkpoint, progress, evidence references and provisional job stay
unchanged, including an unconfirmed paid outcome. No target is committed and no
transition call is retried. Pending host recovery remains excluded while a
provisional ordinary transition exists. Hosts still own any eventual transition
admission or uncertain-operation settlement; this option does not adopt a new
model checkpoint or require a new host transaction API.

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
Trigger, retry, observer, call-allowance, `compaction_schedule`, and
`foreground_continuation` changes do not invalidate an otherwise compatible note;
summary-generation settings still do. Configuration versions 2–5 also accept a
historical digest only when it is exactly reproduced from the current summary
configuration and prompt with the former `foreground_continuation: false` or
`true` key. Restored state retains that proven hash for exact export and paid
progress resume; no record, provider identity, source, or fingerprint is rewritten.
For these versions, restore rejects a non-boolean foreground policy even when the
normalized digest matches. Passive transition classification reports the verified
source record's configuration, independent of the caller's active checkpoint.
Other configuration differences still reject. Older
checkpoints retain their original, stricter configuration validation.
Normalized omitted-engine mounts keep the configuration identity of old explicit
`engine: boundary` mounts. Legacy summary markers are not Boundary checkpoints.

Rejected incoming restores preserve the previous active derived unit. This is not
a promise that every preparation failure rolls back an intentionally invalidated
completed note: required-original retention may discard that note and its legacy
hash witness, after which fresh work uses the canonical configuration. A retained
paid partial instead refuses before that discard. The same-model native recovery
entry also refuses an active paid partial rather than snapshotting only its
completed note; reconciliation of that paid work is not implicit recovery consent.
These in-memory boundaries do not rewrite or delete the host's source checkpoint.

### Returning to Unified 0.111.4

Before upgrading, let active work settle and back up the conversation history,
checkpoint files, settings and operation records together. Keep that backup if
you may need to return to 0.111.4.

An unchanged checkpoint imported from 0.111.4 can be read again by that release
with its original model, history and settings. Once this version creates a
fresh checkpoint, the standard 0.111.4 configuration rejects it. This affects
both completed summaries and partially completed summary work. The original
messages remain available. Reading the rejected checkpoint issues no model
call and runs no tool. A later compaction can incur new charges.

If you have done more work since upgrading, save the current state separately
before downgrading. Use a copy of the pre-upgrade backup for 0.111.4. Keep the
newer checkpoint for use with this version. Before resuming actions, reconcile
any work completed after the backup with its recorded effects.

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
or persisting a replacement. Hosts explicitly implementing the journal admission
protocol register `context.transition_policy()` returning `host-managed`; the
presence of a mounted `run_pending` wrapper alone is not participation.
For other **non-durable hosts without a persistence callback**, ordinary preparation
can reuse a locally qualified checkpoint, or create a readable handoff from
the older prefix and retain recent exchanges verbatim. A provider may declare
`supports_native_compaction(*, model)` synchronously. The older no-argument
declaration remains supported. Trigger, output allowance and per-preparation call caps are
unchanged; it does not invoke chronological native recovery, archive-omission
migration or a native-error portable fallback.

Ordinary regeneration holds the source unit inactive and one provisional target
job in memory until final full-request fitting and source/selection/configuration
guards succeed. Completed target pieces/candidates survive a later failure or
cancellation within that instance and are not repurchased or rebound. A paid call
with an unconfirmed outcome refuses automatic continuation. Changed source,
configuration, or target quarantines that one job rather than replacing it.
This memory-only behavior does **not** promise restart durability or recover
private results after the host discards the Context instance.

**Durable legacy-host compatibility limit:** a nonparticipating host configured
with `durable_checkpoints` or a zero-argument `persist_checkpoint` callback can still
return fitting originals without paid work. If regeneration is required, it raises
`transition_persistence_required` **before generation**, preserving the old
checkpoint. Such hosts must integrate the existing explicit transition
`source_checkpoint`/`save_progress`/`install` transaction: the legacy callback
cannot safely save detached target pieces while retaining the source unit.
It is never repointed to a clone, called to export the old mounted context as if
it saved target progress, or treated as a fake acknowledgement. This deliberately
does not preserve c18's destructive overwrite behavior for durable model switches.
Single-window native overflow still refuses without extra recovery batches.
Incompatible
paid portable progress retains the existing `summary_progress_incompatible`
refusal. A host may admit a bounded transition on a genuine foreground input
through the same `context.recovery.recover` entry, before ordinary preparation.
This is not an automatic overflow fallback or permission to retry UNKNOWN work.


A host that records a user's explicit allowance for one exhausted transition
can test `context.recovery.additional_call_allowance() == 2` and pass
`additional_call_allowance` to `context.recovery.recover`. The argument is a
version-1 object with the host's `authorizationId`, `originalMaxCalls`,
`confirmedCalls` and `additionalCalls` (1 through 32). The host must bind the
grant to the existing session, source checkpoint, target selection and confirmed
progress. Every historical reservation consumes the cumulative allowance. An
unconfirmed reservation requires reconciliation. The component accepts at most
`originalMaxCalls + additionalCalls - confirmedCalls` in `max_calls` and
requires the restored progress. Native recovery can finish at a verified safe
boundary once the saved prefix and complete original suffix fit the measured
target request. Normal installation and exact checkpoint acknowledgement apply.

`context.checkpoint.prepared_originals(identity)` is a pure synchronous bool
attestation: the current identity getter must equal the exact target and summary,
portable progress and provisional ordinary-transition state must all be empty.
It neither clears foreign state nor validates a settled host receipt or grants
fresh model work. Compatible target-derived state uses ordinary strict validation,
not this empty-state shortcut.
An ordinary compaction failure/cooldown latch is not derived request state: it
does not invalidate this attestation and is neither cleared nor rewritten by it.

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

On the first real turn after a selection change, Managed Context verifies a
qualified native checkpoint, fitting originals, or a verified readable checkpoint
and tail before generating anything. An adapter authorizes same-connection native
reuse through its synchronous `rebind_compacted_context` method. For native
artifacts whose complete summary is portable, the source adapter can set
`metadata["context-managed:portable-summary"] = True` and validate that the
carrier's text is exactly that summary. An account switch transfers only the
readable text when all omitted rows still fit the portable projection.

For a new cross-provider handoff, the destination's complete input allowance and
explicit local limits determine how many whole recent exchanges fit alongside
the note. The summarizer receives the earlier eligible prefix, using previous
readable progress where available. The portable result is reference data.
Authoritative counts verify the full assembled target request before installation.
Discovery of additional context capacity retains completed work. An unconfirmed
native reservation retains its existing method and budget on recovery. From the
last verified prefix, continuation checks the untouched suffix at installable
batch ends and stops once it fits. Durable admission remains required for paid
work. Guards fence callbacks,
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
The host count callback must also return `effective_identity` with the exact
selected provider-instance/model after measuring through that same selected
adapter. The current identity getter must confirm it. A bare provider default,
missing/stale account binding or explicit different request model is rejected.

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
