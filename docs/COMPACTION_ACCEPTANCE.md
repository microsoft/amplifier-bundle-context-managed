# Compaction continuity acceptance

Research and evaluation: 2026-09-21–22. All model generation used synthetic,
disposable conversations. The affected Mac session was only read; no production
session was replayed, restarted, installed into, or modified.

## Research contract and implementation choice

Primary sources consulted:

- [OpenAI compaction guide](https://developers.openai.com/api/docs/guides/compaction)
- [OpenAI conversation state guide](https://developers.openai.com/api/docs/guides/conversation-state)
- [OpenAI Agents SDK session memory cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory)

The standalone Responses compact endpoint returns the entire canonical next
input window. Retained items and encrypted items travel together, unchanged;
keeping only the encrypted item is incorrect. Its input must fit before compact
is called. Server-managed compaction has a separate request/chaining contract.
This implementation uses standalone compaction; it does not enable automatic
server-managed compaction.

Native compaction is preferred when the continuation provider explicitly supports
it and can count the complete returned window. A configured utility summarizer or
utility model must not create opaque state for a different continuation provider.
The provider contract therefore transports an opaque, model-scoped window inside
`Message.metadata`; Core preserves it and the provider expands the whole window.
This is a derived request view, never a replacement for canonical source history.
The provider also validates that a restored carrier contains its actual opaque payload; a recomputed checkpoint digest cannot make a missing transport payload valid. Providers without this optional contract use a portable natural-language note.
The two methods remain explicitly distinguishable in diagnostics and checkpoints.

Portable summaries project public messages, receipts, tool results, and input
provenance, omitting duplicated tool calls and private reasoning serialization.
Bounded source fragments update a continuation note incrementally, with output
reserve and provider input preflight. A note commits only after the complete
eligible prefix succeeds, at a boundary preserving complete tool exchanges.
The summary instructions retain the objective, corrections, constraints,
decisions, verified results, pending work, identifiers and artifact references;
quoted tool data cannot authorize new work. This applies the
cookbook's continuity/evaluation guidance, not a claim that a text note is native
model compaction.

## Diagnosed failure

Read-only native evidence showed **50 compaction attempts and 50 fallbacks**
alongside **49 successful main responses**, with no provider retry events.
The original failure records lacked exception details; those records alone
could not establish the cause.

A standalone preflight reproduction against the installed provider subsequently
raised `ContextLengthError` before generation: 929 canonical source messages
serialized to 4,525,742 bytes, yielding 4,816,079 assembled request bytes and
**1,689,336 provider-counted input tokens** against an allowance of **916,404**
(output reserve 1,500). The preflight took 1.933 seconds. No tool was replayed.
The old engine repeatedly serialized the entire oversized prefix, including
private/duplicate state, and retried after every current-turn revision.

The fix bounds each summary request, adaptively splits on input overflow even
for providers without `request_budget`, retains safe failure categories, and
suppresses unchanged-prefix permanent failures. Transient failures use an
exponential cooldown with three attempts per unchanged prefix. Changed eligible
history gets a fresh attempt. Partial notes and cancelled/stale work never commit.

Real native API testing also exposed an SDK transport bug: OpenAI SDK 3.16.1
adds unset `phase: null` to retained messages when using plain `model_dump()`.
That becomes an invalid input parameter on continuation. Serialization now uses
`exclude_unset=True`, preserving explicitly returned null fields while omitting
SDK-invented defaults. Raw and serialized synthetic canonical windows matched,
and actual count/continuation then succeeded.

## Acceptance matrix

| Area | Verified evidence |
| --- | --- |
| Original failure | Installed-provider native preflight reproduced the exact overflow category and scalar budget above. |
| Bounded summaries | Low-context provider without preflight adapts after overflow; a single large tool record preserves critical facts at both edges and its last correction. |
| Atomicity | A later-fragment failure never commits a partial note; source originals remain unchanged. |
| Repeated boundaries | Four component compact/continue cycles for each method; eight factual/steering assertions per cycle; checkpoint restart after cycle two. |
| Full runtime | Five real Core + loop-live turns for each method, with actual fixture tools, steering while a tool is active, and restart after turn three. |
| Tool integrity | Each of five unique fixture operations executes once; every call has a matching result. Canonical earlier messages, including tool output and metadata, compare unchanged each turn. |
| Native identity | Native state uses the main continuation provider, even when a separate utility model/provider is configured. |
| Native measurement | Count the whole returned window plus new tail. Reject a larger candidate even if its carrier label is short. Tiny newly eligible prefixes wait unless the whole request needs fitting; substantial later prefixes compact. |
| Native retention | The complete canonical output is preserved, including retained items around encrypted state and call items whose results are inside encrypted state; no synthetic repair is inserted inside it. Required persisted reminders occur once outside the native window. |
| Restart / rollback | Valid checkpoint restores; changed identity, source prefix, configuration or digest rejects it. An older provider without lossless transport, absent measurement, or a raising count capability rebuilds originals even below threshold. Cancellation propagates. |
| Failure recovery | Unchanged permanent failures are not retried on every tool revision; transient cooldown and three-attempt ceiling are deterministic tests. |
| Observability | Auxiliary calls are labelled Context compaction, retain usage, and cannot publish summary prose as a user-facing assistant response. Diagnostics retain only fixed host label and safe exception category, not arbitrary labels or exception bodies. |
| Accounting | Input/output/cache-read/cache-write buckets remain separate. Known summary costs are accumulated with priced-call coverage; native compact cost is explicitly unavailable, not zero. |

## Live evaluation and limits

`evidence/compaction-component.json` and `evidence/compaction-loop.json` contain
only synthetic fixture identifiers, hashes, scalar measurements and assertions.
They contain no prompts, tool output, encrypted windows, credentials, or native
production history. Both paths are compared with the same objective, facts and
latest correction: ORBIT, budget 25, Tuesday then Thursday, initial receipt R-314,
artifact path, publication still pending, preserve originals, no replay.

These are bounded long-context simulations with deliberately low compaction
thresholds, not a statistical quality benchmark or a claim to have replayed an
entire production conversation. The integrated harness uses real Core and
loop-live orchestration, a real OpenAI model, and a safe fixture tool. It proves
more than provider/component continuation; it does not claim browser UX or
production-install acceptance. The component harness never executes tools.
Native and portable paths both passed the fact assertions; native used fewer
auxiliary requests and less time for these fixtures. Exact latency and usage are
in the evidence; token counts include cache-write buckets when calculating gross
input. A lower normalized `input_tokens` does not mean the model saw only three
tokens. Provider costs are estimates from its existing pricing table, not invoices;
native compact responses do not expose a cost here, so total native cost remains
unknown.

## Configuration and compatibility

Optional boundary-engine settings (existing trigger/output settings still apply):

| Setting | Default | Purpose |
| --- | --- | --- |
| `native_compaction` | `true` | Prefer the continuation provider's optional native contract. |
| `native_min_new_tokens` | `500` | Avoid costly recompaction of a tiny eligible prefix, except when fitting is required. |
| `summary_max_source_chars` | `512000` | Maximum source fragment; also bounded conservatively by provider context and exact request count. |
| `summary_max_calls` | `16` | Bound portable text-summary attempts for one atomic summary; the optional native attempt does not consume this allowance. |
| `native_compaction_timeout` | unset | Optional caller-selected native-call deadline in seconds. By default wait until the provider completes, fails, or the user cancels. |
| `summary_timeout` | unset | Optional caller-selected deadline for the portable summary phase only. Native compaction never consumes this budget. |
| `summary_reasoning_effort` | `low` | Avoid spending the continuation model's high reasoning setting on routine notes. |
| `summary_retry_delay` | `60` | Initial transient-failure cooldown in seconds; exponential backoff, three attempts. |

`summary_target_tokens` still controls output reserve (default 1,500), and
`summarization_model` now reaches the OpenAI request correctly. A source requiring
more than the configured call limit or an explicitly selected deadline falls back to the existing fitter;
this is explicit failure, not a partial successful summary. Provider retries
remain owned by the provider and are separate from this prefix-level cooldown.

The fallback order is native compaction, then a portable text summary, then the
request fitter as a last resort if summarization fails or cannot fit the request.
Elapsed time alone does not trigger a fallback by default. Actual native provider
errors still enter the summary phase, which gets its own optional deadline.
User cancellation ends the operation without launching another model call.
Canonical messages are preserved on every path. Provider transport settings must
also permit long-running requests; this context-manager setting cannot override a
provider's own SDK or transport deadline.

The earlier shared 120-second default was incorrect for long native compaction:
it cancelled the entire preparation coroutine and skipped the text-summary phase.
The regression tests cover this ordering and cancellation with controlled providers;
they do not establish the completion time of any production native request.

No Core protocol change is required. Deploy the optional OpenAI provider transport
before or with this context-manager update; older providers get portable notes.
Unified auxiliary-call observation is independent but needed to distinguish those
calls in its UI. A rollback to an older provider invalidates native checkpoints
and rebuilds originals, never treating the short carrier label as the history.
No tool execution is part of compaction itself.

## Reproducing the checks

The tested dependency environment used Core 2.0.0, context-simple 1.0.0,
loop-live 0.2.0, OpenAI SDK 3.16.1 and Python 3.13. Package versions alone do not
identify local patches: use the matching provider/context/Unified PR revisions.
The integrated harness requires loop-live to be installed as a Core entry point.

Run context module tests with this module on `PYTHONPATH`. Provider tests use
`pytest tests -m 'not live'`; their structural mount fixtures need a non-secret
placeholder `OPENAI_API_KEY`, since mounting without any key intentionally fails.
Run Unified `tests/test_execution_events.py` and `tests/test_diagnostics.py`.

For explicitly opted-in synthetic live checks, put the matching context module,
provider and Unified checkouts on `PYTHONPATH`, use an isolated environment with
those dependencies, and supply an existing API key in the environment without
printing it:

```sh
python scripts/validate_compaction_live.py --run-live --mode native semantic --output /tmp/compaction-component.json
python scripts/validate_compaction_loop_live.py --run-live --mode native semantic --output /tmp/compaction-loop.json
```

Each script has bounded call/iteration/time limits. The runtime harness stores
its tool invocation ledger and transcript only in a temporary directory and
removes it on completion. It publishes a scalar evidence report separately.

## Recorded test results

- Context-managed module: **324 passed**.
- OpenAI provider full non-live suite: **1,121 passed, 2 deselected**; after the
  final native-envelope validator addition, the targeted compaction/budget/
  checkpoint/serialization suite passed **131 tests**. The initial full run
  without a fixture key could not mount six structural/behavioral fixtures;
  rerunning with a non-secret placeholder resolved the environment issue.
- Unified execution-event and diagnostics suites: **39 passed**.
- New Python files passed Ruff, and every change passed `git diff --check`.

| Live case | Completed boundaries | Auxiliary calls | Total compaction time | Continuity |
| --- | ---: | ---: | ---: | --- |
| Component, native | 4 | 4 | 43.017s | 32/32 assertions; restart passed |
| Component, semantic | 4 | 11 | 158.661s | 32/32 assertions; restart passed |
| Core + loop-live, native | 3 | 3 | 31.706s | 40/40 assertions; restart passed |
| Core + loop-live, semantic | 4 | 12 | 197.683s | 40/40 assertions; restart passed |

The integrated final native run additionally validated actual carrier payloads
before continuation and restore. Each integrated case recorded five actual tool
invocations, five intact call/result pairs, five public assistant messages, and
no leaked auxiliary summary messages. The opaque path observed three separately
labelled auxiliary calls; the portable path observed twelve. These timings are
individual runs, not a statistical performance estimate.

## Reviewable implementation

- [Context manager and this acceptance report](https://github.com/microsoft/amplifier-bundle-context-managed/pull/6)
- [OpenAI canonical compaction transport and validation](https://github.com/microsoft/amplifier-module-provider-openai/pull/105), tested source `6e99b8cabb5d7819f31f9f0005c983e763f6b076`
- [Unified auxiliary-call observation and safe diagnostics](https://github.com/microsoft/amplifier-unified/pull/108), tested source `e745eb9a6d79e2de090bbdb01f99cab542f1be47`

The evidence JSON records these companion source revisions and SHA-256 hashes
of the context implementation files. Install the provider capability before or
with the context update; the Unified observer is independent. These PRs were
published for review without a production installation or restart.
