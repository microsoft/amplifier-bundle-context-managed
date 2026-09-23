# Offline compaction wait and continuity acceptance

These checks complement the opted-in live model evaluations in
[COMPACTION_ACCEPTANCE.md](COMPACTION_ACCEPTANCE.md). They use actual Core,
loop-live, loop-streaming, context-managed, the OpenAI provider and the OpenAI
SDK, but HTTP replies are synthetic and remain in memory. Socket connections and
DNS resolution are denied during the checks. Tools only update an in-process
counter. No credentials, paid model calls, user history or production service are
needed.

They establish transport, orchestration, history and fallback behavior. They do
not establish real model summary quality, production deployment, browser behavior
or how long OpenAI will actually take to compact a production request.

## Wait, cancellation and fallback

```sh
python scripts/validate_compaction_wait_offline.py --delay 125 --output /tmp/compaction-wait.json
```

Six cases exercise the actual provider/SDK boundary:

- Native compaction completes after 125 seconds, past the former shared deadline.
- Portable summarization also completes after 125 seconds.
- A real SDK native error goes to portable summarization before fitting.
- Errors from both native and portable calls go to a bounded request view as the
  last resort; canonical originals and the latest correction remain intact.
- Cancellation during native compaction propagates without launching fallback.
- Cancellation during portable summarization propagates without another call.

The SDK starts with an intentionally tiny default timeout. The test asserts that
actual generation/compaction request extensions have no read or write deadline;
connection and pool acquisition keep their finite limits. The HTTP transport is
synthetic, so this proves the selected SDK configuration and that our context
layer does not cancel a healthy wait; it does not emulate real socket health or
packet loss.

## Long conversation and checkpoint restore

```sh
python scripts/validate_compaction_flow_offline.py --turns 180 --output /tmp/compaction-flow.json
```

The harness mounts an ordinary Core session through its module loader and runs a
real loop-live/loop-streaming turn flow in three modes: native, portable, and
native failure followed by portable fallback. Each turn produces one synthetic
tool call and result. Halfway through each mode it saves a context checkpoint,
cleans up the session, creates a fresh session and restores canonical history plus
the checkpoint.

Each turn checks that the newest user correction reaches the provider, completed
tool receipts survive the derived context view, previous canonical messages are
unchanged, and every synthetic operation executes exactly once. Native mode also
checks that the whole previously returned native window reappears unchanged on
subsequent compact requests. Portable replies use a deterministic receipt oracle;
passing that oracle is a transport continuity assertion, not an LLM quality score.

The low trigger is deliberate: repeated boundaries make state-loss defects easier
to expose. At 180 turns per mode the current fixture produces 178 compactions per
mode, 721 final canonical messages, and one checkpoint restore per mode.

## Dependency requirements and recorded boundary

Use a separate Python 3.13 environment with the candidate context module,
OpenAI provider, Core, context-simple, loop-live and loop-streaming installed.
The module loader and Core validation remain enabled. Source checkouts may be
supplied on `PYTHONPATH`, but an installed-runtime acceptance must additionally
record each actually imported module path and source hash.

The September 23, 2026 offline run used context-managed `8162dfa`, OpenAI provider
`5cabc7d`, Core `4f53e0f`, loop-live `1d7be38`, loop-streaming `4cc86dd`, and
context-simple `7b10071`. The context module suite passed 332 tests. Both 125-second
cases passed; fallback and cancellation passed. Across the three long modes,
540 turns, 534 compactions, three restores and 540 unique tool operations passed.
The former context revision `1549edf` fails the delayed native case because its
shared 120-second deadline expires before completion.

Exact interpreter/import receipts, measured timings, request counters and
production rollout evidence belong with the deployment qualification record.
Do not treat successful offline acceptance as proof that a running worker loaded
these revisions. Check a fresh worker's actual sources after component activation,
and separately observe an authorized real conversation completing compaction.
