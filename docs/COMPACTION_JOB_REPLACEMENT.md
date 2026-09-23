# Checkpoint preservation after background job completion

`loop-live` replaces a pending tool receipt with its completed result by calling
`context.set_messages`. That is an authoritative history replacement, but it often
changes only the current, uncompacted tail. Discarding the checkpoint in that case
causes the same historical prefix to be compacted again at the next request.

The boundary manager now keeps an existing native or semantic checkpoint only
when its covered canonical prefix has the same canonical JSON digest. This uses
the same digest as durable checkpoint validation, including metadata and scalar
types. A changed/reordered prefix, shortened covered history, or unverifiable
prefix invalidates it. Appending or shortening only the uncovered tail is safe.

Every replacement still advances the history revision, rejects older in-flight
compaction/request results, and clears whole-history evidence references so the
host refreshes them. Provider/model identity, explicit retention, native-envelope
validation and authoritative token-count checks still run at request preparation.
The original history and newly completed results remain intact.

## Reproduce without a model or app

Use an environment that already contains context-managed, Core, context-simple
and loop-live:

```sh
python scripts/validate_compaction_job_replacement_offline.py --output candidate.json
```

For a baseline run, use the old installed context and add
`--allow-baseline-failure`. For the candidate, set `PYTHONPATH` to the candidate's
`modules/context-managed` directory. The script imports the real
`BundleLiveOrchestrator._synchronize_job_results` method, blocks socket access and
uses only synthetic provider responses. It neither starts an app nor accesses a
saved session.

The recorded 2026-09-23 comparison exercised 50 completed jobs on the same covered
prefix, followed by checkpoint restore and one further tail update:

| Path | Before: calls after 50 jobs / restore | After | Replacement median |
| --- | --- | --- | --- |
| Native, 3.6 million source characters | 51 / 52 | 1 / 1 | 20.015 ms |
| Text summary, 30,000 source characters | 51 / 52 | 1 / 1 | 0.312 ms |

The new prefix comparison adds bounded local work; its purpose is to avoid an
otherwise unnecessary provider compaction, not to make the replacement itself
faster. Exact completed results survived every replacement; changing a covered
objective still invalidated the checkpoint. Source hashes and timings are in
[`evidence/compaction-job-replacement-20260923.json`](evidence/compaction-job-replacement-20260923.json).

The module suite covers both checkpoint types, append/truncate/replace behavior,
metadata/type changes, cancellation, concurrent supersession, evidence refresh,
restore, and downstream safety checks. This is an offline integration regression,
not new evidence of a model's summary quality or the separate large live-chat
acceptance.
