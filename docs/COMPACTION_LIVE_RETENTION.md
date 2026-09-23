# Real-provider retention acceptance

This complements the [offline wait and loop tests](COMPACTION_OFFLINE_ACCEPTANCE.md).
It uses installed Core, context-managed and OpenAI provider modules, but sends
new synthetic history to the configured OpenAI endpoint. It never reads a user
transcript, creates an application chat, executes a tool, or writes a production
session store. It makes paid model calls, so run it only with explicit live-test
authorization and a configured `OPENAI_API_KEY`. `OPENAI_BASE_URL`, if present,
must select the official endpoint for these native acceptance cases.

```sh
python scripts/validate_compaction_retention_live.py --run-live \
  --mode native semantic native-error-to-semantic \
  --output /tmp/compaction-retention.json
```

No finite deadline is imposed on model completion or compaction. User
cancellation still propagates. The script stops after the first failed case;
inspect its evidence before paying to repeat it. Safe event metadata is appended
to the corresponding `.events.jsonl` file so a failed assertion does not erase
the preceding native/fallback observations. API response bodies, credentials and
encrypted compaction content are not logged.

## What is tested

Each mode crosses two compaction boundaries. After the first it restores the
checkpoint into a fresh context manager, adds a newer user correction, and
continues. Ten exact assertions per boundary check identifiers, latest budget and
launch-day corrections, completed receipts, artifact references, pending
publication, unchanged production state, preservation of originals, and no
replay. Canonical source hashes must remain unchanged.

- **Native:** the real compact endpoint must return a valid native checkpoint;
  the next real completion must accept its full canonical window.
- **Semantic:** the real provider summarizes ordered evidence fragments, then a
  real continuation must answer from the retained context.
- **Native failure to semantic:** a typed native failure is injected on the exact
  provider instance; the semantic calls and continuation remain real. A trimmed
  request does not count as success.

The harness uses the exact `OpenAIProvider` class. Subclassing it would disable
its standard-endpoint native capability by design, since Azure/proxy adapters
must not inherit that capability accidentally. Instance-bound instrumentation
preserves native support and records actual native attempts. Capability
assertions fail before inference if the harness changes that contract.

## September 23, 2026 result

[Recorded evidence](evidence/compaction-retention-20260923.json) identifies the
actual installed Core `4f53e0f`, context-managed `8162dfa`, OpenAI provider
`5cabc7d`, and SDK `3.19.0`, with `gpt-6-astra` at the official endpoint.

| Mode | Real compactions | Compaction elapsed time | Retention assertions |
| --- | --- | --- | --- |
| Native | 2 | 10.181s, 14.705s | 20/20 |
| Semantic | 2 | 41.425s, 16.879s | 20/20 |
| Injected native failure to real semantic | 2 | 40.065s, 14.805s | 20/20 |

All three checkpoint restores and all original-history checks passed. The first
native operation reduced the measured whole request from 22,754 to 556 tokens;
the second reduced 7,287 to 4,698, including the recent tail. Known reported cost
for the successful run was $1.2330325; native compact operations reported usage
but no priced cost, so that figure is not the complete billed total.

Two earlier diagnostic attempts accidentally subclassed the provider and thereby
disabled native support. Their semantic completion is not native acceptance. One
of those separate diagnostic continuations passed ten retention assertions, but
neither is included in the successful run above.

These are bounded synthetic standalone quality checks with a deliberately low
compaction trigger. They do not establish a large live application conversation,
a 900,000-token request, statistical summary quality, or browser behavior. A
separate new application chat must verify that deployment-level path using
actual admitted inputs, observed compaction, and post-compaction continuation.
