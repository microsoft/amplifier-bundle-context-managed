# Native provider compaction audit

Audited Microsoft provider modules and primary vendor documentation on 2026-09-22.
Exact starting revisions are recorded in [the audit manifest](evidence/native-provider-revisions.json).
Native compaction is a provider capability, not a context-management strategy.
This manager owns triggers, prefix selection, output budgets, retry suppression,
semantic fallback, and checkpoint installation. A different context manager may
use different policies or never compact.

## Findings

| Provider module | Native opportunity on its current transport | Disposition |
|---|---|---|
| OpenAI | Responses `/responses/compact` returns a canonical window; authoritative input count is available. | Existing adapter and measured native context path remain supported. |
| Anthropic | Messages on-demand compaction returns one signed block, with a count endpoint. | New adapter; full request-envelope integration in this manager. Live standalone and repeated conversation qualification. |
| Azure OpenAI | v1 Responses supports `/responses/compact`; deployment names do not identify capabilities. | New explicit opt-in mechanism. Automatic native selection remains unavailable without authoritative counting. |
| GitHub Copilot | SDK persistent sessions offer automatic compaction and experimental `session.rpc.history.compact()`. | Current provider creates/destroys ephemeral sessions and deliberately disables SDK-owned compaction. A portable checkpoint export/import contract or a separately designed persistent-session transport is needed. |
| Gemini | Live has context-window compression; Managed Agents has server-owned compaction. | Current provider uses `generateContent`. Neither is a verified detached compact/continue operation for that path. Keep semantic compaction. |
| OpenAI ChatGPT | Codex source uses a streaming Responses compaction trigger with retained-history bookkeeping. | Subscription backend is not the public API-key Responses endpoint. This audit has not verified a supported detached canonical-window contract or authoritative counter for this adapter. Do not guess an endpoint or copy Codex's retention policy into the provider. |
| LiteLLM | `compact_responses` forwards native OpenAI compaction. | Current module uses `acompletion`, not Responses. Requires a Responses transport and opaque continuation serialization before enabling this mechanism. |
| vLLM | OpenAI-compatible serving does not itself guarantee `/responses/compact`. | No verified standalone compaction contract for the current supported server path. Keep semantic compaction. |
| Ollama | Chat API has no verified signed/opaque compact-and-resume operation. | Keep semantic compaction. |
| Chat Completions | Generic compatible endpoints expose no common compaction contract. | Keep semantic compaction; require a verified vendor-specific adapter. |
| Mock | No vendor backend. | Test double only. |

“No verified contract” is an audit result, not a claim that a vendor can never add
one. Prefix caching, token truncation, sliding windows, and an LLM-written summary
are not interchangeable with a portable native checkpoint.

Primary references:

- [OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction)
- [Anthropic on-demand compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)
- [Anthropic preserved thinking](https://platform.claude.com/docs/en/build-with-claude/compaction-thinking-blocks)
- [Azure Responses](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses?view=foundry-classic)
- [Copilot session persistence](https://docs.github.com/en/copilot/how-tos/copilot-sdk/features/session-persistence)
- [Copilot SDK compatibility](https://docs.github.com/en/copilot/how-tos/copilot-sdk/troubleshooting/compatibility)
- [Gemini generateContent and Live API](https://ai.google.dev/api/generate-content)
- [Gemini Managed Agents](https://ai.google.dev/gemini-api/docs/managed-agents-quickstart)
- [OpenAI Codex compaction request implementation](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact_remote_v2_attempt.rs)
- [LiteLLM Responses compaction](https://docs.litellm.ai/docs/response_api_compact)
- [vLLM compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [Ollama Chat API](https://docs.ollama.com/api/chat)

## Context boundary changes

The measured-request callback already builds the actual `ChatRequest`. Native
compaction now reuses its model, tools, and current system prompt, replacing only
the conversation with the eligible prefix. Current-turn overlays stay outside
that prefix. The system-prompt factory runs once for both compaction and dispatch.
Candidate counting uses the same full request construction path.

Providers may declare `native_compaction_requires_request_context = True`. The
legacy messages-only context entrypoint then uses semantic compaction. Requests
built from a full template carry `metadata.native_compaction_request_context = True`.
The Anthropic adapter rejects older context-managed requests lacking that marker.
Provider code supplies no summary prompt or compaction trigger.

The boundary engine's `native_compaction_max_output_tokens` defaults to 4096;
`summary_target_tokens` remains the semantic-summary setting. Required persisted
reminders follow the native checkpoint so the signed block remains the first
conversation item. Native state still requires a measured reduction before it is
installed; errors leave canonical history available for semantic fallback.

## Qualification and limits

`scripts/qualify_anthropic_native.py` is an opt-in live synthetic test. Install the
new Anthropic adapter and this manager, supply `ANTHROPIC_API_KEY`, and run the
script. It uses no saved user conversations and never executes tools. Set
`COMPACTION_TEST_MODEL` to qualify another admitted model.

The Sonnet 5 run retained the original codename, corrected port, and latest color
across three native cycles and checkpoint JSON save/reload, with originals intact:

| Cycle | Full request input before | After |
|---|---:|---:|
| 1 | 12,062 | 2,785 |
| 2 | 7,852 | 3,407 |
| 3 | 8,473 | 3,536 |

This is continuation/transport evidence, not a universal summary-quality benchmark
or full-context stress test. No live Azure deployment was available; its mechanism
is transport-tested and opt-in. No live host installation or restart was performed
for this audit. The remaining providers retain their existing semantic path.
