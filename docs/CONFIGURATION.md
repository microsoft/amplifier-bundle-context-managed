# Configuration Reference

## Context Manager Module (`context-managed`)

| Parameter | Default | Description |
|---|---|---|
| `max_tokens` | 200,000 | Fallback context budget when no provider is given. |
| `verbatim_window_tokens` | 40,000 | Tokens of recent conversation to keep verbatim. |
| `max_summary_tiers` | 3 | Maximum rolling summary tiers before merging. |
| `summary_target_tokens` | 1,500 | Target size per summary tier. |
| `summarization_model` | session provider | Which model to use for summarization. |
| `summarization_prompt_path` | built-in | Path to custom summarization prompt markdown. |
| `summarization_retries_before_fallback` | 3 | Consecutive failures before emergency fallback. |
| `emergency_target_usage` | 0.50 | Fraction to compact down to on emergency fallback. |
| `large_result_threshold` | 50,000 | Characters before tool results get pointer-filed. |
| `pressure_warning` | 0.70 | Usage fraction that emits `context:budget_pressure`. |
| `summarize_trigger` | 0.80 | Usage fraction that triggers async summarization. |
| `emergency_fallback` | 0.92 | Usage fraction for mechanical fallback safety net. |

## Transcript Tool (`tool-transcript`)

| Parameter | Default | Description |
|---|---|---|
| `rate_limit_per_turn` | 3 | Max transcript tool calls per iteration. |
