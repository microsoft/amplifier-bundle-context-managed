"""Synthetic regressions for the active-history whole-response budget."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from amplifier_module_tool_transcript import ReadTranscriptTool


def host_tool(messages):
    capabilities = {
        "context.history_authority": "host",
        "context.history": AsyncMock(return_value=messages),
    }
    return ReadTranscriptTool(SimpleNamespace(get_capability=capabilities.get))


def assert_bounded(output):
    # Budget is the ASCII JSON encoding of the output string, including quotes.
    # The enclosing ToolResult / host receipt has separate, host-owned overhead.
    chars = len(output)
    utf8_bytes = len(output.encode("utf-8"))
    json_bytes = len(json.dumps(output, ensure_ascii=True).encode("ascii"))
    assert chars <= 100_000
    assert utf8_bytes <= 100_000
    assert json_bytes <= 100_000


def large_messages(kind):
    messages = [
        {"role": "user", "content": "needle: inspect the result"},
        {"role": "assistant", "content": "Reading.", "tool_calls": [
            {"id": "call_original", "name": "read_transcript", "input": {}}
        ]},
    ]
    if kind == "many_messages":
        messages.extend(
            {"role": "tool", "tool_call_id": f"call_{i}", "content": "x" * 101}
            for i in range(1500)
        )
    else:
        content = {
            "giant": "x" * 2_000_000,
            "unicode": "\U0001f642\u00e9\u6f22" * 30_000,
            "escaped": '"\\\n\x00' * 30_000,
            "nested": json.dumps({"output": "--- Turn 1 ---\n" + "x" * 2_000_000}),
        }[kind]
        messages.append({
            "role": "tool", "tool_call_id": "call_original", "content": content,
        })
    messages.append({"role": "assistant", "content": "OMITTED_TAIL"})
    return messages


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["giant", "unicode", "escaped", "nested", "many_messages"])
@pytest.mark.parametrize("selector", [{"search": "needle"}, {"start_turn": 1, "end_turn": 1}, {}])
async def test_first_matching_turn_is_bounded_and_history_unchanged(kind, selector):
    messages = large_messages(kind)
    original = copy.deepcopy(messages)
    result = await host_tool(messages).execute(selector)

    assert result.success
    assert_bounded(result.output)
    assert result.output.startswith("--- Turn 1 ---\n[user] needle")
    assert "[assistant] Reading." in result.output
    assert "[tool result: call_" in result.output
    assert "OMITTED_TAIL" not in result.output
    assert "incomplete" in result.output.lower()
    assert "canonical history is unchanged" in result.output.lower()
    assert "cannot recover omitted content within an oversized turn" in result.output
    assert "Showing turns" not in result.output
    assert messages == original


@pytest.mark.asyncio
async def test_file_history_bound_preserves_original_bytes(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    original = "\n".join(json.dumps(m) for m in large_messages("giant")).encode()
    transcript.write_bytes(original)
    capabilities = {"context-managed.transcript_path": str(transcript)}
    result = await ReadTranscriptTool(
        SimpleNamespace(get_capability=capabilities.get)
    ).execute({"search": "needle"})
    assert result.success
    assert_bounded(result.output)
    assert "incomplete" in result.output.lower()
    assert transcript.read_bytes() == original


@pytest.mark.asyncio
async def test_later_matching_turn_keeps_number_and_range_semantics():
    messages = [
        {"role": "user", "content": "nonmatching"},
        {"role": "assistant", "content": "excluded answer"},
        *large_messages("giant"),
        {"role": "user", "content": "needle later"},
        {"role": "assistant", "content": "third answer"},
    ]
    tool = host_tool(messages)
    result = await tool.execute({"search": "needle"})
    assert_bounded(result.output)
    assert result.output.startswith("--- Turn 2 ---")
    assert "excluded answer" not in result.output
    assert "--- Turn 3 ---" not in result.output
    later = await tool.execute({"start_turn": 3, "search": "needle"})
    assert later.output == "--- Turn 3 ---\n[user] needle later\n[assistant] third answer"
    no_match = await tool.execute({"end_turn": 1, "search": "needle"})
    assert no_match.output == "No matches found."


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", [-1, 0, 1])
async def test_exact_encoded_boundary_includes_headers_and_notice(delta):
    messages = [
        {"role": "user", "content": "q"},
        {"role": "tool", "tool_call_id": "id", "content": "x"},
    ]
    prefix = "--- Turn 1 ---\n[user] q\n[tool result: id]\n"
    overhead = len(json.dumps(prefix))
    messages[-1]["content"] = "x" * (100_000 - overhead + delta)
    result = await host_tool(messages).execute({})
    assert_bounded(result.output)
    if delta <= 0:
        assert result.output == prefix + messages[-1]["content"]
    else:
        assert "incomplete" in result.output.lower()


@pytest.mark.asyncio
async def test_search_tail_match_is_returned_before_bounded_turn_rendering():
    messages = large_messages("giant")
    messages[-2]["content"] += " only_at_end"
    result = await host_tool(messages).execute({"search": "only_at_end"})
    assert_bounded(result.output)
    assert result.output.startswith("--- Turn 1 ---")
    assert "Search match in message" in result.output
    assert "only_at_end" in result.output
    assert "matching text may be omitted" in result.output


@pytest.mark.asyncio
async def test_regex_and_cross_message_matching_are_unchanged():
    messages = [
        {"role": "user", "content": "first [literal"},
        {"role": "assistant", "content": "second"},
    ]
    tool = host_tool(messages)
    expected = "--- Turn 1 ---\n[user] first [literal\n[assistant] second"
    for search in ("FIRST", "[literal", r"literal second$"):
        result = await tool.execute({"search": search})
        assert result.output == expected


@pytest.mark.asyncio
async def test_formatting_stops_before_omitted_messages():
    class Unformattable:
        def __str__(self):
            raise AssertionError("formatted an omitted message")

    messages = large_messages("giant")
    messages.append({"role": "tool", "content": [Unformattable()]})
    result = await host_tool(messages).execute({})
    assert_bounded(result.output)
    assert "incomplete" in result.output.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool_blocks", "assistant_calls", "text_blocks"])
async def test_aggregate_content_blocks_are_bounded(kind):
    if kind == "tool_blocks":
        message = {"role": "tool", "tool_call_id": "blocks", "content": [
            {"type": "text", "text": "\u6f22" * 101} for _ in range(1500)
        ]}
    elif kind == "assistant_calls":
        message = {"role": "assistant", "content": [
            {"type": "tool_use", "id": f"call_{i}", "name": "grep",
             "input": {"pattern": "x" * 100}} for i in range(1500)
        ]}
    else:
        message = {"role": "assistant", "content": [
            {"type": "text", "text": "x" * 101} for _ in range(1500)
        ]}
    result = await host_tool([{"role": "user", "content": "q"}, message]).execute({})
    assert_bounded(result.output)
    assert "truncated" in result.output


@pytest.mark.asyncio
async def test_private_reasoning_blocks_not_exposed():
    result = await host_tool([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [
            {"type": "thinking", "text": "PRIVATE_THINKING"},
            {"type": "reasoning", "text": "PRIVATE_REASONING"},
            {"type": "redacted_thinking", "text": "PRIVATE_REDACTED"},
            {"type": "text", "text": "Visible answer"},
        ]},
    ]).execute({})
    assert "PRIVATE" not in result.output
    assert "[assistant] Visible answer" in result.output


@pytest.mark.asyncio
async def test_null_assistant_content_preserves_call_and_result():
    result = await host_tool([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_null", "name": "grep", "input": {"pattern": "q"}},
        ]},
        {"role": "tool", "tool_call_id": "call_null", "content": "answer"},
    ]).execute({})
    assert result.output == (
        '--- Turn 1 ---\n[user] q\n[tool_call: grep({"pattern":"q"})]'
        '\n[tool result: call_null]\nanswer'
    )