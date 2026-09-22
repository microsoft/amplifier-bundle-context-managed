"""Native compaction uses the same request envelope as the eventual dispatch."""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_core.message_models import ToolSpec
from amplifier_module_context_managed.boundary import BoundaryContextManager


@pytest.mark.asyncio
async def test_repeated_native_compaction_preserves_request_envelope_and_excludes_current_overlays():
    manager = BoundaryContextManager(
        {
            "max_tokens": 100000,
            "summarize_trigger": 0.01,
            "native_min_new_tokens": 0,
            "compaction_notice_enabled": False,
        }
    )
    originals = [
        {"role": "system", "content": "obsolete system"},
        {
            "role": "system",
            "content": "hook instruction",
            "metadata": {"source": "hook"},
        },
    ]
    for i in range(4):
        originals += [
            {"role": "user", "content": f"turn {i}"},
            {"role": "assistant", "content": "verified result " * 2000},
        ]
    reminder = {
        "role": "user",
        "content": "keep this reminder",
        "metadata": {"ephemeral": True, "persisted": True},
    }
    originals.insert(3, reminder)
    await manager.set_messages(originals)
    factory = AsyncMock(return_value="current system")
    await manager.set_system_prompt_factory(factory)
    tools = [ToolSpec(name="read", parameters={"type": "object"})]
    compacted = []

    async def compact(request):
        assert request.model == "native-fixture"
        assert request.tools == tools
        assert request.max_output_tokens == 4096
        assert request.metadata["native_compaction_request_context"] is True
        assert [row.content for row in request.messages if row.role == "system"] == [
            "current system",
            "hook instruction",
        ]
        assert all(row.content != "CURRENT OVERLAY" for row in request.messages)
        assert all(row.content != reminder["content"] for row in request.messages)
        assert all(row.content != "obsolete system" for row in request.messages)
        compacted.append(copy.deepcopy(request))
        return {
            "kind": "native",
            "message": {
                "role": "user",
                "content": "Checkpoint",
                "metadata": {
                    "source": "context-managed",
                    "ephemeral": True,
                    "persisted": True,
                    "fixture:state": len(compacted),
                },
            },
        }

    model = SimpleNamespace(
        name="fixture",
        default_model="native-fixture",
        native_compaction_requires_request_context=True,
        supports_native_compaction=lambda: True,
        validate_compacted_context=lambda row: (
            "fixture:state" in row.get("metadata", {})
        ),
        compact_context=compact,
        complete=AsyncMock(side_effect=AssertionError("No semantic fallback")),
    )

    async def count_view(rows):
        req = ChatRequest(
            model="native-fixture",
            messages=[Message(**row) for row in rows]
            + [Message(role="user", content="CURRENT OVERLAY")],
            tools=tools,
        )
        tokens = sum(len(str(row.content)) // 4 + 1 for row in req.messages)
        return {
            "dispatch": req,
            "budget_decision": {
                "estimated_input_tokens": tokens,
                "input_limit_tokens": 100000,
                "measurement": {"kind": "provider_count", "input_tokens": tokens},
            },
        }

    for cycle in range(3):
        result = await manager.get_measured_request_view(
            provider=model, retain_contents=[reminder["content"]], count_view=count_view
        )
        final = result["final_attempt"]["dispatch"]
        assert final.tools == tools
        assert sum(row.content == "CURRENT OVERLAY" for row in final.messages) == 1
        assert sum(row.content == reminder["content"] for row in final.messages) == 1
        conversational = [row for row in final.messages if row.role != "system"]
        assert conversational[0].metadata["fixture:state"] == cycle + 1
        assert factory.await_count == cycle + 1
        assert await manager.get_messages() == originals
        if cycle:
            assert any(
                (row.metadata or {}).get("fixture:state") == cycle
                for row in compacted[-1].messages
            )
        for i in range(2):
            for row in [
                {"role": "user", "content": f"next {cycle}/{i}"},
                {"role": "assistant", "content": "new evidence " * 2000},
            ]:
                originals.append(row)
                await manager.add_message(row)
    assert len(compacted) == 3


@pytest.mark.asyncio
async def test_message_only_entrypoint_does_not_compact_without_required_tools_context():
    manager = BoundaryContextManager({"max_tokens": 100000, "summarize_trigger": 0.001})
    await manager.set_messages(
        [
            row
            for i in range(4)
            for row in [
                {"role": "user", "content": f"turn {i}"},
                {"role": "assistant", "content": "evidence " * 1000},
            ]
        ]
    )
    model = SimpleNamespace(
        native_compaction_requires_request_context=True,
        supports_native_compaction=lambda: True,
        validate_compacted_context=lambda row: True,
        compact_context=AsyncMock(),
        complete=AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Preserved evidence")],
                usage=None,
            )
        ),
    )
    await manager.get_messages_for_request(provider=model)
    model.compact_context.assert_not_awaited()
    model.complete.assert_awaited_once()
