"""Opt-in live synthetic qualification; requires the Anthropic native adapter.

Uses ANTHROPIC_API_KEY, never real history, and never executes tools. Prints only
scalar evidence. Run with both context-managed and provider-anthropic installed.
"""

import asyncio
import json
import logging
import os
import traceback

from amplifier_core import ChatRequest, Message
from amplifier_core.message_models import ToolSpec
from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_provider_anthropic import AnthropicProvider


async def main():
    logging.disable(logging.CRITICAL)
    model = os.environ.get("COMPACTION_TEST_MODEL", "claude-sonnet-5")
    provider = AnthropicProvider(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        config={
            "default_model": model,
            "extended_thinking": False,
            "shared_rate_limit_state_path": "",
        },
    )
    compact = provider.compact_context

    async def diagnose_compact(request):
        try:
            return await compact(request)
        except Exception as exc:
            body = getattr(exc, "body", None)
            error = body.get("error", {}) if isinstance(body, dict) else {}
            # This fixture contains no user data. Print the vendor's validation
            # message only, never SDK exception text or request/credential data.
            print(
                json.dumps(
                    {"native_error": error.get("type"), "message": error.get("message")}
                ),
                flush=True,
            )
            raise

    provider.compact_context = diagnose_compact
    events = []

    class Hooks:
        async def emit(self, name, data):
            if name == "context:compaction_finished":
                events.append(
                    {
                        key: data[key]
                        for key in (
                            "outcome",
                            "method",
                            "input_tokens_before",
                            "input_tokens_after",
                            "elapsed_ms",
                            "native_failure",
                            "failure",
                            "native_input_tokens_after",
                        )
                        if key in data
                    }
                )
                print(json.dumps({"event": events[-1]}), flush=True)

    config = {
        "max_tokens": 100000,
        "summarize_trigger": 0.001,
        "durable_checkpoints": True,
        "native_min_new_tokens": 0,
        "compaction_notice_enabled": False,
    }
    identity = {"provider": "anthropic", "model": model}
    tools = [
        ToolSpec(
            name="fixture_lookup",
            description="Unused fixture. Never call.",
            parameters={"type": "object", "properties": {}},
        )
    ]

    async def system():
        return "Retain exact user corrections. This is a synthetic memory test. Do not call tools."

    async def count_view(rows):
        req = ChatRequest(
            model=model,
            messages=[Message(**row) for row in rows],
            tools=tools,
            max_output_tokens=256,
            metadata={"stream": False},
        )
        return {
            "dispatch": req,
            "budget_decision": await provider.request_budget(req, context_estimate=0),
        }

    manager = BoundaryContextManager(config, Hooks())
    await manager.set_system_prompt_factory(system)
    manager.checkpoint_identity = lambda: identity
    originals = [
        {"role": "user", "content": "Project ORCHID-782. Use port 1234."},
        {
            "role": "assistant",
            "content": "Recorded. " + "Fixture evidence is verified. " * 250,
        },
        {"role": "user", "content": "Correction: use port 4317. Color is turquoise."},
        {
            "role": "assistant",
            "content": "Recorded. " + "More fixture evidence is verified. " * 250,
        },
    ]
    for i in range(3):
        originals += [
            {"role": "user", "content": f"Review fixture segment {i}."},
            {"role": "assistant", "content": "Verified completed segment. " * 250},
        ]
    await manager.set_messages(originals)
    checks = []
    try:
        for cycle in range(3):
            question = {
                "role": "user",
                "content": "Give only the project codename, corrected port, and color.",
            }
            originals.append(question)
            await manager.add_message(question)
            fitted = await manager.get_measured_request_view(
                provider=provider, retain_contents=[], count_view=count_view
            )
            assert (
                events[-1]["method"] == "native"
                and events[-1]["outcome"] == "completed"
            )
            answer = await provider.complete(fitted["final_attempt"]["dispatch"])
            text = " ".join(
                block.text
                for block in answer.content
                if getattr(block, "type", None) == "text"
            )
            check = {
                "cycle": cycle + 1,
                "codename": "ORCHID-782" in text,
                "port": "4317" in text,
                "color": "turquoise" in text.lower(),
                "no_tool_calls": not answer.tool_calls,
                "originals_intact": await manager.get_messages() == originals,
            }
            checks.append(check)
            assert all(value for key, value in check.items() if key != "cycle")
            reply = {"role": "assistant", "content": text}
            originals.append(reply)
            await manager.add_message(reply)
            for i in range(2):
                for row in [
                    {"role": "user", "content": f"Review added segment {cycle}/{i}."},
                    {
                        "role": "assistant",
                        "content": "Verified additional fixture material. " * 250,
                    },
                ]:
                    originals.append(row)
                    await manager.add_message(row)
            # Persist/reload the checkpoint, not just the in-memory Message.
            record = json.loads(json.dumps(manager.export_checkpoint(identity)))
            restored = BoundaryContextManager(config, Hooks())
            await restored.set_messages(originals)
            await restored.set_system_prompt_factory(system)
            restored.checkpoint_identity = lambda: identity
            assert restored.restore_checkpoint(record, identity)["status"] == "restored"
            manager = restored
        print(
            json.dumps(
                {
                    "passed": True,
                    "model": model,
                    "checks": checks,
                    "compactions": events,
                }
            )
        )
    finally:
        await provider.client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001 - prevent credential-bearing SDK tracebacks
        # SDK exception text may carry request data; report only safe categories.
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "status": getattr(exc, "status_code", None),
                    "frames": [
                        {"function": frame.name, "line": frame.lineno}
                        for frame in traceback.extract_tb(exc.__traceback__)
                    ],
                }
            )
        )
        raise SystemExit(1)
