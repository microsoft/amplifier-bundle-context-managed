"""A paid settled prefix and an affordable new reply have separate lifetimes."""
import asyncio
import copy
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message, ToolSpec

from .test_settled_usage import offer, setup
from .test_summary_work_budget import IDENTITY, response


class PendingConversation:
    async def start(self):
        self.ctx, self.provider, self.rows = await setup()
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.open = False
        self.saved = []
        self.ctx.persist_checkpoint = lambda: self.saved.append(copy.deepcopy(self.ctx.export_checkpoint(IDENTITY)))
        self.ctx.preserve_evidence = AsyncMock(side_effect=lambda rows: [{"rows": len(rows)}])

        async def summary(_request):
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                return response("Cancelled result")
            return response("ORBIT grounded note")

        self.provider.complete = AsyncMock(side_effect=summary)
        self.envelope = None

        @asynccontextmanager
        async def prepare(request, *, request_options):
            self.open = True
            try:
                yield self.envelope(request) if self.envelope else request
            finally:
                self.open = False
        self.provider.prepared_foreground = prepare
        assert offer(self.ctx, self.provider)()
        await asyncio.wait_for(self.entered.wait(), 2)
        self.task = self.ctx._maintenance_task
        return self

    def begin(self):
        return self.ctx.begin_settled_foreground(provider=self.provider)

    def project(self, rows):
        self.logical = ChatRequest(messages=[Message(**row) for row in rows])
        return self.logical


@pytest.mark.asyncio
async def test_complete_followup_precedes_paid_prefix_then_newest_evidence_and_tail_survive_reopen():
    c = await PendingConversation().start()
    ticket = c.begin()
    assert callable(ticket)
    await c.ctx.add_message({"role": "user", "content": "Keep my new question"})
    async with ticket(retain_contents=[], project_view=c.project, request_options={}) as accepted:
        assert accepted["dispatch"] is c.logical and accepted["base_view"][-1]["content"] == "Keep my new question"
        assert c.open and not c.task.done() and not c.saved
        accepted["validate"]()
        await c.ctx.add_message({"role": "assistant", "content": "ORBIT. Here is the fresh reply."})
        assert not c.task.done()
    assert not c.open
    c.release.set()
    await c.task
    c.provider.complete.assert_awaited_once()
    assert len(c.saved) == 1 and c.saved[0]["summary"]["text"] == "ORBIT grounded note"
    assert c.saved[0]["evidenceRefs"] == [{"rows": len(c.rows) + 1}]
    after = await c.ctx.get_messages()
    assert after[:-2] == c.rows and after[-1]["content"].startswith("ORBIT.")
    fresh, _, _ = await setup()
    await fresh.set_messages(after)
    assert fresh.restore_checkpoint(c.saved[0], IDENTITY)["status"] == "restored"
    assert fresh.summary[0] < len(c.rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["expanded_tools", "unpriced", "image", "reserve"])
async def test_unknown_or_large_final_projection_releases_host_then_joins(reason):
    c = await PendingConversation().start()
    ticket = c.begin()
    await c.ctx.add_message({"role": "user", "content": "Follow up"})
    if reason == "expanded_tools":
        c.envelope = lambda r: r.model_copy(update={"tools": [ToolSpec(name="large", description="z" * 85000,
                                                                       parameters={})]})
    elif reason == "image":
        c.envelope = lambda r: r.model_copy(update={"messages": [Message(role="user",
            content=[{"type": "image", "source": {"type":"base64","data":"YQ==","media_type":"image/png"}}])]})
    elif reason == "reserve":
        c.provider.get_info = lambda: SimpleNamespace(defaults={"context_window": 28000, "max_output_tokens": 17000})
    else:
        c.envelope = lambda r: None

    async def attempt():
        async with ticket(retain_contents=[], project_view=c.project, request_options={}) as result:
            return result
    trying = asyncio.create_task(attempt())
    await asyncio.sleep(0)
    assert not trying.done() and not c.open
    c.release.set()
    assert await trying is None
    assert len(c.saved) == 1
    c.provider.complete.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["stop", "replace", "alter_prefix", "writer", "operations", "model"])
async def test_late_paid_note_refused_on_source_or_authority_change(change):
    c = await PendingConversation().start()
    assert c.begin()
    await c.ctx.add_message({"role":"user", "content":"New user input before save"})
    if change == "stop":
        c.ctx._is_cancelled = lambda: True
        c.ctx.cancel_settled_compaction()
    elif change == "replace":
        await c.ctx.set_messages(await c.ctx.get_messages())
    elif change == "alter_prefix":
        # Emulate a caller breaking its read-only borrowed row contract.
        c.ctx.messages[0]["content"] = "Different source"
    elif change == "writer":
        c.ctx.persist_checkpoint = lambda: c.saved.append("new writer")
    elif change == "operations":
        c.ctx.active_operations = lambda: [{"job_id":"other"}]
    else:
        c.ctx.checkpoint_identity = lambda: {**IDENTITY,"model":"other"}
    c.release.set()
    await asyncio.gather(c.task, return_exceptions=True)
    assert not c.saved and c.ctx.summary is None
    c.provider.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_other_provider_or_absent_host_projection_uses_existing_join():
    c = await PendingConversation().start()
    assert c.ctx.begin_settled_foreground(provider=copy.copy(c.provider)) is None
    del c.provider.prepared_foreground
    assert c.begin() is None
    waiter = asyncio.create_task(c.ctx.wait_for_settled_checkpoint())
    await asyncio.sleep(0)
    assert not waiter.done()
    c.release.set()
    await waiter
    assert len(c.saved) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["compatible_save", "new_input", "operation", "account", "replaced_writer"])
async def test_predispatch_guard_allows_its_own_save_and_rejects_changed_request(change):
    c = await PendingConversation().start()
    ticket = c.begin()
    await c.ctx.add_message({"role": "user", "content": "Follow up"})
    async with ticket(retain_contents=[], project_view=c.project, request_options={}) as grant:
        if change == "compatible_save":
            c.release.set()
            await c.task
            assert c.saved
            grant["validate"]()
        else:
            if change == "new_input":
                await c.ctx.add_message({"role": "user", "content": "Replacement"})
            elif change == "operation":
                c.ctx.active_operations = lambda: [{"job_id":"other"}]
            elif change == "account":
                c.ctx.checkpoint_identity = lambda: {**IDENTITY, "provider":"changed"}
            else:
                c.ctx.persist_checkpoint = lambda: None
            with pytest.raises(RuntimeError, match="changed before dispatch"):
                grant["validate"]()
    c.release.set()
    await asyncio.gather(c.task, return_exceptions=True)
