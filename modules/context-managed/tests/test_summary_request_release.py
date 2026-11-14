"""Summary-only sentinels; actual Core/Foundation pairing is tested separately."""
import asyncio
from types import SimpleNamespace

import pytest
from amplifier_core.llm_errors import ContextLengthError
from amplifier_module_context_managed.boundary import BoundaryContextManager


class SummaryOwner:
    def __init__(self, edge="success"):
        self.edge = edge
        self.counts, self.sends, self.releases = [], [], []
        self.active = None
        self.failure = RuntimeError("original failure")
        self.entered, self.allow_settle = asyncio.Event(), asyncio.Event()
        self.send_active = False

    async def request_budget(self, request, **kwargs):
        assert self.active is None  # Not just last-fragment cleanup.
        assert "request_options" not in kwargs  # Original count options are omitted.
        self.active = request
        self.counts.append(request)
        if self.edge == "count_error":
            raise self.failure
        if self.edge == "cancel_count":
            self.entered.set()
            await asyncio.Event().wait()
        n = 1200 if self.edge == "refuse" else 10
        return {"estimated_input_tokens": n, "input_limit_tokens": 1000,
                "measurement": {"kind": "provider_count", "source": "fixture", "input_tokens": n}}

    def get_info(self):
        if self.active is not None and self.edge == "info_error":
            raise self.failure
        if self.active is not None and self.edge == "cancel_info":
            async def pause():
                self.entered.set()
                await asyncio.Event().wait()
            return pause()
        return {"capabilities": ["completion:auto_continue:v1"]}

    async def complete(self, request, **kwargs):
        assert request is self.active
        assert kwargs == {"request_options": {"auto_continue": False}}
        self.sends.append(request)
        self.send_active = True
        try:
            if self.edge == "send_error":
                raise self.failure
            if self.edge == "split_send" and len(self.sends) == 1:
                raise ContextLengthError("provider overflow")
            if self.edge == "cancel_send":
                self.entered.set()
                await asyncio.Event().wait()
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Continuation.")])
        finally:
            if self.edge == "cancel_send":
                await self.allow_settle.wait()
            self.send_active = False

    def release_media_request(self, request, *, request_options):
        assert not self.send_active
        assert request is self.active
        assert request_options is None  # Count options, never completion options.
        self.releases.append(request)
        self.active = None
        return True


async def summarize(owner, context=None):
    context = context or BoundaryContextManager({"max_tokens": 1000})
    return await context._summarize(owner, [{"role": "user", "content": "x" * 1500}],
        context.revision, {"calls": 0}, fingerprint="fixture",
        source_revision={"messages": 1}, identity={})


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["success", "refuse", "count_error", "info_error", "send_error", "split_send"])
async def test_summary_releases_every_original_candidate_on_each_exit(edge):
    owner = SummaryOwner(edge)
    if edge == "refuse":
        with pytest.raises(ContextLengthError):
            await summarize(owner)
        assert len(owner.counts) >= 3 and not owner.sends
    elif edge.endswith("_error"):
        with pytest.raises(RuntimeError) as caught:
            await summarize(owner)
        assert caught.value is owner.failure
    else:
        assert await summarize(owner) == "Continuation."
        assert len(owner.sends) == (3 if edge == "split_send" else 1)
    assert len(owner.counts) == len(owner.releases)
    assert all(q is released for q, released in zip(owner.counts, owner.releases))
    assert owner.active is None


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["cancel_count", "cancel_info", "cancel_send"])
async def test_summary_cancel_releases_only_after_owned_send_settles(edge):
    owner = SummaryOwner(edge)
    task = asyncio.create_task(summarize(owner))
    try:
        await asyncio.wait_for(owner.entered.wait(), timeout=5)
        task.cancel()
        if edge == "cancel_send":
            await asyncio.sleep(0)
            assert owner.send_active and not owner.releases and not task.done()
            owner.allow_settle.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        owner.allow_settle.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert owner.active is None and owner.releases == owner.counts


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["async", "false", "raises", "type_error", "lookup_raises"])
@pytest.mark.parametrize("edge", ["success", "count_error", "cancel_info"])
async def test_summary_optional_cleanup_is_single_shot_static_and_preserves_original(mode, edge, caplog):
    owner = SummaryOwner(edge)
    calls = []

    def release(request, *, request_options):
        calls.append((request, request_options))
        assert request is owner.active and request_options is None
        if mode in {"raises", "type_error"}:
            raise (TypeError if mode == "type_error" else RuntimeError)("PRIVATE_TRANSPORT")
        if mode == "false":
            return False
        async def finish():
            owner.active = None
            return True
        return finish()

    owner.release_media_request = release
    if mode == "lookup_raises":
        class LookupOwner(SummaryOwner):
            @property
            def release_media_request(self):
                calls.append("lookup")
                raise RuntimeError("PRIVATE_TRANSPORT")
        owner = LookupOwner(edge)
    if edge == "count_error":
        with pytest.raises(RuntimeError) as caught:
            await summarize(owner)
        assert caught.value is owner.failure
    elif edge == "cancel_info":
        task = asyncio.create_task(summarize(owner))
        try:
            await asyncio.wait_for(owner.entered.wait(), timeout=5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    else:
        assert await summarize(owner) == "Continuation."
    assert len(calls) == 1
    assert "PRIVATE_TRANSPORT" not in caplog.text
    if mode != "async":
        assert "release was not confirmed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["success", "count_error", "cancel_info"])
async def test_summary_cleanup_cancel_does_not_mask_original(edge, caplog):
    owner = SummaryOwner(edge)

    async def release(request, *, request_options):
        owner.releases.append(request)
        raise asyncio.CancelledError("PRIVATE_TRANSPORT")

    owner.release_media_request = release
    if edge == "count_error":
        with pytest.raises(RuntimeError) as caught:
            await summarize(owner)
        assert caught.value is owner.failure
    elif edge == "cancel_info":
        task = asyncio.create_task(summarize(owner))
        try:
            await asyncio.wait_for(owner.entered.wait(), timeout=5)
        finally:
            task.cancel("original cancellation")
            with pytest.raises(asyncio.CancelledError, match="original cancellation"):
                await task
    else:
        with pytest.raises(asyncio.CancelledError):
            await summarize(owner)
    assert len(owner.releases) == 1
    assert "PRIVATE_TRANSPORT" not in caplog.text


@pytest.mark.asyncio
async def test_summary_ordinary_provider_without_release_or_count_is_compatible():
    async def complete(request):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Continuation.")])
    # This also retains the pre-existing conservative fragmentation path.
    assert await summarize(SimpleNamespace(complete=complete)) == "Continuation."


@pytest.mark.asyncio
async def test_post_count_source_refusal_releases_without_send():
    context = BoundaryContextManager({"max_tokens": 1000})

    class ChangedOwner(SummaryOwner):
        async def request_budget(self, request, **kwargs):
            result = await super().request_budget(request, **kwargs)
            context.revision += 1
            return result

    owner = ChangedOwner()
    with pytest.raises(RuntimeError, match="History changed"):
        await summarize(owner, context)
    assert owner.releases == owner.counts and owner.active is None
    assert not owner.sends


@pytest.mark.asyncio
@pytest.mark.parametrize("ambient_exception", [False, True])
async def test_new_task_cancel_during_async_release_is_not_swallowed(ambient_exception):
    owner = SummaryOwner()
    entered = asyncio.Event()
    results = []

    async def release(request, *, request_options):
        owner.releases.append(request)
        entered.set()
        await asyncio.Event().wait()

    owner.release_media_request = release

    async def run():
        try:
            if ambient_exception:
                raise ValueError("caller handling unrelated exception")
        except ValueError:
            results.append(await summarize(owner))
        else:
            results.append(await summarize(owner))

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not results
    assert len(owner.releases) == len(owner.counts) == 1