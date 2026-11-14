"""The rows-only boundary owns its preliminary request, not the later dispatch."""
import asyncio
import copy
from types import SimpleNamespace

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import ContextLengthError

from amplifier_module_context_managed.boundary import BoundaryContextManager
from amplifier_module_context_managed.summary import request_fits


class Owner:
    """Exact-object release sentinel, not a substitute for Foundation integration."""

    def __init__(self, *, count=10, limit=10000):
        self.count = count
        self.limit = limit
        self.requests = []
        self.releases = []
        self.active = None
        self.failure = None

    async def request_budget(self, request, **kwargs):
        self.requests.append(request)
        self.active = request
        if self.failure:
            raise self.failure
        return {"estimated_input_tokens": self.count, "input_limit_tokens": self.limit}

    def release_media_request(self, request, *, request_options):
        self.releases.append((request, request_options))
        assert request is self.active
        self.active = None
        return True


async def context_with_text(hooks=None):
    context = BoundaryContextManager({"max_tokens": 10000}, hooks)
    await context.set_messages([{"role": "user", "content": "Keep the original."}])
    return context


@pytest.mark.asyncio
@pytest.mark.parametrize("retaining", [False, True])
async def test_public_rows_return_releases_exact_preliminary_request(retaining):
    owner = Owner()
    context = await context_with_text()
    original = await context.get_messages()
    if retaining:
        rows = await context.get_messages_for_request_retaining(provider=owner, retain_contents=[])
    else:
        rows = await context.get_messages_for_request(provider=owner)
    assert rows[0]["content"] == original[0]["content"]
    assert owner.releases == [(owner.requests[0], None)]
    assert owner.active is None
    rows[0]["content"] = "detached"
    assert await context.get_messages() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("count failed"), asyncio.CancelledError("count cancelled")])
async def test_count_failure_or_cancellation_keeps_original_and_releases(failure):
    owner = Owner()
    owner.failure = failure
    context = await context_with_text()
    with pytest.raises(type(failure)) as caught:
        await context.get_messages_for_request(provider=owner)
    assert caught.value is failure
    assert owner.releases == [(owner.requests[0], None)]
    assert owner.active is None


@pytest.mark.asyncio
async def test_budget_observer_sees_preliminary_request_already_released():
    owner = Owner(count=10001)
    observed = []

    async def emit(event, data):
        if event == "context:budget_exceeded":
            observed.append(event)
            assert owner.active is None
            raise asyncio.CancelledError("observer cancelled")

    context = await context_with_text(SimpleNamespace(emit=emit))
    with pytest.raises(asyncio.CancelledError, match="observer cancelled"):
        await context.get_messages_for_request(provider=owner)
    assert observed == ["context:budget_exceeded"]
    assert owner.releases == [(owner.requests[0], None)]


@pytest.mark.asyncio
async def test_oversize_releases_before_propagating_refusal():
    owner = Owner(count=10001)
    context = await context_with_text()
    with pytest.raises(ContextLengthError):
        await context.get_messages_for_request(provider=owner)
    assert owner.active is None
    assert len(owner.requests) == len(owner.releases) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["async", "false", "raises", "type_error", "lookup_raises"])
async def test_optional_cleanup_is_single_shot_and_diagnostics_are_static(mode, caplog):
    owner = Owner()
    calls = []

    def release(request, *, request_options):
        calls.append((request, request_options))
        if mode in {"raises", "type_error"}:
            error = TypeError if mode == "type_error" else RuntimeError
            raise error("PRIVATE_PIXEL_TRANSPORT")
        if mode == "false":
            return False

        async def finish():
            owner.active = None
            return True

        return finish()

    owner.release_media_request = release
    if mode == "lookup_raises":
        class LookupOwner(Owner):
            @property
            def release_media_request(self):
                calls.append("lookup")
                raise RuntimeError("PRIVATE_PIXEL_TRANSPORT")
        owner = LookupOwner()
    context = await context_with_text()
    result = await context.get_messages_for_request(provider=owner)
    assert result[0]["content"] == "Keep the original."
    assert len(calls) == 1
    assert len(owner.requests) == 1
    assert "PRIVATE_PIXEL_TRANSPORT" not in caplog.text
    assert "released successfully" not in caplog.text
    if mode in {"false", "raises", "type_error", "lookup_raises"}:
        assert "release was not confirmed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("original", [RuntimeError("original"), asyncio.CancelledError("original")])
@pytest.mark.parametrize("cleanup", [RuntimeError("PRIVATE_PIXEL_TRANSPORT"), asyncio.CancelledError("PRIVATE_PIXEL_TRANSPORT")])
async def test_cleanup_cannot_mask_original_error_or_cancel(original, cleanup, caplog):
    owner = Owner()
    owner.failure = original

    async def release(request, *, request_options):
        owner.releases.append((request, request_options))
        raise cleanup

    owner.release_media_request = release
    context = await context_with_text()
    with pytest.raises(type(original)) as caught:
        await context.get_messages_for_request(provider=owner)
    assert caught.value is original
    assert len(owner.releases) == 1
    assert "PRIVATE_PIXEL_TRANSPORT" not in caplog.text


@pytest.mark.asyncio
async def test_new_cancel_during_async_cleanup_prevents_rows_return():
    owner = Owner()

    async def release(request, *, request_options):
        owner.releases.append((request, request_options))
        raise asyncio.CancelledError("cleanup cancelled")

    owner.release_media_request = release
    context = await context_with_text()
    with pytest.raises(asyncio.CancelledError, match="cleanup cancelled"):
        await context.get_messages_for_request(provider=owner)
    assert len(owner.requests) == len(owner.releases) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ambient_exception", [False, True])
async def test_task_cancel_during_cleanup_is_not_swallowed_by_callers_except(ambient_exception):
    owner = Owner()
    entered = asyncio.Event()
    rows_returned = []

    async def release(request, *, request_options):
        owner.releases.append((request, request_options))
        entered.set()
        await asyncio.Event().wait()

    owner.release_media_request = release
    context = await context_with_text()

    async def prepare():
        try:
            if ambient_exception:
                raise ValueError("caller is handling a different exception")
        except ValueError:
            rows_returned.append(await context.get_messages_for_request(provider=owner))
        else:
            rows_returned.append(await context.get_messages_for_request(provider=owner))

    task = asyncio.create_task(prepare())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not rows_returned
    assert len(owner.requests) == len(owner.releases) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [None, SimpleNamespace(), SimpleNamespace(release_media_request=None)])
async def test_text_provider_without_optional_release_or_counter_keeps_compatibility(provider):
    context = await context_with_text()
    rows = await context.get_messages_for_request(provider=provider)
    assert rows[0]["content"] == "Keep the original."


@pytest.mark.asyncio
async def test_shared_request_fits_does_not_release_callers_request():
    owner = Owner()
    request = ChatRequest(messages=[Message(role="user", content="Owned by caller.")])
    assert (await request_fits(owner, request))[0]
    assert owner.active is request
    assert owner.releases == []


@pytest.mark.asyncio
async def test_measured_view_preserves_dispatch_and_options_for_callers_release():
    owner = Owner()
    context = await context_with_text()
    options = {"nested": {"selected": "original"}}

    async def count_view(view):
        dispatch = ChatRequest(messages=[Message(**row) for row in view])
        return {"dispatch": dispatch, "request_options": copy.deepcopy(options),
                "budget_decision": await owner.request_budget(dispatch)}

    result = await context.get_measured_request_view(provider=owner, retain_contents=[], count_view=count_view)
    attempt = result["final_attempt"]
    assert owner.active is attempt["dispatch"]
    assert owner.releases == []
    assert attempt["request_options"] == options
    assert owner.release_media_request(attempt["dispatch"], request_options=attempt["request_options"])