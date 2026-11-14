"""Explicit actual Core/Foundation integration with structural in-memory admission.

Run separately from module tests with exact Foundation source on PYTHONPATH and
MEDIA_PROVIDER_SHA256 / MEDIA_PROJECTOR_SHA256 set to the reviewed file hashes.
The real capability/projector/wrapper run, but store authority and graph admission
are fixtures: this is not filesystem-store, host-bridge or vendor qualification.
"""
import asyncio
import copy
import hashlib
import io
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from amplifier_core import ChatRequest, ImageBlock, Message
from amplifier_module_context_simple import SimpleContextManager

from amplifier_module_context_managed.boundary import BoundaryContextManager


@pytest.fixture(scope="module")
def foundation():
    from amplifier_foundation.session import media, media_projection, media_provider

    for module, variable in ((media_provider, "MEDIA_PROVIDER_SHA256"),
                             (media_projection, "MEDIA_PROJECTOR_SHA256")):
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == os.environ[variable]
    return media, media_provider


def fixture_data():
    # Deterministic valid 2x2 purple PNG, constructed locally without a decoder.
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\x80\0\x80" * 2) * 2)) + chunk(b"IEND", b""))
    ref = {"v": 1, "session_id": "fixture", "sha256": hashlib.sha256(data).hexdigest(),
           "mime": "image/png", "bytes": len(data), "width": 2, "height": 2,
           "provenance": {"kind": "host", "session_id": "fixture", "id": "observation",
                          "created_at": "2026-01-01T00:00:00Z", "expires_at": None}}
    return data, ref


class MemoryAdmission:
    """Structural authority fixture, deliberately not SessionMediaStore."""

    session_id = "fixture"
    epoch = 1
    policy = {}

    def __init__(self):
        self.data, self.ref = fixture_data()
        self._prepared = {}
        self.registered = []
        self.unregistered = []

    def validate_ref(self, ref, *, committed=True, check_expiry=True):
        assert ref == self.ref and committed
        return copy.deepcopy(ref)

    def verify_checkpoint(self, ref):
        assert ref == self.ref
        return True

    def resolve_ref(self, ref, *, committed=True):
        self.validate_ref(ref, committed=committed)
        return io.BytesIO(self.data)

    def register_prepared(self, prepared):
        self._prepared[id(prepared.request)] = prepared
        self.registered.append(prepared)

    def unregister_prepared(self, prepared):
        self._prepared.pop(id(prepared.request), None)
        self.unregistered.append(prepared)

    def history(self):
        return [{"role": "user", "content": "Observe.",
                 "metadata": {"experimental.media.v1": {"refs": [copy.deepcopy(self.ref)]}}}]


class GraphAdmission:
    def __init__(self):
        self.active = 0
        self.generation = 1

    @contextmanager
    def scope(self):
        generation = self.generation
        self.active += 1
        try:
            def valid():
                assert generation == self.generation
                return True
            yield valid
        finally:
            self.active -= 1

    def transition(self):
        assert self.active == 0
        self.generation += 1


class Observer:
    name = "offline-observer"
    default_model = "fixture-model"
    config = {}

    def __init__(self, capability):
        self.capability = capability
        self.counted = []
        self.sent = []
        self.tokens = 300
        self.count_entered = None

    def get_media_request_support(self, *, model, request_options=None):
        return {"v": 1, "positions": ["user"], "mime_types": ["image/png"]}

    async def request_budget(self, request, *, context_estimate, request_options=None):
        self.capability.validate(request, purpose="count", request_options=request_options or {})
        assert any(isinstance(block, ImageBlock) for block in request.messages[0].content)
        self.counted.append(request)
        if self.count_entered:
            self.count_entered.set()
            await asyncio.Event().wait()
        return {"estimated_input_tokens": self.tokens, "input_limit_tokens": 1000,
                "context_token_budget": 1000,
                "measurement": {"kind": "provider_count", "source": "offline.fixture",
                                "input_tokens": self.tokens}}

    async def complete(self, request, *, request_options=None):
        self.capability.validate(request, purpose="transport", request_options=request_options or {})
        self.sent.append(request)
        return "offline complete"


def pairing(foundation):
    media, provider = foundation
    store, graph = MemoryAdmission(), GraphAdmission()
    capability = media.SessionMediaCapability(store, enabled=True)
    observer = Observer(capability)
    wrapper = provider.MediaPreparedProvider(
        observer, capability, provider_identity={"provider": "offline-observer", "account": "fixture", "generation": 1},
        request_scope=graph.scope)
    return store, graph, observer, wrapper


async def managed(store, hooks=None):
    context = BoundaryContextManager(
        {"max_tokens": 1000, "token_meter": "actual", "compaction_notice_enabled": False}, hooks=hooks)
    await context.set_messages(store.history())
    return context


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["rows", "cancel_after_rows", "final_constructor_failure",
                                "budget_observer_cancel", "inside_count_cancel"])
async def test_actual_preliminary_request_released_at_public_rows_boundary(foundation, edge):
    store, graph, observer, wrapper = pairing(foundation)
    entered = asyncio.Event()

    async def emit(event, data):
        if event == "context:budget_exceeded":
            entered.set()
            await asyncio.Event().wait()

    context = await managed(store, SimpleNamespace(emit=emit))
    original = await context.get_messages()
    if edge == "budget_observer_cancel":
        observer.tokens = 1200
    if edge == "inside_count_cancel":
        observer.count_entered = entered

    async def prepare():
        rows = await context.get_messages_for_request(provider=wrapper)
        if edge == "cancel_after_rows":
            entered.set()
            await asyncio.Event().wait()
        if edge == "final_constructor_failure":
            # Actual Core validation fails before a later dispatch can exist.
            ChatRequest(messages=[Message(**row) for row in rows], tools=[object()])
        return rows

    try:
        if edge in {"cancel_after_rows", "budget_observer_cancel", "inside_count_cancel"}:
            task = asyncio.create_task(prepare())
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                # Capture before cancellation too, especially the post-count observer.
                active_before_cancel = (len(store._prepared), graph.active)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            if edge != "inside_count_cancel":
                assert active_before_cancel == (0, 0)
        elif edge == "final_constructor_failure":
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                await prepare()
        else:
            rows = await prepare()
            assert rows == original
        assert len(observer.counted) == 1
        assert not observer.sent
        assert len(store.registered) == len(store.unregistered) == 1
        assert not store._prepared and graph.active == 0
        assert await context.get_messages() == original
        graph.transition()
        # No provider close/recreation, refit or retry is needed for the next call.
        observer.tokens, observer.count_entered = 300, None
        await context.get_messages_for_request(provider=wrapper)
        assert len(observer.counted) == 2 and not observer.sent
        assert len(store.registered) == len(store.unregistered) == 2
        assert not store._prepared and graph.active == 0
    finally:
        wrapper.close()  # Fixture teardown only, after all acceptance assertions.


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["send", "abandon"])
async def test_actual_measured_handoff_retains_exact_dispatch_until_caller_finishes(foundation, finish):
    store, graph, observer, wrapper = pairing(foundation)
    context = await managed(store)
    options = {"nested": {"effective": "original"}}

    async def count_view(rows):
        dispatch = ChatRequest(messages=[Message(**row) for row in rows])
        decision = await wrapper.request_budget(dispatch, request_options=options, context_estimate=1)
        return {"dispatch": dispatch, "request_options": copy.deepcopy(options), "budget_decision": decision}

    try:
        result = await context.get_measured_request_view(provider=wrapper, retain_contents=[], count_view=count_view)
        attempt = result["final_attempt"]
        assert len(store._prepared) == graph.active == 1
        assert not store.unregistered
        assert wrapper.release_media_request(attempt["dispatch"].model_copy(deep=True),
                                             request_options=options) is False
        if finish == "send":
            assert await wrapper.complete(attempt["dispatch"], request_options=attempt["request_options"]) == "offline complete"
            assert observer.counted[0] is observer.sent[0]
        else:
            assert wrapper.release_media_request(attempt["dispatch"], request_options=attempt["request_options"]) is True
            assert not observer.sent
        assert len(observer.counted) == 1
        assert not store._prepared and graph.active == 0
        assert len(store.unregistered) == 1
        assert await context.get_messages() == store.history()
    finally:
        wrapper.close()


def summary_history(store):
    return [
        {"role": "user", "content": "Original objective."},
        {"role": "assistant", "content": "Historical reference. " * 800},
        {"role": "user", "content": "Second turn."},
        {"role": "assistant", "content": "Reply."},
        *store.history(),
    ]


class SummaryObserver(Observer):
    """Observe real summary requests, without manually counting a candidate."""

    def __init__(self, capability, mode):
        super().__init__(capability)
        self.mode = mode
        self.ready = asyncio.Event()
        self.wait_info = False
        self.count_options = []
        self.send_options = []
        self.send_settled = False

    @staticmethod
    def is_summary(request):
        return (request.metadata or {}).get("purpose") == "context-compaction"

    def get_info(self):
        if self.wait_info:
            async def pause():
                self.ready.set()
                await asyncio.Event().wait()
            return pause()
        return {}

    async def request_budget(self, request, *, context_estimate, request_options=None):
        self.capability.validate(request, purpose="count", request_options=request_options or {})
        self.counted.append(request)
        self.count_options.append(request_options)
        summary = self.is_summary(request)
        if summary and self.mode == "cancel_info":
            self.wait_info = True
        if summary and self.mode == "cancel_count":
            self.ready.set()
            await asyncio.Event().wait()
        if summary and self.mode == "count_error":
            raise RuntimeError("fixture count failure")
        n = 1200 if summary and (self.mode == "refuse" or
                                 self.mode == "split" and len(self.counted) == 1) else 300
        return {"estimated_input_tokens": n, "input_limit_tokens": 1000,
                "context_token_budget": 1000,
                "measurement": {"kind": "provider_count", "source": "offline.fixture", "input_tokens": n}}

    async def complete(self, request, *, request_options=None):
        self.capability.validate(request, purpose="transport", request_options=request_options or {})
        self.sent.append(request)
        self.send_options.append(request_options)
        try:
            if self.mode == "cancel_send":
                self.ready.set()
                await asyncio.Event().wait()
            if self.mode == "send_error":
                raise RuntimeError("fixture send failure")
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Verified continuation.")])
        finally:
            self.send_settled = True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancel_info", "refuse", "success", "split", "cancel_count",
                                 "cancel_send", "count_error", "send_error"])
async def test_actual_summary_owned_request_lifetime(foundation, mode):
    media, provider = foundation
    store, graph = MemoryAdmission(), GraphAdmission()
    capability = media.SessionMediaCapability(store, enabled=True)
    observer = SummaryObserver(capability, mode)
    wrapper = provider.MediaPreparedProvider(
        observer, capability, provider_identity={"provider": observer.name, "account": "fixture", "generation": 1},
        request_scope=graph.scope)
    context = await managed(store)
    await context.set_messages(summary_history(store))
    original = await context.get_messages()
    try:
        if mode.startswith("cancel_"):
            task = asyncio.create_task(context.get_messages_for_request(provider=wrapper))
            try:
                await asyncio.wait_for(observer.ready.wait(), timeout=5)
                assert len(observer.counted) == 1
                assert len(store._prepared) == graph.active == 1
                assert len(observer.sent) == (1 if mode == "cancel_send" else 0)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        elif mode in {"refuse", "count_error", "send_error"}:
            from amplifier_module_context_managed.errors import CompactionError
            with pytest.raises(CompactionError):
                await context.get_messages_for_request(provider=wrapper)
        else:
            await context.get_messages_for_request(provider=wrapper)
            summary_counts = [q for q in observer.counted if observer.is_summary(q)]
            assert len(summary_counts) == (3 if mode == "split" else 1)
            assert len(observer.sent) == (2 if mode == "split" else 1)
            assert all(any(q is sent for q in summary_counts) for sent in observer.sent)
            assert context.summary is not None
        if mode == "refuse":
            assert len(observer.counted) == 7 and not observer.sent
        if mode in {"cancel_send", "send_error"}:
            assert observer.send_settled
        # Observe before fixture teardown. No close or later request may fix a leak.
        assert not store._prepared and graph.active == 0
        assert len(store.registered) == len(store.unregistered)
        assert all(options is None for options in observer.count_options)
        assert await context.get_messages() == original
        graph.transition()
        # Same provider remains usable; a fresh context avoids the deliberate
        # compaction-failure retry latch, without changing canonical history.
        observer.mode, observer.wait_info = "success", False
        next_context = await managed(store)
        await next_context.get_messages_for_request(provider=wrapper)
        assert not store._prepared and graph.active == 0
        assert len(store.registered) == len(store.unregistered)
    finally:
        wrapper.close()


@pytest.mark.asyncio
async def test_actual_simple_preliminary_rows_do_not_count_or_acquire_media(foundation):
    store, graph, observer, wrapper = pairing(foundation)
    context = SimpleContextManager(max_tokens=1000)
    await context.set_messages(store.history())
    try:
        await context.get_messages_for_request(provider=wrapper)
        assert not observer.counted and not observer.sent
        assert not store.registered and not store._prepared and graph.active == 0
    finally:
        wrapper.close()