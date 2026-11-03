# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK translation and ownership exercised through the worker's public queue."""

import ctypes
import os
import socket
import threading
import time
from collections import deque

import msgspec
import pytest

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.kvcr_worker import _BlobKeyAdapter, _KvcrWorker
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverRejected,
    RegionLayout,
    RegionSpec,
    RowBatch,
    RowIO,
    RowStatus,
    Span,
)
from tensorrt_llm.kv_hints.protocol import KvHintAction, KvHintsEnvelope, snapshot_kv_hints

pytestmark = pytest.mark.cpu_only
_NATIVE_BUFFERS = []


class _RecordingRuntime:
    def __init__(self, config, bindings, backend_configs):
        self.owner = threading.get_ident()
        self.config, self.bindings, self.backend_configs = config, bindings, backend_configs
        self.calls, self.peer_calls = [], []
        self.completed, self.events = deque(), deque()
        self.automatic = True
        self.missing = False
        self.closed = self.fail_close = False

    def _check_owner(self):
        assert threading.get_ident() == self.owner

    def deposit(self, blocks):
        from kvcr.types import OpEntryResult, OpEntryStatus

        self._check_owner()
        self.calls.append(("deposit", blocks))
        handle = len(self.calls)
        if self.automatic:
            self.completed.append(
                (handle, {key: OpEntryResult(OpEntryStatus.SUCCESS) for key in blocks})
            )
        return handle

    def deliver(self, blocks, request_id=None):
        self.peer_calls.append(("deliver", request_id))
        handle = self.deposit(blocks)
        self.calls[-1] = ("deliver", blocks)
        return handle

    def query(self, keys, request_id=None):
        from kvcr.types import CacheTier, QueryStatus

        self._check_owner()
        self.peer_calls.append(("query", request_id))
        return [
            (QueryStatus.MISS, None) if self.missing else (QueryStatus.HIT, CacheTier.LOCAL_G2)
        ] * len(keys)

    def poll_completed(self):
        self._check_owner()
        while self.events:
            self.bindings.on_resilience_event(self.events.popleft())
        try:
            return (self.completed.popleft(),)
        except IndexError:
            return ()

    def submit_hint(self, hints, request_id):
        self._check_owner()
        self.peer_calls.append(("hint", request_id, hints))

    def discard_hint(self, request_id):
        self._check_owner()
        self.peer_calls.append(("discard", request_id))

    def close(self):
        self._check_owner()
        if self.fail_close:
            raise RuntimeError("transport still active")
        self.closed = True


def _hint(endpoint):
    envelope = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="sdk-test",
        actions=[
            KvHintAction(
                action_id="fetch",
                action_type="kv.fetch",
                action_version="1.0",
                payload={
                    "source_control_endpoint": endpoint,
                    "block_hashes": [_BlobKeyAdapter().decode(b"key/\xce\xb1")],
                },
            )
        ],
    )
    snapshot = snapshot_kv_hints(envelope)
    assert snapshot is not None
    return snapshot


def _until(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.001)
    pytest.fail("worker did not publish expected state")


def _region():
    return RegionSpec(1024, 128, RegionLayout(64, 2))


def _rows(address=1024, nbytes=64):
    return RowBatch.of([RowIO("key/α", (Span(address, nbytes),))])


@pytest.fixture
def runtimes(monkeypatch):
    sdk = pytest.importorskip("kvcr")
    created = []

    def create(*args):
        result = _RecordingRuntime(*args)
        created.append(result)
        return result

    monkeypatch.setattr(sdk, "KVCR", create)
    return created


@pytest.fixture
def worker(runtimes):
    result = _KvcrWorker(_region(), Span(4096, 256), endpoint="local", listen_port=19341)
    try:
        yield result, runtimes[0]
    finally:
        runtimes[0].fail_close = False
        assert result.close(3)


def test_sdk_calls_use_one_owner_and_translate_registered_slots(worker):
    queue, sdk = worker
    first, second = queue.put(_rows()), queue.get(_rows(1088))
    _until(lambda: queue.access_ended(first) and queue.access_ended(second))
    assert all(queue.poll(op)[0].status is RowStatus.READY for op in (first, second))
    assert sdk.owner != threading.get_ident()
    (registration,) = sdk.backend_configs.framework_regions
    assert (registration.addr, registration.size, registration.count) == (1024, 64, 2)
    assert sdk.backend_configs.local_dram.pools == [("", 4096, 256)]
    assert sdk.config.pool_layouts == [("", 64)]
    for (method, blocks), expected in zip(sdk.calls, [("deposit", 0), ("deliver", 1)]):
        assert method == expected[0]
        (ref,) = blocks["key/α".encode()]
        assert (ref.end_point_name, ref.label, ref.element_index) == ("local", "", expected[1])
    assert queue.close(3)
    assert queue.close(0)
    assert sdk.closed


@pytest.mark.parametrize("address,nbytes", [(1025, 64), (1024, 32), (1152, 64)])
def test_invalid_rows_never_reach_sdk(worker, address, nbytes):
    queue, sdk = worker
    try:
        op = queue.put(_rows(address, nbytes))
    except DriverRejected:
        pass
    else:
        _until(lambda: queue.poll(op)[0].status is RowStatus.FAULT)
    assert not sdk.calls


@pytest.mark.parametrize(
    "region,cache",
    [
        (
            RegionSpec(1024, 128, RegionLayout(64, 2, memory_kind=MemoryKind.DEVICE)),
            Span(4096, 256),
        ),
        (_region(), Span(1088, 256)),
        (_region(), Span(4096, 65)),
    ],
)
def test_bad_registration_never_opens_sdk(runtimes, region, cache):
    with pytest.raises((ValueError, RuntimeError)):
        _KvcrWorker(region, cache, endpoint="local", listen_port=19341)
    assert not runtimes


def test_resilience_events_are_processed_before_access_is_released(worker):
    from kvcr.types import MemoryRef, TransferError

    queue, sdk = worker
    fields = dict(
        destination_regions=[MemoryRef(end_point_name="local", element_index=0)],
        destination_owner="framework",
    )
    sdk.events.append(TransferError("copy", -1, **fields))
    changed = threading.Event()
    op = queue.get(_rows(), on_change=lambda _: changed.set())
    assert changed.wait(3)
    assert queue.poll(op)[0].status is RowStatus.READY
    assert not queue.access_ended(op)
    sdk.events.append(TransferError("copy", -1, state="quiesced", **fields))
    _until(lambda: queue.access_ended(op))


@pytest.mark.parametrize(
    "endpoint,label,index", [("other", "", 0), ("local", "bad", 0), ("local", "", 2)]
)
def test_invalid_event_reference_fences_access(worker, endpoint, label, index):
    from kvcr.types import MemoryRef, TransferError

    queue, sdk = worker
    sdk.automatic = False
    op = queue.get(_rows())
    _until(lambda: bool(sdk.calls))
    ref = MemoryRef(end_point_name=endpoint, label=label, element_index=index)
    sdk.events.append(
        TransferError("copy", -1, destination_regions=[ref], destination_owner="framework")
    )
    _until(lambda: queue.access_unprovable(op))
    assert not queue.access_ended(op)


def test_failed_sdk_close_can_be_retried(worker):
    queue, sdk = worker
    sdk.fail_close = True
    assert not queue.close(1)
    assert not sdk.closed
    sdk.fail_close = False
    assert queue.close(3)
    assert sdk.closed


def test_confirmed_miss_never_delivers_to_destination(worker):
    queue, sdk = worker
    sdk.missing = True
    op = queue.get(_rows())
    _until(lambda: queue.access_ended(op))
    assert queue.poll(op)[0].status is RowStatus.MISS
    assert not sdk.calls


def test_peer_hint_payload_is_forwarded_for_sdk_interpretation(worker):
    queue, sdk = worker
    envelope = _hint("provider-specific-source").envelope()
    envelope.actions[0].payload.update(
        {"block_hashes": [], "mode": "provider-mode", "provider_metadata": {"trace": [7, None]}}
    )
    hint = snapshot_kv_hints(envelope)
    assert hint is not None

    op = queue.get(_rows(), hint)
    _until(lambda: queue.poll(op)[0].status is not RowStatus.PENDING)

    calls = [call for call in sdk.peer_calls if call[0] == "hint"]
    assert len(calls) == 1
    assert isinstance(calls[0][2], msgspec.Struct)
    assert isinstance(calls[0][2].actions[0], msgspec.Struct)
    assert msgspec.to_builtins(calls[0][2]) == envelope.model_dump(mode="json")


def test_peer_hints_are_isolated_until_their_own_completion(worker):
    from kvcr.types import OpEntryResult, OpEntryStatus

    queue, sdk = worker
    sdk.automatic = False
    first = queue.get(_rows(), _hint("tcp://source-a:19001"))
    second = queue.get(_rows(1088), _hint("tcp://source-b:19002"))
    _until(lambda: len(sdk.calls) == 2)
    hints = [call for call in sdk.peer_calls if call[0] == "hint"]
    assert len(hints) == 2
    ids = [call[1] for call in hints]
    assert ids[0] != ids[1]
    for call, source in zip(hints, ("source-a", "source-b")):
        (action,) = call[2].actions
        assert source in action.payload["source_control_endpoint"]
        assert action.payload["block_hashes"] == [sdk.bindings.key_adapter.decode(b"key/\xce\xb1")]
        assert ("query", call[1]) in sdk.peer_calls
        assert ("deliver", call[1]) in sdk.peer_calls
        assert ("discard", call[1]) not in sdk.peer_calls
    sdk.completed.append((1, {b"key/\xce\xb1": OpEntryResult(OpEntryStatus.SUCCESS)}))
    _until(lambda: queue.access_ended(first))
    assert ("discard", ids[0]) in sdk.peer_calls
    assert ("discard", ids[1]) not in sdk.peer_calls
    sdk.completed.append((2, {b"key/\xce\xb1": OpEntryResult(OpEntryStatus.SUCCESS)}))
    _until(lambda: queue.access_ended(second))
    assert ("discard", ids[1]) in sdk.peer_calls


@pytest.mark.skipif(os.getenv("TRTLLM_TEST_KVCR_NATIVE") != "1", reason="requires native KVCR/UCX")
def test_native_worker_host_round_trip():
    slot_bytes = 4096
    host = ctypes.create_string_buffer(2 * slot_bytes)
    cache = ctypes.create_string_buffer(4 * slot_bytes)
    allocations = host, cache
    _NATIVE_BUFFERS.append(allocations)
    address = ctypes.addressof(host)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    queue = _KvcrWorker(
        RegionSpec(address, len(host), RegionLayout(slot_bytes, 2)),
        Span(ctypes.addressof(cache), len(cache)),
        endpoint="worker-roundtrip",
        listen_port=port,
    )
    payload = bytes(range(256)) * 16

    def wait(op):
        _until(lambda: queue.access_ended(op))
        assert queue.poll(op)[0].status is RowStatus.READY

    try:
        ctypes.memmove(address, payload, slot_bytes)
        wait(queue.put(RowBatch.of([RowIO("roundtrip", (Span(address, slot_bytes),))])))
        ctypes.memset(address, 0, slot_bytes)
        ctypes.memmove(address + slot_bytes, bytes(value ^ 0xFF for value in payload), slot_bytes)
        wait(
            queue.get(RowBatch.of([RowIO("roundtrip", (Span(address + slot_bytes, slot_bytes),))]))
        )
        assert host.raw[:slot_bytes] == bytes(slot_bytes)
        assert host.raw[slot_bytes:] == payload
    finally:
        assert queue.close(10)
        _NATIVE_BUFFERS.remove(allocations)
