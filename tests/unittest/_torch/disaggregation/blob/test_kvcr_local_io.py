# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local KVCR request translation and an opt-in native host-memory round trip."""

import ctypes
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.kvcr_local_io import _KvcrLocalIO
from tensorrt_llm._torch.disaggregation.blob.store import (
    RegionLayout,
    RegionSpec,
    RowBatch,
    RowIO,
    RowStatus,
    Span,
)

pytestmark = pytest.mark.cpu_only

# Native startup/teardown failure may leave transport accesses alive until exit.
_NATIVE_BUFFERS = []


class _RecordingRuntime:
    def __init__(self, config, bindings, backend_configs):
        self.config = config
        self.bindings = bindings
        self.backend_configs = backend_configs
        self.calls = []
        self.closed = False
        self.peer_calls = []
        self.completed = ()

    def deposit(self, blocks):
        self.calls.append(("deposit", blocks))
        return len(self.calls)

    def deliver(self, blocks, request_id=None):
        self.peer_calls.append(("deliver", request_id))
        self.calls.append(("deliver", blocks))
        return len(self.calls)

    def query(self, keys, request_id=None):
        from kvcr.types import CacheTier, QueryStatus

        self.peer_calls.append(("query", request_id))
        return [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * len(keys)

    def poll_completed(self):
        completed, self.completed = self.completed, ()
        return completed

    def submit_hint(self, envelope, request_id):
        self.peer_calls.append(("hint", request_id, envelope))

    def discard_hint(self, request_id):
        self.peer_calls.append(("discard", request_id))

    def close(self):
        self.closed = True


@pytest.fixture
def runtime(monkeypatch):
    sdk = pytest.importorskip("kvcr")
    runtimes = []

    def create(*args):
        result = _RecordingRuntime(*args)
        runtimes.append(result)
        return result

    monkeypatch.setattr(sdk, "KVCR", create)
    return runtimes


def _region():
    return RegionSpec(1024, 128, RegionLayout(64, 2))


def _rows(address=1024, nbytes=64):
    return RowBatch.of([RowIO("key/α", (Span(address, nbytes),))])


def test_requests_select_whole_registered_slots(runtime):
    io = _KvcrLocalIO(_region(), Span(4096, 256), endpoint="local", listen_port=19341)
    assert io.put(_rows()) == 1
    assert io.get(_rows(1088)) == 2
    sdk = runtime[0]
    registration = sdk.backend_configs.framework_regions[0]
    assert (registration.addr, registration.size, registration.count) == (1024, 64, 2)
    assert sdk.backend_configs.local_dram.pools == [("", 4096, 256)]
    assert sdk.config.pool_layouts == [("", 64)]
    for (method, blocks), expected in zip(sdk.calls, [("deposit", 0), ("deliver", 1)]):
        assert method == expected[0]
        (ref,) = blocks["key/α".encode()]
        assert (ref.end_point_name, ref.label, ref.element_index) == ("local", "", expected[1])
    assert io.progress() == ()
    io.close()
    assert sdk.closed


@pytest.mark.parametrize("address,nbytes", [(1025, 64), (1024, 32), (1152, 64)])
def test_bad_row_rejected_before_submitting_anything(runtime, address, nbytes):
    io = _KvcrLocalIO(_region(), Span(4096, 256), endpoint="local", listen_port=19341)
    with pytest.raises(ValueError, match="whole registered slot"):
        io.put(_rows(address, nbytes))
    assert not runtime[0].calls
    io.close()


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
def test_bad_registration_rejected_before_opening_kvcr(runtime, region, cache):
    with pytest.raises(ValueError):
        _KvcrLocalIO(region, cache, endpoint="local", listen_port=19341)
    assert not runtime


def test_only_owner_thread_calls_kvcr(runtime):
    io = _KvcrLocalIO(_region(), Span(4096, 256), endpoint="local", listen_port=19341)
    with ThreadPoolExecutor(1) as executor:
        with pytest.raises(RuntimeError, match="owner thread"):
            executor.submit(io.put, _rows()).result()
    assert not runtime[0].calls
    io.close()
    io.close()
    with pytest.raises(RuntimeError, match="closed"):
        io.get(_rows())


def test_existing_resilience_callback_resolves_framework_slots(runtime):
    from kvcr.types import MemoryRef, TransferError

    io = _KvcrLocalIO(_region(), Span(4096, 256), endpoint="local", listen_port=19341)
    ref = MemoryRef(end_point_name="local", element_index=1)
    fields = dict(destination_regions=[ref], destination_owner="framework")
    callback = runtime[0].bindings.on_resilience_event
    callback(TransferError("copy", -1, **fields))
    assert io.blocks_reuse((Span(1088, 64),))
    assert not io.blocks_reuse((Span(1024, 64),))
    callback(TransferError("copy", -1, state="quiesced", **fields))
    assert not io.blocks_reuse((Span(1088, 64),))
    io.close()


def test_entire_batch_validated_before_submission(runtime):
    io = _KvcrLocalIO(_region(), Span(4096, 256), endpoint="local", listen_port=19341)
    rows = RowBatch.of([RowIO("valid", (Span(1024, 64),)), RowIO("invalid", (Span(1152, 64),))])
    with pytest.raises(ValueError, match="whole registered slot"):
        io.put(rows)
    assert not runtime[0].calls
    io.close()


@pytest.mark.parametrize(
    "endpoint,label,index", [("other", "", 0), ("local", "bad", 0), ("local", "", 2)]
)
def test_event_references_must_match_registration(runtime, endpoint, label, index):
    from kvcr.types import MemoryRef, TransferError

    io = _KvcrLocalIO(_region(), Span(4096, 256), endpoint="local", listen_port=19341)
    ref = MemoryRef(end_point_name=endpoint, label=label, element_index=index)
    with pytest.raises(ValueError, match="registered framework slot"):
        runtime[0].bindings.on_resilience_event(
            TransferError("copy", -1, destination_regions=[ref], destination_owner="framework")
        )
    io.close()


def test_failed_close_can_be_retried(runtime, monkeypatch):
    io = _KvcrLocalIO(_region(), Span(4096, 256), endpoint="local", listen_port=19341)

    def failed_close():
        raise RuntimeError("transport still active")

    with monkeypatch.context() as patch:
        patch.setattr(runtime[0], "close", failed_close)
        with pytest.raises(RuntimeError, match="transport still active"):
            io.close()
    io.close()
    assert runtime[0].closed


def test_confirmed_absence_never_delivers_into_the_caller_buffer(runtime):
    from kvcr.types import QueryStatus

    io = _KvcrLocalIO(_region(), Span(4096, 256), endpoint="local", listen_port=19341)
    runtime[0].query = lambda keys: [(QueryStatus.MISS, None)] * len(keys)
    op = io.get(_rows())
    ((handle, results),) = io.progress()
    assert handle == op
    assert results[b"key/\xce\xb1"].status is RowStatus.MISS
    assert not runtime[0].calls
    io.close()


def test_peer_hints_are_scoped_to_each_delivery_and_discarded_after_completion(runtime):
    from kvcr.types import OpEntryResult, OpEntryStatus

    from tensorrt_llm._torch.disaggregation.blob.kvcr_peer import KvcrPeerHint

    io = _KvcrLocalIO(_region(), Span(4096, 256), endpoint="local", listen_port=19341)
    sdk = runtime[0]
    first = io.get(_rows(), KvcrPeerHint("tcp://source-a:19001"))
    second = io.get(_rows(1088), KvcrPeerHint("tcp://source-b:19002"))
    hints = [call for call in sdk.peer_calls if call[0] == "hint"]
    ids = [call[1] for call in hints]
    assert ids[0] != ids[1]
    for call, source in zip(hints, ("source-a", "source-b")):
        envelope = call[2]
        assert envelope.protocol_version == "0.1"
        (action,) = envelope.actions
        assert action.action_type == "kv.fetch" and action.action_version == "1.0"
        assert source in action.payload["source_control_endpoint"]
        assert action.payload["block_hashes"] == [sdk.bindings.key_adapter.decode(b"key/\xce\xb1")]
    for request_id in ids:
        assert ("query", request_id) in sdk.peer_calls
        assert ("deliver", request_id) in sdk.peer_calls
        assert ("discard", request_id) not in sdk.peer_calls
    sdk.completed = ((first, {b"key/\xce\xb1": OpEntryResult(OpEntryStatus.SUCCESS)}),)
    io.progress()
    assert ("discard", ids[0]) in sdk.peer_calls
    assert ("discard", ids[1]) not in sdk.peer_calls
    sdk.completed = ((second, {b"key/\xce\xb1": OpEntryResult(OpEntryStatus.SUCCESS)}),)
    io.progress()
    assert ("discard", ids[1]) in sdk.peer_calls
    io.close()


@pytest.mark.skipif(os.getenv("TRTLLM_TEST_KVCR_NATIVE") != "1", reason="requires native KVCR/UCX")
@pytest.mark.parametrize("queued", [False, True], ids=["direct", "worker"])
def test_native_local_round_trip(queued):
    from tensorrt_llm._torch.disaggregation.blob.kvcr_worker import _KvcrWorker
    from tensorrt_llm._torch.disaggregation.blob.store import RowStatus

    slot_bytes = 4096
    host = ctypes.create_string_buffer(2 * slot_bytes)
    cache = ctypes.create_string_buffer(4 * slot_bytes)
    allocations = (host, cache)
    _NATIVE_BUFFERS.append(allocations)
    address = ctypes.addressof(host)
    region = RegionSpec(address, len(host), RegionLayout(slot_bytes, 2))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listen_port = listener.getsockname()[1]
    factory = _KvcrWorker if queued else _KvcrLocalIO
    io = factory(
        region,
        Span(ctypes.addressof(cache), len(cache)),
        endpoint="trtllm-roundtrip",
        listen_port=listen_port,
    )
    payload = bytes(range(256)) * 16
    source = RowBatch.of([RowIO("roundtrip", (Span(address, slot_bytes),))])
    destination = RowBatch.of([RowIO("roundtrip", (Span(address + slot_bytes, slot_bytes),))])

    def wait(handle):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if queued:
                results = io.poll(handle)
                if all(result.concluded for result in results):
                    assert all(result.status is RowStatus.READY for result in results)
                    if io.access_ended(handle):
                        return
            else:
                for completed, entries in io.progress():
                    assert completed == handle
                    assert entries[b"roundtrip"].status is RowStatus.READY
                    return
            threading.Event().wait(0.001)
        pytest.fail("native KVCR operation did not complete within 10 seconds")

    try:
        ctypes.memmove(address, payload, slot_bytes)
        wait(io.put(source))
        ctypes.memset(address, 0, slot_bytes)
        destination_pattern = bytes(value ^ 0xFF for value in payload)
        ctypes.memmove(address + slot_bytes, destination_pattern, slot_bytes)
        assert host.raw[slot_bytes:] == destination_pattern
        delivery = io.get(destination)
        wait(delivery)
        assert host.raw[:slot_bytes] == bytes(slot_bytes)
        assert host.raw[slot_bytes:] == payload
        if queued:
            assert io.access_ended(delivery)
        else:
            assert not io.blocks_reuse(destination.extents)
    finally:
        if queued:
            assert io.close(10)
        else:
            io.close()
        _NATIVE_BUFFERS.remove(allocations)
