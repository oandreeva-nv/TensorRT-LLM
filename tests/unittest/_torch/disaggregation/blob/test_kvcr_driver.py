# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KVCR's StoreDriver lifecycle and an opt-in native BlobBackend round trip."""

import os
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.kvcr_driver import KvcrDriver
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverRejected,
    RegionLayout,
    RegionSpec,
    RegistersRegionsTogether,
    RegistrationError,
    RowBatch,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    StoreDriver,
    StoreError,
)

pytestmark = pytest.mark.cpu_only
_RETAINED_NATIVE_ENDPOINTS = []


@dataclass(eq=False)
class _Op:
    rows: RowBatch
    on_change: object
    ended: bool = False


class _Worker:
    def __init__(self, region, cache, **kwargs):
        self.region = region
        self.cache = cache
        self.cache_owner = kwargs["cache_owner"]
        self.operations = []
        self.closed = False
        self.fail_close = False
        self.complete_on_submit = False
        self.return_entered = threading.Event()
        self.return_gate = threading.Event()
        self.return_gate.set()

    def put(self, rows, *, on_change=None):
        op = _Op(rows, on_change)
        self.operations.append(op)
        if self.complete_on_submit:
            op.ended = True
            on_change(op)
        self.return_entered.set()
        assert self.return_gate.wait(5)
        return op

    get = put

    def poll(self, op):
        return (RowResult(RowStatus.READY if op.ended else RowStatus.PENDING),) * len(op.rows)

    def access_ended(self, op):
        return op.ended

    def access_unprovable(self, op):
        return False

    def drain(self, budget_s):
        return all(op.ended for op in self.operations)

    def close(self, budget_s):
        if self.fail_close:
            return False
        self.closed = True
        self.cache_owner = None
        return True


@pytest.fixture
def workers(monkeypatch):
    created = []

    def create(*args, **kwargs):
        worker = _Worker(*args, **kwargs)
        created.append(worker)
        return worker

    monkeypatch.setattr(
        sys.modules["tensorrt_llm._torch.disaggregation.blob.kvcr_driver"], "_KvcrWorker", create
    )
    return created


def _driver(**kwargs):
    return KvcrDriver(cache_slots=4, endpoint="driver", listen_port=19341, **kwargs)


def _spec():
    return RegionSpec(1024, 128, RegionLayout(64, 2))


def _rows(key="key", address=1024):
    return RowBatch.of([RowIO(key, (Span(address, 64),))])


def test_implements_existing_driver_protocols(workers):
    driver = _driver()
    assert isinstance(driver, StoreDriver)
    assert isinstance(driver, RegistersRegionsTogether)
    handle = driver.register_regions([_spec()])
    assert len(workers) == 1
    assert workers[0].cache.nbytes == 256
    assert workers[0].cache_owner is not None
    assert driver.capabilities.get and driver.capabilities.put
    assert not driver.capabilities.hints
    assert not driver.capabilities.device_memory
    assert not driver.capabilities.multi_span
    assert driver.drain(1)
    assert not workers[0].closed
    with pytest.raises(DriverRejected):
        driver.close()
    driver.unregister_regions(handle)
    assert workers[0].closed
    driver.unregister_regions(handle)
    driver.close()
    driver.close()


@pytest.mark.parametrize(
    "specs",
    [
        [],
        [_spec(), RegionSpec(2048, 128, RegionLayout(64, 2))],
        [RegionSpec(1024, 128, RegionLayout(64, 2, memory_kind=MemoryKind.DEVICE))],
    ],
)
def test_unsupported_registration_rejected_before_sdk(workers, specs):
    driver = _driver()
    with pytest.raises(DriverRejected):
        driver.register_regions(specs)
    assert not workers
    assert driver.drain(0)
    driver.close()


def test_pending_memory_cannot_be_unregistered(workers):
    driver = _driver()
    handle = driver.register_regions([_spec()])
    op = driver.put(_rows())
    with pytest.raises(DriverRejected):
        driver.unregister_regions(handle)
    assert not workers[0].closed
    assert not driver.access_ended(op)
    assert not driver.drain(0)
    op.ended = True
    assert driver.drain(1)
    driver.unregister_regions(handle)
    driver.close()


def test_failed_unregistration_retains_registration_for_retry(workers):
    driver = _driver()
    handle = driver.register_regions([_spec()])
    assert driver.drain(1)
    workers[0].fail_close = True
    with pytest.raises(StoreError, match="retain"):
        driver.unregister_regions(handle)
    assert workers[0].cache_owner is not None
    with pytest.raises(DriverRejected):
        driver.close()
    workers[0].fail_close = False
    driver.unregister_regions(handle)
    driver.close()


def test_callback_can_query_handle_before_submission_returns(workers):
    driver = _driver()
    handle = driver.register_regions([_spec()])
    workers[0].complete_on_submit = True
    seen = []

    def changed(current):
        with ThreadPoolExecutor(1) as executor:
            result = executor.submit(driver.poll, current).result(timeout=1)
        seen.append((result, driver.access_ended(current)))

    op = driver.put(
        _rows(),
        on_change=changed,
    )
    assert seen == [(driver.poll(op), True)]
    assert driver.drain(1)
    driver.unregister_regions(handle)
    driver.close()
    with pytest.raises(RuntimeError, match="closed"):
        driver.poll(op)


def test_rejects_unsupported_hint_and_excess_rows_before_submission(workers):
    driver = _driver(max_rows_per_op=1)
    handle = driver.register_regions([_spec()])
    with pytest.raises(DriverRejected):
        driver.get(_rows(), hint="peer")
    with pytest.raises(DriverRejected):
        driver.put(RowBatch.of([RowIO("a", (Span(1024, 64),)), RowIO("b", (Span(1088, 64),))]))
    assert not workers[0].operations
    assert driver.drain(1)
    driver.unregister_regions(handle)
    driver.close()


def test_unregister_cannot_race_a_handle_being_handed_over(workers):
    driver = _driver()
    handle = driver.register_regions([_spec()])
    worker = workers[0]
    worker.complete_on_submit = True
    worker.return_gate.clear()
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(driver.put, _rows())
        try:
            assert worker.return_entered.wait(2)
            with pytest.raises(DriverRejected):
                driver.unregister_regions(handle)
            assert not worker.closed
        finally:
            worker.return_gate.set()
        op = future.result(timeout=2)
    assert driver.access_ended(op)
    assert driver.drain(1)
    driver.unregister_regions(handle)
    driver.close()


def test_startup_failure_keeps_a_registration_handle_and_owned_cache(workers, monkeypatch):
    module = sys.modules["tensorrt_llm._torch.disaggregation.blob.kvcr_driver"]

    def fail(*args, **kwargs):
        raise RuntimeError("native startup failed")

    monkeypatch.setattr(module, "_KvcrWorker", fail)
    driver = _driver()
    with pytest.raises(RegistrationError) as caught:
        driver.register_regions([_spec()])
    assert caught.value.handle is not None
    assert not driver.drain(0)
    with pytest.raises(StoreError):
        driver.unregister_regions(caught.value.handle)
    with pytest.raises(DriverRejected):
        driver.close()


def test_allocation_failure_is_certified_before_contacting_sdk(workers, monkeypatch):
    import ctypes

    def fail(size):
        raise MemoryError("no cache capacity")

    monkeypatch.setattr(ctypes, "create_string_buffer", fail)
    driver = _driver()
    with pytest.raises(RegistrationError) as caught:
        driver.register_regions([_spec()])
    assert caught.value.handle is None
    assert not workers
    assert driver.drain(0)
    driver.close()


@pytest.mark.skipif(os.getenv("TRTLLM_TEST_KVCR_NATIVE") != "1", reason="requires native KVCR/UCX")
@pytest.mark.parametrize("scenario", ["roundtrip", "duplicate", "existing"])
def test_native_shared_driver_contract(scenario):
    from tensorrt_llm._torch.disaggregation.testing.driver_conformance import DriverConformance
    from tensorrt_llm._torch.disaggregation.testing.host_memory import (
        GUARD,
        HostBuffer,
        MemoryWindow,
    )

    memory = HostBuffer(512)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    driver = KvcrDriver(cache_slots=64, endpoint=f"kvcr-contract-{scenario}", listen_port=port)
    retained = (memory, driver)
    _RETAINED_NATIVE_ENDPOINTS.append(retained)
    handle = driver.register_regions(
        [RegionSpec(memory.address, memory.nbytes, RegionLayout(8, 64))]
    )

    class PoolContract(DriverConformance):
        registers_on_demand = False
        registers_together = True
        offset = 0

        def make_driver(self):
            return driver

        def allocate(self, nbytes, fill, memory_kind):
            assert memory_kind is MemoryKind.HOST
            assert self.offset + nbytes + 2 * GUARD <= memory.nbytes
            window = MemoryWindow(memory.address + self.offset + GUARD, nbytes, fill, owner=memory)
            self.offset += nbytes + 2 * GUARD
            return window, handle

        def shutdown(self, current):
            current.unregister_regions(handle)
            current.close()

    suite = PoolContract()
    checks = {
        "roundtrip": suite.test_put_then_get_moves_whole_rows_and_never_aliases_the_source,
        "duplicate": suite.test_a_duplicate_put_of_a_committed_object_is_ready,
        "existing": suite.test_a_put_under_an_existing_key_leaves_the_object_whole,
    }
    checks[scenario]()
    driver.close()
    assert memory.canaries_intact()
    _RETAINED_NATIVE_ENDPOINTS.remove(retained)


@pytest.mark.skipif(os.getenv("TRTLLM_TEST_KVCR_NATIVE") != "1", reason="requires native KVCR/UCX")
def test_native_blob_backend_round_trip_and_registration_hold():
    from cpu_fixtures import CpuRegion, CpuView, FakeHold, Rows, masks, object_key, wait_until

    from tensorrt_llm._torch.disaggregation.blob.backend import (
        BlobBackend,
        BlobConfig,
        RegionBinding,
    )

    region = CpuRegion(4096, 4)
    hold = FakeHold()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    driver = KvcrDriver(
        cache_slots=8, endpoint="kvcr-blob-test", listen_port=port, max_rows_per_op=1
    )
    binding = RegionBinding([region], driver, hold)
    retained = (region, driver, binding)
    _RETAINED_NATIVE_ENDPOINTS.append(retained)
    binding.register()
    wake = threading.Event()
    backend = BlobBackend(binding, wake.set, BlobConfig())
    keys = np.stack([object_key("A"), object_key("B")])
    payloads = [bytes(range(256)) * 16, bytes(value ^ 0x55 for value in range(256)) * 16]

    def wait(attempt):
        wait_until(lambda: attempt.poll() is not None, "the backend outcome")
        outcome = attempt.poll()
        assert outcome.fault is None
        assert masks(outcome) == [[True, True]]
        wait_until(attempt.quiesced, "the backend to end access")

    try:
        for index, payload in enumerate(payloads):
            region.write(index, payload)
        wait(backend.publish(CpuView([Rows(0, keys, region.slots(0, 1), 0)]), None))
        for index, payload in enumerate(payloads):
            region.write(index, bytes(4096))
            region.write(index + 2, bytes(value ^ 0xFF for value in payload))
        wait(backend.fetch(CpuView([Rows(0, keys, region.slots(2, 3), 0)]), None))
        assert [region.read(2), region.read(3)] == payloads
        assert region.read(0) == region.read(1) == bytes(4096)
        assert region.buffer.canaries_intact()
        assert wake.is_set()
        assert not hold.released
    finally:
        labels = backend.shutdown(10)
        if not labels:
            binding.release_hold()
            driver.close()
            _RETAINED_NATIVE_ENDPOINTS.remove(retained)
        assert not labels
    assert hold.release_threads == [threading.get_ident()]
