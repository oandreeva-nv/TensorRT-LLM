# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-facing queue and cached KVCR operation state."""

import gc
import threading
import time
import weakref
from collections import deque

import pytest

from tensorrt_llm._torch.disaggregation.blob.kvcr_worker import _KvcrWorker
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverRejected,
    RegionLayout,
    RegionSpec,
    RowBatch,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    faulted,
)

pytestmark = pytest.mark.cpu_only


class _ControlledIO:
    def __init__(self, *args, **kwargs):
        self.owner = threading.get_ident()
        self.started = threading.Event()
        self.resume = threading.Event()
        self.resume.set()
        self.operations = []
        self.completed = deque()
        self.hazard = threading.Event()
        self.fail_submit = False
        self.fail_progress = False
        self.fail_close = False
        self.closed = False

    def _check_owner(self):
        assert threading.get_ident() == self.owner

    def put(self, rows):
        self._check_owner()
        self.started.set()
        assert self.resume.wait(5)
        if self.fail_submit:
            raise RuntimeError("ambiguous submit")
        self.operations.append(rows)
        return len(self.operations)

    get = put

    def progress(self):
        self._check_owner()
        if self.fail_progress:
            raise RuntimeError("progress failed")
        try:
            return (self.completed.popleft(),)
        except IndexError:
            return ()

    def blocks_reuse(self, spans):
        self._check_owner()
        return self.hazard.is_set()

    def close(self):
        self._check_owner()
        if self.fail_close:
            raise RuntimeError("not quiescent")
        self.closed = True


def _rows(key="key", address=1024):
    return RowBatch.of([RowIO(key, (Span(address, 64),))])


def _until(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.001)
    pytest.fail("worker state did not change within 3 seconds")


def _stub_sdk(monkeypatch, create):
    def initialize(self, *args, **kwargs):
        self._runtime = create(*args, **kwargs)

    monkeypatch.setattr(_KvcrWorker, "_initialize_runtime", initialize)
    monkeypatch.setattr(_KvcrWorker, "_sdk_put", lambda self, rows: self._runtime.put(rows))
    monkeypatch.setattr(
        _KvcrWorker, "_sdk_get", lambda self, rows, hint=None: self._runtime.get(rows)
    )
    monkeypatch.setattr(_KvcrWorker, "_poll_completed", lambda self: self._runtime.progress())
    monkeypatch.setattr(
        _KvcrWorker, "_blocks_reuse", lambda self, spans: self._runtime.blocks_reuse(spans)
    )
    monkeypatch.setattr(_KvcrWorker, "_close_runtime", lambda self: self._runtime.close())


@pytest.fixture
def worker(monkeypatch):
    created = []

    def open_io(*args, **kwargs):
        io = _ControlledIO(*args, **kwargs)
        created.append(io)
        return io

    _stub_sdk(monkeypatch, open_io)
    result = _KvcrWorker(
        RegionSpec(1024, 136, RegionLayout(64, 2)),
        Span(4096, 256),
        endpoint="worker",
        listen_port=19341,
    )
    try:
        yield result, created[0]
    finally:
        created[0].resume.set()
        created[0].fail_close = False
        assert result.close(3)


def _complete(io, handle=1, key=b"key", success=True):
    result = RowResult(RowStatus.READY) if success else faulted("transfer failed")
    io.completed.append((handle, {key: result}))


def test_submission_and_queries_return_while_sdk_is_blocked(worker):
    driver, io = worker
    io.resume.clear()
    first = driver.put(_rows())
    assert io.started.wait(2)
    # The owner is blocked inside put; neither call below can wait for it.
    second = driver.get(_rows("second", 1088))
    assert driver.poll(first)[0].status is RowStatus.PENDING
    assert driver.poll(second)[0].status is RowStatus.PENDING
    assert not driver.access_ended(first)
    assert not driver.access_unprovable(first)
    assert io.owner != threading.get_ident()


def test_callbacks_observe_results_and_access_outside_locks(worker):
    driver, io = worker
    seen = []
    op = driver.put(
        _rows(),
        on_change=lambda handle: seen.append((driver.poll(handle), driver.access_ended(handle))),
    )
    _until(lambda: len(io.operations) == 1)
    _complete(io)
    _until(lambda: bool(seen))
    assert seen[-1][0][0].status is RowStatus.READY
    assert seen[-1][1]
    assert driver.access_ended(op)
    assert driver.drain(1)


def test_success_waits_for_late_quiescence_even_without_new_submissions(worker):
    driver, io = worker
    io.hazard.set()
    op = driver.get(_rows())
    _until(lambda: len(io.operations) == 1)
    _complete(io)
    _until(lambda: driver.poll(op)[0].status is RowStatus.READY)
    assert not driver.access_ended(op)
    io.hazard.clear()
    _until(lambda: driver.access_ended(op))


def test_failed_result_is_not_memory_release_evidence(worker):
    driver, io = worker
    op = driver.get(_rows())
    _until(lambda: len(io.operations) == 1)
    _complete(io, success=False)
    _until(lambda: driver.poll(op)[0].status is RowStatus.FAULT)
    assert not driver.access_ended(op)
    assert not driver.drain(0)
    io.fail_close = True
    assert not driver.close(1)
    assert not driver.access_ended(op)
    io.fail_close = False
    assert driver.close(2)
    assert driver.access_ended(op)
    assert driver.poll(op)[0].status is RowStatus.FAULT


def test_drain_cancels_queued_work_but_keeps_started_memory(worker):
    driver, io = worker
    io.resume.clear()
    first = driver.put(_rows())
    assert io.started.wait(2)
    second = driver.put(_rows("queued", 1088))
    assert not driver.drain(0)
    assert driver.poll(second)[0].status is RowStatus.CANCELLED
    assert driver.access_ended(second)
    assert not driver.access_ended(first)
    with pytest.raises(DriverRejected):
        driver.get(_rows())
    io.resume.set()
    _until(lambda: len(io.operations) == 1)
    _complete(io)
    assert driver.drain(2)
    assert len(io.operations) == 1


def test_ambiguous_submit_fences_and_never_releases_started_memory(worker):
    driver, io = worker
    io.fail_submit = True
    op = driver.put(_rows())
    _until(lambda: driver.access_unprovable(op))
    assert driver.poll(op)[0].status is RowStatus.FAULT
    assert not driver.access_ended(op)
    with pytest.raises(DriverRejected):
        driver.put(_rows())
    assert not driver.drain(0)


def test_wrong_completion_keys_fence_instead_of_publishing_success(worker):
    driver, io = worker
    op = driver.put(_rows())
    _until(lambda: len(io.operations) == 1)
    _complete(io, key=b"wrong")
    _until(lambda: driver.access_unprovable(op))
    assert driver.poll(op)[0].status is RowStatus.FAULT
    assert not driver.access_ended(op)


def test_rejects_geometry_before_queueing(worker):
    driver, io = worker
    with pytest.raises(DriverRejected):
        driver.put(_rows(address=1152))
    assert not io.operations


def test_region_extent_can_include_trailing_padding(worker):
    driver, io = worker
    rows = _rows()
    batch = RowBatch(rows.keys(), rows.addresses, rows.span_bytes, (Span(1024, 136),))
    op = driver.put(batch)
    _until(lambda: len(io.operations) == 1)
    _complete(io)
    _until(lambda: driver.access_ended(op))


def test_handle_from_elsewhere_is_rejected(worker):
    driver, _ = worker
    with pytest.raises(ValueError, match="handle"):
        driver.poll(object())


def test_progress_failure_fences_pending_operations(worker):
    driver, io = worker
    op = driver.get(_rows())
    _until(lambda: len(io.operations) == 1)
    io.fail_progress = True
    _until(lambda: driver.access_unprovable(op))
    assert driver.poll(op)[0].status is RowStatus.FAULT
    assert not driver.access_ended(op)
    assert driver.close(2)
    assert driver.access_ended(op)


def test_close_budget_does_not_interrupt_running_sdk_call(worker):
    driver, io = worker
    io.resume.clear()
    op = driver.put(_rows())
    assert io.started.wait(2)
    assert not driver.close(0)
    assert not driver.access_ended(op)
    assert not io.closed
    io.resume.set()
    assert driver.close(2)
    assert driver.access_ended(op)


def test_drain_waits_for_callback_to_return(worker):
    driver, io = worker
    entered = threading.Event()
    resume = threading.Event()

    def callback(op):
        entered.set()
        assert resume.wait(3)

    try:
        driver.put(_rows(), on_change=callback)
        _until(lambda: len(io.operations) == 1)
        _complete(io)
        assert entered.wait(2)
        assert not driver.drain(0)
    finally:
        resume.set()
    assert driver.drain(2)


def test_mixed_batch_results_follow_row_order(worker):
    driver, io = worker
    batch = RowBatch.of([RowIO("a", (Span(1024, 64),)), RowIO("b", (Span(1088, 64),))])
    op = driver.get(batch)
    _until(lambda: len(io.operations) == 1)
    io.completed.append((1, {b"b": faulted("transfer failed"), b"a": RowResult(RowStatus.READY)}))
    _until(lambda: driver.poll(op)[0].concluded)
    assert tuple(row.status for row in driver.poll(op)) == (RowStatus.READY, RowStatus.FAULT)
    assert not driver.access_ended(op)


def test_worker_retains_cache_owner_until_teardown(monkeypatch):
    _stub_sdk(monkeypatch, _ControlledIO)

    class CacheOwner:
        pass

    owner = CacheOwner()
    reference = weakref.ref(owner)
    driver = _KvcrWorker(
        RegionSpec(1024, 128, RegionLayout(64, 2)),
        Span(4096, 256),
        endpoint="cache-lifetime",
        listen_port=19341,
        cache_owner=owner,
    )
    try:
        del owner
        gc.collect()
        assert reference() is not None
    finally:
        assert driver.close(3)
    gc.collect()
    assert reference() is None
