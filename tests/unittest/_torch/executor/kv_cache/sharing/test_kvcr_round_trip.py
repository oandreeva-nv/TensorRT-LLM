# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in native KVCR transfers through real lender staging, without serving hooks."""

import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event
from time import monotonic

import pytest
import torch

from tensorrt_llm._torch.disaggregation.base.backend import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.backend import BlobBackend, BlobConfig, RegionBinding
from tensorrt_llm._torch.disaggregation.blob.kvcr_driver import KvcrDriver
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import StagingOptions, attach_lender
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.skipif(
    os.getenv("TRTLLM_TEST_KVCR_NATIVE") != "1" or not torch.cuda.is_available(),
    reason="requires native KVCR/UCX and CUDA",
)

_END = 96
_PROMPT = list(range(1000, 1097))


@dataclass(frozen=True)
class _HostRegion:
    """The host lender descriptor with the backend's explicit memory kind."""

    layout_label: str
    address: int
    nbytes: int
    slot_bytes: int
    slot_count: int
    memory_kind: MemoryKind = MemoryKind.HOST


def _ready_view(lease, manager):
    for _ in range(3):
        manager._stream.synchronize()
        view = lease.poll()
        if view is not None:
            return view
    pytest.fail(f"lender lease did not become ready: {lease.failure}")


def _consume(manager, lender, request, lease, outcome):
    lease.mark_arrived(list(outcome.delivered_masks))
    lease.release()
    for _ in range(3):
        manager._stream.synchronize()
        bounds = lender.resume_bounds(request)
        if bounds is not None:
            return bounds
    pytest.fail("lender did not finish GPU incorporation")


@contextmanager
def _store(manager, *, control_port=None):
    lender = attach_lender(manager, scope=b"kvcr-native-round-trip", staging=StagingOptions(_END))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    options = {} if control_port is None else {"control_port": control_port}
    driver = KvcrDriver(
        cache_slots=32, endpoint=f"kvcr-round-trip-{port}", listen_port=port, **options
    )
    regions = tuple(
        _HostRegion(r.layout_label, r.address, r.nbytes, r.slot_bytes, r.slot_count)
        for r in lender.regions
    )
    binding = RegionBinding(regions, driver, lender.hold_regions())
    binding.register()
    wake = Event()
    backend = BlobBackend(binding, wake.set, BlobConfig(key_prefix="native-test"))
    try:
        yield lender, backend, wake
    finally:
        assert not backend.shutdown(10), "KVCR still owns staging access"
        binding.release_hold()
        driver.close()


def _complete(attempt, wake):
    deadline = monotonic() + 10
    while monotonic() < deadline:
        outcome = attempt.poll()
        if outcome is not None and attempt.quiesced():
            assert outcome.fault is None
            return outcome
        wake.wait(0.001)
        wake.clear()
    pytest.fail("KVCR did not finish its staging access within 10 seconds")


def _write_lease(manager, lender, kit, request):
    lease = lender.lend_write(
        request,
        0,
        _END,
    )
    view = _ready_view(lease, manager)
    kit.fill_sentinel(manager, request)
    manager._stream.synchronize()
    return lease, view


def _pages(manager, kit, request):
    device = kit.DevicePages(manager)
    cache = kit.kv(manager, request)
    return {
        (group, ordinal): device.read(group, kit.pages(cache, group)[ordinal])
        for group in range(kit.num_layer_groups(manager))
        for ordinal in range(_END // manager.tokens_per_block)
    }


def test_native_cold_miss_preserves_destination_pages(real_manager, kit):
    with real_manager(dtype=DataType.BF16) as manager, _store(manager) as (lender, backend, wake):
        request = kit.admitted(manager, 1, _PROMPT)
        lease, view = _write_lease(manager, lender, kit, request)
        before = _pages(manager, kit, request)
        outcome = _complete(backend.fetch(view, None), wake)
        assert all(not mask.any() for mask in outcome.delivered_masks)
        bounds = _consume(manager, lender, request, lease, outcome)
        assert bounds.max_position == 0
        assert _pages(manager, kit, request) == before
        manager.free_resources(request)


def test_native_round_trip_restores_gpu_pages_from_kvcr(real_manager, kit):
    with real_manager(dtype=DataType.BF16) as manager, _store(manager) as (lender, backend, wake):
        producer = kit.published(manager, 1, _PROMPT)
        expected = _pages(manager, kit, producer)
        read = lender.lend_read(
            producer,
            0,
            _END,
        )
        source = _ready_view(read, manager)
        outcome = _complete(backend.publish(source, None), wake)
        assert all(mask.all() for mask in outcome.delivered_masks)
        read.release()
        # Destroy the producer bytes so they cannot satisfy the later comparison.
        kit.fill_sentinel(manager, producer)
        manager.free_resources(producer)

        consumer = kit.make_request(2, _PROMPT)
        assert manager.prepare_context_cache(consumer, reuse_limit=0) == 0
        kit.kv(manager, consumer).enable_swa_scratch_reuse = False
        write, destination = _write_lease(manager, lender, kit, consumer)
        assert _pages(manager, kit, consumer) != expected
        outcome = _complete(backend.fetch(destination, None), wake)
        assert all(mask.all() for mask in outcome.delivered_masks)
        bounds = _consume(manager, lender, consumer, write, outcome)
        assert bounds.max_position == _END
        assert _pages(manager, kit, consumer) == expected
        manager.free_resources(consumer)


def test_native_peer_fetch_between_lenders(real_manager, kit):
    from tensorrt_llm._torch.disaggregation.blob.kvcr_peer import KvcrPeerHint

    with socket.socket() as a, socket.socket() as b:
        a.bind(("127.0.0.1", 0))
        b.bind(("127.0.0.1", 0))
        source_port, target_port = a.getsockname()[1], b.getsockname()[1]
    with (
        real_manager(dtype=DataType.BF16) as source_manager,
        real_manager(dtype=DataType.BF16) as target_manager,
        _store(source_manager, control_port=source_port) as (
            source_lender,
            source_backend,
            source_wake,
        ),
        _store(target_manager, control_port=target_port) as (
            target_lender,
            target_backend,
            target_wake,
        ),
    ):
        producer = kit.published(source_manager, 1, _PROMPT)
        expected = _pages(source_manager, kit, producer)
        read = source_lender.lend_read(
            producer,
            0,
            _END,
        )
        source_view = _ready_view(read, source_manager)
        result = _complete(source_backend.publish(source_view, None), source_wake)
        assert all(mask.all() for mask in result.delivered_masks)
        read.release()
        kit.fill_sentinel(source_manager, producer)
        source_manager.free_resources(producer)

        consumer = kit.admitted(target_manager, 2, _PROMPT)
        write, destination = _write_lease(target_manager, target_lender, kit, consumer)
        before = _pages(target_manager, kit, consumer)
        assert before != expected
        # The target has its own empty KVCR cache: a local lookup must miss.
        miss = _complete(target_backend.fetch(destination, None), target_wake)
        assert all(not mask.any() for mask in miss.delivered_masks)
        assert _pages(target_manager, kit, consumer) == before

        hint = KvcrPeerHint(f"tcp://127.0.0.1:{source_port}")
        result = _complete(target_backend.fetch(destination, hint), target_wake)
        assert all(mask.all() for mask in result.delivered_masks)
        bounds = _consume(target_manager, target_lender, consumer, write, result)
        assert bounds.max_position == _END
        assert _pages(target_manager, kit, consumer) == expected
        target_manager.free_resources(consumer)
