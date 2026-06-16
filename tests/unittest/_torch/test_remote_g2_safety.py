# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CONNECTOR_PACKAGE = "tensorrt_llm._torch.pyexecutor.connectors"
_REMOTE_G2_PATH = (
    _ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "connectors" / "remote_g2.py"
)
_REMOTE_G2_CONNECTOR_PATH = (
    _ROOT
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
    / "remote_g2_connector.py"
)
_REMOTE_G2_TRANSFER_PATH = (
    _ROOT
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
    / "remote_g2_transfer.py"
)
_REMOTE_G2_OBSERVABILITY_PATH = (
    _ROOT
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
    / "remote_g2_observability.py"
)

ROADMAP_FAILURE_SET = {
    "stale_router_event",
    "dropped_event_no_plan",
    "duplicate_event_callback",
    "source_restart_stale_generation",
    "target_cancellation",
    "allocation_failure",
    "transfer_failure",
    "timeout",
    "lease_leak",
}


def _install_package(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_connector_modules():
    _install_package("tensorrt_llm")
    _install_package("tensorrt_llm._torch")
    _install_package("tensorrt_llm._torch.pyexecutor")
    _install_package(_CONNECTOR_PACKAGE)

    kv_cache_connector = types.ModuleType(f"{_CONNECTOR_PACKAGE}.kv_cache_connector")

    class KvCacheConnectorScheduler:
        def __init__(self, llm_args):
            self._llm_args = llm_args

    class KvCacheConnectorWorker:
        def __init__(self, llm_args):
            self._llm_args = llm_args
            self._metadata = None

        def bind_connector_meta(self, metadata):
            self._metadata = metadata

        def get_connector_meta(self):
            return self._metadata

    kv_cache_connector.KvCacheConnectorScheduler = KvCacheConnectorScheduler
    kv_cache_connector.KvCacheConnectorWorker = KvCacheConnectorWorker
    kv_cache_connector.SchedulerOutput = object
    sys.modules[f"{_CONNECTOR_PACKAGE}.kv_cache_connector"] = kv_cache_connector

    observability = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_observability",
        _REMOTE_G2_OBSERVABILITY_PATH,
    )
    remote_g2 = _load_module(f"{_CONNECTOR_PACKAGE}.remote_g2", _REMOTE_G2_PATH)
    transfer = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_transfer", _REMOTE_G2_TRANSFER_PATH
    )
    connector = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_connector", _REMOTE_G2_CONNECTOR_PATH
    )
    return observability, remote_g2, transfer, connector


OBSERVABILITY, REMOTE_G2, TRANSFER, CONNECTOR = _load_connector_modules()


@pytest.fixture(autouse=True)
def _install_identity_slot_lookup():
    """Production wires this up via maybe_start_remote_g2_target_client; the
    unit tests construct TargetRemoteG2BindingStore directly and bypass that
    setup, so without a stub bind_target_blocks fails with
    target_slot_lookup_failed.  Install an identity stub (slot_idx == block_id)
    for the test, restore previous value after."""
    saved = CONNECTOR._installed_block_id_to_slot_idx
    CONNECTOR.install_block_id_to_slot_idx(lambda ids: list(ids))
    yield
    CONNECTOR._installed_block_id_to_slot_idx = saved


InMemoryRemoteG2ObservabilitySink = OBSERVABILITY.InMemoryRemoteG2ObservabilitySink
RemoteG2ConnectorMetadata = CONNECTOR.RemoteG2ConnectorMetadata
RemoteG2Descriptor = REMOTE_G2.RemoteG2Descriptor
RemoteG2ResolveResult = REMOTE_G2.RemoteG2ResolveResult
RemoteKvReusePlan = REMOTE_G2.RemoteKvReusePlan
TargetRemoteG2BindingStore = REMOTE_G2.TargetRemoteG2BindingStore
TargetRemotePlanStore = REMOTE_G2.TargetRemotePlanStore
compute_remote_g2_matched_tokens = REMOTE_G2.compute_remote_g2_matched_tokens

RemoteG2NixlTransferAdapter = TRANSFER.RemoteG2NixlTransferAdapter
RemoteG2SourceMetadata = TRANSFER.RemoteG2SourceMetadata
RemoteG2TransferDescriptor = TRANSFER.RemoteG2TransferDescriptor


class _FakeTransferResult:
    def __init__(self, record, completed=True, fail=False):
        self.record = record
        self.completed = completed
        self.fail = fail
        self.released = 0

    def is_completed(self):
        if self.fail:
            raise RuntimeError("transfer failed")
        return self.completed

    def release(self):
        self.released += 1


class _FakeTransferAdapter:
    def __init__(self, result_factory=None):
        self.started = []
        self.result_factory = result_factory

    def start_transfer(self, record):
        self.started.append(record)
        if self.result_factory is not None:
            return self.result_factory(record)
        return _FakeTransferResult(record)


class _FakeMemoryDescs:
    def __init__(self, type, descs):
        self.type = type
        self.descs = descs


class _FakeRegMemoryDescs(_FakeMemoryDescs):
    pass


class _FakeTransferOp:
    READ = "READ"


@dataclass
class _FakeTransferRequest:
    op: str
    src_descs: _FakeMemoryDescs
    dst_descs: _FakeMemoryDescs
    remote_name: str
    sync_message: str | None = None


class _FakeAgent:
    def __init__(self):
        self.loaded = []
        self.registered = []
        self.deregistered = []
        self.requests = []

    def load_remote_agent(self, name, agent_desc):
        self.loaded.append((name, agent_desc))

    def register_memory(self, descs):
        self.registered.append(descs)

    def deregister_memory(self, descs):
        self.deregistered.append(descs)

    def submit_transfer_requests(self, request):
        self.requests.append(request)
        return _FakeTransferResult(SimpleNamespace())


_FAKE_TRANSFER_TYPES = SimpleNamespace(
    MemoryDescs=_FakeMemoryDescs,
    RegMemoryDescs=_FakeRegMemoryDescs,
    TransferOp=_FakeTransferOp,
    TransferRequest=_FakeTransferRequest,
)


def _plan(**overrides):
    plan = {
        "plan_id": "plan-1",
        "request_id": "dynamo-request-1",
        "target_worker_id": 42,
        "target_dp_rank": 2,
        "source_worker_id": 7,
        "source_dp_rank": 0,
        "source_tier": "host_pinned",
        "block_hashes": [11, 22, 33],
        "start_block_index": 0,
        "planned_prefix_blocks": 3,
        "block_size_tokens": 16,
        "created_at_ms": 100,
        "expires_at_ms": 10_000,
    }
    plan.update(overrides)
    return plan


def _descriptor(block_hash):
    return RemoteG2Descriptor(
        block_hash=block_hash,
        descriptor_generation=1,
        pool_id=f"source-{block_hash}",
        byte_offset=0,
        byte_length=4096,
        metadata={
            "nixl_memory_desc": {
                "ptr": block_hash * 8192,
                "size": 4096,
                "device_id": 0,
                "memory_type": "DRAM",
                "name": f"source-{block_hash}",
            }
        },
    )


def _resolve_result(block_hashes=(11, 22, 33), num_tokens=48, lease_id="lease-1"):
    return RemoteG2ResolveResult(
        lease_id=lease_id,
        descriptors=tuple(_descriptor(block_hash) for block_hash in block_hashes),
        num_tokens=num_tokens,
        source_generation=99,
    )


def _bound_record(request_id=1234, lease_id="lease-bound"):
    store = TargetRemoteG2BindingStore(release_lease=lambda lease_id, reason: True)
    record = store.resolve_for_request(
        request_id,
        RemoteKvReusePlan.from_dict(_plan(plan_id=f"plan-{request_id}")),
        16,
        lambda plan: _resolve_result(lease_id=lease_id),
    )
    store.bind_target_blocks(request_id, [100, 101, 102])
    return record


def _source_metadata(worker_id=7, generation=99):
    return RemoteG2SourceMetadata(
        source_worker_id=worker_id,
        source_generation=generation,
        remote_name=f"source-{worker_id}",
        agent_desc=b"agent-desc",
    )


def _target_descriptors(record):
    return [
        RemoteG2TransferDescriptor(
            ptr=target_block.target_block_id * 16384,
            size=4096,
            device_id=0,
            memory_type="VRAM",
            name=f"target-{target_block.target_block_id}",
        )
        for target_block in record.bound_blocks
    ]


def _event_names(sink):
    return [event.event for event in sink.events]


def test_remote_g2_decision_equivalence_for_representative_prefixes():
    cases = [
        ("zero_remote_match", (), 0, 0, 0, "no_remote_g2_match"),
        ("one_block", (11,), 16, 0, 16, None),
        ("multi_block", (11, 22, 33), 48, 0, 48, None),
        ("first_miss_truncation", (11, 22), 32, 0, 32, None),
        (
            "unaligned_computed_boundary",
            (11, 22),
            32,
            7,
            0,
            "unaligned_num_computed_tokens",
        ),
    ]
    sink = InMemoryRemoteG2ObservabilitySink()
    released = []
    store = TargetRemoteG2BindingStore(
        release_lease=lambda lease_id, reason: released.append((lease_id, reason)) or True,
        observability=sink,
    )

    for index, (name, hashes, num_tokens, computed, expected, release_reason) in enumerate(
        cases
    ):
        result = _resolve_result(
            block_hashes=hashes,
            num_tokens=num_tokens,
            lease_id=f"lease-{name}",
        )

        # D-06/D-07/D-08/D-09: local validation asserts decision equivalence.
        assert (
            compute_remote_g2_matched_tokens(RemoteKvReusePlan.from_dict(_plan(plan_id=f"plan-{name}")), result, computed, 16) == expected
        ), name
        record = store.resolve_for_request(
            1000 + index,
            RemoteKvReusePlan.from_dict(_plan(plan_id=f"plan-{name}")),
            computed,
            lambda plan, result=result: result,
        )
        if expected:
            assert record is not None
            assert record.matched_tokens == expected
        else:
            assert record is None
            assert (f"lease-{name}", release_reason) in released

    names = _event_names(sink)
    assert "resolved" in names
    assert "truncated" in names
    assert "fallback" in names
    assert "released" in names


def test_compute_matched_returns_zero_when_b_prefix_short_of_plan_start():
    # Scenario: source A has request blocks 0..4 (Device 0-3, HostPinned 4).
    # Planner emits plan covering position 4 only (start_block_index=4).
    # Target B has 0 computed tokens — its prefix ends before the plan starts.
    # Attaching the plan's block 4 at B's position 0 would corrupt the cache,
    # so the function must return 0 and let B recompute locally.
    plan = RemoteKvReusePlan.from_dict(
        _plan(
            plan_id="gap",
            block_hashes=[44],
            start_block_index=4,
            planned_prefix_blocks=1,
        )
    )
    result = _resolve_result(block_hashes=(44,), num_tokens=16, lease_id="lease-gap")
    assert compute_remote_g2_matched_tokens(plan, result, 0, 16) == 0


def test_compute_matched_handles_partial_overlap_with_target_prefix():
    # Plan covers request positions [4, 7) (3 blocks). Target B has positions
    # 0..4 cached on Device (computed_blocks = 5). B already has the plan's
    # first block; the remaining two are net-new. Expect 2 * block_size matched.
    plan = RemoteKvReusePlan.from_dict(
        _plan(
            plan_id="overlap",
            block_hashes=[44, 55, 66],
            start_block_index=4,
            planned_prefix_blocks=3,
        )
    )
    result = _resolve_result(
        block_hashes=(44, 55, 66), num_tokens=48, lease_id="lease-overlap"
    )
    assert compute_remote_g2_matched_tokens(plan, result, 5 * 16, 16) == 2 * 16


def test_compute_matched_returns_zero_when_b_prefix_past_plan_end():
    # Plan covers [4, 7); B has 8 blocks cached already → plan is redundant.
    plan = RemoteKvReusePlan.from_dict(
        _plan(
            plan_id="past",
            block_hashes=[44, 55, 66],
            start_block_index=4,
            planned_prefix_blocks=3,
        )
    )
    result = _resolve_result(
        block_hashes=(44, 55, 66), num_tokens=48, lease_id="lease-past"
    )
    assert compute_remote_g2_matched_tokens(plan, result, 8 * 16, 16) == 0


def test_remote_g2_fault_injection_covers_roadmap_failure_set():
    assert ROADMAP_FAILURE_SET == {
        "stale_router_event",
        "dropped_event_no_plan",
        "duplicate_event_callback",
        "source_restart_stale_generation",
        "target_cancellation",
        "allocation_failure",
        "transfer_failure",
        "timeout",
        "lease_leak",
    }

    sink = InMemoryRemoteG2ObservabilitySink()
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 20_000, observability=sink)
    assert plan_store.put(1, _plan(expires_at_ms=20_000)) is None

    scheduler = CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=TargetRemotePlanStore(clock_ms=lambda: 500),
        resolve_and_lease=lambda plan: _resolve_result(),
        release_lease=lambda lease_id, reason: True,
        observability=sink,
    )
    assert scheduler.get_num_new_matched_tokens(SimpleNamespace(request_id=404), 0) == (
        0,
        False,
    )

    adapter = _FakeTransferAdapter()
    released = []
    worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=adapter,
        release_lease=lambda lease_id, reason: released.append((lease_id, reason))
        or True,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
        observability=sink,
    )
    record = _bound_record()
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(record,)))
    worker.start_load_kv(None)
    worker.start_load_kv(None)
    assert adapter.started == [record]
    assert worker.get_finished([], [1234]) == ([], [1234])
    # Fix 1 (deferred release): release is deferred until allgather.
    assert released == []
    worker.on_globally_finished_loading({1234})
    assert released == [("lease-bound", "transfer_succeeded")]


def test_remote_g2_duplicate_callbacks_do_not_double_transfer_or_release():
    adapter = _FakeTransferAdapter()
    released = []
    worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=adapter,
        release_lease=lambda lease_id, reason: released.append((lease_id, reason))
        or True,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
    )
    record = _bound_record()
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(record,)))

    worker.start_load_kv(None)
    worker.start_load_kv(None)
    assert adapter.started == [record]
    assert worker.get_finished([], [1234]) == ([], [1234])
    assert worker.get_finished([], [1234]) == ([], [])
    # Fix 1 (deferred release): release is deferred until allgather.
    assert released == []
    worker.on_globally_finished_loading({1234})
    assert released == [("lease-bound", "transfer_succeeded")]


def test_remote_g2_target_cancellation_releases_once_and_publishes_nothing():
    released = []
    store = TargetRemoteG2BindingStore(
        release_lease=lambda lease_id, reason: released.append((lease_id, reason)) or True
    )
    record = store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        0,
        lambda plan: _resolve_result(lease_id="lease-cancelled"),
    )

    assert store.discard(1234, "cancelled") is True
    assert store.discard(1234, "cancelled") is False
    assert record.state == REMOTE_G2.RemoteG2BindingState.CANCELLED
    assert released == [("lease-cancelled", "cancelled")]


def test_remote_g2_allocation_failure_releases_once_and_publishes_nothing():
    released = []
    store = TargetRemoteG2BindingStore(
        release_lease=lambda lease_id, reason: released.append((lease_id, reason)) or True
    )
    record = store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        0,
        lambda plan: _resolve_result(lease_id="lease-allocation-failed"),
    )

    failed = store.bind_target_blocks(1234, [100, 101])
    assert failed is record
    assert record.state == REMOTE_G2.RemoteG2BindingState.BIND_FAILED
    assert released == [("lease-allocation-failed", "target_binding_failed")]


def test_remote_g2_transfer_failure_and_timeout_release_once():
    failure_released = []

    def failing_result(record):
        return _FakeTransferResult(record, completed=False, fail=True)

    failure_worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(failing_result),
        release_lease=lambda lease_id, reason: failure_released.append(
            (lease_id, reason)
        )
        or True,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
    )
    failure_worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(bindings=(_bound_record(lease_id="lease-failed"),))
    )
    failure_worker.start_load_kv(None)

    with pytest.raises(RuntimeError, match="failed closed"):
        failure_worker.get_finished([], [1234])
    assert failure_released == [("lease-failed", "transfer_failed")]

    timeout_released = []
    timeout_worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(
            lambda record: _FakeTransferResult(record, completed=False)
        ),
        release_lease=lambda lease_id, reason: timeout_released.append(
            (lease_id, reason)
        )
        or True,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
        transfer_timeout_ms=-1,
    )
    timeout_worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(bindings=(_bound_record(lease_id="lease-timeout"),))
    )
    timeout_worker.start_load_kv(None)

    with pytest.raises(RuntimeError, match="timed out"):
        timeout_worker.get_finished([], [1234])
    assert timeout_released == [("lease-timeout", "transfer_timeout")]


def test_remote_g2_source_restart_metadata_mismatch_cleans_up_without_publication():
    agent = _FakeAgent()
    sink = InMemoryRemoteG2ObservabilitySink()
    released = []
    marked_valid = []
    published = []
    adapter = RemoteG2NixlTransferAdapter(
        source_metadata_fetcher=lambda worker_id, generation: _source_metadata(
            worker_id, generation + 1
        ),
        target_descriptor_resolver=_target_descriptors,
        agent_factory=lambda: agent,
        transfer_types=_FAKE_TRANSFER_TYPES,
    )
    worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=adapter,
        release_lease=lambda lease_id, reason: released.append((lease_id, reason))
        or True,
        mark_local_valid=marked_valid.append,
        publish_binding=published.append,
        observability=sink,
    )
    worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(bindings=(_bound_record(lease_id="lease-restart"),))
    )

    with pytest.raises(RuntimeError, match="failed to start"):
        worker.start_load_kv(None)

    assert agent.requests == []
    assert marked_valid == []
    assert published == []
    assert released == [("lease-restart", "transfer_start_failed")]
    names = _event_names(sink)
    assert "fallback" in names
    assert "released" in names
    assert "transferred" not in names


def test_remote_g2_lease_release_is_exactly_once_per_attempt():
    released = []
    worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(),
        release_lease=lambda lease_id, reason: released.append((lease_id, reason))
        or True,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
    )
    record = _bound_record(lease_id="lease-exact-once")
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(record,)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [1234])
    # Fix 1 (deferred release): release is deferred until allgather.
    assert released == []
    worker.on_globally_finished_loading({1234})
    assert released == [("lease-exact-once", "transfer_succeeded")]
    # Duplicate call must not produce a second release.
    worker._release_record_once(record, "duplicate_after_success")
    assert released == [("lease-exact-once", "transfer_succeeded")]


def test_remote_g2_observability_contract_covers_required_events_without_raw_addresses():
    sink = InMemoryRemoteG2ObservabilitySink()
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500, observability=sink)
    plan_store.put(1234, _plan())

    store = TargetRemoteG2BindingStore(
        release_lease=lambda lease_id, reason: True,
        observability=sink,
    )
    store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        0,
        lambda plan: _resolve_result(block_hashes=(11, 22), num_tokens=32),
    )
    store.resolve_for_request(
        5678,
        RemoteKvReusePlan.from_dict(_plan(plan_id="plan-fallback")),
        7,
        lambda plan: _resolve_result(lease_id="lease-fallback"),
    )

    success_worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(),
        release_lease=lambda lease_id, reason: True,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
        observability=sink,
    )
    success_worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(bindings=(_bound_record(lease_id="lease-success"),))
    )
    success_worker.start_load_kv(None)
    success_worker.get_finished([], [1234])

    failed_worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(),
        release_lease=lambda lease_id, reason: True,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: (_ for _ in ()).throw(
            RuntimeError("publish failed")
        ),
        observability=sink,
    )
    failed_worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(bindings=(_bound_record(lease_id="lease-failed"),))
    )
    failed_worker.start_load_kv(None)
    with pytest.raises(RuntimeError, match="failed closed"):
        failed_worker.get_finished([], [1234])

    assert {
        "planned",
        "resolved",
        "truncated",
        "transferred",
        "fallback",
        "failed",
        "released",
    }.issubset(set(_event_names(sink)))

    forbidden = (
        "ptr",
        "address",
        "descriptor",
        "nixl",
        "transfer_tuple",
        "memory_desc",
        "metadata",
    )
    high_cardinality = ("1234", "plan-1", "lease-", "source_worker", "generation")
    for key in sink.counts:
        assert len(key) == 4
        key_text = " ".join(key)
        assert not any(value in key_text for value in high_cardinality)
    for event in sink.events:
        detail_text = " ".join(event.details)
        assert not any(value in detail_text for value in forbidden)


# ── Gather rollback prefill-pin test ────────────────────────────────


def _load_source_setup():
    """Load remote_g2_source_setup for prefill pin helpers.

    remote_g2_source_setup does ``from .remote_g2_source_adapter import ...``,
    so the adapter must be pre-loaded under the package namespace before we
    attempt to load the setup module (same pattern _load_connector_modules
    uses for the other connector siblings).
    """
    connectors = (
        _ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "connectors"
    )
    _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_source_adapter",
        connectors / "remote_g2_source_adapter.py",
    )
    return _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_source_setup",
        connectors / "remote_g2_source_setup.py",
    )


SOURCE_SETUP = _load_source_setup()


def test_gather_rollback_pops_prefill_pins():
    """Fix 6 completeness: verify that prefill pins registered via
    register_prefill_pins are removed by pop_prefill_pin during
    gather rollback, matching the release_lease handler's behavior.
    """
    register = SOURCE_SETUP.register_prefill_pins
    pop = SOURCE_SETUP.pop_prefill_pin
    has_pins = SOURCE_SETUP.has_prefill_pins
    pinned = SOURCE_SETUP._prefill_pinned
    lock = SOURCE_SETUP._prefill_pin_lock

    # Clear any leftover state from other tests.
    with lock:
        pinned.clear()

    block_ids = [100, 200, 300]
    register(block_ids)
    assert has_pins()
    assert len(pinned) == 3

    # Simulate rollback: pop each pin (mirrors the rollback path
    # which calls pop_prefill_pin for each lease pin ref).
    popped = [pop(bid) for bid in block_ids]
    assert popped == [True, True, True]
    assert not has_pins()

    # Double-pop returns False (idempotent).
    assert pop(100) is False

    # Clean up.
    with lock:
        pinned.clear()


def test_should_prefill_pin_lease_gate():
    """Fix 10: should_prefill_pin returns False when no remote resolve has
    been seen, True after notify_remote_resolve_seen, and False again when
    free blocks drop to or below the safety floor (max_blocks_per_seq).

    Fix 10a: est_new_blocks overshoot protection and auto-cap
    (total_primary_blocks // 4).
    """
    should_pin = SOURCE_SETUP.should_prefill_pin
    notify = SOURCE_SETUP.notify_remote_resolve_seen
    pinned = SOURCE_SETUP._prefill_pinned
    lock = SOURCE_SETUP._prefill_pin_lock

    # Save and reset module-level state.
    _saved_flag = SOURCE_SETUP._remote_resolve_seen
    SOURCE_SETUP._remote_resolve_seen = False
    with lock:
        _saved_pins = dict(pinned)
        pinned.clear()

    try:
        # ── Lease gate ──────────────────────────────────────────────
        # No remote resolve seen → never pin, regardless of free blocks.
        assert should_pin(free_blocks=100, max_blocks_per_seq=10,
                          pin_budget=64) is False

        # Simulate first remote resolve arriving.
        notify()
        assert SOURCE_SETUP.is_remote_resolve_active()

        # Plenty of free blocks → should pin.
        assert should_pin(free_blocks=100, max_blocks_per_seq=10,
                          pin_budget=64) is True

        # ── Safety floor ────────────────────────────────────────────
        # Free blocks at safety floor → refuse to pin.
        assert should_pin(free_blocks=10, max_blocks_per_seq=10,
                          pin_budget=64) is False

        # Free blocks below safety floor → refuse to pin.
        assert should_pin(free_blocks=5, max_blocks_per_seq=10,
                          pin_budget=64) is False

        # ── Budget exhaustion ───────────────────────────────────────
        with lock:
            for i in range(64):
                pinned[i] = 0.0
        assert should_pin(free_blocks=100, max_blocks_per_seq=10,
                          pin_budget=64) is False

        # ── Overshoot protection (Fix 10a) ──────────────────────────
        with lock:
            pinned.clear()
            # 60 pins already registered.
            for i in range(60):
                pinned[i] = 0.0
        # A 10-block request would push to 70, exceeding budget=64.
        assert should_pin(free_blocks=100, max_blocks_per_seq=10,
                          pin_budget=64, est_new_blocks=10) is False
        # A 4-block request would push to 64 — exactly at limit → ok.
        assert should_pin(free_blocks=100, max_blocks_per_seq=10,
                          pin_budget=64, est_new_blocks=4) is True
        # A 5-block request would push to 65 → over limit.
        assert should_pin(free_blocks=100, max_blocks_per_seq=10,
                          pin_budget=64, est_new_blocks=5) is False

        # ── Auto-cap (25 % of primary pool, Fix 10a) ───────────────
        with lock:
            pinned.clear()
        # total_primary=80 → effective budget = min(64, 80//4) = 20.
        assert should_pin(free_blocks=100, max_blocks_per_seq=10,
                          pin_budget=64, total_primary_blocks=80) is True
        with lock:
            for i in range(20):
                pinned[i] = 0.0
        # 20 pins == effective budget of 20 → refuse.
        assert should_pin(free_blocks=100, max_blocks_per_seq=10,
                          pin_budget=64, total_primary_blocks=80) is False
        # total_primary=0 (unknown) → skip auto-cap, use pin_budget.
        with lock:
            pinned.clear()
            for i in range(32):
                pinned[i] = 0.0
        assert should_pin(free_blocks=100, max_blocks_per_seq=10,
                          pin_budget=64, total_primary_blocks=0) is True

    finally:
        # Restore module-level state.
        SOURCE_SETUP._remote_resolve_seen = _saved_flag
        with lock:
            pinned.clear()
            pinned.update(_saved_pins)


def test_register_prefill_pins_returns_stale():
    """Fix 10c: register_prefill_pins returns stale block IDs after every
    _PREFILL_SWEEP_INTERVAL calls so stale sweep runs independently of
    release_lease.
    """
    register = SOURCE_SETUP.register_prefill_pins
    pinned = SOURCE_SETUP._prefill_pinned
    lock = SOURCE_SETUP._prefill_pin_lock
    import time as _time

    _saved_count = SOURCE_SETUP._prefill_pin_register_count
    with lock:
        _saved_pins = dict(pinned)
        pinned.clear()

    try:
        # Manually insert a "stale" pin with timestamp far in the past.
        with lock:
            pinned[9999] = _time.monotonic() - SOURCE_SETUP._PREFILL_PIN_TTL_S - 10
        # Force the counter to one less than the sweep interval so the
        # next register call triggers the sweep.
        SOURCE_SETUP._prefill_pin_register_count = (
            SOURCE_SETUP._PREFILL_SWEEP_INTERVAL - 1)

        stale = register([5000])
        assert 9999 in stale, f"Expected 9999 in stale, got {stale}"
        with lock:
            assert 9999 not in pinned, "stale pin should be removed"
            assert 5000 in pinned, "new pin should be present"
    finally:
        SOURCE_SETUP._prefill_pin_register_count = _saved_count
        with lock:
            pinned.clear()
            pinned.update(_saved_pins)
