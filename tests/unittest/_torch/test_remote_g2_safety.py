# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]
_CONNECTOR_PACKAGE = "tensorrt_llm._torch.pyexecutor.connectors"
_REMOTE_G2_PATH = _ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "connectors" / "remote_g2.py"
_REMOTE_G2_CONNECTOR_PATH = (
    _ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "connectors" / "remote_g2_connector.py"
)
_REMOTE_G2_TRANSFER_PATH = (
    _ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "connectors" / "remote_g2_transfer.py"
)
_REMOTE_G2_OBSERVABILITY_PATH = (
    _ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "connectors" / "remote_g2_observability.py"
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
    transfer = _load_module(f"{_CONNECTOR_PACKAGE}.remote_g2_transfer", _REMOTE_G2_TRANSFER_PATH)
    connector = _load_module(f"{_CONNECTOR_PACKAGE}.remote_g2_connector", _REMOTE_G2_CONNECTOR_PATH)
    return observability, remote_g2, transfer, connector


OBSERVABILITY, REMOTE_G2, TRANSFER, CONNECTOR = _load_connector_modules()

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
        self.initial_state = (
            TRANSFER.RemoteG2TransferState.SUCCEEDED
            if completed and not fail
            else TRANSFER.RemoteG2TransferState.IN_PROGRESS
        )

    def poll_state(self):
        if self.fail:
            raise RuntimeError("transfer failed")
        return (
            TRANSFER.RemoteG2TransferState.SUCCEEDED
            if self.completed
            else TRANSFER.RemoteG2TransferState.IN_PROGRESS
        )

    def release_transfer(self):
        self.released += 1
        return True


class _FakeTransferAdapter:
    supports_retryable_release = True

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

    for index, (name, hashes, num_tokens, computed, expected, release_reason) in enumerate(cases):
        result = _resolve_result(
            block_hashes=hashes,
            num_tokens=num_tokens,
            lease_id=f"lease-{name}",
        )

        # D-06/D-07/D-08/D-09: local validation asserts decision equivalence.
        assert (
            compute_remote_g2_matched_tokens(
                RemoteKvReusePlan.from_dict(_plan(plan_id=f"plan-{name}")),
                result,
                computed,
                16,
            )
            == expected
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
    result = _resolve_result(block_hashes=(44, 55, 66), num_tokens=48, lease_id="lease-overlap")
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
    result = _resolve_result(block_hashes=(44, 55, 66), num_tokens=48, lease_id="lease-past")
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
    worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=adapter,
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


def test_remote_g2_duplicate_callbacks_do_not_double_transfer_or_release():
    adapter = _FakeTransferAdapter()
    worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=adapter,
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
    def failing_result(record):
        return _FakeTransferResult(record, completed=False, fail=True)

    failure_worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(failing_result),
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
    )
    failure_worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(bindings=(_bound_record(lease_id="lease-failed"),))
    )
    failure_worker.start_load_kv(None)

    assert failure_worker.get_finished([], [1234]) == ([], [])
    assert failure_worker.take_failed_load_request_ids() == {1234}

    timeout_worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(
            lambda record: _FakeTransferResult(record, completed=False)
        ),
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
        transfer_timeout_ms=-1,
    )
    timeout_worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(bindings=(_bound_record(lease_id="lease-timeout"),))
    )
    timeout_worker.start_load_kv(None)

    assert timeout_worker.get_finished([], [1234]) == ([], [])
    assert timeout_worker.take_failed_load_request_ids() == {1234}


def test_remote_g2_legacy_adapter_is_rejected_without_starting_transfer():
    agent = _FakeAgent()
    sink = InMemoryRemoteG2ObservabilitySink()
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
        mark_local_valid=marked_valid.append,
        publish_binding=published.append,
        observability=sink,
    )
    worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(bindings=(_bound_record(lease_id="lease-restart"),))
    )

    worker.start_load_kv(None)
    worker.get_finished([], [1234])

    assert agent.requests == []
    assert marked_valid == []
    assert published == []
    names = _event_names(sink)
    assert "failed" in names
    assert "released" not in names
    assert "transferred" not in names


def test_remote_g2_capable_adapter_metadata_mismatch_is_a_request_failure():
    agent = _FakeAgent()
    adapter = RemoteG2NixlTransferAdapter(
        source_metadata_fetcher=lambda worker_id, generation: _source_metadata(
            worker_id, generation + 1
        ),
        target_descriptor_resolver=_target_descriptors,
        agent_factory=lambda: agent,
        transfer_types=_FAKE_TRANSFER_TYPES,
    )
    # Exercise the pre-handle metadata/start failure path independently of
    # the production legacy-adapter capability rejection.
    adapter.supports_retryable_release = True
    worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=adapter,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
    )
    worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(bindings=(_bound_record(lease_id="lease-mismatch"),))
    )

    worker.start_load_kv(None)

    assert worker.take_failed_load_request_ids() == {1234}
    assert worker.get_finished([], [1234]) == ([], [])
    assert agent.requests == []


def test_remote_g2_handle_release_is_exactly_once_per_attempt():
    record = _bound_record(lease_id="lease-exact-once")
    result = _FakeTransferResult(record)
    worker = CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(lambda record: result),
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(record,)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [1234])
    assert worker.try_abort_request(1234) is True
    assert result.released == 1


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
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: (_ for _ in ()).throw(RuntimeError("publish failed")),
        observability=sink,
    )
    failed_worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(bindings=(_bound_record(lease_id="lease-failed"),))
    )
    failed_worker.start_load_kv(None)
    assert failed_worker.get_finished([], [1234]) == ([], [])

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
