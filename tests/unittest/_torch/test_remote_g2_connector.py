# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    remote_g2_transfer = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_transfer", _REMOTE_G2_TRANSFER_PATH
    )
    remote_g2_connector = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_connector", _REMOTE_G2_CONNECTOR_PATH
    )
    return remote_g2, remote_g2_transfer, remote_g2_connector, observability


REMOTE_G2, REMOTE_G2_TRANSFER, REMOTE_G2_CONNECTOR, OBSERVABILITY = _load_connector_modules()

RemoteG2ConnectorMetadata = REMOTE_G2_CONNECTOR.RemoteG2ConnectorMetadata
RemoteG2Descriptor = REMOTE_G2.RemoteG2Descriptor
RemoteG2ResolveResult = REMOTE_G2.RemoteG2ResolveResult
TargetRemotePlanStore = REMOTE_G2.TargetRemotePlanStore
TargetRemoteG2BindingStore = REMOTE_G2.TargetRemoteG2BindingStore
InMemoryRemoteG2ObservabilitySink = OBSERVABILITY.InMemoryRemoteG2ObservabilitySink


@pytest.fixture(autouse=True)
def _install_identity_slot_lookup():
    """Production wires this up via maybe_start_remote_g2_target_client; the
    unit tests construct TargetRemoteG2BindingStore directly and bypass that
    setup, so without a stub bind_target_blocks fails with
    target_slot_lookup_failed. Install an identity stub (slot_idx == block_id)
    for the test, restore previous value after."""
    saved = REMOTE_G2_CONNECTOR._installed_block_id_to_slot_idx
    current = sys.modules[f"{_CONNECTOR_PACKAGE}.remote_g2_connector"]
    current_saved = getattr(current, "_installed_block_id_to_slot_idx", None)
    REMOTE_G2_CONNECTOR.install_block_id_to_slot_idx(lambda ids: list(ids))
    current._installed_block_id_to_slot_idx = lambda ids: list(ids)
    yield
    REMOTE_G2_CONNECTOR._installed_block_id_to_slot_idx = saved
    current._installed_block_id_to_slot_idx = current_saved


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
        pool_id="host-pool-0",
        byte_offset=block_hash * 4096,
        byte_length=4096,
    )


def _resolve_result(block_hashes=(11, 22, 33), num_tokens=48, lease_id="lease-1"):
    return RemoteG2ResolveResult(
        lease_id=lease_id,
        descriptors=tuple(_descriptor(block_hash) for block_hash in block_hashes),
        num_tokens=num_tokens,
        source_generation=99,
    )


class _FakeTransferResult:
    def __init__(self, record, completed=True, fail=False):
        self.record = record
        self.completed = completed
        self.fail = fail
        self.released = 0
        self.initial_state = (
            REMOTE_G2_CONNECTOR.RemoteG2TransferState.SUCCEEDED
            if completed and not fail
            else REMOTE_G2_CONNECTOR.RemoteG2TransferState.IN_PROGRESS
        )

    def poll_state(self):
        if self.fail:
            raise RuntimeError("transfer failed")
        return (
            REMOTE_G2_CONNECTOR.RemoteG2TransferState.SUCCEEDED
            if self.completed
            else REMOTE_G2_CONNECTOR.RemoteG2TransferState.IN_PROGRESS
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


def _bound_record(lease_id="lease-bound", request_id=1234):
    store = TargetRemoteG2BindingStore(release_lease=lambda lease_id, reason: True)
    record = store.resolve_for_request(
        request_id,
        _plan(),
        16,
        lambda plan: _resolve_result(lease_id=lease_id),
    )
    store.bind_target_blocks(request_id, [100, 101, 102])
    return record


def _event_names(sink):
    return [event.event for event in sink.events]


def test_remote_g2_connector_resolves_before_reporting_tokens():
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(1234, _plan())
    resolve_calls = []
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=plan_store,
        resolve_and_lease=lambda plan: (
            resolve_calls.append(plan.plan_id) or _resolve_result(num_tokens=32)
        ),
        release_lease=lambda lease_id, reason: True,
    )

    tokens, load_kv_async = scheduler.get_num_new_matched_tokens(
        SimpleNamespace(request_id=1234), 0
    )

    assert (tokens, load_kv_async) == (32, True)
    assert resolve_calls == ["plan-1"]


def test_remote_g2_connector_returns_zero_without_plan():
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=TargetRemotePlanStore(clock_ms=lambda: 500),
        resolve_and_lease=lambda plan: _resolve_result(),
        release_lease=lambda lease_id, reason: True,
    )

    assert scheduler.get_num_new_matched_tokens(SimpleNamespace(request_id=1234), 0) == (
        0,
        False,
    )

    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(5678, _plan())
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=plan_store,
        release_lease=lambda lease_id, reason: True,
    )

    assert scheduler.get_num_new_matched_tokens(SimpleNamespace(request_id=5678), 0) == (
        0,
        False,
    )


def test_remote_g2_connector_preserves_explicit_empty_plan_store():
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=plan_store,
        resolve_and_lease=lambda plan: _resolve_result(),
        release_lease=lambda lease_id, reason: True,
    )
    plan_store.put(1234, _plan())

    assert scheduler.get_num_new_matched_tokens(SimpleNamespace(request_id=1234), 0) == (
        48,
        True,
    )


def test_remote_g2_connector_binds_after_allocated_block_ids():
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(1234, _plan())
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=plan_store,
        resolve_and_lease=lambda plan: _resolve_result(),
        release_lease=lambda lease_id, reason: True,
    )
    request = SimpleNamespace(request_id=1234)

    assert scheduler.get_num_new_matched_tokens(request, 16) == (32, True)
    scheduler.update_state_after_alloc(request, [100, 101, 102])
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(new_requests=[SimpleNamespace(request_id=1234)], cached_requests=[])
    )

    assert isinstance(metadata, RemoteG2ConnectorMetadata)
    assert len(metadata.bindings) == 1
    assert [block.target_block_id for block in metadata.bindings[0].bound_blocks] == [
        101,
        102,
    ]


def test_remote_g2_connector_releases_once_on_request_finished():
    released = []
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(1234, _plan())
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=plan_store,
        resolve_and_lease=lambda plan: _resolve_result(lease_id="lease-finished"),
        release_lease=lambda lease_id, reason: released.append((lease_id, reason)) or True,
    )
    request = SimpleNamespace(request_id=1234)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (48, True)
    assert scheduler.request_finished(request, []) is False
    assert scheduler.request_finished(request, []) is False

    assert released == [("lease-finished", "request_finished")]
    assert plan_store.get(1234) is None


def test_remote_g2_scheduler_abort_releases_once_then_forgets_binding_and_plan():
    released = []
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(1234, _plan())
    binding_store = TargetRemoteG2BindingStore(
        release_lease=lambda lease, reason: released.append((lease, reason)) or True
    )
    binding_store.resolve_for_request(
        1234, _plan(), 0, lambda plan: _resolve_result(lease_id="lease-abort")
    )
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None, plan_store=plan_store, binding_store=binding_store
    )

    assert scheduler.try_abort_request(1234, "transfer_failed") is True
    assert scheduler.try_abort_request(1234, "transfer_failed") is True
    assert binding_store.get(1234) is None
    assert plan_store.get(1234) is None
    assert released == [("lease-abort", "transfer_failed")]


def test_remote_g2_scheduler_abort_retries_genuine_cleanup_failure():
    class FailingBindingStore:
        def discard_without_release(self, request_id):
            raise RuntimeError("store unavailable")

    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=TargetRemotePlanStore(clock_ms=lambda: 500),
        binding_store=FailingBindingStore(),
    )
    assert scheduler.try_abort_request(1234, "transfer_failed") is False


def test_remote_g2_scheduler_release_exception_retries_before_forgetting():
    responses = [RuntimeError("timeout"), True]
    released = []

    def release(lease, reason):
        released.append((lease, reason))
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(1234, _plan())
    binding_store = TargetRemoteG2BindingStore(release_lease=release)
    binding_store.resolve_for_request(
        1234, _plan(), 0, lambda plan: _resolve_result(lease_id="lease-retry")
    )
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None, plan_store=plan_store, binding_store=binding_store
    )

    assert scheduler.try_abort_request(1234, "transfer_failed") is False
    assert binding_store.get(1234) is not None
    assert plan_store.get(1234) is not None
    assert scheduler.try_abort_request(1234, "transfer_failed") is True
    assert binding_store.get(1234) is None
    assert plan_store.get(1234) is None
    assert released == [
        ("lease-retry", "transfer_failed"),
        ("lease-retry", "transfer_failed"),
    ]


def test_remote_g2_scheduler_finish_false_release_is_definitive_and_idempotent():
    released = []
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(1234, _plan())
    binding_store = TargetRemoteG2BindingStore(
        release_lease=lambda lease, reason: released.append((lease, reason)) or False
    )
    binding_store.resolve_for_request(
        1234, _plan(), 0, lambda plan: _resolve_result(lease_id="lease-absent")
    )
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None, plan_store=plan_store, binding_store=binding_store
    )

    assert scheduler.finalize_successful_load(1234) is True
    assert scheduler.finalize_successful_load(1234) is True
    assert released == [("lease-absent", "transfer_succeeded")]


def test_remote_g2_worker_refuses_transfer_before_phase5():
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(None)

    worker.bind_connector_meta(RemoteG2ConnectorMetadata())
    worker.start_load_kv(None)

    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(_bound_record(),)))
    worker.start_load_kv(None)
    assert worker.get_finished([], [1234]) == ([], [])
    assert worker.take_failed_load_request_ids() == {1234}


def test_remote_g2_worker_starts_transfer_for_bound_metadata():
    adapter = _FakeTransferAdapter()
    record = _bound_record()
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=adapter,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
    )

    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(record,)))
    worker.start_load_kv(None)
    worker.start_load_kv(None)

    assert adapter.started == [record]


def test_remote_g2_worker_reports_finished_only_after_transfer_success():
    result = None

    def make_result(record):
        nonlocal result
        result = _FakeTransferResult(record, completed=False)
        return result

    adapter = _FakeTransferAdapter(make_result)
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=adapter,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
    )
    record = _bound_record()
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(record,)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [])
    assert result.released == 0
    result.completed = True
    assert worker.get_finished([], [1234]) == ([], [1234])
    assert result.released == 1


def test_remote_g2_worker_failure_releases_once_and_publishes_nothing():
    published = []
    result = None

    def make_result(record):
        nonlocal result
        result = _FakeTransferResult(record, completed=False, fail=True)
        return result

    adapter = _FakeTransferAdapter(make_result)
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=adapter,
        mark_local_valid=lambda record: None,
        publish_binding=published.append,
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(_bound_record(),)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [])
    assert worker.take_failed_load_request_ids() == {1234}
    assert worker.get_finished([], [1234]) == ([], [])
    assert result.released == 1
    assert published == []


def test_remote_g2_worker_timeout_releases_handle_once():
    result = None

    def make_result(record):
        nonlocal result
        result = _FakeTransferResult(record, completed=False)
        return result

    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(make_result),
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
        transfer_timeout_ms=-1,
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(_bound_record(),)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [])
    assert worker.take_failed_load_request_ids() == {1234}
    assert worker.get_finished([], [1234]) == ([], [])
    assert result.released == 1


def test_remote_g2_worker_publishes_after_local_validity():
    order = []
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(),
        mark_local_valid=lambda record: order.append("valid"),
        publish_binding=lambda record: order.append("publish"),
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(_bound_record(),)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [1234])
    assert order == ["valid", "publish"]


def test_remote_g2_worker_emits_transferred_without_releasing_shared_lease():
    sink = InMemoryRemoteG2ObservabilitySink()
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(),
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
        observability=sink,
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(_bound_record(),)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [1234])
    names = _event_names(sink)
    assert "transferred" in names
    assert "released" not in names


def test_remote_g2_worker_emits_fallback_before_validity_or_publication():
    sink = InMemoryRemoteG2ObservabilitySink()
    published = []
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(),
        mark_local_valid=None,
        publish_binding=published.append,
        observability=sink,
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(_bound_record(),)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [])
    failed = [event for event in sink.events if event.event == "failed"]
    assert failed
    assert failed[0].outcome == "request_failed"
    assert published == []


def test_remote_g2_worker_emits_failed_after_validity_is_marked():
    sink = InMemoryRemoteG2ObservabilitySink()
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(),
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: (_ for _ in ()).throw(RuntimeError("publish failed")),
        observability=sink,
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(_bound_record(),)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [])
    failed = [event for event in sink.events if event.event == "failed"]
    assert failed
    assert failed[0].outcome == "request_failed"


def test_remote_g2_worker_observability_never_logs_raw_descriptors():
    sink = InMemoryRemoteG2ObservabilitySink()
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_FakeTransferAdapter(),
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
        observability=sink,
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata(bindings=(_bound_record(),)))
    worker.start_load_kv(None)
    worker.get_finished([], [1234])

    forbidden = {"ptr", "nixl_memory_desc", "descriptor", "metadata", "transfer_tuple"}
    for event in sink.events:
        detail_text = " ".join(event.details)
        assert not any(value in detail_text for value in forbidden)


class _ScriptedResult:
    def __init__(self, record, states, release_transfer=(True,), start_error=None):
        self.record = record
        self.initial_state = states[0]
        self._states = list(states[1:])
        self._release_transfer_outcomes = list(release_transfer)
        self.start_error = start_error
        self.release_transfer_calls = 0

    def poll_state(self):
        state = self._states.pop(0) if self._states else self.initial_state
        if isinstance(state, BaseException):
            raise state
        return state

    def release_transfer(self):
        self.release_transfer_calls += 1
        result = self._release_transfer_outcomes.pop(0) if self._release_transfer_outcomes else True
        if isinstance(result, BaseException):
            raise result
        return result


class _ScriptedAdapter:
    supports_retryable_release = True

    def __init__(self, start):
        self.start = start

    def start_transfer(self, record):
        return self.start(record)


def test_remote_g2_start_exception_releases_handle_and_continues_bindings():
    failed = _bound_record("lease-failed", 1234)
    succeeded = _bound_record("lease-ok", 5678)
    failed_result = _ScriptedResult(failed, [REMOTE_G2_TRANSFER.RemoteG2TransferState.IN_PROGRESS])

    class StartError(RuntimeError):
        transfer_result = failed_result

    def start(record):
        if record.request_id == 1234:
            raise StartError("post failed")
        return _ScriptedResult(record, [REMOTE_G2_TRANSFER.RemoteG2TransferState.SUCCEEDED])

    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_ScriptedAdapter(start),
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata((failed, succeeded)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234, 5678]) == ([], [5678])
    assert worker.take_failed_load_request_ids() == {1234}
    assert failed_result.release_transfer_calls == 1


def test_remote_g2_initial_failure_and_failed_notifications_drain_once():
    record = _bound_record()
    result = _ScriptedResult(record, [REMOTE_G2_TRANSFER.RemoteG2TransferState.FAILED])
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_ScriptedAdapter(lambda record: result),
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata((record,)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [])
    assert worker.take_failed_load_request_ids() == {1234}
    assert worker.take_failed_load_request_ids() == set()


def test_remote_g2_abort_releases_handle_once():
    record = _bound_record()
    result = _ScriptedResult(
        record,
        [REMOTE_G2_TRANSFER.RemoteG2TransferState.IN_PROGRESS],
    )
    sink = InMemoryRemoteG2ObservabilitySink()
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_ScriptedAdapter(lambda record: result),
        observability=sink,
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata((record,)))
    worker.start_load_kv(None)

    assert worker.try_abort_request(9999) is True
    assert worker.try_abort_request(1234) is True
    assert worker.try_abort_request(1234) is True
    assert result.release_transfer_calls == 1
    aborted = [event for event in sink.events if event.outcome == "request_aborted"]
    assert len(aborted) == 1
    assert aborted[0].reason == "cancelled"
    assert aborted[0].outcome == "request_aborted"


def test_remote_g2_success_cleanup_order_and_hooks_run_once():
    record = _bound_record()
    order = []

    class Result(_ScriptedResult):
        def release_transfer(self):
            order.append("release_transfer")
            return super().release_transfer()

    result = Result(record, [REMOTE_G2_TRANSFER.RemoteG2TransferState.SUCCEEDED])
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_ScriptedAdapter(lambda record: result),
        mark_local_valid=lambda record: order.append("valid"),
        publish_binding=lambda record: order.append("publish"),
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata((record,)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [1234])
    assert worker.get_finished([], []) == ([], [])
    assert order == ["release_transfer", "valid", "publish"]


@pytest.mark.parametrize(
    "release_transfer_outcome,match",
    [
        (False, "did not confirm release"),
        (RuntimeError("release failed"), "release failed"),
    ],
)
def test_remote_g2_release_failure_is_fatal(release_transfer_outcome, match):
    record = _bound_record()
    result = _ScriptedResult(
        record,
        [REMOTE_G2_TRANSFER.RemoteG2TransferState.FAILED],
        release_transfer=(release_transfer_outcome,),
    )
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_ScriptedAdapter(lambda record: result),
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata((record,)))
    worker.start_load_kv(None)

    with pytest.raises(RuntimeError, match=match):
        worker.get_finished([], [1234])
    assert result.release_transfer_calls == 1


@pytest.mark.parametrize("failing_hook", ["mark", "publish"])
def test_remote_g2_mark_or_publish_failure_becomes_request_failure(failing_hook):
    record = _bound_record()

    def hook(name):
        if name == failing_hook:
            raise RuntimeError(name)

    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=_ScriptedAdapter(
            lambda record: _ScriptedResult(
                record, [REMOTE_G2_TRANSFER.RemoteG2TransferState.SUCCEEDED]
            )
        ),
        mark_local_valid=lambda record: hook("mark"),
        publish_binding=lambda record: hook("publish"),
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata((record,)))
    worker.start_load_kv(None)

    assert worker.get_finished([], [1234]) == ([], [])
    assert worker.take_failed_load_request_ids() == {1234}


def test_remote_g2_rejects_incapable_adapter():
    class Incapable:
        def start_transfer(self, record):
            raise AssertionError("must not start")

    record = _bound_record("lease-incapable")
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=Incapable(),
    )
    worker.bind_connector_meta(RemoteG2ConnectorMetadata((record,)))
    worker.start_load_kv(None)
    worker.get_finished([], [1234])
    assert worker.take_failed_load_request_ids() == {1234}


def test_remote_g2_reused_request_id_starts_new_binding_and_prunes_terminals():
    adapter = _FakeTransferAdapter()
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(
        None,
        transfer_adapter=adapter,
        mark_local_valid=lambda record: None,
        publish_binding=lambda record: None,
    )
    first = _bound_record("lease-first", 1234)
    worker.bind_connector_meta(RemoteG2ConnectorMetadata((first,)))
    worker.start_load_kv(None)
    assert worker.get_finished([], [1234]) == ([], [1234])
    worker.start_load_kv(None)
    assert adapter.started == [first]

    second = _bound_record("lease-second", 1234)
    second.plan = REMOTE_G2.RemoteKvReusePlan.from_dict(_plan(plan_id="plan-second"))
    worker.bind_connector_meta(RemoteG2ConnectorMetadata((second,)))
    worker.start_load_kv(None)
    assert worker.get_finished([], [1234]) == ([], [1234])
    assert adapter.started == [first, second]
    assert len(worker._terminal_binding_fingerprints) == 1

    worker.bind_connector_meta(RemoteG2ConnectorMetadata())
    worker.start_load_kv(None)
    assert worker._terminal_binding_fingerprints == set()


# Remote-G2 depends on retryable KV admission in KVCM V1, which is not validated
# with the overlap scheduler because local offload/onboard can interleave with forward.
def test_remote_g2_requires_overlap_scheduler_disabled():
    scheduler_cls = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler
    worker_cls = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker

    assert scheduler_cls.requires_disable_overlap_scheduler
    assert worker_cls.requires_disable_overlap_scheduler


# Startup check: kv_cache_config.enable_partial_reuse must be False when the
# remote-G2 connector is constructed. Partial reuse silently stops remote-G2
# fetch from triggering, so misconfiguration must fail fast.


def _llm_args_with_partial_reuse(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(kv_cache_config=SimpleNamespace(enable_partial_reuse=enabled))


def test_scheduler_init_fails_when_partial_reuse_enabled():
    with pytest.raises(RuntimeError, match="enable_partial_reuse"):
        REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(_llm_args_with_partial_reuse(True))


def test_worker_init_fails_when_partial_reuse_enabled():
    with pytest.raises(RuntimeError, match="enable_partial_reuse"):
        REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(_llm_args_with_partial_reuse(True))


def test_scheduler_init_succeeds_when_partial_reuse_disabled():
    REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(_llm_args_with_partial_reuse(False))


def test_worker_init_succeeds_when_partial_reuse_disabled():
    REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(_llm_args_with_partial_reuse(False))
