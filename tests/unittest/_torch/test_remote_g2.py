# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
import types
from pathlib import Path

_CONNECTOR_PACKAGE = "tensorrt_llm._torch.pyexecutor.connectors"
_REMOTE_G2_PATH = (
    Path(__file__).resolve().parents[3]
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
    / "remote_g2.py"
)
_REMOTE_G2_OBSERVABILITY_PATH = (
    Path(__file__).resolve().parents[3]
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
    / "remote_g2_observability.py"
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


def _load_remote_g2_modules():
    _install_package("tensorrt_llm")
    _install_package("tensorrt_llm._torch")
    _install_package("tensorrt_llm._torch.pyexecutor")
    _install_package(_CONNECTOR_PACKAGE)
    observability = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_observability",
        _REMOTE_G2_OBSERVABILITY_PATH,
    )
    remote_g2 = _load_module(f"{_CONNECTOR_PACKAGE}.remote_g2", _REMOTE_G2_PATH)
    return observability, remote_g2


OBSERVABILITY, _REMOTE_G2 = _load_remote_g2_modules()


# bind_target_blocks does a lazy `from .remote_g2_connector import
# _installed_block_id_to_slot_idx`. Production installs the callable via
# maybe_start_remote_g2_target_client; these tests bypass that setup, so we
# preload a stub remote_g2_connector module in sys.modules with an identity
# callable so the lazy import resolves and binding can proceed.
_g2_connector_stub = types.ModuleType(
    f"{_CONNECTOR_PACKAGE}.remote_g2_connector"
)
_g2_connector_stub._installed_block_id_to_slot_idx = lambda ids: list(ids)
sys.modules[f"{_CONNECTOR_PACKAGE}.remote_g2_connector"] = _g2_connector_stub


import pytest


@pytest.fixture(autouse=True)
def _identity_slot_lookup_for_lazy_import():
    """The lazy import in bind_target_blocks resolves to whatever module
    happens to be at sys.modules[...remote_g2_connector] when the test runs.
    Other test files (e.g. test_remote_g2_connector.py) load the real module
    and may leave `_installed_block_id_to_slot_idx` at None on teardown, so
    we re-stamp it before each test here and restore on exit."""
    module = sys.modules[f"{_CONNECTOR_PACKAGE}.remote_g2_connector"]
    saved = getattr(module, "_installed_block_id_to_slot_idx", None)
    module._installed_block_id_to_slot_idx = lambda ids: list(ids)
    yield
    module._installed_block_id_to_slot_idx = saved

REMOTE_G2_REUSE_ENABLED_ENV = _REMOTE_G2.REMOTE_G2_REUSE_ENABLED_ENV
REMOTE_KV_REUSE_PLAN_VERSION = _REMOTE_G2.REMOTE_KV_REUSE_PLAN_VERSION
RemoteKvReusePlan = _REMOTE_G2.RemoteKvReusePlan
RemoteG2BindingState = _REMOTE_G2.RemoteG2BindingState
RemoteG2Descriptor = _REMOTE_G2.RemoteG2Descriptor
RemoteG2ResolveResult = _REMOTE_G2.RemoteG2ResolveResult
SourceG2DescriptorRecord = _REMOTE_G2.SourceG2DescriptorRecord
SourceG2DescriptorRegistry = _REMOTE_G2.SourceG2DescriptorRegistry
TargetRemoteG2BindingStore = _REMOTE_G2.TargetRemoteG2BindingStore
TargetRemotePlanStore = _REMOTE_G2.TargetRemotePlanStore
compute_remote_g2_matched_tokens = _REMOTE_G2.compute_remote_g2_matched_tokens
InMemoryRemoteG2ObservabilitySink = OBSERVABILITY.InMemoryRemoteG2ObservabilitySink
RemoteG2LifecycleEvent = OBSERVABILITY.RemoteG2LifecycleEvent
sanitize_remote_g2_event_details = OBSERVABILITY.sanitize_remote_g2_event_details


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
        "plan_version": REMOTE_KV_REUSE_PLAN_VERSION,
    }
    plan.update(overrides)
    return plan


def _record(block_hash, generation=1):
    return SourceG2DescriptorRecord(
        block_hash=block_hash,
        source_worker_id=7,
        source_dp_rank=0,
        tier="G2",
        descriptor_generation=generation,
        pool_id="host-pool-0",
        byte_offset=block_hash * 4096,
        byte_length=4096,
    )


def _descriptor(block_hash, generation=1):
    return RemoteG2Descriptor(
        block_hash=block_hash,
        descriptor_generation=generation,
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


def test_target_store_keys_plan_by_trtllm_request_id_and_discards():
    store = TargetRemotePlanStore(clock_ms=lambda: 500)

    stored = store.put(1234, _plan())

    assert stored is not None
    assert store.get(1234).plan_id == "plan-1"
    assert store.get("1234").source_worker_id == 7
    store.discard(1234)
    assert store.get(1234) is None


def test_target_store_rejects_expired_or_non_g2_plan():
    store = TargetRemotePlanStore(clock_ms=lambda: 5_000)

    assert store.put(1, _plan(expires_at_ms=5_000)) is None
    assert store.put(2, _plan(source_tier="device")) is None
    assert store.put(3, {"plan_id": "missing-required-fields"}) is None
    assert len(store) == 0


def test_target_store_respects_remote_g2_kill_switch():
    old_value = os.environ.get(REMOTE_G2_REUSE_ENABLED_ENV)
    os.environ[REMOTE_G2_REUSE_ENABLED_ENV] = "false"
    try:
        store = TargetRemotePlanStore(clock_ms=lambda: 500)
        assert store.put(1234, _plan()) is None
        assert len(store) == 0
    finally:
        if old_value is None:
            os.environ.pop(REMOTE_G2_REUSE_ENABLED_ENV, None)
        else:
            os.environ[REMOTE_G2_REUSE_ENABLED_ENV] = old_value


def test_remote_g2_plan_store_emits_planned_event():
    sink = InMemoryRemoteG2ObservabilitySink()
    store = TargetRemotePlanStore(clock_ms=lambda: 500, observability=sink)

    stored = store.put(1234, _plan())

    assert stored is not None
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.event == "planned"
    assert event.outcome == "accepted"
    assert event.reason == "ok"
    assert event.request_id == 1234
    assert event.plan_id == "plan-1"
    assert event.source_worker_id == 7


def test_remote_g2_observability_counts_are_low_cardinality():
    sink = InMemoryRemoteG2ObservabilitySink()

    # D-14/D-15/D-16: counters use only event, reason, tier, and outcome.
    sink.emit(
        RemoteG2LifecycleEvent(
            event="planned",
            reason="ok",
            tier="host_pinned",
            outcome="accepted",
            request_id=1234,
            plan_id="plan-1",
            lease_id="lease-1",
            source_worker_id=7,
            source_generation=99,
            token_count=32,
            byte_count=8192,
        )
    )

    assert sink.counts == {("planned", "ok", "host_pinned", "accepted"): 1}
    assert sink.token_histogram == [32]
    assert sink.byte_histogram == [8192]


def test_remote_g2_observability_sanitizes_raw_transfer_details():
    details = {
        "ptr": 123,
        "nixl_memory_desc": {"ptr": 456},
        "descriptor": "raw",
        "transfer_tuple": (1, 2, 3),
        "request_id": 1234,
        "safe_reason": "ok",
    }

    # D-17: raw memory, descriptor, and pointer-like details are not observable.
    sanitized = sanitize_remote_g2_event_details(details)

    assert "ptr" not in sanitized
    assert "nixl_memory_desc" not in sanitized
    assert "descriptor" not in sanitized
    assert "transfer_tuple" not in sanitized
    assert sanitized == {"request_id": 1234, "safe_reason": "ok"}


def test_source_registry_resolve_and_lease_returns_live_contiguous_prefix():
    released = []
    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        source_generation=99,
        clock_ms=lambda: 1_000,
        acquire_pin=lambda record, lease_id: f"{lease_id}:{record.block_hash}",
        release_pin=released.append,
        require_trtllm_pin=True,
    )
    first = _record(11, generation=4)
    second = _record(22, generation=5)
    registry.upsert_descriptor(first)
    registry.upsert_descriptor(second)

    result = registry.resolve_and_lease(_plan())

    assert result.reason == "ok"
    assert result.num_tokens == 32
    assert result.source_generation == 99
    assert [d.block_hash for d in result.descriptors] == [11, 22]
    assert [d.descriptor_generation for d in result.descriptors] == [4, 5]
    assert [status.status for status in result.per_block_status] == [
        "live",
        "live",
        "missing",
    ]
    assert first.lease_count == 1
    assert second.lease_count == 1

    assert registry.release_lease(result.lease_id, "success") is True
    assert registry.release_lease(result.lease_id, "duplicate") is False
    assert first.lease_count == 0
    assert second.lease_count == 0
    assert len(released) == 2


def test_source_registry_fails_closed_for_wrong_source_or_tier():
    registry = SourceG2DescriptorRegistry(source_worker_id=7, source_dp_rank=0)
    registry.upsert_descriptor(_record(11))

    assert (
        registry.resolve_and_lease(_plan(source_worker_id=8)).reason
        == "wrong_source_worker"
    )
    assert (
        registry.resolve_and_lease(_plan(source_dp_rank=1)).reason == "wrong_source_rank"
    )
    assert registry.resolve_and_lease(_plan(source_tier="device")).reason == "wrong_source_tier"


def test_source_registry_reports_missing_pin_hook_when_required():
    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        clock_ms=lambda: 1_000,
        require_trtllm_pin=True,
    )
    registry.upsert_descriptor(_record(11))

    result = registry.resolve_and_lease(_plan(planned_prefix_blocks=1))

    assert result.lease_id is None
    assert result.reason == "missing_trtllm_pin_hook"


def test_source_registry_reports_invalid_plan_without_leasing():
    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        clock_ms=lambda: 1_000,
    )
    registry.upsert_descriptor(_record(11))

    result = registry.resolve_and_lease({"plan_id": "missing-required-fields"})

    assert result.lease_id is None
    assert result.reason == "invalid_plan"


def test_source_registry_reports_first_missing_block_status():
    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        clock_ms=lambda: 1_000,
    )

    result = registry.resolve_and_lease(_plan(planned_prefix_blocks=1))

    assert result.reason == "no_live_remote_g2_prefix"
    assert result.per_block_status[0].block_hash == 11
    assert result.per_block_status[0].status == "missing"


def test_source_registry_falls_back_to_find_and_pin_blocks_when_kv_provided():
    # When the in-memory _records cache is empty and a kv handle is
    # configured, resolve_and_lease should discover blocks via
    # kv.find_and_pin_blocks_by_hash and synthesize ephemeral records whose
    # byte_offset and nixl_memory_desc.ptr are derived from the returned slot
    # and the pool's base pointer. The C++ call atomically pins each block
    # under the lookup mutex, so the resolve-time acquire_pin sees the
    # _pinned_by_lookup flag and skips double-pinning.

    BLOCK_SIZE_BYTES = 4096
    POOL_BASE_PTR = 0x1000_0000
    WINDOW_SIZE = 4096
    locations = {
        11: (42, 5),
        22: (88, 9),
    }
    lookups: list[tuple[list[int], int, str, bool]] = []
    pins: list[tuple[int, int]] = []
    unpinned: list[int] = []

    class FakeKv:
        def find_and_pin_blocks_by_hash(
            self, block_hashes, window_size, tier="host_pinned", stop_on_miss=True
        ):
            lookups.append(
                ([int(block_hash) for block_hash in block_hashes], int(window_size), tier, stop_on_miss)
            )
            results = []
            for block_hash in block_hashes:
                loc = locations.get(int(block_hash))
                if loc is None:
                    results.append(
                        {
                            "block_hash": int(block_hash),
                            "pinned": False,
                            "found_tier": None,
                            "block_id": None,
                            "slot_idx": None,
                        }
                    )
                    break
                block_id, slot_idx = loc
                results.append(
                    {
                        "block_hash": int(block_hash),
                        "pinned": True,
                        "found_tier": "host_pinned",
                        "block_id": block_id,
                        "slot_idx": slot_idx,
                    }
                )
            return results

    def acquire_pin(record, lease_id):
        pins.append((record.block_id, record.byte_offset))
        return record.block_id

    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        source_generation=99,
        clock_ms=lambda: 1_000,
        acquire_pin=acquire_pin,
        release_pin=unpinned.append,
        require_trtllm_pin=True,
        kv=FakeKv(),
        window_size=WINDOW_SIZE,
        pool_id="host-pool-0",
        pool_base_ptr=POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
        tier="host_pinned",
    )

    result = registry.resolve_and_lease(_plan())

    assert result.reason == "ok"
    assert result.lease_id is not None
    assert result.num_tokens == 2 * 16
    assert [d.block_hash for d in result.descriptors] == [11, 22]
    assert [d.byte_offset for d in result.descriptors] == [
        5 * BLOCK_SIZE_BYTES,
        9 * BLOCK_SIZE_BYTES,
    ]
    assert [d.metadata["nixl_memory_desc"]["ptr"] for d in result.descriptors] == [
        POOL_BASE_PTR + 5 * BLOCK_SIZE_BYTES,
        POOL_BASE_PTR + 9 * BLOCK_SIZE_BYTES,
    ]
    assert [status.status for status in result.per_block_status] == [
        "live",
        "live",
        "missing",
    ]
    assert lookups == [([11, 22, 33], WINDOW_SIZE, "host_pinned", True)]
    assert pins == [(42, 5 * BLOCK_SIZE_BYTES), (88, 9 * BLOCK_SIZE_BYTES)]

    assert registry.release_lease(result.lease_id, "success") is True
    assert unpinned == [42, 88]


def test_source_registry_kv_fallback_is_disabled_when_kv_is_none():
    # Without a kv handle, resolve_and_lease must not synthesize records.
    # An empty _records cache should return "missing" on the first hash,
    # preserving the alpha-only behavior for legacy callers.
    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        clock_ms=lambda: 1_000,
    )

    result = registry.resolve_and_lease(_plan(planned_prefix_blocks=1))

    assert result.reason == "no_live_remote_g2_prefix"
    assert result.per_block_status[0].status == "missing"


def test_source_registry_uses_kv_block_hashes_for_lookup_when_present():
    # When a plan carries `kv_block_hashes` parallel to `block_hashes`,
    # the source side must use the kv-side hash to look up records but
    # report the router-side identity (block_hashes / tokens_hash) back
    # in the descriptors and per-block status.

    BLOCK_SIZE_BYTES = 4096
    POOL_BASE_PTR = 0x1000_0000
    WINDOW_SIZE = 4096
    # Router-side identity: block_hashes (tokens hashes, e.g. 11/22/33).
    # KV-side identity: kv_block_hashes (splitmix, distinct values).
    tokens_hashes = [11, 22, 33]
    kv_hashes = [0xAAAA_AAAA_AAAA_AAA1, 0xBBBB_BBBB_BBBB_BBB2, 0xCCCC_CCCC_CCCC_CCC3]
    locations = {
        kv_hashes[0]: (42, 5),
        kv_hashes[1]: (88, 9),
        # kv_hashes[2] intentionally absent - last block is "missing"
    }
    lookups: list[list[int]] = []

    class FakeKv:
        def find_and_pin_blocks_by_hash(
            self, block_hashes, window_size, tier="host_pinned", stop_on_miss=True
        ):
            lookups.append([int(block_hash) for block_hash in block_hashes])
            results = []
            for block_hash in block_hashes:
                loc = locations.get(int(block_hash))
                if loc is None:
                    results.append(
                        {
                            "block_hash": int(block_hash),
                            "pinned": False,
                            "found_tier": None,
                            "block_id": None,
                            "slot_idx": None,
                        }
                    )
                    break
                block_id, slot_idx = loc
                results.append(
                    {
                        "block_hash": int(block_hash),
                        "pinned": True,
                        "found_tier": "host_pinned",
                        "block_id": block_id,
                        "slot_idx": slot_idx,
                    }
                )
            return results

    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        source_generation=99,
        clock_ms=lambda: 1_000,
        acquire_pin=lambda record, lease_id: record.block_id,
        release_pin=lambda _pin: None,
        require_trtllm_pin=True,
        kv=FakeKv(),
        window_size=WINDOW_SIZE,
        pool_id="host-pool-0",
        pool_base_ptr=POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
        tier="host_pinned",
    )

    plan = _plan()
    plan["block_hashes"] = tokens_hashes
    plan["kv_block_hashes"] = kv_hashes
    result = registry.resolve_and_lease(plan)

    assert result.reason == "ok"
    assert result.num_tokens == 2 * 16
    # The descriptors must carry tokens hashes so the router can correlate.
    assert [d.block_hash for d in result.descriptors] == [11, 22]
    # The kv-side hashes are the ones used against kv.find_and_pin_blocks_by_hash.
    assert lookups == [kv_hashes]
    # Per-block status reports the tokens (router-side) hash.
    statuses = [(s.block_hash, s.status) for s in result.per_block_status]
    assert statuses == [(11, "live"), (22, "live"), (33, "missing")]
    # Slot/byte_offset derived from the FakeKv's slot returns, which are
    # bound to kv-side hashes, not tokens-side.
    assert [d.byte_offset for d in result.descriptors] == [
        5 * BLOCK_SIZE_BYTES,
        9 * BLOCK_SIZE_BYTES,
    ]


def test_source_registry_reports_promoted_primary_from_tier_aware_lookup():
    BLOCK_SIZE_BYTES = 4096
    WINDOW_SIZE = 4096

    class FakeKv:
        def find_and_pin_blocks_by_hash(
            self, block_hashes, window_size, tier="host_pinned", stop_on_miss=True
        ):
            return [
                {
                    "block_hash": int(block_hashes[0]),
                    "pinned": False,
                    "found_tier": "primary",
                    "block_id": None,
                    "slot_idx": None,
                }
            ]

    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        clock_ms=lambda: 1_000,
        kv=FakeKv(),
        window_size=WINDOW_SIZE,
        pool_id="host-pool-0",
        pool_base_ptr=0,
        block_size_bytes=BLOCK_SIZE_BYTES,
        tier="host_pinned",
    )

    result = registry.resolve_and_lease(_plan(planned_prefix_blocks=1))

    assert result.reason == "no_live_remote_g2_prefix"
    assert result.lease_id is None
    assert [(s.block_hash, s.status) for s in result.per_block_status] == [
        (11, "promoted_primary")
    ]


def test_remote_plan_parser_truncates_prefix_to_hash_count():
    parsed = RemoteKvReusePlan.from_dict(_plan(planned_prefix_blocks=10))

    assert parsed.planned_prefix_blocks == 3
    assert parsed.planned_hashes == (11, 22, 33)


def test_remote_g2_matched_tokens_use_source_resolved_block_aligned_prefix():
    result = _resolve_result(num_tokens=32)
    plan = RemoteKvReusePlan.from_dict(_plan())

    assert compute_remote_g2_matched_tokens(plan, result, 0, 16) == 32
    assert compute_remote_g2_matched_tokens(plan, result, 16, 16) == 16
    assert compute_remote_g2_matched_tokens(plan, result, 32, 16) == 0
    assert compute_remote_g2_matched_tokens(plan, _resolve_result(lease_id=None), 0, 16) == 0


def test_remote_g2_matched_tokens_release_lease_for_unaligned_computed_boundary():
    released = []
    store = TargetRemoteG2BindingStore(
        release_lease=lambda lease_id, reason: released.append((lease_id, reason)) or True
    )

    record = store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        7,
        lambda plan: _resolve_result(lease_id="lease-unaligned"),
    )

    assert record is None
    assert released == [("lease-unaligned", "unaligned_num_computed_tokens")]
    assert len(store) == 0


def test_remote_g2_binding_uses_exact_allocated_target_block_slice():
    store = TargetRemoteG2BindingStore(release_lease=lambda lease_id, reason: True)
    record = store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        16,
        lambda plan: _resolve_result(),
    )

    bound = store.bind_target_blocks(1234, [100, 101, 102])

    assert bound is record
    assert bound.state is RemoteG2BindingState.BOUND
    assert [block.target_block_id for block in bound.bound_blocks] == [101, 102]
    assert [block.source_descriptor.block_hash for block in bound.bound_blocks] == [22, 33]
    assert [block.source_block_index for block in bound.bound_blocks] == [1, 2]
    assert [block.target_block_index for block in bound.bound_blocks] == [1, 2]


def test_remote_g2_binding_store_emits_resolved_truncated_and_fallback_events():
    sink = InMemoryRemoteG2ObservabilitySink()
    released = []
    store = TargetRemoteG2BindingStore(
        release_lease=lambda lease_id, reason: released.append((lease_id, reason)) or True,
        observability=sink,
    )

    record = store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        0,
        lambda plan: _resolve_result(block_hashes=(11, 22), num_tokens=32),
    )
    fallback = store.resolve_for_request(
        5678,
        RemoteKvReusePlan.from_dict(_plan(plan_id="plan-fallback")),
        7,
        lambda plan: _resolve_result(lease_id="lease-fallback"),
    )

    assert record is not None
    assert fallback is None
    names = [event.event for event in sink.events]
    assert "resolved" in names
    assert "truncated" in names
    assert "fallback" in names
    assert ("lease-fallback", "unaligned_num_computed_tokens") in released


def test_remote_g2_binding_store_release_events_are_exact_once():
    sink = InMemoryRemoteG2ObservabilitySink()
    released = []
    store = TargetRemoteG2BindingStore(
        release_lease=lambda lease_id, reason: released.append((lease_id, reason)) or True,
        observability=sink,
    )
    record = store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        0,
        lambda plan: _resolve_result(lease_id="lease-release-events"),
    )

    assert store.release(1234, "transfer_failed") is True
    assert store.release(1234, "transfer_failed") is False
    assert record.release_attempted is True
    assert released == [("lease-release-events", "transfer_failed")]
    release_events = [event for event in sink.events if event.event == "released"]
    assert release_events[0].outcome == "completed"
    assert release_events[-1].outcome == "already_released"


def test_remote_g2_binding_failure_releases_lease_once():
    released = []
    store = TargetRemoteG2BindingStore(
        release_lease=lambda lease_id, reason: released.append((lease_id, reason)) or True
    )
    record = store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        0,
        lambda plan: _resolve_result(lease_id="lease-bind-failed"),
    )

    first = store.bind_target_blocks(1234, [100, 101])
    second = store.bind_target_blocks(1234, [100, 101])

    assert first is record
    assert second is record
    assert record.state is RemoteG2BindingState.BIND_FAILED
    assert record.release_attempted is True
    assert released == [("lease-bind-failed", "target_binding_failed")]


def test_remote_g2_duplicate_and_reordered_callbacks_are_idempotent():
    released = []
    resolve_calls = []
    store = TargetRemoteG2BindingStore(
        release_lease=lambda lease_id, reason: released.append((lease_id, reason)) or True
    )

    assert store.bind_target_blocks(1234, [100, 101, 102]) is None
    first = store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        0,
        lambda plan: resolve_calls.append(plan.plan_id)
        or _resolve_result(lease_id="lease-idempotent"),
    )
    duplicate = store.resolve_for_request(
        1234,
        RemoteKvReusePlan.from_dict(_plan()),
        0,
        lambda plan: resolve_calls.append(plan.plan_id)
        or _resolve_result(lease_id="lease-duplicate"),
    )

    assert duplicate is first
    assert resolve_calls == ["plan-1"]
    assert store.bind_target_blocks(1234, [100, 101, 102]) is first
    assert store.bind_target_blocks(1234, [200, 201, 202]) is first
    assert [block.target_block_id for block in first.bound_blocks] == [100, 101, 102]
    assert store.release(1234, "transfer_failed") is True
    assert store.release(1234, "transfer_failed") is False
    assert first.state is RemoteG2BindingState.TRANSFER_FAILED
    assert released == [("lease-idempotent", "transfer_failed")]


def test_resolve_hashes_force_offloads_primary_only_blocks():
    """Simulate the resolve_hashes IPC handler's force-offload retry path.

    When find_and_pin_blocks_by_hash reports blocks in primary but not
    secondary (CacheMiss with found_tier="primary"), the handler should
    call force_offload_and_pin_blocks_by_hash to move them to secondary
    and fill in descriptors. This prevents asymmetric secondary-tier
    availability across TP ranks during NIXL RDMA transfers.
    """
    from tensorrt_llm._torch.pyexecutor.connectors.remote_g2 import (
        CacheMiss,
        PinnedCacheBlock,
    )

    BLOCK_SIZE_BYTES = 4096
    POOL_BASE_PTR = 0x2000_0000
    WINDOW_SIZE = 4096

    # Simulate: blocks 11 and 22 are in primary only, block 33 is missing.
    find_calls = []
    force_calls = []

    class FakeKv:
        def find_and_pin_blocks_by_hash(
            self, block_hashes, window_size, tier="host_pinned", stop_on_miss=True
        ):
            find_calls.append(list(block_hashes))
            results = []
            for bh in block_hashes:
                results.append({
                    "block_hash": int(bh),
                    "pinned": False,
                    "found_tier": "primary",
                    "block_id": None,
                    "slot_idx": None,
                })
                if stop_on_miss:
                    # In practice, stop_on_miss=True stops at first miss.
                    # But CacheMiss(found_tier="primary") is not a full miss,
                    # so the handler iterates all results.
                    pass
            return results

        def force_offload_and_pin_blocks_by_hash(self, block_hashes, window_size):
            force_calls.append(list(block_hashes))
            # Simulate successful force-offload for all requested hashes.
            results = []
            for i, bh in enumerate(block_hashes):
                results.append({
                    "block_hash": int(bh),
                    "pinned": True,
                    "found_tier": "host_pinned",
                    "block_id": 100 + i,
                    "slot_idx": 50 + i,
                })
            return results

    kv = FakeKv()

    # Reproduce the resolve_hashes handler logic (from remote_g2_source_setup.py).
    hashes = [11, 22, 33]
    registry_window_size = WINDOW_SIZE
    registry_block_size_bytes = BLOCK_SIZE_BYTES
    registry_pool_id = "host-pool-0"
    registry_pool_base_ptr = POOL_BASE_PTR

    # Step 1: Initial lookup via _find_and_pin_blocks_by_hash pattern.
    raw_results = kv.find_and_pin_blocks_by_hash(
        [int(h) for h in hashes],
        int(registry_window_size),
        tier="host_pinned",
        stop_on_miss=True,
    )

    descs = []
    pinned_count = 0
    primary_only_indices = []
    for idx, raw in enumerate(raw_results):
        if bool(raw.get("pinned", False)):
            slot_idx = int(raw["slot_idx"])
            byte_offset = slot_idx * registry_block_size_bytes
            descs.append({
                "block_hash": int(raw["block_hash"]),
                "byte_offset": byte_offset,
                "byte_length": registry_block_size_bytes,
                "pool_id": registry_pool_id,
            })
            pinned_count += 1
        else:
            descs.append(None)
            if raw.get("found_tier") == "primary":
                primary_only_indices.append(idx)

    # Step 2: Force-offload retry for primary-only blocks.
    assert len(primary_only_indices) == 3, "all 3 blocks should be primary-only"
    assert pinned_count == 0

    if primary_only_indices and registry_window_size is not None:
        force_hashes = [int(hashes[i]) for i in primary_only_indices]
        force_results = kv.force_offload_and_pin_blocks_by_hash(
            force_hashes,
            int(registry_window_size),
        )
        force_ok = 0
        for fi, fr in zip(primary_only_indices, force_results):
            if bool(fr.get("pinned", False)):
                slot_idx = int(fr["slot_idx"])
                byte_offset = slot_idx * registry_block_size_bytes
                descs[fi] = {
                    "block_hash": int(fr["block_hash"]),
                    "byte_offset": byte_offset,
                    "byte_length": registry_block_size_bytes,
                    "pool_id": registry_pool_id,
                    "metadata": {
                        "nixl_memory_desc": {
                            "ptr": registry_pool_base_ptr + byte_offset,
                            "len": registry_block_size_bytes,
                        }
                    },
                }
                pinned_count += 1
                force_ok += 1

    # Verify: all 3 blocks should now have descriptors filled in.
    assert pinned_count == 3
    assert force_ok == 3
    assert all(d is not None for d in descs)

    # Verify the force-offload was called with the correct hashes.
    assert force_calls == [[11, 22, 33]]

    # Verify byte_offset calculation: slot 50 * 4096, slot 51 * 4096, slot 52 * 4096.
    assert descs[0]["byte_offset"] == 50 * BLOCK_SIZE_BYTES
    assert descs[1]["byte_offset"] == 51 * BLOCK_SIZE_BYTES
    assert descs[2]["byte_offset"] == 52 * BLOCK_SIZE_BYTES

    # Verify nixl_memory_desc ptr = pool_base + byte_offset.
    for d in descs:
        assert d["metadata"]["nixl_memory_desc"]["ptr"] == POOL_BASE_PTR + d["byte_offset"]


def test_resolve_hashes_force_offload_partial_success():
    """When force_offload succeeds for some blocks but not others, only
    the successful ones should have descriptors filled in."""
    BLOCK_SIZE_BYTES = 4096
    POOL_BASE_PTR = 0x3000_0000
    WINDOW_SIZE = 4096

    class FakeKv:
        def find_and_pin_blocks_by_hash(
            self, block_hashes, window_size, tier="host_pinned", stop_on_miss=True
        ):
            # Block 11 is already pinned in secondary; block 22 is primary-only.
            results = []
            for bh in block_hashes:
                if int(bh) == 11:
                    results.append({
                        "block_hash": 11,
                        "pinned": True,
                        "found_tier": "host_pinned",
                        "block_id": 42,
                        "slot_idx": 5,
                    })
                else:
                    results.append({
                        "block_hash": int(bh),
                        "pinned": False,
                        "found_tier": "primary",
                        "block_id": None,
                        "slot_idx": None,
                    })
            return results

        def force_offload_and_pin_blocks_by_hash(self, block_hashes, window_size):
            # Force-offload succeeds for 22 but fails for 33 (no secondary space).
            results = []
            for bh in block_hashes:
                if int(bh) == 22:
                    results.append({
                        "block_hash": 22,
                        "pinned": True,
                        "found_tier": "host_pinned",
                        "block_id": 88,
                        "slot_idx": 9,
                    })
                else:
                    results.append({
                        "block_hash": int(bh),
                        "pinned": False,
                        "found_tier": "primary",
                        "block_id": None,
                        "slot_idx": None,
                    })
            return results

    kv = FakeKv()
    hashes = [11, 22, 33]

    # Step 1: Initial lookup.
    raw_results = kv.find_and_pin_blocks_by_hash(hashes, WINDOW_SIZE)
    descs = []
    sibling_pins = {}
    pinned_count = 0
    primary_only_indices = []
    for idx, raw in enumerate(raw_results):
        if bool(raw.get("pinned", False)):
            slot_idx = int(raw["slot_idx"])
            byte_offset = slot_idx * BLOCK_SIZE_BYTES
            descs.append({"block_hash": int(raw["block_hash"]), "byte_offset": byte_offset})
            sibling_pins[int(raw["block_hash"])] = int(raw["block_id"])
            pinned_count += 1
        else:
            descs.append(None)
            if raw.get("found_tier") == "primary":
                primary_only_indices.append(idx)

    # Block 11 pinned, blocks 22 and 33 primary-only.
    assert pinned_count == 1
    assert primary_only_indices == [1, 2]

    # Step 2: Force-offload retry.
    force_hashes = [int(hashes[i]) for i in primary_only_indices]
    force_results = kv.force_offload_and_pin_blocks_by_hash(force_hashes, WINDOW_SIZE)
    for fi, fr in zip(primary_only_indices, force_results):
        if bool(fr.get("pinned", False)):
            slot_idx = int(fr["slot_idx"])
            byte_offset = slot_idx * BLOCK_SIZE_BYTES
            descs[fi] = {"block_hash": int(fr["block_hash"]), "byte_offset": byte_offset}
            sibling_pins[int(fr["block_hash"])] = int(fr["block_id"])
            pinned_count += 1

    # Block 22 was force-offloaded; block 33 failed — desc stays None.
    assert pinned_count == 2
    assert descs[0] is not None  # block 11 (was already secondary)
    assert descs[1] is not None  # block 22 (force-offloaded)
    assert descs[2] is None      # block 33 (force-offload failed)
    assert descs[0]["byte_offset"] == 5 * BLOCK_SIZE_BYTES
    assert descs[1]["byte_offset"] == 9 * BLOCK_SIZE_BYTES
    assert sibling_pins == {11: 42, 22: 88}


def test_remote_g2_terminal_cleanup_releases_lease_once_for_all_reasons():
    released = []
    store = TargetRemoteG2BindingStore(
        release_lease=lambda lease_id, reason: released.append((lease_id, reason)) or True
    )
    cases = (
        ("cancelled", RemoteG2BindingState.CANCELLED),
        ("timeout", RemoteG2BindingState.CANCELLED),
        ("disconnect", RemoteG2BindingState.CANCELLED),
        ("transfer_failed", RemoteG2BindingState.TRANSFER_FAILED),
        ("success", RemoteG2BindingState.RELEASED),
    )

    for index, (reason, expected_state) in enumerate(cases):
        request_id = 2000 + index
        lease_id = f"lease-{reason}"
        record = store.resolve_for_request(
            request_id,
            RemoteKvReusePlan.from_dict(_plan(plan_id=f"plan-{index}")),
            0,
            lambda plan, lease_id=lease_id: _resolve_result(lease_id=lease_id),
        )

        assert store.release(request_id, reason) is True
        assert store.release(request_id, reason) is False
        assert record.state is expected_state

    assert released == [(f"lease-{reason}", reason) for reason, _ in cases]
