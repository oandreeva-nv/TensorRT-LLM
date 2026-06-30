# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_CONNECTOR_PACKAGE = "tensorrt_llm._torch.pyexecutor.connectors"
_CONNECTOR_DIR = (
    Path(__file__).resolve().parents[3]
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
)
_REMOTE_G2_PATH = _CONNECTOR_DIR / "remote_g2.py"
_REMOTE_G2_OBSERVABILITY_PATH = _CONNECTOR_DIR / "remote_g2_observability.py"
_REMOTE_G2_SOURCE_ADAPTER_PATH = _CONNECTOR_DIR / "remote_g2_source_adapter.py"


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


def _load_modules():
    _install_package("tensorrt_llm")
    _install_package("tensorrt_llm._torch")
    _install_package("tensorrt_llm._torch.pyexecutor")
    _install_package(_CONNECTOR_PACKAGE)
    _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_observability",
        _REMOTE_G2_OBSERVABILITY_PATH,
    )
    remote_g2 = _load_module(f"{_CONNECTOR_PACKAGE}.remote_g2", _REMOTE_G2_PATH)
    adapter = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_source_adapter",
        _REMOTE_G2_SOURCE_ADAPTER_PATH,
    )
    return remote_g2, adapter


_REMOTE_G2, _ADAPTER = _load_modules()

SourceG2DescriptorRegistry = _REMOTE_G2.SourceG2DescriptorRegistry
SourceG2DescriptorRecord = _REMOTE_G2.SourceG2DescriptorRecord
SourceG2PublisherEventAdapter = _ADAPTER.SourceG2PublisherEventAdapter
make_kv_pin_callbacks = _ADAPTER.make_kv_pin_callbacks

SOURCE_WORKER_ID = 7
SOURCE_DP_RANK = 0
BLOCK_SIZE_BYTES = 4096
POOL_BASE_PTR = 0x10_0000_0000  # arbitrary host address for test


def _make_registry():
    return SourceG2DescriptorRegistry(
        source_worker_id=SOURCE_WORKER_ID,
        source_dp_rank=SOURCE_DP_RANK,
    )


def _make_adapter(registry=None):
    return SourceG2PublisherEventAdapter(
        registry or _make_registry(),
        source_worker_id=SOURCE_WORKER_ID,
        source_dp_rank=SOURCE_DP_RANK,
        block_size_bytes=BLOCK_SIZE_BYTES,
        secondary_pool_base_ptr=POOL_BASE_PTR,
    )


def _stored_event(blocks, parent_hash=None, event_id=1):
    return {
        "event_id": event_id,
        "data": {
            "type": "stored",
            "parent_hash": parent_hash,
            "blocks": blocks,
        },
    }


def _stored_block(block_hash, block_id, slot_idx, cache_level=1):
    return {
        "block_hash": block_hash,
        "block_id": block_id,
        "slot_idx": slot_idx,
        "cache_level": cache_level,
        "tokens": [],
    }


def _updated_event(
    block_hash,
    *,
    old_level=None,
    new_level=None,
    new_slot_idx=None,
    block_id=None,
    event_id=1,
):
    data = {"type": "updated", "block_hash": block_hash}
    if old_level is not None or new_level is not None:
        data["cache_level"] = {"old_value": old_level, "new_value": new_level}
    if new_slot_idx is not None:
        data["new_slot_idx"] = new_slot_idx
    if block_id is not None:
        data["block_id"] = block_id
    return {"event_id": event_id, "data": data}


def _removed_event(block_hashes, event_id=1):
    return {
        "event_id": event_id,
        "data": {"type": "removed", "block_hashes": list(block_hashes)},
    }


def _stored_event_for_hash(adapter, block_hash, block_id, slot_idx, cache_level=1):
    adapter.apply_event(
        _stored_event([_stored_block(block_hash, block_id, slot_idx, cache_level)])
    )


def _records_dict(registry):
    return registry._records  # type: ignore[attr-defined]


def test_constructor_rejects_invalid_inputs():
    registry = _make_registry()

    with pytest.raises(ValueError):
        SourceG2PublisherEventAdapter(
            registry,
            source_worker_id=SOURCE_WORKER_ID,
            source_dp_rank=SOURCE_DP_RANK,
            block_size_bytes=0,
            secondary_pool_base_ptr=POOL_BASE_PTR,
        )

    with pytest.raises(ValueError):
        SourceG2PublisherEventAdapter(
            registry,
            source_worker_id=SOURCE_WORKER_ID,
            source_dp_rank=SOURCE_DP_RANK,
            block_size_bytes=BLOCK_SIZE_BYTES,
            secondary_pool_base_ptr=0,
        )

    with pytest.raises(ValueError):
        SourceG2PublisherEventAdapter(
            registry,
            source_worker_id=SOURCE_WORKER_ID,
            source_dp_rank=SOURCE_DP_RANK,
            block_size_bytes=BLOCK_SIZE_BYTES,
            secondary_pool_base_ptr=POOL_BASE_PTR,
            tier="gpu",
        )

    with pytest.raises(ValueError):
        SourceG2PublisherEventAdapter(
            registry,
            source_worker_id=SOURCE_WORKER_ID + 1,
            source_dp_rank=SOURCE_DP_RANK,
            block_size_bytes=BLOCK_SIZE_BYTES,
            secondary_pool_base_ptr=POOL_BASE_PTR,
        )

    with pytest.raises(ValueError):
        SourceG2PublisherEventAdapter(
            registry,
            source_worker_id=SOURCE_WORKER_ID,
            source_dp_rank=SOURCE_DP_RANK + 1,
            block_size_bytes=BLOCK_SIZE_BYTES,
            secondary_pool_base_ptr=POOL_BASE_PTR,
        )


def test_stored_event_on_secondary_inserts_record():
    registry = _make_registry()
    adapter = _make_adapter(registry)
    _stored_event_for_hash(adapter, block_hash=111, block_id=42, slot_idx=5)

    record = _records_dict(registry)[111]
    assert record.block_hash == 111
    assert record.block_id == 42
    assert record.byte_offset == 5 * BLOCK_SIZE_BYTES
    assert record.byte_length == BLOCK_SIZE_BYTES
    assert record.tier == "host_pinned"
    assert record.descriptor_generation == 1
    nixl = record.metadata["nixl_memory_desc"]
    assert nixl["ptr"] == POOL_BASE_PTR + 5 * BLOCK_SIZE_BYTES
    assert nixl["size"] == BLOCK_SIZE_BYTES
    assert nixl["memory_type"] == "DRAM"


def test_stored_event_on_primary_is_skipped():
    registry = _make_registry()
    adapter = _make_adapter(registry)
    _stored_event_for_hash(adapter, block_hash=222, block_id=10, slot_idx=2, cache_level=0)
    assert 222 not in _records_dict(registry)


def test_stored_event_with_mixed_levels_only_keeps_secondary():
    registry = _make_registry()
    adapter = _make_adapter(registry)
    adapter.apply_event(
        _stored_event(
            [
                _stored_block(111, 1, 5, cache_level=1),
                _stored_block(222, 2, 6, cache_level=0),
                _stored_block(333, 3, 7, cache_level=1),
            ]
        )
    )
    records = _records_dict(registry)
    assert set(records.keys()) == {111, 333}


def test_stored_event_rejects_sentinel_values():
    registry = _make_registry()
    adapter = _make_adapter(registry)
    adapter.apply_event(
        _stored_event(
            [
                _stored_block(111, -1, 5),
                _stored_block(222, 1, -1),
                _stored_block(333, 3, 7),
            ]
        )
    )
    records = _records_dict(registry)
    assert set(records.keys()) == {333}


def test_updated_offload_inserts_or_refreshes_record():
    registry = _make_registry()
    adapter = _make_adapter(registry)
    adapter.apply_event(
        _updated_event(
            block_hash=111,
            old_level=0,
            new_level=1,
            new_slot_idx=8,
            block_id=42,
        )
    )
    record = _records_dict(registry)[111]
    assert record.block_id == 42
    assert record.byte_offset == 8 * BLOCK_SIZE_BYTES
    assert record.descriptor_generation == 1


def test_updated_onboard_removes_record():
    registry = _make_registry()
    adapter = _make_adapter(registry)
    _stored_event_for_hash(adapter, block_hash=111, block_id=42, slot_idx=5)
    assert 111 in _records_dict(registry)

    adapter.apply_event(
        _updated_event(
            block_hash=111,
            old_level=1,
            new_level=0,
            block_id=42,
        )
    )
    assert 111 not in _records_dict(registry)


def test_updated_priority_only_is_noop():
    registry = _make_registry()
    adapter = _make_adapter(registry)
    _stored_event_for_hash(adapter, block_hash=111, block_id=42, slot_idx=5)
    before_generation = _records_dict(registry)[111].descriptor_generation

    adapter.apply_event(
        _updated_event(block_hash=111, block_id=42)  # no cache_level diff
    )
    after = _records_dict(registry)[111]
    assert after.descriptor_generation == before_generation


def test_removed_event_drops_records_and_generation_counters():
    registry = _make_registry()
    adapter = _make_adapter(registry)
    _stored_event_for_hash(adapter, block_hash=111, block_id=42, slot_idx=5)
    _stored_event_for_hash(adapter, block_hash=222, block_id=43, slot_idx=6)
    assert _records_dict(registry).keys() == {111, 222}

    adapter.apply_event(_removed_event([111, 222]))
    assert _records_dict(registry) == {}

    _stored_event_for_hash(adapter, block_hash=111, block_id=42, slot_idx=5)
    assert _records_dict(registry)[111].descriptor_generation == 1


def test_repeated_upserts_bump_descriptor_generation():
    registry = _make_registry()
    adapter = _make_adapter(registry)
    for _ in range(3):
        _stored_event_for_hash(adapter, block_hash=111, block_id=42, slot_idx=5)
    assert _records_dict(registry)[111].descriptor_generation == 3


def test_malformed_event_is_silent_noop():
    registry = _make_registry()
    adapter = _make_adapter(registry)
    adapter.apply_event({})
    adapter.apply_event({"data": "not a mapping"})
    adapter.apply_event({"data": {"type": "stored"}})
    adapter.apply_event({"data": {"type": "stored", "blocks": []}})
    adapter.apply_event({"data": {"type": "stored", "blocks": [{}]}})
    adapter.apply_event({"data": {"type": "updated"}})
    adapter.apply_event({"data": {"type": "removed"}})
    adapter.apply_event({"data": {"type": "unknown"}})
    adapter.apply_event("not a mapping")  # type: ignore[arg-type]
    assert _records_dict(registry) == {}


# ---------------------------------------------------------------------------
# make_kv_pin_callbacks tests
# ---------------------------------------------------------------------------


class _FakeKvCacheManager:
    def __init__(self, pin_locations):
        # pin_locations maps block_id -> (slot_idx, cache_level)
        self._pin_locations = dict(pin_locations)
        self.pin_calls: list[list[int]] = []
        self.unpin_calls: list[list[int]] = []

    def pin_blocks_by_id(self, block_ids):
        self.pin_calls.append(list(block_ids))
        return [self._pin_locations[bid] for bid in block_ids]

    def unpin_blocks_by_id(self, block_ids):
        self.unpin_calls.append(list(block_ids))


def _build_record(block_hash=111, block_id=42, slot_idx=5, byte_length=4096):
    return SourceG2DescriptorRecord(
        block_hash=block_hash,
        source_worker_id=SOURCE_WORKER_ID,
        source_dp_rank=SOURCE_DP_RANK,
        tier="host_pinned",
        descriptor_generation=1,
        pool_id="g2-host-pinned",
        byte_offset=slot_idx * byte_length,
        byte_length=byte_length,
        block_id=block_id,
        metadata={
            "nixl_memory_desc": {
                "ptr": POOL_BASE_PTR + slot_idx * byte_length,
                "size": byte_length,
                "device_id": 0,
                "memory_type": "DRAM",
                "name": "g2-host-pinned",
            }
        },
    )


def test_make_kv_pin_callbacks_rejects_invalid_inputs():
    fake_kv = _FakeKvCacheManager({})
    with pytest.raises(ValueError):
        make_kv_pin_callbacks(
            fake_kv, secondary_pool_base_ptr=POOL_BASE_PTR, block_size_bytes=0
        )
    with pytest.raises(ValueError):
        make_kv_pin_callbacks(
            fake_kv, secondary_pool_base_ptr=0, block_size_bytes=BLOCK_SIZE_BYTES
        )


def test_acquire_pin_success_refreshes_record_to_post_pin_slot():
    # Block was at slot 5 at event time, but pin reports it's now at slot 9.
    fake_kv = _FakeKvCacheManager({42: (9, 1)})
    acquire, _ = make_kv_pin_callbacks(
        fake_kv,
        secondary_pool_base_ptr=POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
    )
    record = _build_record(block_id=42, slot_idx=5)
    pin_ref = acquire(record, "lease-xyz")
    assert pin_ref == 42
    assert record.byte_offset == 9 * BLOCK_SIZE_BYTES
    assert (
        record.metadata["nixl_memory_desc"]["ptr"]
        == POOL_BASE_PTR + 9 * BLOCK_SIZE_BYTES
    )
    assert fake_kv.pin_calls == [[42]]
    assert fake_kv.unpin_calls == []


def test_acquire_pin_on_primary_unpins_and_raises():
    # Pin reports the block migrated to primary (level 0) between event and pin.
    fake_kv = _FakeKvCacheManager({42: (3, 0)})
    acquire, _ = make_kv_pin_callbacks(
        fake_kv,
        secondary_pool_base_ptr=POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
    )
    record = _build_record(block_id=42, slot_idx=5)
    with pytest.raises(RuntimeError):
        acquire(record, "lease-xyz")
    assert fake_kv.pin_calls == [[42]]
    assert fake_kv.unpin_calls == [[42]]


def test_acquire_pin_invalid_block_id_raises_without_calling_kv():
    fake_kv = _FakeKvCacheManager({})
    acquire, _ = make_kv_pin_callbacks(
        fake_kv,
        secondary_pool_base_ptr=POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
    )
    record = _build_record(block_id=-1)
    with pytest.raises(ValueError):
        acquire(record, "lease-xyz")
    assert fake_kv.pin_calls == []
    assert fake_kv.unpin_calls == []


def test_release_pin_calls_unpin_with_block_id():
    fake_kv = _FakeKvCacheManager({})
    _, release = make_kv_pin_callbacks(
        fake_kv,
        secondary_pool_base_ptr=POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
    )
    release(42)
    assert fake_kv.unpin_calls == [[42]]


def test_release_pin_tolerates_sentinel_values():
    fake_kv = _FakeKvCacheManager({})
    _, release = make_kv_pin_callbacks(
        fake_kv,
        secondary_pool_base_ptr=POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
    )
    release(None)
    release(-1)
    assert fake_kv.unpin_calls == []


def test_acquire_pin_then_release_round_trips_through_registry():
    # End-to-end: register, resolve-and-lease through registry, release_lease.
    # Verifies the callbacks integrate correctly with the registry's existing
    # lease lifecycle code.
    fake_kv = _FakeKvCacheManager({42: (5, 1), 43: (6, 1)})
    registry = _make_registry()
    acquire, release = make_kv_pin_callbacks(
        fake_kv,
        secondary_pool_base_ptr=POOL_BASE_PTR,
        block_size_bytes=BLOCK_SIZE_BYTES,
    )
    # Rebuild registry with the callbacks plumbed in.
    registry = SourceG2DescriptorRegistry(
        source_worker_id=SOURCE_WORKER_ID,
        source_dp_rank=SOURCE_DP_RANK,
        acquire_pin=acquire,
        release_pin=release,
    )
    registry.upsert_descriptor(_build_record(block_hash=111, block_id=42, slot_idx=5))
    registry.upsert_descriptor(_build_record(block_hash=222, block_id=43, slot_idx=6))

    import time

    now_ms = int(time.time() * 1000)
    plan = {
        "plan_id": "plan-1",
        "request_id": "req-1",
        "target_worker_id": 99,
        "target_dp_rank": 1,
        "source_worker_id": SOURCE_WORKER_ID,
        "source_dp_rank": SOURCE_DP_RANK,
        "source_tier": "host_pinned",
        "block_hashes": [111, 222, 333],
        "start_block_index": 0,
        "planned_prefix_blocks": 3,  # +1: from_dict reserves the last block
        "block_size_tokens": 16,
        "created_at_ms": now_ms,
        "expires_at_ms": now_ms + 60_000,
        "plan_version": _REMOTE_G2.REMOTE_KV_REUSE_PLAN_VERSION,
    }
    result = registry.resolve_and_lease(plan)
    assert result.lease_id is not None
    assert len(result.descriptors) == 2
    assert fake_kv.pin_calls == [[42], [43]]
    assert fake_kv.unpin_calls == []

    assert registry.release_lease(result.lease_id, "test_done")
    assert fake_kv.unpin_calls == [[42], [43]]
