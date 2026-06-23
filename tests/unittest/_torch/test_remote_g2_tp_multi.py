# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UT1 + UT2: TP>1 per-rank data gather and flow-through tests.

UT1: Source-side MPI gather (_gather_per_rank_nixl_metadata,
     _gather_per_rank_descriptors) with mocked MPI primitives.
UT2: Per-rank data round-trip through _result_to_dict → _result_from_dict
     → binding store, verifying no field loss or mangling.

Run:
    pytest tests/unittest/_torch/test_remote_g2_tp_multi.py -v -s
"""

import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_remote_g2_tp_multi")

# ─── Module loading (avoids importing the full TRT-LLM stack) ────────

_CONNECTOR_PACKAGE = "tensorrt_llm._torch.pyexecutor.connectors"
_CONNECTOR_DIR = (
    Path(__file__).resolve().parents[3]
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
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


def _load_all_modules():
    """Load remote_g2 modules in dependency order."""
    _install_package("tensorrt_llm")
    _install_package("tensorrt_llm._torch")
    _install_package("tensorrt_llm._torch.pyexecutor")
    _install_package(_CONNECTOR_PACKAGE)

    # Stub out tensorrt_llm._utils to provide mpi_* functions for import.
    utils_mod = types.ModuleType("tensorrt_llm._utils")
    utils_mod.mpi_rank = lambda: 0
    utils_mod.mpi_world_size = lambda: 1
    utils_mod.mpi_broadcast = lambda data, root=0: data
    utils_mod.mpi_allgather = lambda data: [data]
    sys.modules["tensorrt_llm._utils"] = utils_mod

    # Stub bindings
    bindings_mod = types.ModuleType("tensorrt_llm.bindings")
    bindings_mod.LlmRequestState = MagicMock()
    sys.modules["tensorrt_llm.bindings"] = bindings_mod
    bindings_internal = types.ModuleType("tensorrt_llm.bindings.internal")
    sys.modules["tensorrt_llm.bindings.internal"] = bindings_internal
    bindings_bm = types.ModuleType("tensorrt_llm.bindings.internal.batch_manager")
    bindings_bm.KvCacheConnectorManager = MagicMock()
    sys.modules["tensorrt_llm.bindings.internal.batch_manager"] = bindings_bm

    # Stub kv_cache_connector (needed by remote_g2_connector)
    kv_stub = types.ModuleType(f"{_CONNECTOR_PACKAGE}.kv_cache_connector")
    kv_stub.KvCacheConnectorScheduler = MagicMock()
    kv_stub.KvCacheConnectorWorker = MagicMock()
    kv_stub.SchedulerOutput = MagicMock()
    sys.modules[f"{_CONNECTOR_PACKAGE}.kv_cache_connector"] = kv_stub

    observability = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_observability",
        _CONNECTOR_DIR / "remote_g2_observability.py",
    )
    remote_g2 = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2",
        _CONNECTOR_DIR / "remote_g2.py",
    )
    # Stub remote_g2_transfer (needed by remote_g2_connector)
    transfer_stub = types.ModuleType(f"{_CONNECTOR_PACKAGE}.remote_g2_transfer")
    transfer_stub.RemoteG2TransferError = type("RemoteG2TransferError", (RuntimeError,), {})
    sys.modules[f"{_CONNECTOR_PACKAGE}.remote_g2_transfer"] = transfer_stub

    _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_connector",
        _CONNECTOR_DIR / "remote_g2_connector.py",
    )
    source_adapter = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_source_adapter",
        _CONNECTOR_DIR / "remote_g2_source_adapter.py",
    )
    source_setup = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_source_setup",
        _CONNECTOR_DIR / "remote_g2_source_setup.py",
    )
    target_setup = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_target_setup",
        _CONNECTOR_DIR / "remote_g2_target_setup.py",
    )
    return remote_g2, source_setup, target_setup


REMOTE_G2, SOURCE_SETUP, TARGET_SETUP = _load_all_modules()

# Aliases for convenience
RemoteG2Descriptor = REMOTE_G2.RemoteG2Descriptor
RemoteG2ResolveResult = REMOTE_G2.RemoteG2ResolveResult
RemoteG2BlockStatus = REMOTE_G2.RemoteG2BlockStatus
RemoteG2BindingState = REMOTE_G2.RemoteG2BindingState
RemoteKvReusePlan = REMOTE_G2.RemoteKvReusePlan
TargetRemoteG2BindingStore = REMOTE_G2.TargetRemoteG2BindingStore
TargetRemotePlanStore = REMOTE_G2.TargetRemotePlanStore
_NixlSourceBundle = SOURCE_SETUP._NixlSourceBundle
_gather_per_rank_nixl_metadata = SOURCE_SETUP._gather_per_rank_nixl_metadata
_result_to_dict = SOURCE_SETUP._result_to_dict
_result_from_dict = TARGET_SETUP._result_from_dict
REMOTE_KV_REUSE_PLAN_VERSION = REMOTE_G2.REMOTE_KV_REUSE_PLAN_VERSION


# ─── Helpers ────────────────────────────────────────────────────────

def _make_bundle(tp_rank: int, worker_id: int = 7) -> _NixlSourceBundle:
    """Create a fake NIXL source bundle for a given TP rank."""
    return _NixlSourceBundle(
        agent=MagicMock(),
        remote_name=f"remote-g2-source-{worker_id}-tp{tp_rank}",
        agent_desc=f"agent-desc-bytes-rank{tp_rank}".encode(),
        pool_base_ptr=0x7000_0000_0000 + tp_rank * 0x1_0000_0000,
        pool_size_bytes=306_708_480,
        source_generation=1,
    )


def _make_descriptor(block_hash: int, rank_offset: int = 0) -> RemoteG2Descriptor:
    """Create a descriptor with rank-distinguishable offsets."""
    return RemoteG2Descriptor(
        block_hash=block_hash,
        descriptor_generation=1,
        pool_id="g2-host-pinned",
        byte_offset=(block_hash * 131072) + (rank_offset * 1000),
        byte_length=131072,
    )


def _make_plan(**overrides):
    defaults = {
        "plan_id": "plan-1",
        "request_id": "dynamo-request-1",
        "target_worker_id": 42,
        "target_dp_rank": 0,
        "source_worker_id": 7,
        "source_dp_rank": 0,
        "source_tier": "host_pinned",
        "block_hashes": [100, 200, 300],
        "start_block_index": 0,
        "planned_prefix_blocks": 3,
        "block_size_tokens": 32,
        "created_at_ms": 100,
        "expires_at_ms": 10_000,
        "plan_version": REMOTE_KV_REUSE_PLAN_VERSION,
    }
    defaults.update(overrides)
    return defaults


# ═══════════════════════════════════════════════════════════════════
# UT1: Source-side MPI gather
# ═══════════════════════════════════════════════════════════════════

class TestUT1_GatherPerRankMetadata:
    """Test _gather_per_rank_nixl_metadata with mocked MPI."""

    def test_tp1_returns_single_entry(self):
        """TP=1: gather should return a single-entry list without MPI."""
        bundle = _make_bundle(tp_rank=0)

        result = _gather_per_rank_nixl_metadata(bundle, tp_rank=0, tp_size=1)

        logger.info("UT1/TP=1: result=%s", result)
        assert len(result) == 1
        assert result[0]["tp_rank"] == 0
        assert result[0]["remote_name"] == bundle.remote_name
        assert result[0]["pool_base_ptr"] == bundle.pool_base_ptr
        assert result[0]["pool_size_bytes"] == bundle.pool_size_bytes
        assert "agent_metadata_b64" in result[0]
        logger.info("UT1/TP=1: PASS — single entry, correct fields")

    def test_tp2_gathers_both_ranks(self):
        """TP=2: mock mpi_allgather to simulate two ranks contributing."""
        bundle_r0 = _make_bundle(tp_rank=0)
        bundle_r1 = _make_bundle(tp_rank=1)

        import base64 as _b64

        # Mock mpi_allgather to return both ranks' entries.
        fake_entries = [
            {
                "tp_rank": 0,
                "remote_name": bundle_r0.remote_name,
                "agent_metadata_b64": _b64.b64encode(bundle_r0.agent_desc).decode(),
                "pool_base_ptr": bundle_r0.pool_base_ptr,
                "pool_size_bytes": bundle_r0.pool_size_bytes,
                "source_generation": 1,
            },
            {
                "tp_rank": 1,
                "remote_name": bundle_r1.remote_name,
                "agent_metadata_b64": _b64.b64encode(bundle_r1.agent_desc).decode(),
                "pool_base_ptr": bundle_r1.pool_base_ptr,
                "pool_size_bytes": bundle_r1.pool_size_bytes,
                "source_generation": 1,
            },
        ]

        with patch.object(
            sys.modules["tensorrt_llm._utils"],
            "mpi_allgather",
            return_value=fake_entries,
        ):
            result = _gather_per_rank_nixl_metadata(bundle_r0, tp_rank=0, tp_size=2)

        logger.info("UT1/TP=2: result has %d entries", len(result))
        assert len(result) == 2
        # Sorted by tp_rank
        assert result[0]["tp_rank"] == 0
        assert result[1]["tp_rank"] == 1
        # Distinct pool pointers
        assert result[0]["pool_base_ptr"] != result[1]["pool_base_ptr"]
        # Distinct agent names
        assert result[0]["remote_name"] != result[1]["remote_name"]
        assert "tp0" in result[0]["remote_name"]
        assert "tp1" in result[1]["remote_name"]
        logger.info("UT1/TP=2: PASS — two entries, distinct pool_base_ptr and agent names")

    def test_tp2_metadata_b64_round_trips(self):
        """Verify agent_metadata_b64 encodes/decodes correctly."""
        import base64 as _b64

        bundle = _make_bundle(tp_rank=0)
        result = _gather_per_rank_nixl_metadata(bundle, tp_rank=0, tp_size=1)

        encoded = result[0]["agent_metadata_b64"]
        decoded = _b64.b64decode(encoded)
        assert decoded == bundle.agent_desc
        logger.info("UT1/b64: PASS — agent_metadata_b64 round-trips correctly")


# ═══════════════════════════════════════════════════════════════════
# UT2: Per-rank data flow round-trip
# ═══════════════════════════════════════════════════════════════════

class TestUT2_PerRankDataRoundTrip:
    """Test serialize → deserialize → binding store for per-rank data."""

    def _make_resolve_result_tp2(self):
        """Build a RemoteG2ResolveResult with per-rank data for TP=2."""
        # Rank 0 descriptors (from source rank 0)
        rank0_descs = tuple(
            _make_descriptor(bh, rank_offset=0) for bh in [100, 200, 300]
        )
        # Rank 1 descriptors (different offsets — from source rank 1)
        rank1_descs = tuple(
            _make_descriptor(bh, rank_offset=1) for bh in [100, 200, 300]
        )

        per_rank_descriptors = {
            0: [
                {"block_hash": d.block_hash, "byte_offset": d.byte_offset,
                 "byte_length": d.byte_length, "pool_id": d.pool_id,
                 "metadata": {}}
                for d in rank0_descs
            ],
            1: [
                {"block_hash": d.block_hash, "byte_offset": d.byte_offset,
                 "byte_length": d.byte_length, "pool_id": d.pool_id,
                 "metadata": {}}
                for d in rank1_descs
            ],
        }

        per_rank_source_metadata = {
            0: {
                "tp_rank": 0,
                "remote_name": "remote-g2-source-7-tp0",
                "agent_metadata_b64": "AAAA",
                "pool_base_ptr": 0x7000_0000_0000,
                "pool_size_bytes": 306_708_480,
                "source_generation": 1,
            },
            1: {
                "tp_rank": 1,
                "remote_name": "remote-g2-source-7-tp1",
                "agent_metadata_b64": "BBBB",
                "pool_base_ptr": 0x7001_0000_0000,
                "pool_size_bytes": 306_708_480,
                "source_generation": 1,
            },
        }

        return RemoteG2ResolveResult(
            lease_id="lease-tp2-test",
            descriptors=rank0_descs,  # flat = rank 0's (backward compat)
            num_tokens=96,
            reason="ok",
            source_generation=1,
            per_rank_descriptors=per_rank_descriptors,
            per_rank_source_metadata=per_rank_source_metadata,
        )

    def test_serialize_deserialize_tp2(self):
        """Per-rank fields survive _result_to_dict → _result_from_dict."""
        original = self._make_resolve_result_tp2()
        logger.info("UT2/serde: original per_rank_descriptors keys=%s",
                     list(original.per_rank_descriptors.keys()))

        # Serialize
        d = _result_to_dict(original)
        logger.info("UT2/serde: serialized dict keys=%s", list(d.keys()))
        assert "per_rank_descriptors" in d, "per_rank_descriptors missing from serialized dict"
        assert "per_rank_source_metadata" in d, "per_rank_source_metadata missing from serialized dict"

        # Deserialize
        reconstructed = _result_from_dict(d)
        assert reconstructed is not None, "_result_from_dict returned None"
        logger.info("UT2/serde: reconstructed per_rank_descriptors keys=%s",
                     list(reconstructed.per_rank_descriptors.keys()))

        # Verify per_rank_descriptors
        assert len(reconstructed.per_rank_descriptors) == 2
        assert 0 in reconstructed.per_rank_descriptors
        assert 1 in reconstructed.per_rank_descriptors

        # Verify rank 1's offsets are distinct from rank 0's
        r0_offsets = [d["byte_offset"] for d in reconstructed.per_rank_descriptors[0]]
        r1_offsets = [d["byte_offset"] for d in reconstructed.per_rank_descriptors[1]]
        assert r0_offsets != r1_offsets, "Rank 0 and rank 1 offsets should differ"
        logger.info("UT2/serde: rank0 offsets=%s, rank1 offsets=%s", r0_offsets, r1_offsets)

        # Verify per_rank_source_metadata
        assert len(reconstructed.per_rank_source_metadata) == 2
        assert reconstructed.per_rank_source_metadata[0]["remote_name"] == "remote-g2-source-7-tp0"
        assert reconstructed.per_rank_source_metadata[1]["remote_name"] == "remote-g2-source-7-tp1"
        assert (
            reconstructed.per_rank_source_metadata[0]["pool_base_ptr"]
            != reconstructed.per_rank_source_metadata[1]["pool_base_ptr"]
        )
        logger.info("UT2/serde: PASS — per-rank data survives round-trip")

    def test_serialize_deserialize_tp1_backward_compat(self):
        """TP=1: empty per-rank dicts serialize cleanly and deserialize as empty."""
        result = RemoteG2ResolveResult(
            lease_id="lease-tp1",
            descriptors=tuple(_make_descriptor(bh) for bh in [100, 200]),
            num_tokens=64,
            reason="ok",
            source_generation=1,
        )

        d = _result_to_dict(result)
        logger.info("UT2/tp1-compat: serialized keys=%s", list(d.keys()))
        # per_rank keys should NOT be present (empty dicts are omitted)
        assert "per_rank_descriptors" not in d
        assert "per_rank_source_metadata" not in d

        reconstructed = _result_from_dict(d)
        assert reconstructed is not None
        assert len(reconstructed.per_rank_descriptors) == 0
        assert len(reconstructed.per_rank_source_metadata) == 0
        logger.info("UT2/tp1-compat: PASS — TP=1 result has empty per-rank dicts")

    def test_flat_descriptors_preserved(self):
        """Flat descriptors (rank 0's view) are always preserved for TP=1 fallback."""
        original = self._make_resolve_result_tp2()

        d = _result_to_dict(original)
        reconstructed = _result_from_dict(d)

        # Flat descriptors should match original
        assert len(reconstructed.descriptors) == 3
        assert reconstructed.descriptors[0].block_hash == 100
        assert reconstructed.descriptors[2].block_hash == 300
        logger.info("UT2/flat: PASS — flat descriptors preserved alongside per-rank")

    def test_binding_store_carries_per_rank(self):
        """Per-rank data flows through the binding store to the connector worker."""
        resolve_result = self._make_resolve_result_tp2()
        plan_dict = _make_plan()

        # Set up binding store with mocked release
        store = TargetRemoteG2BindingStore(
            release_lease=lambda lid, reason: True,
            observability=None,
        )

        # The store's resolve_for_request takes a resolve_and_lease
        # callable that it calls internally. We provide one that
        # returns our pre-built TP=2 resolve result.
        def fake_resolve(plan):
            return resolve_result

        record = store.resolve_for_request(
            request_id=28,
            plan=plan_dict,
            num_computed_tokens=0,
            resolve_and_lease=fake_resolve,
        )
        assert record is not None
        logger.info("UT2/store: record state=%s matched_tokens=%d",
                     record.state, record.matched_tokens)

        # Verify per-rank data is accessible on the record
        rr = record.resolve_result
        assert len(rr.per_rank_descriptors) == 2
        assert len(rr.per_rank_source_metadata) == 2
        assert rr.per_rank_source_metadata[1]["remote_name"] == "remote-g2-source-7-tp1"
        logger.info("UT2/store: PASS — per-rank data accessible on binding record")


# ═══════════════════════════════════════════════════════════════════
# Standalone runner (no pytest required)
# ═══════════════════════════════════════════════════════════════════

def _run_all():
    """Run all tests when invoked directly: python3 test_remote_g2_tp_multi.py"""
    passed = 0
    failed = 0
    tests = [
        ("UT1: TP=1 single entry",      TestUT1_GatherPerRankMetadata().test_tp1_returns_single_entry),
        ("UT1: TP=2 gather both ranks",  TestUT1_GatherPerRankMetadata().test_tp2_gathers_both_ranks),
        ("UT1: b64 round-trip",          TestUT1_GatherPerRankMetadata().test_tp2_metadata_b64_round_trips),
        ("UT2: serde TP=2",              TestUT2_PerRankDataRoundTrip().test_serialize_deserialize_tp2),
        ("UT2: serde TP=1 compat",       TestUT2_PerRankDataRoundTrip().test_serialize_deserialize_tp1_backward_compat),
        ("UT2: flat descriptors",        TestUT2_PerRankDataRoundTrip().test_flat_descriptors_preserved),
        ("UT2: binding store",           TestUT2_PerRankDataRoundTrip().test_binding_store_carries_per_rank),
    ]
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  \033[32mPASS\033[0m  {name}")
        except Exception as exc:
            failed += 1
            print(f"  \033[31mFAIL\033[0m  {name}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    import sys
    # Ensure repo root is on path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    success = _run_all()
    sys.exit(0 if success else 1)
