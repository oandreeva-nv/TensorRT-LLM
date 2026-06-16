# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap the source-side SourceG2DescriptorRegistry from inside the
engine subprocess.

The KV cache manager (and the C++ APIs we depend on - find_and_pin_blocks_by_hash,
pin_blocks_by_id, get_secondary_pool_data) only exists in the engine
subprocess that PyExecutor runs in. So the registry has to be built there.
The connector worker's register_kv_caches hook is the natural anchor: it
runs in that subprocess, right after the KV cache pool is allocated.

Identity (source_worker_id, source_dp_rank) is read from environment
variables that the dynamo worker process sets before spawning the engine.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from .remote_g2 import SourceG2DescriptorRegistry
from .remote_g2_source_adapter import make_kv_pin_callbacks


@dataclass
class _NixlSourceBundle:
    """Source-side NIXL agent + metadata captured for the metadata RPC.

    `agent` is the live NixlTransferAgent (kept alive for the worker's
    lifetime — its memory registrations and the daemon polling thread
    expire when the agent is GCed). `agent_desc` is the opaque
    bytes-blob other workers' agents need to call `load_remote_agent`.
    `remote_name` is the agent's identifier used as the third argument
    to `TransferRequest(..., remote_name)` from a peer.
    """

    agent: Any
    remote_name: str
    agent_desc: bytes
    pool_base_ptr: int
    pool_size_bytes: int
    source_generation: int = 1


# Process-wide singleton — populated by maybe_start_remote_g2_service after
# the NIXL agent is constructed, read by the ZMQ REP loop when answering
# get_metadata RPCs (Stage T2).
_GLOBAL_NIXL_SOURCE_BUNDLE: Optional[_NixlSourceBundle] = None

# Per-rank NIXL bundles gathered from all TP siblings at startup (S1).
# Indexed by TP rank. TP=1 has a single entry at index 0.
# Populated by _gather_per_rank_nixl_metadata() after each rank builds
# its local NIXL agent.
_GLOBAL_PER_RANK_NIXL_BUNDLES: Optional[list[dict]] = None

# ---------------------------------------------------------------------------
# Prefill-pin tracker: blocks pinned by store_blocks_for_reuse(pin=True)
# during disagg prefill termination so they survive in the radix trie
# until the decode worker completes KV transfer (release_lease).
#
# Written by the executor thread (register_prefill_pins), read/popped by
# the ZMQ REP thread (pop_prefill_pin).  Access is protected by a lock.
#
# Each entry stores a timestamp so stale pins (e.g. due to decode worker
# crash / timeout) can be cleaned up periodically.
#
# Fix 10 — lease-gated pinning: prefill pins are only applied when remote
# resolve activity has been observed (at least one resolve_and_lease
# received).  Without remote activity, pinning is wasteful and can starve
# the GUARANTEED_NO_EVICT scheduler.  Additionally, pinning is refused
# when free primary blocks drop below a safety floor to prevent OOM-style
# stalls.
# ---------------------------------------------------------------------------
import time as _time

_PREFILL_PIN_TTL_S = 120.0  # seconds before a prefill pin is considered stale

_prefill_pin_lock = threading.Lock()
_prefill_pinned: dict[int, float] = {}  # block_id → registration timestamp
_prefill_pin_register_count: int = 0  # monotonic; drives periodic stale sweep

# ---------------------------------------------------------------------------
# Remote-resolve activity gate for prefill pinning (Fix 10).
#
# Set to True by the ZMQ REP handler when the first resolve_and_lease
# arrives from any remote worker.  Once True, never reverts — the
# presence of remote resolvers means future blocks may be requested.
# ---------------------------------------------------------------------------
_remote_resolve_seen: bool = False


def notify_remote_resolve_seen() -> None:
    """Mark that at least one remote resolve_and_lease has been received."""
    global _remote_resolve_seen
    _remote_resolve_seen = True


def is_remote_resolve_active() -> bool:
    """Return True if any remote resolve_and_lease has been received."""
    return _remote_resolve_seen


def should_prefill_pin(free_blocks: int, max_blocks_per_seq: int,
                       pin_budget: int, total_primary_blocks: int = 0,
                       est_new_blocks: int = 0) -> bool:
    """Decide whether store_blocks_for_reuse should use pin_blocks=True.

    Returns False (fail-open, no pin) when:
    - No remote resolve_and_lease or resolve_hashes has ever been
      received (no point in pinning blocks no remote worker will
      request).
    - Free primary blocks are at or below a safety floor.  The floor is
      ``max_blocks_per_seq`` — enough to admit one full-length request —
      so the GUARANTEED_NO_EVICT scheduler never starves.
    - Adding ``est_new_blocks`` would push the pin count past the
      effective budget (the lesser of ``pin_budget`` and
      ``total_primary_blocks // 4``).

    Args:
        free_blocks: current free primary blocks from kv_cache_manager.
        max_blocks_per_seq: blocks needed for one max-length request.
        pin_budget: hard cap on total outstanding prefill pins (env var).
        total_primary_blocks: total block count for auto-cap.  NOTE:
            currently sourced from get_kv_cache_stats().max_num_blocks
            which is the total across all pools (primary + secondary),
            not primary-only.  The auto-cap is therefore looser than
            intended; the hard budget (default 32) governs in practice.
            When >0, effective budget = min(pin_budget,
            total_primary_blocks // 4).  Pass 0 to skip auto-cap.
        est_new_blocks: estimated blocks this request will pin.  If
            current pins + est_new_blocks > effective budget, refuse.
    """
    if not _remote_resolve_seen:
        return False
    # Safety floor: keep enough free blocks for at least one max-len request.
    if free_blocks <= max_blocks_per_seq:
        return False
    # Auto-cap: total_primary_blocks // 4.  See docstring re: pool count caveat.
    effective_budget = pin_budget
    if total_primary_blocks > 0:
        effective_budget = min(pin_budget, total_primary_blocks // 4)
    with _prefill_pin_lock:
        current = len(_prefill_pinned)
        if current >= effective_budget:
            return False
        # Prevent overshoot: refuse if this request would blow the cap.
        if est_new_blocks > 0 and current + est_new_blocks > effective_budget:
            return False
    return True


_PREFILL_SWEEP_INTERVAL = 8  # sweep every N register_prefill_pins calls


def register_prefill_pins(block_ids) -> list[int]:
    """Record block IDs pinned by store_blocks_for_reuse for disagg prefill.

    Called from the executor thread after pinning blocks.  *block_ids* is the
    list returned by ``kv_cache_manager.store_blocks_for_reuse(req, True)``.

    Returns a (possibly empty) list of stale block IDs whose TTL has
    expired.  The caller **must** unpin them via
    ``kv_cache_manager.unpin_blocks_by_id(stale_ids)`` if non-empty.
    This ensures stale sweep runs from the executor thread even when no
    ``release_lease`` arrives.
    """
    global _prefill_pin_register_count
    now = _time.monotonic()
    stale: list[int] = []
    with _prefill_pin_lock:
        for bid in block_ids:
            _prefill_pinned[int(bid)] = now
        _prefill_pin_register_count += 1
        # Periodic inline sweep — avoids a separate timer thread.
        if _prefill_pin_register_count % _PREFILL_SWEEP_INTERVAL == 0:
            cutoff = now - _PREFILL_PIN_TTL_S
            expired = [b for b, ts in _prefill_pinned.items()
                       if ts < cutoff]
            for b in expired:
                del _prefill_pinned[b]
                stale.append(b)
    return stale


def pop_prefill_pin(block_id: int) -> bool:
    """Remove *block_id* from the prefill-pin tracker.  Return True if present."""
    with _prefill_pin_lock:
        return _prefill_pinned.pop(int(block_id), None) is not None


def has_prefill_pins() -> bool:
    """Return True if there are any outstanding prefill pins."""
    with _prefill_pin_lock:
        return bool(_prefill_pinned)


def sweep_stale_prefill_pins() -> list[int]:
    """Pop and return block IDs whose prefill pin has exceeded the TTL.

    Caller is responsible for unpinning the returned IDs via
    ``kv.unpin_blocks_by_id(stale_ids)``.
    """
    cutoff = _time.monotonic() - _PREFILL_PIN_TTL_S
    stale: list[int] = []
    with _prefill_pin_lock:
        to_remove = [
            bid for bid, ts in _prefill_pinned.items() if ts < cutoff
        ]
        for bid in to_remove:
            del _prefill_pinned[bid]
            stale.append(bid)
    return stale


def get_nixl_source_bundle() -> Optional[_NixlSourceBundle]:
    return _GLOBAL_NIXL_SOURCE_BUNDLE


def get_per_rank_nixl_bundles() -> Optional[list[dict]]:
    """Return the per-rank NIXL metadata list gathered at startup (S1).

    Each entry is a dict with keys: remote_name, agent_metadata_b64,
    pool_base_ptr, pool_size_bytes, source_generation, tp_rank.
    Returns None if the gather hasn't completed or TP=1 hasn't been
    initialized yet.
    """
    return _GLOBAL_PER_RANK_NIXL_BUNDLES


def _gather_per_rank_nixl_metadata(
    local_bundle: _NixlSourceBundle,
    tp_rank: int,
    tp_size: int,
) -> list[dict]:
    """S1: MPI allgather of per-rank NIXL agent metadata at startup.

    Every TP rank contributes its own (agent_name, agent_desc,
    pool_base_ptr, pool_size_bytes) and receives the full list.
    Uses MPI world communicator (safe when DP OFF, i.e. world = TP group).

    For DP ON deployments, this must be replaced with a TP-group-scoped
    communicator to avoid cross-DP-group contamination.
    """
    import base64 as _b64

    from tensorrt_llm._utils import mpi_allgather

    local_entry = {
        "tp_rank": tp_rank,
        "remote_name": local_bundle.remote_name,
        "agent_metadata_b64": _b64.b64encode(
            local_bundle.agent_desc
        ).decode("ascii"),
        "pool_base_ptr": local_bundle.pool_base_ptr,
        "pool_size_bytes": local_bundle.pool_size_bytes,
        "source_generation": local_bundle.source_generation,
    }

    if tp_size <= 1:
        return [local_entry]

    # MPI allgather — every rank gets the full list.
    all_entries = mpi_allgather(local_entry)
    # Sort by tp_rank so indexing is deterministic.
    all_entries.sort(key=lambda e: e["tp_rank"])

    logging.warning(
        "remote_g2: S1 per-rank NIXL metadata gathered: tp_size=%d "
        "ranks=[%s]",
        tp_size,
        ", ".join(
            f"{e['tp_rank']}:{e['remote_name']}" for e in all_entries
        ),
    )
    return all_entries


def _result_to_dict(result: Any) -> dict:
    """Convert a RemoteG2ResolveResult dataclass into a plain dict for
    wire transport. The dynamo parent and downstream consumers see only
    dicts and do not need to import RemoteG2ResolveResult / its nested
    types.
    """
    descriptors = [
        {
            "block_hash": d.block_hash,
            "descriptor_generation": d.descriptor_generation,
            "pool_id": d.pool_id,
            "byte_offset": d.byte_offset,
            "byte_length": d.byte_length,
            "metadata": dict(d.metadata or {}),
        }
        for d in (result.descriptors or ())
    ]
    per_block_status = [
        {
            "block_hash": s.block_hash,
            "status": s.status,
            "descriptor_generation": s.descriptor_generation,
        }
        for s in (result.per_block_status or ())
    ]
    out = {
        "lease_id": result.lease_id,
        "descriptors": descriptors,
        "num_tokens": result.num_tokens,
        "reason": result.reason,
        "source_generation": result.source_generation,
        "per_block_status": per_block_status,
    }
    # S4: Include per-rank data when available (TP>1).
    per_rank_descs = getattr(result, "per_rank_descriptors", {})
    if per_rank_descs:
        out["per_rank_descriptors"] = per_rank_descs
    per_rank_meta = getattr(result, "per_rank_source_metadata", {})
    if per_rank_meta:
        out["per_rank_source_metadata"] = per_rank_meta
    return out


def _ipc_socket_path(dynamo_pid: int, tp_rank: int = 0, tp_size: int = 1) -> str:
    """Return the ZMQ IPC socket path for a given TP rank.

    TP=1: /tmp/dynamo_remote_g2_ipc_{pid}.sock (backward-compatible)
    TP>1: /tmp/dynamo_remote_g2_ipc_{pid}_tp{rank}.sock (per-rank)
    """
    if tp_size <= 1:
        return f"/tmp/dynamo_remote_g2_ipc_{dynamo_pid}.sock"
    return f"/tmp/dynamo_remote_g2_ipc_{dynamo_pid}_tp{tp_rank}.sock"


def _query_sibling_rank(
    dynamo_pid: int, sibling_rank: int, tp_size: int, block_hashes: list[int],
    lease_id: str = "",
) -> list[dict]:
    """Query a sibling TP rank's ZMQ REP for descriptors via intra-pod IPC.

    Returns a list of descriptor dicts (one per block_hash, None entries
    for blocks not found on that rank). Sub-millisecond — Unix domain
    socket on the same pod.

    ``lease_id`` is passed through so the sibling can track pinned
    block_ids under the lease (Fix 2: lease-scoped sibling tracker).
    """
    import zmq

    sibling_path = _ipc_socket_path(dynamo_pid, sibling_rank, tp_size)
    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.RCVTIMEO = 5000
    req.SNDTIMEO = 5000
    try:
        req.connect(f"ipc://{sibling_path}")
        req.send(pickle.dumps({
            "method": "resolve_hashes",
            "payload": {
                "block_hashes": block_hashes,
                "lease_id": lease_id,
            },
        }))
        raw = req.recv()
        resp = pickle.loads(raw)
        if resp.get("ok"):
            return resp.get("result", [])
        logging.warning(
            "remote_g2: sibling rank %d resolve_hashes returned not-ok: %s",
            sibling_rank, resp.get("error"),
        )
        return []
    except Exception:
        logging.exception(
            "remote_g2: sibling rank %d IPC query failed (path=%s)",
            sibling_rank, sibling_path,
        )
        return []
    finally:
        req.close()


def _release_sibling_hashes(
    dynamo_pid: int, sibling_rank: int, tp_size: int, block_hashes: list[int],
    lease_id: str = "",
) -> int:
    """Send release_lease_pins to a sibling TP rank's ZMQ REP via intra-pod IPC.

    Returns the number of blocks successfully unpinned on the sibling.
    Fire-and-forget semantics: failures are logged but do not propagate.

    Fix 2: Uses lease_id for lease-scoped unpin when available, falling
    back to the legacy block_hashes path for backward compatibility.
    """
    import zmq

    sibling_path = _ipc_socket_path(dynamo_pid, sibling_rank, tp_size)
    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.RCVTIMEO = 5000
    req.SNDTIMEO = 5000
    try:
        req.connect(f"ipc://{sibling_path}")
        if lease_id:
            # Fix 2: lease-scoped release — sibling looks up pins by
            # lease_id instead of iterating block hashes.
            req.send(pickle.dumps({
                "method": "release_lease_pins",
                "payload": {"lease_id": lease_id},
            }))
        else:
            # Legacy path: release by block hashes.
            req.send(pickle.dumps({
                "method": "release_hashes",
                "payload": {"block_hashes": block_hashes},
            }))
        raw = req.recv()
        resp = pickle.loads(raw)
        if resp.get("ok"):
            return resp.get("result", 0)
        logging.warning(
            "remote_g2: sibling rank %d release returned not-ok: %s",
            sibling_rank, resp.get("error"),
        )
        return 0
    except Exception:
        logging.exception(
            "remote_g2: sibling rank %d release IPC failed (path=%s)",
            sibling_rank, sibling_path,
        )
        return 0
    finally:
        req.close()


def _start_zmq_rep_service(
    registry: SourceG2DescriptorRegistry,
    dynamo_pid: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> str:
    """Start a ZMQ REP daemon thread bound to a Unix domain socket.

    Every TP rank binds its own socket (per-rank path). Rank 0's handler
    serves external resolve RPCs AND queries sibling ranks via intra-pod
    ZMQ IPC. Non-rank-0 handlers only serve intra-pod queries
    (resolve_hashes method) from rank 0.

    Returns the socket path for logging.
    """
    import zmq

    socket_path = _ipc_socket_path(dynamo_pid, tp_rank, tp_size)
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.bind(f"ipc://{socket_path}")

    # Fix 2: Lease-scoped sibling pin tracker.  Keyed by
    # lease_id → list[block_id].  Each lease's pins are tracked
    # independently, eliminating hash-collision ambiguity between
    # concurrent resolves and avoiding the dual-hash mismatch
    # (identity hashes vs KV hashes) entirely.
    # Legacy block_hash tracker kept temporarily for backward compat.
    from collections import defaultdict
    sibling_pin_tracker_by_lease: dict[str, list[int]] = {}
    sibling_pin_tracker: dict[int, list[int]] = defaultdict(list)

    def _loop() -> None:
        while True:
            method = "<unparsed>"
            try:
                raw = rep.recv()
            except Exception:
                logging.exception("remote_g2: ZMQ REP recv failed; exiting loop")
                return
            try:
                req = pickle.loads(raw)
                method = req.get("method")
                payload = req.get("payload") or {}
                if method == "resolve_hashes":
                    # Fix 10b: sibling ranks also see remote resolve
                    # activity so executor-side prefill pinning is
                    # symmetric across TP ranks.
                    notify_remote_resolve_seen()
                    # Intra-pod query from rank 0: look up block hashes
                    # on THIS rank's registry and return descriptors.
                    # Uses the batch tier-aware lookup which reports
                    # primary-only blocks as CacheMiss(found_tier="primary").
                    from .remote_g2 import PinnedCacheBlock, CacheMiss
                    hashes = payload.get("block_hashes", [])
                    resolve_lease_id = str(payload.get("lease_id", ""))
                    lookup_results = registry._find_and_pin_blocks_by_hash(
                        tuple(int(h) for h in hashes)
                    )
                    descs = []
                    pinned_count = 0
                    primary_count = 0
                    missing_count = 0
                    # Fix 2: Collect block_ids pinned under this lease.
                    lease_pinned_block_ids: list[int] = []
                    # Collect indices of blocks that are in primary but
                    # not secondary — candidates for force-offload.
                    primary_only_indices = []
                    for idx, lr in enumerate(lookup_results):
                        if isinstance(lr, PinnedCacheBlock):
                            byte_offset = int(lr.slot_idx) * registry._block_size_bytes
                            descs.append({
                                "block_hash": lr.block_hash,
                                "byte_offset": byte_offset,
                                "byte_length": registry._block_size_bytes,
                                "pool_id": registry._pool_id,
                                "metadata": {
                                    "nixl_memory_desc": {
                                        "ptr": registry._pool_base_ptr + byte_offset,
                                        "len": registry._block_size_bytes,
                                    }
                                },
                            })
                            bid = int(lr.block_id)
                            # Fix 8: Only populate the legacy hash-based
                            # tracker when no lease_id is available.
                            # Otherwise the block_ids end up in BOTH
                            # trackers and release_lease_pins won't clean
                            # the legacy one — a later release_hashes
                            # could double-unpin.
                            if resolve_lease_id:
                                lease_pinned_block_ids.append(bid)
                            else:
                                sibling_pin_tracker[lr.block_hash].append(bid)
                            pinned_count += 1
                        else:
                            descs.append(None)
                            if isinstance(lr, CacheMiss) and lr.found_tier == "primary":
                                primary_only_indices.append(idx)
                                primary_count += 1
                            else:
                                missing_count += 1

                    # Force-offload blocks stuck in primary to secondary
                    # and pin them. This ensures TP>1 symmetric secondary
                    # availability for NIXL RDMA transfers.
                    if primary_only_indices and registry._window_size is not None:
                        force_hashes = [
                            int(hashes[i]) for i in primary_only_indices
                        ]
                        try:
                            force_results = registry._kv.force_offload_and_pin_blocks_by_hash(
                                force_hashes,
                                int(registry._window_size),
                            )
                            force_ok = 0
                            for fi, fr in zip(primary_only_indices, force_results):
                                if bool(fr.get("pinned", False)):
                                    slot_idx = int(fr["slot_idx"])
                                    byte_offset = slot_idx * registry._block_size_bytes
                                    descs[fi] = {
                                        "block_hash": int(fr["block_hash"]),
                                        "byte_offset": byte_offset,
                                        "byte_length": registry._block_size_bytes,
                                        "pool_id": registry._pool_id,
                                        "metadata": {
                                            "nixl_memory_desc": {
                                                "ptr": registry._pool_base_ptr + byte_offset,
                                                "len": registry._block_size_bytes,
                                            }
                                        },
                                    }
                                    bid = int(fr["block_id"])
                                    # Fix 8: same as above — only populate
                                    # one tracker to avoid double-unpin.
                                    if resolve_lease_id:
                                        lease_pinned_block_ids.append(bid)
                                    else:
                                        sibling_pin_tracker[int(fr["block_hash"])].append(bid)
                                    pinned_count += 1
                                    primary_count -= 1
                                    force_ok += 1
                            logging.info(
                                "remote_g2: resolve_hashes: tp_rank=%d "
                                "force_offload attempted=%d succeeded=%d",
                                tp_rank, len(force_hashes), force_ok,
                            )
                        except Exception as e:
                            logging.warning(
                                "remote_g2: resolve_hashes: tp_rank=%d "
                                "force_offload failed: %s",
                                tp_rank, e,
                            )
                    # Pad remaining hashes with None if stop_on_miss
                    # truncated the results.
                    while len(descs) < len(hashes):
                        descs.append(None)
                    # Fix 2: Store pinned block_ids under lease_id so
                    # release_lease_pins can unpin by lease, not by hash.
                    if resolve_lease_id and lease_pinned_block_ids:
                        sibling_pin_tracker_by_lease[resolve_lease_id] = (
                            list(lease_pinned_block_ids)
                        )
                    # Diagnostic: log secondary pool utilization to
                    # confirm whether blocks exist in secondary on
                    # this rank.
                    _sec_diag = ""
                    try:
                        _iter_stats = registry._kv.get_iteration_stats()
                        for _ws, _st in _iter_stats.items():
                            _sec_diag += (
                                f" ws={_ws}:sec_used={_st.secondary_used_num_blocks}"
                                f"/sec_free={_st.secondary_free_num_blocks}"
                                f"/sec_max={_st.secondary_max_num_blocks}"
                                f"/pri_used={_st.primary_used_num_blocks}"
                                f"/pri_free={_st.primary_free_num_blocks}"
                            )
                    except Exception as _e:
                        _sec_diag = f" (stats unavailable: {_e})"
                    logging.info(
                        "remote_g2: resolve_hashes: tp_rank=%d "
                        "hashes=%d pinned=%d in_primary=%d "
                        "missing=%d tracker_size=%d "
                        "lease_tracker_size=%d lease_id=%s%s",
                        tp_rank, len(hashes), pinned_count,
                        primary_count, missing_count,
                        sum(len(v) for v in sibling_pin_tracker.values()),
                        len(sibling_pin_tracker_by_lease),
                        resolve_lease_id or "(none)",
                        _sec_diag,
                    )
                    response = {"ok": True, "result": descs}

                elif method == "release_lease_pins":
                    # Fix 2: Lease-scoped unpin — unpin all block_ids
                    # tracked under the given lease_id.  Replaces the
                    # hash-based release_hashes for new resolves.
                    release_lid = str(payload.get("lease_id", ""))
                    block_ids = sibling_pin_tracker_by_lease.pop(
                        release_lid, []
                    )
                    unpinned = 0
                    prefill_unpinned = 0
                    for block_id in block_ids:
                        registry._release_pin_ref(block_id)
                        unpinned += 1
                        if pop_prefill_pin(block_id):
                            registry._release_pin_ref(block_id)
                            prefill_unpinned += 1
                    logging.info(
                        "remote_g2: release_lease_pins: tp_rank=%d "
                        "lease_id=%s unpinned=%d prefill=%d "
                        "remaining_leases=%d",
                        tp_rank, release_lid, unpinned,
                        prefill_unpinned,
                        len(sibling_pin_tracker_by_lease),
                    )
                    response = {"ok": True, "result": unpinned}

                elif method == "release_hashes":
                    # Legacy intra-pod unpin from rank 0: unpin blocks
                    # by block_hash.  Kept for backward compatibility
                    # with resolves that didn't pass lease_id.
                    hashes = payload.get("block_hashes", [])
                    logging.info(
                        "remote_g2: release_hashes: tp_rank=%d "
                        "incoming=%d tracker_keys=%d tracker_total=%d",
                        tp_rank, len(hashes),
                        len(sibling_pin_tracker),
                        sum(len(v) for v in sibling_pin_tracker.values()),
                    )
                    unpinned = 0
                    prefill_unpinned = 0
                    for bh in hashes:
                        ids = sibling_pin_tracker.get(bh)
                        if ids:
                            block_id = ids.pop(0)
                            registry._release_pin_ref(block_id)
                            unpinned += 1
                            if not ids:
                                del sibling_pin_tracker[bh]
                            # Also unpin the prefill-time
                            # store_blocks_for_reuse pin if present.
                            if pop_prefill_pin(block_id):
                                registry._release_pin_ref(block_id)
                                prefill_unpinned += 1
                    logging.info(
                        "remote_g2: release_hashes: unpinned %d/%d "
                        "blocks on tp_rank=%d (prefill_pins=%d)",
                        unpinned, len(hashes), tp_rank,
                        prefill_unpinned,
                    )
                    response = {"ok": True, "result": unpinned}

                elif method == "resolve_and_lease":
                    # Fix 10: signal that remote resolves are happening
                    # so the executor thread enables prefill pinning.
                    notify_remote_resolve_seen()
                    result = registry.resolve_and_lease(payload.get("plan"))
                    result_dict = _result_to_dict(result)

                    # Diagnostic: log secondary pool stats on rank 0
                    # for comparison with sibling ranks.
                    _r0_diag = ""
                    try:
                        _r0_stats = registry._kv.get_iteration_stats()
                        for _ws, _st in _r0_stats.items():
                            _r0_diag += (
                                f" ws={_ws}:sec_used={_st.secondary_used_num_blocks}"
                                f"/sec_free={_st.secondary_free_num_blocks}"
                                f"/sec_max={_st.secondary_max_num_blocks}"
                                f"/pri_used={_st.primary_used_num_blocks}"
                                f"/pri_free={_st.primary_free_num_blocks}"
                            )
                    except Exception as _e:
                        _r0_diag = f" (stats unavailable: {_e})"
                    logging.info(
                        "remote_g2: resolve_and_lease: tp_rank=%d "
                        "reason=%s n_descs=%d%s",
                        tp_rank, result.reason,
                        len(result.descriptors) if result.descriptors else 0,
                        _r0_diag,
                    )

                    # Intra-pod per-rank gather: query sibling ranks
                    # via ZMQ IPC (no MPI, no dynamo RPC).
                    if tp_size > 1 and result.reason == "ok" and result.descriptors:
                        # Fix 7: Use KV block hashes for sibling lookups,
                        # not identity hashes from descriptors.  The
                        # lease's block_hashes are the KV-trie hashes
                        # that rank 0 resolved; siblings need the same
                        # namespace for _find_and_pin_blocks_by_hash.
                        lease = registry.get_lease(result.lease_id)
                        if lease and lease.block_hashes:
                            block_hashes = list(lease.block_hashes)
                        else:
                            # Fallback: identity hashes (legacy plans
                            # where identity == kv hash).
                            block_hashes = [
                                d.block_hash for d in result.descriptors
                            ]
                        per_rank_descs = {tp_rank: [
                            {
                                "block_hash": d.block_hash,
                                "byte_offset": d.byte_offset,
                                "byte_length": d.byte_length,
                                "pool_id": d.pool_id,
                                "metadata": dict(d.metadata or {}),
                            }
                            for d in result.descriptors
                        ]}
                        # Query each sibling rank via local ZMQ IPC.
                        # Fix 2: pass lease_id so siblings track pins
                        # per-lease instead of per-hash.
                        resolve_lease_id = str(result.lease_id or "")
                        for sibling in range(tp_size):
                            if sibling == tp_rank:
                                continue
                            sibling_descs = _query_sibling_rank(
                                dynamo_pid, sibling, tp_size, block_hashes,
                                lease_id=resolve_lease_id,
                            )
                            if sibling_descs:
                                per_rank_descs[sibling] = sibling_descs
                        # Validate: every TP rank must have ALL blocks
                        # available in the secondary (host-pinned) tier.
                        # If any rank is missing blocks (None entries),
                        # the target would read from an empty secondary
                        # pool → data corruption.
                        missing_ranks = []
                        for rank_id in range(tp_size):
                            if rank_id not in per_rank_descs:
                                missing_ranks.append(
                                    (rank_id, "not_queried"))
                                continue
                            rd = per_rank_descs[rank_id]
                            none_indices = [
                                j for j, d in enumerate(rd)
                                if d is None
                            ]
                            if none_indices:
                                missing_ranks.append(
                                    (rank_id, f"blocks_not_in_secondary:"
                                     f"{none_indices[:8]}"))
                        if missing_ranks:
                            logging.warning(
                                "remote_g2: resolve_and_lease FAILED — "
                                "asymmetric secondary offload across TP "
                                "ranks. Ranks with missing blocks: %s. "
                                "Returning cache_miss to force fallback.",
                                missing_ranks,
                            )
                            # Fix 6: Rollback — release rank0 lease,
                            # prefill pins, and sibling pins acquired
                            # during the gather.  Without this, rank0
                            # refs and sibling refs leak on the
                            # cache_miss fallback path.
                            _rollback_prefill = 0
                            try:
                                # Grab pin refs before release consumes
                                # the lease (mirrors release_lease handler).
                                _rb_lease = registry.get_lease(
                                    result.lease_id)
                                _rb_pin_refs = (
                                    list(_rb_lease.trtllm_pin_refs)
                                    if _rb_lease is not None else [])
                                registry.release_lease(
                                    result.lease_id, "gather_rollback")
                                # Release prefill pins (same logic as
                                # the normal release_lease handler).
                                if _rb_pin_refs and has_prefill_pins():
                                    _rb_to_unpin = []
                                    for _bid in _rb_pin_refs:
                                        if pop_prefill_pin(int(_bid)):
                                            _rb_to_unpin.append(int(_bid))
                                    if _rb_to_unpin:
                                        registry._kv.unpin_blocks_by_id(
                                            _rb_to_unpin)
                                        _rollback_prefill = len(
                                            _rb_to_unpin)
                            except Exception:
                                logging.warning(
                                    "remote_g2: gather rollback: failed "
                                    "to release rank0 lease %s",
                                    result.lease_id, exc_info=True)
                            if resolve_lease_id:
                                for sibling in range(tp_size):
                                    if sibling == tp_rank:
                                        continue
                                    try:
                                        _release_sibling_hashes(
                                            dynamo_pid, sibling, tp_size,
                                            block_hashes,
                                            lease_id=resolve_lease_id,
                                        )
                                    except Exception:
                                        logging.warning(
                                            "remote_g2: gather rollback: "
                                            "failed to release sibling %d "
                                            "pins for lease %s",
                                            sibling, resolve_lease_id,
                                            exc_info=True)
                            logging.info(
                                "remote_g2: gather rollback: lease=%s "
                                "prefill_unpinned=%d siblings=%d",
                                result.lease_id, _rollback_prefill,
                                tp_size - 1,
                            )
                            # Override the result to signal cache miss
                            # so the target doesn't attempt a partial
                            # transfer with wrong offsets.  Clear
                            # lease_id/num_tokens so the target does not
                            # send a duplicate release for a lease that
                            # was already rolled back.
                            result_dict["reason"] = "cache_miss"
                            result_dict["lease_id"] = None
                            result_dict["num_tokens"] = 0
                            result_dict["descriptors"] = None
                            result_dict["per_rank_descriptors"] = None
                            response = {
                                "ok": True, "result": result_dict}
                            rep.send(pickle.dumps(response))
                            continue

                        result_dict["per_rank_descriptors"] = per_rank_descs
                        logging.info(
                            "remote_g2: intra-pod gather: %d ranks, "
                            "%d blocks each — all ranks validated",
                            len(per_rank_descs),
                            len(block_hashes),
                        )

                        # Attach per-rank source metadata from S1.
                        per_rank_bundles = get_per_rank_nixl_bundles()
                        if per_rank_bundles:
                            result_dict["per_rank_source_metadata"] = {
                                entry["tp_rank"]: entry
                                for entry in per_rank_bundles
                            }

                    response = {"ok": True, "result": result_dict}
                elif method == "release_lease":
                    lease_id = payload["lease_id"]
                    reason = payload.get("reason", "ack")
                    # Grab block_hashes and trtllm_pin_refs BEFORE
                    # release consumes the lease, so we can fan out
                    # unpin to sibling ranks and release prefill pins.
                    lease = registry.get_lease(lease_id)
                    block_hashes_to_release = (
                        list(lease.block_hashes)
                        if lease is not None else []
                    )
                    lease_pin_refs = (
                        list(lease.trtllm_pin_refs)
                        if lease is not None else []
                    )
                    completed = registry.release_lease(lease_id, reason)

                    # Unpin prefill-time store_blocks_for_reuse pins
                    # on rank 0.  The lease's trtllm_pin_refs contain
                    # the block_ids that were pinned during
                    # resolve_and_lease — these are the same physical
                    # blocks that store_blocks_for_reuse pinned at
                    # prefill termination.
                    prefill_unpinned = 0
                    if completed and lease_pin_refs and has_prefill_pins():
                        prefill_to_unpin = []
                        for bid in lease_pin_refs:
                            if pop_prefill_pin(int(bid)):
                                prefill_to_unpin.append(int(bid))
                        if prefill_to_unpin:
                            registry._kv.unpin_blocks_by_id(
                                prefill_to_unpin)
                            prefill_unpinned = len(prefill_to_unpin)

                    # Fan out unpin to sibling ranks (TP>1 only).
                    # Fix 2: pass lease_id so siblings can use the
                    # lease-scoped tracker (release_lease_pins method).
                    if completed and tp_size > 1:
                        for sibling in range(tp_size):
                            if sibling == tp_rank:
                                continue
                            _release_sibling_hashes(
                                dynamo_pid, sibling, tp_size,
                                block_hashes_to_release,
                                lease_id=lease_id,
                            )
                        logging.info(
                            "remote_g2: release fan-out: lease=%s, "
                            "%d hashes to %d siblings "
                            "(rank0_prefill_pins=%d)",
                            lease_id, len(block_hashes_to_release),
                            tp_size - 1, prefill_unpinned,
                        )
                    # Periodic sweep: unpin any prefill pins that
                    # have exceeded the TTL (decode worker may have
                    # crashed or timed out without sending release).
                    stale_ids = sweep_stale_prefill_pins()
                    if stale_ids:
                        try:
                            registry._kv.unpin_blocks_by_id(stale_ids)
                            logging.warning(
                                "remote_g2: swept %d stale prefill "
                                "pins (ttl=%.0fs) on tp_rank=%d",
                                len(stale_ids),
                                _PREFILL_PIN_TTL_S, tp_rank,
                            )
                        except Exception:
                            logging.exception(
                                "remote_g2: stale prefill pin "
                                "sweep unpin failed")

                    response = {"ok": True, "result": completed}
                elif method == "get_metadata":
                    bundle = get_nixl_source_bundle()
                    per_rank = get_per_rank_nixl_bundles()
                    if bundle is None:
                        response = {
                            "ok": False,
                            "error": "nixl_source_bundle_not_ready",
                        }
                    else:
                        # Bidirectional peer load (raw NIXL): if the
                        # caller sent its serialized agent metadata,
                        # call add_remote_agent on it BEFORE returning
                        # our own metadata.
                        import base64 as _b64
                        peer_metadata_b64 = payload.get("peer_metadata_b64")
                        if peer_metadata_b64:
                            try:
                                peer_bytes = _b64.b64decode(peer_metadata_b64)
                                loaded_name = bundle.agent.add_remote_agent(peer_bytes)
                                logging.warning(
                                    "remote_g2: source add_remote_agent "
                                    "loaded peer name=%s (bytes=%d)",
                                    loaded_name, len(peer_bytes),
                                )
                            except Exception:
                                logging.exception(
                                    "remote_g2: source add_remote_agent failed"
                                )
                        # Rank 0's own metadata (backward-compatible with
                        # TP=1 callers that don't read per_rank_metadata).
                        result = {
                            "source_worker_id": registry.source_worker_id,
                            "source_dp_rank": registry.source_dp_rank,
                            "source_generation": bundle.source_generation,
                            "remote_name": bundle.remote_name,
                            "agent_metadata_b64": _b64.b64encode(
                                bundle.agent_desc
                            ).decode("ascii"),
                            "pool_base_ptr": bundle.pool_base_ptr,
                            "pool_size_bytes": bundle.pool_size_bytes,
                        }
                        # S5 — per-rank metadata for TP>1 targets.
                        if per_rank is not None:
                            result["per_rank_metadata"] = per_rank
                        response = {"ok": True, "result": result}
                else:
                    response = {"ok": False, "error": f"unknown method: {method!r}"}
            except Exception as exc:
                logging.exception("remote_g2: ZMQ REP handler raised")
                response = {"ok": False, "error": repr(exc)}
            try:
                rep.send(pickle.dumps(response))
            except Exception:
                logging.exception("remote_g2: ZMQ REP send failed")

    thread = threading.Thread(target=_loop, name="remote_g2_zmq_rep", daemon=True)
    thread.start()
    return socket_path


def _walk_to_dynamo_worker_pid(max_depth: int = 10) -> Optional[int]:
    """Walk up the process tree from this process and return the first
    ancestor whose cmdline mentions 'dynamo.trtllm'. OpenMPI's orted
    strips arbitrary env vars when spawning ranks, so the engine
    subprocess can't read DYNAMO_REMOTE_G2_WORKER_ID directly; this
    helper finds the dynamo parent so we can read a sidecar file
    /tmp/dynamo_remote_g2_worker_<pid>.txt instead.

    When TP=1 (no MPI spawn), the engine runs inline in the dynamo
    process itself — there is no parent to walk to. In that case we
    check if the current process IS the dynamo process and return our
    own PID.
    """
    try:
        my_pid = os.getpid()
        # TP=1 fast path: check if *this* process is the dynamo worker
        # (no MPI subprocess when tensor_parallel_size == 1).
        try:
            with open(f"/proc/{my_pid}/cmdline") as f:
                my_cmdline = f.read().replace("\0", " ")
        except (FileNotFoundError, PermissionError):
            my_cmdline = ""
        if "dynamo.trtllm" in my_cmdline or "dynamo/trtllm" in my_cmdline:
            return my_pid

        # TP>1 path: walk ancestors to find the dynamo parent.
        pid = my_pid
        for _ in range(max_depth):
            try:
                with open(f"/proc/{pid}/status") as f:
                    status = f.read()
            except (FileNotFoundError, PermissionError):
                return None
            ppid = None
            for line in status.splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                    break
            if ppid is None or ppid < 1:
                return None
            try:
                with open(f"/proc/{ppid}/cmdline") as f:
                    cmdline = f.read().replace("\0", " ")
            except (FileNotFoundError, PermissionError):
                cmdline = ""
            if "dynamo.trtllm" in cmdline or "dynamo/trtllm" in cmdline:
                return ppid
            pid = ppid
        return None
    except Exception:
        return None


def _resolve_source_identity() -> Optional[tuple[int, int]]:
    """Return (source_worker_id, dynamo_parent_pid) on success, None when
    the dynamo identity cannot be reached from the engine subprocess.

    Reads the worker_id from env var first, falling back to a sidecar
    file written by the dynamo parent (since OpenMPI orted strips env
    vars across the spawn boundary). The dynamo parent's PID is also
    needed so the ZMQ REP service can bind to a Unix domain socket the
    parent can find.
    """
    dynamo_pid = _walk_to_dynamo_worker_pid()
    env_value = os.environ.get("DYNAMO_REMOTE_G2_WORKER_ID")
    if env_value and dynamo_pid is not None:
        try:
            return int(env_value), dynamo_pid
        except ValueError:
            pass
    if dynamo_pid is None:
        return None
    sidecar = f"/tmp/dynamo_remote_g2_worker_{dynamo_pid}.txt"
    try:
        with open(sidecar) as f:
            worker_id = int(f.read().strip())
        return worker_id, dynamo_pid
    except Exception:
        return None


def _get_secondary_pool(kv: Any) -> Any:
    """Return the unsliced secondary pool tensor, falling back to
    get_secondary_pool_data(0) on older TRT-LLM builds that lack
    get_unique_secondary_pool().
    """
    if hasattr(kv, "get_unique_secondary_pool"):
        return kv.get_unique_secondary_pool()
    # Fallback: layer-0 slice — data_ptr coincides with allocation base
    # under block-major layout (the standard for transformer models).
    return kv.get_secondary_pool_data(0)


def _secondary_pool_base_ptr(kv: Any) -> int:
    """Return the secondary KV cache pool's base host address, or 0 when
    not available (no host pool allocated, exposure binding missing, etc.).
    """
    try:
        pool = _get_secondary_pool(kv)
        if pool is None or pool.numel() == 0:
            return 0
        return int(pool.data_ptr())
    except Exception as exc:
        logging.warning("remote_g2: _get_secondary_pool raised: %r", exc)
        return 0


def _derive_block_size_bytes(kv: Any) -> Optional[int]:
    """Compute per-LOGICAL-block byte size from the unsliced secondary
    pool tensor (parity with how the source-side NIXL agent registers
    its memory).

    Uses ``get_unique_secondary_pool()`` — the unsliced full secondary
    pool with shape ``(num_blocks, num_layers, kv_factor, blockSize)``
    for block-major layouts. One logical block occupies
    ``num_layers × kv_factor × blockSize × element_size`` bytes, which
    is the same as ``element_size × prod(shape[1:])``.

    Layout assumption: standard transformers run block-major. For
    recurrent-state / linear-attention models the first dim is
    num_layers instead of num_blocks and this derivation would be
    wrong; the modulo sanity check at the end catches that case as a
    fail-loud rather than a silent corruption.
    """
    try:
        pool = _get_secondary_pool(kv)
    except Exception:
        return None
    if pool is None or pool.numel() == 0 or pool.ndim < 2:
        return None

    used_fallback = not hasattr(kv, "get_unique_secondary_pool")
    per_block_elems = 1
    for d in pool.shape[1:]:
        per_block_elems *= int(d)
    if per_block_elems <= 0:
        return None

    per_block_bytes = int(pool.element_size()) * per_block_elems

    # When using get_secondary_pool_data(0) fallback, the tensor is a
    # per-layer slice — multiply by num_pools (== num_layers) to get
    # the full logical block size across all layers.
    if used_fallback:
        try:
            num_pools = int(kv.num_pools)
        except Exception:
            return None
        per_block_bytes *= num_pools

    total_bytes = int(pool.element_size()) * int(pool.numel())
    if used_fallback:
        total_bytes *= num_pools
    if total_bytes % per_block_bytes != 0:
        return None

    return per_block_bytes


def _derive_window_size(kv: Any) -> Optional[int]:
    """Return the attention window size the KV cache manager is configured
    with. Reads from KvCacheIterationStats keys; under the single-window-
    block-manager constraint this connector operates under there is
    exactly one entry."""
    try:
        iter_stats = kv.get_iteration_stats()
    except Exception:
        return None
    if not iter_stats:
        return None
    try:
        return int(next(iter(iter_stats.keys())))
    except (StopIteration, TypeError, ValueError):
        return None


def _pool_size_bytes(kv: Any) -> Optional[int]:
    """Total byte size of the host_pinned secondary pool covering all
    layers. We register this whole range with the NIXL agent so remote
    workers can issue READs against any slot in it.

    Uses ``get_unique_secondary_pool()`` — the unsliced full secondary
    pool tensor — so the total size is the direct
    ``element_size × numel`` of the underlying allocation, without
    needing to multiply by num_layers manually.
    """
    try:
        pool = _get_secondary_pool(kv)
    except Exception:
        return None
    if pool is None or pool.numel() == 0:
        return None
    size = int(pool.element_size() * pool.numel())
    # Fallback pool is per-layer; multiply by num_pools for full size.
    if not hasattr(kv, "get_unique_secondary_pool"):
        try:
            size *= int(kv.num_pools)
        except Exception:
            return None
    return size


def _setup_nixl_source_agent(
    pool_base_ptr: int,
    pool_size_bytes: int,
    source_worker_id: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> Optional[_NixlSourceBundle]:
    """Build a raw nixl_agent on the source side and register the
    host_pinned secondary pool memory range so it can be read remotely.

    With TP>1, each rank builds its own agent with a rank-qualified
    name so peers can load multiple source agents (one per TP rank).
    """
    if tp_size > 1:
        agent_name = f"remote-g2-source-{source_worker_id}-tp{tp_rank}"
    else:
        agent_name = f"remote-g2-source-{source_worker_id}"
    from .remote_g2_raw_nixl_adapter import build_raw_nixl_source_agent

    handle = build_raw_nixl_source_agent(
        agent_name=agent_name,
        pool_base_ptr=pool_base_ptr,
        pool_size_bytes=pool_size_bytes,
    )
    if handle is None:
        return None
    # build_raw_nixl_source_agent has already constructed the agent,
    # registered the host_pinned pool, and captured the agent metadata
    # bytes. Mirror those values into _NixlSourceBundle so the existing
    # metadata RPC handler can read them via the same field names it
    # used to use with the TRT-LLM wrapper.
    return _NixlSourceBundle(
        agent=handle.agent,
        remote_name=handle.agent_name,
        agent_desc=handle.agent_metadata,
        pool_base_ptr=handle.pool_base_ptr,
        pool_size_bytes=handle.pool_size_bytes,
        source_generation=handle.source_generation,
    )


# ─── S2: Per-rank descriptor gather on resolve ─────────────────────
#
# When TP>1, each rank holds its own slice of every KV block. On a
# resolve RPC, rank 0 must gather per-rank descriptors from all TP
# siblings so the target can issue per-rank NIXL READs.

# Sentinel values for the sibling resolve loop (S3).
_SIBLING_RESOLVE_TAG = "__remote_g2_resolve_hashes__"
_SIBLING_SHUTDOWN_TAG = "__remote_g2_shutdown__"


def _gather_per_rank_descriptors(
    block_hashes: list[int],
    local_registry: "SourceG2DescriptorRegistry",
    tp_size: int,
) -> dict[int, list[dict]]:
    """S2: Rank 0 broadcasts block hashes, all ranks do local lookup,
    rank 0 gathers per-rank descriptor lists.

    Returns {tp_rank: [descriptor_dict, ...]} where each descriptor_dict
    has keys matching the flat descriptor format (byte_offset, byte_length,
    pool_id, block_hash, etc.).

    For TP=1, returns {0: [local descriptors]}.
    """
    from tensorrt_llm._utils import mpi_allgather, mpi_broadcast

    # Broadcast hashes from rank 0 to all ranks.
    hashes = mpi_broadcast(block_hashes, root=0)

    # Each rank resolves locally using its own registry. We use the
    # internal find-and-pin path to get descriptors without creating
    # a lease (the lease is managed by rank 0's resolve_and_lease).
    local_descs = []
    for bh in hashes:
        record = local_registry._lookup_via_find_block_by_hash(bh)
        if record is not None:
            local_descs.append({
                "block_hash": record.block_hash,
                "byte_offset": record.byte_offset,
                "byte_length": record.byte_length,
                "pool_id": record.pool_id,
                "metadata": record.metadata,
            })
        else:
            # Block not found on this rank — signal with None.
            local_descs.append(None)

    # Gather from all ranks. Each rank contributes its descriptor list.
    from tensorrt_llm._utils import mpi_rank
    all_rank_descs = mpi_allgather({"tp_rank": mpi_rank(), "descs": local_descs})

    # Assemble into {rank: descs} dict.
    per_rank = {}
    for entry in all_rank_descs:
        per_rank[entry["tp_rank"]] = entry["descs"]

    return per_rank


def _start_sibling_rank_resolve_loop(
    kv: Any,
    registry: "SourceG2DescriptorRegistry",
    tp_rank: int,
    tp_size: int,
) -> None:
    """S3: Non-rank-0 TP siblings run a background thread that
    participates in MPI collectives during resolve RPCs.

    The loop blocks on mpi_broadcast (waiting for rank 0 to send
    block hashes), does local findAndPinSecondaryBlockByHash, then
    participates in the mpi_allgather back to rank 0.

    Runs as a daemon thread so it doesn't block process shutdown.
    """

    def _loop():
        from tensorrt_llm._utils import mpi_allgather, mpi_broadcast, mpi_rank

        logging.info(
            "remote_g2: S3 sibling resolve loop started on tp_rank=%d",
            tp_rank,
        )
        while True:
            try:
                # Block until rank 0 broadcasts hashes (or shutdown).
                hashes = mpi_broadcast(None, root=0)

                if hashes == _SIBLING_SHUTDOWN_TAG:
                    logging.info(
                        "remote_g2: S3 sibling loop shutdown on tp_rank=%d",
                        tp_rank,
                    )
                    return

                # Local resolve for each hash.
                local_descs = []
                for bh in hashes:
                    record = registry._lookup_via_find_block_by_hash(bh)
                    if record is not None:
                        local_descs.append({
                            "block_hash": record.block_hash,
                            "byte_offset": record.byte_offset,
                            "byte_length": record.byte_length,
                            "pool_id": record.pool_id,
                            "metadata": record.metadata,
                        })
                    else:
                        local_descs.append(None)

                # Participate in the gather back to rank 0.
                mpi_allgather({
                    "tp_rank": mpi_rank(),
                    "descs": local_descs,
                })

            except Exception:
                logging.exception(
                    "remote_g2: S3 sibling resolve loop error on tp_rank=%d",
                    tp_rank,
                )
                # Don't exit on transient errors — keep looping.

    t = threading.Thread(target=_loop, daemon=True, name=f"g2-sibling-{tp_rank}")
    t.start()


def maybe_start_remote_g2_service(
    kv: Any,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    lease_ttl_ms: int = 30_000,
    pool_id: str = "g2-host-pinned",
    tier: str = "host_pinned",
) -> Optional[SourceG2DescriptorRegistry]:
    """Start the source-side remote-G2 service against a live kv_cache_manager.

    Called from PyExecutor right after kv_cache_manager is constructed,
    inside the engine subprocess. Builds a SourceG2DescriptorRegistry
    and (in future iterations) spawns a daemon thread that exposes it
    over ZMQ for the dynamo parent process to forward RPC calls into.

    Returns None when the deployment isn't configured for remote-G2
    (env var missing), or when prerequisites aren't met (no secondary
    pool, no host_pinned blocks yet, etc.). Caller treats None as
    "remote-G2 service not started".

    Identity (source_worker_id, source_dp_rank) is read from env vars
    set by the dynamo parent process:
      - DYNAMO_REMOTE_G2_WORKER_ID  (required; matches dynamo
        endpoint.connection_id() for the owning dynamo worker process)
      - DYNAMO_REMOTE_G2_DP_RANK    (defaults to 0)

    TP>1 support (S1-S3):
      - tp_rank / tp_size control MPI gather of per-rank NIXL metadata.
      - Rank 0 runs the ZMQ REP server and answers resolve RPCs.
      - Non-rank-0 ranks participate in MPI gather during resolve (S3)
        but do NOT run a ZMQ server.
    """
    # Auto-detect tp_rank/tp_size from MPI when not passed explicitly.
    # This handles deployments where py_executor.py doesn't pass the
    # TP info (e.g. patched connectors without patched py_executor).
    if tp_rank == 0 and tp_size == 1:
        try:
            from tensorrt_llm._utils import mpi_rank, mpi_world_size
            detected_rank = mpi_rank()
            detected_size = mpi_world_size()
            if detected_size > 1:
                tp_rank = detected_rank
                tp_size = detected_size
                logging.warning(
                    "remote_g2: auto-detected TP from MPI: "
                    "tp_rank=%d tp_size=%d",
                    tp_rank, tp_size,
                )
        except Exception:
            pass

    identity = _resolve_source_identity()
    if identity is None:
        logging.info(
            "remote_g2: source registry skipped "
            "(DYNAMO_REMOTE_G2_WORKER_ID not reachable via env var or sidecar)"
        )
        return None
    source_worker_id, dynamo_pid = identity

    # PyExecutor.kv_cache_manager is a Python wrapper class
    # (resource_manager.KVCacheManager); the C++ binding with
    # get_secondary_pool_data / find_and_pin_blocks_by_hash / pin_blocks_by_id
    # sits at .impl. Unwrap once so the rest of the code (and the
    # SourceG2DescriptorRegistry it builds) talks to the C++ object
    # directly.
    kv = getattr(kv, "impl", kv)
    try:
        source_dp_rank = int(os.environ.get("DYNAMO_REMOTE_G2_DP_RANK", "0"))
    except ValueError:
        source_dp_rank = 0

    pool_base_ptr = _secondary_pool_base_ptr(kv)
    if pool_base_ptr == 0:
        logging.info(
            "remote_g2: source registry skipped (secondary pool unavailable)"
        )
        return None

    block_size_bytes = _derive_block_size_bytes(kv)
    if not block_size_bytes or block_size_bytes <= 0:
        logging.warning(
            "remote_g2: source registry skipped (block_size_bytes unknown)"
        )
        return None

    window_size = _derive_window_size(kv)
    if window_size is None or window_size <= 0:
        logging.warning(
            "remote_g2: source registry skipped (window_size unknown)"
        )
        return None

    acquire_pin, release_pin = make_kv_pin_callbacks(
        kv,
        secondary_pool_base_ptr=pool_base_ptr,
        block_size_bytes=block_size_bytes,
    )

    registry = SourceG2DescriptorRegistry(
        source_worker_id=source_worker_id,
        source_dp_rank=source_dp_rank,
        lease_ttl_ms=lease_ttl_ms,
        acquire_pin=acquire_pin,
        release_pin=release_pin,
        require_trtllm_pin=True,
        kv=kv,
        window_size=window_size,
        pool_id=pool_id,
        pool_base_ptr=pool_base_ptr,
        block_size_bytes=block_size_bytes,
        tier=tier,
    )

    logging.warning(
        "remote_g2: service started "
        "(source_worker_id=%s dp_rank=%s window_size=%s "
        "block_size_bytes=%s pool_base_ptr=0x%x)",
        source_worker_id,
        source_dp_rank,
        window_size,
        block_size_bytes,
        pool_base_ptr,
    )

    # ZMQ REP server — every rank binds its own socket for intra-pod
    # IPC queries. Rank 0 serves external resolve RPCs from the dynamo
    # parent AND queries sibling ranks via their sockets. Non-rank-0
    # ranks serve only intra-pod resolve_hashes queries from rank 0.
    try:
        socket_path = _start_zmq_rep_service(
            registry, dynamo_pid,
            tp_rank=tp_rank, tp_size=tp_size,
        )
        logging.warning(
            "remote_g2: ZMQ REP service bound at %s "
            "(source_worker_id=%s tp_rank=%d tp_size=%d)",
            socket_path,
            source_worker_id,
            tp_rank,
            tp_size,
        )
    except Exception:
        logging.exception(
            "remote_g2: failed to start ZMQ REP service; registry built but "
            "not reachable"
        )

    # Stage T1 — bootstrap the NIXL agent and register the host_pinned
    # secondary pool. Each TP rank builds its own agent with a rank-
    # qualified name. The bundle is stashed as a process-wide singleton
    # so the ZMQ REP service can answer the get_metadata RPC (Stage T2).
    pool_size_bytes = _pool_size_bytes(kv)
    if not pool_size_bytes or pool_size_bytes <= 0:
        logging.warning(
            "remote_g2: NIXL agent skipped (pool_size_bytes unknown)"
        )
    else:
        bundle = _setup_nixl_source_agent(
            pool_base_ptr=pool_base_ptr,
            pool_size_bytes=pool_size_bytes,
            source_worker_id=source_worker_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if bundle is not None:
            global _GLOBAL_NIXL_SOURCE_BUNDLE
            _GLOBAL_NIXL_SOURCE_BUNDLE = bundle
            logging.warning(
                "remote_g2: source NIXL agent built: agent_name=%s "
                "pool_base_ptr=0x%x pool_size=%d tp_rank=%d",
                bundle.remote_name,
                bundle.pool_base_ptr,
                bundle.pool_size_bytes,
                tp_rank,
            )

            # S1 — gather per-rank NIXL metadata from all TP siblings.
            # Every rank must participate (MPI allgather is collective).
            try:
                global _GLOBAL_PER_RANK_NIXL_BUNDLES
                _GLOBAL_PER_RANK_NIXL_BUNDLES = (
                    _gather_per_rank_nixl_metadata(
                        bundle, tp_rank=tp_rank, tp_size=tp_size,
                    )
                )
            except Exception:
                logging.exception(
                    "remote_g2: S1 per-rank metadata gather failed"
                )
        else:
            logging.warning(
                "remote_g2: NIXL source agent bootstrap failed; "
                "resolve will still work, but transfer is disabled"
            )

    # Per-rank resolve is now handled via intra-pod ZMQ IPC (no MPI).
    # Each rank's ZMQ REP serves resolve_hashes queries from rank 0.
    # No S3 sibling loop needed.

    return registry
