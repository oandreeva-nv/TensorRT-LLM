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


def get_nixl_source_bundle() -> Optional[_NixlSourceBundle]:
    return _GLOBAL_NIXL_SOURCE_BUNDLE


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
    return {
        "lease_id": result.lease_id,
        "descriptors": descriptors,
        "num_tokens": result.num_tokens,
        "reason": result.reason,
        "source_generation": result.source_generation,
        "per_block_status": per_block_status,
    }


def _start_zmq_rep_service(registry: SourceG2DescriptorRegistry, dynamo_pid: int) -> str:
    """Start a ZMQ REP daemon thread bound to a Unix domain socket
    tagged with the dynamo parent's PID. The dynamo parent process
    constructs the matching REQ client using the same path. Returns
    the socket path for logging.

    Wire format: pickle-encoded {"method": <name>, "payload": <dict>}
    request; pickle-encoded {"ok": bool, "result"|"error": <value>}
    response. Pickle is safe here since both ends are colocated Python
    processes on the same host.
    """
    import zmq  # imported lazily so this module stays importable on hosts without zmq

    socket_path = f"/tmp/dynamo_remote_g2_ipc_{dynamo_pid}.sock"
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.bind(f"ipc://{socket_path}")

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
                if method == "resolve_and_lease":
                    result = registry.resolve_and_lease(payload.get("plan"))
                    response = {"ok": True, "result": _result_to_dict(result)}
                elif method == "release_lease":
                    completed = registry.release_lease(
                        payload["lease_id"], payload.get("reason", "ack")
                    )
                    response = {"ok": True, "result": completed}
                elif method == "get_metadata":
                    bundle = get_nixl_source_bundle()
                    if bundle is None:
                        response = {
                            "ok": False,
                            "error": "nixl_source_bundle_not_ready",
                        }
                    else:
                        # Bidirectional peer load (raw NIXL): if the
                        # caller sent its serialized agent metadata,
                        # call add_remote_agent on it BEFORE returning
                        # our own metadata. This is the equivalent of
                        # the old load_remote_agent_by_connection
                        # handshake but uses the bytes blob NIXL's raw
                        # API expects.
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
                        response = {
                            "ok": True,
                            "result": {
                                "source_worker_id": registry.source_worker_id,
                                "source_dp_rank": registry.source_dp_rank,
                                "source_generation": bundle.source_generation,
                                "remote_name": bundle.remote_name,
                                # Raw NIXL agent metadata bytes — target
                                # passes this to its agent.add_remote_agent
                                # and uses the rkeys it contains for the
                                # prepped-flow dlist.
                                "agent_metadata_b64": _b64.b64encode(
                                    bundle.agent_desc
                                ).decode("ascii"),
                                # Pool extent so target can build a
                                # block-indexed remote dlist that lines
                                # up with the source's registered region.
                                "pool_base_ptr": bundle.pool_base_ptr,
                                "pool_size_bytes": bundle.pool_size_bytes,
                            },
                        }
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
            logging.warning(
                "PROBE _walk_to_dynamo_worker_pid: current process IS dynamo "
                "(TP=1 inline mode) pid=%d", my_pid,
            )
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
                logging.warning(
                    "PROBE _walk_to_dynamo_worker_pid: found dynamo parent "
                    "ppid=%d from pid=%d", ppid, pid,
                )
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
        logging.warning(
            "PROBE _resolve_source_identity: env worker_id=%s dynamo_pid=%d",
            env_value, dynamo_pid,
        )
        try:
            return int(env_value), dynamo_pid
        except ValueError:
            pass
    if dynamo_pid is None:
        logging.warning(
            "PROBE _resolve_source_identity: dynamo_pid is None — "
            "source registry will be skipped (need SYS_PTRACE + runAsUser:0)",
        )
        return None
    sidecar = f"/tmp/dynamo_remote_g2_worker_{dynamo_pid}.txt"
    try:
        with open(sidecar) as f:
            worker_id = int(f.read().strip())
        logging.warning(
            "PROBE _resolve_source_identity: sidecar worker_id=%d dynamo_pid=%d",
            worker_id, dynamo_pid,
        )
        return worker_id, dynamo_pid
    except Exception as exc:
        logging.warning(
            "PROBE _resolve_source_identity: sidecar read failed %r", exc,
        )
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
) -> Optional[_NixlSourceBundle]:
    """Build a raw nixl_agent on the source side and register the
    host_pinned secondary pool memory range so it can be read remotely.

    Switched from TRT-LLM's BindingsNixlTransferAgent wrapper to raw
    `nixl.nixl_agent` because the wrapper exposes only the combined
    transfer flow (createXferReq → NIXL_ERR_NOT_FOUND for cross-process
    READ). The prepped flow needed for proper rkey exchange requires
    raw NIXL API surface.
    """
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


def maybe_start_remote_g2_service(
    kv: Any,
    *,
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

    try:
        socket_path = _start_zmq_rep_service(registry, dynamo_pid)
        logging.warning(
            "remote_g2: ZMQ REP service bound at %s (source_worker_id=%s)",
            socket_path,
            source_worker_id,
        )
    except Exception:
        logging.exception(
            "remote_g2: failed to start ZMQ REP service; registry built but "
            "not reachable from dynamo parent"
        )

    # Stage T1 — bootstrap the NIXL agent and register the host_pinned
    # secondary pool. The bundle is stashed as a process-wide singleton
    # so the ZMQ REP service can answer the get_metadata RPC (Stage T2)
    # without threading the bundle through the registry constructor.
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
        )
        if bundle is not None:
            global _GLOBAL_NIXL_SOURCE_BUNDLE
            _GLOBAL_NIXL_SOURCE_BUNDLE = bundle
            logging.warning(
                "PROBE remote_g2_source_nixl: agent_name=%s pool_base_ptr=0x%x "
                "pool_size=%d agent_desc_bytes=%d source_generation=%d",
                bundle.remote_name,
                bundle.pool_base_ptr,
                bundle.pool_size_bytes,
                len(bundle.agent_desc),
                bundle.source_generation,
            )
        else:
            logging.warning(
                "remote_g2: NIXL source agent bootstrap failed; "
                "resolve will still work, but transfer is disabled"
            )

    return registry
