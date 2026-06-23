# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap the target-side RPC bridge from inside the engine subprocess.

Mirror of remote_g2_source_setup.py for the other direction: the connector
scheduler runs in the engine subprocess, but the dynamo runtime client
that knows how to reach the source worker lives in the dynamo parent.
This module opens a ZMQ REQ socket connecting to the parent's local REP
loop and installs sync callables into the connector's module state so
the scheduler picks them up at call time.

Wire format mirrors the source side: pickle-encoded request/response,
``{"method": ..., "payload": ...}`` ⇄ ``{"ok": bool, "result"|"error": ...}``.

Note on installation order: PyExecutor constructs the connector
scheduler *before* this bootstrap runs (see py_executor_creator.py
where ``scheduler_cls(llm_args)`` is invoked). The scheduler stashes
``self._resolve_and_lease = None`` in that case; ``get_num_new_matched_tokens``
falls back to reading ``remote_g2_connector._installed_resolve_and_lease``
at call time, so installation can happen later.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pickle
import threading
import time
from typing import Any, Mapping, Optional, Sequence

from . import remote_g2_connector
from .remote_g2 import (
    RemoteG2BindingRecord,
    RemoteG2BlockStatus,
    RemoteG2Descriptor,
    RemoteG2ResolveResult,
    RemoteKvReusePlan,
)
from .remote_g2_source_setup import (
    _derive_block_size_bytes,
    _derive_window_size,
    _resolve_source_identity,
    _walk_to_dynamo_worker_pid,
)


class _TargetReqWrapper:
    """Thread-safe ZMQ REQ client to the dynamo parent's local REP bridge.

    REQ/REP is strict send-then-recv on a single socket, so concurrent
    callers serialize on the lock. The connector scheduler already runs
    in TP rank 0 only, so contention is minimal — at most one in-flight
    RPC per scheduler tick.
    """

    def __init__(self, socket_path: str, timeout_ms: int = 30_000) -> None:
        import zmq

        self._socket_path = socket_path
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.RCVTIMEO = timeout_ms
        self._socket.SNDTIMEO = timeout_ms
        self._socket.connect(f"ipc://{socket_path}")
        self._lock = threading.Lock()

    def request(self, method: str, payload: dict) -> dict:
        with self._lock:
            self._socket.send(pickle.dumps({"method": method, "payload": payload}))
            raw = self._socket.recv()
        return pickle.loads(raw)


def _plan_to_dict(plan: RemoteKvReusePlan) -> dict:
    """Convert a plan dataclass to a dict for wire transport. The parent
    REP loop forwards the dict to client.direct() as-is."""
    return dataclasses.asdict(plan)


def _result_from_dict(data: dict) -> Optional[RemoteG2ResolveResult]:
    """Reconstruct RemoteG2ResolveResult from the source's dict response.

    Returns None when the response shape is malformed. Caller treats
    that as "plan not resolvable" — same as a transport failure.
    """
    if not isinstance(data, dict):
        return None
    try:
        descriptors = tuple(
            RemoteG2Descriptor(
                block_hash=int(d["block_hash"]),
                descriptor_generation=int(d["descriptor_generation"]),
                pool_id=str(d["pool_id"]),
                byte_offset=int(d["byte_offset"]),
                byte_length=int(d["byte_length"]),
                metadata=dict(d.get("metadata") or {}),
            )
            for d in (data.get("descriptors") or ())
        )
        per_block_status = tuple(
            RemoteG2BlockStatus(
                block_hash=int(s["block_hash"]),
                status=str(s["status"]),
                descriptor_generation=(
                    int(s["descriptor_generation"])
                    if s.get("descriptor_generation") is not None
                    else None
                ),
            )
            for s in (data.get("per_block_status") or ())
        )
        # T1/T2: Extract per-rank data if present (TP>1).
        per_rank_descriptors: dict = {}
        per_rank_source_metadata: dict = {}
        raw_prd = data.get("per_rank_descriptors")
        if isinstance(raw_prd, dict):
            for rank_key, desc_list in raw_prd.items():
                rank = int(rank_key)
                per_rank_descriptors[rank] = desc_list  # raw dicts
        raw_prm = data.get("per_rank_source_metadata")
        if isinstance(raw_prm, (dict, list)):
            import base64 as _b64_prm
            if isinstance(raw_prm, list):
                # List indexed by rank.
                for i, meta in enumerate(raw_prm):
                    per_rank_source_metadata[i] = meta
            else:
                for rank_key, meta in raw_prm.items():
                    per_rank_source_metadata[int(rank_key)] = meta
            # Decode agent_metadata_b64 → agent_metadata (raw bytes)
            # so _ensure_peer_loaded can pass it to add_remote_agent.
            for rank, meta in per_rank_source_metadata.items():
                if isinstance(meta, dict) and "agent_metadata_b64" in meta:
                    meta = dict(meta)  # don't mutate the original
                    meta["agent_metadata"] = _b64_prm.b64decode(
                        meta["agent_metadata_b64"]
                    )
                    per_rank_source_metadata[rank] = meta

        return RemoteG2ResolveResult(
            lease_id=data.get("lease_id"),
            descriptors=descriptors,
            num_tokens=int(data.get("num_tokens", 0)),
            reason=str(data.get("reason", "ok")),
            source_generation=int(data.get("source_generation", 0)),
            per_block_status=per_block_status,
            per_rank_descriptors=per_rank_descriptors,
            per_rank_source_metadata=per_rank_source_metadata,
        )
    except (KeyError, TypeError, ValueError):
        logging.exception("remote_g2: malformed resolve response dict: %r", data)
        return None


def _empty_result(reason: str) -> RemoteG2ResolveResult:
    """Build a no-lease result the BindingStore will treat as 'not
    resolvable' so the scheduler falls back to local recompute."""
    return RemoteG2ResolveResult(
        lease_id=None,
        descriptors=(),
        num_tokens=0,
        reason=reason,
        source_generation=0,
        per_block_status=(),
    )


def _make_resolve_callable(wrapper: _TargetReqWrapper):
    def _resolve(plan: RemoteKvReusePlan) -> RemoteG2ResolveResult:
        logging.warning(
            "PROBE rpc_chain target_resolve_callable pid=%d plan_id=%s "
            "source_worker_id=%s",
            os.getpid(),
            plan.plan_id,
            plan.source_worker_id,
        )
        try:
            response = wrapper.request(
                "resolve",
                {
                    "plan": _plan_to_dict(plan),
                    "source_worker_id": int(plan.source_worker_id),
                },
            )
        except Exception:
            logging.exception("remote_g2: target REQ resolve raised")
            return _empty_result("transport_failure")

        if not isinstance(response, dict):
            return _empty_result("malformed_response")
        if not response.get("ok"):
            err = response.get("error", "unknown")
            logging.warning("remote_g2: target REQ resolve returned not-ok: %s", err)
            return _empty_result(f"rpc_error:{err}")
        result = _result_from_dict(response.get("result") or {})
        return result or _empty_result("malformed_result")

    return _resolve


def _make_release_callable(wrapper: _TargetReqWrapper, default_source_worker_id: int):
    """The connector's release_lease signature is ``(lease_id, reason) -> bool``;
    it does not carry source_worker_id. We close over a default ID — the
    parent's REP loop is responsible for routing the release back to the
    correct source via the lease_id-to-source mapping it maintains.

    For the POC, we pass the connector worker's own worker_id as the
    default. The parent's REP loop ignores it for release calls and
    extracts the source from the lease_id prefix instead.
    """

    def _release(lease_id: str, reason: str) -> bool:
        try:
            response = wrapper.request(
                "release",
                {
                    "lease_id": str(lease_id),
                    "reason": str(reason),
                    "source_worker_id": int(default_source_worker_id),
                },
            )
        except Exception:
            logging.exception("remote_g2: target REQ release raised")
            return False
        if not isinstance(response, dict) or not response.get("ok"):
            return False
        return bool(response.get("result"))

    return _release


def _make_source_metadata_fetcher(wrapper: _TargetReqWrapper, peer_info_provider=None):
    """Build the source_metadata_fetcher callable the NIXL transfer
    adapter calls with (source_worker_id, source_generation) to learn
    the source's NIXL agent identity. Round-trips through the same
    engine→parent ZMQ bridge as resolve/release.

    When ``peer_info_provider`` is supplied, it's called inside _fetch
    to obtain ``(target_name, target_connection_info)``. These are
    threaded into the RPC payload so the source side can pre-load us
    as a peer via load_remote_agent_by_connection BEFORE returning its
    metadata. That bidirectional handshake is what makes the subsequent
    createXferReq's rkey lookup succeed."""

    from .remote_g2_transfer import RemoteG2SourceMetadata, RemoteG2TransferError

    def _fetch(source_worker_id: int, source_generation: int) -> RemoteG2SourceMetadata:
        payload = {"source_worker_id": int(source_worker_id)}
        if peer_info_provider is not None:
            try:
                peer_name, peer_conn = peer_info_provider()
                if peer_name and peer_conn:
                    payload["peer_name"] = str(peer_name)
                    payload["peer_connection_info"] = str(peer_conn)
            except Exception:
                logging.exception("remote_g2: peer_info_provider raised")
        try:
            response = wrapper.request("metadata", payload)
        except Exception as exc:
            raise RemoteG2TransferError(
                f"metadata RPC raised: {exc!r}"
            ) from exc
        if not isinstance(response, dict) or not response.get("ok"):
            err = response.get("error") if isinstance(response, dict) else "non_dict"
            raise RemoteG2TransferError(f"metadata RPC not ok: {err}")
        inner = response.get("result")
        # Unwrap double envelope: parent's REP returns {"ok": True, "result": <inner>}
        # where <inner> is the source's response envelope {"ok": True, "result": <flat>}.
        if isinstance(inner, dict) and "result" in inner and "ok" in inner:
            inner = inner.get("result")
        if not isinstance(inner, dict):
            raise RemoteG2TransferError("metadata RPC malformed response")
        # Source encodes agent_desc bytes as base64 over the dynamo wire;
        # decode back to raw bytes before handing to the NIXL adapter.
        import base64 as _b64
        agent_desc_b64 = inner.get("agent_desc_b64")
        if agent_desc_b64 is not None:
            agent_desc = _b64.b64decode(agent_desc_b64)
        else:
            # Backwards-compat for any older source that still sends raw bytes
            # (won't survive dynamo transport, but covers in-process tests).
            agent_desc = bytes(inner.get("agent_desc", b""))
        connection_info = str(inner.get("connection_info", ""))
        meta = RemoteG2SourceMetadata(
            source_worker_id=int(inner["source_worker_id"]),
            source_generation=int(inner.get("source_generation", source_generation)),
            remote_name=str(inner["remote_name"]),
            agent_desc=agent_desc,
        )
        # Stash extra fields on the dataclass instance for downstream use.
        object.__setattr__(meta, "connection_info", connection_info)
        # T4: Stash per-rank metadata (S5 response) so the adapter can
        # index by mpi_rank() at transfer time.
        per_rank_metadata = inner.get("per_rank_metadata")
        if per_rank_metadata is not None:
            object.__setattr__(meta, "per_rank_metadata", per_rank_metadata)
        return meta

    return _fetch


def _make_target_descriptor_resolver(
    primary_pool_base_ptr: int,
    block_size_bytes: int,
    device_id: int,
):
    """Build the target_descriptor_resolver callable the adapter uses to
    convert a RemoteG2BindingRecord's bound_blocks into VRAM descriptors
    pointing at the target's primary KV pool. Standard TRT-LLM layout:
    block N lives at ``primary_base + N * block_size_bytes``.
    """
    from .remote_g2_transfer import RemoteG2TransferDescriptor

    def _resolve(record: RemoteG2BindingRecord) -> Sequence[RemoteG2TransferDescriptor]:
        descs = []
        for block in record.bound_blocks:
            # Use the primary-pool slot index (resolved at bind time) rather
            # than the engine's block_id; block_ids are globally unique
            # across all blocks ever allocated and may exceed the primary
            # pool's slot count, while slot_idx is the dense per-pool index
            # that ptr arithmetic needs.
            slot_idx = int(getattr(block, "target_slot_idx", -1))
            if slot_idx < 0:
                slot_idx = int(block.target_block_id)  # legacy fallback
            ptr = int(primary_pool_base_ptr) + slot_idx * int(block_size_bytes)
            descs.append(
                RemoteG2TransferDescriptor(
                    ptr=ptr,
                    size=int(block_size_bytes),
                    device_id=int(device_id),
                    memory_type="VRAM",
                    name=f"remote-g2-target-block-{block.target_block_id}",
                )
            )
        # PROBE: emit the first/last target descriptor + source counterpart
        # so we can verify they reference valid registered ranges on each side
        if descs:
            head_src = record.bound_blocks[0].source_descriptor
            head_tgt = descs[0]
            tail_tgt = descs[-1]
            logging.warning(
                "PROBE rpc_chain target_resolver request_id=%s n_blocks=%d "
                "tgt_dev_id=%d block_size=%d "
                "src_head ptr=0x%x len=%d pool=%s "
                "tgt_head ptr=0x%x size=%d "
                "tgt_tail ptr=0x%x size=%d",
                record.request_id,
                len(descs),
                int(device_id),
                int(block_size_bytes),
                int(head_src.metadata.get("nixl_memory_desc", {}).get("ptr", 0))
                  if isinstance(head_src.metadata.get("nixl_memory_desc"), dict)
                  else 0,
                int(head_src.byte_length),
                str(head_src.pool_id),
                head_tgt.ptr,
                head_tgt.size,
                tail_tgt.ptr,
                tail_tgt.size,
            )
        return descs

    return _resolve


def _build_listening_nixl_agent(name: str):
    """Build a BindingsNixlTransferAgent with use_listen_thread=True so
    the target also exposes a NIXL data port. Without this, the source
    can't push its memory descriptors back to the target during the
    fetchRemoteMD handshake (it hangs in the checkRemoteMD polling
    loop)."""
    from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import (
        BindingsNixlTransferAgent,
    )
    from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
        BaseAgentConfig,
        NixlTransferAgent as CppNixlTransferAgent,
    )

    config = BaseAgentConfig(
        name,
        True,  # use_prog_thread
        multi_thread=False,
        use_listen_thread=True,
        enable_telemetry=False,
        backend_params={"num_threads": "1"},
    )
    agent = BindingsNixlTransferAgent.__new__(BindingsNixlTransferAgent)
    agent._cpp_agent = CppNixlTransferAgent(config)
    agent.name = name
    logging.warning(
        "PROBE remote_g2_target_listening_agent: name=%s connection_info=%s",
        name, agent.get_local_connection_info(),
    )
    return agent


class _ConnectionInfoNixlAdapter:
    """Local subclass-by-composition of RemoteG2NixlTransferAdapter that
    swaps the broken ``agent.load_remote_agent(name, agent_desc_bytes)``
    handshake for ``agent.load_remote_agent_by_connection(name, ip:port)``.

    The cpp NIXL agent's bytes-based load_remote_agent only deserializes
    a pre-fetched MD blob locally; the connection-info-based variant
    actively fetches remote MD over the data port and polls
    ``checkRemoteMD`` until the remote's registered memory rkeys arrive.
    The former is enough for connection setup but ``createXferReq``
    fails with NIXL_ERR_NOT_FOUND because UCX has no rkey for the
    remote DRAM region. The latter is what we actually need.

    Wraps the upstream adapter rather than subclassing because
    RemoteG2NixlTransferAdapter is a regular class without a clean
    extension hook for the load_remote_agent step. We delegate
    construction; only ``start_transfer`` is overridden.
    """

    def __init__(self, _peer_info_state=None, **kwargs):
        from .remote_g2_transfer import RemoteG2NixlTransferAdapter
        # Pop our adapter-only kwarg before forwarding to upstream.
        self._peer_info_state = _peer_info_state or {"peer_name": "", "peer_conn": ""}
        # Override agent_factory so the upstream adapter builds an agent
        # with use_listen_thread=True. NIXL needs *both* peers to have
        # listener threads so fetchRemoteMD can resolve the remote's
        # rkeys bidirectionally; the stock NixlTransferAgent(name) wraps
        # use_listen_thread=False, breaking that handshake.
        agent_name = kwargs.get("agent_name", "remote-g2-target")
        kwargs.setdefault("agent_factory", lambda: _build_listening_nixl_agent(agent_name))
        self._inner = RemoteG2NixlTransferAdapter(**kwargs)
        # Track which (name, generation) pairs we've already loaded so
        # we don't re-fetch on every transfer.
        self._loaded_remote_agents: set[tuple[str, int]] = set()

    def start_transfer(self, record):
        from .remote_g2_transfer import (
            RemoteG2TransferDescriptor,
            RemoteG2TransferError,
            RemoteG2TransferResult,
        )
        if not record.is_transfer_ready:
            raise RemoteG2TransferError("remote G2 binding is not transfer-ready")

        # Step 1 — Build target NIXL agent EARLY so we have our own
        # connection_info to send to the source as part of the metadata
        # RPC (bidirectional handshake). The upstream
        # adapter built this lazily after metadata fetch; we flip the
        # order so the fetcher's peer_info_provider closure sees an
        # already-constructed agent.
        types = self._inner._get_transfer_types()
        agent = self._inner._get_agent(types)
        try:
            name = agent.name if hasattr(agent, "name") else ""
            conn = agent.get_local_connection_info()
        except Exception:
            name, conn = "", ""
        # Update the shared peer-info state the fetcher closure reads.
        self._peer_info_state["peer_name"] = name
        self._peer_info_state["peer_conn"] = conn
        self._inner._local_peer_name = name
        self._inner._local_peer_conn = conn

        # Step 2 — Resolve source metadata. The fetcher's
        # peer_info_provider closure (set up in _build_target_nixl_adapter)
        # reads _local_peer_name / _local_peer_conn from self._inner and
        # threads them into the RPC payload. Source then pre-loads us
        # as a peer before returning its metadata.
        source_metadata = self._inner._source_metadata_cache.get_or_refresh(
            record.plan.source_worker_id,
            record.source_generation,
            self._inner._source_metadata_fetcher,
        )

        # Step 3 — Build source / target descriptors (same as upstream).
        source_descs = tuple(
            RemoteG2TransferDescriptor.from_source_descriptor(block.source_descriptor)
            for block in record.bound_blocks
        )
        target_descs = tuple(
            self._inner._normalize_target_descriptor(descriptor)
            for descriptor in self._inner._target_descriptor_resolver(record)
        )
        self._inner._validate_descriptors(source_descs, target_descs)

        # Step 4 — Connection-info-based load on our side (matching
        # what source already did for us via the metadata RPC).
        connection_info = getattr(source_metadata, "connection_info", "") or ""
        key = (source_metadata.remote_name, int(source_metadata.source_generation))
        if connection_info and key not in self._loaded_remote_agents:
            logging.warning(
                "PROBE rpc_chain target_load_remote_by_connection name=%s connection_info=%s "
                "(peer_handshake target=%s -> source=%s)",
                source_metadata.remote_name, connection_info,
                self._inner._local_peer_name, source_metadata.remote_name,
            )
            agent.load_remote_agent_by_connection(
                source_metadata.remote_name, connection_info
            )
            self._loaded_remote_agents.add(key)
        elif not connection_info:
            logging.warning(
                "remote_g2: no connection_info from source — falling back to bytes load (may fail at createXferReq)"
            )
            agent.load_remote_agent(
                source_metadata.remote_name, source_metadata.agent_desc
            )

        # Step 4 — Register the target VRAM blocks just-in-time.
        target_registration = types.RegMemoryDescs(
            "VRAM", [d.registration_tuple() for d in target_descs]
        )
        agent.register_memory(target_registration)

        # Step 5 — Construct and submit the transfer (cpp types via shim).
        request = types.TransferRequest(
            types.TransferOp.READ,
            types.MemoryDescs("DRAM", [d.transfer_tuple() for d in source_descs]),
            types.MemoryDescs("VRAM", [d.transfer_tuple() for d in target_descs]),
            source_metadata.remote_name,
        )
        status = agent.submit_transfer_requests(request)
        logging.warning(
            "PROBE rpc_chain nixl_read_submitted request_id=%s blocks=%d",
            record.request_id, len(source_descs),
        )
        return RemoteG2TransferResult(
            record=record,
            source_metadata=source_metadata,
            source_descs=source_descs,
            target_descs=target_descs,
            status=status,
            agent=agent,
            target_registration=target_registration,
        )


def _build_cpp_transfer_types_shim():
    """Build a SimpleNamespace of transfer types that the adapter can
    use, backed by C++ binding classes where needed.

    Why this shim:
    The adapter calls ``types.MemoryDescs("DRAM", [...])`` and
    ``types.TransferRequest(op, src, dst, name)``. ``submit_transfer_requests``
    on the C++-bound agent expects the C++ classes for MemoryDescs /
    TransferRequest, but the adapter's default ``_get_transfer_types``
    imports the Python wrappers from ``base.agent``. Mismatch → TypeError.

    We supply factory callables matching the adapter's call signature
    that translate the Python string into the C++ MemoryType enum and
    return real C++ binding instances.
    """
    from types import SimpleNamespace

    try:
        from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
            MemoryDescs as CppMemoryDescs,
            MemoryType as CppMemoryType,
            TransferOp as CppTransferOp,
            TransferRequest as CppTransferRequest,
        )
    except Exception:
        logging.exception("remote_g2: failed to import cpp transfer binding types")
        return None

    # RegMemoryDescs stays Python — _agent_cpp.register_memory converts it
    # internally via _convert_reg_memory_descs.
    from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs

    from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent

    _STRING_TO_CPP_MEMTYPE = {
        "DRAM": CppMemoryType.DRAM,
        "HOST": CppMemoryType.DRAM,
        "HOST_PINNED": CppMemoryType.DRAM,
        "CPU": CppMemoryType.DRAM,
        "VRAM": CppMemoryType.VRAM,
        "GPU": CppMemoryType.VRAM,
        "CUDA": CppMemoryType.VRAM,
    }

    def _make_memory_descs(mem_type, tuples):
        """Adapter passes strings like 'DRAM'/'VRAM'; cpp binding wants
        CppMemoryType enum + a sequence of (ptr, size, device_id) tuples."""
        if isinstance(mem_type, str):
            cpp_type = _STRING_TO_CPP_MEMTYPE.get(mem_type.upper())
            if cpp_type is None:
                raise ValueError(f"unsupported memory type string: {mem_type!r}")
        else:
            cpp_type = mem_type
        return CppMemoryDescs(cpp_type, list(tuples))

    def _make_transfer_request(op, src_descs, dst_descs, remote_name):
        # All four args already in the right type:
        #   op           → CppTransferOp enum (we pass types.TransferOp.READ below)
        #   src/dst      → CppMemoryDescs (from _make_memory_descs above)
        #   remote_name  → str
        return CppTransferRequest(op, src_descs, dst_descs, remote_name)

    return SimpleNamespace(
        MemoryDescs=_make_memory_descs,
        RegMemoryDescs=RegMemoryDescs,
        TransferOp=CppTransferOp,
        TransferRequest=_make_transfer_request,
        NixlTransferAgent=NixlTransferAgent,
    )


def _build_target_nixl_adapter(
    wrapper: _TargetReqWrapper,
    kv: Any,
    own_worker_id: int,
) -> Optional[Any]:
    """Construct the raw-nixl-based target transfer adapter.

    Switched from TRT-LLM's RemoteG2NixlTransferAdapter (wraps
    BindingsNixlTransferAgent) to raw nixl_agent because the wrapper
    only supports the combined transfer flow, which doesn't propagate
    rkeys properly for cross-process READ.

    Returns None if any prerequisite is missing.
    """
    from .remote_g2_raw_nixl_adapter import RawNixlRemoteG2Adapter

    # Primary pool gives us VRAM base + per-block byte stride. Mirror the
    # secondary-pool derivation we use on the source side.
    # Use get_unique_primary_pool() — returns the FULL primary KV cache
    # tensor (this is what TRT-LLM passes into register_kv_caches).
    # get_primary_pool_data(0) returns only a per-window slice and gave
    # us an undersized range (~86 MiB vs the full multi-GiB cache),
    # which made block_ids overflow our prepared dlist with
    # NIXL_ERR_INVALID_PARAM at make_prepped_xfer time.
    try:
        primary_pool = kv.get_unique_primary_pool()
    except Exception:
        logging.exception("remote_g2: cannot access primary pool tensor")
        return None

    if primary_pool is None or primary_pool.numel() == 0:
        logging.warning("remote_g2: primary pool tensor is empty")
        return None

    primary_base_ptr = int(primary_pool.data_ptr())
    block_size_bytes = _derive_block_size_bytes(kv)
    if not block_size_bytes or block_size_bytes <= 0:
        logging.warning(
            "remote_g2: primary block_size_bytes unknown; cannot build target descriptor resolver"
        )
        return None

    try:
        device_id = int(getattr(primary_pool.device, "index", 0) or 0)
    except Exception:
        device_id = 0

    # Compute the full primary VRAM pool size — needed by the raw NIXL
    # adapter to pre-register the entire pool at construction (rather
    # than registering each block just-in-time).
    try:
        primary_pool_size_bytes = int(
            primary_pool.element_size() * primary_pool.numel()
        )
    except Exception:
        logging.exception("remote_g2: failed to compute primary pool size")
        return None

    # The raw adapter's source_metadata_fetcher returns a plain dict
    # (not a RemoteG2SourceMetadata dataclass). The fetcher passes the
    # target's serialized agent metadata bytes via the metadata RPC
    # payload, so source can add_remote_agent on us BEFORE returning
    # its own metadata bytes.
    def _raw_metadata_fetcher(source_worker_id: int, source_generation: int) -> dict:
        import base64 as _b64
        # The local NIXL agent for the raw adapter isn't yet available
        # at this point (the adapter populates _local_peer_metadata_b64
        # on construction). We rely on the adapter setting it via the
        # shared state dict below.
        payload = {"source_worker_id": int(source_worker_id)}
        peer_b64 = adapter_state.get("peer_metadata_b64", "")
        if peer_b64:
            payload["peer_metadata_b64"] = peer_b64
        response = wrapper.request("metadata", payload)
        if not isinstance(response, dict) or not response.get("ok"):
            err = response.get("error") if isinstance(response, dict) else "non_dict"
            raise RuntimeError(f"metadata RPC not ok: {err}")
        inner = response.get("result")
        # Unwrap nested envelope from the dynamo round trip.
        if isinstance(inner, dict) and "result" in inner and "ok" in inner:
            inner = inner.get("result")
        if not isinstance(inner, dict):
            raise RuntimeError("metadata RPC malformed response")
        # Decode the source's raw NIXL metadata bytes.
        if "agent_metadata_b64" in inner:
            inner = dict(inner)
            inner["agent_metadata"] = _b64.b64decode(inner["agent_metadata_b64"])
        return inner

    adapter_state: dict[str, str] = {"peer_metadata_b64": ""}
    descriptor_resolver = _make_target_descriptor_resolver(
        primary_base_ptr, block_size_bytes, device_id
    )

    agent_name = f"remote-g2-target-{own_worker_id}"
    try:
        adapter = RawNixlRemoteG2Adapter(
            source_metadata_fetcher=_raw_metadata_fetcher,
            target_descriptor_resolver=descriptor_resolver,
            agent_name=agent_name,
            primary_pool_base_ptr=primary_base_ptr,
            primary_pool_size_bytes=primary_pool_size_bytes,
            device_id=device_id,
        )
        # Populate our own metadata bytes for the bidirectional
        # handshake so the source can add_remote_agent on us.
        import base64 as _b64
        adapter_state["peer_metadata_b64"] = _b64.b64encode(
            adapter._agent.get_agent_metadata()
        ).decode("ascii")
    except Exception:
        logging.exception("remote_g2: RawNixlRemoteG2Adapter construction failed")
        return None

    logging.warning(
        "PROBE remote_g2_target_adapter: agent_name=%s primary_pool_base=0x%x "
        "block_size_bytes=%d device_id=%d",
        agent_name,
        primary_base_ptr,
        block_size_bytes,
        device_id,
    )
    return adapter


def _wait_for_socket(path: str, timeout_s: float = 30.0) -> bool:
    """Poll for the parent-side REP socket file to appear. The dynamo
    parent binds it during init_llm_worker, which races slightly with
    engine subprocess setup; a short poll catches the case where this
    bootstrap runs first."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.25)
    return False


def maybe_start_remote_g2_target_client(
    kv: Optional[Any] = None,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> bool:
    """Open the engine→parent ZMQ REQ socket and install module-state
    callables on remote_g2_connector. Returns True on success, False
    when not configured (no dynamo parent reachable) or when the
    parent's REP socket never appears.

    TP>1: tp_rank and tp_size are passed through so the target adapter
    knows which rank's source metadata to use for NIXL peer loads.

    When ``kv`` (the C++ kv_cache_manager) is provided, also constructs
    the NIXL transfer adapter and installs it + no-op mark_local_valid /
    publish_binding hooks. Without ``kv``, the resolver/release path is
    still wired but the transfer adapter slot stays empty.

    Called from PyExecutor right after the source-side service is
    bootstrapped. Order-independent with respect to scheduler
    construction because the connector reads module state at call time.
    """
    dynamo_pid = _walk_to_dynamo_worker_pid()
    if dynamo_pid is None:
        logging.info(
            "remote_g2: target client skipped "
            "(dynamo parent not reachable from engine subprocess)"
        )
        return False

    socket_path = f"/tmp/dynamo_remote_g2_target_{dynamo_pid}.sock"
    if not _wait_for_socket(socket_path):
        logging.warning(
            "remote_g2: target client skipped (parent REP socket %s never appeared)",
            socket_path,
        )
        return False

    # Resolve own_worker_id via the same sidecar-file fallback the
    # source side uses — env vars are stripped by orted when MPI spawns
    # the engine subprocess, so plain os.environ.get(...) returns ""
    # and we'd otherwise fall back to "0" (which collides between
    # workers when used as a NIXL agent name).
    identity = _resolve_source_identity()
    if identity is not None:
        own_worker_id = identity[0]
    else:
        try:
            own_worker_id = int(os.environ.get("DYNAMO_REMOTE_G2_WORKER_ID", "0"))
        except ValueError:
            own_worker_id = 0

    try:
        wrapper = _TargetReqWrapper(socket_path)
    except Exception:
        logging.exception("remote_g2: failed to open target REQ socket at %s", socket_path)
        return False

    resolve_fn = _make_resolve_callable(wrapper)
    release_fn = _make_release_callable(wrapper, own_worker_id)

    remote_g2_connector.install_resolve_and_lease(resolve_fn)
    remote_g2_connector.install_release_lease(release_fn)

    # Install the block_id → primary-pool slot_idx lookup the binding
    # store uses to build NIXL local-dlist indices.
    # Unwrap .impl to get the C++ KvCacheManager binding — the Python
    # wrapper may not expose pin_blocks_by_id / get_slot_idx_by_block_id.
    _kv_impl = getattr(kv, "impl", kv)
    window_size = _derive_window_size(_kv_impl)

    # Detect whether the non-pinning accessor exists (added after rc15).
    _has_get_slot = hasattr(_kv_impl, "get_slot_idx_by_block_id")
    _has_pin_unpin = hasattr(_kv_impl, "pin_blocks_by_id") and hasattr(
        _kv_impl, "unpin_blocks_by_id"
    )
    logging.warning(
        "remote_g2: block_id_to_slot_idx setup: "
        "has_get_slot=%s has_pin_unpin=%s window_size=%s kv_type=%s",
        _has_get_slot,
        _has_pin_unpin,
        window_size,
        type(_kv_impl).__name__,
    )

    def _block_id_to_slot_idx(block_ids: list[int]) -> list[int]:
        if not block_ids or window_size is None:
            return []
        int_ids = [int(b) for b in block_ids]

        # Preferred: non-pinning single-call accessor (post-rc15).
        if _has_get_slot:
            try:
                return [
                    int(_kv_impl.get_slot_idx_by_block_id(b, int(window_size)))
                    for b in int_ids
                ]
            except Exception as exc:
                logging.warning(
                    "remote_g2: get_slot_idx_by_block_id failed: %s", exc
                )
                return []

        # Fallback: pin_blocks_by_id returns [(pool_type, slot_idx), ...].
        # We immediately unpin so refcounts stay balanced.
        if _has_pin_unpin:
            try:
                pairs = _kv_impl.pin_blocks_by_id(int_ids)
                # Immediately unpin — we only need the slot indices.
                _kv_impl.unpin_blocks_by_id(int_ids)
                return [int(p[1]) for p in pairs]
            except Exception as exc:
                logging.warning(
                    "remote_g2: pin_blocks_by_id fallback failed: %s", exc
                )
                try:
                    _kv_impl.unpin_blocks_by_id(int_ids)
                except Exception:
                    pass
                return []

        logging.warning(
            "remote_g2: no block_id→slot_idx method available"
        )
        return []

    remote_g2_connector.install_block_id_to_slot_idx(_block_id_to_slot_idx)

    logging.warning(
        "remote_g2: target client installed "
        "(socket=%s own_worker_id=%s)",
        socket_path,
        own_worker_id,
    )

    # Stage T3 — also build and install the NIXL transfer adapter +
    # no-op mark_local_valid / publish_binding hooks. The connector
    # worker reads these from module state at start_load_kv time, so
    # installation order vs. worker construction doesn't matter.
    if kv is None:
        logging.info(
            "remote_g2: transfer adapter not installed (kv_cache_manager not passed)"
        )
        return True

    # Mirror the source-side .impl unwrap; we need the C++ binding for
    # get_primary_pool_data + get_iteration_stats.
    kv = getattr(kv, "impl", kv)
    adapter = _build_target_nixl_adapter(wrapper, kv, own_worker_id)
    if adapter is None:
        logging.warning(
            "remote_g2: transfer adapter not installed (build failed); "
            "resolve still works, transfer disabled"
        )
        return True

    def _noop_mark_local_valid(record: RemoteG2BindingRecord) -> None:
        logging.info(
            "PROBE remote_g2_mark_local_valid (no-op): request_id=%s lease_id=%s blocks=%d",
            record.request_id,
            record.lease_id,
            len(record.bound_blocks),
        )

    def _noop_publish_binding(record: RemoteG2BindingRecord) -> None:
        logging.info(
            "PROBE remote_g2_publish_binding (no-op): request_id=%s lease_id=%s blocks=%d",
            record.request_id,
            record.lease_id,
            len(record.bound_blocks),
        )

    remote_g2_connector.install_transfer_adapter(adapter)
    remote_g2_connector.install_mark_local_valid(_noop_mark_local_valid)
    remote_g2_connector.install_publish_binding(_noop_publish_binding)

    logging.warning(
        "remote_g2: transfer adapter installed (no-op mark_local_valid / publish_binding)"
    )
    return True
