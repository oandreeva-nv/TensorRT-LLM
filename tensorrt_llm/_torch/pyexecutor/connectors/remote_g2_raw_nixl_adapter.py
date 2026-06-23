# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw `nixl.nixl_agent` adapter for remote-G2 transfers.

Why this exists: TRT-LLM's wrapper (`_torch/disaggregation/nixl/_agent_cpp.py`)
exposes only the combined flow (`submit_transfer_requests` → cpp
`createXferReq`). For cross-process READ, peer rkeys must be fetched
via NIXL's prepped flow (`prep_xfer_dlist` → `make_prepped_xfer` →
`transfer`) — the combined flow assumes rkeys are already cached and
fails with NIXL_ERR_NOT_FOUND when they aren't.

This module bypasses the TRT-LLM wrapper and uses raw NIXL directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass
class _NixlSourceHandle:
    """Source-side NIXL state, captured for the metadata RPC."""

    agent: Any  # raw nixl.nixl_agent
    agent_name: str
    agent_metadata: bytes  # bytes blob peers pass to add_remote_agent
    connection_ip: str
    connection_port: int
    pool_base_ptr: int
    pool_size_bytes: int
    source_generation: int = 1


def build_raw_nixl_source_agent(
    *,
    agent_name: str,
    pool_base_ptr: int,
    pool_size_bytes: int,
) -> Optional[_NixlSourceHandle]:
    """Construct a raw nixl_agent on the source side and register the
    host_pinned secondary pool. Returns None if NIXL isn't available
    or any step fails."""
    try:
        from nixl import nixl_agent, nixl_agent_config
    except Exception:
        logging.exception("remote_g2: raw nixl import failed")
        return None

    try:
        config = nixl_agent_config(
            enable_prog_thread=True,
            # Disable the listen thread to avoid metadata stream port
            # conflicts when multiple NIXL agents coexist in the same
            # pod (TP>1 or dynamo's own NIXL agent). We exchange agent
            # metadata explicitly via add_remote_agent, not via NIXL's
            # auto-discovery metadata stream.
            enable_listen_thread=False,
            backends=["UCX"],
        )
        agent = nixl_agent(agent_name, config, instantiate_all=False)
    except Exception:
        logging.exception("remote_g2: raw nixl_agent construction failed")
        return None

    # Register the entire host_pinned secondary pool as a single DRAM
    # region. device_id=0 is the standard for host memory in NIXL.
    reg_desc = (pool_base_ptr, pool_size_bytes, 0, f"{agent_name}-host-pool")
    try:
        reg_list = agent.get_reg_descs([reg_desc], mem_type="DRAM")
        agent.register_memory(reg_list)
    except Exception:
        logging.exception(
            "remote_g2: raw register_memory failed for source pool 0x%x size=%d",
            pool_base_ptr,
            pool_size_bytes,
        )
        return None

    try:
        metadata = agent.get_agent_metadata()
    except Exception:
        logging.exception("remote_g2: get_agent_metadata failed")
        return None

    # NIXL's get_agent_metadata returns bytes containing the agent name +
    # registered MD; this is what peers pass to add_remote_agent.
    # The listener IP / port aren't exposed via a single getter on raw
    # nixl_agent — we'll let peers use fetch_remote_metadata which
    # discovers them automatically when given just the agent name.
    # For our same-host case, the IP is the container's IP.
    import socket
    try:
        # NIXL's listener binds on all interfaces. We need the container's
        # routable IP. Use a UDP socket trick to discover ourselves.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    # Listener port: raw nixl doesn't expose this directly via the
    # public API. Parse from the metadata blob if it's there, else
    # fall back to the default 8888.
    listen_port = _extract_listen_port_from_metadata(metadata) or 8888

    logging.info(
        "[NIXL-XFER] source_agent_ready: agent=%s ip=%s port=%d "
        "pool_base=0x%x pool_size=%d",
        agent_name, local_ip, listen_port,
        pool_base_ptr, pool_size_bytes,
    )
    return _NixlSourceHandle(
        agent=agent,
        agent_name=agent_name,
        agent_metadata=metadata,
        connection_ip=local_ip,
        connection_port=listen_port,
        pool_base_ptr=pool_base_ptr,
        pool_size_bytes=pool_size_bytes,
    )


def _extract_listen_port_from_metadata(metadata: bytes) -> Optional[int]:
    """Heuristic: raw NIXL's metadata blob includes the listener
    address in a discoverable but not strictly-stable layout. For the
    POC we look for a ':<port>' pattern in the printable parts. If
    not found, caller falls back to the default port.

    Long-term fix: file an upstream request to expose
    `get_local_connection_info()` on raw nixl_agent the way
    TRT-LLM's wrapper does."""
    try:
        # Try common ports range — most likely format embeds ASCII port.
        printable = metadata.decode("latin-1", errors="ignore")
        import re
        m = re.search(r":(\d{4,5})\b", printable)
        if m:
            port = int(m.group(1))
            if 1024 <= port <= 65535:
                return port
    except Exception:
        pass
    return None


class RawNixlRemoteG2Adapter:
    """Target-side adapter using raw nixl_agent with the prepped flow.

    Drop-in replacement for `RemoteG2NixlTransferAdapter`. Constructor
    takes the same shape (source_metadata_fetcher, target_descriptor_resolver)
    but the fetcher returns a dict (not RemoteG2SourceMetadata) carrying
    the raw NIXL agent metadata bytes.
    """

    def __init__(
        self,
        *,
        source_metadata_fetcher,  # (worker_id, generation) -> dict
        target_descriptor_resolver,  # (record) -> Sequence[descriptor]
        agent_name: str,
        primary_pool_base_ptr: int,
        primary_pool_size_bytes: int,
        device_id: int = 0,
    ):
        try:
            from nixl import nixl_agent, nixl_agent_config
        except Exception:
            raise RuntimeError("remote_g2: nixl import failed in target adapter")

        config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=False,
            backends=["UCX"],
        )
        self._agent = nixl_agent(agent_name, config, instantiate_all=False)
        self._agent_name = agent_name
        self._source_metadata_fetcher = source_metadata_fetcher
        self._target_descriptor_resolver = target_descriptor_resolver
        self._device_id = int(device_id)
        self._block_size_bytes = 0  # set on first transfer

        # Pre-register the entire primary VRAM pool — sub-gap "Buffer
        # pre-registration lifecycle" calls this out as required. The
        # whole-pool registration means prep_xfer_dlist can build the
        # local dlist without per-transfer re-registration.
        try:
            reg_desc = (
                primary_pool_base_ptr,
                primary_pool_size_bytes,
                self._device_id,
                f"{agent_name}-primary-pool",
            )
            reg_list = self._agent.get_reg_descs([reg_desc], mem_type="VRAM")
            self._agent.register_memory(reg_list)
            self._primary_pool_base_ptr = primary_pool_base_ptr
            self._primary_pool_size_bytes = primary_pool_size_bytes
            logging.info(
                "[NIXL-XFER] target_agent_ready: agent=%s "
                "vram_pool_base=0x%x vram_pool_size=%d device_id=%d",
                agent_name, primary_pool_base_ptr,
                primary_pool_size_bytes, self._device_id,
            )
        except Exception:
            logging.exception(
                "remote_g2: raw nixl primary pool register_memory failed "
                "(ptr=0x%x size=%d)",
                primary_pool_base_ptr,
                primary_pool_size_bytes,
            )
            raise

        # Per-peer prepped-dlist handles. Keyed by (source_agent_name,
        # source_generation). The local dlist is also cached per-peer
        # because NIXL's make_prepped_xfer requires both local and
        # remote handles to come from prep_xfer_dlist.
        self._peer_handles: dict[tuple[str, int], tuple[Any, Any]] = {}
        logging.warning(
            "PROBE remote_g2_raw_target_adapter: agent_name=%s primary_pool_base=0x%x "
            "primary_pool_size=%d device_id=%d",
            agent_name,
            primary_pool_base_ptr,
            primary_pool_size_bytes,
            device_id,
        )

    def _ensure_peer_loaded(
        self,
        source_meta: dict,
    ) -> tuple[Any, Any]:
        """Idempotent peer setup: add_remote_agent, prep local + remote
        dlist handles. Returns (local_handle, remote_handle) for the
        peer described by source_meta."""
        peer_name = source_meta["remote_name"]
        peer_generation = int(source_meta.get("source_generation", 1))
        key = (peer_name, peer_generation)
        if key in self._peer_handles:
            return self._peer_handles[key]

        # Load the peer's metadata (includes rkeys for its registered
        # memory regions).
        peer_metadata = source_meta.get("agent_metadata")
        if not peer_metadata:
            raise RuntimeError(
                f"remote_g2: peer {peer_name} metadata missing from source response"
            )
        loaded_name_raw = self._agent.add_remote_agent(peer_metadata)
        # add_remote_agent returns bytes; decode for string comparison.
        loaded_name = (
            loaded_name_raw.decode() if isinstance(loaded_name_raw, bytes)
            else loaded_name_raw
        )
        if loaded_name != peer_name:
            logging.warning(
                "remote_g2: add_remote_agent returned %r but expected %r",
                loaded_name, peer_name,
            )
        logging.info(
            "[NIXL-XFER] peer_loaded: local_agent=%s remote_agent=%s "
            "remote_pool_base=0x%x remote_pool_size=%d device_id=%d",
            self._agent_name, peer_name,
            int(source_meta.get("pool_base_ptr", 0)),
            int(source_meta.get("pool_size_bytes", 0)),
            self._device_id,
        )

        # Local dlist: covers our entire primary VRAM pool, indexed by
        # block. We pre-built block-aligned tuples so make_prepped_xfer's
        # index lookup is a 1:1 block_id-to-slot mapping.
        block_size = self._block_size_bytes
        num_blocks = self._primary_pool_size_bytes // block_size if block_size else 0
        local_descs = [
            (
                self._primary_pool_base_ptr + i * block_size,
                block_size,
                self._device_id,
            )
            for i in range(num_blocks)
        ]
        # NIXL distinguishes local vs remote dlists by agent_name:
        # empty string = local, non-empty = remote. Passing our own
        # name for local fails with "invalid sides (local must be
        # local, remote must be remote)" at make_prepped_xfer time.
        local_handle = self._agent.prep_xfer_dlist(
            "",  # local
            local_descs,
            mem_type="VRAM",
        )

        # Remote dlist: covers the source's host_pinned pool, also
        # block-aligned. The source pool base + size came back in the
        # metadata; we replicate the per-block index layout so the
        # source/target index arrays line up 1:1.
        remote_pool_base = int(source_meta["pool_base_ptr"])
        remote_pool_size = int(source_meta["pool_size_bytes"])
        remote_num_blocks = remote_pool_size // block_size if block_size else 0
        remote_descs = [
            (remote_pool_base + i * block_size, block_size, 0)
            for i in range(remote_num_blocks)
        ]
        remote_handle = self._agent.prep_xfer_dlist(
            peer_name,
            remote_descs,
            mem_type="DRAM",
        )

        self._peer_handles[key] = (local_handle, remote_handle)
        logging.warning(
            "PROBE remote_g2_raw_peer_loaded peer=%s gen=%d "
            "local_blocks=%d remote_blocks=%d",
            peer_name, peer_generation, num_blocks, remote_num_blocks,
        )
        return local_handle, remote_handle

    def start_transfer(self, record):
        """Issue a prepped-flow NIXL READ for the bound blocks of
        `record`. Returns a transfer-result object exposing
        is_completed() / wait() / release().

        NVTX-labeled regions (visible in nsys traces):
          - "remote_g2 start_transfer"      (the whole call)
          - "remote_g2 metadata fetch"      (source metadata RPC)
          - "remote_g2 ensure_peer_loaded"  (handshake + dlist build)
          - "remote_g2 nixl_post"           (transfer submit)
        """
        try:
            import torch.cuda.nvtx as _nvtx
        except Exception:
            _nvtx = None
        if _nvtx is not None:
            _nvtx.range_push(f"remote_g2 start_transfer req={record.request_id}")
        try:
            return self._start_transfer_impl(record, _nvtx)
        finally:
            if _nvtx is not None:
                _nvtx.range_pop()

    def _start_transfer_impl(self, record, _nvtx):
        if not record.is_transfer_ready:
            raise RuntimeError("remote G2 binding is not transfer-ready")

        # Block size — pick up from record on first call.
        block_size = record.plan.block_size_tokens * 0
        for block in record.bound_blocks:
            block_size = int(block.source_descriptor.byte_length)
            break
        if not block_size:
            raise RuntimeError("remote G2 record has zero-sized blocks")
        self._block_size_bytes = block_size

        # T3: Determine which source rank's metadata to use.
        # With TP>1, each target rank loads its corresponding source
        # rank's NIXL agent and uses that rank's descriptors.
        from tensorrt_llm._utils import mpi_rank as _mpi_rank
        my_rank = _mpi_rank()

        resolve_result = record.resolve_result
        per_rank_meta = getattr(resolve_result, "per_rank_source_metadata", {})
        per_rank_descs = getattr(resolve_result, "per_rank_descriptors", {})

        # Source metadata — drives add_remote_agent + remote dlist.
        if _nvtx is not None:
            _nvtx.range_push("remote_g2 metadata fetch")
        try:
            if per_rank_meta and my_rank in per_rank_meta:
                # TP>1: use this rank's source metadata directly.
                source_meta = per_rank_meta[my_rank]
                logging.info(
                    "remote_g2: T3 using per-rank metadata for tp_rank=%d "
                    "(source=%s)",
                    my_rank, source_meta.get("remote_name"),
                )
            else:
                # TP=1 or fallback: fetch via RPC (rank 0's metadata).
                source_meta = self._source_metadata_fetcher(
                    record.plan.source_worker_id,
                    int(record.source_generation),
                )
        finally:
            if _nvtx is not None:
                _nvtx.range_pop()
        if not isinstance(source_meta, dict):
            raise RuntimeError("remote_g2: source_metadata_fetcher returned non-dict")

        if _nvtx is not None:
            _nvtx.range_push("remote_g2 ensure_peer_loaded")
        try:
            local_handle, remote_handle = self._ensure_peer_loaded(source_meta)
        finally:
            if _nvtx is not None:
                _nvtx.range_pop()

        # Build index arrays. Block indices into the local dlist =
        # target_block_id; into the remote dlist = source byte_offset
        # divided by block_size.
        #
        # T3: With TP>1, use per_rank_descriptors for this rank's
        # source offsets. Fall back to bound_blocks' descriptors for
        # TP=1 (where all descriptors are rank 0's).
        local_indices: list[int] = []
        remote_indices: list[int] = []

        if per_rank_descs and my_rank in per_rank_descs:
            # TP>1: this rank's descriptors from the per-rank gather.
            rank_descs = per_rank_descs[my_rank]
            for i, block in enumerate(record.bound_blocks):
                slot_idx = int(getattr(block, "target_slot_idx", -1))
                if slot_idx < 0:
                    slot_idx = int(block.target_block_id)
                local_indices.append(slot_idx)
                if i < len(rank_descs) and rank_descs[i] is not None:
                    src_offset = int(rank_descs[i].get("byte_offset", 0))
                else:
                    # Block not available in secondary tier on this rank.
                    # Using rank 0's byte_offset would read from a
                    # different rank's (possibly empty) secondary pool
                    # causing data corruption.  Abort the transfer.
                    raise RuntimeError(
                        f"remote_g2: NIXL transfer aborted — source "
                        f"rank {my_rank} does not have block {i} "
                        f"(hash={getattr(block, 'source_block_hash', '?')}) "
                        f"in secondary (host-pinned) tier. "
                        f"rank_descs[{i}] is None; falling back to "
                        f"rank 0's byte_offset would cause data "
                        f"corruption. This indicates an asymmetric "
                        f"offload across TP ranks — blocks were "
                        f"offloaded to secondary on rank 0 but not "
                        f"on rank {my_rank}."
                    )
                remote_indices.append(src_offset // block_size)
        else:
            # TP=1: use bound_blocks directly.
            for block in record.bound_blocks:
                slot_idx = int(getattr(block, "target_slot_idx", -1))
                if slot_idx < 0:
                    slot_idx = int(block.target_block_id)
                local_indices.append(slot_idx)
                src_offset = int(block.source_descriptor.byte_offset)
                remote_indices.append(src_offset // block_size)

        logging.warning(
            "PROBE remote_g2_raw_make_prepped request_id=%s blocks=%d "
            "local_head=%d remote_head=%d",
            record.request_id, len(local_indices),
            local_indices[0] if local_indices else -1,
            remote_indices[0] if remote_indices else -1,
        )
        source_agent_name = source_meta.get("remote_name", "unknown")
        logging.info(
            "[NIXL-XFER] prep: request_id=%s tp_rank=%d blocks=%d "
            "source_agent=%s local_indices=%s remote_indices=%s "
            "block_size=%d device_id=%d",
            record.request_id, my_rank, len(local_indices),
            source_agent_name,
            local_indices[:4] if len(local_indices) > 4
            else local_indices,
            remote_indices[:4] if len(remote_indices) > 4
            else remote_indices,
            block_size, self._device_id,
        )

        if _nvtx is not None:
            _nvtx.range_push("remote_g2 make_prepped_xfer")
        try:
            handle = self._agent.make_prepped_xfer(
                "READ",
                local_handle,
                local_indices,
                remote_handle,
                remote_indices,
            )
        finally:
            if _nvtx is not None:
                _nvtx.range_pop()

        # Kick off the transfer. NIXL's `transfer(handle)` is the
        # post_xfer equivalent and returns the initial state. The
        # connector worker's get_finished now iterates self._active_loads
        # every tick (not just started_loading_req_ids), so we don't
        # need to block here — the framework will poll until DONE.
        if _nvtx is not None:
            _nvtx.range_push("remote_g2 nixl_post")
        try:
            state = self._agent.transfer(handle)
        finally:
            if _nvtx is not None:
                _nvtx.range_pop()
        logging.warning(
            "PROBE remote_g2_raw_transfer_submitted request_id=%s initial_state=%s",
            record.request_id, state,
        )
        logging.info(
            "[NIXL-XFER] submitted: request_id=%s tp_rank=%d "
            "source_worker=%s source_agent=%s "
            "blocks=%d initial_state=%s",
            record.request_id, my_rank,
            record.plan.source_worker_id,
            source_agent_name,
            len(local_indices), state,
        )
        return _RawNixlTransferResult(
            agent=self._agent,
            handle=handle,
            record=record,
        )


@dataclass
class _RawNixlTransferResult:
    agent: Any
    handle: Any
    record: Any
    _released: bool = False
    _logged_done: bool = False
    _poll_count: int = 0

    def is_completed(self) -> bool:
        state = self.agent.check_xfer_state(self.handle)
        self._poll_count += 1
        state_str = str(state).upper()
        if (
            self._poll_count == 1
            or self._poll_count % 100 == 0
            or state_str not in ("PROC", "PROCESSING", "PENDING")
        ):
            logging.warning(
                "PROBE remote_g2_raw_is_completed request_id=%s poll=%d state=%s",
                getattr(self.record, "request_id", "?"),
                self._poll_count,
                state_str,
            )
        if state_str in ("DONE", "SUCCESS") and not self._logged_done:
            self._logged_done = True
            logging.info(
                "[NIXL-XFER] completed: request_id=%s state=%s",
                self.record.request_id, state_str,
            )
        elif state_str in ("ERROR", "FAILED") and not self._logged_done:
            self._logged_done = True
            logging.error(
                "[NIXL-XFER] FAILED: request_id=%s state=%s",
                self.record.request_id, state_str,
            )
        return state_str in ("DONE", "SUCCESS")

    def wait(self, timeout_ms: Optional[int] = None) -> bool:
        import time
        deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000.0
        while True:
            state = self.agent.check_xfer_state(self.handle)
            if str(state).upper() in ("DONE", "SUCCESS"):
                return True
            if str(state).upper() in ("ERROR", "FAILED"):
                return False
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.001)

    def release(self) -> None:
        if self._released:
            return
        try:
            self.agent.release_xfer_handle(self.handle)
        except Exception:
            logging.exception("remote_g2: release_xfer_handle failed")
        self._released = True
