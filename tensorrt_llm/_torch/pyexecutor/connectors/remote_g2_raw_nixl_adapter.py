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
from typing import Any, Optional

from .remote_g2_transfer import RemoteG2TransferState, validate_remote_g2_transfer_result


def _classify_nixl_transfer_state(state: Any) -> RemoteG2TransferState:
    """Normalize raw NIXL enum and string statuses, failing closed."""
    if state is None:
        return RemoteG2TransferState.FAILED

    raw_name = getattr(state, "name", state)
    value = str(raw_name).strip().upper().rsplit(".", 1)[-1]
    if value in {"DONE", "SUCCESS"}:
        return RemoteG2TransferState.SUCCEEDED
    if value in {"PROC", "PROCESSING", "PENDING"}:
        return RemoteG2TransferState.IN_PROGRESS
    return RemoteG2TransferState.FAILED


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
            enable_listen_thread=True,
            listen_port=0,  # OS-assigned
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

    supports_retryable_release = True

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
            enable_listen_thread=True,
            listen_port=0,
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
        except Exception:
            logging.exception(
                "remote_g2: raw nixl primary pool register_memory failed (ptr=0x%x size=%d)",
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
            raise RuntimeError(f"remote_g2: peer {peer_name} metadata missing from source response")
        loaded_name_raw = self._agent.add_remote_agent(peer_metadata)
        # add_remote_agent returns bytes; decode for string comparison.
        loaded_name = (
            loaded_name_raw.decode() if isinstance(loaded_name_raw, bytes) else loaded_name_raw
        )
        if loaded_name != peer_name:
            logging.warning(
                "remote_g2: add_remote_agent returned %r but expected %r",
                loaded_name,
                peer_name,
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
            (remote_pool_base + i * block_size, block_size, 0) for i in range(remote_num_blocks)
        ]
        remote_handle = self._agent.prep_xfer_dlist(
            peer_name,
            remote_descs,
            mem_type="DRAM",
        )

        self._peer_handles[key] = (local_handle, remote_handle)
        logging.warning(
            "PROBE remote_g2_raw_peer_loaded peer=%s gen=%d local_blocks=%d remote_blocks=%d",
            peer_name,
            peer_generation,
            num_blocks,
            remote_num_blocks,
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

        # Source metadata — drives add_remote_agent + remote dlist.
        if _nvtx is not None:
            _nvtx.range_push("remote_g2 metadata fetch")
        try:
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
        local_indices: list[int] = []
        remote_indices: list[int] = []
        for block in record.bound_blocks:
            # Use the primary-pool slot index (resolved at bind time) — NIXL's
            # local dlist is dense over slots; block_ids are globally-unique
            # engine identifiers that can exceed the slot count.
            slot_idx = int(getattr(block, "target_slot_idx", -1))
            if slot_idx < 0:
                slot_idx = int(block.target_block_id)  # legacy fallback
            local_indices.append(slot_idx)
            src_offset = int(block.source_descriptor.byte_offset)
            remote_indices.append(src_offset // block_size)

        logging.warning(
            "PROBE remote_g2_raw_make_prepped request_id=%s blocks=%d local_head=%d remote_head=%d",
            record.request_id,
            len(local_indices),
            local_indices[0] if local_indices else -1,
            remote_indices[0] if remote_indices else -1,
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

        # Take ownership before posting: transfer() can fail after NIXL has
        # created a live handle, and cleanup must still be retryable.
        result = _RawNixlTransferResult(
            agent=self._agent,
            handle=handle,
            record=record,
            initial_state=RemoteG2TransferState.FAILED,
        )
        validate_remote_g2_transfer_result(result)

        # Kick off the transfer. NIXL's `transfer(handle)` is the
        # post_xfer equivalent and returns the initial state. The
        # connector worker's get_finished now iterates self._active_loads
        # every tick (not just started_loading_req_ids), so we don't
        # need to block here — the framework will poll until DONE.
        if _nvtx is not None:
            _nvtx.range_push("remote_g2 nixl_post")
        try:
            try:
                state = self._agent.transfer(handle)
            except Exception as exc:
                result.start_error = exc
                return result
        finally:
            if _nvtx is not None:
                _nvtx.range_pop()
        result.initial_state = _classify_nixl_transfer_state(state)
        logging.warning(
            "PROBE remote_g2_raw_transfer_submitted request_id=%s initial_state=%s",
            record.request_id,
            state,
        )

        return result


@dataclass
class _RawNixlTransferResult:
    agent: Any
    handle: Any
    record: Any
    initial_state: RemoteG2TransferState
    start_error: Optional[BaseException] = None
    _released: bool = False

    def poll_state(self) -> RemoteG2TransferState:
        return _classify_nixl_transfer_state(self.agent.check_xfer_state(self.handle))

    def is_completed(self) -> bool:
        return self.poll_state() is RemoteG2TransferState.SUCCEEDED

    def wait(self, timeout_ms: Optional[int] = None) -> bool:
        import time

        deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000.0
        while True:
            state = self.poll_state()
            if state is RemoteG2TransferState.SUCCEEDED:
                return True
            if state is RemoteG2TransferState.FAILED:
                return False
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.001)

    def release_transfer(self) -> bool:
        if self._released:
            return True
        self.agent.release_xfer_handle(self.handle)
        self._released = True
        return True

    def release(self) -> None:
        self.release_transfer()
