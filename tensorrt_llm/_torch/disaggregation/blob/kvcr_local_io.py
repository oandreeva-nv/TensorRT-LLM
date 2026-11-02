# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-thread local KVCR I/O for whole host slots."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, NoReturn

from ..base.memory import MemoryKind
from .kvcr_lifecycle import _KvcrAccessTracker, _MemoryReference
from .kvcr_peer import KvcrPeerHint, _BlobKeyAdapter, _fetch_envelope
from .store import RegionSpec, RowBatch, RowResult, RowStatus, Span, faulted

if TYPE_CHECKING:
    from kvcr.types import BlockKey, MemoryRef

_Completion = tuple[int, dict[bytes, RowResult]]


def _unsupported_pin(*args: object) -> NoReturn:
    raise RuntimeError("framework source pinning is not enabled for local KVCR I/O")


class _KvcrLocalIO:
    """Local I/O primitive for a future StoreDriver's owner thread.

    Args:
        region: One contiguous host region split into equal whole-object slots.
        cache: Disjoint host memory used exclusively by KVCR, in whole slots.
        endpoint: Unique name for this KVCR instance.
        listen_port: Explicit NIXL listener port, required even for local I/O.

    The caller owns both allocations and keeps them alive until ``close`` succeeds,
    including after a startup or close failure. All methods run on the constructing
    thread. Requests and progress call KVCR directly; this is not the nonblocking
    StoreDriver boundary. Progress returns normalized per-key row verdicts.
    ``blocks_reuse`` reports exceptional hazards only, not ordinary completion.
    """

    def __init__(
        self,
        region: RegionSpec,
        cache: Span,
        *,
        endpoint: str,
        listen_port: int,
        control_port: int | None = None,
        control_host: str = "127.0.0.1",
    ) -> None:
        if region.layout.memory_kind is not MemoryKind.HOST:
            raise ValueError("local KVCR I/O requires host memory")
        if region.span.overlaps(cache):
            raise ValueError("KVCR cache must not overlap framework memory")
        if cache.nbytes % region.layout.slot_bytes:
            raise ValueError("KVCR cache must contain whole slots")
        if not endpoint or not 0 < listen_port < 65536:
            raise ValueError("an endpoint name and valid listener port are required")

        from kvcr import KVCR, KVCRBindings
        from kvcr.config import KVCRBackendConfigs, KVCRConfig, LocalDramOptions
        from kvcr.types import RegionDescriptor

        self._owner = threading.get_ident()
        self._closed = False
        self._region = region
        self._endpoint = endpoint
        self._access = _KvcrAccessTracker(self._resolve)
        self._absent: dict[int, tuple[bytes, ...]] = {}
        self._completed: list[_Completion] = []
        self._next_local_handle = -1
        self._peer_requests: dict[int, str] = {}
        self._next_peer_request = 0
        control = None
        if control_port is not None:
            from kvcr.runtime.control_channels import ZmqPeerControlChannel

            control = ZmqPeerControlChannel(control_host, control_port, control_host)
        self._runtime = KVCR(
            KVCRConfig(
                nixl_agent_name=endpoint,
                pool_layouts=[("", region.layout.slot_bytes)],
                nixl_listen_port=listen_port,
            ),
            KVCRBindings(
                request_pin=_unsupported_pin,
                poll_pin_results=lambda: (),
                release_pin=_unsupported_pin,
                on_resilience_event=self._access.on_resilience_event,
                framework_control=control,
                key_adapter=_BlobKeyAdapter(),
            ),
            KVCRBackendConfigs(
                framework_regions=[
                    RegionDescriptor(
                        addr=region.address,
                        size=region.layout.slot_bytes,
                        count=region.layout.slot_count,
                    )
                ],
                local_dram=LocalDramOptions([("", cache.address, cache.nbytes)]),
            ),
        )

    def put(self, rows: RowBatch) -> int:
        """Deposit whole rows into KVCR-owned DRAM; return the SDK handle."""
        self._check_open()
        return self._runtime.deposit(self._references(rows))

    def get(self, rows: RowBatch, hint: KvcrPeerHint | None = None) -> int:
        """Deliver whole rows into the caller's slots; return an opaque handle."""
        from kvcr.types import QueryStatus

        self._check_open()
        blocks = self._references(rows)
        keys = tuple(blocks)
        request_id = None
        if hint is not None:
            self._next_peer_request += 1
            request_id = f"{self._endpoint}:{self._next_peer_request}"
            self._runtime.submit_hint(_fetch_envelope(hint, keys, request_id), request_id)
        statuses = (
            self._runtime.query(keys)
            if request_id is None
            else self._runtime.query(keys, request_id=request_id)
        )
        if len(statuses) != len(keys):
            raise RuntimeError("KVCR query returned the wrong number of results")
        absent = tuple(
            key for key, (status, _) in zip(keys, statuses) if status is QueryStatus.MISS
        )
        for key in absent:
            del blocks[key]
        if not blocks:
            if request_id is not None:
                self._runtime.discard_hint(request_id)
            handle = self._next_local_handle
            self._next_local_handle -= 1
            self._completed.append((handle, dict.fromkeys(absent, RowResult(RowStatus.MISS))))
            return handle
        handle = (
            self._runtime.deliver(blocks)
            if request_id is None
            else self._runtime.deliver(blocks, request_id=request_id)
        )
        if request_id is not None:
            self._peer_requests[handle] = request_id
        self._absent[handle] = absent
        return handle

    def progress(self) -> tuple[_Completion, ...]:
        """Process resilience events before publishing normalized row verdicts."""
        from kvcr.types import OpEntryStatus

        self._check_open()
        completed = []
        for handle, entries in self._runtime.poll_completed():
            request_id = self._peer_requests.pop(handle, None)
            if request_id is not None:
                self._runtime.discard_hint(request_id)
            results = {
                key: RowResult(RowStatus.READY)
                if entry.status is OpEntryStatus.SUCCESS
                else faulted(f"KVCR returned {entry.status.value}")
                for key, entry in entries.items()
            }
            for key in self._absent.pop(handle, ()):
                results[key] = RowResult(RowStatus.MISS)
            completed.append((handle, results))
        completed.extend(self._completed)
        self._completed.clear()
        return tuple(completed)

    def blocks_reuse(self, spans: Sequence[Span]) -> bool:
        """Whether exceptional accesses still quarantine any of these spans."""
        self._check_open()
        return self._access.blocks_reuse(spans)

    def close(self) -> None:
        """Release the runtime; failure leaves caller allocations protected."""
        self._check_owner()
        if not self._closed:
            self._runtime.close()
            self._closed = True

    def _references(self, rows: RowBatch) -> dict[BlockKey, list[MemoryRef]]:
        from kvcr.types import BlockKey, MemoryRef

        result = {}
        for row in rows:
            if len(row.spans) != 1:
                raise ValueError("a row must occupy one whole registered slot")
            span = row.spans[0]
            index, remainder = divmod(
                span.address - self._region.address, self._region.layout.slot_bytes
            )
            if (
                remainder
                or span.nbytes != self._region.layout.slot_bytes
                or not 0 <= index < self._region.layout.slot_count
            ):
                raise ValueError("a row must occupy one whole registered slot")
            result[BlockKey(row.key.encode("utf-8"))] = [
                MemoryRef(end_point_name=self._endpoint, element_index=index)
            ]
        return result

    def _resolve(self, ref: _MemoryReference) -> Span:
        if (
            ref.end_point_name != self._endpoint
            or ref.label != ""
            or not 0 <= ref.element_index < self._region.layout.slot_count
        ):
            raise ValueError("KVCR event does not identify a registered framework slot")
        size = self._region.layout.slot_bytes
        return Span(self._region.address + ref.element_index * size, size)

    def _check_owner(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("KVCR I/O must run on its owner thread")

    def _check_open(self) -> None:
        self._check_owner()
        if self._closed:
            raise RuntimeError("KVCR I/O is closed")
