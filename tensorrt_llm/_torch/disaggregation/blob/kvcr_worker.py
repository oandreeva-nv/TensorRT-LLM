# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-owner KVCR SDK worker and cached operation state."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NoReturn

from ..base.memory import MemoryKind
from .kvcr_lifecycle import _KvcrAccessTracker, _MemoryReference
from .kvcr_peer import KvcrPeerHint, _BlobKeyAdapter, _fetch_envelope
from .store import (
    DriverRejected,
    OnChange,
    RegionSpec,
    RowBatch,
    RowResult,
    RowStatus,
    Span,
    faulted,
)

if TYPE_CHECKING:
    from kvcr.types import BlockKey, MemoryRef

_Completion = tuple[int, dict[bytes, RowResult]]
_LOGGER = logging.getLogger(__name__)


def _unsupported_pin(*args: object) -> NoReturn:
    raise RuntimeError("framework source pinning is not enabled for KVCR")


@dataclass(eq=False)
class _Work:
    method: Literal["put", "get"]
    rows: RowBatch
    results: tuple[RowResult, ...]
    on_change: OnChange | None
    hint: KvcrPeerHint | None = None
    ended: bool = False
    unprovable: bool = False


class _KvcrWorker:
    """Own the KVCR SDK on one thread and expose nonblocking queued operations.

    Construction waits for registration on a single owner thread. ``put``, ``get``
    and queries use short state locks; the owner never holds them during SDK calls.
    The caller keeps framework and cache allocations alive until ``close`` returns
    True, including after startup failure. Failed rows remain protected until
    successful teardown; this component does not infer access end from a failure.
    An optional ``cache_owner`` keeps driver-owned cache memory alive until teardown.

    Registration is fixed at construction. The owner translates whole host slots,
    processes SDK results and resilience events, and performs teardown. Cached
    queries remain available after teardown to inspect the final state.
    """

    def __init__(
        self,
        region: RegionSpec,
        cache: Span,
        *,
        endpoint: str,
        listen_port: int,
        cache_owner: object = None,
        control_port: int | None = None,
        control_host: str = "127.0.0.1",
    ) -> None:
        self._region = region
        self._control_options = (
            {}
            if control_port is None
            else {"control_port": control_port, "control_host": control_host}
        )
        self._cache_owner = cache_owner
        self._condition = threading.Condition(threading.Lock())
        self._queue: deque[_Work] = deque()
        self._operations: set[_Work] = set()
        self._active: dict[int, _Work] = {}
        self._watching: set[_Work] = set()
        self._draining = False
        self._fenced = False
        self._busy = False
        self._close_requested = False
        self._close_attempt = 0
        self._stopped = False
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            args=(region, cache, endpoint, listen_port),
            name=f"kvcr-{endpoint}",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError(
                "KVCR worker startup failed; retain caller allocations"
            ) from self._startup_error

    def put(self, rows: RowBatch, *, on_change: OnChange | None = None) -> _Work:
        """Queue a deposit without waiting on KVCR or iterating rows."""
        return self._submit("put", rows, on_change)

    def get(
        self,
        rows: RowBatch,
        hint: KvcrPeerHint | None = None,
        *,
        on_change: OnChange | None = None,
    ) -> _Work:
        """Queue a delivery without waiting on KVCR or iterating rows."""
        return self._submit("get", rows, on_change, hint)

    def poll(self, op: object) -> tuple[RowResult, ...]:
        """Read cached row verdicts without consulting the SDK."""
        with self._condition:
            return self._operation(op).results

    def access_ended(self, op: object) -> bool:
        """Read the monotonic, separately recorded memory-release verdict."""
        with self._condition:
            return self._operation(op).ended

    def access_unprovable(self, op: object) -> bool:
        """Whether a fenced operation lost its normal completion mechanism."""
        with self._condition:
            return self._operation(op).unprovable

    def drain(self, budget_s: float) -> bool:
        """Reject new work, cancel queued work, and wait within the given budget."""
        self._check_waiter()
        with self._condition:
            self._draining = True
            changed = self._cancel_queued()
            self._condition.notify_all()
        self._notify(changed)
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._drained() or any(op.unprovable for op in self._operations if not op.ended)
                ),
                timeout=max(0, budget_s),
            )
            return self._drained()

    def close(self, budget_s: float) -> bool:
        """Request owner-thread teardown; False means retain all allocations and retry.

        A budget bounds the caller's wait, not a running SDK call. Successful SDK
        teardown is independent access-end evidence, including for failed rows.
        """
        self._check_waiter()
        with self._condition:
            if self._stopped:
                return True
            self._draining = True
            changed = self._cancel_queued()
            attempt = self._close_attempt + 1
            self._close_requested = True
            self._condition.notify_all()
        self._notify(changed)
        with self._condition:
            self._condition.wait_for(
                lambda: self._stopped or (self._close_attempt >= attempt and not self._busy),
                timeout=max(0, budget_s),
            )
            return self._stopped

    def _submit(
        self,
        method: Literal["put", "get"],
        rows: RowBatch,
        on_change: OnChange | None,
        hint: KvcrPeerHint | None = None,
    ) -> _Work:
        if rows.span_bytes != (self._region.layout.slot_bytes,) or not all(
            extent.within(self._region.span) for extent in rows.extents
        ):
            raise DriverRejected("rows must occupy whole registered host slots")
        op = _Work(method, rows, (RowResult(RowStatus.PENDING),) * len(rows), on_change, hint)
        with self._condition:
            if self._draining:
                raise DriverRejected("KVCR worker is draining or fenced")
            self._operations.add(op)
            self._queue.append(op)
            self._condition.notify_all()
        return op

    def _operation(self, op: object) -> _Work:
        if not isinstance(op, _Work) or op not in self._operations:
            raise ValueError("unknown KVCR operation handle")
        return op

    def _drained(self) -> bool:
        return not self._busy and all(op.ended for op in self._operations)

    def _check_waiter(self) -> None:
        if threading.current_thread() is self._thread:
            raise RuntimeError("cannot wait for the KVCR worker on its owner thread")

    def _cancel_queued(self) -> set[_Work]:
        changed = set(self._queue)
        self._queue.clear()
        for op in changed:
            op.results = (RowResult(RowStatus.CANCELLED),) * len(op.rows)
            op.ended = True
        return changed

    def _run(self, region: RegionSpec, cache: Span, endpoint: str, listen_port: int) -> None:
        try:
            self._initialize_runtime(
                region, cache, endpoint=endpoint, listen_port=listen_port, **self._control_options
            )
        except Exception as error:
            # Thread boundary: report failure to the constructor without losing it.
            self._startup_error = error
            self._ready.set()
            return
        self._ready.set()
        while True:
            with self._condition:
                closing = self._close_requested
                self._close_requested = False
                op = self._queue.popleft() if self._queue and not closing else None
                self._busy = True
            changed: set[_Work] = set()
            stopped = False
            try:
                if closing:
                    self._close_runtime()
                    changed = self._finish_close()
                    stopped = True
                elif not self._fenced:
                    if op is not None:
                        if op.method == "put":
                            handle = self._sdk_put(op.rows)
                        else:
                            handle = self._sdk_get(op.rows, op.hint)
                        if handle in self._active:
                            raise RuntimeError("KVCR reused an active SDK handle")
                        self._active[handle] = op
                    self._progress(changed)
            except Exception as error:
                # No exception may strand issued handles or imply access has ended.
                changed |= self._fence(error)
            self._notify(changed)
            with self._condition:
                self._busy = False
                self._stopped = stopped
                if stopped:
                    self._cache_owner = None
                if closing:
                    self._close_attempt += 1
                self._condition.notify_all()
                if stopped:
                    return
                if not self._queue and not self._close_requested:
                    self._condition.wait(timeout=None if self._fenced else 0.001)

    def _progress(self, changed: set[_Work]) -> None:
        for handle, entries in self._poll_completed():
            op = self._active.pop(handle)
            keys = [key.encode("utf-8") for key in op.rows.keys()]
            if entries.keys() != set(keys):
                raise RuntimeError("KVCR completion keys do not match the submitted rows")
            results = tuple(entries[key] for key in keys)
            successful = all(
                result.status in (RowStatus.READY, RowStatus.MISS) for result in results
            )
            with self._condition:
                op.results = results
                if not successful:
                    op.unprovable = True
            if successful:
                self._watching.add(op)
            changed.add(op)
        for op in tuple(self._watching):
            if not self._blocks_reuse(op.rows.extents):
                with self._condition:
                    op.ended = True
                self._watching.remove(op)
                changed.add(op)

    def _fence(self, error: Exception) -> set[_Work]:
        with self._condition:
            self._fenced = self._draining = True
            changed = self._cancel_queued()
            for op in self._operations:
                if not op.ended:
                    if not op.results[0].concluded:
                        op.results = (faulted(str(error)),) * len(op.rows)
                    op.unprovable = True
                    changed.add(op)
            return changed

    def _finish_close(self) -> set[_Work]:
        with self._condition:
            changed = set()
            for op in self._operations:
                if not op.ended:
                    if not op.results[0].concluded:
                        op.results = (faulted("KVCR closed before completion"),) * len(op.rows)
                    op.ended = True
                    changed.add(op)
            return changed

    @staticmethod
    def _notify(changed: set[_Work]) -> None:
        for op in changed:
            if op.on_change is not None:
                try:
                    op.on_change(op)
                except Exception:
                    # A caller's broken callback cannot stop native cleanup/progress.
                    _LOGGER.exception("KVCR on_change callback raised")

    def _initialize_runtime(
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
        self._sdk_closed = False
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

    def _sdk_put(self, rows: RowBatch) -> int:
        """Deposit whole rows into KVCR-owned DRAM; return the SDK handle."""
        self._check_sdk_open()
        return self._runtime.deposit(self._references(rows))

    def _sdk_get(self, rows: RowBatch, hint: KvcrPeerHint | None = None) -> int:
        """Deliver whole rows into the caller's slots; return an opaque handle."""
        from kvcr.types import QueryStatus

        self._check_sdk_open()
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

    def _poll_completed(self) -> tuple[_Completion, ...]:
        """Process resilience events before publishing normalized row verdicts."""
        from kvcr.types import OpEntryStatus

        self._check_sdk_open()
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

    def _blocks_reuse(self, spans: Sequence[Span]) -> bool:
        """Whether exceptional accesses still quarantine any of these spans."""
        self._check_sdk_open()
        return self._access.blocks_reuse(spans)

    def _close_runtime(self) -> None:
        """Release the runtime; failure leaves caller allocations protected."""
        self._check_owner()
        if not self._sdk_closed:
            self._runtime.close()
            self._sdk_closed = True

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

    def _check_sdk_open(self) -> None:
        self._check_owner()
        if self._sdk_closed:
            raise RuntimeError("KVCR I/O is closed")
