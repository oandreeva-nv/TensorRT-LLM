# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner worker and cached operation state for local KVCR I/O."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Literal

from .kvcr_local_io import _KvcrLocalIO
from .kvcr_peer import KvcrPeerHint
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

_LOGGER = logging.getLogger(__name__)


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
    """Queue local I/O and publish cached results without calling KVCR on submit.

    Construction waits for registration on a single owner thread. ``put``, ``get``
    and queries use short state locks; the owner never holds them during SDK calls.
    The caller keeps framework and cache allocations alive until ``close`` returns
    True, including after startup failure. Failed rows remain protected until
    successful teardown; this component does not infer access end from a failure.
    An optional ``cache_owner`` keeps driver-owned cache memory alive until teardown.

    This is not yet a StoreDriver: registration is fixed at construction, failed
    access recovery is conservative, and close performs the SDK teardown itself.
    Cached queries remain available after teardown to inspect the final state.
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
            io = _KvcrLocalIO(
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
                    io.close()
                    changed = self._finish_close()
                    stopped = True
                elif not self._fenced:
                    if op is not None:
                        if op.method == "put":
                            handle = io.put(op.rows)
                        else:
                            handle = (
                                io.get(op.rows) if op.hint is None else io.get(op.rows, op.hint)
                            )
                        if handle in self._active:
                            raise RuntimeError("KVCR reused an active SDK handle")
                        self._active[handle] = op
                    self._progress(io, changed)
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

    def _progress(self, io: _KvcrLocalIO, changed: set[_Work]) -> None:
        for handle, entries in io.progress():
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
            if not io.blocks_reuse(op.rows.extents):
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
