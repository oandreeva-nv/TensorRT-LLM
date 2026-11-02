# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KVCR implementation of the blob backend's private StoreDriver interface."""

from __future__ import annotations

import ctypes
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from ..base.memory import MemoryKind
from .kvcr_peer import KvcrPeerHint
from .kvcr_worker import _KvcrWorker, _Work
from .store import (
    DriverCapabilities,
    DriverRejected,
    Hint,
    OnChange,
    OpHandle,
    ProbeStatus,
    RegionHandle,
    RegionLayout,
    RegionSpec,
    RegistrationError,
    RowBatch,
    RowResult,
    Span,
    StoreError,
)

__all__ = ["KvcrDriver"]


@dataclass(eq=False)
class _Batch:
    cache: ctypes.Array[ctypes.c_char] | None
    worker: _KvcrWorker | None = None
    revoking: bool = False
    revoked: bool = False
    submissions: int = 0


# A startup failure may leave native accesses with no worker that can revoke them.
_UNRELEASED_REGISTRATIONS: list[_Batch] = []


class KvcrDriver:
    """One local KVCR runtime behind the existing StoreDriver interface.

    Args:
        cache_slots: Number of whole objects the driver-owned DRAM cache holds.
        endpoint: Unique KVCR endpoint name.
        listen_port: NIXL listener port.
        max_rows_per_op: Maximum rows per queued operation, or None for unlimited.
        close_timeout_s: Maximum wait for each SDK unregistration attempt.
        control_port: Optional peer control listener, distinct from the NIXL port.
        control_host: Bind and advertise address for the peer control listener.

    Registers one host region, once per driver lifetime, through batch registration.
    The worker keeps the cache allocation alive until native teardown succeeds.
    The caller keeps the lender's region hold until registration is revoked.
    Peer hints require a control port. Sources serve already deposited KVCR DRAM;
    framework source pinning, device memory and multi-span rows are not supported. Failed operations
    without access-end proof remain unprovable and retain their registration.
    """

    def __init__(
        self,
        *,
        cache_slots: int,
        endpoint: str,
        listen_port: int,
        max_rows_per_op: int | None = 64,
        close_timeout_s: float = 5.0,
        control_port: int | None = None,
        control_host: str = "127.0.0.1",
    ) -> None:
        if type(cache_slots) is not int or cache_slots <= 0:
            raise ValueError("cache_slots must be a positive integer")
        if not endpoint or not 0 < listen_port < 65536:
            raise ValueError("an endpoint name and valid listener port are required")
        if control_port is not None and (
            type(control_port) is not int
            or not 0 < control_port < 65536
            or not isinstance(control_host, str)
            or not control_host
        ):
            raise ValueError("a valid control host and port are required for peer fetch")
        if not math.isfinite(close_timeout_s) or close_timeout_s < 0:
            raise ValueError("close_timeout_s must be finite and non-negative")
        self._capabilities = DriverCapabilities(
            get=True,
            put=True,
            cancel=False,
            probe=False,
            hints=control_port is not None,
            max_rows_per_op=max_rows_per_op,
        )
        self._cache_slots = cache_slots
        self._endpoint = endpoint
        self._listen_port = listen_port
        self._close_timeout_s = close_timeout_s
        self._control_options = (
            {}
            if control_port is None
            else {"control_port": control_port, "control_host": control_host}
        )
        self._lock = threading.Lock()
        self._batch: _Batch | None = None
        self._issued_batches: set[_Batch] = set()
        self._operations: dict[OpHandle, _KvcrWorker] = {}
        self._draining = False
        self._drained = False
        self._closed = False

    @property
    def capabilities(self) -> DriverCapabilities:
        return self._capabilities

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> RegionHandle:
        """Only batch registration can open the KVCR runtime."""
        raise DriverRejected("KVCR requires register_regions")

    def unregister(self, handle: RegionHandle) -> None:
        """Only batch unregistration can close the KVCR runtime."""
        raise DriverRejected("KVCR requires unregister_regions")

    def register_regions(self, specs: Sequence[RegionSpec]) -> RegionHandle:
        """Open KVCR with one region and allocate its separate DRAM cache."""
        with self._lock:
            if self._closed or self._draining or self._issued_batches:
                raise DriverRejected("KVCR accepts one batch per driver lifetime")
        if len(specs) != 1 or specs[0].layout.memory_kind is not MemoryKind.HOST:
            raise DriverRejected("KVCR requires exactly one host region")
        spec = specs[0]
        try:
            cache = ctypes.create_string_buffer(spec.layout.slot_bytes * self._cache_slots)
        except (MemoryError, OverflowError) as error:
            raise RegistrationError("could not allocate KVCR DRAM") from error
        span = Span(ctypes.addressof(cache), len(cache))
        if span.overlaps(spec.span):
            raise DriverRejected("KVCR cache overlaps the framework region")
        batch = _Batch(cache)
        with self._lock:
            self._batch = batch
            self._issued_batches.add(batch)
        try:
            worker = _KvcrWorker(
                spec,
                span,
                endpoint=self._endpoint,
                listen_port=self._listen_port,
                cache_owner=cache,
                **self._control_options,
            )
        except Exception as error:
            # Registration may have escaped; retain the owned cache and give a handle.
            _UNRELEASED_REGISTRATIONS.append(batch)
            raise RegistrationError("KVCR startup failed; retain the region hold", batch) from error
        with self._lock:
            batch.worker = worker
        return batch

    def unregister_regions(self, handle: RegionHandle) -> None:
        """Revoke only after every access ended; failed teardown stays retryable."""
        with self._lock:
            self._check_open()
            if not isinstance(handle, _Batch) or handle not in self._issued_batches:
                raise ValueError("unknown KVCR registration handle")
            if handle.revoked:
                return
            worker = handle.worker
            if worker is None:
                raise StoreError("KVCR startup was uncertain; retain the region hold")
            if handle.submissions or any(
                not worker.access_ended(op)
                for op, owner in self._operations.items()
                if owner is worker
            ):
                raise DriverRejected("KVCR operation can still access the region")
            handle.revoking = True
        if not worker.close(self._close_timeout_s):
            raise StoreError("KVCR teardown did not finish; retain the region hold and retry")
        with self._lock:
            handle.revoked = True
            handle.cache = None
            self._batch = None

    def put(self, rows: RowBatch, *, on_change: OnChange | None = None) -> OpHandle:
        """Queue a local publication through the owner worker."""
        return self._submit(rows, on_change, is_get=False)

    def get(
        self, rows: RowBatch, hint: Hint = None, *, on_change: OnChange | None = None
    ) -> OpHandle:
        """Queue a local fetch, or an explicitly configured peer fetch."""
        if hint is not None and (
            not self._capabilities.hints or not isinstance(hint, KvcrPeerHint)
        ):
            raise DriverRejected("KVCR peer fetch requires a control port and KvcrPeerHint")
        return self._submit(rows, on_change, is_get=True, hint=hint)

    def poll(self, op: OpHandle) -> tuple[RowResult, ...]:
        """Read cached results from the worker that issued this handle."""
        with self._lock:
            return self._owner(op).poll(op)

    def access_ended(self, op: OpHandle) -> bool:
        """Read cached access-end evidence."""
        with self._lock:
            return self._owner(op).access_ended(op)

    def access_unprovable(self, op: OpHandle) -> bool:
        """Read whether ordinary access-end proof was lost."""
        with self._lock:
            return self._owner(op).access_unprovable(op)

    def cancel(self, op: OpHandle) -> None:
        """Cancellation is not a declared capability."""
        raise DriverRejected("KVCR cancellation is not enabled")

    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        """Probing is not a declared capability."""
        raise DriverRejected("KVCR probing is not enabled")

    def drain(self, budget_s: float) -> bool:
        """Drain operations without closing the still-registered runtime."""
        with self._lock:
            self._check_open()
            self._draining = True
            batch = self._batch
        drained = batch is None or (batch.worker is not None and batch.worker.drain(budget_s))
        with self._lock:
            self._drained = drained and (batch is None or batch.submissions == 0)
            return self._drained

    def close(self) -> None:
        """Finish the driver after draining and revoking its registration."""
        with self._lock:
            if self._closed:
                return
            if not self._drained or self._batch is not None:
                raise DriverRejected("drain and unregister the KVCR region before close")
            self._closed = True

    def _submit(
        self,
        rows: RowBatch,
        on_change: OnChange | None,
        *,
        is_get: bool,
        hint: KvcrPeerHint | None = None,
    ) -> OpHandle:
        limit = self._capabilities.max_rows_per_op
        if limit is not None and len(rows) > limit:
            raise DriverRejected("KVCR row limit exceeded")
        with self._lock:
            self._check_open()
            batch = self._batch
            if self._draining or batch is None or batch.revoking or batch.worker is None:
                raise DriverRejected("KVCR region is unavailable or draining")
            worker = batch.worker
            batch.submissions += 1

        def changed(op: _Work) -> None:
            with self._lock:
                self._operations[op] = worker
            if on_change is not None:
                on_change(op)

        try:
            submit = worker.get if is_get else worker.put
            op = (
                submit(rows, on_change=changed)
                if hint is None
                else submit(rows, hint, on_change=changed)
            )
            with self._lock:
                self._operations[op] = worker
            return op
        finally:
            with self._lock:
                batch.submissions -= 1

    def _owner(self, op: OpHandle) -> _KvcrWorker:
        self._check_open()
        worker = self._operations.get(op)
        if worker is None:
            raise ValueError("unknown KVCR operation handle")
        return worker

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("KVCR driver is closed")
