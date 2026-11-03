# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track exceptional accesses to framework memory from KVCR resilience events."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Protocol

from .store import Span

if TYPE_CHECKING:
    from kvcr.types import TransferError


class _MemoryReference(Protocol):
    end_point_name: str
    label: str
    element_index: int


_Access = tuple[int, Literal["source", "destination"]]


class _KvcrAccessTracker:
    """Owner-thread quarantine for the regions named by one KVCR runtime.

    Pass ``on_resilience_event`` to the existing KVCR binding. ``resolve`` must
    validate framework references against live registrations and return their
    physical address ranges, including resolving labels that alias one buffer.
    Event IDs are local to this runtime and side, not public submission handles.

    This records exceptional hazards only. Absence of a hazard does not prove
    an operation has ended; the driver must also establish ordinary completion
    and stop its own accesses before allowing the lease to return. Unprovable
    accesses remain quarantined; cancellation and timeouts do not remove them.
    """

    def __init__(self, resolve: Callable[[_MemoryReference], Span]) -> None:
        self._resolve = resolve
        self._hazards: dict[_Access, frozenset[Span]] = {}

    def uncertain(self, access: _Access, spans: Sequence[Span]) -> None:
        memory = self._check_memory(access, spans)
        self._hazards[access] = memory

    def quiesced(self, access: _Access, spans: Sequence[Span]) -> None:
        self._check_memory(access, spans)
        self._hazards.pop(access, None)

    def blocks_reuse(self, spans: Sequence[Span]) -> bool:
        return any(
            protected.overlaps(candidate)
            for memory in self._hazards.values()
            for protected in memory
            for candidate in spans
        )

    def _check_memory(self, access: _Access, spans: Sequence[Span]) -> frozenset[Span]:
        memory = frozenset(spans)
        if not memory:
            raise ValueError("an access must identify memory")
        previous = self._hazards.get(access)
        if previous is not None and previous != memory:
            raise ValueError("memory changed for an existing access")
        return memory

    def on_resilience_event(self, error: Exception) -> None:
        """Consume memory events; let other failures reach the driver's fence.

        Invoke on the KVCR owner thread before interpreting returned data
        results. The driver owns synchronization with its nonblocking queries.
        """
        from kvcr.types import TransferError

        if not isinstance(error, TransferError):
            raise error
        if error.state not in ("uncertain", "quiesced"):
            raise ValueError("unknown KVCR access state")
        accesses = self._framework_spans(error)
        # Validate the entire event before clearing any part of its protection.
        for direction, spans in accesses.items():
            self._check_memory((error.op_handle, direction), spans)
        for direction, spans in accesses.items():
            access = (error.op_handle, direction)
            if error.state == "uncertain":
                self.uncertain(access, spans)
            else:
                self.quiesced(access, spans)

    def _framework_spans(
        self, error: TransferError
    ) -> dict[Literal["source", "destination"], tuple[Span, ...]]:
        if error.source_blocks is None and error.destination_regions is None:
            raise ValueError("a memory event must identify affected memory")
        accesses: dict[Literal["source", "destination"], tuple[Span, ...]] = {}
        sources = self._source_spans(error)
        if sources:
            accesses["source"] = sources
        destinations = self._destination_spans(error)
        if destinations:
            accesses["destination"] = destinations
        return accesses

    def _source_spans(self, error: TransferError) -> tuple[Span, ...]:
        blocks = error.source_blocks
        if blocks is None:
            return ()
        owners = error.source_owners
        if owners is None or owners.keys() != blocks.keys():
            raise ValueError("source memory ownership is incomplete")
        spans = []
        for key, refs in blocks.items():
            owner = owners[key]
            if owner not in ("framework", "kvcr"):
                raise ValueError("unknown source memory ownership")
            if owner == "framework":
                if not refs:
                    raise ValueError("framework source memory is missing")
                spans.extend(self._resolve(ref) for ref in refs)
        return tuple(spans)

    def _destination_spans(self, error: TransferError) -> tuple[Span, ...]:
        if error.destination_regions is None:
            return ()
        if error.destination_owner not in ("framework", "kvcr"):
            raise ValueError("destination memory ownership is incomplete")
        if error.destination_owner == "kvcr":
            return ()
        if not error.destination_regions:
            raise ValueError("framework destination memory is missing")
        return tuple(self._resolve(ref) for ref in error.destination_regions)
