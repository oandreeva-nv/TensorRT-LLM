# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

from .remote_g2 import RemoteG2BindingRecord, RemoteG2Descriptor


class RemoteG2TransferError(RuntimeError):
    pass


class RemoteG2TransferContractError(RemoteG2TransferError):
    """A transfer result cannot prove that its handle is safe to release."""

    def __init__(self, message: str, transfer_result: Any) -> None:
        super().__init__(message)
        self.transfer_result = transfer_result


class RemoteG2TransferState(Enum):
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def validate_remote_g2_transfer_result(result: Any) -> Any:
    """Fail closed while retaining ownership of a possibly-live result."""
    missing = []
    if not isinstance(getattr(result, "initial_state", None), RemoteG2TransferState):
        missing.append("initial_state")
    for method_name in ("poll_state", "quiesce"):
        if not callable(getattr(result, method_name, None)):
            missing.append(method_name)
    if missing:
        raise RemoteG2TransferContractError(
            "remote G2 transfer result lacks retryable cleanup contract: " + ", ".join(missing),
            result,
        )
    return result


@dataclass(frozen=True)
class RemoteG2SourceMetadata:
    source_worker_id: int
    source_generation: int
    remote_name: str
    agent_desc: bytes
    backend: str = "NIXL"


@dataclass(frozen=True)
class RemoteG2TransferDescriptor:
    ptr: int
    size: int
    device_id: int
    memory_type: str
    name: str = ""

    @classmethod
    def from_source_descriptor(cls, descriptor: RemoteG2Descriptor) -> "RemoteG2TransferDescriptor":
        metadata = descriptor.metadata
        raw = metadata.get("nixl_memory_desc")
        if isinstance(raw, Mapping):
            ptr = raw.get("ptr", metadata.get("ptr"))
            size = raw.get("size", metadata.get("size", descriptor.byte_length))
            device_id = raw.get("device_id", metadata.get("device_id", 0))
            memory_type = raw.get(
                "memory_type", raw.get("type", metadata.get("memory_type", "DRAM"))
            )
            name = raw.get("name", metadata.get("name", descriptor.pool_id))
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            if len(raw) < 3:
                raise RemoteG2TransferError("source descriptor tuple is incomplete")
            ptr, size, device_id = raw[:3]
            memory_type = raw[3] if len(raw) >= 4 else metadata.get("memory_type", "DRAM")
            name = raw[4] if len(raw) >= 5 else metadata.get("name", descriptor.pool_id)
        else:
            ptr = metadata.get("ptr")
            size = metadata.get("size", descriptor.byte_length)
            device_id = metadata.get("device_id", 0)
            memory_type = metadata.get("memory_type", "DRAM")
            name = metadata.get("name", descriptor.pool_id)

        if ptr is None:
            raise RemoteG2TransferError("source descriptor missing transfer pointer")
        return cls(
            ptr=int(ptr),
            size=int(size),
            device_id=int(device_id),
            memory_type=_canonical_memory_type(str(memory_type)),
            name=str(name),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RemoteG2TransferDescriptor":
        return cls(
            ptr=int(data["ptr"]),
            size=int(data["size"]),
            device_id=int(data.get("device_id", 0)),
            memory_type=_canonical_memory_type(str(data["memory_type"])),
            name=str(data.get("name", "")),
        )

    def transfer_tuple(self) -> tuple[int, int, int]:
        return (self.ptr, self.size, self.device_id)

    def registration_tuple(self) -> tuple[int, int, int, str]:
        return (
            self.ptr,
            self.size,
            self.device_id,
            self.name or f"remote_g2_target_{self.ptr:x}",
        )


@dataclass
class RemoteG2TransferResult:
    record: RemoteG2BindingRecord
    source_metadata: RemoteG2SourceMetadata
    source_descs: tuple[RemoteG2TransferDescriptor, ...]
    target_descs: tuple[RemoteG2TransferDescriptor, ...]
    status: Any
    agent: Any
    target_registration: Any
    released: bool = False

    def is_completed(self) -> bool:
        return bool(self.status.is_completed())

    def wait(self, timeout_ms: Optional[int] = None) -> bool:
        return bool(self.status.wait(timeout_ms))

    def release(self) -> None:
        if self.released:
            return
        self.agent.deregister_memory(self.target_registration)
        self.released = True


class RemoteG2SourceMetadataCache:
    def __init__(self) -> None:
        self._metadata: dict[tuple[int, int], RemoteG2SourceMetadata] = {}

    def get_or_refresh(
        self,
        source_worker_id: int,
        source_generation: int,
        fetcher: Callable[[int, int], RemoteG2SourceMetadata],
    ) -> RemoteG2SourceMetadata:
        key = (int(source_worker_id), int(source_generation))
        cached = self._metadata.get(key)
        if cached is not None:
            return cached

        metadata = fetcher(int(source_worker_id), int(source_generation))
        if metadata.source_worker_id != int(source_worker_id):
            raise RemoteG2TransferError("source metadata worker mismatch")
        if metadata.source_generation != int(source_generation):
            raise RemoteG2TransferError("source metadata generation mismatch")
        if not metadata.agent_desc:
            raise RemoteG2TransferError("source metadata missing agent descriptor")

        self._metadata[key] = metadata
        return metadata


class RemoteG2NixlTransferAdapter:
    supports_synchronous_release = False

    def __init__(
        self,
        *,
        source_metadata_fetcher: Callable[[int, int], RemoteG2SourceMetadata],
        target_descriptor_resolver: Callable[
            [RemoteG2BindingRecord], Sequence[RemoteG2TransferDescriptor | Mapping[str, Any]]
        ],
        source_metadata_cache: Optional[RemoteG2SourceMetadataCache] = None,
        agent_factory: Optional[Callable[[], Any]] = None,
        transfer_types: Optional[Any] = None,
        agent_name: str = "remote-g2-target",
    ) -> None:
        self._source_metadata_fetcher = source_metadata_fetcher
        self._target_descriptor_resolver = target_descriptor_resolver
        self._source_metadata_cache = source_metadata_cache or RemoteG2SourceMetadataCache()
        self._agent_factory = agent_factory
        self._transfer_types = transfer_types
        self._agent_name = agent_name
        self._agent: Optional[Any] = None

    def start_transfer(self, record: RemoteG2BindingRecord) -> RemoteG2TransferResult:
        if not record.is_transfer_ready:
            raise RemoteG2TransferError("remote G2 binding is not transfer-ready")

        source_metadata = self._source_metadata_cache.get_or_refresh(
            record.plan.source_worker_id,
            record.source_generation,
            self._source_metadata_fetcher,
        )
        source_descs = tuple(
            RemoteG2TransferDescriptor.from_source_descriptor(block.source_descriptor)
            for block in record.bound_blocks
        )
        target_descs = tuple(
            self._normalize_target_descriptor(descriptor)
            for descriptor in self._target_descriptor_resolver(record)
        )
        self._validate_descriptors(source_descs, target_descs)

        types = self._get_transfer_types()
        agent = self._get_agent(types)
        agent.load_remote_agent(source_metadata.remote_name, source_metadata.agent_desc)
        target_registration = types.RegMemoryDescs(
            "VRAM", [descriptor.registration_tuple() for descriptor in target_descs]
        )
        agent.register_memory(target_registration)
        request = types.TransferRequest(
            types.TransferOp.READ,
            types.MemoryDescs("DRAM", [descriptor.transfer_tuple() for descriptor in source_descs]),
            types.MemoryDescs("VRAM", [descriptor.transfer_tuple() for descriptor in target_descs]),
            source_metadata.remote_name,
        )
        status = agent.submit_transfer_requests(request)
        return RemoteG2TransferResult(
            record=record,
            source_metadata=source_metadata,
            source_descs=source_descs,
            target_descs=target_descs,
            status=status,
            agent=agent,
            target_registration=target_registration,
        )

    def _get_transfer_types(self) -> Any:
        if self._transfer_types is not None:
            return self._transfer_types

        from tensorrt_llm._torch.disaggregation.base.agent import (  # noqa: PLC0415
            MemoryDescs,
            RegMemoryDescs,
            TransferOp,
            TransferRequest,
        )
        from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent  # noqa: PLC0415

        self._transfer_types = SimpleNamespace(
            MemoryDescs=MemoryDescs,
            RegMemoryDescs=RegMemoryDescs,
            TransferOp=TransferOp,
            TransferRequest=TransferRequest,
            NixlTransferAgent=NixlTransferAgent,
        )
        return self._transfer_types

    def _get_agent(self, types: Any) -> Any:
        if self._agent is None:
            self._agent = (
                self._agent_factory()
                if self._agent_factory is not None
                else types.NixlTransferAgent(self._agent_name)
            )
        return self._agent

    def _normalize_target_descriptor(
        self, descriptor: RemoteG2TransferDescriptor | Mapping[str, Any]
    ) -> RemoteG2TransferDescriptor:
        if isinstance(descriptor, RemoteG2TransferDescriptor):
            return descriptor
        return RemoteG2TransferDescriptor.from_mapping(descriptor)

    def _validate_descriptors(
        self,
        source_descs: tuple[RemoteG2TransferDescriptor, ...],
        target_descs: tuple[RemoteG2TransferDescriptor, ...],
    ) -> None:
        if not source_descs:
            raise RemoteG2TransferError("remote G2 transfer has no source descriptors")
        if len(source_descs) != len(target_descs):
            raise RemoteG2TransferError("source and target descriptor counts differ")
        for source, target in zip(source_descs, target_descs):
            if source.size != target.size:
                raise RemoteG2TransferError("source and target descriptor sizes differ")
            if source.memory_type != "DRAM":
                raise RemoteG2TransferError("remote G2 source descriptor must be DRAM")
            if target.memory_type != "VRAM":
                raise RemoteG2TransferError("target descriptor must be VRAM")


def _canonical_memory_type(memory_type: str) -> str:
    value = memory_type.strip().upper()
    if value in {"CPU", "HOST", "HOST_PINNED", "DRAM"}:
        return "DRAM"
    if value in {"GPU", "CUDA", "G1", "VRAM"}:
        return "VRAM"
    return value
