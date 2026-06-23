# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional

try:
    from .remote_g2_observability import (
        NullRemoteG2ObservabilitySink,
        RemoteG2LifecycleEvent,
        RemoteG2ObservabilitySink,
    )
except ImportError:
    from remote_g2_observability import (  # type: ignore[no-redef]
        NullRemoteG2ObservabilitySink,
        RemoteG2LifecycleEvent,
        RemoteG2ObservabilitySink,
    )

REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY = "remote_kv_reuse_plan"
REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY = "remote_kv_reuse_no_plan_reason"
REMOTE_KV_REUSE_PLAN_VERSION = 1
REMOTE_G2_REUSE_ENABLED_ENV = "DYN_REMOTE_G2_REUSE_ENABLED"

_REMOTE_G2_TIERS = {"g2", "host_pinned", "hostpinned", "cpu_pinned", "cpu_tier1"}
_CACHE_TIER_PRIMARY = "primary"
_CACHE_TIER_HOST_PINNED = "host_pinned"


def _now_ms() -> int:
    return int(time.time() * 1000)


def remote_g2_reuse_enabled() -> bool:
    value = os.getenv(REMOTE_G2_REUSE_ENABLED_ENV)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _normalize_request_id(request_id: int | str) -> int | str:
    try:
        return int(request_id)
    except (TypeError, ValueError):
        return str(request_id)


def _normalize_tier(tier: str) -> str:
    return tier.lower().replace("-", "_")


def _is_remote_g2_tier(tier: str) -> bool:
    return _normalize_tier(tier) in _REMOTE_G2_TIERS


@dataclass(frozen=True)
class RemoteKvReusePlan:
    plan_id: str
    request_id: str
    target_worker_id: int
    target_dp_rank: int
    source_worker_id: int
    source_dp_rank: int
    source_tier: str
    block_hashes: tuple[int, ...]
    start_block_index: int
    planned_prefix_blocks: int
    block_size_tokens: int
    created_at_ms: int
    expires_at_ms: int
    plan_version: int = REMOTE_KV_REUSE_PLAN_VERSION
    # Parallel to `block_hashes`, but carrying the source worker's
    # KV-cache-manager-side hash (TRT-LLM splitmix) rather than the
    # router-side hash (XXH3 tokens_hash). The source side uses these
    # values to look up blocks; the router-side block_hashes are kept
    # for plan identity. Empty when the producer has not been updated
    # to populate the new field — in that case the source side falls
    # back to using `block_hashes` for the lookup (legacy behavior).
    kv_block_hashes: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RemoteKvReusePlan":
        missing = {
            field_name
            for field_name in (
                "plan_id",
                "request_id",
                "target_worker_id",
                "target_dp_rank",
                "source_worker_id",
                "source_dp_rank",
                "source_tier",
                "start_block_index",
                "block_hashes",
                "planned_prefix_blocks",
                "block_size_tokens",
                "created_at_ms",
                "expires_at_ms",
            )
            if field_name not in data
        }
        if missing:
            raise ValueError(f"remote G2 plan missing fields: {sorted(missing)}")

        block_hashes = tuple(int(block_hash) for block_hash in data["block_hashes"])
        kv_block_hashes_raw = data.get("kv_block_hashes", ())
        kv_block_hashes = tuple(int(h) for h in kv_block_hashes_raw)
        if kv_block_hashes and len(kv_block_hashes) != len(block_hashes):
            raise ValueError(
                "kv_block_hashes length must match block_hashes when provided"
            )
        planned_prefix_blocks = int(data["planned_prefix_blocks"])
        if planned_prefix_blocks < 0:
            raise ValueError("planned_prefix_blocks must be non-negative")
        start_block_index = int(data["start_block_index"])
        if start_block_index < 0:
            raise ValueError("start_block_index must be non-negative")

        return cls(
            plan_id=str(data["plan_id"]),
            request_id=str(data["request_id"]),
            target_worker_id=int(data["target_worker_id"]),
            target_dp_rank=int(data["target_dp_rank"]),
            source_worker_id=int(data["source_worker_id"]),
            source_dp_rank=int(data["source_dp_rank"]),
            source_tier=str(data["source_tier"]),
            block_hashes=block_hashes,
            kv_block_hashes=kv_block_hashes,
            start_block_index=start_block_index,
            planned_prefix_blocks=min(planned_prefix_blocks, len(block_hashes)),
            block_size_tokens=int(data["block_size_tokens"]),
            created_at_ms=int(data["created_at_ms"]),
            expires_at_ms=int(data["expires_at_ms"]),
            plan_version=int(data.get("plan_version", REMOTE_KV_REUSE_PLAN_VERSION)),
        )

    def is_remote_g2(self) -> bool:
        return _is_remote_g2_tier(self.source_tier)

    def is_expired(self, now_ms: Optional[int] = None) -> bool:
        return self.expires_at_ms <= (now_ms if now_ms is not None else _now_ms())

    @property
    def planned_hashes(self) -> tuple[int, ...]:
        return self.block_hashes[: self.planned_prefix_blocks]

    @property
    def planned_kv_block_hashes(self) -> tuple[int, ...]:
        return self.kv_block_hashes[: self.planned_prefix_blocks]


@dataclass
class TargetRemotePlanEntry:
    plan: RemoteKvReusePlan
    resolved: Optional["RemoteG2ResolveResult"] = None


class TargetRemotePlanStore:
    """Target-worker request-scoped remote G2 plan cache.

    The key is the TensorRT-LLM numeric request id, not Dynamo extra_args.
    Dynamo only uses extra_args to carry the router plan to target admission.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        clock_ms: Callable[[], int] = _now_ms,
        observability: Optional[RemoteG2ObservabilitySink] = None,
    ) -> None:
        self.enabled = enabled
        self._clock_ms = clock_ms
        self._observability = observability or NullRemoteG2ObservabilitySink()
        self._plans: dict[int | str, TargetRemotePlanEntry] = {}
        self._lock = threading.RLock()

    def put(
        self, trtllm_request_id: int | str, plan: Mapping[str, Any] | RemoteKvReusePlan
    ) -> Optional[RemoteKvReusePlan]:
        if not self.enabled or not remote_g2_reuse_enabled():
            return None
        try:
            parsed = (
                plan
                if isinstance(plan, RemoteKvReusePlan)
                else RemoteKvReusePlan.from_dict(plan)
            )
        except (TypeError, ValueError):
            return None
        # PROBE: confirm the target worker received the plan and parsed the
        # new kv_block_hashes field from the wire. Investigation-only.
        import logging as _logging
        import os as _os
        _logging.warning(
            "PROBE rpc_chain put_plan pid=%d trtllm_req_id=%s plan_id=%s "
            "source=%s/%s target=%s/%s block_hashes_count=%d "
            "kv_block_hashes_count=%d store_id=%d",
            _os.getpid(),
            trtllm_request_id,
            parsed.plan_id,
            parsed.source_worker_id,
            parsed.source_dp_rank,
            parsed.target_worker_id,
            parsed.target_dp_rank,
            len(parsed.block_hashes),
            len(parsed.kv_block_hashes),
            id(_GLOBAL_TARGET_PLAN_STORE),
        )
        now_ms = self._clock_ms()
        if not parsed.is_remote_g2() or parsed.is_expired(now_ms):
            return None

        with self._lock:
            self._plans[_normalize_request_id(trtllm_request_id)] = TargetRemotePlanEntry(parsed)
        self._observability.emit(
            RemoteG2LifecycleEvent(
                event="planned",
                reason="ok",
                tier=parsed.source_tier,
                outcome="accepted",
                request_id=trtllm_request_id,
                plan_id=parsed.plan_id,
                source_worker_id=parsed.source_worker_id,
                block_count=parsed.planned_prefix_blocks,
                token_count=parsed.planned_prefix_blocks * parsed.block_size_tokens,
            )
        )
        return parsed

    def get(self, trtllm_request_id: int | str) -> Optional[RemoteKvReusePlan]:
        key = _normalize_request_id(trtllm_request_id)
        with self._lock:
            entry = self._plans.get(key)
            if entry is None:
                return None
            if entry.plan.is_expired(self._clock_ms()):
                self._plans.pop(key, None)
                return None
            return entry.plan

    def bind_resolution(
        self, trtllm_request_id: int | str, result: "RemoteG2ResolveResult"
    ) -> None:
        key = _normalize_request_id(trtllm_request_id)
        with self._lock:
            entry = self._plans[key]
            entry.resolved = result

    def get_resolution(
        self, trtllm_request_id: int | str
    ) -> Optional["RemoteG2ResolveResult"]:
        key = _normalize_request_id(trtllm_request_id)
        with self._lock:
            entry = self._plans.get(key)
            return None if entry is None else entry.resolved

    def discard(self, trtllm_request_id: int | str) -> None:
        with self._lock:
            self._plans.pop(_normalize_request_id(trtllm_request_id), None)

    def clear(self) -> None:
        with self._lock:
            self._plans.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._plans)


_GLOBAL_TARGET_PLAN_STORE = TargetRemotePlanStore()


def target_remote_g2_plan_store() -> TargetRemotePlanStore:
    return _GLOBAL_TARGET_PLAN_STORE


@dataclass
class SourceG2DescriptorRecord:
    block_hash: int
    source_worker_id: int
    source_dp_rank: int
    tier: str
    descriptor_generation: int
    pool_id: str
    byte_offset: int
    byte_length: int
    block_id: int = -1
    live: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    lease_count: int = 0
    # True when find_and_pin_blocks_by_hash already bumped refcount on
    # this block; the resolve-time acquire_pin should not pin again.
    _pinned_by_lookup: bool = False

    def is_resolvable_for(self, source_worker_id: int, source_dp_rank: int) -> bool:
        return (
            self.live
            and self.source_worker_id == source_worker_id
            and self.source_dp_rank == source_dp_rank
            and _is_remote_g2_tier(self.tier)
        )


@dataclass(frozen=True)
class RemoteG2Descriptor:
    block_hash: int
    descriptor_generation: int
    pool_id: str
    byte_offset: int
    byte_length: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteG2BlockStatus:
    block_hash: int
    status: str
    descriptor_generation: Optional[int] = None


@dataclass(frozen=True)
class PinnedCacheBlock:
    block_hash: int
    block_id: int
    slot_idx: int
    tier: str = _CACHE_TIER_HOST_PINNED


@dataclass(frozen=True)
class CacheMiss:
    block_hash: int
    found_tier: Optional[str] = None


CacheLookupResult = PinnedCacheBlock | CacheMiss


@dataclass
class RemoteG2Lease:
    lease_id: str
    plan_id: str
    request_id: str
    target_worker_id: int
    target_dp_rank: int
    block_hashes: tuple[int, ...]
    descriptor_generations: tuple[int, ...]
    expires_at_ms: int
    trtllm_pin_refs: tuple[Any, ...] = ()
    released: bool = False
    release_reason: Optional[str] = None


@dataclass(frozen=True)
class RemoteG2ResolveResult:
    lease_id: Optional[str]
    descriptors: tuple[RemoteG2Descriptor, ...]
    num_tokens: int
    reason: str = "ok"
    source_generation: int = 0
    per_block_status: tuple[RemoteG2BlockStatus, ...] = ()
    # T1: Per-rank descriptors for TP>1. Keyed by tp_rank (int).
    # Each value is a tuple of descriptors for that rank's KV slice.
    # TP=1 callers can ignore this (empty dict).
    per_rank_descriptors: dict = field(default_factory=dict)
    # T2: Per-rank source metadata for TP>1. Keyed by tp_rank (int).
    # Each value is a dict with keys: remote_name, agent_metadata_b64,
    # pool_base_ptr, pool_size_bytes, source_generation.
    per_rank_source_metadata: dict = field(default_factory=dict)


class RemoteG2BindingState(str, Enum):
    RESOLVED = "resolved"
    BOUND = "bound"
    BIND_FAILED = "bind_failed"
    RELEASED = "released"
    CANCELLED = "cancelled"
    TRANSFER_FAILED = "transfer_failed"


@dataclass(frozen=True)
class RemoteG2BoundBlock:
    source_descriptor: RemoteG2Descriptor
    target_block_id: int
    source_block_index: int
    target_block_index: int
    # Primary-pool slot index for this block, used as the NIXL local
    # dlist index. Engine-allocated block_id is globally unique and
    # may exceed the primary pool's slot count (which dimensions the
    # dlist); the slot_idx is the right dense index. -1 means
    # unresolved (legacy bindings created without the lookup).
    target_slot_idx: int = -1


@dataclass
class RemoteG2BindingRecord:
    request_id: int | str
    plan: RemoteKvReusePlan
    resolve_result: RemoteG2ResolveResult
    matched_tokens: int
    num_computed_tokens: int
    block_size_tokens: int
    state: RemoteG2BindingState = RemoteG2BindingState.RESOLVED
    bound_blocks: tuple[RemoteG2BoundBlock, ...] = ()
    release_attempted: bool = False
    release_completed: bool = False
    release_reason: Optional[str] = None

    @property
    def lease_id(self) -> Optional[str]:
        return self.resolve_result.lease_id

    @property
    def source_generation(self) -> int:
        return self.resolve_result.source_generation

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            RemoteG2BindingState.BIND_FAILED,
            RemoteG2BindingState.RELEASED,
            RemoteG2BindingState.CANCELLED,
            RemoteG2BindingState.TRANSFER_FAILED,
        }

    @property
    def is_transfer_ready(self) -> bool:
        return self.state is RemoteG2BindingState.BOUND and bool(self.bound_blocks)


def compute_remote_g2_matched_tokens(
    plan: RemoteKvReusePlan,
    result: RemoteG2ResolveResult,
    num_computed_tokens: int,
    block_size_tokens: int,
) -> int:
    if result.lease_id is None:
        return 0
    if block_size_tokens <= 0 or num_computed_tokens < 0:
        return 0
    if num_computed_tokens % block_size_tokens != 0:
        return 0

    computed_blocks = num_computed_tokens // block_size_tokens
    # B's prefix ends before the plan begins → gap. Attaching would put the
    # plan's blocks at the wrong target positions (silent KV corruption).
    if computed_blocks < plan.start_block_index:
        return 0

    resolved_tokens = min(
        max(result.num_tokens, 0), len(result.descriptors) * block_size_tokens
    )
    resolved_blocks = resolved_tokens // block_size_tokens
    plan_end = plan.start_block_index + resolved_blocks
    # B already has every position the plan covers → nothing to transfer.
    if computed_blocks >= plan_end:
        return 0

    skip = computed_blocks - plan.start_block_index
    matched_blocks = resolved_blocks - skip
    return matched_blocks * block_size_tokens


class TargetRemoteG2BindingStore:
    """Request-scoped target binding state for remote G2 reuse."""

    def __init__(
        self,
        release_lease: Callable[[str, str], bool],
        *,
        observability: Optional[RemoteG2ObservabilitySink] = None,
    ) -> None:
        self._release_lease = release_lease
        self._observability = observability or NullRemoteG2ObservabilitySink()
        self._records: dict[int | str, RemoteG2BindingRecord] = {}
        self._lock = threading.RLock()

    def resolve_for_request(
        self,
        request_id: int | str,
        plan: Mapping[str, Any] | RemoteKvReusePlan,
        num_computed_tokens: int,
        resolve_and_lease: Callable[[RemoteKvReusePlan], RemoteG2ResolveResult],
    ) -> Optional[RemoteG2BindingRecord]:
        key = _normalize_request_id(request_id)
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return None if existing.is_terminal else existing

        try:
            parsed = (
                plan
                if isinstance(plan, RemoteKvReusePlan)
                else RemoteKvReusePlan.from_dict(plan)
            )
        except (TypeError, ValueError):
            return None

        result = resolve_and_lease(parsed)
        matched_tokens = compute_remote_g2_matched_tokens(
            parsed, result, num_computed_tokens, parsed.block_size_tokens
        )
        planned_tokens = parsed.planned_prefix_blocks * parsed.block_size_tokens
        if matched_tokens == 0:
            reason = self._zero_match_release_reason(
                num_computed_tokens, parsed.block_size_tokens
            )
            if planned_tokens > 0:
                self._emit_event(
                    "truncated",
                    parsed,
                    result,
                    request_id=key,
                    reason=result.reason,
                    outcome="reduced",
                    token_count=0,
                )
            self._emit_event(
                "fallback",
                parsed,
                result,
                request_id=key,
                reason=reason,
                outcome="local_recompute",
                token_count=0,
            )
            self._release_positive_lease(
                result,
                reason,
                parsed,
                key,
            )
            return None

        record = RemoteG2BindingRecord(
            request_id=key,
            plan=parsed,
            resolve_result=result,
            matched_tokens=matched_tokens,
            num_computed_tokens=num_computed_tokens,
            block_size_tokens=parsed.block_size_tokens,
        )
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                self._release_record_once(
                    record,
                    "duplicate_resolve",
                    RemoteG2BindingState.RELEASED,
                )
                return None if existing.is_terminal else existing
            self._records[key] = record
        self._emit_event(
            "resolved",
            parsed,
            result,
            request_id=key,
            reason=result.reason,
            outcome="accepted",
            token_count=matched_tokens,
            block_count=matched_tokens // parsed.block_size_tokens,
        )
        if matched_tokens < planned_tokens or result.reason != "ok":
            self._emit_event(
                "truncated",
                parsed,
                result,
                request_id=key,
                reason=result.reason,
                outcome="reduced",
                token_count=matched_tokens,
                block_count=matched_tokens // parsed.block_size_tokens,
            )
        return record

    def bind_target_blocks(
        self, request_id: int | str, block_ids: list[int] | tuple[int, ...]
    ) -> Optional[RemoteG2BindingRecord]:
        key = _normalize_request_id(request_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if record.state is RemoteG2BindingState.BOUND or record.is_terminal:
                return record

            block_size = record.block_size_tokens
            if block_size <= 0 or record.num_computed_tokens % block_size != 0:
                self._release_record_once(
                    record,
                    "target_binding_failed",
                    RemoteG2BindingState.BIND_FAILED,
                )
                return record

            plan_start = record.plan.start_block_index
            computed_blocks = record.num_computed_tokens // block_size
            matched_blocks = record.matched_tokens // block_size
            # `skip` is how many of the plan's blocks B already has on Device;
            # those descriptors are dropped from the front of the transfer.
            # compute_remote_g2_matched_tokens guarantees skip >= 0 and
            # skip + matched_blocks <= len(descriptors).
            skip = computed_blocks - plan_start
            source_end = skip + matched_blocks
            source_descriptors = record.resolve_result.descriptors[skip:source_end]
            target_end = computed_blocks + matched_blocks

            if (
                skip < 0
                or len(block_ids) < target_end
                or len(source_descriptors) != matched_blocks
            ):
                self._emit_record_event(
                    "fallback",
                    record,
                    reason="target_binding_failed",
                    outcome="local_recompute",
                )
                self._release_record_once(
                    record,
                    "target_binding_failed",
                    RemoteG2BindingState.BIND_FAILED,
                )
                return record

            target_block_ids = block_ids[computed_blocks:target_end]

            # Resolve engine block_id → primary-pool slot_idx for each
            # bound block. NIXL's local dlist is dense over primary
            # pool slots, so we have to index by slot, not block_id.
            # The lookup callable is installed by the target setup
            # (remote_g2_connector._installed_block_id_to_slot_idx),
            # since the engine block_id namespace is wider than the
            # primary pool's slot space and we cannot use block_ids
            # directly as NIXL local-dlist indices.
            target_slot_indices: list[int] = []
            target_block_ids_list = [int(b) for b in target_block_ids]
            if target_block_ids_list:
                from .remote_g2_connector import _installed_block_id_to_slot_idx
                if _installed_block_id_to_slot_idx is not None:
                    try:
                        target_slot_indices = list(
                            _installed_block_id_to_slot_idx(target_block_ids_list)
                        )
                    except Exception:
                        target_slot_indices = []
            if len(target_slot_indices) != len(target_block_ids_list):
                # Couldn't resolve all slots; abort the binding so the
                # request falls back to local recompute rather than
                # issuing NIXL with stale/garbage indices.
                self._release_record_once(
                    record,
                    "target_slot_lookup_failed",
                    RemoteG2BindingState.BIND_FAILED,
                )
                return record

            record.bound_blocks = tuple(
                RemoteG2BoundBlock(
                    source_descriptor=descriptor,
                    target_block_id=int(target_block_id),
                    source_block_index=plan_start + skip + offset,
                    target_block_index=computed_blocks + offset,
                    target_slot_idx=target_slot_indices[offset],
                )
                for offset, (descriptor, target_block_id) in enumerate(
                    zip(source_descriptors, target_block_ids)
                )
            )
            record.state = RemoteG2BindingState.BOUND
            return record

    def release(self, request_id: int | str, reason: str = "released") -> bool:
        with self._lock:
            record = self._records.get(_normalize_request_id(request_id))
            if record is None:
                return False
            return self._release_record_once(
                record, reason, self._terminal_state_for_release(reason)
            )

    def discard(self, request_id: int | str, reason: str = "discarded") -> bool:
        key = _normalize_request_id(request_id)
        with self._lock:
            record = self._records.pop(key, None)
            if record is None:
                return False
            return self._release_record_once(
                record, reason, self._terminal_state_for_release(reason)
            )

    def get(self, request_id: int | str) -> Optional[RemoteG2BindingRecord]:
        with self._lock:
            return self._records.get(_normalize_request_id(request_id))

    def iter_records(self):
        """Snapshot of (request_id, record) for all currently-tracked
        records. Snapshot is taken under the lock; iteration is safe
        without it. Used by the connector to scan for transfer-ready
        bindings without relying on the scheduler_output filtering."""
        with self._lock:
            return list(self._records.items())

    def clear(self) -> None:
        with self._lock:
            records = tuple(self._records.values())
            self._records.clear()
        for record in records:
            self._release_record_once(record, "clear", RemoteG2BindingState.RELEASED)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def _release_record_once(
        self,
        record: RemoteG2BindingRecord,
        reason: str,
        state: RemoteG2BindingState,
    ) -> bool:
        if record.release_attempted:
            self._emit_record_event(
                "released", record, reason=reason, outcome="already_released"
            )
            return False

        record.release_attempted = True
        record.release_reason = reason
        record.state = state
        lease_id = record.lease_id
        if lease_id is None:
            self._emit_record_event(
                "released", record, reason=reason, outcome="no_lease"
            )
            return False

        record.release_completed = self._release_lease(lease_id, reason)
        self._emit_record_event(
            "released",
            record,
            reason=reason,
            outcome="completed" if record.release_completed else "already_released",
        )
        return record.release_completed

    def _release_positive_lease(
        self,
        result: RemoteG2ResolveResult,
        reason: str,
        plan: Optional[RemoteKvReusePlan] = None,
        request_id: Optional[int | str] = None,
    ) -> None:
        if result.lease_id is not None:
            completed = self._release_lease(result.lease_id, reason)
            self._observability.emit(
                RemoteG2LifecycleEvent(
                    event="released",
                    reason=reason,
                    tier=plan.source_tier if plan is not None else "g2",
                    outcome="completed" if completed else "already_released",
                    request_id=request_id,
                    plan_id=plan.plan_id if plan is not None else None,
                    lease_id=result.lease_id,
                    source_worker_id=(
                        plan.source_worker_id if plan is not None else None
                    ),
                    source_generation=result.source_generation,
                    block_count=len(result.descriptors),
                    byte_count=sum(
                        descriptor.byte_length for descriptor in result.descriptors
                    ),
                    token_count=result.num_tokens,
                )
            )

    def _zero_match_release_reason(
        self, num_computed_tokens: int, block_size_tokens: int
    ) -> str:
        if (
            block_size_tokens > 0
            and num_computed_tokens >= 0
            and num_computed_tokens % block_size_tokens != 0
        ):
            return "unaligned_num_computed_tokens"
        return "no_remote_g2_match"

    def _terminal_state_for_release(self, reason: str) -> RemoteG2BindingState:
        if reason in {"cancelled", "timeout", "disconnect"}:
            return RemoteG2BindingState.CANCELLED
        if reason == "transfer_failed":
            return RemoteG2BindingState.TRANSFER_FAILED
        return RemoteG2BindingState.RELEASED

    def _emit_event(
        self,
        event: str,
        plan: RemoteKvReusePlan,
        result: RemoteG2ResolveResult,
        *,
        request_id: int | str,
        reason: str,
        outcome: str,
        token_count: int,
        block_count: Optional[int] = None,
    ) -> None:
        count = len(result.descriptors) if block_count is None else block_count
        self._observability.emit(
            RemoteG2LifecycleEvent(
                event=event,
                reason=reason,
                tier=plan.source_tier,
                outcome=outcome,
                request_id=request_id,
                plan_id=plan.plan_id,
                lease_id=result.lease_id,
                source_worker_id=plan.source_worker_id,
                source_generation=result.source_generation,
                block_count=count,
                byte_count=sum(
                    descriptor.byte_length for descriptor in result.descriptors
                ),
                token_count=token_count,
            )
        )

    def _emit_record_event(
        self,
        event: str,
        record: RemoteG2BindingRecord,
        *,
        reason: str,
        outcome: str,
    ) -> None:
        self._observability.emit(
            RemoteG2LifecycleEvent(
                event=event,
                reason=reason,
                tier=record.plan.source_tier,
                outcome=outcome,
                request_id=record.request_id,
                plan_id=record.plan.plan_id,
                lease_id=record.lease_id,
                source_worker_id=record.plan.source_worker_id,
                source_generation=record.source_generation,
                block_count=len(record.bound_blocks) or len(record.resolve_result.descriptors),
                byte_count=sum(
                    block.source_descriptor.byte_length for block in record.bound_blocks
                )
                or sum(
                    descriptor.byte_length
                    for descriptor in record.resolve_result.descriptors
                ),
                token_count=record.matched_tokens,
            )
        )


class SourceG2DescriptorRegistry:
    """Source-worker owned live remote G2 descriptor and lease registry."""

    def __init__(
        self,
        *,
        source_worker_id: int,
        source_dp_rank: int,
        source_generation: int = 1,
        lease_ttl_ms: int = 30_000,
        clock_ms: Callable[[], int] = _now_ms,
        acquire_pin: Optional[Callable[[SourceG2DescriptorRecord, str], Any]] = None,
        release_pin: Optional[Callable[[Any], None]] = None,
        require_trtllm_pin: bool = False,
        kv: Optional[Any] = None,
        window_size: Optional[int] = None,
        pool_id: str = "",
        pool_base_ptr: int = 0,
        block_size_bytes: int = 0,
        tier: str = "",
    ) -> None:
        self.source_worker_id = source_worker_id
        self.source_dp_rank = source_dp_rank
        self.source_generation = source_generation
        self.lease_ttl_ms = lease_ttl_ms
        self._clock_ms = clock_ms
        self._acquire_pin = acquire_pin
        self._release_pin = release_pin
        self._require_trtllm_pin = require_trtllm_pin
        self._kv = kv
        self._window_size = window_size
        self._pool_id = pool_id
        self._pool_base_ptr = int(pool_base_ptr)
        self._block_size_bytes = int(block_size_bytes)
        self._tier = tier
        self._records: dict[int, SourceG2DescriptorRecord] = {}
        self._leases: dict[str, RemoteG2Lease] = {}
        self._lock = threading.RLock()

    def upsert_descriptor(self, record: SourceG2DescriptorRecord) -> None:
        if record.source_worker_id != self.source_worker_id:
            raise ValueError("descriptor source_worker_id does not match registry")
        if record.source_dp_rank != self.source_dp_rank:
            raise ValueError("descriptor source_dp_rank does not match registry")
        if not _is_remote_g2_tier(record.tier):
            raise ValueError("source descriptor registry accepts remote G2 records only")
        with self._lock:
            self._records[int(record.block_hash)] = record

    def remove_descriptor(self, block_hash: int) -> None:
        with self._lock:
            record = self._records.get(int(block_hash))
            if record is not None:
                record.live = False
                self._records.pop(int(block_hash), None)

    def resolve_and_lease(
        self, plan: Mapping[str, Any] | RemoteKvReusePlan
    ) -> RemoteG2ResolveResult:
        now_ms = self._clock_ms()
        try:
            parsed = (
                plan
                if isinstance(plan, RemoteKvReusePlan)
                else RemoteKvReusePlan.from_dict(plan)
            )
        except (TypeError, ValueError):
            return RemoteG2ResolveResult(
                None, (), 0, "invalid_plan", self.source_generation
            )

        if parsed.plan_version != REMOTE_KV_REUSE_PLAN_VERSION:
            return RemoteG2ResolveResult(
                None, (), 0, "unsupported_plan_version", self.source_generation
            )
        if parsed.source_worker_id != self.source_worker_id:
            return RemoteG2ResolveResult(
                None, (), 0, "wrong_source_worker", self.source_generation
            )
        if parsed.source_dp_rank != self.source_dp_rank:
            return RemoteG2ResolveResult(
                None, (), 0, "wrong_source_rank", self.source_generation
            )
        if not parsed.is_remote_g2():
            return RemoteG2ResolveResult(
                None, (), 0, "wrong_source_tier", self.source_generation
            )
        if parsed.is_expired(now_ms):
            return RemoteG2ResolveResult(
                None, (), 0, "plan_expired", self.source_generation
            )
        if self._require_trtllm_pin and self._acquire_pin is None:
            return RemoteG2ResolveResult(
                None, (), 0, "missing_trtllm_pin_hook", self.source_generation
            )

        identity_hashes = parsed.planned_hashes
        # Producers that have not been updated to populate kv_block_hashes
        # leave the field empty, in which case we use the plan's
        # block_hashes for both identity and lookup. This preserves the
        # pre-dual-hash behavior for legacy plans.
        kv_hashes = parsed.planned_kv_block_hashes or identity_hashes

        with self._lock:
            records: list[SourceG2DescriptorRecord] = []
            per_block_status: list[RemoteG2BlockStatus] = []
            i = 0
            while i < len(identity_hashes):
                identity_hash = identity_hashes[i]
                kv_hash = int(kv_hashes[i])
                record = self._records.get(kv_hash)
                if record is None and self._kv is not None:
                    lookup_results = self._find_and_pin_blocks_by_hash(kv_hashes[i:])
                    if not lookup_results:
                        per_block_status.append(
                            RemoteG2BlockStatus(int(identity_hash), "missing")
                        )
                        break
                    for offset, lookup_result in enumerate(lookup_results):
                        current_identity_hash = int(identity_hashes[i + offset])
                        if isinstance(lookup_result, CacheMiss):
                            status = (
                                "promoted_primary"
                                if lookup_result.found_tier == _CACHE_TIER_PRIMARY
                                else "missing"
                            )
                            per_block_status.append(
                                RemoteG2BlockStatus(current_identity_hash, status)
                            )
                            break
                        record = self._record_from_pinned_cache_block(lookup_result)
                        records.append(record)
                        per_block_status.append(
                            RemoteG2BlockStatus(
                                current_identity_hash,
                                "live",
                                record.descriptor_generation,
                            )
                        )
                    break
                if record is None:
                    per_block_status.append(RemoteG2BlockStatus(int(identity_hash), "missing"))
                    break
                if not record.live:
                    per_block_status.append(RemoteG2BlockStatus(int(identity_hash), "non_live"))
                    break
                if not _is_remote_g2_tier(record.tier):
                    per_block_status.append(RemoteG2BlockStatus(int(identity_hash), "wrong_tier"))
                    break
                if (
                    record.source_worker_id != self.source_worker_id
                    or record.source_dp_rank != self.source_dp_rank
                ):
                    per_block_status.append(
                        RemoteG2BlockStatus(
                            int(identity_hash), "wrong_source", record.descriptor_generation
                        )
                    )
                    break
                records.append(record)
                per_block_status.append(
                    RemoteG2BlockStatus(
                        int(identity_hash), "live", record.descriptor_generation
                    )
                )
                i += 1

            if not records:
                return RemoteG2ResolveResult(
                    None,
                    (),
                    0,
                    "no_live_remote_g2_prefix",
                    self.source_generation,
                    tuple(per_block_status),
                )

            lease_id = f"{parsed.plan_id}:{uuid.uuid4().hex}"
            pin_refs: list[Any] = []
            try:
                if self._acquire_pin is not None:
                    pin_refs = [self._acquire_pin(record, lease_id) for record in records]

                for record in records:
                    record.lease_count += 1

                lease = RemoteG2Lease(
                    lease_id=lease_id,
                    plan_id=parsed.plan_id,
                    request_id=parsed.request_id,
                    target_worker_id=parsed.target_worker_id,
                    target_dp_rank=parsed.target_dp_rank,
                    block_hashes=tuple(record.block_hash for record in records),
                    descriptor_generations=tuple(
                        record.descriptor_generation for record in records
                    ),
                    expires_at_ms=now_ms + self.lease_ttl_ms,
                    trtllm_pin_refs=tuple(pin_refs),
                )
                self._leases[lease_id] = lease
            except Exception:
                for pin_ref in pin_refs:
                    self._release_pin_ref(pin_ref)
                raise

            # Descriptors carry the router-facing identity (block_hashes /
            # tokens_hash). The record's internal block_hash is the KV-side
            # value used for in-process lookup; the router does not need it.
            descriptors = tuple(
                RemoteG2Descriptor(
                    block_hash=int(identity_hashes[i]),
                    descriptor_generation=record.descriptor_generation,
                    pool_id=record.pool_id,
                    byte_offset=record.byte_offset,
                    byte_length=record.byte_length,
                    metadata=dict(record.metadata),
                )
                for i, record in enumerate(records)
            )
            return RemoteG2ResolveResult(
                lease_id=lease_id,
                descriptors=descriptors,
                num_tokens=len(descriptors) * parsed.block_size_tokens,
                source_generation=self.source_generation,
                per_block_status=tuple(per_block_status),
            )

    def release_lease(self, lease_id: str, reason: str) -> bool:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.released:
                return False

            lease.released = True
            lease.release_reason = reason
            for block_hash in lease.block_hashes:
                record = self._records.get(block_hash)
                if record is not None and record.lease_count > 0:
                    record.lease_count -= 1
            for pin_ref in lease.trtllm_pin_refs:
                self._release_pin_ref(pin_ref)
            return True

    def expire_leases(self) -> list[str]:
        now_ms = self._clock_ms()
        with self._lock:
            expired = [
                lease_id
                for lease_id, lease in self._leases.items()
                if not lease.released and lease.expires_at_ms <= now_ms
            ]
        for lease_id in expired:
            self.release_lease(lease_id, "ttl_expired")
        return expired

    def get_lease(self, lease_id: str) -> Optional[RemoteG2Lease]:
        with self._lock:
            return self._leases.get(lease_id)

    def _release_pin_ref(self, pin_ref: Any) -> None:
        if self._release_pin is not None:
            self._release_pin(pin_ref)

    def _find_and_pin_blocks_by_hash(
        self, block_hashes: tuple[int, ...]
    ) -> list[CacheLookupResult]:
        """Resolve blocks through the live C++ KV cache manager.

        Returned PinnedCacheBlock entries are already pinned in the requested
        host-pinned tier. CacheMiss(found_tier="primary") means the block was
        observed in primary and was not pinned.
        """
        if self._window_size is None or self._block_size_bytes <= 0:
            return [CacheMiss(int(block_hashes[0]))] if block_hashes else []
        lookup_hashes = [int(block_hash) for block_hash in block_hashes]
        try:
            raw_results = self._kv.find_and_pin_blocks_by_hash(
                lookup_hashes,
                int(self._window_size),
                tier=_CACHE_TIER_HOST_PINNED,
                stop_on_miss=True,
            )
        except Exception:
            logging.exception("remote_g2: find_and_pin raised n=%d", len(lookup_hashes))
            return [CacheMiss(int(block_hashes[0]))] if block_hashes else []

        results: list[CacheLookupResult] = []
        for raw in raw_results:
            if bool(raw.get("pinned", False)):
                results.append(
                    PinnedCacheBlock(
                        block_hash=int(raw["block_hash"]),
                        block_id=int(raw["block_id"]),
                        slot_idx=int(raw["slot_idx"]),
                        tier=str(raw.get("found_tier") or _CACHE_TIER_HOST_PINNED),
                    )
                )
            else:
                found_tier = raw.get("found_tier")
                results.append(
                    CacheMiss(
                        block_hash=int(raw["block_hash"]),
                        found_tier=str(found_tier) if found_tier is not None else None,
                    )
                )
        return results

    def _record_from_pinned_cache_block(
        self, pinned: PinnedCacheBlock
    ) -> SourceG2DescriptorRecord:
        byte_offset = int(pinned.slot_idx) * self._block_size_bytes
        record = SourceG2DescriptorRecord(
            block_hash=pinned.block_hash,
            source_worker_id=self.source_worker_id,
            source_dp_rank=self.source_dp_rank,
            tier=pinned.tier,
            descriptor_generation=1,
            pool_id=self._pool_id,
            byte_offset=byte_offset,
            byte_length=self._block_size_bytes,
            block_id=int(pinned.block_id),
            live=True,
            metadata={
                "nixl_memory_desc": {
                    "ptr": self._pool_base_ptr + byte_offset,
                    "len": self._block_size_bytes,
                }
            },
        )
        # The C++ tier-aware lookup already bumped refcount on this block. The
        # downstream acquire_pin callback must NOT call pin_blocks_by_id again
        # - it should only register this pin with the lease so release_lease
        # can unpin once the transfer completes.
        record._pinned_by_lookup = True
        return record
