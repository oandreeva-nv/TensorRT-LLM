# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .kv_cache_connector import (
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from .remote_g2 import (
    RemoteG2BindingRecord,
    RemoteG2ResolveResult,
    RemoteKvReusePlan,
    TargetRemoteG2BindingStore,
    TargetRemotePlanStore,
    target_remote_g2_plan_store,
)
from .remote_g2_observability import (
    NullRemoteG2ObservabilitySink,
    RemoteG2LifecycleEvent,
    RemoteG2ObservabilitySink,
)
from .remote_g2_transfer import RemoteG2TransferError


@dataclass(frozen=True)
class RemoteG2ConnectorMetadata:
    bindings: tuple[RemoteG2BindingRecord, ...] = ()


def _missing_release_lease(lease_id: str, reason: str) -> bool:
    return False


def _assert_partial_reuse_disabled(llm_args: Any) -> None:
    """Hard-fail at construction if ``kv_cache_config.enable_partial_reuse``
    is left on the default ``True`` while the remote-G2 connector is active.

    Partial reuse silently stops remote-G2 fetch from triggering on
    subsequent requests, so misconfiguration must surface as a startup error
    rather than as a quiet drop in hit rate. The connector class being
    constructed at all means the user has selected remote_g2 — no further
    gate is needed.
    """
    kv_cache_config = getattr(llm_args, "kv_cache_config", None)
    if kv_cache_config is None:
        return
    if getattr(kv_cache_config, "enable_partial_reuse", False):
        raise RuntimeError(
            "remote_g2: kv_cache_config.enable_partial_reuse must be set to "
            "False when the remote-G2 connector is enabled (currently True, "
            "the default). Leaving it on prevents remote-G2 fetch from "
            "triggering on subsequent requests; the failure mode is silent "
            "(no error, just no remote-G2 hits)."
        )


# Module-state slots for callables installed from outside (typically by
# remote_g2_target_setup.maybe_start_remote_g2_target_client). The
# connector scheduler/worker read these lazily at call time so that
# installation order vs. construction order doesn't matter.
_installed_resolve_and_lease: Optional[
    Callable[["RemoteKvReusePlan"], "RemoteG2ResolveResult"]
] = None
_installed_release_lease: Optional[Callable[[str, str], bool]] = None
_installed_transfer_adapter: Optional[Any] = None
_installed_mark_local_valid: Optional[Callable[[RemoteG2BindingRecord], None]] = None
_installed_publish_binding: Optional[Callable[[RemoteG2BindingRecord], None]] = None
# Maps engine block_ids → primary-pool slot indices. Installed by the
# target setup once the KV cache manager is available. The binding store
# calls this when binding so NIXL's local dlist gets the right
# dense per-slot index instead of the engine's globally-unique block_id.
_installed_block_id_to_slot_idx: Optional[
    Callable[[list[int]], list[int]]
] = None


def install_block_id_to_slot_idx(
    fn: Callable[[list[int]], list[int]]
) -> None:
    global _installed_block_id_to_slot_idx
    _installed_block_id_to_slot_idx = fn


def install_resolve_and_lease(
    fn: Callable[["RemoteKvReusePlan"], "RemoteG2ResolveResult"]
) -> None:
    global _installed_resolve_and_lease
    _installed_resolve_and_lease = fn


def install_release_lease(fn: Callable[[str, str], bool]) -> None:
    global _installed_release_lease
    _installed_release_lease = fn


def install_transfer_adapter(adapter: Any) -> None:
    """Install the NIXL transfer adapter that the connector worker uses
    in start_load_kv. Read lazily by the worker so installation order
    relative to worker construction doesn't matter."""
    global _installed_transfer_adapter
    _installed_transfer_adapter = adapter


def install_mark_local_valid(fn: Callable[[RemoteG2BindingRecord], None]) -> None:
    global _installed_mark_local_valid
    _installed_mark_local_valid = fn


def install_publish_binding(fn: Callable[[RemoteG2BindingRecord], None]) -> None:
    global _installed_publish_binding
    _installed_publish_binding = fn


def _resolve_release_lease(explicit: Optional[Callable[[str, str], bool]]):
    """Return a callable that defers lookup to call time so module-state
    installation that happens after the scheduler/worker is constructed
    is still picked up."""

    def _call(lease_id: str, reason: str) -> bool:
        fn = explicit if explicit is not None else _installed_release_lease
        if fn is None:
            return _missing_release_lease(lease_id, reason)
        return fn(lease_id, reason)

    return _call


class RemoteG2KvCacheConnectorScheduler(KvCacheConnectorScheduler):
    requires_retryable_kv_admission = True
    # KVCM V1 local offload/onboard is not safe under overlap scheduler.
    # See NVBug 6293536.
    requires_disable_overlap_scheduler = True
    requires_disable_attention_dp = True
    requires_uniform_attention_window = True

    def __init__(
        self,
        llm_args: Any,
        *,
        plan_store: Optional[TargetRemotePlanStore] = None,
        binding_store: Optional[TargetRemoteG2BindingStore] = None,
        resolve_and_lease: Optional[
            Callable[[RemoteKvReusePlan], RemoteG2ResolveResult]
        ] = None,
        release_lease: Optional[Callable[[str, str], bool]] = None,
        observability: Optional[RemoteG2ObservabilitySink] = None,
    ) -> None:
        super().__init__(llm_args)
        _assert_partial_reuse_disabled(llm_args)
        self._observability = observability or NullRemoteG2ObservabilitySink()
        self._plan_store = (
            plan_store if plan_store is not None else target_remote_g2_plan_store()
        )
        self._explicit_resolve_and_lease = resolve_and_lease
        self._binding_store = (
            binding_store
            if binding_store is not None
            else TargetRemoteG2BindingStore(
                _resolve_release_lease(release_lease),
                observability=self._observability,
            )
        )

    @property
    def _resolve_and_lease(
        self,
    ) -> Optional[Callable[["RemoteKvReusePlan"], "RemoteG2ResolveResult"]]:
        """Prefer the explicit kwarg, fall back to module-state install
        at access time so late installation is still picked up."""
        if self._explicit_resolve_and_lease is not None:
            return self._explicit_resolve_and_lease
        return _installed_resolve_and_lease

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        plan = self._plan_store.get(request.request_id)
        resolver = self._resolve_and_lease
        if plan is None or resolver is None:
            return (0, False)

        record = self._binding_store.resolve_for_request(
            request.request_id,
            plan,
            num_computed_tokens,
            resolver,
        )
        if record is None:
            return (0, False)
        return (record.matched_tokens, True)

    def update_state_after_alloc(self, request: Any, block_ids: list[int]) -> None:
        self._binding_store.bind_target_blocks(request.request_id, block_ids)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> RemoteG2ConnectorMetadata:
        # The official contract is to filter records by what's in
        # scheduler_output. Empirically that input arrives with the
        # newly-allocated request missing during the same tick the
        # request was bound, so the connector would never emit any
        # transfer-ready record. Scan binding_store directly instead —
        # is_transfer_ready already gates on (state=BOUND and bound_blocks),
        # and the worker tracks per-request _active_loads / _completed_loads
        # to avoid double-start, so we don't actually need scheduler_output
        # to dedupe.
        bindings: list[RemoteG2BindingRecord] = []
        for request_id, record in self._binding_store.iter_records():
            if record.is_transfer_ready:
                bindings.append(record)
        return RemoteG2ConnectorMetadata(tuple(bindings))

    def request_finished(self, request: Any, cache_block_ids: list[int]) -> bool:
        self._binding_store.discard(request.request_id, "request_finished")
        self._plan_store.discard(request.request_id)
        return False


class RemoteG2KvCacheConnectorWorker(KvCacheConnectorWorker):
    requires_retryable_kv_admission = True
    # Keep scheduler and worker capability flags aligned.
    requires_disable_overlap_scheduler = True
    requires_disable_attention_dp = True
    requires_uniform_attention_window = True

    def __init__(
        self,
        llm_args: Any,
        *,
        transfer_adapter: Optional[Any] = None,
        release_lease: Optional[Callable[[str, str], bool]] = None,
        mark_local_valid: Optional[Callable[[RemoteG2BindingRecord], None]] = None,
        publish_binding: Optional[Callable[[RemoteG2BindingRecord], None]] = None,
        transfer_timeout_ms: int = 30_000,
        observability: Optional[RemoteG2ObservabilitySink] = None,
    ) -> None:
        super().__init__(llm_args)
        _assert_partial_reuse_disabled(llm_args)
        # Worker is constructed by PyExecutor with just llm_args, before
        # maybe_start_remote_g2_target_client runs - none of the
        # adapter / hooks can be wired at that point. Stash whatever was
        # passed explicitly; the accessor properties below fall back to
        # module-state slots at call time.
        self._explicit_transfer_adapter = transfer_adapter
        self._release_lease = _resolve_release_lease(release_lease)
        self._explicit_mark_local_valid = mark_local_valid
        self._explicit_publish_binding = publish_binding
        self._transfer_timeout_ms = transfer_timeout_ms
        self._observability = observability or NullRemoteG2ObservabilitySink()
        self._active_loads: dict[int | str, _RemoteG2ActiveLoad] = {}
        self._completed_loads: set[int | str] = set()
        self._released_leases: set[str] = set()
        # Fix 1: Deferred release — records whose transfers completed
        # locally but whose release RPC has not yet been sent.  The
        # release is deferred until the connector manager's allgather
        # confirms ALL TP ranks finished loading (see
        # on_globally_finished_loading).  This prevents the source from
        # unpinning blocks while a sibling target rank is still reading.
        self._completed_pending_release: dict[
            int | str, RemoteG2BindingRecord
        ] = {}

    @property
    def _transfer_adapter(self) -> Optional[Any]:
        if self._explicit_transfer_adapter is not None:
            return self._explicit_transfer_adapter
        return _installed_transfer_adapter

    @property
    def _mark_local_valid(self) -> Optional[Callable[[RemoteG2BindingRecord], None]]:
        if self._explicit_mark_local_valid is not None:
            return self._explicit_mark_local_valid
        return _installed_mark_local_valid

    @property
    def _publish_binding(self) -> Optional[Callable[[RemoteG2BindingRecord], None]]:
        if self._explicit_publish_binding is not None:
            return self._explicit_publish_binding
        return _installed_publish_binding

    def register_kv_caches(self, kv_cache_tensor: Any) -> None:
        self._kv_cache_tensor = kv_cache_tensor

    def start_load_kv(self, stream: Any) -> None:
        metadata = self.get_connector_meta()
        if not isinstance(metadata, RemoteG2ConnectorMetadata) or not metadata.bindings:
            return
        if self._transfer_adapter is None:
            for record in metadata.bindings:
                self._emit_record_event(
                    "fallback",
                    record,
                    reason="transfer_adapter_missing",
                    outcome="local_recompute",
                )
                self._release_record_once(record, "transfer_adapter_missing")
            raise RuntimeError("remote G2 transfer adapter is not configured")

        for record in metadata.bindings:
            request_id = record.request_id
            if request_id in self._active_loads or request_id in self._completed_loads:
                continue
            try:
                result = self._transfer_adapter.start_transfer(record)
            except Exception as exc:
                self._emit_record_event(
                    "fallback",
                    record,
                    reason="transfer_start_failed",
                    outcome="local_recompute",
                )
                self._release_record_once(record, "transfer_start_failed")
                raise RuntimeError("remote G2 transfer failed to start") from exc
            self._active_loads[request_id] = _RemoteG2ActiveLoad(
                result=result, started_at_ms=_now_ms()
            )

    def wait_for_layer_load(self, layer_idx: int, stream: Any) -> None:
        return

    def save_kv_layer(self, layer_idx: int, stream: Any) -> None:
        return

    def wait_for_save(self, stream: Any) -> None:
        return

    def get_finished(
        self, finished_gen_req_ids: list[int], started_loading_req_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        finished_loading: list[int] = []
        # Iterate self._active_loads (all in-flight transfers), not just
        # started_loading_req_ids (NEW this tick). The connector
        # framework's get_finished moves the request from
        # new_async_requests to pending_async_requests on the first
        # tick, so subsequent ticks call us with empty
        # started_loading_req_ids — without this iteration we'd only
        # ever poll each transfer ONCE, and slow/in-progress transfers
        # would never get reported as finished.
        for request_id in list(self._active_loads.keys()):
            active = self._active_loads.get(request_id)
            if active is None:
                continue
            record = active.result.record
            try:
                completed = active.result.is_completed()
            except Exception as exc:
                self._active_loads.pop(request_id, None)
                try:
                    self._release_transfer_result_once(active.result)
                finally:
                    self._emit_record_event(
                        "fallback",
                        record,
                        reason="transfer_failed",
                        outcome="local_recompute",
                    )
                    self._release_record_once(record, "transfer_failed")
                raise RuntimeError("remote G2 transfer failed closed") from exc

            if completed:
                try:
                    self._release_transfer_result_once(active.result)
                    self._emit_record_event(
                        "transferred",
                        record,
                        reason="ok",
                        outcome="completed",
                    )
                    self._complete_success(active)
                    self._active_loads.pop(request_id, None)
                    self._completed_loads.add(request_id)
                    finished_loading.append(int(request_id))
                except Exception as exc:
                    self._active_loads.pop(request_id, None)
                    try:
                        self._release_transfer_result_once(active.result)
                    finally:
                        self._release_record_once(record, "transfer_failed")
                    raise RuntimeError("remote G2 transfer failed closed") from exc
            elif _now_ms() - active.started_at_ms > self._transfer_timeout_ms:
                self._active_loads.pop(request_id, None)
                try:
                    self._release_transfer_result_once(active.result)
                finally:
                    self._emit_record_event(
                        "fallback",
                        record,
                        reason="transfer_timeout",
                        outcome="local_recompute",
                    )
                    self._release_record_once(record, "transfer_timeout")
                raise RuntimeError("remote G2 transfer timed out")
        return ([], finished_loading)

    def _complete_success(self, active: "_RemoteG2ActiveLoad") -> None:
        record = active.result.record
        if self._mark_local_valid is None:
            self._emit_record_event(
                "fallback",
                record,
                reason="local_validity_missing",
                outcome="local_recompute",
            )
            self._release_record_once(record, "local_validity_missing")
            raise RemoteG2TransferError("remote G2 local validity hook is not configured")
        if self._publish_binding is None:
            self._emit_record_event(
                "fallback",
                record,
                reason="publication_missing",
                outcome="local_recompute",
            )
            self._release_record_once(record, "publication_missing")
            raise RemoteG2TransferError("remote G2 publication hook is not configured")
        try:
            self._mark_local_valid(record)
            active.local_valid_marked = True
        except Exception:
            self._emit_record_event(
                "fallback",
                record,
                reason="local_validity_failed",
                outcome="local_recompute",
            )
            self._release_record_once(record, "local_validity_failed")
            raise
        try:
            self._publish_binding(record)
            active.published = True
        except Exception:
            self._emit_record_event(
                "failed",
                record,
                reason="publication_failed",
                outcome="fail_closed",
            )
            self._release_record_once(record, "publication_failed")
            raise
        # Fix 1: Do NOT release immediately — defer until the connector
        # manager's allgather confirms all TP ranks are done loading.
        # on_globally_finished_loading() will call _release_record_once
        # for each request in the globally-confirmed set.
        self._completed_pending_release[record.request_id] = record

    def on_globally_finished_loading(
        self, globally_finished_ids: set,
    ) -> None:
        """Called by KvCacheConnectorManager after mpi_allgather confirms
        all TP ranks finished loading these request IDs.

        This is the safe point to release source leases — all target
        ranks have completed their NIXL RDMA reads, so unpinning source
        blocks cannot cause corruption.

        Non-leader ranks don't have ``_installed_release_lease`` wired,
        so their ``_release_record_once`` calls fall through to
        ``_missing_release_lease`` (a no-op).  Only rank 0 actually
        sends the release RPC.
        """
        for req_id in list(self._completed_pending_release.keys()):
            if req_id in globally_finished_ids:
                record = self._completed_pending_release.pop(req_id)
                self._release_record_once(record, "transfer_succeeded")

    def _release_record_once(self, record: RemoteG2BindingRecord, reason: str) -> bool:
        lease_id = record.lease_id
        if lease_id is None:
            self._emit_record_event(
                "released", record, reason=reason, outcome="no_lease"
            )
            return False
        if lease_id in self._released_leases:
            self._emit_record_event(
                "released", record, reason=reason, outcome="already_released"
            )
            return False
        self._released_leases.add(lease_id)
        completed = self._release_lease(lease_id, reason)
        self._emit_record_event(
            "released",
            record,
            reason=reason,
            outcome="completed" if completed else "already_released",
        )
        return completed

    def _release_transfer_result_once(self, result: Any) -> None:
        release = getattr(result, "release", None)
        if release is not None:
            release()

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
                block_count=len(record.bound_blocks),
                byte_count=sum(
                    block.source_descriptor.byte_length for block in record.bound_blocks
                ),
                token_count=record.matched_tokens,
            )
        )


@dataclass
class _RemoteG2ActiveLoad:
    result: Any
    started_at_ms: int
    local_valid_marked: bool = False
    published: bool = False


def _now_ms() -> int:
    return int(time.time() * 1000)
