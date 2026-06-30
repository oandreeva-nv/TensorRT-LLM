# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from .kv_cache_connector import KvCacheConnectorScheduler, KvCacheConnectorWorker, SchedulerOutput
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
from .remote_g2_transfer import RemoteG2TransferState, validate_remote_g2_transfer_result


@dataclass(frozen=True)
class RemoteG2ConnectorMetadata:
    bindings: tuple[RemoteG2BindingRecord, ...] = ()


def _missing_release_lease(lease_id: str, reason: str) -> bool:
    raise RuntimeError("remote G2 lease release is not configured")


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
_installed_resolve_and_lease: Optional[Callable[["RemoteKvReusePlan"], "RemoteG2ResolveResult"]] = (
    None
)
_installed_release_lease: Optional[Callable[[str, str], bool]] = None
_installed_transfer_adapter: Optional[Any] = None
_installed_mark_local_valid: Optional[Callable[[RemoteG2BindingRecord], None]] = None
_installed_publish_binding: Optional[Callable[[RemoteG2BindingRecord], None]] = None
# Maps engine block_ids → primary-pool slot indices. Installed by the
# target setup once the KV cache manager is available. The binding store
# calls this when binding so NIXL's local dlist gets the right
# dense per-slot index instead of the engine's globally-unique block_id.
_installed_block_id_to_slot_idx: Optional[Callable[[list[int]], list[int]]] = None


def install_block_id_to_slot_idx(fn: Callable[[list[int]], list[int]]) -> None:
    global _installed_block_id_to_slot_idx
    _installed_block_id_to_slot_idx = fn


def install_resolve_and_lease(fn: Callable[["RemoteKvReusePlan"], "RemoteG2ResolveResult"]) -> None:
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
        resolve_and_lease: Optional[Callable[[RemoteKvReusePlan], RemoteG2ResolveResult]] = None,
        release_lease: Optional[Callable[[str, str], bool]] = None,
        observability: Optional[RemoteG2ObservabilitySink] = None,
    ) -> None:
        super().__init__(llm_args)
        _assert_partial_reuse_disabled(llm_args)
        self._observability = observability or NullRemoteG2ObservabilitySink()
        self._plan_store = plan_store if plan_store is not None else target_remote_g2_plan_store()
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
        import logging as _logging
        import os as _os

        _logging.warning(
            "PROBE rpc_chain get_num_new_matched_tokens pid=%d req_id=%s "
            "plan_found=%s resolver_set=%s store_id=%d",
            _os.getpid(),
            request.request_id,
            plan is not None,
            resolver is not None,
            id(self._plan_store),
        )
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
        import logging as _logging

        record_before = self._binding_store.get(request.request_id)
        self._binding_store.bind_target_blocks(request.request_id, block_ids)
        record_after = self._binding_store.get(request.request_id)
        _logging.warning(
            "PROBE rpc_chain update_state_after_alloc req_id=%s block_ids_count=%d "
            "pre_state=%s post_state=%s post_bound_blocks=%d post_is_transfer_ready=%s",
            request.request_id,
            len(block_ids),
            getattr(record_before, "state", None) if record_before else None,
            getattr(record_after, "state", None) if record_after else None,
            len(getattr(record_after, "bound_blocks", ()) or ()) if record_after else 0,
            getattr(record_after, "is_transfer_ready", False) if record_after else False,
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> RemoteG2ConnectorMetadata:
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
        states_snapshot = []
        for request_id, record in self._binding_store.iter_records():
            states_snapshot.append(
                (
                    request_id,
                    getattr(record, "state", None),
                    bool(record.is_transfer_ready),
                )
            )
            if record.is_transfer_ready:
                bindings.append(record)
        import logging as _logging

        _logging.warning(
            "PROBE rpc_chain build_connector_meta scanned=%d transfer_ready=%d states=%s "
            "scheduler_output_size=%d",
            len(states_snapshot),
            len(bindings),
            states_snapshot[:5],
            len(scheduler_output.new_requests) + len(scheduler_output.cached_requests),
        )
        return RemoteG2ConnectorMetadata(tuple(bindings))

    def request_finished(self, request: Any, cache_block_ids: list[int]) -> bool:
        self._release_and_forget(request.request_id, "request_finished")
        return False

    def try_abort_request(self, request_id: int, reason: str) -> bool:
        return self._release_and_forget(request_id, reason)

    def finalize_successful_load(self, request_id: int) -> bool:
        return self._release_and_forget(request_id, "transfer_succeeded")

    def _release_and_forget(self, request_id: int, reason: str) -> bool:
        """Release the shared lease on rank zero, then forget state."""
        try:
            if not self._binding_store.release_and_discard(request_id, reason):
                return False
            self._plan_store.discard(request_id)
        except Exception:
            logging.exception(
                "remote_g2: scheduler cleanup failed request_id=%s reason=%s",
                request_id,
                reason,
            )
            return False
        return True


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
        self._explicit_mark_local_valid = mark_local_valid
        self._explicit_publish_binding = publish_binding
        self._transfer_timeout_ms = transfer_timeout_ms
        self._observability = observability or NullRemoteG2ObservabilitySink()
        self._active_loads: dict[int | str, _RemoteG2LoadAttempt] = {}
        self._terminal_binding_fingerprints: set[tuple[Optional[str], str, int]] = set()
        self._failed_load_request_ids: set[int] = set()

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
        if not isinstance(metadata, RemoteG2ConnectorMetadata):
            self._terminal_binding_fingerprints.clear()
            return
        current_fingerprints = {self._binding_fingerprint(record) for record in metadata.bindings}
        self._terminal_binding_fingerprints.intersection_update(current_fingerprints)
        if not metadata.bindings:
            return

        for record in metadata.bindings:
            request_id = record.request_id
            if (
                request_id in self._active_loads
                or self._binding_fingerprint(record) in self._terminal_binding_fingerprints
            ):
                continue
            load = _RemoteG2LoadAttempt(
                record=record,
                state=_RemoteG2LoadState.STARTING,
                started_at_ms=_now_ms(),
            )
            self._active_loads[request_id] = load
            adapter = self._transfer_adapter
            if adapter is None:
                self._mark_load_failed(load, "transfer_adapter_missing")
                continue
            if not bool(getattr(adapter, "supports_retryable_release", False)):
                self._mark_load_failed(load, "transfer_adapter_incapable")
                continue
            try:
                result = adapter.start_transfer(record)
            except Exception as exc:
                transfer_result = getattr(exc, "transfer_result", None)
                load.result = transfer_result
                if transfer_result is not None:
                    try:
                        validate_remote_g2_transfer_result(transfer_result)
                        load.release_contract_valid = True
                    except Exception:
                        pass
                self._mark_load_failed(load, "transfer_start_failed")
                continue
            load.result = result
            try:
                validate_remote_g2_transfer_result(result)
                start_error = getattr(result, "start_error", None)
                initial_state = result.initial_state
            except Exception:
                # Ownership may already have crossed into the adapter.  An
                # invalid result cannot prove that handle cleanup is safe.
                self._mark_load_failed(load, "transfer_result_unsafe")
                continue
            load.release_contract_valid = True
            if start_error is not None:
                self._mark_load_failed(load, "transfer_start_failed")
            elif initial_state is RemoteG2TransferState.FAILED:
                self._mark_load_failed(load, "transfer_failed")
            elif initial_state is RemoteG2TransferState.SUCCEEDED:
                load.state = _RemoteG2LoadState.SUCCEEDED_PENDING_CLEANUP
            else:
                load.state = _RemoteG2LoadState.ACTIVE

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
            load = self._active_loads.get(request_id)
            if load is None:
                continue
            if load.state is _RemoteG2LoadState.ACTIVE:
                try:
                    state = load.result.poll_state()
                except Exception:
                    self._mark_load_failed(load, "transfer_poll_failed")
                else:
                    if state is RemoteG2TransferState.SUCCEEDED:
                        load.state = _RemoteG2LoadState.SUCCEEDED_PENDING_CLEANUP
                    elif state is RemoteG2TransferState.FAILED:
                        self._mark_load_failed(load, "transfer_failed")
                    elif state is not RemoteG2TransferState.IN_PROGRESS:
                        self._mark_load_failed(load, "transfer_state_invalid")
                    elif _now_ms() - load.started_at_ms > self._transfer_timeout_ms:
                        self._mark_load_failed(load, "transfer_timeout")
            if self._advance_load_cleanup(load):
                self._finalize_load(request_id, load, finished_loading)
        return ([], finished_loading)

    def take_failed_load_request_ids(self) -> set[int]:
        failed = self._failed_load_request_ids
        self._failed_load_request_ids = set()
        return failed

    def try_abort_request(self, request_id: int) -> bool:
        load = self._active_loads.get(request_id)
        if load is None:
            return True
        if load.state is not _RemoteG2LoadState.FAILED_PENDING_CLEANUP:
            load.state = _RemoteG2LoadState.ABORTED_PENDING_CLEANUP
            load.failure_reason = "cancelled"
        self._advance_load_cleanup(load)
        self._finalize_load(request_id, load, [])
        return True

    def _mark_load_failed(self, load: "_RemoteG2LoadAttempt", reason: str) -> None:
        first_failure = load.state is not _RemoteG2LoadState.FAILED_PENDING_CLEANUP
        load.state = _RemoteG2LoadState.FAILED_PENDING_CLEANUP
        load.failure_reason = reason
        if first_failure:
            self._failed_load_request_ids.add(int(load.record.request_id))

    def _advance_load_cleanup(self, load: "_RemoteG2LoadAttempt") -> bool:
        if load.state in {_RemoteG2LoadState.STARTING, _RemoteG2LoadState.ACTIVE}:
            return False
        if load.result is not None:
            self._release_transfer_handle(load)

        if load.state is _RemoteG2LoadState.SUCCEEDED_PENDING_CLEANUP:
            if not load.mark_local_valid_attempted:
                load.mark_local_valid_attempted = True
                if self._mark_local_valid is None:
                    self._mark_load_failed(load, "local_validity_missing")
                else:
                    try:
                        self._mark_local_valid(load.record)
                        load.local_valid_marked = True
                    except Exception:
                        self._mark_load_failed(load, "local_validity_failed")
            if (
                load.state is _RemoteG2LoadState.SUCCEEDED_PENDING_CLEANUP
                and not load.publish_binding_attempted
            ):
                load.publish_binding_attempted = True
                if self._publish_binding is None:
                    self._mark_load_failed(load, "publication_missing")
                else:
                    try:
                        self._publish_binding(load.record)
                        load.binding_published = True
                    except Exception:
                        self._mark_load_failed(load, "publication_failed")

        return True

    def _release_transfer_handle(self, load: "_RemoteG2LoadAttempt") -> None:
        if not load.release_contract_valid:
            raise RuntimeError("remote_g2: transfer handle release contract is unavailable")
        release_transfer = getattr(load.result, "release_transfer", None)
        if not callable(release_transfer):
            raise RuntimeError("remote_g2: transfer result has no handle release operation")
        if release_transfer() is not True:
            raise RuntimeError("remote_g2: transfer handle release did not confirm release")

    def _finalize_load(
        self,
        request_id: int | str,
        load: "_RemoteG2LoadAttempt",
        finished_loading: list[int],
    ) -> None:
        self._active_loads.pop(request_id, None)
        self._terminal_binding_fingerprints.add(self._binding_fingerprint(load.record))
        reason = load.failure_reason or "transfer_succeeded"
        if load.state is _RemoteG2LoadState.SUCCEEDED_PENDING_CLEANUP:
            self._emit_record_event("transferred", load.record, reason="ok", outcome="completed")
            finished_loading.append(int(request_id))
        elif load.state is _RemoteG2LoadState.FAILED_PENDING_CLEANUP:
            self._emit_record_event("failed", load.record, reason=reason, outcome="request_failed")
        elif load.state is _RemoteG2LoadState.ABORTED_PENDING_CLEANUP:
            self._emit_record_event("failed", load.record, reason=reason, outcome="request_aborted")

    @staticmethod
    def _binding_fingerprint(
        record: RemoteG2BindingRecord,
    ) -> tuple[Optional[str], str, int]:
        return (record.lease_id, record.plan.plan_id, record.source_generation)

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


class _RemoteG2LoadState(Enum):
    STARTING = "starting"
    ACTIVE = "active"
    SUCCEEDED_PENDING_CLEANUP = "succeeded_pending_cleanup"
    FAILED_PENDING_CLEANUP = "failed_pending_cleanup"
    ABORTED_PENDING_CLEANUP = "aborted_pending_cleanup"


@dataclass
class _RemoteG2LoadAttempt:
    record: RemoteG2BindingRecord
    state: _RemoteG2LoadState
    started_at_ms: int
    result: Any = None
    failure_reason: Optional[str] = None
    release_contract_valid: bool = False
    mark_local_valid_attempted: bool = False
    local_valid_marked: bool = False
    publish_binding_attempted: bool = False
    binding_published: bool = False


def _now_ms() -> int:
    return int(time.time() * 1000)
