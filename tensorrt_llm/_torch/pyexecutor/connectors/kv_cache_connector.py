# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file contains the primary interface for the KV Cache Connector.

The KV Cache Connector is a component that allows for remote KV cache access.
It is responsible for:
- Orchestrating the loading and saving of KV cache blocks.
- Managing asynchronous block tx/rx.

It can be used to provide functionalities such as:
1. Disagg
2. KV offload/onboard
3. KV cache sharing
4. P2P KV cache transfer
etc.

The Connector API is split into two parts:
1. The scheduler, which is responsible for orchestration, and building metadata for the workers.
2. The worker, which performs and monitors transfers indicated by the scheduler's metadata.

To implement a custom KV connector, you need to implement both the scheduler and worker-side interfaces.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from tensorrt_llm._utils import mpi_allgather, mpi_broadcast, mpi_rank
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.internal.batch_manager import (
    KvCacheConnectorManager as KvCacheConnectorManagerCpp,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from ..llm_request import get_draft_token_length
from ..scheduler import ScheduledRequests

if TYPE_CHECKING:
    from ..resource_manager import KVCacheManager


# Used to store data for a single inflight request.
@dataclass
class RequestData:
    # The request ID.
    request_id: int
    # The new tokens that were generated in the prior forward pass.
    new_tokens: List[int]
    # The new block IDs allocated in the prior forward pass.
    new_block_ids: List[int]
    # The position of the latest token with computed (valid) kv cache values.
    computed_position: int
    # The number of scheduled tokens for the upcoming forward pass.
    num_scheduled_tokens: int
    # The cumulative chain of block hashes for full blocks of beam 0. Each entry
    # is the hash that KV cache events will report for the corresponding block;
    # the chain is read directly from the KV cache manager's stored block hashes
    # rather than recomputed Python-side. May front-run the corresponding KV cache
    # event emission slightly: when a block becomes full during generation, its
    # hash is committed in the same scheduler step.
    block_hashes: List[int] = field(default_factory=list)
    # The retention priorities for each new block (same length as new_block_ids).
    # Used for priority-based offload filtering. None means use default priority.
    priorities: Optional[List[int]] = None
    # Per-request cache salt that the KV cache manager uses to isolate reuse
    # between requests carrying different salts. Connectors that key cached
    # content on token sequences (e.g. by hashing tokens to a file path or
    # remote object id) MUST mix cache_salt into their identifiers,
    # otherwise blocks from a different salt could be incorrectly reused.
    cache_salt: Optional[str] = None


# A class to store some basic data regarding all inflight requests.
# This is used when calling `build_connector_meta` on the scheduler.
@dataclass
class SchedulerOutput:
    # Requests being scheduled for the first time. Requests will show up in `new_request` exactly once.
    new_requests: List[RequestData] = field(default_factory=list)

    # Requests being scheduled, that have already shown up in `new_requests`.
    cached_requests: List[RequestData] = field(default_factory=list)


class KvCacheConnectorWorker(ABC):
    requires_retryable_kv_admission = False
    requires_disable_overlap_scheduler = False
    requires_disable_attention_dp = False
    requires_uniform_attention_window = False

    def __init__(self, llm_args: TorchLlmArgs):
        self._llm_args = llm_args
        self._metadata = None
        super().__init__()

    def bind_connector_meta(self, metadata: object):
        self._metadata = metadata

    def get_connector_meta(self) -> object:
        return self._metadata

    def _clear_connector_meta(self):
        self._metadata = None

    def register_forward_pass_callable(self) -> Callable:
        """
        This callable will be called at the end of the forward pass.

        Any CUDA calls which happen in the callable will execute on the
        same stream as the forward pass.

        This method is typically used by the connector to insert a
        cuda event into the forward pass cuda stream to obtain a
        signal of when it's appropriate to start offloading cache blocks.
        """

    @abstractmethod
    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        """
        Register the KV cache tensors to the worker.
        This can be used for something like NIXL registration.

        Args:
            kv_cache_tensor: The contiguous KV cache tensor.
        """

    @abstractmethod
    def start_load_kv(self, stream: torch.cuda.Stream):
        """
        Begin loading the KV cache in preparation for the next forward pass.
        Specific blocks to transfer are indicated by the scheduler's metadata.
        """

    @abstractmethod
    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        """
        Wait for a layer to finish being loaded before proceeding with the forward pass on the layer.
        Note: This function is called immediately before the layer's work is enqueued into the stream.

        Args:
            layer_idx: The index of the layer to wait for.
            stream: The stream the forward pass is being executed on.
        """

    @abstractmethod
    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        """
        Begin saving the KV cache for a layer.
        Note: This function is called immediately after the layer's work is enqueued into the stream.

        Args:
            layer_idx: The index of the layer to save.
            stream: The stream the forward pass is being executed on.
        """

    @abstractmethod
    def wait_for_save(self, stream: torch.cuda.Stream):
        """
        Block until all synchronous saving operations are complete. Called at the end of the forward pass.
        """

    @abstractmethod
    def get_finished(
        self, finished_gen_req_ids: List[int], started_loading_req_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Get the requests that have finished loading and saving.

        Args:
            finished_gen_req_ids: The IDs of the requests that have
                finished generating tokens, and are now asynchronously saving.
            started_loading_req_ids: The IDs of the requests that have
                started asynchronously loading.

        Returns:
            The IDs of the requests that have finished saving.
            The IDs of the requests that have finished loading.

        Note: IDs may only be returned from this call after they've been
        provided in the ``finished_gen_req_ids`` and
        ``started_loading_req_ids`` arguments.  Additionally, the runtime
        will only take action based on these returned IDs once they've
        been returned by ALL workers. This allows some workers to take
        longer than others to complete the operations.
        """

    def take_failed_load_request_ids(self) -> set[int]:
        """Drain request IDs whose asynchronous KV loads failed."""
        return set()

    def try_abort_request(self, request_id: int) -> bool:
        """Return whether connector-owned resources are fully cleaned."""
        return True


class KvCacheConnectorScheduler(ABC):
    requires_retryable_kv_admission = False
    requires_disable_overlap_scheduler = False
    requires_disable_attention_dp = False
    requires_uniform_attention_window = False

    def __init__(self, llm_args: TorchLlmArgs):
        self._llm_args = llm_args
        super().__init__()

    @abstractmethod
    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        """
        Build the metadata for the worker.
        This is called by the KV Cache Manager when adding a sequence.
        Args:
            scheduler_output: The data for all inflight requests.

        Returns:
            The metadata for the workers.
        """

    @abstractmethod
    def get_num_new_matched_tokens(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> Tuple[int, bool]:
        """
        Get the number of tokens that can be loaded from remote KV cache.
        This does not include the tokens already matched on device (indicated by `num_computed_tokens`).

        Args:
            request: The request to get the number of tokens for.
            num_computed_tokens: The number of tokens already matched on device.

        Returns:
            The number of tokens that can be loaded from remote KV cache.
            Whether the tokens will be loaded asynchronously.
        """

    @abstractmethod
    def request_finished(self, request: LlmRequest, cache_block_ids: List[int]) -> bool:
        """
        Called when a request is finished generating tokens.

        Args:
            request: The request that finished generating tokens.

        Returns:
            Whether the request is performing asynchronous saving operations.
            If true, this indicates that the kv cache manager should wait
            to deallocate the blocks until the saving has completed
            (determined by ``get_finished`` on the workers).
        """

    @abstractmethod
    def update_state_after_alloc(self, request: LlmRequest, block_ids: List[int]):
        """
        Called after get_num_new_matched_tokens is called to provide the block ids to the scheduler.

        Args:
            request: The request that was allocated resources.
            block_ids: The KV cacheblock IDs that were allocated.
        """

    def wait_for_initialization(self):
        """
        Some connectors need to wait for some resources to be initialized.
        For example, FlexKV needs to wait for the FlexKV manager to be initialized.
        """
        return

    def try_abort_request(self, request_id: int, reason: str) -> bool:
        """Return whether scheduler-side connector state is fully cleaned."""
        return True

    def finalize_successful_load(self, request_id: int) -> bool:
        """Return whether scheduler state for a successful load is cleaned."""
        return True


# An internal dataclass to handle async saving/loading requests.
@dataclass
class AsyncRequests:
    saving: Dict[int, LlmRequest]
    loading: Dict[int, LlmRequest]

    def add_from(self, other: "AsyncRequests"):
        """
        Remove requests from the other `AsyncRequests` object, and add them to this one.
        """
        self.saving.update(other.saving)
        self.loading.update(other.loading)

        other.saving = dict()
        other.loading = dict()

    def extract_by_id(self, saving_ids: List[int], loading_ids: List[int]) -> "AsyncRequests":
        """
        Extract the requests with the given IDs from this `AsyncRequests` object.

        Args:
            saving_ids: The IDs of the requests to extract.
            loading_ids: The IDs of the requests to extract.
        """
        new_async_requests = AsyncRequests(dict(), dict())

        for req_id in saving_ids:
            new_async_requests.saving[req_id] = self.saving[req_id]
            del self.saving[req_id]
        for req_id in loading_ids:
            new_async_requests.loading[req_id] = self.loading[req_id]
            del self.loading[req_id]

        return new_async_requests

    def discard_request_id(self, request_id: int) -> Optional[LlmRequest]:
        """Discard an asynchronous request if it is currently tracked."""
        request = self.loading.pop(request_id, None)
        saving_request = self.saving.pop(request_id, None)
        return request if request is not None else saving_request

    @property
    def saving_ids(self) -> Set[int]:
        """
        Get the IDs of the requests that are being saved asynchronously.
        """
        return set(self.saving.keys())

    @property
    def loading_ids(self) -> Set[int]:
        """
        Get the IDs of the requests that are being loaded asynchronously.
        """
        return set(self.loading.keys())


@dataclass
class KvCacheConnectorPollResult:
    """Request-scoped results from one connector progress poll."""

    finished_save_requests: List[LlmRequest] = field(default_factory=list)
    failed_load_requests: List[LlmRequest] = field(default_factory=list)


@dataclass(frozen=True)
class _ConnectorFatalErrorPayload:
    """A picklable, sanitized worker-hook error shared by every rank."""

    exception_type: str
    message: str


class _CoordinationMode(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    ABORT = "abort"


class _SchedulerAck(Enum):
    SUCCESS = "success"
    ABORT = "abort"


@dataclass
class _RequestCoordinationState:
    """Cross-rank lifecycle state for one asynchronously loaded request."""

    request: LlmRequest
    mode: _CoordinationMode = _CoordinationMode.PENDING
    local_abort_reason: Optional[str] = None
    consensus_abort_reason: Optional[str] = None
    local_transfer_released: bool = False
    all_worker_releases_complete: bool = False
    local_bookkeeping_cleaned: bool = False
    scheduler_ack: Optional[_SchedulerAck] = None

    @staticmethod
    def _merge_abort_reason(current: Optional[str], incoming: str) -> str:
        if current == "transfer_failed" or incoming == "transfer_failed":
            return "transfer_failed"
        return current or incoming

    def record_local_abort(self, reason: str) -> None:
        self.local_abort_reason = self._merge_abort_reason(self.local_abort_reason, reason)
        self.mode = _CoordinationMode.ABORT
        if self.scheduler_ack is _SchedulerAck.SUCCESS:
            self.scheduler_ack = None

    def observe_consensus_abort(self, reason: str) -> None:
        self.record_local_abort(reason)
        self.consensus_abort_reason = self._merge_abort_reason(self.consensus_abort_reason, reason)

    def mark_local_transfer_released(self) -> None:
        if self.mode is not _CoordinationMode.ABORT:
            raise RuntimeError("transfer release requires an abort")
        self.local_transfer_released = True

    def mark_all_worker_releases_complete(self) -> None:
        if self.mode is not _CoordinationMode.ABORT:
            raise RuntimeError("all worker releases require an abort")
        if not self.local_transfer_released:
            raise RuntimeError("all worker releases require local transfer release")
        self.all_worker_releases_complete = True

    def mark_local_bookkeeping_cleaned(self) -> None:
        if not self.local_transfer_released:
            raise RuntimeError("local bookkeeping cleanup requires transfer release")
        self.local_bookkeeping_cleaned = True

    def mark_scheduler_acked(self, ack: _SchedulerAck) -> None:
        if ack is _SchedulerAck.ABORT:
            if self.mode is not _CoordinationMode.ABORT:
                raise RuntimeError("scheduler abort acknowledgement requires an abort")
            if not self.all_worker_releases_complete:
                raise RuntimeError("scheduler abort acknowledgement requires all worker releases")
        elif self.mode is _CoordinationMode.ABORT:
            raise RuntimeError("scheduler success acknowledgement cannot follow an abort")
        else:
            self.mode = _CoordinationMode.SUCCESS
        self.scheduler_ack = ack


@dataclass(frozen=True)
class _ConnectorPollPayload:
    """Immutable request-ID-only snapshot exchanged by connector workers."""

    finished_save_request_ids: tuple[int, ...] = ()
    finished_load_request_ids: tuple[int, ...] = ()
    abort_intents: tuple[tuple[int, str], ...] = ()
    locally_released_request_ids: frozenset[int] = frozenset()
    locally_cleaned_request_ids: frozenset[int] = frozenset()
    scheduler_success_ack_request_ids: frozenset[int] = frozenset()
    scheduler_abort_ack_request_ids: frozenset[int] = frozenset()
    fatal_error: Optional[_ConnectorFatalErrorPayload] = None


class KvCacheConnectorSchedulerOutputRequest:
    def __init__(self):
        self.block_ids = []
        self.tokens = []

    def update_and_build_data(self, req: LlmRequest, kv_cache_manager: "KVCacheManager"):
        block_ids = kv_cache_manager.get_cache_indices(req)
        tokens = req.get_tokens(0)

        # Commit hashes for any blocks that have become full since the last call
        # and read back the full cumulative chain. The C++ side sets each block's
        # mBlockKey/mHash on first call, so subsequent calls become pure lookups.
        block_hashes = kv_cache_manager.commit_and_get_block_hashes(req)

        new_block_ids = block_ids[len(self.block_ids) :]
        new_tokens = tokens[len(self.tokens) :]

        self.block_ids.extend(new_block_ids)
        self.tokens.extend(new_tokens)

        if req.state in (
            LlmRequestState.CONTEXT_INIT,
            LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
        ):
            computed_position = req.context_current_position
            num_scheduled_tokens = min(req.context_remaining_length, req.context_chunk_size)
        else:
            computed_position = len(tokens) - 1
            num_scheduled_tokens = 1 + get_draft_token_length(
                req
            )  # Specdec with draft tokens is not supported yet.

        # Get retention priority for each new block only if retention config is provided
        # (for priority-based offload filtering)
        priorities = None
        if req.kv_cache_retention_config is not None:
            priorities = [
                kv_cache_manager.get_priority_by_block_id(block_id) for block_id in new_block_ids
            ]

        return RequestData(
            req.request_id,
            new_tokens,
            new_block_ids,
            computed_position,
            num_scheduled_tokens,
            block_hashes=block_hashes,
            priorities=priorities,
            cache_salt=req.cache_salt,
        )


class KvCacheConnectorSchedulerOutputManager:
    def __init__(self):
        self.requests = defaultdict(KvCacheConnectorSchedulerOutputRequest)
        self.external_loads = dict()

    def build_scheduler_output(
        self,
        scheduled_batch: ScheduledRequests,
        new_async_requests: AsyncRequests,
        kv_cache_manager: "KVCacheManager",
    ):
        scheduler_output = SchedulerOutput()

        for req in scheduled_batch.context_requests:
            if req.request_id in new_async_requests.loading_ids:
                continue

            is_new = req.request_id not in self.requests

            request_data = self.requests[req.request_id].update_and_build_data(
                req, kv_cache_manager
            )

            # Don't include the connector matched tokens in the initial scheduler output.
            if req.request_id in self.external_loads:
                request_data.computed_position -= self.external_loads[req.request_id]

            if is_new:
                scheduler_output.new_requests.append(request_data)
            else:
                scheduler_output.cached_requests.append(request_data)

        for req in scheduled_batch.generation_requests:
            request_data = self.requests[req.request_id].update_and_build_data(
                req, kv_cache_manager
            )

            scheduler_output.cached_requests.append(request_data)

        self.external_loads = dict()

        return scheduler_output

    def record_new_matched_tokens(self, request: LlmRequest, num_new_matched_tokens: int):
        self.external_loads[request.request_id] = num_new_matched_tokens

    def discard_request_id(self, request_id: int) -> None:
        """Discard all scheduler-output state for a request."""
        self.requests.pop(request_id, None)
        self.external_loads.pop(request_id, None)


class KvCacheConnectorManager(KvCacheConnectorManagerCpp):
    """
    The KvCacheConnectorManager is used to manager connector-related state.

    It has the following responsibilities:
    1. Managing the state of async requests (both offload and onboard)
    2. Handling MPI communication. We only run the leader on one rank,
       but need the results of the leader API on all ranks.

    Note: This class is solely an implementation detail, and is not part of the connector interface itself.
    When implementing a connector API, you do not need to implement this class.
    """

    def __init__(
        self, worker: KvCacheConnectorWorker, scheduler: Optional[KvCacheConnectorScheduler]
    ):
        assert (scheduler is not None) == (mpi_rank() == 0), (
            "The scheduler may only exist on rank 0!"
        )

        super().__init__()

        self.worker = worker
        self.scheduler = scheduler

        # Requests that haven't yet been passed into get_finished.
        self.new_async_requests = AsyncRequests(dict(), dict())

        # Requests that have been passed into get_finished, but haven't yet been returned.
        self.pending_async_requests = AsyncRequests(dict(), dict())

        # Requests that have been returned from get_finished locally, but haven't yet been returned by all workers.
        self.local_finished_async_requests = AsyncRequests(dict(), dict())

        # Requests that have finished loading asynchronously.
        self.finished_async_loading_requests = dict()

        # Abort coordination is deliberately request-ID based. Connector block
        # IDs are rank-local and therefore must never cross the TP collective.
        self._request_coordination: Dict[int, _RequestCoordinationState] = {}

        self._scheduler_output = None
        self.scheduler_output_manager = KvCacheConnectorSchedulerOutputManager()

    def _connector_requires(self, attr: str) -> bool:
        return any(
            bool(getattr(connector, attr, False))
            for connector in (self.worker, self.scheduler)
            if connector is not None
        )

    @property
    def requires_retryable_kv_admission(self) -> bool:
        return self._connector_requires("requires_retryable_kv_admission")

    @property
    def requires_disable_overlap_scheduler(self) -> bool:
        return self._connector_requires("requires_disable_overlap_scheduler")

    @property
    def requires_disable_attention_dp(self) -> bool:
        return self._connector_requires("requires_disable_attention_dp")

    @property
    def requires_uniform_attention_window(self) -> bool:
        return self._connector_requires("requires_uniform_attention_window")

    def _run_on_leader(self, f: Callable[[], Any]) -> Any:
        """
        Run a function on the leader rank, and broadcast the result to all other ranks.
        """
        if self.scheduler is not None:
            assert mpi_rank() == 0, "The scheduler may only exist on rank 0!"
            res = f()
        else:
            res = None
        return mpi_broadcast(res, root=0)

    def get_num_new_matched_tokens(self, request: LlmRequest, num_computed_tokens: int) -> int:
        if request.is_generation_only_request:
            raise RuntimeError("Connector API is not supported for generation-only requests!")

        num_tokens, load_kv_async = self._run_on_leader(
            lambda: self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)
        )

        if num_tokens == 0 and load_kv_async:
            raise RuntimeError("load_kv_async must be False when num_tokens is 0!")

        # TODO(jthomson04): This part is a bit ugly.
        # When the connector indicates that a request will be loaded
        # asynchronously, we need to suspend its execution. This is
        # problematic, since at the point when this function is called,
        # the request has already been scheduled! Because of this, we
        # need to remove it from our list of scheduled requests
        # (see `take_scheduled_requests_pending_load`).
        if load_kv_async:
            self.new_async_requests.loading[request.request_id] = request

        self.scheduler_output_manager.record_new_matched_tokens(request, num_tokens)

        request.py_num_connector_matched_tokens = num_tokens

        return num_tokens

    def should_add_sequence(self, request: LlmRequest) -> bool:
        req_id = request.request_id
        return req_id not in self.finished_async_loading_requests

    def build_scheduler_output(
        self, scheduled_batch: ScheduledRequests, kv_cache_manager: "KVCacheManager"
    ):
        self._scheduler_output = self.scheduler_output_manager.build_scheduler_output(
            scheduled_batch, self.new_async_requests, kv_cache_manager
        )

    def take_scheduled_requests_pending_load(self, scheduled_requests: ScheduledRequests):
        """
        Remove context requests from our list of scheduled requests that are being loaded asynchronously.
        This is done to prevent the runtime from attempting to load the KV cache for these requests.

        Args:
            scheduled_requests: The scheduled requests.

        Returns:
            The scheduled requests with the context requests that are being loaded asynchronously removed.
        """

        for key in ["context_requests_chunking", "context_requests_last_chunk"]:
            allowed_context_requests = []
            for req in getattr(scheduled_requests, key):
                # If this request is being loaded asynchronously, in
                # addition to removing it from the list of scheduled
                # requests, we also need to update its state.
                if req.request_id in self.new_async_requests.loading.keys():
                    req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS

                    # Replace the request with the canonical request.
                    self.new_async_requests.loading[req.request_id] = req
                else:
                    allowed_context_requests.append(req)
            setattr(scheduled_requests, key, allowed_context_requests)

    def handle_metadata(self) -> object:
        if self._scheduler_output is None:
            return

        metadata = self._run_on_leader(
            lambda: self.scheduler.build_connector_meta(self._scheduler_output)
        )

        self._scheduler_output = None

        self.worker.bind_connector_meta(metadata)

    def request_finished(self, req: LlmRequest, cache_block_ids: List[int]) -> bool:
        """
        Called when a request is finished generating tokens.

        Args:
            req: The request that finished generating tokens.

        Returns:
            Whether the request is performing asynchronous saving
            operations. If true, we do not immediately call
            free_resources on the request.
        """

        if req.request_id in self.finished_async_loading_requests:
            del self.finished_async_loading_requests[req.request_id]

        saving_async = self._run_on_leader(
            lambda: self.scheduler.request_finished(req, cache_block_ids)
        )

        # This is similar to take_scheduled_requests_pending_load.
        # We need to update the request's state to indicate that it's still being used, but isn't schedulable.
        if saving_async:
            self.new_async_requests.saving[req.request_id] = req
            req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

        return saving_async

    def _find_tracked_request(self, request_id: int) -> Optional[LlmRequest]:
        for async_requests in (
            self.new_async_requests,
            self.pending_async_requests,
            self.local_finished_async_requests,
        ):
            request = async_requests.loading.get(request_id)
            if request is None:
                request = async_requests.saving.get(request_id)
            if request is not None:
                return request
        request = self.finished_async_loading_requests.get(request_id)
        if request is not None:
            return request
        state = self._request_coordination.get(request_id)
        return None if state is None else state.request

    def _get_or_create_coordination_state(
        self, request_id: int
    ) -> Optional[_RequestCoordinationState]:
        state = self._request_coordination.get(request_id)
        if state is not None:
            return state
        request = self._find_tracked_request(request_id)
        if request is None:
            return None
        state = _RequestCoordinationState(request)
        self._request_coordination[request_id] = state
        return state

    def _record_abort_intent(self, request_id: int, reason: str) -> bool:
        state = self._get_or_create_coordination_state(request_id)
        if state is None:
            return False
        state.record_local_abort(reason)
        return True

    def try_abort_request(self, request_id: int, reason: str) -> bool:
        """Queue a noncollective request abort.

        Returning ``False`` asks the caller to retry until TP consensus has
        drained all connector-owned state. An untracked request already has
        complete cleanup and therefore returns ``True`` immediately.
        """
        if not self._record_abort_intent(request_id, reason):
            return True
        return False

    def _tracked_ids(self) -> Set[int]:
        return set().union(
            self.new_async_requests.loading_ids,
            self.new_async_requests.saving_ids,
            self.pending_async_requests.loading_ids,
            self.pending_async_requests.saving_ids,
            self.local_finished_async_requests.loading_ids,
            self.local_finished_async_requests.saving_ids,
            self.finished_async_loading_requests.keys(),
            self._request_coordination.keys(),
        )

    def _filter_tracked_request_ids(self, values: object) -> Set[int]:
        """Return valid, currently tracked integer request IDs."""
        if isinstance(values, (str, bytes)):
            return set()
        try:
            candidates = list(values)  # type: ignore[arg-type]
        except Exception:
            return set()
        tracked_ids = self._tracked_ids()
        return {
            request_id
            for request_id in candidates
            if type(request_id) is int and request_id in tracked_ids
        }

    @staticmethod
    def _to_fatal_error_payload(error: Exception) -> _ConnectorFatalErrorPayload:
        return _ConnectorFatalErrorPayload(type(error).__name__, str(error))

    def _discard_request_bookkeeping(self, request_id: int) -> None:
        """Idempotently discard non-coordination state for a request.

        Scheduler output is transient and may contain connector-provided or
        test-injected malformed entries. Such entries cannot safely be
        associated with a future request, so discard them. Other unexpected
        exceptions are retried before the next collective while canonical and
        coordination state remains intact.
        """
        if self._scheduler_output is not None:

            def keep_request_data(data: object) -> bool:
                try:
                    return data.request_id != request_id  # type: ignore[attr-defined]
                except Exception:
                    return False

            try:
                new_requests = list(self._scheduler_output.new_requests)
                cached_requests = list(self._scheduler_output.cached_requests)
            except Exception:
                # A corrupt transient output cannot be reused safely. Clearing
                # it is preferable to retaining a request whose cleanup has
                # already reached cross-rank consensus.
                self._scheduler_output = None
            else:
                try:
                    self._scheduler_output.new_requests = [
                        data for data in new_requests if keep_request_data(data)
                    ]
                    self._scheduler_output.cached_requests = [
                        data for data in cached_requests if keep_request_data(data)
                    ]
                except Exception:
                    self._scheduler_output = None

        try:
            self.scheduler_output_manager.discard_request_id(request_id)
        except Exception:
            # The concrete manager uses ordinary dictionaries, but retain an
            # infallible fallback so finalization cannot escape the executor
            # loop if instrumentation or a connector replaces the helper.
            try:
                self.scheduler_output_manager.requests.pop(request_id, None)
                self.scheduler_output_manager.external_loads.pop(request_id, None)
            except Exception:
                # The concrete stores are ordinary dictionaries; this guard is
                # solely to keep post-collective finalization non-throwing.
                pass

        for requests in (
            self.new_async_requests,
            self.pending_async_requests,
            self.local_finished_async_requests,
        ):
            requests.discard_request_id(request_id)
        self.finished_async_loading_requests.pop(request_id, None)

    def _finalize_aborted_request(self, request_id: int) -> Optional[LlmRequest]:
        """Remove coordination only after every rank cleaned request data."""
        state = self._request_coordination.pop(request_id, None)
        return None if state is None else state.request

    def _record_local_poll_progress(
        self,
        finished_save_request_ids: object,
        finished_load_request_ids: object,
        failed_load_request_ids: object,
    ) -> None:
        """Fold this rank's worker poll results into local bookkeeping.

        Runs on the path to the TP collective, so it is intentionally free of
        broadcasts; the caller converts any failure here into a shared fatal.
        """
        finished_save_request_ids = self._filter_tracked_request_ids(finished_save_request_ids)
        finished_load_request_ids = self._filter_tracked_request_ids(finished_load_request_ids)
        failed_load_request_ids = self._filter_tracked_request_ids(failed_load_request_ids)
        for request_id in failed_load_request_ids:
            self._record_abort_intent(request_id, "transfer_failed")

        # Remove the requests from our pending list that have finished locally.
        finished_save_request_ids &= self.pending_async_requests.saving_ids
        finished_load_request_ids &= self.pending_async_requests.loading_ids
        new_local_finished_async_requests = self.pending_async_requests.extract_by_id(
            sorted(finished_save_request_ids), sorted(finished_load_request_ids)
        )

        # Add these requests to our list of locally finished requests.
        self.local_finished_async_requests.add_from(new_local_finished_async_requests)

    def _release_aborted_transfers(self) -> None:
        """Complete rank-local connector and data cleanup for abort intents.

        The snapshot is taken over a copy of the intent set so a concurrently
        mutated dict cannot raise on the path to the collective.
        """
        for request_id, state in list(self._request_coordination.items()):
            if state.consensus_abort_reason is None:
                continue
            if state.local_transfer_released:
                continue
            if self.worker.try_abort_request(request_id) is not True:
                raise RuntimeError(f"KV connector worker failed to release request {request_id}")
            state.mark_local_transfer_released()

        # Data cleanup is rank-local and idempotent. Attempt it as soon as the
        # local worker has released its transfer, but retain the canonical
        # request and coordination until the collective proves every rank succeeded.
        for request_id, state in list(self._request_coordination.items()):
            if state.consensus_abort_reason is None:
                continue
            if not state.local_transfer_released:
                continue
            if state.local_bookkeeping_cleaned:
                continue
            try:
                self._discard_request_bookkeeping(request_id)
            except Exception:
                continue
            state.mark_local_bookkeeping_cleaned()

    def _build_collective_poll_payload(
        self, fatal_error: "Optional[_ConnectorFatalErrorPayload]"
    ) -> _ConnectorPollPayload:
        """Snapshot rank-local state into the allgather payload.

        Must never raise: a payload snapshot is not allowed to be the reason a
        rank misses the collective. On failure it returns an empty, fatal-flagged
        payload so every rank still rendezvous and fails symmetrically.
        """
        is_leader = mpi_rank() == 0
        try:
            states = tuple(self._request_coordination.items())
            return _ConnectorPollPayload(
                finished_save_request_ids=tuple(
                    sorted(self.local_finished_async_requests.saving_ids)
                ),
                finished_load_request_ids=tuple(
                    sorted(self.local_finished_async_requests.loading_ids)
                ),
                abort_intents=tuple(
                    sorted(
                        (request_id, state.local_abort_reason)
                        for request_id, state in states
                        if state.local_abort_reason is not None
                    )
                ),
                locally_released_request_ids=frozenset(
                    request_id for request_id, state in states if state.local_transfer_released
                ),
                locally_cleaned_request_ids=frozenset(
                    request_id for request_id, state in states if state.local_bookkeeping_cleaned
                ),
                scheduler_success_ack_request_ids=frozenset(
                    request_id
                    for request_id, state in states
                    if is_leader and state.scheduler_ack is _SchedulerAck.SUCCESS
                ),
                scheduler_abort_ack_request_ids=frozenset(
                    request_id
                    for request_id, state in states
                    if is_leader and state.scheduler_ack is _SchedulerAck.ABORT
                ),
                fatal_error=fatal_error,
            )
        except Exception as error:
            return _ConnectorPollPayload(
                fatal_error=fatal_error or self._to_fatal_error_payload(error)
            )

    def get_finished(self) -> KvCacheConnectorPollResult:
        """
        Process requests that have finished loading and saving.

        Returns:
            The requests that have newly finished saving.
        """
        started_loading_req_ids = list(self.new_async_requests.loading_ids)
        finished_gen_req_ids = list(self.new_async_requests.saving_ids)

        # Add the requests to our list of outstanding (still in progress)
        # requests.
        self.pending_async_requests.add_from(self.new_async_requests)

        # Pass these newly finished requests into get_finished, and get
        # the list of requests that have finished saving and loading.
        fatal_error = None
        try:
            worker_result = self.worker.get_finished(finished_gen_req_ids, started_loading_req_ids)
            finished_save_request_ids, finished_load_request_ids = worker_result
        except Exception as error:
            fatal_error = self._to_fatal_error_payload(error)
            finished_save_request_ids, finished_load_request_ids = (), ()

        try:
            failed_load_request_ids = self.worker.take_failed_load_request_ids()
        except Exception as error:
            if fatal_error is None:
                fatal_error = self._to_fatal_error_payload(error)
            failed_load_request_ids = ()

        # Everything from here to the collective mutates only rank-local
        # bookkeeping, but it MUST NOT raise out of this method: if a single
        # rank skipped the mpi_allgather below, the TP group would desync and
        # every rank would hang. Convert any unexpected failure into a shared
        # fatal that all ranks observe symmetrically *after* the collective.
        try:
            self._record_local_poll_progress(
                finished_save_request_ids,
                finished_load_request_ids,
                failed_load_request_ids,
            )
            self._release_aborted_transfers()
        except Exception as error:
            if fatal_error is None:
                fatal_error = self._to_fatal_error_payload(error)

        local_poll_payload = self._build_collective_poll_payload(fatal_error)
        rank_poll_payloads = mpi_allgather(local_poll_payload)

        invalid_payload_ranks = [
            rank
            for rank, payload in enumerate(rank_poll_payloads)
            if not isinstance(payload, _ConnectorPollPayload)
        ]
        if invalid_payload_ranks:
            raise RuntimeError(
                "KV connector worker returned an invalid collective payload on rank "
                f"{min(invalid_payload_ranks)}"
            )

        fatal_errors = [
            (rank, payload.fatal_error)
            for rank, payload in enumerate(rank_poll_payloads)
            if isinstance(payload.fatal_error, _ConnectorFatalErrorPayload)
        ]
        if fatal_errors:
            _, shared_error = min(fatal_errors, key=lambda item: item[0])
            raise RuntimeError(
                f"KV connector worker hook failed: "
                f"{shared_error.exception_type}: {shared_error.message}"
            )

        # Merge request-ID intents deterministically. A transfer failure wins
        # over cancellation irrespective of rank order.
        for payload in rank_poll_payloads:
            for request_id, reason in payload.abort_intents:
                if type(request_id) is not int or not isinstance(reason, str):
                    continue
                state = self._get_or_create_coordination_state(request_id)
                if state is None:
                    continue
                state.observe_consensus_abort(reason)

        released_request_id_sets = [
            {
                request_id
                for request_id in payload.locally_released_request_ids
                if type(request_id) is int
            }
            for payload in rank_poll_payloads
        ]
        globally_released_request_ids = (
            set.intersection(*released_request_id_sets) if released_request_id_sets else set()
        )
        for request_id in globally_released_request_ids:
            state = self._request_coordination.get(request_id)
            if state is not None and state.consensus_abort_reason is not None:
                state.mark_all_worker_releases_complete()

        gathered_abort_ack_request_ids = set().union(
            *(
                {
                    request_id
                    for request_id in payload.scheduler_abort_ack_request_ids
                    if type(request_id) is int
                }
                for payload in rank_poll_payloads
            )
        )
        data_cleaned_sets = [
            {
                request_id
                for request_id in payload.locally_cleaned_request_ids
                if type(request_id) is int
            }
            for payload in rank_poll_payloads
        ]
        all_ranks_data_cleaned = (
            set.intersection(*data_cleaned_sets) if data_cleaned_sets else set()
        )
        gathered_success_ack_request_ids = set().union(
            *(
                {
                    request_id
                    for request_id in payload.scheduler_success_ack_request_ids
                    if type(request_id) is int
                }
                for payload in rank_poll_payloads
            )
        )

        # Find only the requests that have been reported complete by all workers.
        intersect_finished_save_request_ids = set.intersection(
            *[set(payload.finished_save_request_ids) for payload in rank_poll_payloads]
        )
        intersect_finished_load_request_ids = set.intersection(
            *[set(payload.finished_load_request_ids) for payload in rank_poll_payloads]
        )
        abort_ids = {
            request_id
            for request_id, state in self._request_coordination.items()
            if state.consensus_abort_reason is not None
        }
        intersect_finished_save_request_ids -= abort_ids
        intersect_finished_load_request_ids -= abort_ids

        # Successful loads also own scheduler-side lease state. Rank 0 cleans
        # it only after every worker has reported local success, then publishes
        # the acknowledgement through the next poll's sole collective.
        if self.scheduler is not None:
            for request_id in intersect_finished_load_request_ids:
                state = self._get_or_create_coordination_state(request_id)
                if state is None or state.mode is _CoordinationMode.ABORT:
                    continue
                if state.scheduler_ack is _SchedulerAck.SUCCESS:
                    continue
                try:
                    if self.scheduler.finalize_successful_load(request_id) is True:
                        state.mark_scheduler_acked(_SchedulerAck.SUCCESS)
                except Exception:
                    pass

        resumable_load_request_ids = (
            intersect_finished_load_request_ids & gathered_success_ack_request_ids
        )

        # Remove these requests from our list of locally finished requests.
        all_finished = self.local_finished_async_requests.extract_by_id(
            intersect_finished_save_request_ids, resumable_load_request_ids
        )

        # For requests that have finished loading, move them back to the context state.
        for request_id, request in all_finished.loading.items():
            request.state = LlmRequestState.CONTEXT_INIT
            self.finished_async_loading_requests[request_id] = request
            self._request_coordination.pop(request_id, None)

        # Rank 0 performs scheduler cleanup only after the current collective
        # proves every worker released its transfer. The acknowledgement is
        # intentionally published by the next poll's single collective.
        if self.scheduler is not None:
            for request_id, state in list(self._request_coordination.items()):
                reason = state.consensus_abort_reason
                if reason is None or not state.all_worker_releases_complete:
                    continue
                if state.scheduler_ack is _SchedulerAck.ABORT:
                    continue
                try:
                    if self.scheduler.try_abort_request(request_id, reason) is True:
                        state.mark_scheduler_acked(_SchedulerAck.ABORT)
                except Exception:
                    pass

        failed_requests = []
        completed_abort_request_ids = gathered_abort_ack_request_ids & all_ranks_data_cleaned
        for request_id in sorted(completed_abort_request_ids):
            state = self._request_coordination.get(request_id)
            if state is None or state.consensus_abort_reason is None:
                continue
            reason = state.consensus_abort_reason
            request = self._finalize_aborted_request(request_id)
            if reason == "transfer_failed" and request is not None:
                failed_requests.append(request)

        # Return the requests that have finished saving.
        # The execution loop will call _terminate_request on these requests.
        return KvCacheConnectorPollResult(
            finished_save_requests=list(all_finished.saving.values()),
            failed_load_requests=failed_requests,
        )

    def update_state_after_alloc(self, req: LlmRequest, block_ids: List[int]):
        if self.scheduler is not None:
            self.scheduler.update_state_after_alloc(req, block_ids)

    def set_scheduler_output(self, scheduler_output: SchedulerOutput):
        self._scheduler_output = scheduler_output

    def layer_pre_hook(self, module, *args):
        self.worker.wait_for_layer_load(module.layer_idx, torch.cuda.current_stream())

    def layer_post_hook(self, module, *args):
        self.worker.save_kv_layer(module.layer_idx, torch.cuda.current_stream())

    def wait_for_initialization(self):
        if self.scheduler is not None:
            self.scheduler.wait_for_initialization()
