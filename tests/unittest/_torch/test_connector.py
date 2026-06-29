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

import pickle
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import cloudpickle
import mpi4py
import pytest

from tensorrt_llm import mpi_rank
from tensorrt_llm._torch.pyexecutor.connectors import kv_cache_connector
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    AsyncRequests,
    KvCacheConnectorManager,
    KvCacheConnectorPollResult,
    KvCacheConnectorScheduler,
    KvCacheConnectorSchedulerOutputManager,
    KvCacheConnectorWorker,
)
from tensorrt_llm._torch.pyexecutor.connectors.remote_g2 import (
    RemoteG2BindingState,
    TargetRemoteG2BindingStore,
)
from tensorrt_llm._torch.pyexecutor.connectors.remote_g2_connector import (
    RemoteG2ConnectorMetadata,
    RemoteG2KvCacheConnectorScheduler,
    RemoteG2KvCacheConnectorWorker,
)
from tensorrt_llm._torch.pyexecutor.connectors.remote_g2_transfer import RemoteG2TransferState
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

cloudpickle.register_pickle_by_value(sys.modules[__name__])
mpi4py.MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def run_across_mpi(executor, fun, num_ranks):
    return list(executor.starmap(fun, [() for i in range(num_ranks)]))


class _ExplodingRequestData:

    @property
    def request_id(self):
        raise RuntimeError("malformed scheduler output")


class _RankZeroReleaseStore:

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def release_and_discard(self, request_id, reason):
        self.calls.append((request_id, reason))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _RemoteG2MpiResult:

    def __init__(self, record, initial_state, poll_states=(), quiesce=(True, )):
        self.record = record
        self.initial_state = initial_state
        self.poll_states = list(poll_states)
        self.quiesce_outcomes = list(quiesce)

    def poll_state(self):
        if self.poll_states:
            return self.poll_states.pop(0)
        return self.initial_state

    def quiesce(self):
        if self.quiesce_outcomes:
            return self.quiesce_outcomes.pop(0)
        return True


class _RemoteG2MpiAdapter:
    supports_synchronous_release = True

    def __init__(self, result):
        self.result = result

    def start_transfer(self, record):
        return self.result


def _remote_g2_mpi_record():
    return SimpleNamespace(
        request_id=42,
        lease_id="lease-tp",
        plan=SimpleNamespace(
            plan_id="plan-tp",
            source_tier="g2",
            source_worker_id=7,
        ),
        source_generation=1,
        resolve_result=SimpleNamespace(descriptors=()),
        bound_blocks=(),
        matched_tokens=16,
        release_attempted=False,
        release_completed=False,
        release_reason=None,
        state=RemoteG2BindingState.RESOLVED,
    )


def test_connector_failure_abort_defaults_are_backward_compatible():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    assert KvCacheConnectorWorker.take_failed_load_request_ids(worker) == set()
    assert KvCacheConnectorWorker.abort_request(worker, 42) is True

    scheduler = MagicMock(spec=KvCacheConnectorScheduler)
    assert KvCacheConnectorScheduler.abort_request(scheduler, 42,
                                                   "cancelled") is True
    assert KvCacheConnectorScheduler.finish_load(scheduler, 42) is True


def test_async_requests_discard_request_id_is_idempotent():
    req = MagicMock(request_id=42)
    async_requests = AsyncRequests(saving={}, loading={42: req})

    assert async_requests.discard_request_id(42) is req
    assert async_requests.discard_request_id(42) is None


def test_async_requests_discard_request_id_returns_saving_request():
    req = MagicMock(request_id=42)
    async_requests = AsyncRequests(saving={42: req}, loading={})

    assert async_requests.discard_request_id(42) is req
    assert async_requests.saving == {}


def test_async_requests_discard_request_id_prefers_loading_and_clears_both():
    loading_req = MagicMock(request_id=42)
    saving_req = MagicMock(request_id=42)
    async_requests = AsyncRequests(saving={42: saving_req},
                                   loading={42: loading_req})

    assert async_requests.discard_request_id(42) is loading_req
    assert async_requests.loading == {}
    assert async_requests.saving == {}


def test_scheduler_output_manager_discard_request_id_clears_all_state():
    manager = KvCacheConnectorSchedulerOutputManager()
    manager.requests[42]
    manager.external_loads[42] = 16

    manager.discard_request_id(42)

    assert 42 not in manager.requests
    assert 42 not in manager.external_loads


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
# TODO(jthomson04): I don't have the slightest idea why this test is leaking threads.
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_get_finished_allgather(mpi_pool_executor):

    def test():
        worker = MagicMock()

        if mpi_rank() == 0:
            scheduler = MagicMock()

            scheduler.request_finished.return_value = True
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock()

        req.request_id = 42

        manager.request_finished(req, [])

        # To start, make both workers return nothing.
        worker.get_finished.return_value = ([], [])

        assert manager.get_finished() == KvCacheConnectorPollResult()

        assert worker.get_finished.call_count == 1
        assert worker.get_finished.call_args[0] == ([42], [])

        worker.get_finished.reset_mock()

        # Now, only return the request id on one worker.
        if mpi_rank() == 0:
            worker.get_finished.return_value = ([42], [])
        else:
            worker.get_finished.return_value = ([], [])

        # It should still return nothing, since rank 1 is still saving.
        assert manager.get_finished() == KvCacheConnectorPollResult()

        assert worker.get_finished.call_count == 1
        assert worker.get_finished.call_args[0] == ([], [])

        # Now, also return it on worker 1.
        if mpi_rank() == 0:
            worker.get_finished.return_value = ([], [])
        else:
            worker.get_finished.return_value = ([42], [])

        assert manager.get_finished() == KvCacheConnectorPollResult(
            finished_saving=[req])

    run_across_mpi(mpi_pool_executor, test, 2)


def test_connector_manager_request_abort_filters_untracked_and_reused_ids(
        monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector.mpi_rank",
        lambda: 0)
    worker = MagicMock()
    scheduler = MagicMock()
    manager = KvCacheConnectorManager(worker, scheduler)

    assert manager.request_abort(42, "cancelled") is True

    old_request = MagicMock(request_id=42)
    manager.new_async_requests.loading[42] = old_request
    assert manager.request_abort(42, "cancelled") is False
    assert manager._abort_requests[42] is old_request

    manager._discard_request_data(42)
    manager._finalize_request_id(42)
    new_request = MagicMock(request_id=42)
    manager.new_async_requests.loading[42] = new_request
    assert manager.request_abort(42, "cancelled") is False
    assert manager._abort_requests[42] is new_request


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_failed_load_reaches_tp_consensus_and_drains_state(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.get_finished.return_value = (([], [42]) if mpi_rank() == 0 else
                                            ([], []))
        worker.take_failed_load_request_ids.return_value = set()
        worker.abort_request.return_value = True
        scheduler = MagicMock() if mpi_rank() == 0 else None
        if scheduler is not None:
            scheduler.abort_request.return_value = True
        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        manager.new_async_requests.loading[42] = request
        scheduler_request = manager.scheduler_output_manager.requests[42]
        # Seed production scheduler bookkeeping with deliberately different
        # rank-local allocator IDs. Consensus must never inspect or compare it.
        scheduler_request.block_ids = [1000 + mpi_rank()]
        manager.scheduler_output_manager.external_loads[42] = 16
        manager.scheduler_output_manager.discard_request_id = MagicMock(
            side_effect=RuntimeError("injected discard failure"))
        manager._scheduler_output = MagicMock()
        unrelated_output = MagicMock(request_id=77)
        manager._scheduler_output.new_requests = [
            _ExplodingRequestData(),
            MagicMock(request_id=42)
        ]
        manager._scheduler_output.cached_requests = [
            MagicMock(request_id=42), unrelated_output
        ]
        original_pending_discard = manager.pending_async_requests.discard_request_id
        pending_discard_calls = 0

        def flaky_pending_discard(request_id):
            nonlocal pending_discard_calls
            pending_discard_calls += 1
            if pending_discard_calls == 1:
                raise RuntimeError("injected finalization failure")
            return original_pending_discard(request_id)

        if mpi_rank() == 1:
            manager.pending_async_requests.discard_request_id = flaky_pending_discard

        if mpi_rank() == 1:
            worker.take_failed_load_request_ids.return_value = {42, 999, "bad"}
        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert request.state != LlmRequestState.CONTEXT_INIT

        # Failure beats a peer completion and a simultaneous cancellation.
        manager.request_abort(42, "cancelled")
        worker.get_finished.return_value = ([], [])
        worker.take_failed_load_request_ids.return_value = set()
        assert manager.get_finished() == KvCacheConnectorPollResult()
        worker.abort_request.assert_called_once_with(42)
        assert request.state != LlmRequestState.CONTEXT_INIT
        # Rank 1's local discard failed, but every rank preserves coordination
        # and the canonical request until the next collective retry succeeds.
        assert manager._global_abort_intents[42] == "transfer_failed"
        assert manager._abort_requests[42] is request
        assert (42 in manager._local_data_cleaned) == (mpi_rank() == 0)
        assert request.state != LlmRequestState.CONTEXT_INIT

        assert manager.get_finished() == KvCacheConnectorPollResult(
            failed_loading=[request])
        assert request.state != LlmRequestState.CONTEXT_INIT
        assert manager.request_abort(42, "cancelled") is True
        assert manager.new_async_requests.loading == {}
        assert manager.pending_async_requests.loading == {}
        assert manager.local_finished_async_requests.loading == {}
        assert manager.finished_async_loading_requests == {}
        assert manager._abort_requests == {}
        assert manager._local_abort_intents == {}
        assert manager._global_abort_intents == {}
        assert manager._local_quiescent == set()
        assert manager._all_ranks_quiescent == set()
        assert manager._local_data_cleaned == set()
        assert manager._scheduler_cleanup_acks == set()
        assert manager._scheduler_success_acks == set()
        assert 42 not in manager.scheduler_output_manager.requests
        assert 42 not in manager.scheduler_output_manager.external_loads
        discard_calls = (
            manager.scheduler_output_manager.discard_request_id.call_args_list)
        expected_discard_calls = 1 if mpi_rank() == 0 else 2
        assert [call.args
                for call in discard_calls] == [(42, )] * expected_discard_calls
        assert manager._scheduler_output.new_requests == []
        assert manager._scheduler_output.cached_requests == [unrelated_output]
        cleanup_calls = worker.abort_request.call_count
        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert worker.abort_request.call_count == cleanup_calls
        assert request.state != LlmRequestState.CONTEXT_INIT
        if scheduler is not None:
            scheduler.finish_load.assert_not_called()
            scheduler.abort_request.assert_called_once_with(
                42, "transfer_failed")

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("finish_outcomes",
                         [[True], [False, True], [RuntimeError("retry"), True]])
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_success_waits_for_all_ranks_and_scheduler_ack(
        mpi_pool_executor, finish_outcomes):

    def test():
        worker = MagicMock()
        worker.take_failed_load_request_ids.return_value = set()
        worker.get_finished.return_value = (([], [42]) if mpi_rank() == 0 else
                                            ([], []))
        scheduler = MagicMock() if mpi_rank() == 0 else None
        if scheduler is not None:
            scheduler.finish_load.side_effect = finish_outcomes
        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        manager.new_async_requests.loading[42] = request

        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert request.state != LlmRequestState.CONTEXT_INIT
        if scheduler is not None:
            scheduler.finish_load.assert_not_called()

        worker.get_finished.return_value = (([], []) if mpi_rank() == 0 else
                                            ([], [42]))
        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert request.state != LlmRequestState.CONTEXT_INIT

        worker.get_finished.return_value = ([], [])
        expected_calls = len(finish_outcomes)
        while scheduler is not None and scheduler.finish_load.call_count < expected_calls:
            assert manager.get_finished() == KvCacheConnectorPollResult()
            assert request.state != LlmRequestState.CONTEXT_INIT
        # Non-leaders execute the same number of polls as rank 0.
        for _ in range(expected_calls - 1):
            if scheduler is None:
                assert manager.get_finished() == KvCacheConnectorPollResult()

        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert request.state == LlmRequestState.CONTEXT_INIT
        if scheduler is not None:
            assert scheduler.finish_load.call_count == expected_calls

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_cancellation_after_success_callback_never_promotes(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.get_finished.return_value = ([], [42])
        worker.take_failed_load_request_ids.return_value = set()
        worker.abort_request.return_value = True
        scheduler = MagicMock() if mpi_rank() == 0 else None
        if scheduler is not None:
            scheduler.finish_load.return_value = True
            scheduler.abort_request.return_value = True
        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        manager.new_async_requests.loading[42] = request

        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert request.state != LlmRequestState.CONTEXT_INIT
        if scheduler is not None:
            scheduler.finish_load.assert_called_once_with(42)
            assert manager.request_abort(42, "cancelled") is False

        worker.get_finished.return_value = ([], [])
        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert request.state != LlmRequestState.CONTEXT_INIT
        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert request.state != LlmRequestState.CONTEXT_INIT
        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert request.state != LlmRequestState.CONTEXT_INIT
        assert manager.request_abort(42, "cancelled") is True
        if scheduler is not None:
            scheduler.finish_load.assert_called_once_with(42)
            scheduler.abort_request.assert_called_once_with(42, "cancelled")

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_failure_waits_for_all_worker_releases(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.get_finished.return_value = ([], [])
        worker.take_failed_load_request_ids.return_value = ({42} if mpi_rank()
                                                            == 1 else set())
        worker.abort_request.return_value = True
        scheduler = MagicMock() if mpi_rank() == 0 else None
        if scheduler is not None:
            scheduler.abort_request.return_value = True
        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        manager.new_async_requests.loading[42] = request

        assert manager.get_finished() == KvCacheConnectorPollResult()
        worker.take_failed_load_request_ids.return_value = set()
        assert manager.get_finished() == KvCacheConnectorPollResult()
        if scheduler is not None:
            scheduler.abort_request.assert_called_once_with(
                42, "transfer_failed")
        assert manager.get_finished() == KvCacheConnectorPollResult(
            failed_loading=[request])
        if scheduler is not None:
            scheduler.abort_request.assert_called_once_with(
                42, "transfer_failed")

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_remote_g2_tp_success_releases_once_after_all_ranks_finish(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.take_failed_load_request_ids.return_value = set()
        worker.get_finished.return_value = (([], [42]) if mpi_rank() == 0 else
                                            ([], []))
        release_store = _RankZeroReleaseStore([True])
        scheduler = None
        if mpi_rank() == 0:
            scheduler = RemoteG2KvCacheConnectorScheduler(
                None,
                binding_store=release_store,
                plan_store=MagicMock())
        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        manager.new_async_requests.loading[42] = request

        assert manager.get_finished() == KvCacheConnectorPollResult()
        if scheduler is not None:
            assert release_store.calls == []
        worker.get_finished.return_value = (([], []) if mpi_rank() == 0 else
                                            ([], [42]))
        assert manager.get_finished() == KvCacheConnectorPollResult()
        if scheduler is not None:
            assert release_store.calls == [(42, "transfer_succeeded")]
        assert request.state != LlmRequestState.CONTEXT_INIT

        worker.get_finished.return_value = ([], [])
        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert request.state == LlmRequestState.CONTEXT_INIT
        if scheduler is not None:
            assert release_store.calls == [(42, "transfer_succeeded")]

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_remote_g2_tp_failure_waits_for_peer_release_before_rank0_release(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.get_finished.return_value = ([], [])
        worker.take_failed_load_request_ids.return_value = ({42} if mpi_rank()
                                                            == 1 else set())
        worker.abort_request.return_value = True
        release_store = _RankZeroReleaseStore([True])
        scheduler = None
        if mpi_rank() == 0:
            scheduler = RemoteG2KvCacheConnectorScheduler(
                None,
                binding_store=release_store,
                plan_store=MagicMock())
        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        manager.new_async_requests.loading[42] = request

        assert manager.get_finished() == KvCacheConnectorPollResult()
        worker.take_failed_load_request_ids.return_value = set()
        assert manager.get_finished() == KvCacheConnectorPollResult()
        if scheduler is not None:
            assert release_store.calls == [(42, "transfer_failed")]
        assert manager.get_finished() == KvCacheConnectorPollResult(
            failed_loading=[request])
        if scheduler is not None:
            assert release_store.calls == [(42, "transfer_failed")]

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_remote_g2_tp_rank0_release_exception_retries_before_finalization(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.get_finished.return_value = ([], [])
        worker.take_failed_load_request_ids.return_value = ({42} if mpi_rank()
                                                            == 1 else set())
        worker.abort_request.return_value = True
        release_store = _RankZeroReleaseStore(
            [RuntimeError("release timeout"), True])
        scheduler = None
        if mpi_rank() == 0:
            scheduler = RemoteG2KvCacheConnectorScheduler(
                None,
                binding_store=release_store,
                plan_store=MagicMock())
        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        manager.new_async_requests.loading[42] = request

        assert manager.get_finished() == KvCacheConnectorPollResult()
        worker.take_failed_load_request_ids.return_value = set()
        assert manager.get_finished() == KvCacheConnectorPollResult()
        result = KvCacheConnectorPollResult()
        for _ in range(3):
            result = manager.get_finished()
            if result.failed_loading:
                break
        assert result == KvCacheConnectorPollResult(failed_loading=[request])
        if scheduler is not None:
            assert release_store.calls == [
                (42, "transfer_failed"),
                (42, "transfer_failed"),
            ]

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("release_outcomes", [[True], [RuntimeError("timeout"), True]])
@pytest.mark.threadleak(enabled=False)
def test_remote_g2_actual_worker_and_store_hold_lease_until_tp_failure_consensus(
        mpi_pool_executor, release_outcomes):

    def test():
        record = _remote_g2_mpi_record()
        if mpi_rank() == 0:
            result = _RemoteG2MpiResult(
                record,
                RemoteG2TransferState.IN_PROGRESS,
            )
        else:
            result = _RemoteG2MpiResult(
                record,
                RemoteG2TransferState.FAILED,
            )
        worker = RemoteG2KvCacheConnectorWorker(
            None,
            transfer_adapter=_RemoteG2MpiAdapter(result),
            mark_local_valid=lambda record: None,
            publish_binding=lambda record: None,
        )
        worker.bind_connector_meta(RemoteG2ConnectorMetadata((record, )))
        worker.start_load_kv(None)

        scheduler_release_calls = []
        scheduler = None
        if mpi_rank() == 0:
            outcomes = list(release_outcomes)

            def release(lease, reason):
                scheduler_release_calls.append((lease, reason))
                outcome = outcomes.pop(0)
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

            binding_store = TargetRemoteG2BindingStore(release_lease=release)
            binding_store._records[42] = record
            scheduler = RemoteG2KvCacheConnectorScheduler(
                None,
                binding_store=binding_store,
                plan_store=MagicMock(),
            )

        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        manager.new_async_requests.loading[42] = request

        assert manager.get_finished() == KvCacheConnectorPollResult()
        if scheduler is not None:
            assert scheduler_release_calls == []

        final = KvCacheConnectorPollResult()
        for _ in range(6):
            final = manager.get_finished()
            if final.failed_loading:
                break
        assert final == KvCacheConnectorPollResult(failed_loading=[request])
        if scheduler is not None:
            assert scheduler_release_calls == [
                ("lease-tp", "transfer_failed")
            ] * len(release_outcomes)
            assert sum(
                not isinstance(outcome, BaseException)
                for outcome in release_outcomes
            ) == 1

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_remote_g2_actual_worker_success_releases_only_after_tp_consensus(
        mpi_pool_executor):

    def test():
        record = _remote_g2_mpi_record()
        result = _RemoteG2MpiResult(
            record,
            (RemoteG2TransferState.SUCCEEDED if mpi_rank() == 0 else
             RemoteG2TransferState.IN_PROGRESS),
            poll_states=(
                () if mpi_rank() == 0 else (
                    RemoteG2TransferState.IN_PROGRESS,
                    RemoteG2TransferState.SUCCEEDED,
                )),
        )
        worker = RemoteG2KvCacheConnectorWorker(
            None,
            transfer_adapter=_RemoteG2MpiAdapter(result),
            mark_local_valid=lambda record: None,
            publish_binding=lambda record: None,
        )
        worker.bind_connector_meta(RemoteG2ConnectorMetadata((record, )))
        worker.start_load_kv(None)

        scheduler_release_calls = []
        scheduler = None
        if mpi_rank() == 0:
            binding_store = TargetRemoteG2BindingStore(
                release_lease=lambda lease, reason: scheduler_release_calls.append(
                    (lease, reason)
                )
                or True)
            binding_store._records[42] = record
            scheduler = RemoteG2KvCacheConnectorScheduler(
                None,
                binding_store=binding_store,
                plan_store=MagicMock(),
            )

        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
        manager.new_async_requests.loading[42] = request

        assert manager.get_finished() == KvCacheConnectorPollResult()
        if scheduler is not None:
            assert scheduler_release_calls == []
        assert manager.get_finished() == KvCacheConnectorPollResult()
        if scheduler is not None:
            assert scheduler_release_calls == [("lease-tp", "transfer_succeeded")]
        assert request.state != LlmRequestState.CONTEXT_INIT
        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert request.state == LlmRequestState.CONTEXT_INIT

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_cancellation_waits_for_every_rank_release(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.get_finished.return_value = ([], [])
        worker.take_failed_load_request_ids.return_value = set()
        worker.abort_request.return_value = True
        scheduler = MagicMock() if mpi_rank() == 0 else None
        if scheduler is not None:
            scheduler.abort_request.return_value = True
        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        manager.new_async_requests.loading[42] = request

        allgather_calls = 0
        original_allgather = kv_cache_connector.mpi_allgather

        def counted_allgather(payload):
            nonlocal allgather_calls
            allgather_calls += 1
            return original_allgather(payload)

        kv_cache_connector.mpi_allgather = counted_allgather
        try:
            if mpi_rank() == 0:
                assert manager.request_abort(42, "cancelled") is False
            else:
                assert manager._local_abort_intents == {}
            for poll in range(3):
                assert manager.get_finished() == KvCacheConnectorPollResult()
                assert allgather_calls == poll + 1
                if poll == 0:
                    assert manager._global_abort_intents[42] == "cancelled"
                    if mpi_rank() == 1:
                        assert manager._local_abort_intents[42] == "cancelled"
        finally:
            kv_cache_connector.mpi_allgather = original_allgather

        assert manager.request_abort(42, "cancelled") is True
        assert worker.abort_request.call_count == 1
        if scheduler is not None:
            scheduler.abort_request.assert_called_once_with(42, "cancelled")

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("scheduler_outcomes",
                         [[False, True], [RuntimeError("retry"), True]])
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_retries_scheduler_cleanup(mpi_pool_executor,
                                                     scheduler_outcomes):

    def test():
        worker = MagicMock()
        worker.get_finished.return_value = ([], [])
        worker.take_failed_load_request_ids.return_value = ({42} if mpi_rank()
                                                            == 0 else set())
        worker.abort_request.return_value = True
        scheduler = MagicMock() if mpi_rank() == 0 else None
        if scheduler is not None:
            scheduler.abort_request.side_effect = scheduler_outcomes
        manager = KvCacheConnectorManager(worker, scheduler)
        request = MagicMock(request_id=42)
        manager.new_async_requests.loading[42] = request

        assert manager.get_finished() == KvCacheConnectorPollResult()
        worker.take_failed_load_request_ids.return_value = set()
        assert manager.get_finished() == KvCacheConnectorPollResult()
        assert manager.get_finished() == KvCacheConnectorPollResult()
        result = manager.get_finished()
        assert result == KvCacheConnectorPollResult(failed_loading=[request])
        if scheduler is not None:
            assert scheduler.abort_request.call_count == 2

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("failing_hook",
                         ["get_finished", "take_failed_load_request_ids"])
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_worker_hook_failure_is_same_on_all_ranks(
        mpi_pool_executor, failing_hook):

    def test():
        worker = MagicMock()
        worker.get_finished.return_value = ([], [])
        worker.take_failed_load_request_ids.return_value = set()
        getattr(
            worker,
            failing_hook).side_effect = ValueError(f"rank {mpi_rank()} secret")
        scheduler = MagicMock() if mpi_rank() == 0 else None
        manager = KvCacheConnectorManager(worker, scheduler)

        with pytest.raises(
                RuntimeError,
                match=("KV connector worker hook failed: ValueError: "
                       "rank 0 secret")):
            manager.get_finished()

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_abort_release_error_is_same_on_all_ranks(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.get_finished.return_value = ([], [])
        worker.take_failed_load_request_ids.return_value = set()
        worker.abort_request.side_effect = RuntimeError(
            f"rank {mpi_rank()} release failed")
        scheduler = MagicMock() if mpi_rank() == 0 else None
        manager = KvCacheConnectorManager(worker, scheduler)
        manager.new_async_requests.loading[42] = MagicMock(request_id=42)
        assert manager.request_abort(42, "cancelled") is False

        assert manager.get_finished() == KvCacheConnectorPollResult()
        with pytest.raises(
                RuntimeError,
                match=("KV connector worker hook failed: RuntimeError: "
                       "rank 0 release failed")):
            manager.get_finished()

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_num_matched_tokens(mpi_pool_executor):

    def test():
        worker = MagicMock()

        if mpi_rank() == 0:
            scheduler = MagicMock()
            scheduler.get_num_new_matched_tokens.return_value = (16, True)
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock()

        req.request_id = 42
        req.is_generation_only_request = False

        assert manager.get_num_new_matched_tokens(req, 32) == 16

        if mpi_rank() == 0:
            assert scheduler.get_num_new_matched_tokens.call_count == 1
            assert scheduler.get_num_new_matched_tokens.call_args[0] == (req,
                                                                         32)

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_take_scheduled_requests(mpi_pool_executor):

    def test():
        worker = MagicMock()

        if mpi_rank() == 0:
            scheduler = MagicMock()
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        scheduled_requests = ScheduledRequests()

        req0 = MagicMock()
        req0.request_id = 0
        req0.is_generation_only_request = False

        req1 = MagicMock()
        req1.request_id = 1
        req1.is_generation_only_request = False

        if mpi_rank() == 0:
            scheduler.get_num_new_matched_tokens.return_value = (16, True)

        assert manager.get_num_new_matched_tokens(req0, 0) == 16
        if mpi_rank() == 0:
            assert scheduler.get_num_new_matched_tokens.call_count == 1
            assert scheduler.get_num_new_matched_tokens.call_args[0] == (req0,
                                                                         0)

            scheduler.get_num_new_matched_tokens.reset_mock()
            scheduler.get_num_new_matched_tokens.return_value = (32, False)

        assert manager.get_num_new_matched_tokens(req1, 0) == 32
        if mpi_rank() == 0:
            assert scheduler.get_num_new_matched_tokens.call_count == 1
            assert scheduler.get_num_new_matched_tokens.call_args[0] == (req1,
                                                                         0)

        scheduled_requests.context_requests_last_chunk = [req0, req1]

        manager.take_scheduled_requests_pending_load(scheduled_requests)

        assert scheduled_requests.context_requests_last_chunk == [req1]

    run_across_mpi(mpi_pool_executor, test, 2)


def test_scheduler_output_num_scheduled_tokens_with_mtp():
    """Test that num_scheduled_tokens is correctly set for MTP (multi-token prediction)."""
    NUM_DRAFT_TOKENS = 3

    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []

    # Create a mock request in generation state with draft tokens
    req = MagicMock()
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.get_tokens.return_value = [1, 2, 3, 4, 5]  # 5 tokens already generated
    req.py_draft_tokens = [100, 101, 102]  # 3 MTP draft tokens

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = KvCacheConnectorSchedulerOutputManager()
    scheduler_output = manager.build_scheduler_output(scheduled_batch,
                                                      AsyncRequests({}, {}),
                                                      kv_cache_manager)

    assert len(scheduler_output.cached_requests) == 1
    request_data = scheduler_output.cached_requests[0]

    # For generation requests: num_scheduled_tokens = 1 + draft_token_length
    expected_num_scheduled_tokens = 1 + NUM_DRAFT_TOKENS
    assert request_data.num_scheduled_tokens == expected_num_scheduled_tokens, \
        f"Expected {expected_num_scheduled_tokens}, got {request_data.num_scheduled_tokens}"


def test_scheduler_output_block_hashes_read_through():
    """``RequestData.block_hashes`` reflects the chain returned by the KV cache manager.

    The connector path does not recompute hashes Python-side; each scheduler step
    is a pure pass-through of whatever ``commit_and_get_block_hashes`` returns.
    A subsequent step that observes a longer chain simply forwards the longer
    chain. The block-completion semantics (when the next hash actually appears)
    are owned by the C++ KV cache manager and exercised by the C++ unit tests
    for ``commitAndGetBlockHashesForRequest``.
    """
    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0]
    # Two consecutive scheduler steps: first sees no full block yet, second sees
    # one full block whose hash has just been committed by the manager.
    kv_cache_manager.commit_and_get_block_hashes.side_effect = [[], [12345]]

    req = MagicMock()
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.py_draft_tokens = []
    req.get_tokens.return_value = [1, 2, 3]

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = KvCacheConnectorSchedulerOutputManager()

    output = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)
    assert output.cached_requests[0].block_hashes == []

    req.get_tokens.return_value = [1, 2, 3, 4]
    output = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)
    assert output.cached_requests[0].block_hashes == [12345]

    # Each scheduler step asks the manager exactly once per request; no Python
    # caching layer reshapes the request between calls.
    assert kv_cache_manager.commit_and_get_block_hashes.call_count == 2
    for call in kv_cache_manager.commit_and_get_block_hashes.call_args_list:
        assert call.args == (req, )
