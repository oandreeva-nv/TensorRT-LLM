# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = "tensorrt_llm._torch.pyexecutor.connectors"


def _install_package(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


for package in (
    "tensorrt_llm",
    "tensorrt_llm._torch",
    "tensorrt_llm._torch.pyexecutor",
    _PACKAGE,
    "tensorrt_llm.bindings",
    "tensorrt_llm.bindings.internal",
    "tensorrt_llm.bindings.internal.batch_manager",
    "tensorrt_llm.llmapi",
):
    _install_package(package)

torch = types.ModuleType("torch")
torch.Tensor = object
torch.cuda = SimpleNamespace(Stream=object, current_stream=lambda: None)
sys.modules["torch"] = torch

utils = types.ModuleType("tensorrt_llm._utils")
utils.mpi_allgather = lambda value: [value]
utils.mpi_broadcast = lambda value, root=0: value
utils.mpi_rank = lambda: 0
sys.modules[utils.__name__] = utils

bindings = sys.modules["tensorrt_llm.bindings"]
bindings.LlmRequestState = SimpleNamespace(
    CONTEXT_INIT="context_init",
    DISAGG_CONTEXT_TRANS_IN_PROGRESS="context_transfer",
    DISAGG_GENERATION_TRANS_IN_PROGRESS="generation_transfer",
)

batch_manager = sys.modules["tensorrt_llm.bindings.internal.batch_manager"]
batch_manager.KvCacheConnectorManager = object
batch_manager.LlmRequest = object

llm_args = types.ModuleType("tensorrt_llm.llmapi.llm_args")
llm_args.TorchLlmArgs = object
sys.modules[llm_args.__name__] = llm_args

llm_request = types.ModuleType("tensorrt_llm._torch.pyexecutor.llm_request")
llm_request.get_draft_token_length = lambda request: 0
sys.modules[llm_request.__name__] = llm_request

scheduler = types.ModuleType("tensorrt_llm._torch.pyexecutor.scheduler")
scheduler.ScheduledRequests = object
sys.modules[scheduler.__name__] = scheduler

_MODULE_NAME = f"{_PACKAGE}.kv_cache_connector"
_PATH = _ROOT / "tensorrt_llm/_torch/pyexecutor/connectors/kv_cache_connector.py"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _PATH)
assert _SPEC is not None and _SPEC.loader is not None
CONNECTOR = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = CONNECTOR
_SPEC.loader.exec_module(CONNECTOR)


def test_request_coordination_failure_wins_and_abort_replaces_success_ack():
    request = SimpleNamespace(request_id=42)
    state = CONNECTOR._RequestCoordinationState(request)

    state.mark_scheduler_acked(CONNECTOR._SchedulerAck.SUCCESS)
    state.record_local_abort("cancelled")
    state.observe_consensus_abort("transfer_failed")

    assert state.mode is CONNECTOR._CoordinationMode.ABORT
    assert state.local_abort_reason == "transfer_failed"
    assert state.consensus_abort_reason == "transfer_failed"
    assert state.scheduler_ack is None


def test_request_coordination_enforces_abort_cleanup_order():
    state = CONNECTOR._RequestCoordinationState(SimpleNamespace(request_id=42))
    state.observe_consensus_abort("cancelled")

    with pytest.raises(RuntimeError, match="transfer release"):
        state.mark_local_bookkeeping_cleaned()
    with pytest.raises(RuntimeError, match="all worker releases"):
        state.mark_scheduler_acked(CONNECTOR._SchedulerAck.ABORT)

    state.mark_local_transfer_released()
    state.mark_local_bookkeeping_cleaned()
    state.mark_all_worker_releases_complete()
    state.mark_scheduler_acked(CONNECTOR._SchedulerAck.ABORT)

    assert state.local_transfer_released is True
    assert state.local_bookkeeping_cleaned is True
    assert state.all_worker_releases_complete is True
    assert state.scheduler_ack is CONNECTOR._SchedulerAck.ABORT


def test_connector_poll_payload_is_picklable_and_request_id_only():
    payload = CONNECTOR._ConnectorPollPayload(
        finished_save_request_ids=(1,),
        finished_load_request_ids=(2,),
        abort_intents=((3, "transfer_failed"),),
        locally_released_request_ids=frozenset({3}),
        locally_cleaned_request_ids=frozenset({3}),
        scheduler_success_ack_request_ids=frozenset({2}),
        scheduler_abort_ack_request_ids=frozenset({3}),
    )

    current_module = sys.modules.get(_MODULE_NAME)
    try:
        sys.modules[_MODULE_NAME] = CONNECTOR
        assert pickle.loads(pickle.dumps(payload)) == payload
    finally:
        if current_module is None:
            sys.modules.pop(_MODULE_NAME, None)
        else:
            sys.modules[_MODULE_NAME] = current_module
    assert payload.finished_save_request_ids == (1,)
    assert payload.abort_intents == ((3, "transfer_failed"),)


class _Worker:
    def __init__(self):
        self.finished = ([], [])
        self.failed = set()
        self.aborted = []

    def get_finished(self, finished_gen_request_ids, started_load_request_ids):
        return self.finished

    def take_failed_load_request_ids(self):
        failed = self.failed
        self.failed = set()
        return failed

    def try_abort_request(self, request_id):
        self.aborted.append(request_id)
        return True


class _Scheduler:
    def __init__(self):
        self.successful = []
        self.aborted = []

    def finalize_successful_load(self, request_id):
        self.successful.append(request_id)
        return True

    def try_abort_request(self, request_id, reason):
        self.aborted.append((request_id, reason))
        return True


def test_connector_manager_success_uses_request_coordination_state():
    worker = _Worker()
    scheduler = _Scheduler()
    manager = CONNECTOR.KvCacheConnectorManager(worker, scheduler)
    request = SimpleNamespace(request_id=42, state="loading")
    manager.new_async_requests.loading[42] = request
    worker.finished = ([], [42])

    assert manager.get_finished() == CONNECTOR.KvCacheConnectorPollResult()
    assert manager._request_coordination[42].scheduler_ack is CONNECTOR._SchedulerAck.SUCCESS

    worker.finished = ([], [])
    assert manager.get_finished() == CONNECTOR.KvCacheConnectorPollResult()
    assert request.state == bindings.LlmRequestState.CONTEXT_INIT
    assert manager._request_coordination == {}
    assert scheduler.successful == [42]


def test_connector_manager_failure_releases_then_finalizes_request():
    worker = _Worker()
    scheduler = _Scheduler()
    manager = CONNECTOR.KvCacheConnectorManager(worker, scheduler)
    request = SimpleNamespace(request_id=42, state="loading")
    manager.new_async_requests.loading[42] = request
    worker.failed = {42}

    assert manager.get_finished() == CONNECTOR.KvCacheConnectorPollResult()
    assert manager._request_coordination[42].consensus_abort_reason == "transfer_failed"

    assert manager.get_finished() == CONNECTOR.KvCacheConnectorPollResult()
    state = manager._request_coordination[42]
    assert state.local_transfer_released is True
    assert state.all_worker_releases_complete is True
    assert state.local_bookkeeping_cleaned is True
    assert state.scheduler_ack is CONNECTOR._SchedulerAck.ABORT

    assert manager.get_finished() == CONNECTOR.KvCacheConnectorPollResult(
        failed_load_requests=[request]
    )
    assert manager._request_coordination == {}
    assert worker.aborted == [42]
    assert scheduler.aborted == [(42, "transfer_failed")]
