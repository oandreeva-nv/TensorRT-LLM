# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from enum import Enum
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


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package in (
    "tensorrt_llm",
    "tensorrt_llm._torch",
    "tensorrt_llm._torch.pyexecutor",
    _PACKAGE,
):
    _install_package(package)

_load_module(
    f"{_PACKAGE}.remote_g2_observability",
    _ROOT / "tensorrt_llm/_torch/pyexecutor/connectors/remote_g2_observability.py",
)
REMOTE_G2 = _load_module(
    f"{_PACKAGE}.remote_g2",
    _ROOT / "tensorrt_llm/_torch/pyexecutor/connectors/remote_g2.py",
)
TRANSFER = _load_module(
    f"{_PACKAGE}.remote_g2_transfer",
    _ROOT / "tensorrt_llm/_torch/pyexecutor/connectors/remote_g2_transfer.py",
)
RAW = _load_module(
    f"{_PACKAGE}.remote_g2_raw_nixl_adapter",
    _ROOT / "tensorrt_llm/_torch/pyexecutor/connectors/remote_g2_raw_nixl_adapter.py",
)


class _NixlState(Enum):
    DONE = 1
    PROC = 2
    ERR = 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DONE", "SUCCEEDED"),
        ("SUCCESS", "SUCCEEDED"),
        ("PROC", "IN_PROGRESS"),
        ("PROCESSING", "IN_PROGRESS"),
        ("PENDING", "IN_PROGRESS"),
        ("ERR", "FAILED"),
        ("unexpected", "FAILED"),
        (None, "FAILED"),
        (_NixlState.DONE, "SUCCEEDED"),
        (_NixlState.PROC, "IN_PROGRESS"),
        (_NixlState.ERR, "FAILED"),
    ],
)
def test_classify_nixl_transfer_state(raw, expected):
    assert RAW._classify_nixl_transfer_state(raw).name == expected


class _FakeAgent:
    def __init__(self, states=(), release_results=()):
        self.states = iter(states)
        self.release_results = iter(release_results)
        self.release_calls = 0

    def check_xfer_state(self, handle):
        return next(self.states)

    def release_xfer_handle(self, handle):
        self.release_calls += 1
        result = next(self.release_results, None)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DONE", "SUCCEEDED"),
        ("PROC", "IN_PROGRESS"),
        ("ERR", "FAILED"),
    ],
)
def test_raw_result_retains_initial_done_proc_and_err(raw, expected):
    result = RAW._RawNixlTransferResult(
        agent=_FakeAgent(),
        handle=object(),
        record=SimpleNamespace(request_id=17),
        initial_state=RAW._classify_nixl_transfer_state(raw),
    )

    assert result.initial_state.name == expected


def test_raw_result_poll_err_fails_immediately():
    result = RAW._RawNixlTransferResult(
        agent=_FakeAgent(states=["ERR"]),
        handle=object(),
        record=SimpleNamespace(request_id=17),
        initial_state=RAW.RemoteG2TransferState.IN_PROGRESS,
    )

    assert result.poll_state() is RAW.RemoteG2TransferState.FAILED


def test_raw_result_quiesce_failure_is_fatal():
    agent = _FakeAgent(release_results=[RuntimeError("release failed")])
    result = RAW._RawNixlTransferResult(
        agent=agent,
        handle=object(),
        record=SimpleNamespace(request_id=17),
        initial_state=RAW.RemoteG2TransferState.FAILED,
    )

    with pytest.raises(RuntimeError, match="release failed"):
        result.quiesce()
    assert agent.release_calls == 1


def test_raw_result_quiesce_success_is_idempotent():
    agent = _FakeAgent(release_results=[None])
    result = RAW._RawNixlTransferResult(
        agent=agent,
        handle=object(),
        record=SimpleNamespace(request_id=17),
        initial_state=RAW.RemoteG2TransferState.SUCCEEDED,
    )

    assert result.quiesce() is True
    assert result.quiesce() is True
    assert agent.release_calls == 1


class _SubmitAgent(_FakeAgent):
    def __init__(self):
        super().__init__(release_results=[None])
        self.handle = object()

    def make_prepped_xfer(self, *args):
        return self.handle

    def transfer(self, handle):
        assert handle is self.handle
        raise RuntimeError("post failed")


def _record():
    descriptor = SimpleNamespace(byte_length=4096, byte_offset=0)
    block = SimpleNamespace(
        source_descriptor=descriptor,
        target_slot_idx=3,
        target_block_id=99,
    )
    return SimpleNamespace(
        request_id=17,
        is_transfer_ready=True,
        source_generation=1,
        plan=SimpleNamespace(source_worker_id=7, block_size_tokens=16),
        bound_blocks=(block,),
    )


def test_raw_adapter_transfer_raise_after_handle_creation_retains_retryable_handle():
    adapter = object.__new__(RAW.RawNixlRemoteG2Adapter)
    adapter._agent = _SubmitAgent()
    adapter._source_metadata_fetcher = lambda worker_id, generation: {}
    adapter._ensure_peer_loaded = lambda source_meta: (object(), object())

    result = adapter._start_transfer_impl(_record(), None)

    assert result.handle is adapter._agent.handle
    assert result.initial_state is RAW.RemoteG2TransferState.FAILED
    assert isinstance(result.start_error, RuntimeError)
    assert result.quiesce() is True


def test_unsafe_result_contract_is_rejected_without_losing_live_result():
    unsafe = SimpleNamespace(initial_state=RAW.RemoteG2TransferState.IN_PROGRESS)

    with pytest.raises(TRANSFER.RemoteG2TransferContractError) as exc_info:
        TRANSFER.validate_remote_g2_transfer_result(unsafe)

    assert exc_info.value.transfer_result is unsafe


def test_adapter_capabilities_are_explicit():
    assert TRANSFER.RemoteG2NixlTransferAdapter.supports_synchronous_release is False
    assert RAW.RawNixlRemoteG2Adapter.supports_synchronous_release is True
