# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pickle
import sys
import types
from pathlib import Path

import pytest

_PACKAGE = "tensorrt_llm._torch.pyexecutor.connectors"
_DIR = Path(__file__).resolve().parents[3] / "tensorrt_llm" / "_torch" / "pyexecutor" / "connectors"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package in ("tensorrt_llm", "tensorrt_llm._torch", "tensorrt_llm._torch.pyexecutor", _PACKAGE):
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module

_load(f"{_PACKAGE}.remote_g2_observability", _DIR / "remote_g2_observability.py")
_load(f"{_PACKAGE}.remote_g2", _DIR / "remote_g2.py")
sys.modules.setdefault(
    f"{_PACKAGE}.remote_g2_connector",
    types.ModuleType(f"{_PACKAGE}.remote_g2_connector"),
)
source_setup = types.ModuleType(f"{_PACKAGE}.remote_g2_source_setup")
source_setup._derive_block_size_bytes = lambda *args: 0
source_setup._derive_window_size = lambda *args: 0
source_setup._resolve_source_identity = lambda *args: 0
source_setup._walk_to_dynamo_worker_pid = lambda *args: 0
sys.modules[source_setup.__name__] = source_setup
TARGET_SETUP = _load(f"{_PACKAGE}.remote_g2_target_setup", _DIR / "remote_g2_target_setup.py")


class _Wrapper:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def request(self, method, payload):
        if self.error is not None:
            raise self.error
        return self.response


@pytest.mark.parametrize("value", [True, False])
def test_release_callable_preserves_boolean(value):
    release = TARGET_SETUP._make_release_callable(_Wrapper({"ok": True, "result": value}), 7)
    assert release("lease", "reason") is value


@pytest.mark.parametrize(
    "wrapper",
    [
        _Wrapper(error=RuntimeError("transport")),
        _Wrapper("not-a-dict"),
        _Wrapper({"ok": False, "error": "failed"}),
        _Wrapper({"ok": True}),
        _Wrapper({"ok": True, "result": 1}),
    ],
)
def test_release_callable_rejects_uncertain_response(wrapper):
    release = TARGET_SETUP._make_release_callable(wrapper, 7)
    with pytest.raises(RuntimeError):
        release("lease", "reason")


def test_missing_release_callable_raises():
    # The default lives in the connector module; load it through the focused
    # connector test module to avoid importing the full package.
    from test_remote_g2_connector import REMOTE_G2_CONNECTOR

    with pytest.raises(RuntimeError, match="not configured"):
        REMOTE_G2_CONNECTOR._missing_release_lease("lease", "reason")


def test_target_req_wrapper_recreates_req_socket_after_error(monkeypatch):
    closed = []

    class Socket:
        def __init__(self, response=None, error=None):
            self.response = response
            self.error = error
            self.RCVTIMEO = None
            self.SNDTIMEO = None
            self.connected = None

        def connect(self, endpoint):
            self.connected = endpoint

        def send(self, payload):
            self.payload = payload

        def recv(self):
            if self.error is not None:
                raise self.error
            return pickle.dumps(self.response)

        def close(self, linger=None):
            closed.append(linger)

    sockets = [
        Socket(error=RuntimeError("timeout")),
        Socket(response={"ok": True, "result": "recovered"}),
    ]

    class Context:
        def socket(self, kind):
            assert kind == "REQ"
            return sockets.pop(0)

    fake_zmq = types.SimpleNamespace(
        REQ="REQ", Context=types.SimpleNamespace(instance=lambda: Context())
    )
    monkeypatch.setitem(sys.modules, "zmq", fake_zmq)
    wrapper = TARGET_SETUP._TargetReqWrapper("/tmp/remote-g2", timeout_ms=17)

    with pytest.raises(RuntimeError, match="timeout"):
        wrapper.request("release", {})
    assert closed == [0]
    assert wrapper._socket.RCVTIMEO == 17
    assert wrapper._socket.SNDTIMEO == 17
    assert wrapper._socket.connected == "ipc:///tmp/remote-g2"
    assert wrapper.request("release", {}) == {"ok": True, "result": "recovered"}
