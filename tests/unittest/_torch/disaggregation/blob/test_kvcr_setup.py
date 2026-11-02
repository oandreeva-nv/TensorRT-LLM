# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone provider setup without runtime assembly or SDK registration."""

import pytest

from tensorrt_llm._torch.disaggregation.blob.kvcr_setup import open_kvcr_driver

pytestmark = pytest.mark.cpu_only


def _config():
    return {"cache_slots": 8, "endpoint": "setup-test", "listen_port": 19341}


def test_mapping_opens_driver_without_registering_sdk():
    driver = open_kvcr_driver(_config())
    assert driver.capabilities.get and driver.capabilities.put
    assert not driver.capabilities.hints
    assert driver.drain(0)
    driver.close()


def test_yaml_opens_peer_enabled_driver(tmp_path):
    config = tmp_path / "kvcr.yaml"
    config.write_text(
        "cache_slots: 8\nendpoint: setup-test\nlisten_port: 19341\ncontrol_port: 19342\n"
    )
    driver = open_kvcr_driver(config)
    assert driver.capabilities.hints
    assert driver.drain(0)
    driver.close()


@pytest.mark.parametrize("config", [None, [], 42, {"unknown": 1}, {}])
def test_invalid_config_fails_before_registration(config):
    with pytest.raises((TypeError, ValueError)):
        open_kvcr_driver(config)


def test_yaml_requires_a_mapping(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("- not-a-mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        open_kvcr_driver(config)
