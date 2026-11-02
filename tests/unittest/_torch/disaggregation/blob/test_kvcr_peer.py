# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider hint validation without native KVCR or network initialization."""

from dataclasses import FrozenInstanceError

import pytest

from tensorrt_llm._torch.disaggregation.blob.kvcr_peer import KvcrPeerHint

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "source",
        "tcp://source",
        "tcp://source:0",
        "tcp://source:65536",
        "http://source:1234",
        "tcp://source:1234/path",
        "tcp://user@source:1234",
    ],
)
def test_rejects_invalid_peer_endpoint(endpoint):
    with pytest.raises(ValueError):
        KvcrPeerHint(endpoint)


def test_peer_selection_cannot_change_after_submission():
    hint = KvcrPeerHint("tcp://source:1234")
    with pytest.raises(FrozenInstanceError):
        hint.source_control_endpoint = "tcp://other-source:1234"
