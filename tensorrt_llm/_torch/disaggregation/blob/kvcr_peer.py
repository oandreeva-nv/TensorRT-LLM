# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-specific peer selection for exact blob keys."""

from dataclasses import dataclass
from hashlib import blake2b
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    import msgspec


@dataclass(frozen=True)
class KvcrPeerHint:
    """Select a peer's published KVCR DRAM; this is not a serving envelope."""

    source_control_endpoint: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_control_endpoint, str):
            raise TypeError("KVCR peer source must be a string")
        address = urlsplit(self.source_control_endpoint)
        if (
            address.scheme != "tcp"
            or not address.hostname
            or not address.port
            or address.username
            or address.password
            or address.path
            or address.query
            or address.fragment
        ):
            raise ValueError("KVCR peer source must be a tcp://host:port endpoint")


class _BlobKeyAdapter:
    """Map blob keys to advisory 64-bit hint membership.

    Remote transfers still carry the complete key. The abbreviated hash only
    selects a candidate source; it never replaces the object identity.
    """

    def encode(self, framework_key: object) -> bytes:
        if not isinstance(framework_key, str):
            raise TypeError("blob keys must be strings")
        return framework_key.encode("utf-8")

    def decode(self, key: bytes) -> int:
        return int.from_bytes(blake2b(key, digest_size=8, person=b"trtllm-kvcr").digest(), "big")


def _fetch_envelope(
    hint: KvcrPeerHint,
    keys: tuple[bytes, ...],
    request_id: str,
) -> "msgspec.Struct":
    # Keep the optional wire dependency off the local driver's import path.
    import msgspec

    class Action(msgspec.Struct):
        action_id: str
        action_type: str
        action_version: str
        payload: dict[str, object]

    class Envelope(msgspec.Struct):
        protocol_version: str
        message_id: str
        actions: list[Action]

    payload = {
        "source_control_endpoint": hint.source_control_endpoint,
        "block_hashes": [_BlobKeyAdapter().decode(key) for key in keys],
    }
    return Envelope("0.1", request_id, [Action(request_id, "kv.fetch", "1.0", payload)])
