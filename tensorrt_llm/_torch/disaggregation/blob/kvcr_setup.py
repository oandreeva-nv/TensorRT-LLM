# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone KVCR driver setup; registration and backend assembly belong to the caller."""

from collections.abc import Mapping
from pathlib import Path

import yaml

from .kvcr_driver import KvcrDriver

__all__ = ["open_kvcr_driver"]


def open_kvcr_driver(config: Mapping[str, object] | str | Path) -> KvcrDriver:
    """Open an unregistered driver from its option mapping or a YAML file.

    Keys match ``KvcrDriver`` constructor arguments. This imports no KVCR SDK,
    allocates no cache, and opens no listener until region registration. The
    caller closes the driver after backend shutdown and region-hold release.
    """
    if isinstance(config, (str, Path)):
        with Path(config).open() as stream:
            config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError("KVCR driver config must be a mapping")
    return KvcrDriver(**config)
