# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory quarantine driven by KVCR's existing resilience events."""

import pytest

from tensorrt_llm._torch.disaggregation.blob.kvcr_lifecycle import _KvcrAccessTracker
from tensorrt_llm._torch.disaggregation.blob.store import Span

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def sdk():
    return pytest.importorskip("kvcr.types")


def test_quiescence_clears_only_its_own_access():
    tracker = _KvcrAccessTracker(lambda ref: Span(100 + ref.element_index * 16, 16))
    buffer = Span(100, 16)
    tracker.uncertain((1, "destination"), (buffer,))
    tracker.uncertain((2, "destination"), (buffer,))
    tracker.quiesced((1, "destination"), (buffer,))
    assert tracker.blocks_reuse((buffer,))
    tracker.quiesced((2, "destination"), (buffer,))
    assert not tracker.blocks_reuse((buffer,))


def test_overlapping_ranges_are_protected():
    tracker = _KvcrAccessTracker(lambda ref: Span(100, 16))
    tracker.uncertain((1, "source"), (Span(100, 16),))
    assert tracker.blocks_reuse((Span(108, 16),))
    assert not tracker.blocks_reuse((Span(116, 16),))


def test_duplicate_and_unknown_events_do_not_release_other_accesses():
    tracker = _KvcrAccessTracker(lambda ref: Span(100, 16))
    spans = (Span(100, 16),)
    tracker.uncertain((1, "source"), spans)
    tracker.uncertain((1, "source"), spans)
    tracker.quiesced((1, "destination"), spans)
    tracker.quiesced((999, "source"), spans)
    assert tracker.blocks_reuse(spans)
    tracker.quiesced((1, "source"), spans)
    tracker.quiesced((1, "source"), spans)
    assert not tracker.blocks_reuse(spans)


@pytest.mark.parametrize("transition", [_KvcrAccessTracker.uncertain, _KvcrAccessTracker.quiesced])
def test_changed_memory_for_existing_access_is_rejected(transition):
    tracker = _KvcrAccessTracker(lambda ref: Span(100, 16))
    tracker.uncertain((1, "source"), (Span(100, 16),))
    with pytest.raises(ValueError, match="memory changed"):
        transition(tracker, (1, "source"), (Span(200, 16),))
    assert tracker.blocks_reuse((Span(100, 16),))


@pytest.mark.parametrize("direction", ["source", "destination"])
def test_existing_sdk_callback_tracks_framework_memory(sdk, direction):
    tracker = _KvcrAccessTracker(lambda ref: Span(100 + ref.element_index * 16, 16))
    ref = sdk.MemoryRef(end_point_name="local", element_index=1)
    key = sdk.BlockKey(b"key")
    fields = (
        {"source_blocks": {key: [ref]}, "source_owners": {key: "framework"}}
        if direction == "source"
        else {"destination_regions": [ref], "destination_owner": "framework"}
    )
    tracker.on_resilience_event(sdk.TransferError("copy", -10, **fields))
    assert tracker.blocks_reuse((Span(116, 16),))
    assert not tracker.blocks_reuse((Span(100, 16),))
    tracker.on_resilience_event(sdk.TransferError("copy", -10, state="quiesced", **fields))
    assert not tracker.blocks_reuse((Span(116, 16),))


def test_kvcr_owned_memory_is_not_framework_quarantine(sdk):
    tracker = _KvcrAccessTracker(lambda ref: pytest.fail("must not resolve KVCR memory"))
    ref = sdk.MemoryRef(end_point_name="local", element_index=0)
    tracker.on_resilience_event(
        sdk.TransferError("copy", 1, destination_regions=[ref], destination_owner="kvcr")
    )
    assert not tracker.blocks_reuse((Span(100, 16),))


def test_non_memory_resilience_errors_are_propagated(sdk):
    tracker = _KvcrAccessTracker(lambda ref: Span(100, 16))
    error = RuntimeError("source stalled")
    with pytest.raises(RuntimeError, match="source stalled") as caught:
        tracker.on_resilience_event(error)
    assert caught.value is error


def test_unknown_memory_ownership_is_rejected(sdk):
    tracker = _KvcrAccessTracker(lambda ref: Span(100, 16))
    ref = sdk.MemoryRef(end_point_name="local", element_index=0)
    with pytest.raises(ValueError, match="ownership"):
        tracker.on_resilience_event(sdk.TransferError("copy", 1, destination_regions=[ref]))


def test_invalid_quiescence_event_cannot_partially_clear_protection(sdk):
    tracker = _KvcrAccessTracker(lambda ref: Span(100 + ref.element_index * 16, 16))
    source = sdk.MemoryRef(end_point_name="local", element_index=0)
    destination = sdk.MemoryRef(end_point_name="local", element_index=1)
    key = sdk.BlockKey(b"key")
    fields = {
        "source_blocks": {key: [source]},
        "source_owners": {key: "framework"},
        "destination_regions": [destination],
        "destination_owner": "framework",
    }
    tracker.on_resilience_event(sdk.TransferError("copy", 1, **fields))
    fields["destination_regions"] = [source]
    with pytest.raises(ValueError, match="memory changed"):
        tracker.on_resilience_event(sdk.TransferError("copy", 1, state="quiesced", **fields))
    assert tracker.blocks_reuse((Span(100, 16),))
    assert tracker.blocks_reuse((Span(116, 16),))


def test_aliases_are_resolved_to_physical_memory(sdk):
    tracker = _KvcrAccessTracker(lambda ref: Span(100, 16))
    first = sdk.MemoryRef(end_point_name="local", label="pool:a", element_index=0)
    alias = sdk.MemoryRef(end_point_name="local", label="pool:*", element_index=0)
    for handle, ref in [(1, first), (2, alias)]:
        tracker.on_resilience_event(
            sdk.TransferError(
                "copy", handle, destination_regions=[ref], destination_owner="framework"
            )
        )
    tracker.on_resilience_event(
        sdk.TransferError(
            "copy", 1, state="quiesced", destination_regions=[alias], destination_owner="framework"
        )
    )
    assert tracker.blocks_reuse((Span(100, 16),))


def test_incomplete_source_descriptors_are_rejected(sdk):
    tracker = _KvcrAccessTracker(lambda ref: Span(100, 16))
    ref = sdk.MemoryRef(end_point_name="local", element_index=0)
    first, second = sdk.BlockKey(b"a"), sdk.BlockKey(b"b")
    with pytest.raises(ValueError, match="source memory"):
        tracker.on_resilience_event(
            sdk.TransferError(
                "copy",
                1,
                source_blocks={first: [ref], second: []},
                source_owners={first: "framework", second: "framework"},
            )
        )
