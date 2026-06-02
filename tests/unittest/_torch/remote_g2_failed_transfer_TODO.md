# Remote-G2 failed-transfer tests — TODO

Co-located with `test_remote_g2_connector.py`. Lists test gaps we
identified during the transfer-initiation audit and chose not to add in
the first pass. Pick these up when extending failure-mode coverage.

---

## Gap A — `transfer_adapter is None` (misconfiguration path)

`start_load_kv` has two distinct failure code paths today:

| Path | Adapter state | Raise message | Fallback reason |
|---|---|---|---|
| Adapter raised | Adapter exists, `start_transfer` raised | `"remote G2 transfer failed to start"` | `"transfer_start_failed"` |
| Adapter missing | `self._transfer_adapter is None` | `"remote G2 transfer adapter is not configured"` | `"transfer_adapter_missing"` |

Tests #1–#4 (`test_transfer_start_raise_*`) cover the first path. The
second path has no test today.

### Why it matters

In production, `transfer_adapter is None` happens when `install_transfer_adapter`
was never called — typically because `maybe_start_remote_g2_target_client`
failed silently during worker bootstrap (NIXL agent build crashed,
dynamo parent unreachable, etc.). The Mode 1 / Mode 2 source-worker
takedowns land here.

### Tests to add (4, by analogy with #1–#4)

| Test | Asserts |
|---|---|
| `test_transfer_adapter_missing_propagates_runtime_error_today` | `start_load_kv` raises `RuntimeError("not configured")` |
| `test_transfer_adapter_missing_releases_lease_once` | Lease released with reason `"transfer_adapter_missing"` |
| `test_transfer_adapter_missing_does_not_publish_binding` | No binding published |
| `test_transfer_adapter_missing_emits_fallback_event` | Fallback event with reason `"transfer_adapter_missing"` |

Setup is simpler than #1–#4: just `transfer_adapter=None` instead of
`transfer_adapter=_FakeTransferAdapter(...)`.

The first test (`*_today`) flips when the rewrite stops raising on
misconfiguration. The other three are stable invariants — survive the
rewrite unchanged.

---

## Gap B — multiple bindings, one fails mid-loop

Today's loop in `start_load_kv`:

```python
for record in metadata.bindings:
    request_id = record.request_id
    if request_id in self._active_loads or request_id in self._completed_loads:
        continue
    try:
        result = self._transfer_adapter.start_transfer(record)
    except Exception:
        # ...cleanup...
        raise RuntimeError("remote G2 transfer failed to start") from exc
    self._active_loads[request_id] = ...
```

If bindings = [A, B, C, D] and B's `start_transfer` raises, then:
- A made it into `_active_loads` (transfer is running)
- B raised → cleanup for B, then `raise`
- C and D never get processed — silently dropped from this tick

### Why it matters

Real scenario: B receives a burst of 4 requests, source A is briefly
unreachable for one of them. 1 transfer runs, 1 errors with a fallback
event, 2 requests just disappear with no observable signal.

### Tests to add (1-2)

| Test | Asserts |
|---|---|
| `test_transfer_multi_binding_one_fails_aborts_loop_today` | 1st made it into `_active_loads`; 3rd & 4th never started; only 1 fallback event emitted (for the 2nd) |
| `test_transfer_multi_binding_one_fails_other_succeeds_today` (optional) | Just the "1st made it" half, simpler |

### Prereq

`_bound_record()` hardcodes `request_id=1234`. Extend the helper to
take an optional `request_id` parameter so the test can build distinct
bindings:

```python
def _bound_record(lease_id="lease-bound", request_id=1234):
    ...
```

Backward compatible — existing tests use the default.

### Open design question

The `*_today` suffix here is doing double duty. The current "abort the
whole loop on first failure" behavior is *both* today's actual code AND
arguably a bug the rewrite should fix (process remaining bindings, only
the failing one falls back). Need to decide before writing the test
whether we're pinning the bug or pinning the contract. If pinning the
bug: the matching driving test would be
`test_transfer_multi_binding_one_fails_others_continue` (no propagate,
others succeed).

---

## Gaps deferred (low priority)

- **Duplicate request_id in `_active_loads` / `_completed_loads`**:
  silently skipped today. Pin only if it surfaces in practice.
- **Underlying failure reason preserved on the event**: today the
  connector wraps everything as `"transfer_start_failed"`. The
  underlying `RemoteG2TransferError` carries richer info (e.g.,
  `"source metadata generation mismatch"` vs NIXL errors). Better
  captured as a Group E xfail driving test (the rewrite should arguably
  forward the underlying reason for observability).
