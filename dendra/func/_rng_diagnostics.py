"""Scoped global RNG inspection for optional functional diagnostics."""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch


def _numpy_rng_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _global_rng_snapshot(device: torch.device) -> dict[str, object]:
    snapshot = {
        "torch": torch.random.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if device.type == "cuda":
        snapshot["cuda"] = torch.cuda.get_rng_state(device)
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        snapshot["mps"] = torch.mps.get_rng_state()
    return snapshot


def _global_rng_changes(snapshot, device: torch.device) -> set[str]:
    changed = set()
    if not torch.equal(torch.random.get_rng_state(), snapshot["torch"]):
        changed.add("global PyTorch")
    if random.getstate() != snapshot["python"]:
        changed.add("global Python")
    if not _numpy_rng_state_equal(np.random.get_state(), snapshot["numpy"]):
        changed.add("global NumPy")
    if "cuda" in snapshot and not torch.equal(
        torch.cuda.get_rng_state(device), snapshot["cuda"]
    ):
        changed.add("CUDA")
    if "mps" in snapshot and not torch.equal(
        torch.mps.get_rng_state(), snapshot["mps"]
    ):
        changed.add("MPS")
    return changed


def _restore_global_rng(snapshot, device: torch.device) -> None:
    torch.random.set_rng_state(snapshot["torch"])
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    if "cuda" in snapshot:
        torch.cuda.set_rng_state(snapshot["cuda"], device)
    if "mps" in snapshot:
        torch.mps.set_rng_state(snapshot["mps"])


class _RngDiagnostic:
    """Findings only; snapshots belong to the active scope."""

    __slots__ = ("changes",)

    def __init__(self):
        self.changes: set[str] = set()


@contextmanager
def _rng_diagnostic_scope(device: torch.device, *, enabled: bool):
    """Inspect and restore global RNG state under a captured caller policy.

    Callers report findings after the scope, so a probe exception retains
    precedence over a diagnostic finding. Disabled scopes do not inspect RNGs.
    """
    diagnostic = _RngDiagnostic()
    if not enabled:
        yield diagnostic
        return
    snapshot = _global_rng_snapshot(device)
    try:
        yield diagnostic
    finally:
        try:
            diagnostic.changes = _global_rng_changes(snapshot, device)
        finally:
            _restore_global_rng(snapshot, device)
