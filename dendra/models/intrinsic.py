"""Convenience helpers for conductance-driven intrinsic activity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .mod import alphasynapse

if TYPE_CHECKING:
    from .slice import Slice

__all__ = ["insert_intrinsic_activity", "remove_intrinsic_activity"]


_IntrinsicActivity = alphasynapse.rename("_IntrinsicActivity")
_IntrinsicActivity.__module__ = __name__
_IntrinsicActivity.__qualname__ = "_IntrinsicActivity"


def _require_slice(target: Any) -> Slice:
    """Return ``target`` after validating the public Slice-only contract."""
    from .slice import Slice

    if not isinstance(target, Slice):
        raise TypeError(
            "Intrinsic activity must target a dendra.models.slice.Slice; "
            f"got {type(target).__name__}."
        )
    return target


def _normalize_onsets(target: Slice, onsets: Any) -> torch.Tensor:
    """Normalize scalar/shared onset schedules to an explicit event axis."""
    if torch.is_tensor(onsets):
        normalized = onsets
    else:
        root = target.root_model
        try:
            normalized = torch.as_tensor(
                onsets,
                device=root.device(),
                dtype=root.dtype(),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(
                "onsets must be a numeric scalar, vector, or matrix."
            ) from exc

    if normalized.ndim == 0:
        return normalized.reshape(1, 1)
    if normalized.ndim == 1:
        return normalized.reshape(-1, 1)
    if normalized.ndim == 2:
        if normalized.shape[0] and not normalized.shape[1]:
            raise ValueError(
                "A non-empty onset schedule must select at least one location; "
                f"got shape {tuple(normalized.shape)}."
            )
        return normalized
    raise ValueError(
        "onsets must be scalar, a shared one-dimensional event train, or an "
        "explicit two-dimensional (events, locations) array; "
        f"got shape {tuple(normalized.shape)}."
    )


def insert_intrinsic_activity(
    target: Slice,
    onsets: Any,
    *,
    tau: Any = 0.1,
    gmax: Any = 0.1,
    e: Any = 0.0,
) -> None:
    """Insert an excitatory alpha-conductance event train on a Slice.

    Parameters
    ----------
    target : Slice
        Physical compartments that receive the activity. Empty Slices are a
        no-op.
    onsets : scalar or array_like
        Event times in ms. A scalar creates one event. A one-dimensional
        ``(K,)`` value is a shared ``K``-event train and is normalized to
        ``(K, 1)``. Two-dimensional values use explicit
        ``(events, locations)`` layout: ``(K, 1)`` shares a train across the
        Slice, while ``(K, L)`` supplies one onset per event and flattened
        physical location.
    tau : scalar or array_like, optional
        Alpha time constant in ms. Defaults to ``0.1``.
    gmax : scalar or array_like, optional
        Peak point-process conductance in microSiemens. Defaults to ``0.1``.
    e : scalar or array_like, optional
        Reversal potential in mV. Defaults to ``0.0``.

    Notes
    -----
    Repeated calls add independent colocated events. Non-scalar ``tau``,
    ``gmax``, and ``e`` values use the copied RANGE contract: provide
    ``(K, 1)``, ``(1, L)``, ``(K, L)``, or exactly ``K * L`` values rather
    than an ambiguous short one-dimensional vector. This is a structural model
    edit; call ``initialize()`` before the next simulation run.
    """
    target = _require_slice(target)
    if target.is_empty:
        return

    normalized_onsets = _normalize_onsets(target, onsets)
    copies = int(normalized_onsets.shape[0])
    if copies == 0:
        return

    target.insert(
        _IntrinsicActivity,
        preserve_multiplicity=True,
        copies=copies,
        onset=normalized_onsets,
        tau=tau,
        gmax=gmax,
        e=e,
    )


def remove_intrinsic_activity(target: Slice) -> None:
    """Remove helper-created intrinsic activity intersecting a Slice.

    Every event/copy created by :func:`insert_intrinsic_activity` at the
    selected physical compartments is removed. Activity outside the Slice and
    ordinary :class:`~dendra.models.mod.alphasynapse` insertions are preserved.
    Removing absent activity or targeting an empty Slice is a no-op. This is a
    structural model edit; call ``initialize()`` before the next simulation
    run.
    """
    target = _require_slice(target)
    if target.is_empty:
        return
    target.delete(_IntrinsicActivity)
