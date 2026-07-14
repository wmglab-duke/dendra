"""Shared geometry adapters for unbranched voltage solvers."""

from __future__ import annotations

import torch

from .core import _as_solve_matrix, _model_solve_shape


def unbranched_edge_conductance(model) -> torch.Tensor:
    """Return path-edge conductance in siemens as a ``(B, K - 1)`` tensor.

    Generic native :class:`~dendra.models.core.Cable` models carry exact
    center-to-center resistances compiled from their Section geometry. Legacy
    Axon models retain their established half-cylinder reconstruction from
    ``diam``, ``dx``, and ``rhoa``.

    A canonical edge resistance already contains the source Sections' axial
    resistivities. Its optional ``rhoa_scale`` may therefore vary between solve
    rows, but must be spatially uniform within each cable; nonuniform scaling
    cannot be applied exactly after two half-path resistances have been reduced
    to one edge total.
    """
    B, K = _model_solve_shape(model)
    if K <= 1:
        return torch.zeros(B, 0, device=model.device(), dtype=model.dtype())

    exact = getattr(model, "_canonical_edge_resistance_ohm", None)
    if exact is not None:
        validate = getattr(model, "_validate_canonical_geometry", None)
        if validate is not None:
            validate()
        resistance = _as_solve_matrix(exact, model)
        scale = _as_solve_matrix(model.rhoa_scale, model)
        reference = scale[:, :1]
        # This is a semantic eligibility check, not a numerical comparison.
        # Accepting merely "close" entries would silently discard real
        # compartment-wise variation by applying only the first entry below.
        if not torch.equal(scale, reference.expand_as(scale)):
            raise ValueError(
                "A canonical Cable supports only spatially uniform rhoa_scale "
                "within each cable. Exact edge totals cannot recover distinct "
                "left/right half-path scaling."
            )
        return (resistance[:, 1:] * reference).reciprocal()

    diam = _as_solve_matrix(model.diam, model)
    dx = _as_solve_matrix(model.dx, model)
    rhoa = _as_solve_matrix(model.rhoa, model) * _as_solve_matrix(
        model.rhoa_scale, model
    )
    radius_cm = 1e-4 * diam / 2.0
    dx_cm = 1e-4 * dx
    segment_resistance = rhoa * dx_cm / (torch.pi * radius_cm.square())
    return 2.0 / (segment_resistance[:, :-1] + segment_resistance[:, 1:])


__all__ = ["unbranched_edge_conductance"]
