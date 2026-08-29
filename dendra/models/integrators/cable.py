"""Shared geometry adapters for unbranched voltage solvers."""

from __future__ import annotations

import torch

from .core import _as_solve_matrix, _model_solve_shape


def _cylindrical_membrane_area(
    diameter_um: torch.Tensor,
    length_um: torch.Tensor,
) -> torch.Tensor:
    """Return lateral cylindrical membrane area in square centimetres."""
    return diameter_um * 1.0e-4 * torch.pi * length_um * 1.0e-4


def _cylindrical_edge_conductance(
    diameter_um: torch.Tensor,
    length_um: torch.Tensor,
    axial_resistivity_ohm_cm: torch.Tensor,
) -> torch.Tensor:
    """Return centre-to-centre conductance for adjoining cylinders."""
    radius_cm = 1.0e-4 * diameter_um / 2.0
    length_cm = 1.0e-4 * length_um
    segment_resistance = (
        axial_resistivity_ohm_cm * length_cm / (torch.pi * radius_cm.square())
    )
    return 2.0 / (segment_resistance[..., :-1] + segment_resistance[..., 1:])


def _layered_edge_conductance(
    axial_resistance_mohm_per_cm: torch.Tensor,
    length_um: torch.Tensor,
) -> torch.Tensor:
    """Return centre-to-centre conductance for extracellular cable layers.

    ``axial_resistance_mohm_per_cm`` has one final axis per extracellular
    layer.  Each compartment contributes half of its longitudinal resistance
    to either adjoining edge, matching the intracellular cylindrical adapter
    and NEURON's ``xraxial`` convention.
    """
    length_cm = length_um * 1.0e-4
    segment_resistance = axial_resistance_mohm_per_cm * length_cm.unsqueeze(-1) * 1.0e6
    return 2.0 / (segment_resistance[..., :-1, :] + segment_resistance[..., 1:, :])


def _canonical_edge_conductance(
    resistance_ohm: torch.Tensor,
    rhoa_scale: torch.Tensor,
) -> torch.Tensor:
    """Return exact path conductance from child-indexed edge resistance.

    Native ``Cable`` resistance stores a zero placeholder at the root and the
    complete centre-to-centre resistance at every subsequent child.  The
    compiled resistance already incorporates the source morphology's axial
    resistivity, so only the supported spatially uniform runtime scale remains
    to be applied here.
    """
    return (resistance_ohm[..., 1:] * rhoa_scale).reciprocal()


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
        return _canonical_edge_conductance(resistance, reference)

    diam = _as_solve_matrix(model.diam, model)
    dx = _as_solve_matrix(model.dx, model)
    rhoa = _as_solve_matrix(model.rhoa, model) * _as_solve_matrix(
        model.rhoa_scale, model
    )
    return _cylindrical_edge_conductance(diam, dx, rhoa)


__all__ = ["unbranched_edge_conductance"]
