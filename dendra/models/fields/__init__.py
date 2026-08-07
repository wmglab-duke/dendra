"""Extracellular field models and the shared model-shape contract.

Model coordinates are tensors shaped ``(*batch, N, C)``: ``C`` is the final
compartment/morphology axis, ``N`` is the population axis, and any earlier axes
are independent batch axes. Ordinary scalar fields preserve this shape exactly.
Vector E-field wrappers interpolate vectors as ``(*batch, N, C, 3)`` and return
integrated quasipotentials with shape ``(*batch, N, C)``.

``parametric_efield`` is the deliberate exception because it can generate a
direction bank. With ``D = n_azimuthal * n_polar`` and
``Q = prod(model.shape[:-1])``, ``D == Q`` pairs directions with flattened model
lanes, ``D == 1`` broadcasts one direction, and ``Q == 1`` generates ``(D, C)``.
The ``D == 1`` shape-preserving rule takes precedence when ``D == Q == 1``.
Every other combination is rejected instead of creating a Cartesian product.

For ``PreComputedInterpolate1D``, a single LUT broadcasts; ``N`` LUT rows align
with the population axis and repeat over outer batches; ``Q`` rows align with
flattened model lanes. Explicit ``indices`` resolve any other mapping.
"""

from .analytic import anisotropic_point, isotropic_point, parametric_efield
from .line import arbitrary_line, arc_line, helix_line, line3d
from .precomputed import (
    EfieldInterpolate3DMesh,
    EfieldInterpolate3DRect,
    EfieldInterpolate3DScattered,
    PreComputedInterpolate1D,
    PreComputedInterpolate3DMesh,
    PreComputedInterpolate3DRect,
    PreComputedInterpolate3DScattered,
)

# aliases
precomputed_interpolate_1d = PreComputedInterpolate1D
precomputed_interpolate_3d_rect = PreComputedInterpolate3DRect
precomputed_interpolate_3d_scattered = PreComputedInterpolate3DScattered
precomputed_interpolate_3d_mesh = PreComputedInterpolate3DMesh
efield_interpolate_3d_rect = EfieldInterpolate3DRect
efield_interpolate_3d_scattered = EfieldInterpolate3DScattered
efield_interpolate_3d_mesh = EfieldInterpolate3DMesh

__all__ = [
    "anisotropic_point",
    "EfieldInterpolate3DScattered",
    "efield_interpolate_3d_scattered",
    "isotropic_point",
    "parametric_efield",
    "PreComputedInterpolate1D",
    "precomputed_interpolate_1d",
    "PreComputedInterpolate3DRect",
    "precomputed_interpolate_3d_rect",
    "PreComputedInterpolate3DScattered",
    "precomputed_interpolate_3d_scattered",
    "EfieldInterpolate3DRect",
    "efield_interpolate_3d_rect",
    "line3d",
    "arbitrary_line",
    "arc_line",
    "helix_line",
    "EfieldInterpolate3DMesh",
    "efield_interpolate_3d_mesh",
    "PreComputedInterpolate3DMesh",
    "precomputed_interpolate_3d_mesh",
]
