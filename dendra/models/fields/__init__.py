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
