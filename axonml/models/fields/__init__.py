from .analytic import anisotropic_point, isotropic_point, parametric_efield
from .precomputed import (
    EfieldInterpolate3DRect,
    EfieldInterpolate3DScattered,
    PreComputedInterpolate1D,
    PreComputedInterpolate3DRect,
    PreComputedInterpolate3DScattered,
)

# aliases
precomputed_interpolate_1d = PreComputedInterpolate1D
precomputed_interpolate_3d_rect = PreComputedInterpolate3DRect
precomputed_interpolate_3d_scattered = PreComputedInterpolate3DScattered
efield_interpolate_3d_rect = EfieldInterpolate3DRect
efield_interpolate_3d_scattered = EfieldInterpolate3DScattered

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
]
