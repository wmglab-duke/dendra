from .analytic import anisotropic_point, isotropic_point, parametric_efield
from .precomputed import EfieldInterpolate3DScattered, PreComputedInterpolate1D

# aliases
precomputed_interpolate_1d = PreComputedInterpolate1D
efield_interpolate_3d_scattered = EfieldInterpolate3DScattered

__all__ = [
    "anisotropic_point",
    "EfieldInterpolate3DScattered",
    "efield_interpolate_3d_scattered",
    "isotropic_point",
    "parametric_efield",
    "PreComputedInterpolate1D",
    "precomputed_interpolate_1d",
]
