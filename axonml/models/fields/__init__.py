from .analytic import anisotropic_point, isotropic_point, parametric_efield
from .precomputed import EfieldInterpolate3D, PreComputedInterpolate1D

# aliases
precomputed_interpolate_1d = PreComputedInterpolate1D
efield_interpolate_3d = EfieldInterpolate3D

__all__ = [
    "anisotropic_point",
    "EfieldInterpolate3D",
    "efield_interpolate_3d",
    "isotropic_point",
    "parametric_efield",
    "PreComputedInterpolate1D",
    "precomputed_interpolate_1d",
]
