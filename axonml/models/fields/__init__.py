from .analytic import anisotropic_point, isotropic_point, parametric_efield
from .precomputed import PreComputedInterpolate1D

# aliases
precomputed_interpolate_1d = PreComputedInterpolate1D

__all__ = [
    "anisotropic_point",
    "isotropic_point",
    "parametric_efield",
    "PreComputedInterpolate1D",
    "precomputed_interpolate_1d",
]
