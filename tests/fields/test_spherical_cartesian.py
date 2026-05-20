import math

import torch
from hypothesis import given
from hypothesis import strategies as st

from dendra.models.fields.spherical_cartesian import spherical_to_cartesian


def test_spherical_known_axes():
    """Simple sanity checks for the cardinal axes."""
    sph = torch.tensor(
        [
            [0.0, 90.0, 1.0],  # +X axis
            [90.0, 90.0, 1.0],  # +Y axis
            [0.0, 0.0, 1.0],  # +Z axis
        ]
    )
    cart = spherical_to_cartesian(sph)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.allclose(cart, expected, atol=1e-6)


@given(
    phi=st.floats(
        min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False
    ),
    theta=st.floats(
        min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False
    ),
    r=st.floats(min_value=1e-2, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_spherical_length_preserved(phi, theta, r):
    """Magnitude of the Cartesian vector must equal r."""
    inp = torch.tensor([phi, theta, r], dtype=torch.float64)
    out = spherical_to_cartesian(inp)
    mag = torch.linalg.norm(out).item()
    assert math.isclose(mag, float(r), rel_tol=1e-6, abs_tol=1e-6)


def test_spherical_gradcheck():
    """Autograd should compute accurate analytical gradients."""
    # generate one random point in valid domain
    rng = torch.Generator().manual_seed(0)
    vec = torch.empty(3, dtype=torch.float64)
    vec[0] = torch.rand((), dtype=torch.float64, generator=rng) * 360.0
    vec[1] = torch.rand((), dtype=torch.float64, generator=rng) * 180.0
    vec[2] = 1.5
    vec.requires_grad = True

    def func(v):
        return spherical_to_cartesian(v)

    assert torch.autograd.gradcheck(func, (vec,), eps=1e-6, atol=1e-4)
