# tests/test_point_sources.py
import importlib
import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
MODULE_NAME = "dendra.models.fields.analytic"

point_src = importlib.import_module(MODULE_NAME)

isotropic_point = point_src.isotropic_point
anisotropic_point = point_src.anisotropic_point
parametric_efield = point_src.parametric_efield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class DummyModel:
    """
    Minimal stand-in for the neuron model expected by parametric_efield.
    Only the attributes actually used by the file under test are provided.
    """

    def __init__(self, x, y, z, *, device="cpu", dtype=torch.float32):
        # (1, N) tensors so broadcasting behaves exactly like in real code
        self.x = torch.as_tensor(x, device=device, dtype=dtype).unsqueeze(0)
        self.y = torch.as_tensor(y, device=device, dtype=dtype).unsqueeze(0)
        self.z = torch.as_tensor(z, device=device, dtype=dtype).unsqueeze(0)

        # These two are called in .forward()
        self.graph = None  # not used by forward, but expected by __forward

    # The Point base‑class calls these helpers to move itself
    def device(self):
        return self.x.device

    def dtype(self):
        return self.x.dtype


# ---------------------------------------------------------------------------
# Isotropic point source
# ---------------------------------------------------------------------------
def test_isotropic_point_scalar():
    """Analytic check for a single evaluation."""
    src = isotropic_point(rhoe=500.0)  # default location (0,0,0)
    # 100 µm along +x
    x, y, z = (torch.tensor([100.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    v = src.fn(x, y, z)

    # Expected:   V = rho / (4π r_cm)   ,  r_cm = r_µm × 1e‑4
    r_cm = 100.0 * 1e-4
    expected = 500.0 / (4 * math.pi * r_cm)

    assert torch.allclose(v, torch.tensor(expected), rtol=1e-6)


coord = st.floats(min_value=-2e4, max_value=2e4).filter(lambda v: abs(v) > 1.0)


@given(x=coord, y=coord, z=coord)
@settings(deadline=None)
def test_isotropic_point_grad(x, y, z):
    """Ensure gradients exist & are finite except at the source."""
    if x == y == z == 0.0:  # singularity – skip
        return

    src = isotropic_point()
    coords = torch.tensor([x, y, z], requires_grad=True, dtype=torch.double)
    v = src.fn(coords[0:1], coords[1:2], coords[2:3])
    (grad,) = torch.autograd.grad(v, coords, retain_graph=False)

    assert torch.isfinite(v).all()
    assert torch.isfinite(grad).all()


def test_anisotropic_equals_isotropic_when_resistivity_equal():
    """When rhox = rhoy = rhoz, formulas should coincide."""
    src_iso = isotropic_point(rhoe=500.0)
    src_aniso = anisotropic_point(rhox=500.0, rhoy=500.0, rhoz=500.0)

    coords = torch.tensor([[300.0], [400.0], [-200.0]])  # µm
    v_iso = src_iso.fn(*coords)
    v_aniso = src_aniso.fn(*coords)

    assert torch.allclose(v_iso, v_aniso, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# parametric_efield – public .forward()
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "phi, theta, z_um",
    [
        (1, 1, 500.0),
        (1, 2, 1000.0),
    ],
)
def test_parametric_efield_forward_values(phi, theta, z_um):
    """Validate V_e = -E * z * cos(θ) * 1000 mV for x = y = 0."""
    efield = parametric_efield(phi, theta, 0.0)

    model = DummyModel(
        x=[0.0],
        y=[0.0],
        z=[z_um],
        device="cpu",
        dtype=torch.float32,
    )

    out = efield(model, e_field_strength_Vm=1.0)  # (φ·θ, 1)
    assert out.shape == (phi * theta, 1)

    # ---- build analytic expectation ---------------------------------------
    theta_vals = (
        torch.linspace(0.0, 180.0, theta, dtype=out.dtype, device=out.device)
        * math.pi
        / 180.0
    )
    z_m = z_um / 1e6  # µm → m
    expected_theta = -z_m * torch.cos(theta_vals) * 1000  # (θ,)

    expected = expected_theta.repeat(phi)  # (φ·θ,)
    # -----------------------------------------------------------------------

    assert torch.allclose(out.squeeze(-1), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# parametric_efield – private __forward()
# ---------------------------------------------------------------------------
def test_parametric_efield_private_forward_clamps_negative(monkeypatch):
    """
    Patch out the heavy downstream functions so we can inspect the magnitude
    clamping behaviour in isolation.
    """
    # Identity converter so the 3rd column still holds the magnitude (r)
    monkeypatch.setattr(point_src, "spherical_to_cartesian", lambda x: x)

    # Dummy quasi‑potential so we don't need the full graph machinery
    class _QP:
        def __init__(self, data):
            self._d = data

        def contiguous(self):
            return self._d

    def _dummy_qp(**kwargs):
        return _QP(kwargs["e_fields_batch"].sum(-1))

    monkeypatch.setattr(
        point_src, "calculate_quasipotentials_batched_coords", _dummy_qp
    )

    # 200 % / mm → definitely drives magnitude < 0 for z = 1 mm
    efield = parametric_efield(1, 1, relative_mag_change_per_mm=200.0)

    model = DummyModel(x=[0.0], y=[0.0], z=[1000.0])  # 1 mm above origin
    e_fields, qp = efield._parametric_efield__forward(model, e_field_strength_Vm=1.0)

    # Magnitude (3rd component) should have been clamped to 0
    assert e_fields.shape[-1] == 3
    assert torch.allclose(e_fields[..., 2], torch.zeros_like(e_fields[..., 2]))


# ---------------------------------------------------------------------------
# Device / dtype propagation smoke test
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_and_dtype_propagation_cuda():
    device = torch.device("cuda:0")
    efield = parametric_efield(2, 2, 0.0).to(device=device, dtype=torch.float16)

    model = DummyModel(
        x=[0.0, 10.0],
        y=[0.0, 20.0],
        z=[0.0, 30.0],
        device=device,
        dtype=torch.float16,
    )

    out = efield(model)
    assert out.device == device
    assert out.dtype == torch.float16
