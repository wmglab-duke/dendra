import warnings

import pytest
import torch
import torch.nn.functional as F

from dendra.models.integrators.explicit import (
    SymmetricConv1D,
    _conv_last,
    _dufort_frankel,
    _dufort_frankel_homogeneous,
    _euler,
    _filter_last,
    _rk2,
    _rk4,
    ssd_df,
    ssd_df_no_ve,
)
from dendra.models.integrators.implicit import (
    DENDRA_SOLVERS_AVAILABLE,
    _bwd_euler_bt,
    _bwd_euler_sc,
    _bwd_euler_sc_multi,
    _bwd_euler_sc_skip,
    _bwd_euler_ub,
    assemble_rhs,
    assemble_rhs_into,
)
from dendra.models.integrators.tridiag import pcr_solve_t

DTYPE = torch.float64


class CableModel(torch.nn.Module):
    """Small model exposing the geometry/runtime contract used by integrators."""

    def __init__(
        self,
        shape=(1, 5),
        *,
        v_init=-65.0,
        diam=2.0,
        dx=10.0,
        cm=1.0,
        rhoa=100.0,
        n_layers=2,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.v_init = v_init
        self.n_layers = n_layers
        self.celsius = 37.0

        self.jit = False
        self.jit_network_solves = False
        self.jit_network_ops = False
        self.backend = "inductor"
        self.fullgraph = False
        self.dynamic = False
        self.compile_mode = None
        self.compile_options = None

        def full(value, shape=self.shape):
            tensor = torch.as_tensor(value, dtype=DTYPE)
            if tensor.ndim < len(shape):
                tensor = tensor.reshape(
                    (1,) * (len(shape) - tensor.ndim) + tensor.shape
                )
            return tensor.expand(shape).clone()

        self.register_buffer("v", full(v_init))
        self.register_buffer("diam", full(diam))
        self.register_buffer("dx", full(dx))
        self.register_buffer("cm", full(cm))
        self.register_buffer("rhoa", full(rhoa))

        shell_shape = self.shape + (max(n_layers, 1),)
        self.register_buffer("xraxial", torch.full(shell_shape, 2.0, dtype=DTYPE))
        self.register_buffer("xc", torch.full(shell_shape, 0.5, dtype=DTYPE))
        self.register_buffer("xg", torch.full(shell_shape, 1.0e-4, dtype=DTYPE))

    def device(self):
        return self.v.device

    def dtype(self):
        return self.v.dtype

    def expanded_v_init(self):
        value = torch.as_tensor(self.v_init, dtype=self.v.dtype, device=self.v.device)
        if value.ndim == 0:
            return value.expand_as(self.v)
        return value.reshape((1,) * (self.v.ndim - value.ndim) + value.shape).expand_as(
            self.v
        )


class LinearMechanism(torch.nn.Module):
    """Passive current I = g (v - e), with an optional voltage-process shift."""

    def __init__(self, g=0.0, e=0.0, shift=0.0):
        super().__init__()
        self.g = g
        self.e = e
        self.shift = shift
        self.advance_calls = []
        self.i_calls = 0
        self.detached = False
        self.dt = None

    def _g(self, v):
        return torch.as_tensor(self.g, dtype=v.dtype, device=v.device).expand_as(v)

    def _current(self, v):
        return self._g(v) * (
            v - torch.as_tensor(self.e, dtype=v.dtype, device=v.device)
        )

    def update_v(self, v):
        return v + torch.as_tensor(self.shift, dtype=v.dtype, device=v.device)

    def advance(self, v, dt, temp):
        self.advance_calls.append((dt, temp, v.detach().clone()))

    def iexp(self, v):
        return self._current(v)

    def idf(self, v, v_prev):
        return self._current(v), self._g(v)

    def itot(self, v):
        return self._current(v)

    def i(self, v):
        self.i_calls += 1
        return self._current(v), self._g(v)

    def set_dt(self, dt):
        self.dt = dt

    def detach(self):
        self.detached = True


def _explicit_for_ode(cls, *, g=2.0, imem=False, shape=(2, 4)):
    model = CableModel(shape=shape, v_init=1.0)
    mech = LinearMechanism(g=g)
    integrator = cls(model, mech, imem=imem).to(dtype=DTYPE)
    integrator.cm_inv = torch.ones(shape, dtype=DTYPE)
    integrator.ra_inv = torch.zeros(shape, dtype=DTYPE)
    integrator.area_c = torch.ones(shape, dtype=DTYPE)
    integrator.cm_c = torch.ones(shape, dtype=DTYPE)
    integrator.ve_zero = torch.zeros(shape, dtype=DTYPE)
    return model, mech, integrator


def _dense_tridiagonal(lower, main, upper):
    matrix = torch.diag_embed(main)
    rows = torch.arange(main.shape[-1] - 1)
    matrix[:, rows + 1, rows] = lower
    matrix[:, rows, rows + 1] = upper
    return matrix


def _dense_block_solve(lower, main, upper, rhs):
    batch, compartments, block, _ = main.shape
    dense = torch.zeros(
        batch,
        compartments * block,
        compartments * block,
        dtype=main.dtype,
        device=main.device,
    )
    for k in range(compartments):
        sl = slice(k * block, (k + 1) * block)
        dense[:, sl, sl] = main[:, k]
        if k < compartments - 1:
            nxt = slice((k + 1) * block, (k + 2) * block)
            dense[:, nxt, sl] = torch.diag_embed(lower[:, k])
            dense[:, sl, nxt] = torch.diag_embed(upper[:, k])
    result = torch.linalg.solve(dense, rhs.reshape(batch, -1, 1))
    return result.reshape_as(rhs)


def test_symmetric_conv_enforces_symmetric_stencil_only_while_training():
    conv = SymmetricConv1D(1, 1, 3, bias=False).to(dtype=DTYPE)
    conv.weight.data.copy_(torch.tensor([[[1.0, 2.0, 5.0]]], dtype=DTYPE))
    x = torch.tensor([[[1.0, 2.0, 4.0]]], dtype=DTYPE)

    conv.train()
    actual_train = conv(x)
    symmetric = torch.tensor([[[3.0, 2.0, 3.0]]], dtype=DTYPE)
    assert torch.allclose(actual_train, F.conv1d(x, symmetric))

    conv.eval()
    assert torch.allclose(conv(x), F.conv1d(x, conv.weight))
    assert not torch.allclose(conv(x), actual_train)


def test_conv_and_filter_helpers_preserve_arbitrary_leading_dimensions_and_gradients():
    conv = torch.nn.Conv1d(2, 1, 3, padding="same", bias=False).to(dtype=DTYPE)
    conv.weight.data.copy_(
        torch.tensor([[[1.0, -2.0, 1.0], [0.0, 1.0, 0.0]]], dtype=DTYPE)
    )
    x = torch.arange(24.0, dtype=DTYPE).reshape(2, 3, 4).requires_grad_()
    y = torch.ones(4, dtype=DTYPE, requires_grad=True)
    actual = _conv_last(conv, x, y)
    expected = (
        conv(torch.stack([x.reshape(-1, 4), y.expand_as(x).reshape(-1, 4)], dim=1))
        .squeeze(1)
        .reshape_as(x)
    )
    assert actual.shape == x.shape
    assert torch.allclose(actual, expected)

    filt = torch.nn.Conv1d(1, 1, 3, padding=1, bias=False).to(dtype=DTYPE)
    filt.weight.data.fill_(1.0 / 3.0)
    filtered = _filter_last(filt, actual)
    filtered.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert y.grad is not None and torch.isfinite(y.grad).all()


def test_spatial_stencils_have_known_reflect_boundary_values_and_batched_shapes():
    current = torch.tensor([[[1.0, 2.0, 4.0], [3.0, 5.0, 8.0]]], dtype=DTYPE)
    previous = torch.tensor([[[0.5, 1.5, 2.5], [2.0, 4.0, 7.0]]], dtype=DTYPE)
    extracellular = torch.tensor([0.0, 1.0, 3.0], dtype=DTYPE)

    cur_pad = F.pad(current.reshape(-1, 3), (1, 1), mode="reflect")
    ve = extracellular.expand_as(current).reshape(-1, 3)
    ve_pad = F.pad(ve, (1, 1), mode="reflect")
    expected_no_ve = (
        cur_pad[:, :-2] + cur_pad[:, 2:] - previous.reshape(-1, 3)
    ).reshape_as(current)
    expected = (
        expected_no_ve.reshape(-1, 3) + ve_pad[:, :-2] + ve_pad[:, 2:] - 2 * ve
    ).reshape_as(current)

    assert torch.allclose(ssd_df_no_ve(current, previous), expected_no_ve)
    assert torch.allclose(ssd_df(current, previous, extracellular), expected)


def test_spatial_stencils_support_a_single_compartment_as_a_sealed_boundary():
    current = torch.tensor([[2.0]], dtype=DTYPE)
    previous = torch.tensor([[1.0]], dtype=DTYPE)
    extracellular = torch.tensor([[7.0]], dtype=DTYPE)
    # Both reflected neighbours are the only compartment; extracellular
    # differences therefore vanish.
    assert torch.equal(
        ssd_df_no_ve(current, previous), torch.tensor([[3.0]], dtype=DTYPE)
    )
    assert torch.equal(
        ssd_df(current, previous, extracellular), torch.tensor([[3.0]], dtype=DTYPE)
    )


@pytest.mark.parametrize(
    "cls, factor",
    [
        (_euler, lambda z: 1 - z),
        (_rk2, lambda z: 1 - z + z**2 / 2),
        (_rk4, lambda z: 1 - z + z**2 / 2 - z**3 / 6 + z**4 / 24),
    ],
)
def test_explicit_runge_kutta_methods_match_linear_decay_polynomials(cls, factor):
    model, mech, integrator = _explicit_for_ode(cls, g=2.0, imem=True)
    v = torch.tensor([[1.0, -2.0, 0.5, 3.0], [0.25, 0.75, -1.0, 2.0]], dtype=DTYPE)
    dt = 0.1
    actual, imem = integrator._step(
        v, integrator.ve_zero, integrator.area_c, dt, 37.0, integrator.cm_c, None
    )
    assert torch.allclose(actual, factor(2.0 * dt) * v, atol=1e-12)
    assert torch.allclose(imem, torch.zeros_like(v), atol=1e-12)
    assert len(mech.advance_calls) == 1


def test_explicit_euler_injection_broadcasts_and_has_analytic_gradients():
    _, _, integrator = _explicit_for_ode(_euler, g=0.0, shape=(2, 3))
    v = torch.ones((2, 3), dtype=DTYPE, requires_grad=True)
    injection = torch.tensor([0.2, -0.1, 0.4], dtype=DTYPE, requires_grad=True)
    slope, ionic = integrator.FRK(
        v,
        torch.zeros_like(v),
        torch.ones_like(v),
        torch.ones_like(v),
        torch.zeros_like(v),
        injection,
    )
    assert torch.allclose(slope, injection.expand_as(v))
    assert torch.allclose(ionic, -injection.expand_as(v))
    slope.sum().backward()
    assert torch.equal(v.grad, torch.zeros_like(v))
    assert torch.equal(injection.grad, torch.full_like(injection, 2.0))


def test_explicit_initialize_and_step_commit_geometry_and_membrane_current():
    model = CableModel(shape=(2, 3), v_init=-5.0, diam=torch.tensor([2.0, 4.0, 8.0]))
    mech = LinearMechanism(g=0.0)
    integrator = _euler(model, mech, imem=True).to(dtype=DTYPE)
    integrator._initialize(model, 0.05)

    dx_cm = model.dx / 10000.0
    expected_area = torch.pi * (model.diam / 10000.0) * dx_cm
    expected_cm = model.cm / 1000.0 * expected_area
    expected_ra = model.rhoa * dx_cm / (torch.pi * (model.diam / 20000.0) ** 2)
    assert torch.allclose(integrator.area_c, expected_area)
    assert torch.allclose(integrator.cm_c, expected_cm)
    assert torch.allclose(integrator.cm_inv, expected_cm.reciprocal())
    assert torch.allclose(integrator.ra_inv, expected_ra.reciprocal())

    original = model.v.clone()
    integrator.step(model, 0.05, intra=torch.tensor([1.0e-9, 0.0, -1.0e-9]))
    assert model.v.shape == original.shape
    assert not torch.equal(model.v, original)
    assert model.i_membrane.shape == model.shape


def test_explicit_euler_single_compartment_has_no_artificial_axial_current():
    model = CableModel(shape=(3, 1), v_init=torch.tensor([-3.0]))
    integrator = _euler(model, LinearMechanism(g=0.0)).to(dtype=DTYPE)
    integrator._initialize(model, 0.1)
    before = model.v.clone()
    integrator.step(model, 0.1)
    assert torch.equal(model.v, before)


@pytest.mark.parametrize("conv", [False, True])
def test_homogeneous_dufort_frankel_known_injection_balance_and_state_commit(conv):
    model = CableModel(shape=(2, 5), v_init=-2.0)
    mech = LinearMechanism(g=0.0)
    integrator = _dufort_frankel_homogeneous(model, mech, conv=conv, imem=True).to(
        dtype=DTYPE
    )
    integrator._initialize(model, 0.1)
    integrator.s2.zero_()
    integrator.s4.fill_(1.0)
    injection = torch.linspace(-0.2, 0.2, 5, dtype=DTYPE)
    previous = model.v_prev.clone()

    integrator.step(model, 0.1, intra=injection)
    expected = previous + integrator.s1 * injection.expand_as(previous)
    assert torch.allclose(model.v, expected)
    assert torch.equal(model.v_prev, torch.full_like(previous, -2.0))
    assert torch.allclose(model.i_membrane, torch.zeros_like(model.v), atol=1e-12)


def test_homogeneous_dufort_conv_and_direct_stencils_are_equivalent():
    model_direct = CableModel(shape=(2, 5), v_init=-2.0)
    model_conv = CableModel(shape=(2, 5), v_init=-2.0)
    mech_direct = LinearMechanism(g=0.03, e=-4.0)
    mech_conv = LinearMechanism(g=0.03, e=-4.0)
    direct = _dufort_frankel_homogeneous(model_direct, mech_direct, conv=False).to(
        dtype=DTYPE
    )
    conv = _dufort_frankel_homogeneous(model_conv, mech_conv, conv=True).to(dtype=DTYPE)
    direct._initialize(model_direct, 0.01)
    conv._initialize(model_conv, 0.01)

    v = torch.tensor(
        [[-2.0, -1.0, 0.0, 2.0, 1.0], [1.0, 0.5, -0.5, -1.0, -2.0]],
        dtype=DTYPE,
    )
    v_prev = v - 0.1
    ve = torch.linspace(-0.3, 0.2, 5, dtype=DTYPE)
    args = (
        v,
        v_prev,
        ve,
        direct.s1,
        direct.s2,
        direct.s3,
        direct.s4,
        direct.area,
        0.01,
        37.0,
    )
    direct_result = direct._step(*args)[0]
    conv_args = (
        v,
        v_prev,
        ve.expand_as(v),
        conv.s1,
        conv.s2,
        conv.s3,
        conv.s4,
        conv.area,
        0.01,
        37.0,
    )
    conv_result = conv._step_conv(*conv_args)[0]
    assert torch.allclose(direct_result, conv_result, atol=1e-12, rtol=1e-12)


def test_homogeneous_dufort_smoothing_and_init_v_work_for_batched_state():
    model = CableModel(shape=(2, 5), v_init=torch.arange(5.0, dtype=DTYPE))
    integrator = _dufort_frankel_homogeneous(
        model, LinearMechanism(), beta=0.5, imem=True
    ).to(dtype=DTYPE)
    integrator._initialize(model, 0.1)
    v = torch.tensor(
        [[0.0, 0.0, 10.0, 0.0, 0.0], [1.0, 2.0, 4.0, 2.0, 1.0]], dtype=DTYPE
    )
    integrator.s2.zero_()
    integrator.s4.fill_(1.0)
    result, _, _ = integrator._step(
        v,
        v,
        None,
        integrator.s1,
        integrator.s2,
        integrator.s3,
        integrator.s4,
        integrator.area,
        0.1,
        37.0,
    )
    assert result.shape == v.shape
    assert not torch.equal(result, v)

    model.v.fill_(99.0)
    model.v_prev.fill_(98.0)
    integrator.init_v(model)
    assert torch.equal(model.v, model.expanded_v_init())
    assert torch.equal(model.v_prev, model.expanded_v_init())
    assert torch.equal(model.i_membrane, torch.zeros_like(model.v))


def test_heterogeneous_dufort_preserves_uniform_voltage_and_supports_optional_inputs():
    model = CableModel(
        shape=(2, 5),
        v_init=-3.0,
        diam=torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]),
        dx=torch.tensor([8.0, 10.0, 9.0, 12.0, 11.0]),
    )
    integrator = _dufort_frankel(
        model, LinearMechanism(g=0.0), beta=0.75, imem=True
    ).to(dtype=DTYPE)
    integrator._initialize(model, 0.01)
    integrator.step(model, 0.01)
    assert torch.allclose(model.v, torch.full_like(model.v, -3.0))
    assert torch.allclose(model.i_membrane, torch.zeros_like(model.v), atol=1e-12)

    before = model.v.clone()
    ve = torch.linspace(-0.1, 0.2, 5, dtype=DTYPE)
    intra = torch.linspace(0.0, 1.0e-10, 5, dtype=DTYPE)
    integrator.step(model, 0.01, ve=ve, intra=intra)
    assert model.v.shape == model.shape
    assert model.i_membrane.shape == model.shape
    assert torch.isfinite(model.v).all()
    assert not torch.equal(model.v, before)


def test_heterogeneous_and_homogeneous_dufort_match_for_uniform_geometry():
    model_h = CableModel(shape=(1, 5), v_init=-1.0)
    model_u = CableModel(shape=(1, 5), v_init=-1.0)
    mech_h = LinearMechanism(g=0.02, e=-2.0)
    mech_u = LinearMechanism(g=0.02, e=-2.0)
    heterogeneous = _dufort_frankel(model_h, mech_h).to(dtype=DTYPE)
    homogeneous = _dufort_frankel_homogeneous(model_u, mech_u).to(dtype=DTYPE)
    heterogeneous._initialize(model_h, 0.01)
    homogeneous._initialize(model_u, 0.01)
    v = torch.tensor([[-1.0, 0.0, 2.0, 1.0, -2.0]], dtype=DTYPE)
    previous = v - 0.2

    actual = heterogeneous._step(v, previous, None, 0.01, 37.0)[0]
    expected = homogeneous._step(
        v,
        previous,
        None,
        homogeneous.s1,
        homogeneous.s2,
        homogeneous.s3,
        homogeneous.s4,
        homogeneous.area,
        0.01,
        37.0,
    )[0]
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_heterogeneous_dufort_init_v_resets_both_time_levels():
    model = CableModel(shape=(2, 4), v_init=torch.arange(4.0, dtype=DTYPE))
    integrator = _dufort_frankel(model, LinearMechanism(), imem=True).to(dtype=DTYPE)
    model.v.fill_(10.0)
    model.v_prev.fill_(20.0)
    integrator.init_v(model)
    assert torch.equal(model.v, model.expanded_v_init())
    assert torch.equal(model.v_prev, model.expanded_v_init())
    assert torch.equal(model.i_membrane, torch.zeros_like(model.v))


def test_backward_euler_single_compartment_matches_closed_form_and_current_balance():
    model = CableModel(shape=(2, 3), v_init=-4.0, diam=20.0, dx=50.0, cm=2.0)
    mech = LinearMechanism(g=0.4, e=-7.0, shift=0.5)
    integrator = _bwd_euler_sc(model, mech, imem=True).to(dtype=DTYPE)
    integrator._initialize(model, 0.2)
    injection = torch.tensor([1.0e-8, -2.0e-8, 0.0], dtype=DTYPE)
    shifted = model.v + 0.5
    denom = integrator.cmdt + 0.4
    expected = shifted - 0.4 * (shifted + 7.0) / denom
    expected = expected + (injection / integrator.area) / denom

    integrator.step(model, 0.2, intra=injection)
    assert torch.allclose(model.v, expected)
    assert torch.allclose(model.i_membrane, injection.expand_as(model.v), atol=1e-18)
    assert mech.advance_calls[0][:2] == (0.2, 37.0)


def test_backward_euler_single_compartment_gradient_matches_closed_form():
    model = CableModel(shape=(1, 3))
    integrator = _bwd_euler_sc(model, LinearMechanism(g=0.25)).to(dtype=DTYPE)
    integrator.cmdt = torch.tensor(0.75, dtype=DTYPE)
    integrator.area = torch.tensor(2.0, dtype=DTYPE)
    v = torch.tensor([[1.0, -2.0, 4.0]], dtype=DTYPE, requires_grad=True)
    intra = torch.tensor([[0.2, 0.4, -0.1]], dtype=DTYPE, requires_grad=True)
    result, _ = integrator._solve(v, 0.1, 37.0, intra)
    result.sum().backward()
    assert torch.allclose(v.grad, torch.full_like(v, 0.75))
    assert torch.allclose(intra.grad, torch.full_like(intra, 0.5))


def test_backward_euler_area_is_registered_for_state_and_device_migration():
    model = CableModel(shape=(1, 2))
    integrator = _bwd_euler_sc(model, LinearMechanism()).to(dtype=DTYPE)
    integrator._initialize(model, 0.1)
    assert "area" in integrator.state_dict()
    assert integrator.area.dtype == DTYPE


def test_backward_euler_skip_applies_voltage_process_and_evaluates_current():
    model = CableModel(shape=(2, 3), v_init=-5.0)
    mech = LinearMechanism(g=0.1, shift=1.25)
    integrator = _bwd_euler_sc_skip(model, mech)
    integrator._initialize(model, 0.1)
    integrator.step(model, 0.1, intra=torch.ones_like(model.v))
    assert torch.equal(model.v, torch.full_like(model.v, -3.75))
    assert mech.i_calls == 1
    assert len(mech.advance_calls) == 1


def test_backward_euler_multi_commits_and_honors_write_back_switch():
    model = CableModel(shape=(1, 3), v_init=-2.0)
    mech = LinearMechanism(g=0.2, e=-3.0)
    integrator = _bwd_euler_sc_multi(model, mech, write_back=False)
    integrator._initialize(model, 0.1)
    integrator.step(model, 0.1)
    assert model.v.shape == model.shape
    assert not torch.equal(model.v, torch.full_like(model.v, -2.0))


def test_unbranched_implicit_geometry_builds_correct_asymmetric_bands():
    cm = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=DTYPE)
    model = CableModel(shape=(1, 4), cm=cm, diam=2.0, dx=10.0, rhoa=100.0)
    integrator = _bwd_euler_ub(model, LinearMechanism(), method="pcr").to(dtype=DTYPE)
    dt = 0.2
    integrator._initialize(model, dt)

    radius = 1.0e-4 * model.diam.reshape(1, 4) / 2
    length = 1.0e-4 * model.dx.reshape(1, 4)
    area = 2 * torch.pi * radius * length
    capacitance = 1.0e-6 * model.cm.reshape(1, 4) * area
    resistance = model.rhoa.reshape(1, 4) * length / (torch.pi * radius.square())
    edge = 2 / (resistance[:, :-1] + resistance[:, 1:])
    expected_upper = -(dt * 1.0e-3) * edge / capacitance[:, :-1]
    expected_lower = -(dt * 1.0e-3) * edge / capacitance[:, 1:]
    assert torch.allclose(integrator.upper, expected_upper)
    assert torch.allclose(integrator.lower, expected_lower)
    assert integrator.diag_base.shape == (1, 4)


def test_unbranched_implicit_step_matches_dense_linear_system_and_gradients():
    model = CableModel(
        shape=(2, 4),
        v_init=-2.0,
        cm=torch.tensor([1.0, 2.0, 1.5, 3.0]),
        diam=torch.tensor([2.0, 3.0, 2.5, 4.0]),
    )
    mech = LinearMechanism(g=0.03, e=-5.0)
    integrator = _bwd_euler_ub(model, mech, method="pcr", imem=True).to(dtype=DTYPE)
    dt = 0.05
    integrator._initialize(model, dt)
    v = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0], [1.0, 0.5, -0.5, -1.5]],
        dtype=DTYPE,
        requires_grad=True,
    )
    intra = torch.tensor([1.0e-10, 0.0, -2.0e-10, 0.5e-10], dtype=DTYPE)
    actual, imem = integrator._step(v, dt, 37.0, intra=intra)

    gtot = torch.full_like(v, 0.03).reshape(2, 4)
    itot = 0.03 * (v.reshape(2, 4) + 5.0)
    source = (gtot * v.reshape(2, 4) - itot) * integrator.scale
    source = source + intra.expand_as(v).reshape(2, 4) * integrator.cm_inv
    rhs = v.reshape(2, 4) + dt * 1.0e-3 * source
    main = 1 - dt * 1.0e-3 * (integrator.diag_base - gtot * integrator.scale)
    dense = _dense_tridiagonal(integrator.lower, main, integrator.upper)
    expected = torch.linalg.solve(dense, rhs.unsqueeze(-1)).squeeze(-1).reshape_as(v)
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)
    assert imem.shape == model.shape and torch.isfinite(imem).all()
    actual.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert integrator._last_bands is not None


def test_unbranched_implicit_preserves_uniform_intracellular_potential_with_ve():
    model = CableModel(
        shape=(1, 4),
        cm=torch.tensor([1.0, 2.0, 4.0, 8.0]),
        diam=torch.tensor([2.0, 3.0, 4.0, 5.0]),
    )
    ve = torch.tensor([[0.0, 1.0, -0.5, 2.0]], dtype=DTYPE)
    model.v = -ve.clone()
    integrator = _bwd_euler_ub(model, LinearMechanism(), method="pcr").to(dtype=DTYPE)
    integrator._initialize(model, 0.1)
    integrator.step(model, 0.1, ve=ve)
    assert torch.allclose(model.v, -ve, atol=1e-11, rtol=1e-11)


def test_unbranched_implicit_single_compartment_injection_and_imem_balance():
    model = CableModel(shape=(2, 1), v_init=-3.0)
    integrator = _bwd_euler_ub(model, LinearMechanism(), method="pcr", imem=True).to(
        dtype=DTYPE
    )
    integrator._initialize(model, 0.1)
    injection = torch.tensor([[1.0e-10], [-2.0e-10]], dtype=DTYPE)
    integrator.step(model, 0.1, ve=torch.tensor([5.0], dtype=DTYPE), intra=injection)
    assert model.v.shape == (2, 1)
    assert torch.allclose(model.i_membrane, injection, atol=1e-20, rtol=1e-10)


def test_unbranched_solver_selection_alias_fallbacks_and_clip_configuration():
    model = CableModel(shape=(1, 3))
    inv = _bwd_euler_ub(model, LinearMechanism(), method="inv")
    assert inv.method == "thomas"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inv._select_solver(model)
    if not DENDRA_SOLVERS_AVAILABLE:
        assert inv._solve is pcr_solve_t
        assert any("Falling back to PCR" in str(item.message) for item in caught)

    with pytest.warns(UserWarning, match="only supported"):
        clipped = _bwd_euler_ub(
            model, LinearMechanism(), method="pcr", clip_scale_backward=2.0
        )
    assert clipped.use_gc_variant is False

    unknown = _bwd_euler_ub(model, LinearMechanism(), method="not-a-solver")
    with pytest.warns(UserWarning, match="Unknown or unsupported"):
        unknown._select_solver(model)
    assert unknown.method == "thomas"


def test_unbranched_spd_selection_uses_solver_or_warns_and_falls_back():
    model = CableModel(shape=(1, 3))
    integrator = _bwd_euler_ub(model, LinearMechanism(), method="spd")
    if DENDRA_SOLVERS_AVAILABLE:
        integrator._select_solver(model)
        assert callable(integrator._solve)
    else:
        with pytest.warns(UserWarning, match="not available"):
            integrator._select_solver(model)
        assert integrator._solve is pcr_solve_t


def test_assemble_rhs_matches_manual_balance_and_inplace_variant():
    v_prev = torch.tensor(
        [[[[4.0, 1.0, -1.0], [3.0, 0.5, -0.5]]]], dtype=DTYPE
    ).reshape(1, 2, 3)
    c_rad = torch.tensor([[[2.0, 0.5, 0.25], [1.5, 0.75, 0.4]]], dtype=DTYPE)
    d = torch.tensor([[0.3, -0.2]], dtype=DTYPE)
    xg = torch.tensor([[0.1, 0.2]], dtype=DTYPE)
    e_ext = torch.tensor([[2.0, -1.0]], dtype=DTYPE)

    expected = torch.zeros_like(v_prev)
    radial = c_rad[..., :-1] * (v_prev[..., :-1] - v_prev[..., 1:])
    expected[..., :-1] += radial
    expected[..., 1:] -= radial
    expected[..., 0] += d
    expected[..., 1] -= d
    expected[..., -1] += xg * e_ext + c_rad[..., -1] * v_prev[..., -1]
    assert torch.allclose(assemble_rhs(v_prev, c_rad, d, xg, e_ext), expected)

    out = torch.full_like(v_prev, torch.nan)
    returned = assemble_rhs_into(out, v_prev, c_rad, d, xg, e_ext)
    assert returned.data_ptr() == out.data_ptr()
    assert torch.allclose(out, expected)

    without_external = assemble_rhs(v_prev, c_rad, d, xg, None)
    out.fill_(torch.nan)
    assert torch.allclose(
        assemble_rhs_into(out, v_prev, c_rad, d, xg, None), without_external
    )


def test_assemble_rhs_is_differentiable_for_all_physical_inputs():
    v = torch.randn(2, 4, 3, dtype=DTYPE, requires_grad=True)
    c = torch.rand(2, 4, 3, dtype=DTYPE, requires_grad=True)
    d = torch.randn(2, 4, dtype=DTYPE, requires_grad=True)
    xg = torch.rand(2, 4, dtype=DTYPE, requires_grad=True)
    external = torch.randn(2, 4, dtype=DTYPE, requires_grad=True)
    assemble_rhs(v, c, d, xg, external).square().sum().backward()
    for tensor in (v, c, d, xg, external):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_block_implicit_validates_method_and_three_unknown_specialization():
    model = CableModel(shape=(1, 3), n_layers=2)
    with pytest.raises(ValueError, match="Unknown method"):
        _bwd_euler_bt(model, LinearMechanism(), method="bad")
    wrong_layers = CableModel(shape=(1, 3), n_layers=1)
    with pytest.raises(ValueError, match="exactly 3 unknowns"):
        _bwd_euler_bt(wrong_layers, LinearMechanism())
    assert _bwd_euler_bt.shape(2, 5) == (2, 5)


def test_block_implicit_initialization_and_step_match_dense_residual():
    model = CableModel(shape=(1, 3), v_init=-4.0, n_layers=2)
    mech = LinearMechanism(g=0.02, e=-6.0)
    integrator = _bwd_euler_bt(model, mech, method="thomas", imem=True).to(dtype=DTYPE)
    integrator._select_solver = lambda owner: setattr(
        integrator, "_solve", _dense_block_solve
    )
    integrator._initialize(model, 0.1)

    assert integrator.maind.shape == (1, 3, 3, 3)
    assert torch.allclose(integrator.maind, integrator.maind.transpose(-1, -2))
    assert integrator.lower.shape == integrator.upper.shape == (1, 2, 3)

    old_vc = model.vc.reshape(1, 3, 3).clone()
    ve = torch.tensor([[0.0, 0.1, -0.2]], dtype=DTYPE)
    intra = torch.tensor([[1.0e-10, 0.0, -1.0e-10]], dtype=DTYPE)
    integrator.step(model, 0.1, ve=ve, intra=intra)
    assert model.vc.shape == (1, 3, 3)
    assert model.v.shape == (1, 3)
    assert model.i_membrane.shape == (1, 3)
    assert torch.isfinite(model.vc).all()

    gtot = torch.full((1, 3), 0.02, dtype=DTYPE) * integrator.area
    itot = 0.02 * (old_vc[..., 0] + 6.0) * integrator.area
    d = gtot * old_vc[..., 0] - itot + intra
    main = integrator.maind.clone()
    main[..., 0, 0] += gtot
    main[..., 1, 1] += gtot
    main[..., 0, 1] -= gtot
    main[..., 1, 0] -= gtot
    rhs = assemble_rhs(old_vc, integrator.c_rad, d, integrator.xg[..., -1], ve)
    dense_result = _dense_block_solve(integrator.lower, main, integrator.upper, rhs)
    assert torch.allclose(model.vc, dense_result)


def test_block_implicit_init_v_and_detach_reset_public_state():
    model = CableModel(
        shape=(2, 3), v_init=torch.tensor([-3.0, -2.0, -1.0]), n_layers=2
    )
    mech = LinearMechanism()
    integrator = _bwd_euler_bt(model, mech, method="thomas", imem=True)
    model.v.fill_(10.0)
    model.vc.fill_(20.0)
    integrator.init_v(model)
    assert torch.equal(model.v, model.expanded_v_init())
    assert torch.equal(model.vc[..., 0], model.expanded_v_init())
    assert torch.equal(model.vc[..., 1:], torch.zeros_like(model.vc[..., 1:]))
    assert torch.equal(model.i_membrane, torch.zeros_like(model.v))
    integrator.detach(model)
    assert mech.detached
    assert not model.v.requires_grad and not model.vc.requires_grad
