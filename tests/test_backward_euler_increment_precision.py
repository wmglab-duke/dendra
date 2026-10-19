"""Independent physical-circuit checks for point and unbranched BE increments.

The dense oracle is authored in absolute physical units, independently of the
candidate's normalized bands and increment RHS. Critical cancellation fixtures
also have exact analytic references; an absolute dense solve is not an accurate
oracle for updates below the old voltage's ULP.
"""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.integrators import implicit
from dendra.models.integrators.implicit import (
    _bwd_euler_sc,
    _bwd_euler_sc_multi,
    _bwd_euler_ub,
)
from dendra.models.mod import pas

from .test_integrator_explicit_implicit import CableModel, LinearMechanism

DT = 0.025
KINDS = ("point", "packed", "thomas", "pcr")
DTYPES = (torch.float32, torch.float64)


def _hh(v, channels):
    gnabar, gkbar, gl, ena, ek, el = channels
    # A held-fixed physical HH current frame isolates voltage assembly from
    # already-tested gate kinetics. All six channel inputs remain independent.
    m = torch.linspace(0.15, 0.35, v.shape[-1], device=v.device, dtype=v.dtype)
    h = torch.full_like(v, 0.6)
    n = torch.linspace(0.2, 0.45, v.shape[-1], device=v.device, dtype=v.dtype)
    sodium = gnabar * m**3 * h
    potassium = gkbar * n**4
    return sodium + potassium + gl, sodium * ena + potassium * ek + gl * el


class _Step(torch.nn.Module):
    def __init__(
        self, kind, *, k=5, batch=2, dtype=torch.float64, device="cpu", observed=False
    ):
        super().__init__()
        model = CableModel(shape=(batch, k)).to(device=device, dtype=dtype)
        if kind == "point":
            integrator = _bwd_euler_sc(model, LinearMechanism(), imem=observed)
        elif kind == "packed":
            integrator = _bwd_euler_sc_multi(
                model, LinearMechanism(), write_back=False, imem=observed
            )
        else:
            integrator = _bwd_euler_ub(
                model, LinearMechanism(), method=kind, imem=observed
            )
            integrator._select_solver(torch.device(device))
        self.integrator = integrator.to(device=device, dtype=dtype)
        self.solver = (
            None
            if kind in {"point", "packed"}
            else integrator._solve
            if kind == "spd"
            else integrator._functional_solver()
        )
        self.observed = observed
        self.kind, self.k, self.batch = kind, k, batch

    def forward(self, v, conductance, source, ve, intra):
        current = conductance * v - source if self.observed else None
        return self.integrator._voltage_update(
            v,
            DT,
            current,
            conductance,
            ve=ve,
            intra=intra,
            solver=self.solver,
            membrane_source=source,
        )

    def response(
        self, v, amplitude, gnabar, gkbar, gl, ena, ek, el, cm, area, rhoa_scale, ve
    ):
        conductance, source = _hh(v, (gnabar, gkbar, gl, ena, ek, el))
        intra = amplitude * torch.linspace(
            -1e-9, 2e-9, self.k, dtype=v.dtype, device=v.device
        )
        if self.kind in {"point", "packed"}:
            workspace = self.integrator._prepare_workspace(DT, cm=cm, area=area)
        else:
            edge = torch.linspace(
                1.25e-8, 2.5e-8, self.k - 1, dtype=v.dtype, device=v.device
            ).expand_as(v[..., 1:])
            workspace = self.integrator._prepare_workspace(
                DT,
                cm=cm,
                area=area,
                edge_conductance=edge / rhoa_scale,
            )
        return torch.func.functional_call(
            self,
            {"integrator." + name: value for name, value in workspace.items()},
            (v, conductance, source, ve, intra),
        )[0]


def _arguments(dtype, *, k=5, batch=2, device="cpu"):
    v = torch.linspace(-68, -54, batch * k, dtype=dtype, device=device).reshape(
        batch, k
    )
    scalars = tuple(
        torch.tensor(value, dtype=dtype, device=device)
        for value in (0.7, 0.12, 0.036, 0.0003, 50.0, -77.0, -54.3)
    )
    cm = torch.linspace(0.6, 1.8, batch * k, dtype=dtype, device=device).reshape(
        batch, k
    )
    area = torch.linspace(
        0.7e-5, 1.3e-5, batch * k, dtype=dtype, device=device
    ).reshape(batch, k)
    rhoa = torch.tensor(1.2, dtype=dtype, device=device)
    ve = torch.linspace(-1.5, 2.0, batch * k, dtype=dtype, device=device).reshape(
        batch, k
    )
    return (v, *scalars, cm, area, rhoa, ve)


def _physical(kind, *args):
    v, amplitude, *tail = args
    channels, (cm, area, rhoa_scale, ve) = tail[:6], tail[6:]
    conductance, source = _hh(v, channels)
    capacitance = 1e-6 * cm * area
    capacitive = capacitance / (DT * 1e-3)
    rhs = capacitive * v + source * area
    rhs = rhs + amplitude * torch.linspace(
        -1e-9, 2e-9, v.shape[-1], dtype=v.dtype, device=v.device
    )
    if kind in {"point", "packed"} or v.shape[-1] == 1:
        return rhs / (capacitive + conductance * area)
    k = v.shape[-1]
    incidence = torch.eye(k, dtype=v.dtype, device=v.device)[:, 1:]
    incidence = incidence - torch.eye(k, dtype=v.dtype, device=v.device)[:, :-1]
    edge = (
        torch.linspace(1.25e-8, 2.5e-8, k - 1, dtype=v.dtype, device=v.device)
        / rhoa_scale
    )
    laplacian = torch.einsum("ie,e,je->ij", incidence, edge, incidence)
    matrix = torch.diag_embed(capacitive + conductance * area) + laplacian
    rhs = rhs - (laplacian @ ve.unsqueeze(-1)).squeeze(-1)
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)


def _gradient_close(actual, expected, dtype):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = torch.linalg.vector_norm(actual.double() - expected.double())
    scale = torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    assert error <= (2e-3 if dtype == torch.float32 else 2e-9) * scale, (error, scale)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", (1, 5))
def test_physical_dense_voltage_stimulus_all_channel_and_geometry_gradients(
    kind, dtype, k
):
    step = _Step(kind, k=k, dtype=dtype)
    arguments = tuple(value.requires_grad_() for value in _arguments(dtype, k=k))
    snapshot = tuple((value._version, value.detach().clone()) for value in arguments)
    references = tuple(value.detach().double().requires_grad_() for value in arguments)
    actual, expected = step.response(*arguments), _physical(kind, *references)
    torch.testing.assert_close(
        actual.double(),
        expected,
        rtol=2e-5 if dtype == torch.float32 else 2e-12,
        atol=2e-5 if dtype == torch.float32 else 2e-12,
    )
    gradients = torch.autograd.grad(
        actual.square().mean(), arguments, allow_unused=True
    )
    targets = torch.autograd.grad(
        expected.square().mean(), references, allow_unused=True
    )
    for index, (actual_gradient, target) in enumerate(
        zip(gradients, targets, strict=True)
    ):
        if target is None:
            if actual_gradient is not None:
                # K=1 has no physical edges. The UB empty-edge workspace may
                # retain a connected zero dependency on rhoa/ve, whereas the
                # independent isolated-node oracle never uses those inputs.
                assert kind in {"thomas", "pcr"} and k == 1 and index in {10, 11}
                torch.testing.assert_close(
                    actual_gradient, torch.zeros_like(actual_gradient), rtol=0, atol=0
                )
        else:
            assert actual_gradient is not None
            _gradient_close(actual_gradient, target, dtype)
    for value, (version, saved) in zip(arguments, snapshot, strict=True):
        assert value._version == version
        torch.testing.assert_close(value, saved, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_unforced_uniform_voltage_is_exactly_stationary(kind, dtype):
    step = _Step(kind, dtype=dtype)
    args = list(_arguments(dtype))
    args[0] = torch.full_like(args[0], -62.0)
    # All currents vanish independently of voltage and every compartment has
    # different physical C/area. Axial constant mode must not drift.
    for index in (2, 3, 4):
        args[index] = torch.zeros_like(args[index])
    args[1] = torch.zeros_like(args[1])
    args[-1] = torch.zeros_like(args[-1])
    initial = args[0].clone()
    for _ in range(16):
        args[0] = step.response(*args)
        torch.testing.assert_close(args[0], initial, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_nonzero_affine_membrane_equilibrium_is_bitwise_stationary(kind, dtype):
    step = _Step(kind, dtype=dtype)
    args = _arguments(dtype)
    cm, area = args[8:10]
    if kind in {"point", "packed"}:
        workspace = step.integrator._prepare_workspace(DT, cm=cm, area=area)
    else:
        workspace = step.integrator._prepare_workspace(
            DT, cm=cm, area=area, edge_conductance=torch.full((2, 4), 2e-8, dtype=dtype)
        )
    step.integrator._install_prepared_workspace(workspace)
    voltage = torch.full((2, 5), -62.0, dtype=dtype)
    initial = voltage.clone()
    conductance = torch.linspace(0.0001, 0.0004, 5, dtype=dtype).expand_as(voltage)
    source = conductance * voltage
    for _ in range(16):
        voltage, _ = step(
            voltage,
            conductance,
            source,
            torch.zeros_like(voltage),
            torch.zeros_like(voltage),
        )
        torch.testing.assert_close(voltage, initial, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ("point", "packed"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_point_combines_sub_ulp_membrane_and_injected_updates_before_publication(
    kind, dtype
):
    step = _Step(kind, k=1, batch=1, dtype=dtype, observed=True)
    integrator = step.integrator
    integrator.cmdt = torch.tensor(0.75, dtype=dtype)
    integrator.area = torch.tensor(2.0, dtype=dtype)
    v = torch.tensor([[64.0]], dtype=dtype)
    ulp = torch.nextafter(v, torch.full_like(v, float("inf"))) - v
    ionic = -0.4 * ulp
    intra = 0.8 * ulp
    result, current = integrator._voltage_update(
        v, DT, ionic, torch.full_like(v, 0.25), intra=intra
    )
    expected = v + ((-ionic) + intra / integrator.area)
    torch.testing.assert_close(expected, v + ulp, rtol=0, atol=0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    # Observed membrane current is defined from the *published* voltage step.
    expected_current = (
        integrator.cmdt * (result - v) + (ionic + 0.25 * (result - v))
    ) * integrator.area
    torch.testing.assert_close(current, expected_current, rtol=0, atol=0)


def _weak_current(kind, dtype, k):
    step = _Step(kind, k=k, batch=1, dtype=dtype)
    integrator = step.integrator
    v = torch.full((1, k), 64.0, dtype=dtype)
    # Exact normalized C/dt=1 and dt*g_edge/C=.5. This explicitly authored
    # current must not be reconstructed by subtracting two large G*v terms.
    integrator.cm_inv = torch.full_like(v, 40000.0)
    integrator.scale = torch.full_like(v, 40000.0)
    integrator.lower = torch.full((1, k - 1), -0.5, dtype=dtype)
    integrator.upper = integrator.lower.clone()
    integrator.g_edge_Cinv = torch.full((1, k - 1), 20000.0, dtype=dtype)
    integrator.g_edge_Cinv_right = integrator.g_edge_Cinv.clone()
    diagonal = torch.zeros_like(v)
    if k > 1:
        diagonal[:, :-1] -= 20000
        diagonal[:, 1:] -= 20000
    integrator.diag_base = diagonal

    def response(initial, ionic, conductance, injection):
        return integrator._voltage_update(
            initial,
            DT,
            ionic.expand_as(initial),
            conductance.expand_as(initial),
            intra=injection,
            solver=step.solver,
        )[0]

    ionic = torch.tensor(1e-12 if dtype == torch.float32 else 1e-24, dtype=dtype)
    args = (v, ionic, torch.tensor(0.25, dtype=dtype), torch.zeros_like(v))
    return response, args


@pytest.mark.parametrize("kind", ("thomas", "pcr"))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("k", (1, 3))
def test_authored_tiny_current_retains_conductance_sensitivity_below_voltage_ulp(
    kind, dtype, k
):
    response, args = _weak_current(kind, dtype, k)
    args = tuple(value.requires_grad_() for value in args)
    result = response(*args)
    # Publication legitimately rounds the sub-ULP update away, but its
    # conductance sensitivity is nonzero and must survive AD. The historical
    # absolute RHS cancels its G*v derivatives at the rounded ~64mV solution.
    torch.testing.assert_close(result, args[0], rtol=0, atol=0)
    gradients = torch.autograd.grad(result.sum(), args)
    expected = (
        torch.ones_like(args[0]),
        torch.full_like(args[1], -k / 1.25),
        k * args[1].detach() / 1.25**2,
        torch.full_like(args[3], 1 / 1.25),
    )
    assert gradients[2].item() != 0
    for actual, target in zip(gradients, expected, strict=True):
        torch.testing.assert_close(
            actual, target, rtol=32 * torch.finfo(dtype).eps, atol=0
        )


@pytest.mark.slow
def test_cpp_compiled_authored_tiny_current_retains_sub_ulp_conductance_gradient():
    response, args = _weak_current("thomas", torch.float32, 3)
    args = tuple(value.requires_grad_() for value in args)
    compiled = torch.compile(
        response, fullgraph=True, backend="inductor", options={"cpp_wrapper": True}
    )
    result = compiled(*args)
    torch.testing.assert_close(result, args[0], rtol=0, atol=0)
    conductance_gradient = torch.autograd.grad(result.sum(), args[2])[0]
    expected = 3 * args[1].detach() / 1.25**2
    assert conductance_gradient.item() != 0
    torch.testing.assert_close(
        conductance_gradient, expected, rtol=32 * torch.finfo(torch.float32).eps, atol=0
    )


@pytest.mark.parametrize("kind", KINDS)
def test_vmap_empty_lanes_jvp_and_second_derivatives_match_physical_equations(kind):
    step = _Step(kind)
    arguments = _arguments(torch.float64)

    def physical(*values):
        return _physical(kind, *values)

    lanes = tuple(torch.stack((value, value * 1.01)) for value in arguments)
    actual, expected = torch.vmap(step.response)(*lanes), torch.vmap(physical)(*lanes)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    # Exercise empty lanes at the voltage-update boundary, not through HH's
    # known scalar/empty-lane broadcasting limitation. Construct the physical
    # current frame and geometry once outside vmap, then map only node fields.
    voltage, amplitude = arguments[:2]
    conductance, source = _hh(voltage, arguments[2:8])
    injection = amplitude * torch.linspace(-1e-9, 2e-9, 5, dtype=voltage.dtype)
    injection = injection.expand_as(voltage)
    workspace_arguments = {"cm": arguments[8], "area": arguments[9]}
    if kind not in {"point", "packed"}:
        edge = torch.linspace(1.25e-8, 2.5e-8, 4, dtype=voltage.dtype)
        workspace_arguments["edge_conductance"] = (
            edge.expand_as(voltage[..., 1:]) / arguments[10]
        )
    workspace = step.integrator._prepare_workspace(DT, **workspace_arguments)

    def current_frame(*fields):
        return torch.func.functional_call(
            step,
            {"integrator." + name: value for name, value in workspace.items()},
            fields,
        )[0]

    frame = (voltage, conductance, source, arguments[-1], injection)
    empty = tuple(value.new_empty((0, *value.shape)) for value in frame)
    assert torch.vmap(current_frame)(*empty).shape == (0, 2, 5)
    tangents = tuple(torch.full_like(value, 0.0125) for value in arguments)
    for actual, expected in zip(
        torch.func.jvp(step.response, arguments, tangents),
        torch.func.jvp(physical, arguments, tangents),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
    actual_arguments = tuple(value.requires_grad_() for value in arguments)
    expected_arguments = tuple(
        value.detach().clone().requires_grad_() for value in arguments
    )
    first = torch.autograd.grad(
        step.response(*actual_arguments).square().mean(),
        actual_arguments,
        create_graph=True,
        allow_unused=True,
    )
    first_reference = torch.autograd.grad(
        physical(*expected_arguments).square().mean(),
        expected_arguments,
        create_graph=True,
        allow_unused=True,
    )
    # A directional second derivative avoids a large Hessian matrix.
    total = sum(
        (gradient * tangent).sum()
        for gradient, tangent in zip(first, tangents, strict=True)
        if gradient is not None
    )
    total_reference = sum(
        (gradient * tangent).sum()
        for gradient, tangent in zip(first_reference, tangents, strict=True)
        if gradient is not None
    )
    second = torch.autograd.grad(total, actual_arguments, allow_unused=True)
    expected_second = torch.autograd.grad(
        total_reference, expected_arguments, allow_unused=True
    )
    for actual, expected in zip(second, expected_second, strict=True):
        if expected is None:
            assert actual is None
        else:
            _gradient_close(actual, expected, torch.float64)


@pytest.mark.parametrize("kind", KINDS)
def test_chain_reversal_or_point_permutation_preserves_physical_solution(kind):
    step = _Step(kind)
    arguments = list(_arguments(torch.float64))
    permutation = torch.tensor([4, 3, 2, 1, 0])
    # Reversal preserves a path; arbitrary graph permutations are not a UB API.
    # Current-frame and stimulus patterns are authored in local position, so
    # check permutation at the lower numerical interface with explicit fields.
    voltage = arguments[0]
    conductance = torch.linspace(0.001, 0.003, 5).double().expand_as(voltage)
    source = conductance * -60
    intra = torch.linspace(-1e-9, 2e-9, 5).double().expand_as(voltage)
    area, cm = arguments[9], arguments[8]
    edge = torch.full_like(voltage[..., 1:], 2e-8)

    def invoke(v, g, j, external, injection, cap, membrane_area):
        workspace = (
            step.integrator._prepare_workspace(DT, cm=cap, area=membrane_area)
            if kind in {"point", "packed"}
            else step.integrator._prepare_workspace(
                DT, cm=cap, area=membrane_area, edge_conductance=edge
            )
        )
        return torch.func.functional_call(
            step,
            {"integrator." + name: value for name, value in workspace.items()},
            (v, g, j, external, injection),
        )[0]

    values = (voltage, conductance, source, arguments[-1], intra, cm, area)
    expected = invoke(*values).index_select(-1, permutation)
    actual = invoke(*(value.index_select(-1, permutation) for value in values))
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def _legacy_absolute(step, *args):
    """Literal old UB RHS, specifically for custom/surrogate-policy tests."""
    v, amplitude, *tail = args
    channels, (cm, area, rhoa, ve) = tail[:6], tail[6:]
    conductance, source = _hh(v, channels)
    edge = (
        torch.linspace(
            1.25e-8, 2.5e-8, step.k - 1, dtype=v.dtype, device=v.device
        ).expand_as(v[..., 1:])
        / rhoa
    )
    workspace = step.integrator._prepare_workspace(
        DT, cm=cm, area=area, edge_conductance=edge
    )
    current_source = source * workspace["scale"]
    if step.k > 1:
        difference = ve[..., 1:] - ve[..., :-1]
        left = workspace["g_edge_Cinv"] * difference
        right = workspace["g_edge_Cinv_right"] * difference
        current_source = current_source + torch.cat(
            (left[..., :1], left[..., 1:] - right[..., :-1], -right[..., -1:]), -1
        )
    injection = amplitude * torch.linspace(
        -1e-9, 2e-9, step.k, dtype=v.dtype, device=v.device
    )
    current_source = current_source + injection * workspace["cm_inv"]
    rhs = v + (DT * 1e-3) * current_source
    main = 1 - (DT * 1e-3) * (workspace["diag_base"] - conductance * workspace["scale"])
    if step.k == 1:
        return rhs / main, rhs
    arguments = (workspace["lower"], main, workspace["upper"], rhs)
    if step.integrator.use_gc_variant:
        arguments = (*arguments, step.integrator.clip_scale)
    solver = step.integrator._solve if step.solver is None else step.solver
    return solver(*arguments), rhs


@pytest.mark.parametrize("k", (1, 5))
@pytest.mark.parametrize("binding", ("explicit", "replacement"))
def test_custom_solver_binding_keeps_absolute_rhs_and_k1_bypass(k, binding):
    step = _Step("pcr", k=k)
    seen = []

    def custom(lower, main, upper, rhs):
        seen.append(rhs.detach().clone())
        # An authored solver's semantics need not be exactly A^-1 rhs. Do not
        # silently change that observable RHS/output contract by shifting it.
        return rhs + 2

    # Sharing a built-in's name/module must not authorize a different callable.
    custom.__name__ = "pcr_solve_t"
    custom.__module__ = "dendra.models.integrators.tridiag.pcr"
    if binding == "explicit":
        step.solver = custom
    else:
        step.integrator._solve = custom
        step.solver = None
    args = _arguments(torch.float64, k=k)
    actual = step.response(*args)
    expected, rhs = _legacy_absolute(step, *args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if k == 1:
        assert not seen
    else:
        assert len(seen) == 2
        torch.testing.assert_close(seen[0], rhs, rtol=0, atol=0)


@pytest.mark.skipif(
    not implicit.DENDRA_SOLVERS_AVAILABLE, reason="native clipped Thomas required"
)
@pytest.mark.parametrize("k", (1, 5))
def test_clipped_thomas_retains_legacy_primal_and_surrogate_gradients(k):
    step = _Step("thomas", k=k)
    step.integrator.use_gc_variant = True
    step.integrator.register_buffer(
        "clip_scale", torch.tensor(0.02, dtype=torch.float64)
    )
    step.solver = torch.ops.dendra_solvers.thomas_solve_t_gc
    args = tuple(value.requires_grad_() for value in _arguments(torch.float64, k=k))
    reference = tuple(value.detach().clone().requires_grad_() for value in args)
    actual = step.response(*args)
    expected, _ = _legacy_absolute(step, *reference)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual.square().mean(), args, allow_unused=True)
    targets = torch.autograd.grad(
        expected.square().mean(), reference, allow_unused=True
    )
    for gradient, target in zip(gradients, targets, strict=True):
        if target is None:
            assert gradient is None
        else:
            torch.testing.assert_close(gradient, target, rtol=0, atol=0)


@pytest.mark.skipif(
    not callable(getattr(implicit.dendra_solvers, "solve_tri_spd", None)),
    reason="native SPD facade required",
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_spd_keeps_legacy_rhs_and_matches_physical_ordinary_domain_all_eight_gradients(
    dtype,
):
    step = _Step("spd", batch=1, dtype=dtype)
    args = list(_arguments(dtype, batch=1))
    # Row normalization must be symmetric for the exact SPD specialized API.
    # Geometry is held fixed: independent per-node C/area perturbations would
    # leave that constrained symmetry manifold.
    args[8] = torch.ones_like(args[8])
    args[9] = torch.full_like(args[9], 1e-5)
    args = tuple(
        value.requires_grad_() if index < 8 else value
        for index, value in enumerate(args)
    )
    reference = tuple(
        value.detach().double().requires_grad_(index < 8)
        for index, value in enumerate(args)
    )
    actual = step.response(*args)
    legacy, _ = _legacy_absolute(step, *args)
    physical = _physical("spd", *reference)
    torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.double(),
        physical,
        rtol=2e-5 if dtype == torch.float32 else 2e-12,
        atol=2e-5 if dtype == torch.float32 else 2e-12,
    )
    gradients = torch.autograd.grad(actual.square().mean(), args[:8])
    legacy_gradients = torch.autograd.grad(legacy.square().mean(), args[:8])
    physical_gradients = torch.autograd.grad(physical.square().mean(), reference[:8])
    for gradient, previous, expected in zip(
        gradients, legacy_gradients, physical_gradients, strict=True
    ):
        torch.testing.assert_close(gradient, previous, rtol=0, atol=0)
        _gradient_close(gradient, expected, dtype)


@pytest.mark.skipif(
    not callable(getattr(implicit.dendra_solvers, "solve_tri_spd", None)),
    reason="native SPD facade required",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("diagonal_multiple", (1.0, 2.0))
def test_spd_regularized_pivots_keep_legacy_primal_and_rhs_only_gradients(
    dtype, diagonal_multiple
):
    step = _Step("spd", k=3, batch=1, dtype=dtype)
    integrator = step.integrator
    shape = (1, 3)
    integrator.scale = torch.ones(shape, dtype=dtype)
    integrator.cm_inv = torch.ones(shape, dtype=dtype)
    integrator.diag_base = torch.full(shape, diagonal_multiple * 40000.0, dtype=dtype)
    integrator.lower = torch.zeros((1, 2), dtype=dtype)
    integrator.upper = torch.zeros((1, 2), dtype=dtype)
    integrator.g_edge_Cinv = torch.zeros((1, 2), dtype=dtype)
    integrator.g_edge_Cinv_right = torch.zeros((1, 2), dtype=dtype)
    conductance = torch.zeros(shape, dtype=dtype)
    voltage = torch.tensor([[64.0, -32.0, 16.0]], dtype=dtype, requires_grad=True)
    source = torch.full(shape, 0.125, dtype=dtype, requires_grad=True)
    injection = torch.full(shape, -0.0625, dtype=dtype, requires_grad=True)
    actual, _ = integrator._voltage_update(
        voltage,
        DT,
        None,
        conductance,
        intra=injection,
        solver=step.solver,
        membrane_source=source,
    )
    # Literal historical absolute-voltage RHS. Zero/negative Cholesky pivots
    # are intentionally regularized by this facade when only RHS gradients
    # are needed; this is not an independent physical inverse oracle.
    dt_s = DT * 1e-3
    rhs = voltage + dt_s * (source * integrator.scale + injection * integrator.cm_inv)
    main = 1 - dt_s * (integrator.diag_base - conductance * integrator.scale)
    assert (main <= 1e-12).all()
    legacy = step.solver(integrator.lower, main, integrator.upper, rhs)
    torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
    torch.testing.assert_close(
        actual, rhs / 1e-12, rtol=8 * torch.finfo(dtype).eps, atol=0
    )
    weights = torch.tensor([[0.5, 1.0, 1.5]], dtype=dtype)
    inputs = (voltage, source, injection)
    gradients = torch.autograd.grad((actual * weights).sum(), inputs)
    previous = torch.autograd.grad((legacy * weights).sum(), inputs)
    expected = (weights / 1e-12, dt_s * weights / 1e-12, dt_s * weights / 1e-12)
    for gradient, old_gradient, target in zip(
        gradients, previous, expected, strict=True
    ):
        torch.testing.assert_close(gradient, old_gradient, rtol=0, atol=0)
        torch.testing.assert_close(
            gradient, target, rtol=8 * torch.finfo(dtype).eps, atol=0
        )
    # Coefficient-gradient requests must retain the facade's existing safety
    # rejection rather than silently differentiating a regularized inverse.
    with pytest.raises(RuntimeError, match="non-SPD"):
        integrator._voltage_update(
            voltage.detach(),
            DT,
            None,
            conductance.requires_grad_(),
            solver=step.solver,
            membrane_source=source.detach(),
        )


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("dtype", DTYPES)
def test_actual_point_populations_preserve_uniform_unforced_full_recordings(
    packed, dtype
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=dtype):
        left = dn.SingleCompartment(
            N=1,
            C=2,
            cm=0.75,
            dtype=dtype,
            v_init=torch.tensor([-62.0, 17.125], dtype=dtype),
        )
        left.insert(pas, g=0.0)
        if packed:
            right = dn.SingleCompartment(N=2, C=1, cm=1.35, dtype=dtype, v_init=-41.625)
            right.insert(pas, g=0.0)
            model = dn.concat_models({"left": left, "right": right})
        else:
            model = left
        model.initialize()
        model.train()
        before = model.v.detach().clone()
        functional, tensors = dn.func.make_functional(model, dt=DT)
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v",))})
        final, auxiliary = dn.func.run(
            functional.bind(tensors.parameters, prepared),
            tensors.state,
            steps=32,
            callbacks=callbacks,
        )
    trace = auxiliary["callbacks"]["trace"]["v"]
    assert trace.shape[0] == 33
    torch.testing.assert_close(trace, before.expand_as(trace), rtol=0, atol=0)
    torch.testing.assert_close(final["integrator"]["v"], before, rtol=0, atol=0)
    torch.testing.assert_close(model.v, before, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_observed_membrane_current_matches_published_physical_balance(kind):
    step = _Step(kind, observed=True)
    args = _arguments(torch.float64)
    v, amplitude, *tail = args
    channels, (cm, area, rhoa, ve) = tail[:6], tail[6:]
    conductance, source = _hh(v, channels)
    intra = amplitude * torch.linspace(-1e-9, 2e-9, 5).double()
    if kind in {"point", "packed"}:
        workspace = step.integrator._prepare_workspace(DT, cm=cm, area=area)
    else:
        edge = torch.linspace(1.25e-8, 2.5e-8, 4).double().expand_as(v[..., 1:]) / rhoa
        workspace = step.integrator._prepare_workspace(
            DT, cm=cm, area=area, edge_conductance=edge
        )
    new, current = torch.func.functional_call(
        step,
        {"integrator." + name: value for name, value in workspace.items()},
        (v, conductance, source, ve, intra),
    )
    expected = (
        (1e-6 * cm / (DT * 1e-3)) * (new - v) + conductance * new - source
    ) * area
    torch.testing.assert_close(current, expected, rtol=2e-11, atol=1e-18)


@pytest.mark.parametrize("kind", ("point", "thomas", "pcr"))
@pytest.mark.slow
def test_compiled_constant_mode_and_all_input_gradients(kind):
    step = _Step(kind, dtype=torch.float32)
    compiled = torch.compile(
        step.response, backend="inductor", fullgraph=True, options={"cpp_wrapper": True}
    )
    args = tuple(value.requires_grad_() for value in _arguments(torch.float32))
    references = tuple(value.detach().double().requires_grad_() for value in args)
    actual = compiled(*args)
    expected = _physical(kind, *references)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-5)
    gradients = torch.autograd.grad(actual.square().mean(), args, allow_unused=True)
    wanted = torch.autograd.grad(
        expected.square().mean(), references, allow_unused=True
    )
    for gradient, target in zip(gradients, wanted, strict=True):
        if target is None:
            assert gradient is None
        else:
            _gradient_close(gradient, target, torch.float32)
    rest = list(_arguments(torch.float32))
    rest[0] = torch.full_like(rest[0], -62)
    for index in (1, 2, 3, 4):
        rest[index] = torch.zeros_like(rest[index])
    rest[-1] = torch.zeros_like(rest[-1])
    torch.testing.assert_close(compiled(*rest), rest[0], rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", ("point", "thomas", "pcr"))
def test_cuda_physical_dense_primal_all_channel_and_stimulus_gradients(kind):
    step = _Step(kind, dtype=torch.float32, device="cuda")
    arguments = tuple(
        value.requires_grad_() for value in _arguments(torch.float32, device="cuda")
    )
    reference = tuple(
        value.detach().cpu().double().requires_grad_() for value in arguments
    )
    actual = step.response(*arguments)
    expected = _physical(kind, *reference)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=4e-5, atol=4e-5)
    gradients = torch.autograd.grad(
        actual.square().mean(), arguments, allow_unused=True
    )
    wanted = torch.autograd.grad(expected.square().mean(), reference, allow_unused=True)
    for gradient, target in zip(gradients, wanted, strict=True):
        if target is None:
            assert gradient is None
        else:
            _gradient_close(gradient.cpu(), target, torch.float32)
