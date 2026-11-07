"""Spatial plans reject changed programs but accept explicit live coefficients.

Topology and solver choices are fixed when lowering. Native volume and edge
geometry remain differentiable numerical inputs, checked here with independent
finite-volume equations and directional finite differences.
"""

from __future__ import annotations

from types import MethodType

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms._spatial import _FUNCTIONAL_SCALAR_SPD
from tests.func.test_clearance_process import (
    DT,
    NAI,
    POOL,
    PRODUCTION,
    STEPS,
    _close,
    _explicit_reference,
    _key,
    _loss,
    _runner,
)
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_material_diffusion import (
    _arguments,
    _constant_name,
    _gradients,
    _model,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _unexpected_hook(self, *args, **kwargs):
    raise AssertionError("Changed spatial hooks must be rejected before execution")


@pytest.mark.parametrize(
    ("kind", "change"),
    (
        ("path", "hook"),
        ("tree", "hook"),
        ("path", "solver"),
        ("tree", "solver"),
        ("path", "boundary"),
        ("tree", "boundary"),
        ("path", "interface_scheme"),
    ),
)
def test_spatial_program_changes_invalidate_existing_functional_plan(kind, change):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    process = model.mech.material_processes["_FunctionalDiffusion"]
    operator = next(iter(process._spatial_operators.values()))
    if change == "hook":
        operator.prepare_diffusion = MethodType(_unexpected_hook, operator)
    else:
        setattr(
            operator,
            change,
            {"solver": "dense", "boundary": "changed", "interface_scheme": "series"}[
                change
            ],
        )
    for execute in (
        lambda: functional.prepare(tensors.parameters, tensors.constants),
        lambda: functional.step(tensors.parameters, prepared, tensors.state),
    ):
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="[Ss]tructur|changed|[Cc]anonical|[Hh]ook",
        ):
            execute()


def test_raw_chemical_topology_changes_invalidate_source_and_relowering():
    model = _model("tree")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    changed = model.diff_parent_index.clone()
    children = torch.nonzero(changed >= 0, as_tuple=False).flatten()
    assert children.numel()
    changed[children[0]] = -1
    model.diff_parent_index = changed
    with pytest.raises(
        dn.func.FunctionalizationError, match="[Ss]tructur|changed|[Tt]opolog|parent"
    ):
        functional.step(tensors.parameters, prepared, tensors.state)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="[Ss]tructur|changed|[Tt]opolog|parent|[Gg]raph",
    ):
        dn.func.make_functional(model, dt=DT)


@pytest.mark.skipif(
    _FUNCTIONAL_SCALAR_SPD is None,
    reason="dendra-solvers pure scalar SPD facade is required",
)
def test_functional_material_spd_solver_matches_native_without_changing_source():
    model = _model("path", solver="spd")
    native = _model("path", solver="spd")
    before = _module_snapshot(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    actual, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state
    )
    dt = tensors.constants["dt"]
    native.integrator._initialize(native, dt, force=True, compile_scope="population")
    native.integrator.step(native, dt)
    native.t = native.t + dt
    torch.testing.assert_close(
        actual, functional.extract(native).state, rtol=3e-10, atol=3e-11
    )
    _assert_module_snapshot(model, before)


def _chemical_loss(recorded):
    return 0.2 * recorded[POOL].square().mean() + 0.01 * recorded[NAI].square().mean()


def _independent_dense_recurrence(model, initial, parameters, volume, edge_factor, dt):
    """Storage-order finite-volume equations, without a process/operator cache."""
    pool, sodium = initial[POOL].clone(), initial[NAI].clone()
    parents, children = model.material_edge_index
    parents, children = parents.long(), children.long()
    count = pool.shape[-1]
    volume = torch.broadcast_to(volume, pool.shape)
    factor = edge_factor.index_select(-1, children)
    diffusion = parameters[
        _key(parameters, "material_processes._FunctionalDiffusion.D_param")
    ]
    conductance = diffusion * factor
    incidence = pool.new_zeros((children.numel(), count))
    indices = torch.arange(children.numel(), device=pool.device)
    incidence[indices, parents] = 1.0
    incidence[indices, children] = -1.0
    laplace = torch.einsum("...e,ei,ej->...ij", conductance, incidence, incidence)
    storage = volume / dt
    rate = parameters[
        _key(parameters, "material_processes._FunctionalDiffusion.rate_param")
    ]
    target = parameters[
        _key(parameters, "material_processes._FunctionalDiffusion.target_param")
    ]
    reaction = volume * rate
    matrix = torch.diag_embed(storage) + laplace
    pool_matrix = matrix + torch.diag_embed(reaction)
    pools, ions = [pool], [sodium]
    for _ in range(STEPS):
        source = pool + dt * PRODUCTION * (1 + 0.01 * sodium)
        pool = torch.linalg.solve(
            pool_matrix, (storage * source + reaction * target).unsqueeze(-1)
        ).squeeze(-1)
        sodium = torch.linalg.solve(matrix, (storage * sodium).unsqueeze(-1)).squeeze(
            -1
        )
        pools.append(pool)
        ions.append(sodium)
    return {POOL: torch.stack(pools), NAI: torch.stack(ions)}


def test_native_chemical_geometry_is_live_differentiable_and_not_a_cached_snapshot(
    cpu_threads,
):
    model = _model("tree")
    source = _module_snapshot(model)
    runner = _runner(model)
    parameters, initialization, inputs, _targets = _arguments(runner, model)
    baseline = dict(
        runner(parameters=parameters, initialization=initialization, inputs=inputs)[1][
            "callbacks"
        ]["trace"]
    )
    baseline_pool = baseline[POOL].detach().clone()
    volume = runner.constants[
        _constant_name(runner.constants, "volume_i", "_mp_volume_i")
    ]
    factor = runner.constants[
        _constant_name(runner.constants, "diff_geom_um", "_mp_diff_geom_um")
    ]
    with torch.no_grad():
        volume.mul_(1.18)
        factor.mul_(0.87)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    actual = dict(
        runner(parameters=parameters, initialization=initialization, inputs=inputs)[1][
            "callbacks"
        ]["trace"]
    )
    gradients = dict(
        zip(
            targets,
            torch.autograd.grad(
                _loss(actual), tuple(targets.values()), retain_graph=True
            ),
            strict=True,
        )
    )
    for name, value in gradients.items():
        assert torch.isfinite(value).all() and torch.count_nonzero(value), name
    physical_gradients = torch.autograd.grad(_chemical_loss(actual), (volume, factor))
    rp, ri, rx, rt = _arguments(runner, model)
    expected = _explicit_reference(runner, rp, ri, rx)
    expected_recorded = dict(expected[1]["callbacks"]["trace"])
    _close(actual, expected_recorded)
    _close(gradients, _gradients(expected_recorded, rt))
    independent = _independent_dense_recurrence(
        model,
        {name: actual[name][0].detach() for name in (POOL, NAI)},
        parameters,
        volume,
        factor,
        runner.constants["dt"],
    )
    _close({name: actual[name] for name in (POOL, NAI)}, independent)
    independent_gradients = torch.autograd.grad(
        _chemical_loss(independent), (volume, factor)
    )
    for actual_gradient, oracle_gradient in zip(
        physical_gradients, independent_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, oracle_gradient, rtol=3e-7, atol=2e-9
        )
    assert (actual[POOL] - baseline_pool).abs().max() > 1e-12
    for label, leaf, derivative in zip(
        ("volume", "edge_factor"), (volume, factor), physical_gradients, strict=True
    ):
        original = leaf.detach().clone()
        step = 1e-4
        losses = []
        for sign in (1, -1):
            with torch.no_grad():
                leaf.copy_(original * (1 + sign * step))
            recorded = dict(
                runner(
                    parameters=parameters, initialization=initialization, inputs=inputs
                )[1]["callbacks"]["trace"]
            )
            losses.append(_chemical_loss(recorded).detach())
        with torch.no_grad():
            leaf.copy_(original)
        torch.testing.assert_close(
            (derivative * original).sum(),
            (losses[0] - losses[1]) / (2 * step),
            rtol=4e-4,
            atol=1e-9,
            msg=lambda message: f"{label}: {message}",
        )
    _assert_module_snapshot(model, source)
