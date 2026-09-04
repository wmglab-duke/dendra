"""Functional state and transition contracts for ``ExtCellAxon``."""

from __future__ import annotations

import math
from functools import partial

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.implicit import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas

DT = 0.01
DTYPE = torch.float64
STEPS = 3
PAS_G = "integrator.mech.mechanisms.pas.g_param"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE,
        reason="ExtCellAxon CPU execution requires dendra-solvers",
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _model(*, batch_calls=(), n_comp=4):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.ExtCellAxon(
            diameters=[5.0, 8.0],
            n_comp=n_comp,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_bt(method="thomas", imem=False),
        )
        if n_comp == 4:
            model.dx.copy_(
                torch.tensor(
                    [[8.0, 9.0, 10.0, 11.0], [7.0, 9.0, 12.0, 14.0]],
                    dtype=DTYPE,
                )
            )
        else:
            model.dx.copy_(
                torch.linspace(
                    7.0,
                    11.0,
                    model.dx.numel(),
                    dtype=DTYPE,
                ).reshape_as(model.dx)
            )
        model.xraxial.copy_(
            torch.linspace(
                1.5,
                3.0,
                model.xraxial.numel(),
                dtype=DTYPE,
            ).reshape_as(model.xraxial)
        )
        model.xc.copy_(
            torch.linspace(
                0.08,
                0.20,
                model.xc.numel(),
                dtype=DTYPE,
            ).reshape_as(model.xc)
        )
        model.xg.copy_(
            torch.linspace(
                1.0e-4,
                4.0e-4,
                model.xg.numel(),
                dtype=DTYPE,
            ).reshape_as(model.xg)
        )
        model.insert(pas, g=1.0e-3, e=-70.0)
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()

        # Exercise genuine block carry. A transition that reconstructs ``vc``
        # from membrane voltage alone must not pass this fixture.
        inner_shell = torch.linspace(
            -2.0,
            1.0,
            model.v.numel(),
            dtype=DTYPE,
        ).reshape(model.shape)
        outer_shell = torch.linspace(
            0.5,
            -1.5,
            model.v.numel(),
            dtype=DTYPE,
        ).reshape(model.shape)
        with torch.no_grad():
            model.vc[..., 1].copy_(inner_shell)
            model.vc[..., 2].copy_(outer_shell)
            model.vc[..., 0].copy_(model.v + inner_shell)
    return model


def _drives(model, *, steps=STEPS):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -1.5,
        2.0,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -2.0e-9,
        3.0e-9,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _initialize_imperative(model):
    dt = torch.as_tensor(DT, dtype=model.dtype(), device=model.device())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    return dt


def _imperative_step(model, dt, ve, intra):
    model.integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def _assert_tree_close(actual, expected, *, rtol=2.0e-10, atol=2.0e-11):
    actual_with_paths, actual_spec = torch.utils._pytree.tree_flatten_with_path(actual)
    expected_with_paths, expected_spec = torch.utils._pytree.tree_flatten_with_path(
        expected
    )
    assert actual_spec == expected_spec
    for (actual_path, actual_leaf), (expected_path, expected_leaf) in zip(
        actual_with_paths,
        expected_with_paths,
        strict=True,
    ):
        assert actual_path == expected_path
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
            msg=lambda message: f"state leaf {actual_path}: {message}",
        )


def _assert_voltage_relation(state):
    voltage = state["integrator"]["v"]
    circuit_voltage = state["integrator"]["vc"]
    torch.testing.assert_close(
        voltage,
        circuit_voltage[..., 0] - circuit_voltage[..., 1],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def _transform_case():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps=1)
    step_input = dn.func.StepInput(ve=ve[0], intra=intra[0])

    def response(local_xg, local_g):
        parameters = {**tensors.parameters, PAS_G: local_g}
        constants = {**tensors.constants, "xg": local_xg}
        return functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
            step_input,
        )[0]["integrator"]["v"]

    return model, response, tensors.constants["xg"], tensors.parameters[PAS_G]


def test_extcell_state_and_prepared_block_workspace_schema():
    from dendra_solvers import solve_bt

    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    assert functional._transition.solver is solve_bt

    assert set(tensors.state) == {"integrator", "clock", "control"}
    assert set(tensors.state["integrator"]) == {"v", "vc"}
    assert set(tensors.state["clock"]) == {"t"}
    assert set(tensors.state["control"]) == {"duration_remainder"}
    assert tensors.state["integrator"]["v"].shape == model.shape
    assert tensors.state["integrator"]["vc"].shape == (*model.shape, 3)
    assert torch.count_nonzero(tensors.state["integrator"]["vc"][..., 1:])
    _assert_voltage_relation(tensors.state)

    assert tensors.constants["xraxial"].shape == (*model.core_shape(), 2)
    assert tensors.constants["xc"].shape == (*model.core_shape(), 2)
    assert tensors.constants["xg"].shape == (*model.core_shape(), 2)

    batch = math.prod(model.shape[:-1])
    compartments = model.shape[-1]
    workspace_shapes = {
        "maind": (batch, compartments, 3, 3),
        "lower": (batch, compartments - 1, 3),
        "upper": (batch, compartments - 1, 3),
        "area": (batch, compartments),
        "cm_dt": (batch, compartments),
        "xc_dt": (batch, compartments, 2),
        "c_rad": (batch, compartments, 3),
        "xg": (batch, compartments, 2),
    }
    workspace = prepared.values["integrator"]
    assert workspace["dt"].shape == ()
    for name, shape in workspace_shapes.items():
        assert workspace[name].shape == shape
        assert workspace[name].dtype == DTYPE
        assert workspace[name].device.type == "cpu"


def test_extcell_requires_exact_v_and_vc_integrator_state_contract():
    model = _model()
    model.integrator.v_vars = ["v"]

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"must declare exactly the \('v', 'vc'\) integrator state contract",
    ):
        dn.func.make_functional(model, dt=DT)


def test_extcell_single_compartment_has_valid_sealed_end_workspace_and_step():
    source = _model(n_comp=1)
    reference = _model(n_comp=1)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    batch = math.prod(source.shape[:-1])
    assert prepared.values["integrator"]["lower"].shape == (batch, 0, 3)
    assert prepared.values["integrator"]["upper"].shape == (batch, 0, 3)
    assert prepared.values["integrator"]["maind"].shape == (batch, 1, 3, 3)

    ve, intra = _drives(source, steps=1)
    state, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(ve=ve[0], intra=intra[0]),
    )
    dt = _initialize_imperative(reference)
    _imperative_step(reference, dt, ve[0], intra[0])
    _assert_tree_close(state, functional.extract(reference).state)
    _assert_voltage_relation(state)


@pytest.mark.parametrize("batch_calls", [(), (2,)])
def test_extcell_step_and_rollout_match_every_imperative_leaf(batch_calls):
    source = _model(batch_calls=batch_calls)
    reference = _model(batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source)
    dt = _initialize_imperative(reference)
    initial_circuit_voltage = tensors.state["integrator"]["vc"].clone()
    state = tensors.state

    for index in range(STEPS):
        state, auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(reference, dt, ve[index], intra[index])
        _assert_tree_close(state, functional.extract(reference).state)
        _assert_voltage_relation(state)
        torch.testing.assert_close(
            auxiliary["v"],
            state["integrator"]["v"],
            rtol=0.0,
            atol=0.0,
        )

    assert not torch.equal(state["integrator"]["vc"], initial_circuit_voltage)

    rolled, auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_tree_close(rolled, state)
    _assert_voltage_relation(rolled)
    torch.testing.assert_close(
        auxiliary["v"],
        rolled["integrator"]["v"],
        rtol=0.0,
        atol=0.0,
    )

    if batch_calls:
        assert source.shape == (2, *source.core_shape())
        assert rolled["integrator"]["vc"].shape == (*source.shape, 3)


def test_extcell_physical_and_mechanism_jacobians_agree_and_are_nonzero():
    model, response, xg, g = _transform_case()

    reverse = torch.func.jacrev(response, argnums=(0, 1))(xg, g)
    forward = torch.func.jacfwd(response, argnums=(0, 1))(xg, g)

    assert reverse[0].shape == (*model.shape, *xg.shape)
    assert reverse[1].shape == model.shape
    for reverse_value, forward_value in zip(reverse, forward, strict=True):
        torch.testing.assert_close(
            reverse_value,
            forward_value,
            rtol=2.0e-10,
            atol=2.0e-11,
        )
        assert torch.isfinite(reverse_value).all()
        assert torch.isfinite(forward_value).all()
        assert torch.count_nonzero(reverse_value)
        assert torch.count_nonzero(forward_value)


def test_extcell_physical_and_mechanism_vmap_including_zero_lanes():
    model, response, xg, g = _transform_case()
    xg_lanes = xg.unsqueeze(0) * xg.new_tensor([0.8, 1.0, 1.2]).reshape(3, 1, 1, 1)
    g_lanes = g + g.new_tensor([-2.0e-4, 0.0, 3.0e-4])
    expected = torch.stack(
        [
            response(local_xg, local_g)
            for local_xg, local_g in zip(xg_lanes, g_lanes, strict=True)
        ]
    )

    torch.testing.assert_close(
        torch.vmap(response)(xg_lanes, g_lanes),
        expected,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    empty = torch.vmap(response)(
        xg.new_empty((0, *xg.shape)),
        g.new_empty((0, *g.shape)),
    )
    assert empty.shape == (0, *model.shape)


def test_extcell_compile_jacrev_composes_over_block_and_mechanism_parameters():
    _model_value, response, xg, g = _transform_case()
    transformed = torch.func.jacrev(response, argnums=(0, 1))
    expected = transformed(xg, g)

    with torch_compiler_warning_context():
        compiled = torch.compile(transformed, backend="eager", fullgraph=True)
        actual = compiled(xg, g)

    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-10,
            atol=2.0e-11,
        )


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_extcell_host_runners_match_direct_rollout(runner_name):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model)
    inputs = dn.func.RolloutInput(ve=ve, intra=intra)
    expected_state, expected_auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        inputs,
    )
    step = partial(functional.step, tensors.parameters, prepared)

    if runner_name == "run":
        actual_state, actual_auxiliary = dn.func.run(
            functional,
            step,
            tensors.state,
            inputs,
        )
    else:
        runner = getattr(dn.func, runner_name)
        actual_state, actual_auxiliary = runner(
            functional,
            step,
            tensors.state,
            STEPS * DT,
            2,
            inputs,
        )

    _assert_tree_close(actual_state, expected_state)
    _assert_tree_close(actual_auxiliary, expected_auxiliary)
    _assert_voltage_relation(actual_state)


def _runner_gradient_case(functional, tensors, ve, intra, *, checkpointed):
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    constants = {
        name: value.detach().clone() for name, value in tensors.constants.items()
    }
    state = torch.utils._pytree.tree_map(
        lambda value: value.detach().clone(),
        tensors.state,
    )
    parameters[PAS_G].requires_grad_()
    constants["xg"].requires_grad_()
    state["integrator"]["v"].requires_grad_()
    state["integrator"]["vc"].requires_grad_()
    ve = ve.detach().clone().requires_grad_()
    intra = intra.detach().clone().requires_grad_()
    targets = (
        parameters[PAS_G],
        constants["xg"],
        state["integrator"]["v"],
        state["integrator"]["vc"],
        ve,
        intra,
    )

    prepared = functional.prepare(parameters, constants)
    step = partial(functional.step, parameters, prepared)
    runner = dn.func.longrun_checkpointed if checkpointed else dn.func.longrun
    final, auxiliary = runner(
        functional,
        step,
        state,
        STEPS * DT,
        2,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    loss = (
        final["integrator"]["v"].square().mean()
        + 0.01 * final["integrator"]["vc"].square().mean()
        + 0.001 * auxiliary["v"].sin().mean()
    )
    gradients = torch.autograd.grad(loss, targets)
    return final, auxiliary, loss, gradients


def test_extcell_checkpointed_runner_matches_ordinary_block_bptt():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model)

    ordinary = _runner_gradient_case(
        functional,
        tensors,
        ve,
        intra,
        checkpointed=False,
    )
    checkpointed = _runner_gradient_case(
        functional,
        tensors,
        ve,
        intra,
        checkpointed=True,
    )

    _assert_tree_close(checkpointed[0], ordinary[0])
    _assert_tree_close(checkpointed[1], ordinary[1])
    torch.testing.assert_close(checkpointed[2], ordinary[2], rtol=2.0e-10, atol=2.0e-11)
    for actual, expected in zip(checkpointed[3], ordinary[3], strict=True):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual)
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


@pytest.mark.parametrize("batch_calls", [(), (2,)])
def test_extcell_commit_then_imperative_resume_is_transactional(batch_calls):
    source = _model(batch_calls=batch_calls)
    target = _model(batch_calls=batch_calls)
    reference = _model(batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=5)

    checkpoint_state, _auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve[:3], intra=intra[:3]),
    )
    functional.commit_state_(target, checkpoint_state)
    _assert_tree_close(functional.extract(target).state, checkpoint_state)

    target_dt = _initialize_imperative(target)
    for index in range(3, 5):
        _imperative_step(target, target_dt, ve[index], intra[index])
    reference_dt = _initialize_imperative(reference)
    for index in range(5):
        _imperative_step(reference, reference_dt, ve[index], intra[index])
    _assert_tree_close(
        functional.extract(target).state,
        functional.extract(reference).state,
    )

    before = functional.extract(target).state
    invalid = torch.utils._pytree.tree_map(lambda value: value.clone(), before)
    invalid["integrator"]["vc"] = invalid["integrator"]["vc"][..., :2]
    with pytest.raises(ValueError, match="state leaf 'integrator.vc'.*expected"):
        functional.commit_state_(target, invalid)
    _assert_tree_close(functional.extract(target).state, before)


def test_extcell_rejects_rebound_target_vc_layout_before_extract_or_commit():
    source = _model()
    target = _model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    target.vc = target.vc[..., :2].clone()
    voltage_before = target.v.detach().clone()
    circuit_before = target.vc.detach().clone()
    time_before = target.t.detach().clone()

    for operation in (
        lambda: functional.extract(target),
        lambda: functional.commit_state_(target, tensors.state),
    ):
        with pytest.raises(
            dn.func.FunctionalizationError,
            match="runtime-state layout does not match",
        ):
            operation()
        torch.testing.assert_close(target.v, voltage_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(target.vc, circuit_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(target.t, time_before, rtol=0.0, atol=0.0)
