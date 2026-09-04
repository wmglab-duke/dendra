"""Functional transition contracts for ``SingleCompartment`` populations."""

from __future__ import annotations

import copy
from functools import partial

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import hh

DT = 0.01
STEPS = 3
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _model(*, dtype=torch.float64, batch_calls=()):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=2,
            C=3,
            v_init=torch.tensor([-64.0, -59.0, -62.0], dtype=dtype),
            dtype=dtype,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(hh)
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _drives(model, steps=STEPS):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -3.0,
        2.0,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -2.0e-9,
        3.0e-9,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _imperative_step(model, ve=None, intra=None):
    """Advance the imperative integrator with an already assembled drive."""
    dt = torch.as_tensor(DT, device=model.device(), dtype=model.dtype())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    model.integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def _clone_tree(value):
    return torch.utils._pytree.tree_map(
        lambda tensor: tensor.detach().clone(),
        value,
    )


def _assert_tree_close(actual, expected, *, rtol=0.0, atol=0.0):
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
            msg=lambda message: f"tensor leaf {actual_path}: {message}",
        )


def _tensor_snapshot(value):
    return (
        id(value),
        value.untyped_storage().data_ptr(),
        value._version,
        value.detach().clone(),
    )


def _assert_tensor_snapshot(value, snapshot):
    identity, storage, version, expected = snapshot
    assert id(value) == identity
    assert value.untyped_storage().data_ptr() == storage
    assert value._version == version
    torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)


def _source_snapshot(model):
    return {
        "parameters": {
            name: _tensor_snapshot(value)
            for name, value in model.named_parameters(remove_duplicate=False)
        },
        "buffers": {
            name: _tensor_snapshot(value)
            for name, value in model.named_buffers(remove_duplicate=False)
        },
        "buf_i_list": id(model.mech._buf_i),
        "buf_i": tuple(_tensor_snapshot(value) for value in model.mech._buf_i),
        "buf_g_list": id(model.mech._buf_g),
        "buf_g": tuple(_tensor_snapshot(value) for value in model.mech._buf_g),
        "integrator_initialized": model.integrator.initialized,
        "integrator_dt": model.integrator.dt,
        "integrator_shape": model.integrator.shape,
        "compiled_kernels": dict(model.integrator._compiled_kernels),
        "caches": copy.copy(model._caches),
        "training": tuple(
            (name, module.training) for name, module in model.named_modules()
        ),
    }


def _assert_source_unchanged(model, snapshot):
    for name, value in model.named_parameters(remove_duplicate=False):
        _assert_tensor_snapshot(value, snapshot["parameters"][name])
    for name, value in model.named_buffers(remove_duplicate=False):
        _assert_tensor_snapshot(value, snapshot["buffers"][name])
    assert id(model.mech._buf_i) == snapshot["buf_i_list"]
    assert id(model.mech._buf_g) == snapshot["buf_g_list"]
    for value, expected in zip(
        model.mech._buf_i,
        snapshot["buf_i"],
        strict=True,
    ):
        _assert_tensor_snapshot(value, expected)
    for value, expected in zip(
        model.mech._buf_g,
        snapshot["buf_g"],
        strict=True,
    ):
        _assert_tensor_snapshot(value, expected)
    assert model.integrator.initialized == snapshot["integrator_initialized"]
    assert model.integrator.dt == snapshot["integrator_dt"]
    assert model.integrator.shape == snapshot["integrator_shape"]
    assert model.integrator._compiled_kernels == snapshot["compiled_kernels"]
    assert model._caches == snapshot["caches"]
    assert (
        tuple((name, module.training) for name, module in model.named_modules())
        == snapshot["training"]
    )


def _select_drives(ve, intra, drive_kind):
    return (
        ve if drive_kind in {"ve", "both"} else None,
        intra if drive_kind in {"intra", "both"} else None,
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 2.0e-5, 2.0e-6),
        (torch.float64, 2.0e-12, 2.0e-12),
    ],
)
@pytest.mark.parametrize("drive_kind", ["none", "ve", "intra", "both"])
def test_single_compartment_matches_every_imperative_leaf_at_each_step_and_fused(
    dtype,
    rtol,
    atol,
    drive_kind,
):
    source = _model(dtype=dtype)
    imperative = _model(dtype=dtype)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source)
    ve, intra = _select_drives(ve, intra, drive_kind)
    state = tensors.state

    for index in range(STEPS):
        ve_step = None if ve is None else ve[index]
        intra_step = None if intra is None else intra[index]
        state, auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve_step, intra=intra_step),
        )
        _imperative_step(imperative, ve_step, intra_step)
        _assert_tree_close(
            state,
            functional.extract(imperative).state,
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            auxiliary["v"],
            state["integrator"]["v"],
            rtol=0.0,
            atol=0.0,
        )

    rolled, auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
        steps=STEPS,
    )
    _assert_tree_close(rolled, state, rtol=rtol, atol=atol)
    torch.testing.assert_close(auxiliary["v"], rolled["integrator"]["v"])


def test_single_compartment_ignores_extracellular_voltage_but_uses_intra():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model)

    no_drive, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=STEPS,
    )
    ve_only, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve),
    )
    intra_only, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(intra=intra),
    )
    both, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )

    _assert_tree_close(ve_only, no_drive)
    _assert_tree_close(both, intra_only)
    assert not torch.equal(
        intra_only["integrator"]["v"],
        no_drive["integrator"]["v"],
    )

    def voltage_from_ve(local_ve):
        return functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(ve=local_ve),
        )[0]["integrator"]["v"]

    ve_jacobian = torch.func.jacrev(voltage_from_ve)(ve)
    assert torch.count_nonzero(ve_jacobian) == 0

    def voltage_from_intra(local_intra):
        return functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(intra=local_intra),
        )[0]["integrator"]["v"]

    intra_jacobian = torch.func.jacrev(voltage_from_intra)(intra)
    assert torch.isfinite(intra_jacobian).all()
    assert torch.count_nonzero(intra_jacobian) > 0


def test_single_compartment_explicit_batch_matches_imperative_everywhere():
    source = _model(batch_calls=(3,))
    imperative = _model(batch_calls=(3,))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source)

    assert source.shape == (3, *source.core_shape())
    assert functional.shape == source.shape
    assert tensors.constants["diam"].shape == source.core_shape()
    assert tensors.constants["dx"].shape == source.core_shape()

    state = tensors.state
    for index in range(STEPS):
        state, _ = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(imperative, ve[index], intra[index])
        _assert_tree_close(
            state,
            functional.extract(imperative).state,
            rtol=2.0e-12,
            atol=2.0e-12,
        )

    rolled, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_tree_close(rolled, state, rtol=2.0e-12, atol=2.0e-12)


def test_single_compartment_jacrev_jacfwd_vmap_and_zero_lanes():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    _ve, intra = _drives(model, steps=2)
    raw_conductance = tensors.parameters[GNABAR]

    def voltage_from_conductance(local_conductance):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = local_conductance
        return functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(intra=intra),
        )[0]["integrator"]["v"]

    reverse = torch.func.jacrev(voltage_from_conductance)(raw_conductance)
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(voltage_from_conductance)(raw_conductance)
    assert torch.isfinite(reverse).all()
    assert torch.count_nonzero(reverse) > 0
    torch.testing.assert_close(reverse, forward, rtol=2.0e-9, atol=2.0e-10)

    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def run_lane(voltage_offset, local_intra):
        state = {
            **tensors.state,
            "integrator": {
                **tensors.state["integrator"],
                "v": tensors.state["integrator"]["v"] + voltage_offset,
            },
        }
        return functional.rollout(
            tensors.parameters,
            prepared,
            state,
            dn.func.RolloutInput(intra=local_intra),
        )[0]["integrator"]["v"]

    offsets = model.v.new_tensor([-0.25, 0.0, 0.3])
    intra_lanes = torch.stack((0.5 * intra, intra, 1.5 * intra))
    actual = torch.vmap(run_lane)(offsets, intra_lanes)
    expected = torch.stack(
        tuple(
            run_lane(offset, local_intra)
            for offset, local_intra in zip(offsets, intra_lanes, strict=True)
        )
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-12)

    def run_parameter_lane(local_conductance):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = local_conductance
        return functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(intra=intra),
        )[0]["integrator"]["v"]

    conductance_lanes = torch.stack(
        (0.8 * raw_conductance, raw_conductance, 1.2 * raw_conductance)
    )
    parameter_actual = torch.vmap(run_parameter_lane)(conductance_lanes)
    parameter_expected = torch.stack(
        tuple(run_parameter_lane(value) for value in conductance_lanes)
    )
    torch.testing.assert_close(
        parameter_actual,
        parameter_expected,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    empty_parameters = torch.vmap(run_parameter_lane)(
        raw_conductance.new_empty((0, *raw_conductance.shape))
    )
    assert empty_parameters.shape == (0, *model.shape)

    empty_states = torch.utils._pytree.tree_map(
        lambda value: value.new_empty((0, *value.shape)),
        tensors.state,
    )
    empty_intra = intra.new_empty((0, *intra.shape))

    def run_empty_lane(state, local_intra):
        return functional.rollout(
            tensors.parameters,
            prepared,
            state,
            dn.func.RolloutInput(intra=local_intra),
        )[0]["integrator"]["v"]

    empty = torch.vmap(run_empty_lane)(empty_states, empty_intra)
    assert empty.shape == (0, *model.shape)


def test_single_compartment_atomic_rollout_is_fullgraph_aot_differentiable():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    _ve, intra = _drives(model, steps=2)

    def atomic(local_conductance, local_intra):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = local_conductance
        return functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(intra=local_intra),
        )[0]["integrator"]["v"]

    eager_conductance = tensors.parameters[GNABAR].detach().clone().requires_grad_()
    eager_intra = intra.detach().clone().requires_grad_()
    eager_voltage = atomic(eager_conductance, eager_intra)
    eager_gradients = torch.autograd.grad(
        eager_voltage.square().mean(),
        (eager_conductance, eager_intra),
    )

    compiled_conductance = tensors.parameters[GNABAR].detach().clone().requires_grad_()
    compiled_intra = intra.detach().clone().requires_grad_()
    compiled = torch.compile(
        atomic,
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )
    with torch_compiler_warning_context():
        compiled_voltage = compiled(compiled_conductance, compiled_intra)
        compiled_gradients = torch.autograd.grad(
            compiled_voltage.square().mean(),
            (compiled_conductance, compiled_intra),
        )

    torch.testing.assert_close(
        compiled_voltage,
        eager_voltage,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


def test_single_compartment_fixed_chunks_and_host_run_match_eager_rollout():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, steps=4)
    inputs = dn.func.RolloutInput(ve=ve, intra=intra)

    expected, expected_auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        inputs,
    )

    chunk = functional.compile_rollout_chunk(2, backend="aot_eager")
    chunked = tensors.state
    with torch_compiler_warning_context():
        for start in range(0, 4, chunk.steps):
            chunked, chunk_auxiliary = chunk(
                tensors.parameters,
                prepared,
                chunked,
                dn.func.RolloutInput(
                    ve=ve[start : start + chunk.steps],
                    intra=intra[start : start + chunk.steps],
                ),
            )
    _assert_tree_close(chunked, expected, rtol=2.0e-12, atol=2.0e-12)
    _assert_tree_close(
        chunk_auxiliary,
        expected_auxiliary,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    bound_step = partial(functional.step, tensors.parameters, prepared)
    scheduled, scheduled_auxiliary = dn.func.run(
        functional,
        bound_step,
        tensors.state,
        inputs,
    )
    _assert_tree_close(scheduled, expected, rtol=2.0e-12, atol=2.0e-12)
    _assert_tree_close(
        scheduled_auxiliary,
        expected_auxiliary,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def _checkpoint_case(functional, tensors, intra, *, checkpointed):
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    raw_conductance = parameters[GNABAR].requires_grad_()
    constants = {
        name: value.detach().clone() for name, value in tensors.constants.items()
    }
    state = _clone_tree(tensors.state)
    initial_voltage = state["integrator"]["v"].requires_grad_()
    local_intra = intra.detach().clone().requires_grad_()
    prepared = functional.prepare(parameters, constants)
    step = partial(functional.step, parameters, prepared)
    runner = dn.func.longrun_checkpointed if checkpointed else dn.func.longrun
    final, auxiliary = runner(
        functional,
        step,
        state,
        STEPS * DT,
        2,
        dn.func.RolloutInput(intra=local_intra),
    )
    loss = (
        final["integrator"]["v"].square().mean()
        + 0.01 * final["mechanisms"]["hh"]["m"].square().mean()
        + 0.001 * auxiliary["v"].sin().mean()
    )
    gradients = torch.autograd.grad(
        loss,
        (raw_conductance, initial_voltage, local_intra),
    )
    return final, auxiliary, loss, gradients


def test_single_compartment_checkpointed_runner_matches_ordinary_bptt():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    _ve, intra = _drives(model)

    ordinary = _checkpoint_case(functional, tensors, intra, checkpointed=False)
    checkpointed = _checkpoint_case(functional, tensors, intra, checkpointed=True)
    ordinary_state, ordinary_auxiliary, ordinary_loss, ordinary_gradients = ordinary
    (
        checkpointed_state,
        checkpointed_auxiliary,
        checkpointed_loss,
        checkpointed_gradients,
    ) = checkpointed

    _assert_tree_close(
        checkpointed_state,
        ordinary_state,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    _assert_tree_close(
        checkpointed_auxiliary,
        ordinary_auxiliary,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        checkpointed_loss,
        ordinary_loss,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    for actual, expected in zip(
        checkpointed_gradients,
        ordinary_gradients,
        strict=True,
    ):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


def test_single_compartment_commit_resume_and_functional_calls_are_pure():
    source = _model()
    target = _model()
    reference = _model()
    source_snapshot = _source_snapshot(source)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=4)
    state_snapshot = _clone_tree(tensors.state)
    ve_snapshot = _tensor_snapshot(ve)
    intra_snapshot = _tensor_snapshot(intra)

    checkpoint, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve[:2], intra=intra[:2]),
    )
    functional.commit_state_(target, checkpoint)
    for index in range(2, 4):
        _imperative_step(target, ve[index], intra[index])
    for index in range(4):
        _imperative_step(reference, ve[index], intra[index])

    _assert_tree_close(
        functional.extract(target).state,
        functional.extract(reference).state,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    _assert_tree_close(tensors.state, state_snapshot)
    _assert_tensor_snapshot(ve, ve_snapshot)
    _assert_tensor_snapshot(intra, intra_snapshot)
    _assert_source_unchanged(source, source_snapshot)
