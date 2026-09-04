from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._lowered import _LoweredPopulationChunk
from dendra.models.mod import hh

DT = 0.01

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -60.0, -56.0, -59.0, -63.0]),
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(hh)
        model.initialize()
        model.train()
    return model


def _drives(model, steps):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -2.0,
        2.0,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -1.0e-9,
        1.0e-9,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _replace_voltage(state: Mapping[str, object], voltage: torch.Tensor):
    replaced = dict(state)
    replaced["integrator"] = dict(state["integrator"])
    replaced["integrator"]["v"] = voltage
    return replaced


@pytest.fixture()
def lowered_case():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, 2)
    lowered = _LoweredPopulationChunk(functional, steps=2)
    operands = lowered.bind(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    return functional, tensors, prepared, lowered, operands


def test_bind_keeps_all_validation_outside_the_tensor_transition(
    lowered_case,
    monkeypatch,
):
    functional, tensors, prepared, lowered, initial_operands = lowered_case
    expected = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=initial_operands.ve, intra=initial_operands.intra),
    )

    validation_names = (
        "_validate_source",
        "_validate_parameters",
        "_validate_prepared",
        "_validate_state",
        "_resolve_rollout_inputs",
    )
    validation_calls = dict.fromkeys(validation_names, 0)
    for name in validation_names:
        original = getattr(functional, name)

        def count_validation(*args, _name=name, _original=original, **kwargs):
            validation_calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(functional, name, count_validation)

    operands = lowered.bind(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=initial_operands.ve, intra=initial_operands.intra),
    )
    assert validation_calls == dict.fromkeys(validation_names, 1)

    def fail_validation(*_args, **_kwargs):
        raise AssertionError("validation leaked into the tensor-only transition")

    for name in validation_names:
        monkeypatch.setattr(functional, name, fail_validation)

    actual = lowered(*operands)
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        torch.testing.assert_close(actual_leaf, expected_leaf, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("transform_name", ["jacrev", "jacfwd"])
def test_transform_then_compile_chunk_jacobians_match_eager(
    lowered_case,
    transform_name,
):
    _functional, _tensors, _prepared, lowered, operands = lowered_case
    initial_voltage = operands.state["integrator"]["v"]

    def final_voltage(voltage):
        state = _replace_voltage(operands.state, voltage)
        final, _aux = lowered(
            operands.parameters,
            operands.prepared,
            state,
            operands.ve,
            operands.intra,
        )
        return final["integrator"]["v"]

    transform = getattr(torch.func, transform_name)
    transformed = transform(final_voltage)
    expected = transformed(initial_voltage)
    compiled = torch.compile(
        transformed,
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )
    with torch_compiler_warning_context():
        actual = compiled(initial_voltage)

    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) > 0
    torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-10)


def test_vmap_then_compile_chunk_matches_explicit_lanes(lowered_case):
    _functional, _tensors, _prepared, lowered, operands = lowered_case
    initial_voltage = operands.state["integrator"]["v"]
    lanes = torch.stack(
        (
            initial_voltage - 0.25,
            initial_voltage,
            initial_voltage + 0.25,
        )
    )

    def final_voltage(voltage):
        state = _replace_voltage(operands.state, voltage)
        final, _aux = lowered(
            operands.parameters,
            operands.prepared,
            state,
            operands.ve,
            operands.intra,
        )
        return final["integrator"]["v"]

    expected = torch.stack(tuple(final_voltage(lane) for lane in lanes))
    transformed = torch.func.vmap(final_voltage)
    compiled = torch.compile(
        transformed,
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )
    with torch_compiler_warning_context():
        actual = compiled(lanes)

    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


def test_compiled_wrapper_and_transform_composition_share_one_tensor_kernel(
    lowered_case,
):
    functional, tensors, prepared, _lowered, operands = lowered_case
    chunk = functional.compile_rollout_chunk(2, backend="aot_eager")

    assert isinstance(chunk._lowered, _LoweredPopulationChunk)
    with torch_compiler_warning_context():
        actual = chunk(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(ve=operands.ve, intra=operands.intra),
        )
    expected = chunk._lowered(*operands)

    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=2.0e-10,
            atol=2.0e-11,
        )
