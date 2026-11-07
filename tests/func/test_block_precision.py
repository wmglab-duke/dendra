"""Block workspaces and public state retain the requested Population dtype."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra.models.integrators.implicit import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import hh
from tests.func.test_cuda_population import (
    DT,
    _close,
    _close_gradients,
    _functional_result,
    _imperative_result,
)
from tests.func.test_cuda_population import _model as _acceptance_model
from tests.func.test_cuda_population import (
    _seed_circuit_state,
)

requires_solvers = pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE, reason="CPU block execution requires dendra-solvers"
)


def _model(kind, *, dtype=torch.float32, batch_calls=(), single_node=False):
    if single_node:
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE="cpu", DTYPE=dtype):
            model = dn.ExtCellAxon(
                [5.0, 8.0],
                n_comp=1,
                dtype=dtype,
                integrator=dn.bwd_euler_bt(method="thomas", imem=False),
            )
            model.xraxial.fill_(2.0)
            model.xc.fill_(0.1)
            model.xg.fill_(2e-4)
            model.insert(hh)
            for size in batch_calls:
                model.batch(size)
            model.initialize()
            model.train()
    else:
        model = _acceptance_model(
            kind, device="cpu", dtype=dtype, batch_calls=batch_calls
        )
    # Keep the small geometry derivative caused by uniform membrane voltage.
    # A spatial voltage ramp would conceal the coefficient cancellation.
    model.v = torch.full_like(model.v, -65.0 if kind == "ext_axon" else -64.0)
    _seed_circuit_state(model)
    return model


def _assert_public_dtype(state, dtype):
    duration = state["control"]["duration_remainder"]
    assert duration.dtype == torch.float64 and duration.device.type == "cpu"
    for section, values in state.items():
        if section == "control":
            values = {
                name: value
                for name, value in values.items()
                if name != "duration_remainder"
            }
        assert all(value.dtype == dtype for value in pytree.tree_leaves(values))


class _Float32NumericalMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.native_solves = []

    @staticmethod
    def _check(func, values, stage):
        floating = [
            value
            for value in pytree.tree_leaves(values)
            if torch.is_tensor(value) and value.is_floating_point()
        ]
        assert all(value.dtype == torch.float32 for value in floating), (
            f"{func} has non-float32 numerical {stage}: "
            f"{[value.dtype for value in floating]}"
        )
        return floating

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        floating_inputs = self._check(func, (args, kwargs), "inputs")
        result = func(*args, **kwargs)
        floating_outputs = self._check(func, result, "outputs")
        if func._schema.name.startswith("dendra_solvers::") and floating_inputs:
            assert floating_outputs, "the native solve must return numerical tensors"
            self.native_solves.append(func._schema.name)
        return result


@requires_solvers
@pytest.mark.parametrize("kind", ["ext_axon", "ext_tree"])
def test_float32_block_numeric_paths_do_not_promote_intermediates(monkeypatch, kind):
    model = _model(kind)
    integrator = model.integrator
    timestep = torch.tensor(DT, dtype=torch.float32)
    builder = integrator._prepare_workspace
    preparation = {}

    def capture_preparation(dt, **physical_inputs):
        preparation["dt"] = dt
        preparation["inputs"] = physical_inputs
        return builder(dt, **physical_inputs)

    # Keep the host duration clock outside this scope: only solver preparation
    # and numerical state transitions are required to stay entirely float32.
    with monkeypatch.context() as patch:
        patch.setattr(integrator, "_prepare_workspace", capture_preparation)
        with _Float32NumericalMode():
            integrator._initialize(model, timestep, force=True)

    with _Float32NumericalMode():
        builder(preparation["dt"], **preparation["inputs"])

    voltage = model.v.detach().clone()
    circuit = model.vc.detach().clone().requires_grad_()
    with _Float32NumericalMode() as mode:
        result = integrator._step(
            circuit,
            voltage,
            timestep,
            model.celsius,
            ve=torch.zeros_like(voltage),
            intra=torch.full_like(voltage, 1.0e-10),
        )
        loss = result[0].square().mean() + result[1].square().mean()
        gradient = torch.autograd.grad(loss, circuit)[0]
    assert mode.native_solves, "the native numerical solver boundary was not exercised"
    assert gradient.dtype == torch.float32


@requires_solvers
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("kind", "batch_calls", "single_node"),
    [
        ("ext_axon", (), False),
        ("ext_axon", (2,), False),
        ("ext_axon", (), True),
        ("ext_tree", (), False),
        ("ext_tree", (2,), False),
    ],
)
def test_block_workspace_preserves_requested_dtype(
    kind, batch_calls, single_node, dtype
):
    model = _model(kind, dtype=dtype, batch_calls=batch_calls, single_node=single_node)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    workspace = prepared.values["integrator"]

    assert workspace["dt"].dtype == dtype
    assert all(
        workspace[name].dtype == dtype
        for name in functional._integrator_workspace_names
    )
    assert all(value.dtype == dtype for value in tensors.constants.values())
    assert all(value.dtype == dtype for value in tensors.parameters.values())
    _assert_public_dtype(tensors.state, dtype)

    model.integrator._initialize(
        model, tensors.constants["dt"], force=True, compile_scope="population"
    )
    for name in functional._integrator_workspace_names:
        expected = getattr(model.integrator, name)
        assert expected.dtype == dtype
        torch.testing.assert_close(workspace[name], expected, rtol=0, atol=0)
    if single_node:
        assert workspace["lower"].shape[-2:] == (0, 3)

    state, _ = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(
            ve=torch.zeros_like(model.v), intra=torch.full_like(model.v, 1e-10)
        ),
    )
    _assert_public_dtype(state, dtype)


def _rebind_values(prepared, values):
    # Recompute the binding to exercise schema validation independently of
    # the separate mutation/freshness guard.
    return type(prepared)(
        prepared.token,
        prepared.parameter_binding,
        prepared.constant_binding,
        prepared.constant_sources,
        prepared.grad_enabled_at_prepare,
        values,
    )


@requires_solvers
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["ext_axon", "ext_tree"])
def test_block_workspace_precision_retains_schema_and_freshness_checks(kind, dtype):
    model = _model(kind, dtype=dtype)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    area = prepared.values["integrator"]["area"]

    wrong_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    for malformed, message in (
        (
            area.to(dtype=wrong_dtype),
            rf"prepared integrator\.area must use .*{dtype}",
        ),
        (area.flatten(), r"prepared integrator\.area must have shape"),
    ):
        values = {
            **prepared.values,
            "integrator": {**prepared.values["integrator"], "area": malformed},
        }
        with pytest.raises(dn.func.FunctionalizationError, match=message):
            functional.step(
                tensors.parameters,
                _rebind_values(prepared, values),
                tensors.state,
            )

    with torch.no_grad():
        area.add_(0)
    with pytest.raises(
        dn.func.FunctionalizationError, match="prepared workspaces changed"
    ):
        functional.step(tensors.parameters, prepared, tensors.state)


@requires_solvers
@pytest.mark.parametrize("kind", ["ext_axon", "ext_tree"])
def test_uniform_float32_block_gradients_match_imperative_and_float64(kind):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE="cpu", DTYPE=torch.float32):
        source, reference = _model(kind), _model(kind)
        assert torch.unique(source.v).numel() == 1
        functional, _, states, gradients = _functional_result(kind, source)
        expected_states, expected_gradients = _imperative_result(
            kind, reference, functional
        )
    assert len(gradients) == 9
    _close(states, expected_states, dtype=torch.float32)
    _close_gradients(gradients, expected_gradients, dtype=torch.float32)

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE="cpu", DTYPE=torch.float64):
        _, _, _, reference_gradients = _functional_result(
            kind, _model(kind, dtype=torch.float64)
        )
    _close_gradients(
        {name: gradient.double() for name, gradient in gradients.items()},
        reference_gradients,
        dtype=torch.float32,
    )


@requires_solvers
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["ext_axon", "ext_tree"])
def test_imperative_block_membrane_current_retains_requested_dtype(kind, dtype):
    model = _model(kind, dtype=dtype)
    model.integrator.imem = True
    model.initialize()
    timestep = torch.tensor(DT, dtype=dtype)
    model.integrator._initialize(model, timestep, force=True)
    model.integrator.step(
        model, timestep, torch.zeros_like(model.v), torch.full_like(model.v, 1e-10)
    )

    assert model.v.dtype == model.vc.dtype == model.i_membrane.dtype == dtype
    assert torch.isfinite(model.i_membrane).all()
