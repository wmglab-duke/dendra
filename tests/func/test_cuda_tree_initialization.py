"""Pure scalar Tree initialization on real CUDA, including compiled AD."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import hh
from tests.func.test_functional_initialization import (
    AM1,
    _assert_module_snapshot,
    _assert_tensor_snapshot,
    _assert_tree_close,
    _module_snapshot,
    _tensor_snapshot,
)
from tests.func.test_scalar_tree import _morphology

DT = 0.025
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated.*"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated.*"),
]


def _model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cuda", DTYPE=torch.float32):
        model = dn.Tree.from_morphology(
            _morphology(),
            N=2,
            device="cuda",
            dtype=torch.float32,
            celsius=6.3,
            v_init=-62.0,
            integrator=dn.dhs(threads=32, imem=False),
        )
        model.internal_nodes.insert(hh)
        model.initialize()
        model.train()
    return model


@pytest.mark.slow
def test_cuda_tree_compiled_initializer_recomputes_fresh_state_and_gradients():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    # Lowering prepares an independent execution copy. The source's numeric
    # workspace is intentionally still unprepared, so allocator-provided NaNs
    # there must remain bitwise untouched rather than enter initialized state.
    assert not model.integrator.initialized
    with torch.no_grad():
        model.integrator.area[0, 4] = float("nan")
    voltage = (
        torch.linspace(
            -73.0, -51.0, model.v.numel(), device="cuda", dtype=torch.float32
        )
        .reshape(model.shape)
        .requires_grad_()
    )
    rate = tensors.parameters[AM1].detach().clone().requires_grad_()
    source_snapshot = _module_snapshot(model)
    inputs = [
        *tensors.parameters.values(),
        *tensors.constants.values(),
        *torch.utils._pytree.tree_leaves(tensors.state),
        voltage,
        rate,
    ]
    input_snapshots = [_tensor_snapshot(value) for value in inputs]

    def initialize(v, am1):
        parameters = {**tensors.parameters, AM1: am1}
        return functional.initialize(
            parameters,
            tensors.constants,
            tensors.initialization._replace(v_init=v),
        ).state

    def objective(state):
        return state["mechanisms"]["hh"]["m"].square().mean()

    compiled = torch.compile(
        initialize, backend="inductor", fullgraph=True, dynamic=False
    )
    with torch_compiler_warning_context():
        actual = compiled(voltage, rate)
        gradients = torch.autograd.grad(objective(actual), (voltage, rate))
    expected = initialize(voltage, rate)
    expected_gradients = torch.autograd.grad(objective(expected), (voltage, rate))
    _assert_tree_close(actual, expected, rtol=4e-6, atol=3e-7)
    for actual_gradient, expected_gradient in zip(
        gradients, expected_gradients, strict=True
    ):
        assert actual_gradient.dtype == torch.float32
        assert actual_gradient.device.type == "cuda"
        assert torch.isfinite(actual_gradient).all()
        assert torch.count_nonzero(actual_gradient) > 0
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=4e-5, atol=2e-7
        )

    # Independent public initialization at changed values rules out a captured
    # initialized-state reset, while retaining one compiled input schema.
    changed_voltage = (voltage.detach() + 3.0).requires_grad_()
    changed_rate = (rate.detach() * 0.85).requires_grad_()
    with torch_compiler_warning_context():
        changed = compiled(changed_voltage, changed_rate)
    reference = _model()
    reference.set_v_init(changed_voltage.detach())
    with torch.no_grad():
        dict(reference.named_parameters())[AM1].copy_(changed_rate)
    reference.initialize()
    _assert_tree_close(
        changed, functional.extract(reference).state, rtol=4e-6, atol=3e-7
    )
    assert not torch.equal(
        actual["mechanisms"]["hh"]["m"], changed["mechanisms"]["hh"]["m"]
    )
    for state in (actual, changed):
        for group in ("integrator", "mechanisms"):
            for value in torch.utils._pytree.tree_leaves(state[group]):
                assert value.dtype == torch.float32
                assert value.device.type == "cuda"
        assert torch.count_nonzero(state["clock"]["t"]) == 0
        assert torch.count_nonzero(state["control"]["duration_remainder"]) == 0

    # The CUDA initializer also composes with forward AD independently of the
    # compiled reverse pass, including support-local voltage selection.
    direction = torch.linspace(
        -0.2, 0.3, voltage.numel(), device="cuda", dtype=voltage.dtype
    ).reshape_as(voltage)
    rate_direction = torch.full_like(rate, 0.1)
    _, tangent = torch.func.jvp(
        lambda v, a: objective(initialize(v, a)),
        (voltage, rate),
        (direction, rate_direction),
    )
    reverse_tangent = (expected_gradients[0] * direction).sum() + (
        expected_gradients[1] * rate_direction
    ).sum()
    torch.testing.assert_close(tangent, reverse_tangent, rtol=4e-5, atol=2e-7)
    _assert_module_snapshot(model, source_snapshot)
    for value, snapshot in zip(inputs, input_snapshots, strict=True):
        _assert_tensor_snapshot(value, snapshot)
