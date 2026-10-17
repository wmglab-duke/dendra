"""Fresh, pure initialization of the canonical scalar Tree capability slice."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import hh
from tests.func.test_functional_initialization import (
    AM1,
    _assert_module_snapshot,
    _assert_tensor_snapshot,
    _assert_tree_close,
    _module_snapshot,
    _tensor_snapshot,
)
from tests.func.test_scalar_tree import (
    _imperative_step,
    _morphology,
)

DT = 0.025

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE,
        reason="scalar Tree CPU execution requires dendra-solvers",
    ),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated.*"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated.*"),
]


class _TreeSubclass(dn.Tree):
    pass


def _model(*, dtype=torch.float32, batch_calls=(), population_type=dn.Tree):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = population_type.from_morphology(
            _morphology(),
            N=2,
            dtype=dtype,
            celsius=6.3,
            v_init=-62.0,
            integrator=dn.dhs(threads=2, imem=False),
        )
        model.internal_nodes.insert(hh)
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _voltage(model):
    return torch.linspace(-72.0, -51.0, model.v.numel(), dtype=model.dtype()).reshape(
        model.shape
    )


def _initialize(functional, tensors, voltage, am1=None):
    parameters = dict(tensors.parameters)
    if am1 is not None:
        parameters[AM1] = am1
    return functional.initialize(
        parameters,
        tensors.constants,
        tensors.initialization._replace(v_init=voltage),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_calls", [(), (2,)])
def test_tree_initializer_recomputes_all_state_leaves_at_explicit_inputs(
    dtype, batch_calls
):
    model = _model(dtype=dtype, batch_calls=batch_calls)
    # A resumed source clock must not leak into a fresh initialization.
    model.t.fill_(3.0)
    model._duration_remainder.fill_(0.004)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model)
    am1 = tensors.parameters[AM1] * 1.17
    actual = _initialize(functional, tensors, voltage, am1)

    reference = _model(dtype=dtype, batch_calls=batch_calls)
    reference.set_v_init(voltage)
    with torch.no_grad():
        dict(reference.named_parameters())[AM1].copy_(am1)
    reference.initialize()
    _assert_tree_close(actual.state, functional.extract(reference).state)

    # Includes zero-area junctions in voltage while HH stays support-local.
    area = tensors.constants["canonical_area_cm2"]
    assert torch.count_nonzero(area == 0) > 0
    torch.testing.assert_close(actual.state["integrator"]["v"], voltage, rtol=0, atol=0)
    assert actual.state["mechanisms"]["hh"]["m"].numel() < voltage.numel()
    assert not torch.equal(
        actual.state["mechanisms"]["hh"]["m"],
        tensors.state["mechanisms"]["hh"]["m"],
    )
    changed_voltage = _initialize(functional, tensors, voltage + 2.0, am1)
    changed_parameter = _initialize(functional, tensors, voltage, am1 * 0.8)
    for changed in (changed_voltage, changed_parameter):
        assert not torch.equal(
            actual.state["mechanisms"]["hh"]["m"],
            changed.state["mechanisms"]["hh"]["m"],
        )
    assert actual.state["integrator"]["v"] is not voltage
    for name, value in actual.parameters.items():
        assert value is (am1 if name == AM1 else tensors.parameters[name])
    for name, value in actual.constants.items():
        assert value is tensors.constants[name]
    for group in ("integrator", "mechanisms"):
        for value in torch.utils._pytree.tree_leaves(actual.state[group]):
            assert value.dtype == dtype


def test_tree_initializer_does_not_mutate_source_or_explicit_inputs():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model).requires_grad_()
    snapshot = _module_snapshot(model)
    inputs = [
        *tensors.parameters.values(),
        *tensors.constants.values(),
        *torch.utils._pytree.tree_leaves(tensors.state),
        voltage,
    ]
    snapshots = [_tensor_snapshot(value) for value in inputs]
    first = _initialize(functional, tensors, voltage)
    second = _initialize(functional, tensors, voltage)
    _assert_tree_close(first.state, second.state)
    first.state["mechanisms"]["hh"]["m"].sum().backward()
    assert voltage.grad is not None and torch.isfinite(voltage.grad).all()
    _assert_module_snapshot(model, snapshot)
    for value, before in zip(inputs, snapshots, strict=True):
        _assert_tensor_snapshot(value, before)


def test_tree_initializer_ad_and_fixed_initial_gate_boundary():
    model = _model(dtype=torch.float64)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage, am1 = _voltage(model), tensors.parameters[AM1]

    def gates(v, a):
        return _initialize(functional, tensors, v, a).state["mechanisms"]["hh"]["m"]

    backward = torch.func.jacrev(gates, argnums=(0, 1))(voltage, am1)
    forward = torch.func.jacfwd(gates, argnums=(0, 1))(voltage, am1)
    for actual, expected in zip(backward, forward, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
        assert torch.count_nonzero(actual) > 0

    # Callers can keep initial gates fixed while exposing a fresh V0 AD leaf.
    with torch.no_grad():
        initialized = _initialize(functional, tensors, voltage, am1)
    initial_voltage = initialized.state["integrator"]["v"].requires_grad_()
    initial_gates = initialized.state["mechanisms"]["hh"]
    assert not any(value.requires_grad for value in initial_gates.values())
    loss = initial_voltage.square().mean() + sum(
        value.square().mean() for value in initial_gates.values()
    )
    (gradient,) = torch.autograd.grad(loss, (initial_voltage,))
    torch.testing.assert_close(gradient, 2 * initial_voltage / initial_voltage.numel())


def test_tree_initializer_vmap_including_empty_lanes_matches_independent_calls():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model)
    voltages = (
        voltage.unsqueeze(0) + voltage.new_tensor([-1.0, 0.0, 2.0])[:, None, None]
    )
    rates = tensors.parameters[AM1] * voltage.new_tensor([0.8, 1.0, 1.2])

    def initialize(v, a):
        return _initialize(functional, tensors, v, a).state

    actual = torch.vmap(initialize)(voltages, rates)
    expected = torch.utils._pytree.tree_map(
        lambda *values: torch.stack(values),
        *(initialize(v, a) for v, a in zip(voltages, rates, strict=True)),
    )
    _assert_tree_close(actual, expected)
    empty = torch.vmap(initialize)(voltages[:0], rates[:0])
    for left, right in zip(
        torch.utils._pytree.tree_leaves(empty),
        torch.utils._pytree.tree_leaves(actual),
        strict=True,
    ):
        assert left.shape == (0, *right.shape[1:])


def test_tree_initializer_rollout_matches_public_initialization():
    model = _model(dtype=torch.float64)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model)
    initialized = _initialize(functional, tensors, voltage)
    steps = 2
    drive = (
        torch.linspace(-1e-9, 2e-9, steps * voltage.numel())
        .to(voltage)
        .reshape(steps, *voltage.shape)
    )
    actual, _ = functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        dn.func.RolloutInput(intra=drive),
        steps=steps,
    )
    reference = _model(dtype=torch.float64)
    reference.set_v_init(voltage)
    reference.initialize()
    dt = torch.as_tensor(DT, dtype=reference.dtype())
    reference.integrator._initialize(
        reference, dt, force=True, compile_scope="population"
    )
    for current in drive:
        _imperative_step(reference, dt, None, current)
    _assert_tree_close(
        actual, functional.extract(reference).state, rtol=2e-11, atol=2e-11
    )


@pytest.mark.parametrize(
    "mutation", ["pre_hook", "post_hook", "steady_state", "init_v"]
)
def test_tree_initializer_rechecks_imperative_lifecycle_after_lowering(mutation):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    if mutation == "steady_state":
        model.cache("_steady_state")
    elif mutation == "init_v":
        model.integrator.init_v = lambda population: None
    else:
        getattr(model, f"register_{mutation.removesuffix('_hook')}_initialize_hook")(
            lambda population: population.v.add_(1.0)
        )
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="initialization|initialization hooks|steady.state|fresh-initialization",
    ):
        _initialize(functional, tensors, tensors.initialization.v_init)


def test_tree_subclass_initialization_stays_outside_exact_type_admission():
    model = _model(population_type=_TreeSubclass)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(dn.func.FunctionalizationError, match="exact scalar Tree"):
        _initialize(functional, tensors, tensors.initialization.v_init)


def test_tree_initializer_compile_composes_with_voltage_derivative():
    model = _model(dtype=torch.float64)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model)

    def loss(v):
        return (
            _initialize(functional, tensors, v)
            .state["mechanisms"]["hh"]["m"]
            .square()
            .mean()
        )

    gradient = torch.func.grad(loss)
    compiled = torch.compile(gradient, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual = compiled(voltage)
    torch.testing.assert_close(actual, gradient(voltage), rtol=2e-12, atol=2e-12)


@pytest.mark.slow
def test_tree_initializer_real_cpu_inductor_preserves_float32_and_fresh_gates():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = _voltage(model)

    def initialize(v, a):
        return _initialize(functional, tensors, v, a).state

    compiled = torch.compile(initialize, backend="inductor", fullgraph=True)
    with torch_compiler_warning_context():
        actual = compiled(voltage, tensors.parameters[AM1])
        changed = compiled(voltage + 2.0, tensors.parameters[AM1] * 1.2)
    _assert_tree_close(
        actual, initialize(voltage, tensors.parameters[AM1]), rtol=4e-6, atol=2e-7
    )
    _assert_tree_close(
        changed,
        initialize(voltage + 2.0, tensors.parameters[AM1] * 1.2),
        rtol=4e-6,
        atol=2e-7,
    )
    assert not torch.equal(
        actual["mechanisms"]["hh"]["m"], changed["mechanisms"]["hh"]["m"]
    )
    for value in torch.utils._pytree.tree_leaves(actual["mechanisms"]):
        assert value.dtype == torch.float32
