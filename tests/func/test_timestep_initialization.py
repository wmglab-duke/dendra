"""Fresh initialization sees zero timestep workspaces, not builders at dt=0.

Runtime preparation is a distinct differentiable transaction. Keep horizons
short and reuse the same mixed Mechanism/State probe across CPU and CUDA tests.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_initialization_parity import (
    CHANNELS,
    _close_history,
    _constant_target,
    _live_state,
    _read_path,
    _record_paths,
)
from tests.func.test_membrane_current import _model as _membrane_model

DT = 0.01
OLD_DT = 0.0375
STEPS = 4
MECHANISM = "_InitializationTimestepMechanism"
STATE = "_InitializationTimestepState"
PREFIX = f"integrator.mech.mechanisms.{MECHANISM}."
WORKSPACE_TARGETS = (
    "workspace_mechanism",
    "workspace_mechanism_rate",
    "workspace_state",
    "workspace_state_rate",
)
pytestmark = pytest.mark.cpu


class _InitializationTimestepState(State):
    State.STATE("activation")
    State.CARRY("state_seed")
    State.GLOBAL(state_scale=0.25, state_rate=0.45)
    State.TIMESTEP_BUFFER("state_coefficient")
    State.TIMESTEP_BUFFER("state_scalar", shape=())
    State.TIMESTEP_BUFFER("state_table", shape=(1, 3, 1))
    State.ASSIGNED("state_drive")
    State.DERIVATIVE("activation' = state_drive - 0.03 * activation")

    def derive_timestep_buffers(self, dt):
        return {
            "state_coefficient": self.state_scale * (1.0 + 0.02 * self.celsius.mean())
            + 0.006 * self.diam.mean()
            + self.state_rate * dt,
            "state_scalar": self.state_scale.reshape(()) * (1.0 + 2.0 * dt),
            "state_table": (
                self.state_scale.reshape(()) + dt * self.state_rate.reshape(())
            )
            .reshape(1, 1)
            .expand(3, 1),
        }

    def state_defaults(self, v, values):
        return {
            "activation": 0.2
            + 0.001 * v
            + 0.2 * self.state_scale
            + 0.002 * values["diam"]
            + 0.001 * values["celsius"]
            + self.state_coefficient
            + self.state_scalar
            + self.state_table.sum()
        }

    def initial_values(self, v, values):
        del values
        return {
            "state_seed": 0.1
            + 0.001 * v
            + self.state_coefficient
            + self.state_scalar
            + self.state_table.sum()
        }

    def assigned_values(self, v, values):
        del values
        return {
            "state_drive": torch.zeros_like(v)
            + self.state_coefficient
            + self.state_scalar
            + self.state_table.sum()
        }


class _InitializationTimestepMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_InitializationTimestepState)
    Mechanism.GLOBAL(scale=0.35, rate=0.8, g=2.0e-5, e=-61.0)
    Mechanism.CARRY("seed")
    Mechanism.TIMESTEP_BUFFER("mechanism_coefficient")
    Mechanism.TIMESTEP_BUFFER("mechanism_scalar", shape=())
    Mechanism.TIMESTEP_BUFFER("mechanism_table", shape=(1, 2, 1))
    Mechanism.ASSIGNED("conductance")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def derive_timestep_buffers(self, dt):
        return {
            "mechanism_coefficient": self.scale * (1.0 + 0.01 * self.celsius.mean())
            + 0.004 * self.diam.mean()
            + self.rate * dt,
            "mechanism_scalar": self.scale.reshape(()) * (1.0 + 0.5 * dt),
            "mechanism_table": (self.scale.reshape(()) + dt * self.rate.reshape(()))
            .reshape(1, 1)
            .expand(2, 1),
        }

    def initial_values(self, v, values):
        return {
            "seed": 0.3
            + 0.002 * v
            + 0.1 * self.scale
            + 0.001 * values["diam"]
            + 0.001 * values["celsius"]
            + self.mechanism_coefficient
            + self.mechanism_scalar
            + self.mechanism_table.sum()
        }

    def assigned_values(self, v, values):
        del v
        return {
            "conductance": self.g
            * (
                1.0
                + 0.1 * self.activation
                + 0.02 * values["seed"]
                + 0.05
                * (
                    self.mechanism_coefficient
                    + self.mechanism_scalar
                    + self.mechanism_table.sum()
                )
            )
        }

    def advance(self, v, dt, values):
        del v
        return {
            "seed": values["seed"]
            + dt
            * (
                self.mechanism_coefficient
                + self.mechanism_scalar
                + self.mechanism_table.sum()
            )
        }

    def i(self, v):
        return self.conductance * (v - self.e)


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


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected, *, dtype=torch.float64):
    _close_history(actual, expected, dtype=dtype)


def _model(
    kind="path",
    *,
    device="cpu",
    dtype=torch.float64,
    shell_resistance_scale=1.0,
    batch=(2,),
    mechanism=_InitializationTimestepMechanism,
):
    model = _membrane_model(kind, device=device, dtype=dtype, batch=None)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DEVICE=device, DTYPE=dtype):
        components = (
            tuple(model.populations.values()) if kind.startswith("multi_") else (model,)
        )
        for component in components:
            if hasattr(component, "xraxial"):
                component.xraxial.mul_(shell_resistance_scale)
        target = (
            getattr(model, next(iter(model.populations)))
            if kind.startswith("multi_")
            else model
        )
        target[..., ::2].insert(mechanism)
        for size in batch:
            model.batch(size)
        model.initialize()
        model.train()
        # Lowering after runtime preparation must not leak stale workspace
        # values into fresh initialization or a DERIVED_BUFFER dependency.
        model.integrator._initialize(
            model, model.v.new_tensor(OLD_DT), force=True, compile_scope="population"
        )
        model.t.fill_(1.25)
        model._duration_remainder.fill_(0.004)
    return model


def _runner(
    model,
    *,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
    initialization="fresh",
):
    functional, tensors = dn.func.make_functional(model, dt=DT)
    trainable = tuple(name for name in tensors.parameters if name.endswith("_param"))
    options = {}
    if backend == "inductor":
        options = {
            "fullgraph": True,
            "options": {
                "cpp_wrapper": model.device().type == "cpu",
                "compile_threads": 1,
            },
        }
    return dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        initialization=initialization,
        execution=execution,
        inference=execution == "while",
        trainable=trainable,
        checkpoint_every=checkpoint_every,
        callbacks={"trace": dn.func.Recorder(tuple(_record_paths(functional)))},
        backend=backend,
        **options,
    )


def _arguments(runner, model):
    parameters = _clone(dict(runner.parameters))
    targets = {
        name: parameters[f"integrator.mech.mechanisms.hh.{name}_param"].requires_grad_()
        for name in CHANNELS
    }
    workspace_names = (
        PREFIX + "scale_param",
        PREFIX + "rate_param",
        PREFIX + f"DE.{STATE}.state_scale_param",
        PREFIX + f"DE.{STATE}.state_rate_param",
    )
    for label, name in zip(WORKSPACE_TARGETS, workspace_names, strict=True):
        parameters[name] = (1.07 * parameters[name]).requires_grad_()
        targets[label] = parameters[name]
    parameters["celsius_param"] = (parameters["celsius_param"] + 0.4).requires_grad_()
    targets["celsius"] = parameters["celsius_param"]
    constants = dict(runner.constants)
    geometry = (
        "populations.block_path.diam"
        if "populations.block_path.diam" in constants
        else "diam"
    )
    targets["geometry"] = constants[geometry].requires_grad_()
    targets["dt"] = constants["dt"].requires_grad_()
    initial = _clone(runner.initialization_inputs)
    voltage = (
        torch.linspace(
            -72.0, -52.0, model.v.numel(), device=model.device(), dtype=model.dtype()
        )
        .reshape(model.shape)
        .requires_grad_()
    )
    initial = initial._replace(v_init=voltage)
    targets["initial_voltage"] = voltage
    drive = torch.linspace(
        -2e-9, 3e-9, STEPS * model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(STEPS, *model.shape)
    targets["stimulus"] = model.v.new_ones((), requires_grad=True)
    inputs = dn.func.RolloutInput(intra=drive * targets["stimulus"])
    return parameters, constants, initial, inputs, targets


def _recorded(result):
    return dict(result[1]["callbacks"]["trace"])


def _loss(recorded):
    current = recorded["i_membrane"] * 1e6
    return (
        0.001 * recorded["v"].square().mean()
        + 0.0001 * current.square().mean()
        + 0.1 * recorded["hh.m"].square().mean()
        + 0.1 * recorded[f"{MECHANISM}.activation"].square().mean()
        + 0.05 * recorded[f"{MECHANISM}.seed"].square().mean()
        + 0.02 * recorded[f"{MECHANISM}.DE.{STATE}.state_seed"].square().mean()
    )


def _gradients(recorded, targets, *, retain_graph=False):
    values = torch.autograd.grad(
        _loss(recorded), tuple(targets.values()), retain_graph=retain_graph
    )
    gradients = dict(zip(targets, values, strict=True))
    for name, value in gradients.items():
        assert torch.isfinite(value).all() and torch.count_nonzero(value), name
    return gradients


def _explicit_reference(runner, parameters, constants, initial, inputs):
    functional = runner.functional
    initialized = functional.initialize(parameters, constants, initial)
    prepared = functional.prepare(initialized.parameters, initialized.constants)
    return dn.func.run(
        functional.bind(initialized.parameters, prepared),
        initialized.state,
        inputs=inputs,
        steps=STEPS,
        callbacks=runner._callbacks,
    )


def _imperative_reference(runner, reference, parameters, constants, initial, inputs):
    with torch.no_grad():
        raw = dict(reference.named_parameters())
        for name, value in parameters.items():
            raw[name].copy_(value)
        for name, value in constants.items():
            if name != "dt" and not name.startswith("parametrizations."):
                target = _constant_target(reference, name)
                expanded = torch.broadcast_to(value, target.shape)
                if not torch.equal(target, expanded):
                    owner_path, _, buffer_name = name.rpartition(".")
                    owner = (
                        reference.get_submodule(owner_path) if owner_path else reference
                    )
                    assert buffer_name in owner._buffers, name
                    # Population.batch may make the old geometry an expanded
                    # view. Rebind an owned full-shape buffer before the real
                    # initializer refreshes its support-local geometry views.
                    owner._buffers[buffer_name] = expanded.clone()
        reference.set_v_init(initial.v_init)
        reference.initialize()
        history = [_live_state(runner.functional, reference)]
        dt = constants["dt"].detach()
        reference.integrator._initialize(
            reference, dt, force=True, compile_scope="population"
        )
        for index in range(STEPS):
            reference.integrator.step(reference, dt, None, inputs.intra[index].detach())
            reference.t = reference.t + dt
            history.append(_live_state(runner.functional, reference))
        history = pytree.tree_map(lambda *values: torch.stack(values), *history)
        return {
            name: _read_path(history, path)
            for name, path in _record_paths(runner.functional).items()
        }


def _assert_initial_zero_contract(runner, parameters, constants, initial, initialized):
    prepared = runner.functional.prepare(parameters, constants)
    mechanism = runner.functional._population.mech.mechanisms[MECHANISM]
    voltage = mechanism.get(initial.v_init)
    diameter = prepared.values["population"].get(
        "diam", prepared.values["geometry"].get("diam")
    )
    assert diameter is not None
    geometry = mechanism.get(diameter)
    temperature = parameters["celsius_param"]
    expected_activation = (
        0.2
        + 0.001 * voltage
        + 0.2 * parameters[PREFIX + f"DE.{STATE}.state_scale_param"]
        + 0.002 * geometry
        + 0.001 * temperature
    )
    expected_seed = (
        0.3
        + 0.002 * voltage
        + 0.1 * parameters[PREFIX + "scale_param"]
        + 0.001 * geometry
        + 0.001 * temperature
    )
    state = initialized.state
    torch.testing.assert_close(
        state["mechanisms"][MECHANISM]["activation"],
        expected_activation.expand_as(voltage),
    )
    torch.testing.assert_close(
        state["mechanism_buffers"][MECHANISM]["seed"], expected_seed.expand_as(voltage)
    )
    torch.testing.assert_close(
        state["state_buffers"][MECHANISM][STATE]["state_seed"], 0.1 + 0.001 * voltage
    )
    # Every declared workspace remains a prepared value, never evolving carry.
    workspace_values = prepared.values["mechanisms"]
    for owner, prefix, names in (
        (
            mechanism,
            PREFIX,
            ("mechanism_coefficient", "mechanism_scalar", "mechanism_table"),
        ),
        (
            mechanism.DE[STATE],
            PREFIX + f"DE.{STATE}.",
            ("state_coefficient", "state_scalar", "state_table"),
        ),
    ):
        for name in names:
            value = workspace_values[prefix + name]
            expected_shape = (
                tuple(owner.shape_p)
                if name.endswith("coefficient")
                else (
                    ()
                    if name.endswith("scalar")
                    else (1, 2 if prefix == PREFIX else 3, 1)
                )
            )
            assert value.shape == expected_shape
            assert (
                value.dtype == initial.v_init.dtype
                and value.device == initial.v_init.device
            )
            assert torch.isfinite(value).all() and torch.count_nonzero(value)
            assert value.requires_grad
            assert all(
                leaf.module_path != prefix.removesuffix(".") or leaf.buffer_name != name
                for leaf in runner.functional._state_layout
            )


def _exercise(
    kind="path",
    *,
    device="cpu",
    dtype=torch.float64,
    shell_resistance_scale=1.0,
    backend="eager",
    execution="scan",
    checkpoint_every=None,
):
    model = _model(
        kind, device=device, dtype=dtype, shell_resistance_scale=shell_resistance_scale
    )
    runner = _runner(
        model, backend=backend, execution=execution, checkpoint_every=checkpoint_every
    )
    source = _module_snapshot(model)
    parameters, constants, initial, inputs, targets = _arguments(runner, model)
    initialized = runner.functional.initialize(parameters, constants, initial)
    _assert_initial_zero_contract(runner, parameters, constants, initial, initialized)
    result = runner(parameters=parameters, initialization=initial, inputs=inputs)
    actual = _recorded(result)
    gradients = _gradients(actual, targets) if execution == "scan" else {}
    parameters, constants, initial, inputs, targets = _arguments(runner, model)
    if execution == "while":
        with torch.no_grad():
            expected = _explicit_reference(
                runner, parameters, constants, initial, inputs
            )
    else:
        expected = _explicit_reference(runner, parameters, constants, initial, inputs)
    _close(actual, _recorded(expected), dtype=dtype)
    _close(result[0], expected[0], dtype=dtype)
    for leaf in runner.functional._state_layout:
        value = _read_path(result[0], leaf.public_path)
        assert value.device == leaf.device and value.dtype == leaf.dtype
    if execution == "scan":
        _close(gradients, _gradients(_recorded(expected), targets), dtype=dtype)
    assert actual["v"].shape == (STEPS + 1, *model.shape)
    assert (
        actual["t"][0].item() == 0 and torch.count_nonzero(actual["i_membrane"][0]) == 0
    )
    assert runner._prepared is None
    _assert_module_snapshot(model, source)
    return actual, gradients


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_fresh_timestep_buffers_full_traces_and_all_gradients(kind, cpu_threads):
    _exercise(kind)


@pytest.mark.parametrize(
    ("execution", "checkpoint_every"), (("scan", 2), ("while", None))
)
def test_fresh_timestep_buffers_compiled_runner_modes(
    execution, checkpoint_every, cpu_threads
):
    _exercise(execution=execution, checkpoint_every=checkpoint_every)


@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_fresh_timestep_buffers_real_cpp_execution(execution, cpu_threads):
    _exercise(
        backend="inductor",
        execution=execution,
        checkpoint_every=2 if execution == "scan" else None,
    )


def test_initialization_never_dispatches_timestep_builders(monkeypatch, cpu_threads):
    import dendra.func._population as population_module

    runner = _runner(_model())
    parameters, constants, initial, _inputs, _targets = _arguments(
        runner, runner.functional._population
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("timestep builder was dispatched")

    monkeypatch.setattr(population_module, "_evaluate_timestep_buffers", forbidden)
    runner.functional.initialize(parameters, constants, initial)
    with pytest.raises(AssertionError, match="timestep builder was dispatched"):
        runner.functional.prepare(parameters, constants)


@pytest.mark.parametrize("kind", ("path", "multi_block"))
def test_fresh_timestep_buffers_match_actual_imperative_initialize(kind, cpu_threads):
    model = _model(kind)
    runner = _runner(model)
    parameters, constants, initial, inputs, _targets = _arguments(runner, model)
    actual = _recorded(
        runner(parameters=parameters, initialization=initial, inputs=inputs)
    )
    expected = _imperative_reference(
        runner, _model(kind), parameters, constants, initial, inputs
    )
    _close(actual, expected)


def test_fresh_timestep_buffers_gradients_match_imperative_finite_differences(
    cpu_threads,
):
    model = _model()
    runner = _runner(model)
    parameters, constants, initial, inputs, targets = _arguments(runner, model)
    result = runner(parameters=parameters, initialization=initial, inputs=inputs)
    gradients = _gradients(_recorded(result), targets)
    reference = _model()
    base_drive = inputs.intra.detach() / targets["stimulus"].detach()
    # Perturb actual public initialization + runtime execution, not another
    # functional initializer. Directions cover the full initial-voltage field.
    for name, target in targets.items():
        direction = torch.linspace(
            0.7, 1.3, target.numel(), dtype=target.dtype, device=target.device
        ).reshape(target.shape)
        scale = max(float(target.detach().abs().amax()), 1.0e-3)
        epsilon = 1e-4 * scale
        losses = []
        for sign in (-1.0, 1.0):
            with torch.no_grad():
                target.add_(sign * epsilon * direction)
            local_inputs = (
                inputs._replace(intra=base_drive * target)
                if name == "stimulus"
                else inputs
            )
            losses.append(
                _loss(
                    _imperative_reference(
                        runner, reference, parameters, constants, initial, local_inputs
                    )
                )
            )
            with torch.no_grad():
                target.sub_(sign * epsilon * direction)
        numerical = (losses[1] - losses[0]) / (2.0 * epsilon)
        expected = (gradients[name] * direction).sum()
        torch.testing.assert_close(
            numerical,
            expected,
            rtol=3e-4,
            atol=1e-9,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.parametrize("policy", ("snapshot", "fresh"))
def test_timestep_buffers_snapshot_and_explicit_continuation(
    policy, monkeypatch, cpu_threads
):
    runner = _runner(_model(), initialization=policy)
    first = runner()
    if policy == "snapshot":
        torch.testing.assert_close(
            _recorded(first)["t"][0], runner.initial_state["clock"]["t"]
        )
    else:
        assert _recorded(first)["t"][0].item() == 0

    def forbidden(*args, **kwargs):
        pytest.fail("continuation must not reinitialize")

    monkeypatch.setattr(runner.functional, "initialize", forbidden)
    parameters = first[1]["parameters"] if policy == "fresh" else runner.parameters
    continued = runner(
        first[0], parameters=parameters, callback_state=first[1]["callbacks"].state
    )
    prepared = runner.functional.prepare(parameters, runner.constants)
    expected = dn.func.run(
        runner.functional.bind(parameters, prepared),
        first[0],
        steps=STEPS,
        callbacks=runner._callbacks,
        callback_state=first[1]["callbacks"].state,
    )
    _close(continued[0], expected[0])
    _close(_recorded(continued), _recorded(expected))
    assert _recorded(continued)["v"].shape[0] == STEPS


def test_zero_timestep_initialization_is_transformable_and_dt_independent(cpu_threads):
    model = _model()
    runner = _runner(model)
    parameters, constants, initial, _inputs, _targets = _arguments(runner, model)

    def values(voltage, dt):
        local_constants = {**constants, "dt": dt}
        initialized = runner.functional.initialize(
            parameters, local_constants, initial._replace(v_init=voltage)
        )
        return initialized.state["mechanisms"][MECHANISM]["activation"]

    expected = values(initial.v_init, constants["dt"])
    reverse = torch.func.jacrev(values, argnums=1)(initial.v_init, constants["dt"])
    forward = torch.func.jacfwd(values, argnums=1)(initial.v_init, constants["dt"])
    torch.testing.assert_close(reverse, torch.zeros_like(expected), rtol=0, atol=0)
    torch.testing.assert_close(forward, torch.zeros_like(expected), rtol=0, atol=0)
    voltages = torch.stack((initial.v_init, initial.v_init + 0.5))
    times = torch.tensor((DT, 2 * DT), dtype=model.dtype())
    actual = torch.vmap(values)(voltages, times)
    torch.testing.assert_close(
        actual,
        torch.stack([values(v, dt) for v, dt in zip(voltages, times, strict=True)]),
    )
    empty = torch.vmap(values)(voltages[:0], times[:0])
    assert empty.shape == (0, *expected.shape)
    hessian = torch.func.grad(
        torch.func.grad(lambda dt: values(initial.v_init, dt).sum())
    )(constants["dt"])
    torch.testing.assert_close(
        hessian, torch.zeros_like(constants["dt"]), rtol=0, atol=0
    )
    compiled_values = torch.compile(values, backend="eager", fullgraph=True)
    torch.testing.assert_close(
        compiled_values(initial.v_init, constants["dt"]), expected, rtol=0, atol=0
    )
    jacobian = torch.func.jacrev(values, argnums=0)
    compiled_jacobian = torch.compile(jacobian, backend="eager", fullgraph=True)
    torch.testing.assert_close(
        compiled_jacobian(initial.v_init, constants["dt"]),
        jacobian(initial.v_init, constants["dt"]),
        rtol=0,
        atol=0,
    )


class _ShiftVoltage(torch.nn.Module):
    def forward(self, voltage, offset):
        return (voltage + offset,)


def test_timestep_initialization_composes_with_pure_initialization_transforms(
    cpu_threads,
):
    model = _model()
    model.register_pre_initialize_transform(
        "shift",
        _ShiftVoltage(),
        reads=("state.integrator.v",),
        writes=("state.integrator.v",),
        inputs={"offset": model.v.new_tensor(0.3)},
    )
    runner = _runner(model)
    parameters, constants, initial, inputs, targets = _arguments(runner, model)
    initial.transforms["pre.shift.offset"].requires_grad_()
    targets["offset"] = initial.transforms["pre.shift.offset"]
    actual = _recorded(
        runner(parameters=parameters, initialization=initial, inputs=inputs)
    )
    gradients = _gradients(actual, targets, retain_graph=True)
    expected = _recorded(
        _explicit_reference(runner, parameters, constants, initial, inputs)
    )
    _close(actual, expected)
    _close(gradients, _gradients(expected, targets))


class _ZeroDtRebindingDerived(_InitializationTimestepMechanism):
    Mechanism.DERIVED_BUFFER("audit_bias")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reject_zero_context = False

    def derive_buffers(self):
        if self.reject_zero_context and self.dt == 0:
            self.mechanism_coefficient = self.mechanism_coefficient + 1.0
        return {"audit_bias": self.scale * torch.ones_like(self.diam)}


class _ZeroDtPythonMutationDerived(_InitializationTimestepMechanism):
    Mechanism.DERIVED_BUFFER("audit_bias")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_context_calls = 0
        self.reject_zero_context = False

    def derive_buffers(self):
        if self.reject_zero_context and self.dt == 0:
            self.zero_context_calls += 1
        return {"audit_bias": self.scale * torch.ones_like(self.diam)}


@pytest.mark.parametrize(
    ("mechanism", "message"),
    (
        (_ZeroDtRebindingDerived, "mutated or rebound registered preparation inputs"),
        (_ZeroDtPythonMutationDerived, "Python instance state"),
    ),
)
def test_zero_context_derived_mutation_fails_without_touching_source(
    mechanism, message
):
    model = _model(mechanism=mechanism)
    owner = model.mech.mechanisms[mechanism.__name__]
    # The ordinary initializer already rejects illegal builders. Enable this
    # adversary only after a legal model has a nonzero runtime timestep, so
    # functional lowering must audit the otherwise-unvisited zero context.
    owner.reject_zero_context = True
    snapshot = _module_snapshot(model)
    calls = getattr(owner, "zero_context_calls", None)
    # Mutation detection is an optional diagnostic captured by the plan.
    with dn.ctx(CHECK_PURITY=True):
        functional, tensors = dn.func.make_functional(model, dt=DT)
        with pytest.raises(dn.func.FunctionalizationError, match=message):
            functional.initialize(
                tensors.parameters, tensors.constants, tensors.initialization
            )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    continued, _ = functional.step(tensors.parameters, prepared, tensors.state)
    assert torch.isfinite(continued["integrator"]["v"]).all()
    _assert_module_snapshot(model, snapshot)
    assert getattr(owner, "zero_context_calls", None) == calls
