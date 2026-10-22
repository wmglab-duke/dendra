"""TABLE dispatch retains native trajectories, parameter gradients, and restarts."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import graded_syn
from dendra.models.multi import MultiPopulation
from tests.func.test_network_cache_initialization import (
    _cache_initialized_native_packet,
)

DT = 0.125
STEPS = 4
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _curve(value):
    return 0.5 + torch.sigmoid((value + 50.0) / 15.0)


class _TableLeakBase(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.CARRY("gain")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def curve(self, v):
        return _curve(v)

    def initial_values(self, v, values):
        del values
        return {"gain": self.curve(v)}

    def advance(self, v, dt, values):
        del dt, values
        return {"gain": self.curve(v)}

    def i_affine_terms(self, v):
        del v
        conductance = self.g * self.gain
        return conductance, conductance * self.e


class _FixedTableLeak(_TableLeakBase):
    Mechanism.TABLE("curve", -100.0, 20.0, 9)


class _LearnableTableLeak(_TableLeakBase):
    Mechanism.TABLE("curve", -100.0, 20.0, 9, learnable=True)


class _NumericalTableLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.TABLE("curve", -100.0, 20.0, 9, learnable=True)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def curve(self, v):
        return _curve(v)

    def i(self, v):
        return self.g * self.curve(v) * (v - self.e)


class _TableGateBase(State):
    State.STATE("x")
    State.RANGE(tau=2.0)
    State.ASSIGNED("target")
    State.DERIVATIVE("x' = (target - x) / tau")
    State.METHOD("cnexp")

    def curve(self, v):
        return _curve(v)

    def assigned_values(self, v, values):
        del values
        return {"target": self.curve(v)}

    def state_defaults(self, v, values):
        del values
        return {"x": self.curve(v)}


class _FixedGate(_TableGateBase):
    State.TABLE("curve", -100.0, 20.0, 9)


class _LearnableGate(_TableGateBase):
    State.TABLE("curve", -100.0, 20.0, 9, learnable=True)


class _GatedLeakBase(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        conductance = self.g * self.x
        return conductance, conductance * self.e


class _FixedStateTableLeak(_GatedLeakBase):
    Mechanism.STATE_BUNDLE(_FixedGate)


class _LearnableStateTableLeak(_GatedLeakBase):
    Mechanism.STATE_BUNDLE(_LearnableGate)


MECHANISMS = (
    _FixedTableLeak,
    _LearnableTableLeak,
    _FixedStateTableLeak,
    _LearnableStateTableLeak,
)
LEARNABLE = (_LearnableTableLeak, _LearnableStateTableLeak)


def _clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)


def _owner(population):
    mechanism = population.mech.tabulated
    return next(iter(mechanism.DE.values())) if mechanism.DE else mechanism


def _population(mechanism=_LearnableTableLeak, *, enabled=True):
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-47.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(mechanism.rename("tabulated"))
    population.train()
    population.initialize()
    _owner(population).usetables(enabled)
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    population.detach()
    return population


def _network(mechanism=_LearnableTableLeak, *, connected=True, shared=False):
    left, right = _population(mechanism), _population(mechanism)
    if connected:
        right.insert(graded_syn.rename("graded"), e=-10.0)
    network = dn.Network({"left": left, "right": right})
    if connected:
        network.connect_continuous_one_to_one(
            left[:],
            right[:],
            right.mech.graded,
            pre_var="v",
            input="g_pre",
            weight=0.001,
            delay=0.0,
            transform=torch.sigmoid,
        )
    network.train()
    network.initialize(DT)
    if shared:
        _owner(right).curve_table.y = _owner(left).curve_table.y
    for population in network.populations.values():
        population.detach()
    return network


@pytest.mark.parametrize("mechanism", MECHANISMS)
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_population_table_recording_continuation_commit_and_native_checkpoint(
    mechanism, execution
):
    source, native = _population(mechanism), _population(mechanism)
    value_name = "tabulated.x" if source.mech.tabulated.DE else "tabulated.gain"
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=execution == "while",
        execution=execution,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("v", value_name, "t"))},
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    before = _clone(runner.functional.extract())
    prefix, first = runner()
    replay, _ = runner()
    _close(prefix, replay)
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    expected = [native.v.clone()]
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step(DT)
            expected.append(native.v.clone())
    recorded = torch.cat(
        (first["callbacks"]["trace"]["v"], second["callbacks"]["trace"]["v"])
    )
    _close(recorded, torch.stack(expected))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)
    committed = _population(mechanism)
    runner.functional.commit_state_(committed, final)
    _close(runner.functional.extract(committed).state, final)
    restored = _population(mechanism)
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step(DT)
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("mechanism", LEARNABLE)
@pytest.mark.parametrize("checkpoint_every", (None, 2))
def test_table_values_and_query_gradients_match_native_with_checkpointed_scan(
    mechanism, checkpoint_every
):
    source, native = _population(mechanism), _population(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*curve_table.y", "*g_param"),
        checkpoint_every=checkpoint_every,
        backend="aot_eager",
    )
    state = _clone(runner.initial_state)
    query = state["integrator"]["v"].requires_grad_()
    native.v = native.v.detach().clone().requires_grad_()
    native_query = native.v
    native.mech.tabulated.populate()
    selected = runner.get_parameter("*curve_table.y")
    conductance = runner.get_parameter("*g_param")
    final, _ = runner(state)
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().sum(), (selected, conductance, query)
    )
    expected = torch.autograd.grad(
        native.v.square().sum(),
        (_owner(native).curve_table.y, native.mech.tabulated.g_param, native_query),
    )
    for observed, wanted in zip(actual, expected, strict=True):
        assert wanted.abs().sum() > 0 and torch.isfinite(wanted).all()
        _close(observed, wanted)


def test_table_in_numerical_current_preserves_voltage_and_table_value_gradients():
    source, native = _population(_NumericalTableLeak), _population(_NumericalTableLeak)
    native.mech.tabulated.populate()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*curve_table.y",),
        checkpoint_every=2,
        backend="aot_eager",
    )
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().sum(), runner.get_parameter("*curve_table.y")
    )[0]
    expected = torch.autograd.grad(
        native.v.square().sum(), _owner(native).curve_table.y
    )[0]
    assert expected.abs().sum() > 0
    _close(actual, expected)


@pytest.mark.parametrize("mechanism", LEARNABLE)
def test_network_table_shared_parameter_aliases_and_native_gradient_accumulation(
    mechanism,
):
    source, native = (
        _network(mechanism, shared=True),
        _network(mechanism, shared=True),
    )
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*right*curve_table.y",),
        checkpoint_every=2,
        backend="aot_eager",
    )
    left = runner.get_parameter("*left*curve_table.y")
    assert left is runner.get_parameter("*right*curve_table.y")
    assert left is not _owner(source.left).curve_table.y
    final, _ = runner()
    for _ in range(STEPS):
        native.step()
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        sum(
            value["integrator"]["v"].square().sum()
            for value in final["populations"].values()
        ),
        left,
    )[0]
    expected = torch.autograd.grad(
        sum(population.v.square().sum() for population in native.populations.values()),
        _owner(native.left).curve_table.y,
    )[0]
    assert expected.abs().sum() > 0
    _close(actual, expected)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_connected_network_table_recording_and_checkpoint_continuation(execution):
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=execution == "while",
        execution=execution,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
    )
    before = _clone(runner.functional.extract())
    prefix, first = runner()
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    expected = {
        name: [population.v.clone()] for name, population in native.populations.items()
    }
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step()
            for name, population in native.populations.items():
                expected[name].append(population.v.clone())
    for name, values in expected.items():
        observed = torch.cat(
            (
                first["callbacks"]["trace"][f"{name}.v"],
                second["callbacks"]["trace"][f"{name}.v"],
            )
        )
        _close(observed, torch.stack(values))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)
    committed = _network()
    runner.functional.commit_state_(committed, final)
    restored = _network()
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step()
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("mechanism", LEARNABLE)
@pytest.mark.parametrize("cache", (False, True))
def test_fresh_network_initialization_preserves_learned_table_values_and_cache(
    cache, mechanism
):
    source, native = (
        _network(mechanism, connected=False),
        _network(mechanism, connected=False),
    )
    for network in (source, native):
        saved = {
            name: _owner(population).curve_table.y.detach().clone()
            for name, population in network.populations.items()
        }
        with torch.no_grad():
            for _ in range(2):
                network.step()
            if cache:
                for population in network.populations.values():
                    population.cache("_steady_state")
                network.cache_state()
            for population in network.populations.values():
                _owner(population).curve_table.y.add_(0.15)
            for _ in range(2):
                network.step()
    functional, tensors = dn.func.make_functional_network(source)
    before = _clone(tensors)
    parameters = (
        tensors.independent_parameters(trainable=("*curve_table.y",))
        if cache
        else tensors.parameters
    )
    initialized = functional.initialize(
        parameters, tensors.constants, tensors.state, functional.initialization_inputs
    )
    native_leaf = _owner(native.left).curve_table.y
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    assert _owner(native.left).curve_table.y is native_leaf
    for name, population in native.populations.items():
        expected_y = saved[name] if cache else saved[name] + 0.15
        _close(_owner(population).curve_table.y, expected_y)
        selected = [
            value
            for key, value in initialized.parameters["populations"][name].items()
            if key.endswith("curve_table.y")
        ]
        assert len(selected) == 1
        _close(selected[0], expected_y)
    expected = (
        _cache_initialized_native_packet(functional, native)
        if cache
        else functional.extract(native)
    )
    _close(initialized, expected)
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)
    if cache:
        key = next(
            name
            for name in parameters["populations"]["left"]
            if name.endswith("curve_table.y")
        )
        actual_gradient = torch.autograd.grad(
            actual["populations"]["left"]["integrator"]["v"].square().sum(),
            parameters["populations"]["left"][key],
        )[0]
        expected_gradient = torch.autograd.grad(
            native.left.v.square().sum(), native_leaf
        )[0]
        assert expected_gradient.abs().sum() > 0
        _close(actual_gradient, expected_gradient)
    _close(tensors, before)


@pytest.mark.parametrize("mechanism", (_FixedTableLeak, _FixedStateTableLeak))
def test_cached_fixed_table_restores_all_interpolation_buffers_and_trajectory(
    mechanism,
):
    source, native = (
        _network(mechanism, connected=False),
        _network(mechanism, connected=False),
    )
    saved = _owner(native.left).curve_table._y.clone()
    for network in (source, native):
        with torch.no_grad():
            network.step()
            for population in network.populations.values():
                population.cache("_steady_state")
            network.cache_state()
            for population in network.populations.values():
                # The y0 and endpoint buffers are views of _y; a uniform shift
                # leaves the prepared slopes valid while changing the function.
                _owner(population).curve_table._y.add_(0.15)
            network.step()
    _close(_owner(source.left).curve_table._y, saved + 0.15)
    functional, tensors = dn.func.make_functional_network(source)
    before = _clone(tensors)
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(_owner(native.left).curve_table._y, saved)
    _close(initialized, _cache_initialized_native_packet(functional, native))
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)
    _close(tensors, before)


@pytest.mark.parametrize("mechanism", (_FixedTableLeak, _FixedStateTableLeak))
def test_population_owned_cache_restores_fixed_table_evaluation(mechanism):
    source, native = _population(mechanism), _population(mechanism)
    saved = _owner(native).curve_table._y.clone()
    for population in (source, native):
        with torch.no_grad():
            population.step(DT)
            population.cache("_steady_state")
            _owner(population).curve_table._y.add_(0.15)
            population.step(DT)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    before = _clone(tensors)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    _close(_owner(native).curve_table._y, saved)
    expected = functional.extract(native)
    _close(initialized.state, expected.state)
    for name, value in initialized.constants.items():
        if "curve_table." in name:
            _close(value, expected.constants[name])
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    native.step(DT)
    _close(actual, functional.extract(native).state)
    _close(tensors, before)


@pytest.mark.parametrize("mechanism", (_FixedTableLeak, _FixedStateTableLeak))
def test_fixed_table_runtime_reads_explicit_constants_instead_of_source_buffers(
    mechanism,
):
    source, native = _population(mechanism), _population(mechanism)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    before = _clone(tensors)
    constants = dict(tensors.constants)
    changed = []
    for name, value in constants.items():
        if name.endswith(
            tuple(
                f"curve_table.{suffix}"
                for suffix in ("_y", "_y0", "_y_first", "_y_last")
            )
        ):
            constants[name] = value + 0.15
            changed.append(name)
    assert changed
    with torch.no_grad():
        _owner(native).curve_table._y.add_(0.15)
    actual, _ = functional.prepare_and_step(
        tensors.parameters, constants, tensors.state
    )
    native.step(DT)
    _close(actual, functional.extract(native).state)
    _close(tensors, before)


@pytest.mark.parametrize("mechanism", LEARNABLE)
def test_population_fresh_initialization_uses_current_table_values_with_gradients(
    mechanism,
):
    source, native = _population(mechanism), _population(mechanism)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    parameters = tensors.independent_parameters(trainable=("*curve_table.y",))
    name = tensors.parameter_name("*curve_table.y")
    leaf = parameters[name]
    parameters[name] = leaf + 0.15
    with torch.no_grad():
        _owner(native).curve_table.y.add_(0.15)
    initialized = functional.initialize(
        parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    _close(initialized.state, functional.extract(native).state)
    field = "x" if source.mech.tabulated.DE else "gain"
    actual_value = (
        initialized.state["mechanisms"]["tabulated"][field]
        if source.mech.tabulated.DE
        else initialized.state["mechanism_buffers"]["tabulated"]["gain"]
    )
    # -47 mV lies between the -55 and -40 mV samples. This oracle avoids
    # reusing the shared lookup implementation in the gradient assertion.
    expected_value = (
        parameters[name][3] * (7.0 / 15.0) + parameters[name][4] * (8.0 / 15.0)
    ).expand_as(actual_value)
    _close(actual_value, expected_value)
    actual = torch.autograd.grad(actual_value.sum(), leaf)[0]
    expected = torch.autograd.grad(expected_value.sum(), leaf)[0]
    assert expected.abs().sum() > 0
    _close(actual, expected)
    _close(parameters[name], leaf + 0.15)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_disabled_table_uses_original_function_and_preserves_native_flag(mechanism):
    source, native = (
        _population(mechanism, enabled=False),
        _population(mechanism, enabled=False),
    )
    assert not _owner(source).usetable_curve
    runner = dn.compile(
        source, dt=DT, steps=STEPS, inference=False, backend="aot_eager"
    )
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    assert not _owner(source).usetable_curve


def _layout_population(layout, mechanism):
    if layout == "batch":
        population = _population(mechanism)
        population.batch(3)
    elif layout == "regional":
        from dendra.models.mod import pas

        population = dn.SingleCompartment(
            N=2, C=3, v_init=-47.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(pas)
        population[:, :2].insert(mechanism.rename("tabulated"))
    else:
        population = dn.concat_models(
            {"left": _population(mechanism), "right": _population(mechanism)},
            threads=1,
        )
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    population.detach()
    return population


@pytest.mark.parametrize("mechanism", LEARNABLE)
@pytest.mark.parametrize("layout", ("batch", "regional", "packed"))
def test_table_explicit_batch_regional_and_concatenated_layouts(layout, mechanism):
    source, native = (
        _layout_population(layout, mechanism),
        _layout_population(layout, mechanism),
    )
    if layout == "batch":
        assert source.v.shape == (3, 2, 2)
    elif layout == "regional":
        assert source.v.shape == (2, 3)
        field = "x" if source.mech.tabulated.DE else "gain"
        assert getattr(source.mech.tabulated, field).shape == (2, 2)
    else:
        assert isinstance(source, MultiPopulation)
        assert set(source.populations) == {"left", "right"}
    runner = dn.compile(
        source, dt=DT, steps=STEPS, inference=False, backend="aot_eager"
    )
    before = _clone(runner.functional.extract())
    final, _ = runner()
    with torch.no_grad():
        for _ in range(STEPS):
            native.step(DT)
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)


@pytest.mark.parametrize("mechanism", (_FixedTableLeak, _FixedStateTableLeak))
def test_concatenated_fixed_tables_evaluate_supplied_constants(mechanism):
    source, native = (
        _layout_population("packed", mechanism),
        _layout_population("packed", mechanism),
    )
    functional, tensors = dn.func.make_functional(source, dt=DT)
    before = _clone(tensors)
    constants = dict(tensors.constants)
    changed = []
    for name, value in constants.items():
        if name.endswith(
            tuple(
                f"curve_table.{suffix}"
                for suffix in ("_y", "_y0", "_y_first", "_y_last")
            )
        ):
            constants[name] = value + 0.15
            changed.append(name)
    assert changed
    with torch.no_grad():
        _owner(native).curve_table._y.add_(0.15)
    baseline, _ = functional.prepare_and_step(*tensors[:3])
    actual, _ = functional.prepare_and_step(
        tensors.parameters, constants, tensors.state
    )
    native.step(DT)
    _close(actual, functional.extract(native).state)
    field = "x" if source.mech.tabulated.DE else "gain"
    family = "mechanisms" if source.mech.tabulated.DE else "mechanism_buffers"
    assert not torch.equal(
        actual[family]["tabulated"][field], baseline[family]["tabulated"][field]
    )
    _close(tensors, before)
