"""Mechanism delay histories preserve native phases, gradients, and restarts."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import graded_syn, pas
from tests.func.test_network_cache_initialization import (
    _cache_initialized_native_packet,
)

DT = 0.125
STEPS = 4
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    previous_grad = torch.is_grad_enabled()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
            assert torch.is_grad_enabled(), "delay execution leaked disabled autograd"
    finally:
        torch.set_grad_enabled(previous_grad)
        torch.set_num_threads(previous)


class _DelayedLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.CARRY("delayed")
    Mechanism.CARRY("signal_delay_buffer", shape="deferred")
    Mechanism.CARRY("signal_delay_ptr", shape=(), dtype=torch.long)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")
    delay_mode = "shift"
    delay_steps = 2
    delay_axis = 0

    def __init__(self, *args, **kwargs):
        Mechanism.__init__(self, *args, **kwargs)
        self.register_delayed_state(
            "signal",
            self.g.expand(self.shape_f),
            self.delay_steps,
            mode=self.delay_mode,
            insert_axis=self.delay_axis,
        )

    def initial_values(self, v, values):
        return {
            "delayed": torch.zeros_like(v),
            "signal_delay_buffer": torch.zeros_like(values["signal_delay_buffer"]),
            "signal_delay_ptr": torch.zeros_like(values["signal_delay_ptr"]),
        }

    def advance(self, v, dt, values):
        del dt
        source = torch.sigmoid((v + 55.0) / 15.0)
        delayed, updates = self.delayed_state_values("signal", source, values)
        return {**updates, "delayed": delayed}

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.gain * self.delayed


class _CircularLeak(_DelayedLeak):
    delay_mode = "circular"


class _ZeroDelayLeak(_DelayedLeak):
    delay_steps = 0
    delay_axis = -1


class _MultiDelayedLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.CARRY("delayed")
    Mechanism.CARRY("signal_delay_buffer", shape="deferred")
    Mechanism.CARRY("signal_delay_ptr", shape=(), dtype=torch.long)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")
    supports_population_axis_multistream_delays = True
    delay_mode = "shift"
    delay_axis = 0

    def __init__(self, *args, **kwargs):
        Mechanism.__init__(self, *args, **kwargs)
        self.register_delayed_states(
            "signal",
            torch.stack((self.g.expand(self.shape_f),) * 3),
            (0, 1, 3),
            mode=self.delay_mode,
            stream_axis=0,
            delay_axis=self.delay_axis,
        )

    def initial_values(self, v, values):
        return {
            "delayed": torch.zeros_like(v),
            "signal_delay_buffer": torch.zeros_like(values["signal_delay_buffer"]),
            "signal_delay_ptr": torch.zeros_like(values["signal_delay_ptr"]),
        }

    def advance(self, v, dt, values):
        del dt
        source = torch.sigmoid((v + 55.0) / 15.0)
        streams = torch.stack((source, 1.3 * source, 0.7 * source))
        delayed, updates = self.delayed_states_values("signal", streams, values)
        return {**updates, "delayed": delayed.sum(dim=0)}

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.gain * self.delayed


class _MultiCircularLeak(_MultiDelayedLeak):
    delay_mode = "circular"
    delay_axis = -1


class _PersistentDelayedLeak(Mechanism):
    """Registered undeclared history persists unless the author resets it."""

    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.CARRY("delayed")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")
    delay_mode = "shift"
    delay_steps = 2
    delay_axis = 0

    def __init__(self, *args, **kwargs):
        Mechanism.__init__(self, *args, **kwargs)
        self.register_delayed_state(
            "signal", self.g.expand(self.shape_f), 2, mode="shift", insert_axis=0
        )

    advance = _DelayedLeak.advance
    i_affine_terms = _DelayedLeak.i_affine_terms

    def initial_values(self, v, values):
        del values
        return {"delayed": torch.zeros_like(v)}


class _ResetDelayedLeak(_PersistentDelayedLeak):
    def initial_values(self, v, values):
        return {"delayed": torch.zeros_like(v), **self.initial_delay_values(values)}


MECHANISMS = (
    _DelayedLeak,
    _CircularLeak,
    _ZeroDelayLeak,
    _MultiDelayedLeak,
    _MultiCircularLeak,
)


def _clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)


def _population(mechanism=_DelayedLeak, *, layout="dense"):
    population = dn.SingleCompartment(
        N=2, C=3, v_init=-47.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    if layout == "regional":
        population.insert(pas)
        population[:, :2].insert(mechanism.rename("delays"))
    else:
        population.insert(mechanism.rename("delays"))
    if layout == "batch":
        population.batch(2)
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    population.detach()
    return population


def _network(mechanism=_DelayedLeak, *, connected=True):
    left, right = _population(mechanism), _population(mechanism)
    if connected:
        right.insert(graded_syn.rename("graded"), e=-10.0)
    network = dn.Network({"left": left, "right": right})
    if connected:
        network.connect_continuous_one_to_one(
            left[:],
            right[:],
            right.mech.graded,
            pre_var="mech.delays.delayed",
            input="g_pre",
            weight=0.001,
            delay=0.0,
            transform=torch.sigmoid,
        )
    network.train()
    network.initialize(DT)
    for population in network.populations.values():
        population.detach()
    return network


@pytest.mark.parametrize("mechanism", MECHANISMS)
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_population_delay_recording_restart_commit_and_checkpoint(mechanism, execution):
    source, native = _population(mechanism), _population(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=execution == "while",
        execution=execution,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("v", "delays.delayed", "t"))},
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    before = _clone(runner.functional.extract())
    prefix, first = runner()
    assert torch.is_grad_enabled(), "compiled delays must retain caller grad mode"
    replay, _ = runner()
    _close(prefix, replay)
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    voltages, outputs = [native.v.clone()], [native.mech.delays.delayed.clone()]
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step(DT)
            voltages.append(native.v.clone())
            outputs.append(native.mech.delays.delayed.clone())
    for name, expected in (("v", voltages), ("delays.delayed", outputs)):
        observed = torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        _close(observed, torch.stack(expected))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)
    committed, restored = _population(mechanism), _population(mechanism)
    runner.functional.commit_state_(committed, final)
    _close(runner.functional.extract(committed).state, final)
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step(DT)
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize(
    "mechanism", (_DelayedLeak, _MultiDelayedLeak, _CircularLeak, _MultiCircularLeak)
)
@pytest.mark.parametrize("checkpoint_every", (None, 2))
def test_delay_gradients_match_native_for_parameters_voltage_and_history(
    mechanism, checkpoint_every
):
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    source, native = _population(mechanism), _population(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*g_param", "*gain_param"),
        checkpoint_every=checkpoint_every,
        backend="aot_eager",
    )
    state = _clone(runner.initial_state)
    voltage = state["integrator"]["v"].requires_grad_()
    history = state["mechanism_buffers"]["delays"][
        "signal_delay_buffer"
    ].requires_grad_()
    with torch.no_grad():
        history.fill_(0.25)
    native.v = voltage.detach().clone().requires_grad_()
    native_voltage = native.v
    native_history = history.detach().clone().requires_grad_()
    native.mech.delays.signal_delay_buffer = native_history
    native.mech.delays.populate()
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    final, _ = runner(state)
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert final["integrator"]["v"].requires_grad
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    assert final["integrator"]["v"].requires_grad
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().sum(),
        (
            runner.get_parameter("*g_param"),
            runner.get_parameter("*gain_param"),
            voltage,
            history,
        ),
    )
    expected = torch.autograd.grad(
        native.v.square().sum(),
        (
            native.mech.delays.g_param,
            native.mech.delays.gain_param,
            native_voltage,
            native_history,
        ),
    )
    for observed, wanted in zip(actual, expected, strict=True):
        assert wanted.abs().sum() > 0 and torch.isfinite(wanted).all()
        _close(observed, wanted)


@pytest.mark.parametrize(
    "mechanism", (_DelayedLeak, _CircularLeak, _MultiDelayedLeak, _MultiCircularLeak)
)
def test_explicit_history_is_read_only_and_circular_detaches_queued_sources(mechanism):
    source = _population(mechanism)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    state = _clone(tensors.state)
    voltage = state["integrator"]["v"].requires_grad_()
    history = state["mechanism_buffers"]["delays"][
        "signal_delay_buffer"
    ].requires_grad_()
    before = _clone((state, tensors))
    final, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, state, steps=1
    )
    output = final["mechanism_buffers"]["delays"]["delayed"]
    source_gradient, history_gradient = torch.autograd.grad(
        output.sum(), (voltage, history), allow_unused=True
    )
    # Reads retain incoming history gradients. Circular writes detach the
    # newly submitted source, while each zero-delay stream passes it through.
    assert history_gradient is not None and history_gradient.abs().sum() > 0
    if mechanism in (_MultiDelayedLeak, _MultiCircularLeak):
        assert source_gradient is not None and source_gradient.abs().sum() > 0
    else:
        assert source_gradient is None or source_gradient.count_nonzero() == 0
    _close((state, tensors), before)


@pytest.mark.parametrize("mechanism", (_DelayedLeak, _MultiCircularLeak))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_connected_network_delays_preserve_native_order_recording_and_continuation(
    mechanism, execution
):
    source, native = _network(mechanism), _network(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=execution == "while",
        execution=execution,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.delays.delayed", "t"))},
    )
    before = _clone(runner.functional.extract())
    prefix, first = runner()
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    expected = [native.right.mech.delays.delayed.clone()]
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step()
            expected.append(native.right.mech.delays.delayed.clone())
    observed = torch.cat(
        (
            first["callbacks"]["trace"]["right.delays.delayed"],
            second["callbacks"]["trace"]["right.delays.delayed"],
        )
    )
    _close(observed, torch.stack(expected))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)
    committed, restored = _network(mechanism), _network(mechanism)
    runner.functional.commit_state_(committed, final)
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step()
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("mechanism", (_DelayedLeak, _MultiCircularLeak))
@pytest.mark.parametrize("cache", (False, True))
def test_fresh_network_delay_initialization_restores_cache_or_clears_history(
    mechanism, cache
):
    source, native = (
        _network(mechanism, connected=False),
        _network(mechanism, connected=False),
    )
    for network in (source, native):
        with torch.no_grad():
            for _ in range(3):
                network.step()
            if cache:
                for population in network.populations.values():
                    population.cache("_steady_state")
                network.cache_state()
            network.step()
    functional, tensors = dn.func.make_functional_network(source)
    before = _clone(tensors)
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    expected = (
        _cache_initialized_native_packet(functional, native)
        if cache
        else functional.extract(native)
    )
    _close(initialized, expected)
    for population in initialized.state["populations"].values():
        history = population["mechanism_buffers"]["delays"]["signal_delay_buffer"]
        assert bool(history.ne(0).any()) == cache
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)
    _close(tensors, before)


@pytest.mark.parametrize("mechanism", (_DelayedLeak, _MultiDelayedLeak))
@pytest.mark.parametrize("layout", ("batch", "regional", "packed"))
def test_delay_histories_support_population_layouts(mechanism, layout):
    def make():
        if layout == "packed":
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
        return _population(mechanism, layout=layout)

    source, native = make(), make()
    runner = dn.compile(
        source, dt=DT, steps=STEPS, inference=False, backend="aot_eager"
    )
    before = _clone(runner.functional.extract())
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)


@pytest.mark.parametrize(
    "mechanism", (_DelayedLeak, _PersistentDelayedLeak, _ResetDelayedLeak)
)
@pytest.mark.parametrize("cache", (False, True))
def test_population_initialization_preserves_authored_reset_policy_and_cache(
    mechanism, cache
):
    source, native = _population(mechanism), _population(mechanism)
    for population in (source, native):
        with torch.no_grad():
            for _ in range(3):
                population.step(DT)
            if cache:
                population.cache("_steady_state")
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
    _close(initialized.state, functional.extract(native).state)
    history = initialized.state["mechanism_buffers"]["delays"]["signal_delay_buffer"]
    assert bool(history.ne(0).any()) == (cache or mechanism is _PersistentDelayedLeak)
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    native.step(DT)
    _close(actual, functional.extract(native).state)
    _close(tensors, before)


@pytest.mark.parametrize("mechanism", (_DelayedLeak, _MultiDelayedLeak))
def test_connected_network_delay_gradients_cross_population_histories(mechanism):
    source, native = _network(mechanism), _network(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=2 * STEPS,
        inference=False,
        trainable=("*left*gain_param", "*right*gain_param"),
        checkpoint_every=2,
        backend="aot_eager",
    )
    state = _clone(runner.initial_state)
    source_history = state["populations"]["left"]["mechanism_buffers"]["delays"][
        "signal_delay_buffer"
    ].requires_grad_()
    with torch.no_grad():
        source_history.fill_(0.25)
    native_history = source_history.detach().clone().requires_grad_()
    native.left.mech.delays.signal_delay_buffer = native_history
    for population in native.populations.values():
        population.mech.delays.populate()
    final, _ = runner(state)
    for _ in range(2 * STEPS):
        native.step()
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        final["populations"]["right"]["integrator"]["v"].square().sum(),
        (
            source_history,
            runner.get_parameter("*left*gain_param"),
            runner.get_parameter("*right*gain_param"),
        ),
    )
    expected = torch.autograd.grad(
        native.right.v.square().sum(),
        (
            native_history,
            native.left.mech.delays.gain_param,
            native.right.mech.delays.gain_param,
        ),
    )
    for observed, wanted in zip(actual, expected, strict=True):
        assert wanted.abs().sum() > 0 and torch.isfinite(wanted).all()
        _close(observed, wanted)
