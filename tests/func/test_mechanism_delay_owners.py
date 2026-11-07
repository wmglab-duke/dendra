"""Pure VoltageProcess hooks own delay histories through restart and autograd."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, VoltageProcess
from tests.func.test_mechanism_delay_execution import DT, STEPS, _clone, _close

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    grad = torch.is_grad_enabled()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
            assert torch.is_grad_enabled()
    finally:
        torch.set_grad_enabled(grad)
        torch.set_num_threads(threads)


class _DelayedVoltage(VoltageProcess):
    VoltageProcess.GLOBAL(drive=0.7, gain=0.25)
    VoltageProcess.CARRY("owned", "delayed")
    multistream = False

    def __init__(self, *args, **kwargs):
        Mechanism.__init__(self, *args, **kwargs)
        payload = self.drive.expand(self.shape_f)
        if self.multistream:
            self.register_delayed_states(
                "signal",
                torch.stack((payload, payload)),
                (0, 2),
                stream_axis=0,
                delay_axis=-1,
                mode="shift",
            )
        else:
            self.register_delayed_state(
                "signal", payload, 2, mode="shift", insert_axis=0
            )

    def initial_values(self, v, values):
        return {
            "owned": v,
            "delayed": torch.zeros_like(v),
            **self.initial_delay_values(values),
        }

    def voltage_values(self, v, dt, values):
        del v
        signal = torch.sigmoid((values["owned"] + 55.0) / 15.0)
        if self.multistream:
            delayed, updates = self.delayed_states_values(
                "signal", torch.stack((signal, signal)), values
            )
            delayed = delayed.sum(dim=0)
        else:
            delayed, updates = self.delayed_state_values("signal", signal, values)
        owned = values["owned"] + dt * self.drive + self.gain * delayed
        return owned, {**updates, "owned": owned, "delayed": delayed}


class _MultiDelayedVoltage(_DelayedVoltage):
    multistream = True
    supports_population_axis_multistream_delays = True


OWNERS = (_DelayedVoltage, _MultiDelayedVoltage)


def _population(owner=_DelayedVoltage):
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-47.0, integrator=dn.scnv(imem=False)
    )
    population.insert(owner.rename("driver"))
    population.batch(2)
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    population.detach()
    return population


@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_voltage_phase_delay_recording_continuation_commit_and_checkpoint(
    owner, execution
):
    source, native = _population(owner), _population(owner)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        execution=execution,
        initialization="fresh",
        inference=execution == "while",
        backend="aot_eager",
        callbacks={
            "trace": dn.func.Recorder(("v", "driver.owned", "driver.delayed", "t"))
        },
    )
    before = _clone(runner.functional.extract())
    prefix, first = runner()
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    expected = [native.mech.driver.delayed.clone()]
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step(DT)
            expected.append(native.mech.driver.delayed.clone())
    observed = torch.cat(
        (
            first["callbacks"]["trace"]["driver.delayed"],
            second["callbacks"]["trace"]["driver.delayed"],
        )
    )
    _close(observed, torch.stack(expected))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)
    committed, restored = _population(owner), _population(owner)
    runner.functional.commit_state_(committed, final)
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step(DT)
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("checkpoint_every", (None, 2))
def test_voltage_phase_delays_preserve_native_parameter_and_history_gradients(
    owner, checkpoint_every
):
    source, native = _population(owner), _population(owner)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*gain_param", "*drive_param"),
        checkpoint_every=checkpoint_every,
        backend="aot_eager",
    )
    state = _clone(runner.initial_state)
    history = state["mechanism_buffers"]["driver"][
        "signal_delay_buffer"
    ].requires_grad_()
    with torch.no_grad():
        history.fill_(0.2)
    native_history = history.detach().clone().requires_grad_()
    native.mech.driver.signal_delay_buffer = native_history
    native.mech.driver.populate()
    final, _ = runner(state)
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().sum(),
        (
            history,
            runner.get_parameter("*gain_param"),
            runner.get_parameter("*drive_param"),
        ),
    )
    expected = torch.autograd.grad(
        native.v.square().sum(),
        (native_history, native.mech.driver.gain_param, native.mech.driver.drive_param),
    )
    for observed, wanted in zip(actual, expected, strict=True):
        assert wanted.abs().sum() > 0 and torch.isfinite(wanted).all()
        _close(observed, wanted)


@pytest.mark.parametrize("owner", OWNERS)
@pytest.mark.parametrize("cache", (False, True))
def test_voltage_phase_delay_fresh_initialization_and_population_cache(owner, cache):
    source, native = _population(owner), _population(owner)
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
    history = initialized.state["mechanism_buffers"]["driver"]["signal_delay_buffer"]
    assert bool(history.ne(0).any()) == cache
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    native.step(DT)
    _close(actual, functional.extract(native).state)
    _close(tensors, before)


@pytest.mark.parametrize("kind", ("population", "network"))
def test_cached_delay_vector_restores_native_routing_with_same_capacity(kind):
    from tests.func.test_mechanism_delay_execution import _MultiDelayedLeak, _network
    from tests.func.test_mechanism_delay_execution import _population as make_population

    def make():
        return (
            make_population(_MultiDelayedLeak)
            if kind == "population"
            else _network(_MultiDelayedLeak, connected=False)
        )

    source, native = make(), make()
    for model in (source, native):
        with torch.no_grad():
            for _ in range(3):
                model.step(DT) if kind == "population" else model.step()
            populations = (
                (model,) if kind == "population" else tuple(model.populations.values())
            )
            for population in populations:
                population.cache("_steady_state")
            if kind == "network":
                model.cache_state()
            for population in populations:
                population.mech.delays.signal_delay_steps.copy_(
                    torch.tensor((0, 2, 3), dtype=torch.long)
                )
    if kind == "population":
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
        constants = initialized.constants
    else:
        functional, tensors = dn.func.make_functional_network(source)
        before = _clone(tensors)
        initialized = functional.initialize(*tensors, functional.initialization_inputs)
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
        constants = initialized.constants["populations"]["left"]
    steps = next(
        value
        for name, value in constants.items()
        if name.endswith("signal_delay_steps")
    )
    torch.testing.assert_close(steps, steps.new_tensor((0, 1, 3)))
    _close(initialized.state, functional.extract(native).state)
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    native.step(DT) if kind == "population" else native.step()
    _close(actual, functional.extract(native).state)
    committed = make()
    functional.commit_state_(committed, actual)
    _close(functional.extract(committed).state, actual)
    _close(tensors, before)
