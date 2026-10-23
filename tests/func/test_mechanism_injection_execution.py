"""Mechanism-owned waveforms preserve delivery, phases, and continuation."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, VoltageProcess
from dendra.models.mod import graded_syn, pas
from dendra.models.stim.waveform.core import Waveform
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


class _InjectedLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.ASSIGNED("i_inj")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        del kwargs
        return self.register_waveform_injection(
            waveform, index=index, model_shape=model_shape, model=model
        )

    def assigned_values(self, v, values):
        del values
        return {"i_inj": self.injection_values(v, t=self.t)}

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.gain * self.i_inj


class _NamedInjectedLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.ASSIGNED("i_a", "i_b")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        del kwargs
        accepted = self.register_waveform_injection(
            waveform,
            index=index,
            model_shape=model_shape,
            model=model,
            current_name="i_a",
            scale=1.5,
        )
        if accepted:
            self.register_waveform_injection(
                waveform,
                index=index,
                model_shape=model_shape,
                model=model,
                current_name="i_b",
                scale=-0.25,
            )
        return accepted

    def assigned_values(self, v, values):
        del values
        return {
            "i_a": self.injection_values(v, t=self.t, current_name="i_a"),
            "i_b": self.injection_values(v, t=self.t, current_name="i_b"),
        }

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.gain * (self.i_a + self.i_b)


class _InjectedVoltage(VoltageProcess):
    VoltageProcess.GLOBAL(gain=0.7)
    VoltageProcess.CARRY("owned")

    inject = _InjectedLeak.inject

    def initial_values(self, v, values):
        del values
        return {"owned": v}

    def voltage_values(self, v, dt, values):
        voltage = values["owned"] + dt * self.gain * self.injection_values(v, t=self.t)
        return voltage, {"owned": voltage}


class _CurrentClockProbe(_InjectedLeak):
    Mechanism.CARRY("seen")

    def initial_values(self, v, values):
        del values
        return {"seen": self.injection_values(v, t=self.t)}

    def advance(self, v, dt, values):
        del dt, values
        return {"seen": self.injection_values(v, t=self.t)}


class _SelectedWaveform(Waveform):
    FUNCTIONAL_PURE = True
    Waveform.PARAMETER(amp=0.7)

    def __init__(self, *, spatial=(0.4, 0.9), **kwargs):
        Waveform.__init__(self, **kwargs)
        self.register_buffer("spatial", torch.tensor(spatial))

    def fn(self, t):
        return self.amp * self.spatial[..., None] * (1.0 + 0.2 * t)


class _SelectedParameterWaveform(Waveform):
    FUNCTIONAL_PURE = True
    Waveform.PARAMETER(amp=0.7, spatial=(0.4, 0.9))

    def fn(self, t):
        return self.amp * self.spatial[..., None] * (1.0 + 0.2 * t)


def _clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)


def _waveform():
    return dn.sin(amp=0.7, freq=0.23, phase=0.4, delay=-0.5)


def _population(
    mechanism=_InjectedLeak, *, layout="dense", waveform=None, fallback=False
):
    population = dn.SingleCompartment(
        N=2,
        C=3,
        v_init=-47.0,
        integrator=(
            dn.scnv(imem=False)
            if mechanism is _InjectedVoltage
            else dn.bwd_euler_sc(imem=False)
        ),
    )
    if layout == "regional":
        population.insert(pas)
        population[:, :2].insert(mechanism.rename("owner"))
    else:
        population.insert(mechanism.rename("owner"))
    waveform = _waveform() if waveform is None else waveform
    population[:, :].inject(waveform) if fallback else population[:, 0].inject(waveform)
    if layout == "batch":
        population.batch(2)
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    population.detach()
    return population


def _network(mechanism=_InjectedLeak, *, shared=False, connected=True):
    waveform = _waveform() if shared else None
    left = _population(mechanism, waveform=waveform)
    right = _population(mechanism, waveform=waveform)
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
    for population in network.populations.values():
        population.detach()
    return network


@pytest.mark.parametrize(
    "mechanism", (_InjectedLeak, _NamedInjectedLeak, _InjectedVoltage)
)
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_population_owned_waveforms_record_restart_commit_and_checkpoint(
    mechanism, execution
):
    source, native = _population(mechanism), _population(mechanism)
    assert source.mechanism_injection_accepted == [True]
    assert source.build_intra() is None
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=execution == "while",
        execution=execution,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
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
    observed = torch.cat(
        (first["callbacks"]["trace"]["v"], second["callbacks"]["trace"]["v"])
    )
    _close(observed, torch.stack(expected))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)
    committed, restored = _population(mechanism), _population(mechanism)
    runner.functional.commit_state_(committed, final)
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step(DT)
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("layout", ("batch", "regional", "packed"))
@pytest.mark.parametrize("mechanism", (_InjectedLeak, _NamedInjectedLeak))
def test_owned_injection_supports_population_layouts(mechanism, layout):
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
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)


def test_partial_regional_owner_leaves_entire_waveform_to_solver():
    source, native = (
        _population(layout="regional", fallback=True),
        _population(layout="regional", fallback=True),
    )
    assert source.mechanism_injection_accepted == [False]
    assert not source.mech.owner._injection_specs
    assert source.build_intra() is not None
    runner = dn.compile(
        source, dt=DT, steps=STEPS, inference=False, backend="aot_eager"
    )
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)


def test_only_the_first_full_acceptor_owns_and_delivers_the_waveform():
    def make():
        population = _population()
        population.insert(_InjectedLeak.rename("second"), gain=0.005)
        population.initialize()
        population.integrator._initialize(
            population, population.t.new_tensor(DT), force=True
        )
        population.detach()
        return population

    source, native = make(), make()
    assert tuple(source.mech.mechanisms) == ("owner", "second")
    assert len(source.mech.owner._injection_specs) == 1
    assert not source.mech.second._injection_specs
    assert source.mechanism_injection_accepted == [True]
    assert source.build_intra() is None
    runner = dn.compile(
        source, dt=DT, steps=STEPS, inference=False, backend="aot_eager"
    )
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)


@pytest.mark.parametrize("network_cache", (False, True))
def test_shared_waveform_cache_restoration_preserves_last_writer_and_aliases(
    network_cache,
):
    def make():
        network = _network(shared=True, connected=False)
        amplitude = network.left.mech.owner.injected_waveforms[0].amp
        with torch.no_grad():
            for _ in range(2):
                network.step()
            if network_cache:
                network.cache_state()
                for name, saved_amplitude in (("left", 0.25), ("right", 1.1)):
                    # state_dict includes every registered alias of the module.
                    for key, value in network._state_cache[name].items():
                        if key.endswith("injected_waveforms.0.amp"):
                            value.fill_(saved_amplitude)
            else:
                amplitude.fill_(0.25)
                network.left.cache("_steady_state")
                amplitude.fill_(1.1)
                network.right.cache("_steady_state")
            amplitude.fill_(0.8)
        return network

    source, native = make(), make()
    functional, tensors = dn.func.make_functional_network(source)
    before = _clone(tensors)
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    expected = _cache_initialized_native_packet(functional, native)
    _close(initialized, expected)
    amplitudes = tuple(
        next(
            value
            for key, value in initialized.parameters["populations"][name].items()
            if key.endswith("injected_waveforms.0.amp")
        )
        for name in ("left", "right")
    )
    assert amplitudes[0] is amplitudes[1]
    _close(amplitudes[0], amplitudes[0].new_tensor(1.1))
    assert (
        native.left.mech.owner.injected_waveforms[0].amp
        is native.right.mech.owner.injected_waveforms[0].amp
    )
    _close(native.left.mech.owner.injected_waveforms[0].amp, amplitudes[0])
    final, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(final, functional.extract(native).state)
    _close(tensors, before)
    _close(
        source.left.mech.owner.injected_waveforms[0].amp,
        amplitudes[0].new_tensor(0.8),
    )


def test_recorded_current_uses_the_incoming_step_clock():
    source = _population(_CurrentClockProbe)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("owner.seen", "t"))},
    )
    _, auxiliary = runner()
    trace = auxiliary["callbacks"]["trace"]
    waveform = source.mech.owner.injected_waveforms[0]
    # The initial assigned current and the first accepted step both read t=0.
    times = torch.arange(STEPS, dtype=source.dtype()) * DT
    values = waveform(times)
    expected = torch.zeros(STEPS + 1, *source.shape, dtype=source.dtype())
    expected[:, :, 0] = torch.cat((values[:1], values))[:, None]
    _close(trace["owner.seen"], expected)
    _close(trace["t"].reshape(-1), torch.arange(STEPS + 1, dtype=source.dtype()) * DT)


def test_one_waveform_can_share_parameters_between_owner_and_solver_delivery():
    def make():
        waveform = _waveform()
        population = _population(layout="regional", waveform=waveform)
        # The regional owner covers column 0, but cannot claim all columns.
        population[:].inject(waveform)
        population.initialize()
        population.integrator._initialize(
            population, population.t.new_tensor(DT), force=True
        )
        population.detach()
        return population

    source, native = make(), make()
    assert source.mechanism_injection_accepted == [True, False]
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        backend="aot_eager",
        trainable=("*injected_waveforms.*.amp", "stimulation.intra.waveforms.*.amp"),
    )
    # All selected aliases refer to one amplitude, regardless of delivery path.
    amplitudes = tuple(runner.trainable_parameters.values())
    assert len(amplitudes) == 2 and amplitudes[0] is amplitudes[1]
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().sum(), amplitudes[0]
    )[0]
    expected = torch.autograd.grad(
        native.v.square().sum(), native.mech.owner.injected_waveforms[0].amp
    )[0]
    assert expected.abs().sum() > 0
    _close(actual, expected)


@pytest.mark.parametrize("layout", ("batch", "regional"))
def test_custom_selected_waveform_buffers_and_gradients_follow_local_support(layout):
    spatial = ((0.4, 0.9), (0.6, 1.2)) if layout == "batch" else (0.4, 0.9)
    source = _population(layout=layout, waveform=_SelectedWaveform(spatial=spatial))
    native = _population(layout=layout, waveform=_SelectedWaveform(spatial=spatial))
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        backend="aot_eager",
        trainable=("*injected_waveforms.*.amp",),
    )
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().sum(),
        runner.get_parameter("*injected_waveforms.*.amp"),
    )[0]
    expected = torch.autograd.grad(
        native.v.square().sum(), native.mech.owner.injected_waveforms[0].amp
    )[0]
    assert expected.abs().sum() > 0
    _close(actual, expected)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_connected_network_owned_waveforms_preserve_continuation(execution):
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
    expected = [native.right.v.clone()]
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step()
            expected.append(native.right.v.clone())
    _close(
        torch.cat(
            (
                first["callbacks"]["trace"]["right.v"],
                second["callbacks"]["trace"]["right.v"],
            )
        ),
        torch.stack(expected),
    )
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)
    committed, restored = _network(), _network()
    runner.functional.commit_state_(committed, final)
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step()
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("cache", (False, True))
def test_owned_injection_fresh_and_cached_initialization(network, cache):
    make = (lambda: _network(connected=False)) if network else _population
    source, native = make(), make()
    for model in (source, native):
        with torch.no_grad():
            for _ in range(3):
                model.step() if network else model.step(DT)
            if cache:
                if network:
                    for population in model.populations.values():
                        population.cache("_steady_state")
                    model.cache_state()
                else:
                    model.cache("_steady_state")
            model.step() if network else model.step(DT)
    if network:
        functional, tensors = dn.func.make_functional_network(source)
        initialized = functional.initialize(*tensors, functional.initialization_inputs)
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
        # Fresh initialization retains each Population's native materialized
        # buffers in both cached and uncached execution. Snapshot extraction
        # leaves those provenance selectors inactive.
        expected = _cache_initialized_native_packet(functional, native)
        _close(initialized, expected)
        final, _ = functional.prepare_and_step(*initialized)
        native.step()
    else:
        functional, tensors = dn.func.make_functional(source, dt=DT)
        initialized = functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
        )
        native.initialize()
        native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
        _close(initialized.state, functional.extract(native).state)
        final, _ = functional.prepare_and_step(
            initialized.parameters, initialized.constants, initialized.state
        )
        native.step(DT)
    _close(final, functional.extract(native).state)


@pytest.mark.parametrize("checkpoint_every", (None, 2))
def test_owned_waveform_model_voltage_and_clock_gradients_match_native(
    checkpoint_every,
):
    source, native = _population(), _population()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        backend="aot_eager",
        checkpoint_every=checkpoint_every,
        trainable=(
            "*gain_param",
            "*injected_waveforms.*.amp",
            "*injected_waveforms.*.freq",
            "*injected_waveforms.*.phase",
        ),
    )
    state = _clone(runner.initial_state)
    voltage = state["integrator"]["v"].requires_grad_()
    time = state["clock"]["t"].requires_grad_()
    with torch.no_grad():
        time.fill_(0.3)
    native.v = voltage.detach().clone().requires_grad_()
    native_voltage = native.v
    native_time = time.detach().clone().requires_grad_()
    native.t = native_time
    native.mech.owner.populate()
    final, _ = runner(state)
    for _ in range(STEPS):
        native.step(DT)
    _close(final["integrator"]["v"], native.v)
    actual_inputs = [
        runner.get_parameter(pattern)
        for pattern in (
            "*gain_param",
            "*injected_waveforms.*.amp",
            "*injected_waveforms.*.freq",
            "*injected_waveforms.*.phase",
        )
    ]
    waveform = native.mech.owner.injected_waveforms[0]
    expected_inputs = [
        native.mech.owner.gain_param,
        waveform.amp,
        waveform.freq,
        waveform.phase,
    ]
    actual = torch.autograd.grad(
        final["integrator"]["v"].square().sum(), (*actual_inputs, voltage, time)
    )
    expected = torch.autograd.grad(
        native.v.square().sum(), (*expected_inputs, native_voltage, native_time)
    )
    for observed, wanted in zip(actual, expected, strict=True):
        assert torch.isfinite(wanted).all() and wanted.abs().sum() > 0
        _close(observed, wanted)


def test_shared_network_waveform_parameters_preserve_aliases_and_gradients():
    source, native = _network(shared=True), _network(shared=True)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        backend="aot_eager",
        trainable=("*injected_waveforms.*.amp",),
        checkpoint_every=2,
    )
    selected = list(runner.trainable_parameters.values())
    assert len(selected) == 2 and selected[0] is selected[1]
    final, _ = runner()
    for _ in range(STEPS):
        native.step()
    _close(final, runner.functional.extract(native).state)
    actual = torch.autograd.grad(
        final["populations"]["right"]["integrator"]["v"].square().sum(), selected[0]
    )[0]
    expected = torch.autograd.grad(
        native.right.v.square().sum(), native.left.mech.owner.injected_waveforms[0].amp
    )[0]
    assert expected.abs().sum() > 0
    _close(actual, expected)


@pytest.mark.parametrize(
    "initialization,backend",
    (("snapshot", "aot_eager"), ("fresh", "aot_eager"), ("snapshot", "inductor")),
)
def test_arbitrary_waveform_knots_preserve_gradients_and_checkpoint_continuation(
    initialization,
    backend,
):
    def make():
        waveform = dn.arbitrary(
            tpoints=(-0.4, 0.17, 0.69, 1.43), values=(0.2, 0.9, -0.4, 0.8)
        )
        population = _population(waveform=waveform)
        with torch.no_grad():
            for _ in range(3):
                population.step(DT)
        population.detach()
        return population

    source, native = make(), make()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        initialization=initialization,
        backend=backend,
        checkpoint_every=2,
        trainable=("*injected_waveforms.*.values", "*injected_waveforms.*.tpoints"),
    )
    before = _clone(runner.functional.extract())
    prefix, _ = runner()
    replay, _ = runner()
    _close(prefix, replay)
    final, _ = runner(prefix)
    if initialization == "fresh":
        native.initialize()
        native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    for _ in range(2 * STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    waveform = native.mech.owner.injected_waveforms[0]
    selected = (
        runner.get_parameter("*injected_waveforms.*.values"),
        runner.get_parameter("*injected_waveforms.*.tpoints"),
    )
    assert all(value.shape == (4,) for value in selected)
    actual = torch.autograd.grad(final["integrator"]["v"].square().sum(), selected)
    expected = torch.autograd.grad(
        native.v.square().sum(), (waveform.values, waveform.tpoints)
    )
    for observed, wanted in zip(actual, expected, strict=True):
        assert torch.isfinite(wanted).all() and wanted.abs().sum() > 0
        _close(observed, wanted)
    committed, restored = make(), make()
    runner.functional.commit_state_(committed, _clone(final))
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(_clone(final))
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step(DT)
    _close(continued, runner.functional.extract(restored).state)
    _close(runner.functional.extract(), before)


@pytest.mark.parametrize("layout", ("dense", "batch"))
def test_custom_selected_vector_parameters_preserve_native_gradients(layout):
    spatial = ((0.4, 0.9), (0.6, 1.2)) if layout == "batch" else (0.4, 0.9)
    source = _population(
        layout=layout, waveform=_SelectedParameterWaveform(spatial=spatial)
    )
    native = _population(
        layout=layout, waveform=_SelectedParameterWaveform(spatial=spatial)
    )
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        backend="aot_eager",
        checkpoint_every=2,
        trainable=("*injected_waveforms.*.spatial", "*injected_waveforms.*.amp"),
    )
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    selected = (
        runner.get_parameter("*injected_waveforms.*.spatial"),
        runner.get_parameter("*injected_waveforms.*.amp"),
    )
    assert selected[0].shape == torch.tensor(spatial).shape
    actual = torch.autograd.grad(final["integrator"]["v"].square().sum(), selected)
    waveform = native.mech.owner.injected_waveforms[0]
    expected = torch.autograd.grad(
        native.v.square().sum(), (waveform.spatial, waveform.amp)
    )
    for observed, wanted in zip(actual, expected, strict=True):
        assert torch.isfinite(wanted).all() and wanted.abs().sum() > 0
        _close(observed, wanted)
