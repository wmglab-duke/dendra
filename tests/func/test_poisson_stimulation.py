"""Randomized stimulation owns explicit draws and replayable RNG carry."""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.distributions import Normal
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import pas
from dendra.units import nA
from tests.func.test_mechanism_injection_execution import _InjectedLeak
from tests.networks.test_functional_event_delivery import ThresholdRelay

DTYPE = torch.float64
DT = 0.025
STEPS = 4
pytestmark = pytest.mark.cpu


class _NoisyPoissonLeak(_InjectedLeak):
    Mechanism.RANGENOISE("eta", distribution="normal", mu=0.0, sigma=0.00001, seed=43)

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.gain * self.i_inj + self.eta.clone()


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)


def _waveform(*, seed=17, generator=None, amp=0.2):
    return dn.mono_rect(amp=amp, pw=0.045).poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=(
            torch.Generator().manual_seed(seed) if generator is None else generator
        ),
    )


def _population(role="intra", *, waveform=None, mechanism=_InjectedLeak):
    population = dn.Unmyelinated(
        [2.0, 2.5],
        L=4.0,
        dx=1.0,
        v_init=-65.0,
        dtype=DTYPE,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    population.insert(mechanism.rename("owner") if role == "mechanism" else pas)
    waveform = (
        _waveform(amp=0.2 * nA if role == "intra" else 0.2)
        if waveform is None
        else waveform
    )
    if role != "extra":
        population[:, 1].inject(waveform)
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    population.detach()
    field = torch.linspace(-1.25, 1.75, population.v.numel(), dtype=DTYPE).reshape(
        population.shape
    )
    return population, waveform, (field, waveform) if role == "extra" else None


def _network(*, sharing="independent"):
    generator = torch.Generator().manual_seed(29) if sharing == "generator" else None
    waveform = _waveform(seed=41) if sharing == "waveform" else None
    populations = {}
    for index, name in enumerate(("left", "right")):
        current = (
            waveform
            if waveform is not None
            else _waveform(seed=41 + index, generator=generator)
        )
        populations[name], _, _ = _population("mechanism", waveform=current)
    network = dn.Network(populations)
    network.train()
    network.initialize(DT)
    for population in network.populations.values():
        population.detach()
    return network


def _slice(inputs, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], inputs)


@pytest.mark.parametrize("role", ("intra", "mechanism", "extra"))
def test_poisson_explicit_tape_replays_steps_chunks_and_detaches_draw_gradients(role):
    source, waveform, extra = _population(role)
    native, _waveform_source, native_extra = _population(role)
    functional, tensors = dn.func.make_functional(source, dt=DT, extra=extra)
    frozen = _clone(tensors)
    source_rng = waveform.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    tape = functional.sample_stimulation_draw_tape(tensors.state, 2 * STEPS)
    _close(tape, functional.sample_stimulation_draw_tape(tensors.state, 2 * STEPS))
    assert isinstance(tape, dn.func.StimulationDraw)
    tape = tape._replace(
        draws={
            name: value.detach().clone().requires_grad_()
            for name, value in tape.draws.items()
        }
    )
    inputs = dn.func.RolloutInput(stimulation=tape)
    frozen_inputs = _clone(inputs)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    whole, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, inputs, steps=2 * STEPS
    )
    split, _ = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        _slice(inputs, 0, STEPS),
        steps=STEPS,
    )
    split, _ = functional.rollout(
        tensors.parameters,
        prepared,
        split,
        _slice(inputs, STEPS, 2 * STEPS),
        steps=STEPS,
    )
    _close(whole, split)
    stepped = tensors.state
    for index in range(2 * STEPS):
        draw = pytree.tree_map_only(torch.Tensor, lambda value: value[index], tape)
        stepped, _ = functional.step(
            tensors.parameters,
            prepared,
            stepped,
            dn.func.StepInput(stimulation=draw),
        )
        native.step(DT, extra=native_extra)
    _close(whole, stepped)
    _close(whole["integrator"]["v"], native.v)
    gradients = torch.autograd.grad(
        whole["integrator"]["v"].square().mean(),
        tuple(tape.draws.values()),
        allow_unused=True,
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradients)
    _close(tensors, frozen)
    _close(inputs, frozen_inputs)
    _close(waveform.generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


def test_poisson_default_generator_is_captured_without_consuming_global_rng():
    saved_rng = torch.get_rng_state().clone()
    try:
        torch.manual_seed(83)
        source, _, _ = _population(
            "intra", waveform=_waveform(generator=torch.default_generator, amp=0.2 * nA)
        )
        native, _, _ = _population(
            "intra", waveform=_waveform(generator=torch.default_generator, amp=0.2 * nA)
        )
        functional, tensors = dn.func.make_functional(source, dt=DT)
        before = torch.get_rng_state().clone()
        draws = functional.sample_stimulation_draw_tape(tensors.state, STEPS)
        _close(torch.get_rng_state(), before)
        final, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(stimulation=draws),
            steps=STEPS,
        )
        _close(torch.get_rng_state(), before)
        for _ in range(STEPS):
            native.step(DT)
        _close(final["integrator"]["v"], native.v)
    finally:
        torch.set_rng_state(saved_rng)


@pytest.mark.parametrize(
    "defect", ("draw_shape", "draw_dtype", "missing_rng", "rng_dtype")
)
def test_poisson_explicit_packet_validates_schema_without_consuming_rng(defect):
    source, waveform, _ = _population("intra")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    packet = functional.sample_stimulation_draw(tensors.state)
    draw_key = next(iter(packet.draws))
    rng_key = next(iter(packet.rng))
    draws, rng = dict(packet.draws), dict(packet.rng)
    if defect == "draw_shape":
        draws[draw_key] = draws[draw_key][:-1]
    elif defect == "draw_dtype":
        dtype = (
            torch.float64 if draws[draw_key].dtype == torch.float32 else torch.float32
        )
        draws[draw_key] = draws[draw_key].to(dtype)
    elif defect == "missing_rng":
        rng.clear()
    else:
        rng[rng_key] = rng[rng_key].to(torch.int64)
    invalid = dn.func.StimulationDraw(draws, rng)
    before = waveform.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="schema|names|RNG"):
        functional.step(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.StepInput(stimulation=invalid),
        )
    _close(waveform.generator.get_state(), before)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("role", ("intra", "mechanism"))
def test_poisson_in_deterministic_expression_commits_its_nested_schedule(role):
    def make():
        pulse = _waveform(amp=0.2 * nA if role == "intra" else 0.2)
        return _population(role, waveform=0.7 * pulse + dn.constant(value=0.0))

    source, _waveform_source, _ = make()
    native, _native_waveform, _ = make()
    runner = dn.compile(
        source, dt=DT, steps=STEPS, inference=False, backend="aot_eager"
    )
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    target, _waveform_target, _ = make()
    runner.functional.commit_state_(target, final)
    restored, _waveform_restored, _ = make()
    restored.restore_dict_from_checkpoint(target.state_dict_for_checkpoint())
    continued, _ = runner(final)
    for _ in range(STEPS):
        restored.step(DT)
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("role", ("intra", "mechanism"))
def test_poisson_fresh_initialization_preserves_rng_and_population_cache(
    network, cached, role
):
    def make():
        if network:
            populations = {
                name: _population(
                    role,
                    waveform=_waveform(
                        seed=17 + index, amp=0.2 * nA if role == "intra" else 0.2
                    ),
                )[0]
                for index, name in enumerate(("left", "right"))
            }
            source = dn.Network(populations)
            source.train()
            source.initialize(DT)
        else:
            source, _, _ = _population(role)
            populations = {"cell": source}
        if cached:
            for _ in range(2):
                source.step() if network else source.step(DT)
            for population in populations.values():
                population.cache("_steady_state")
            for _ in range(2):
                source.step() if network else source.step(DT)
        return source

    source, native = make(), make()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        backend="aot_eager",
        checkpoint_every=2,
    )
    frozen = _clone(runner.functional.extract())
    global_rng = torch.get_rng_state().clone()
    final, _ = runner()
    replay, _ = runner()
    _close(final, replay)
    native.initialize(DT) if network else native.initialize()
    for _ in range(STEPS):
        native.step() if network else native.step(DT)
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), frozen)
    _close(torch.get_rng_state(), global_rng)


def test_mechanism_poisson_fresh_initialization_requires_replayable_explicit_draw():
    source, waveform, _ = _population("mechanism")
    native, _, _ = _population("mechanism")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors)
    before = waveform.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="draw|Stimulation"
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
        )
    draw = functional.sample_stimulation_initialization_draw(tensors.state)
    _close(draw, functional.sample_stimulation_initialization_draw(tensors.state))
    inputs = tensors.initialization._replace(stimulation=draw)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, inputs, state=tensors.state
    )
    replay = functional.initialize(
        tensors.parameters, tensors.constants, inputs, state=tensors.state
    )
    _close(initialized, replay)
    native.initialize()
    _close(initialized.state, functional.extract(native).state)
    runtime_draw = functional.sample_stimulation_draw(initialized.state)
    final, _ = functional.prepare_and_step(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        dn.func.StepInput(stimulation=runtime_draw),
    )
    native.step(DT)
    _close(final, functional.extract(native).state)
    _close(tensors, frozen)
    _close(waveform.generator.get_state(), before)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("private", (False, True))
def test_poisson_fresh_network_normal_sampling_preserves_default_stream_order(private):
    saved_rng = torch.get_rng_state().clone()

    def make():
        torch.manual_seed(219)
        generator = (
            torch.Generator().manual_seed(91) if private else torch.default_generator
        )
        cell, _, _ = _population("mechanism", waveform=_waveform(generator=generator))
        cell.insert(ThresholdRelay.rename("relay"))
        network = dn.Network(
            {"cell": cell}, netcon_delay_backend="dense", netcon_train_backend="dense"
        )
        network.connect_one_to_one(
            cell[:],
            cell[:],
            cell.mech.relay,
            pre_var="mech.relay.level",
            weight=Normal(mean=0.003, std=0.0002),
            delay=DT,
            allow_autapses=True,
        )
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            train_delay_backend="dense",
        )
        return network

    try:
        source, native = make(), make()
        runner = dn.compile(
            source,
            dt=DT,
            steps=STEPS,
            initialization="fresh",
            inference=False,
            backend="aot_eager",
            checkpoint_every=2,
        )
        before = torch.get_rng_state().clone()
        final, _ = runner()
        replay, _ = runner()
        _close(final, replay)
        _close(torch.get_rng_state(), before)
        native.initialize(DT)
        for _ in range(STEPS):
            native.step()
        _close(final, runner.functional.extract(native).state)
    finally:
        torch.set_rng_state(saved_rng)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_poisson_network_extra_preserves_authored_mapping_order(execution):
    def make():
        generator = torch.Generator().manual_seed(97)
        populations, pairs = {}, {}
        for name in ("left", "right"):
            population, _, extra = _population(
                "extra", waveform=_waveform(generator=generator)
            )
            populations[name], pairs[name] = population, extra
        network = dn.Network(populations)
        network.train()
        network.initialize(DT)
        # Native Network.step visits this mapping before Population-order
        # Intra/mechanism evaluation, so its order owns the shared draw stream.
        return network, {name: pairs[name] for name in ("right", "left")}

    source, extra = make()
    native, native_extra = make()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        extra=extra,
        execution=execution,
        inference=execution == "while",
        backend="aot_eager",
    )
    prefix, _ = runner()
    final, _ = runner(prefix)
    for _ in range(2 * STEPS):
        native.step(extra=native_extra)
    for name, population in native.populations.items():
        _close(final["populations"][name]["integrator"]["v"], population.v)


@pytest.mark.parametrize("entrypoint", ("step", "run", "assemble"))
def test_randomized_runtime_extra_requires_bound_stimulation_and_preserves_rng(
    entrypoint,
):
    population, waveform, extra = _population("extra")
    functional, tensors = dn.func.make_functional(population, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    source_rng = waveform.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with pytest.raises(
        dn.func.FunctionalizationError, match="[Bb]ind|Population|stimulation"
    ):
        if entrypoint == "step":
            functional.step(tensors.parameters, prepared, tensors.state, extra=extra)
        elif entrypoint == "run":
            dn.func.run(
                functional,
                partial(functional.step, tensors.parameters, prepared),
                tensors.state,
                tstop=STEPS * DT,
                extra=extra,
            )
        else:
            plan = functional.make_extra(extra)
            bundle = plan.extract()
            plan.assemble_tensors(bundle, torch.arange(STEPS, dtype=DTYPE) * DT)
    _close(waveform.generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("kind", ("unbounded", "nested", "repeated"))
def test_poisson_unsupported_dynamic_invocations_fail_before_consuming_rng(kind):
    generator = torch.Generator().manual_seed(73)
    if kind == "unbounded":
        waveform = dn.mono_rect(amp=0.2 * nA, pw=0.045).poisson(
            interval=0.055,
            n=None,
            off=0.3,
            randomize_every_call=True,
            generator=generator,
        )
    else:
        waveform = _waveform(generator=generator, amp=0.2 * nA)
        if kind == "nested":
            waveform = waveform.poisson(interval=0.2, n=2, generator=generator)
        else:
            waveform = waveform + waveform
    population, _, _ = _population("intra", waveform=waveform)
    before = generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with pytest.raises(
        dn.func.FunctionalizationError, match="finite n|Nested|repeated"
    ):
        dn.func.make_functional(population, dt=DT)
    _close(generator.get_state(), before)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("role", ("intra", "mechanism", "extra"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_poisson_runner_restart_continuation_recording_and_checkpoint(role, execution):
    source, waveform, extra = _population(role)
    native, _native_waveform, native_extra = _population(role)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        extra=extra,
        inference=execution == "while",
        execution=execution,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    frozen = _clone(runner.functional.extract())
    source_rng = waveform.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    prefix, first = runner()
    replay, replay_auxiliary = runner()
    _close(prefix, replay)
    _close(first["callbacks"]["trace"], replay_auxiliary["callbacks"]["trace"])
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    with torch.no_grad():
        expected = [native.v.clone()]
        for _ in range(2 * STEPS):
            native.step(DT, extra=native_extra)
            expected.append(native.v.clone())
    observed = torch.cat(
        (first["callbacks"]["trace"]["v"], second["callbacks"]["trace"]["v"])
    )
    _close(observed, torch.stack(expected))
    if role != "extra":
        _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), frozen)
    _close(waveform.generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)
    # A pure checkpoint owns the generator suffix rather than consulting the
    # original Python generator again when execution resumes.
    if role != "extra":
        committed, _wave, _extra = _population(role)
        runner.functional.commit_state_(committed, final)
        restored, _wave, _extra = _population(role)
        restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
        continued, _ = runner(final)
        with torch.no_grad():
            for _ in range(STEPS):
                restored.step(DT)
        _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("role", ("intra", "mechanism", "extra"))
def test_poisson_amplitude_gradient_matches_native_checkpointed_scan(role):
    source, _waveform_source, extra = _population(role)
    native, native_waveform, native_extra = _population(role)
    if role == "extra":
        extra[0].requires_grad_()
        native_extra[0].requires_grad_()
    runner = dn.compile(
        source,
        dt=DT,
        steps=2 * STEPS,
        extra=extra,
        inference=False,
        backend="aot_eager",
        checkpoint_every=2,
        trainable=(
            ("*waveform.amp", "stimulation.extra.contacts.*.field")
            if role == "extra"
            else ("*waveform.amp",)
        ),
    )
    selected = tuple(runner.trainable_parameters.values())
    assert len(selected) == (2 if role == "extra" else 1)
    final, _ = runner()
    assert all(
        not schedule.requires_grad
        for schedule in final["stimulation"]["schedules"].values()
    )
    for _ in range(2 * STEPS):
        native.step(DT, extra=native_extra)
    _close(final["integrator"]["v"], native.v)
    sources = {
        name: native_extra[0]
        if name.endswith(".field")
        else native_waveform.waveform.amp
        for name in runner.trainable_parameters
    }
    actual = torch.autograd.grad(final["integrator"]["v"].square().mean(), selected)
    expected = torch.autograd.grad(native.v.square().mean(), tuple(sources.values()))
    for gradient, reference in zip(actual, expected, strict=True):
        assert reference.abs().sum() > 0
        _close(gradient, reference)


def test_shared_network_poisson_amplitude_gradients_keep_schedules_detached():
    source, native = _network(sharing="waveform"), _network(sharing="waveform")
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        execution="scan",
        inference=False,
        trainable=("*waveform.amp",),
        checkpoint_every=2,
        backend="aot_eager",
    )
    amplitudes = tuple(runner.trainable_parameters.values())
    assert len(amplitudes) == 2 and amplitudes[0] is amplitudes[1]
    final, _ = runner()
    for population in final["populations"].values():
        assert all(
            not schedule.requires_grad
            for schedule in population["stimulation"]["schedules"].values()
        )
    actual_loss = sum(
        population["integrator"]["v"].square().mean()
        for population in final["populations"].values()
    )
    actual = torch.autograd.grad(actual_loss, amplitudes[0])[0]
    for _ in range(STEPS):
        native.step()
    _close(final, runner.functional.extract(native).state)
    expected_loss = sum(
        population.v.square().mean() for population in native.populations.values()
    )
    amplitude = native.left.mech.owner.injected_waveforms[0].waveform.amp
    expected = torch.autograd.grad(expected_loss, amplitude)[0]
    assert expected.abs().sum() > 0
    _close(actual, expected)


@pytest.mark.parametrize("sharing", ("independent", "waveform", "generator"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_poisson_network_preserves_shared_call_order_and_checkpoint(sharing, execution):
    source, native = _network(sharing=sharing), _network(sharing=sharing)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=execution == "while",
        execution=execution,
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    frozen = _clone(runner.functional.extract())
    global_rng = torch.get_rng_state().clone()
    prefix, first = runner()
    replay, _ = runner()
    _close(prefix, replay)
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    expected = {name: [cell.v.clone()] for name, cell in native.populations.items()}
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step()
            for name, cell in native.populations.items():
                expected[name].append(cell.v.clone())
    for name in native.populations:
        observed = torch.cat(
            (
                first["callbacks"]["trace"][f"{name}.v"],
                second["callbacks"]["trace"][f"{name}.v"],
            )
        )
        _close(observed, torch.stack(expected[name]))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), frozen)
    _close(torch.get_rng_state(), global_rng)
    if sharing == "waveform":
        amplitude = next(
            name
            for name in runner.named_parameters
            if name.startswith("populations.left.") and name.endswith("waveform.amp")
        )
        assert (
            runner.named_parameters[amplitude]
            is runner.named_parameters[
                amplitude.replace("populations.left.", "populations.right.")
            ]
        )
    committed = _network(sharing=sharing)
    runner.functional.commit_state_(committed, final)
    restored = _network(sharing=sharing)
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step()
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_noise_and_poisson_explicit_tapes_coexist_through_replay_and_continuation(
    execution,
):
    source, waveform, _ = _population("mechanism", mechanism=_NoisyPoissonLeak)
    native, _, _ = _population("mechanism", mechanism=_NoisyPoissonLeak)
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
    functional = runner.functional
    frozen = _clone(functional.extract())
    source_rng = waveform.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    inputs = dn.func.RolloutInput(
        noise=functional.sample_noise_draw_tape(runner.initial_state, 2 * STEPS),
        stimulation=functional.sample_stimulation_draw_tape(
            runner.initial_state, 2 * STEPS
        ),
    )
    assert inputs.noise is not None and inputs.stimulation is not None
    frozen_inputs = _clone(inputs)
    first_inputs = _slice(inputs, 0, STEPS)
    prefix, first = runner(inputs=first_inputs)
    replay, replay_auxiliary = runner(inputs=first_inputs)
    _close(prefix, replay)
    _close(first["callbacks"]["trace"], replay_auxiliary["callbacks"]["trace"])
    final, second = runner(
        prefix,
        inputs=_slice(inputs, STEPS, 2 * STEPS),
        callback_state=first["callbacks"].state,
    )
    expected = [native.v.clone()]
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step(DT)
            expected.append(native.v.clone())
    recorded = torch.cat(
        (first["callbacks"]["trace"]["v"], second["callbacks"]["trace"]["v"])
    )
    _close(recorded, torch.stack(expected))
    _close(final, functional.extract(native).state)
    _close(functional.extract(), frozen)
    _close(inputs, frozen_inputs)
    _close(waveform.generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)
