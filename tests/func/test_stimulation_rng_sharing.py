"""Borrowed stimulation streams preserve mixed native stochastic draw order."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.units import nA
from tests.func.test_mechanism_injection_execution import _InjectedLeak
from tests.func.test_poisson_stimulation import DT, DTYPE, STEPS, _clone, _close

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                IMEM=False,
                DTYPE=DTYPE,
                DEVICE="cpu",
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


class _SharedStochasticGate(State):
    State.STATE("x")
    State.RANGENOISE("forcing", mu=0.0, sigma=0.01, seed=101, rng_name="stream")
    State.RANGE(theta=0.7, sigma=0.04)
    State.DERIVATIVE("x' = theta * (0.3 - x) + forcing")
    State.DIFFUSION("x = sigma")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


class _SharedStochasticLeak(Mechanism):
    Mechanism.STATE_BUNDLE(_SharedStochasticGate)
    Mechanism.RANGE(g=0.001, e=-65.0, gain=0.002)
    Mechanism.GLOBALRAND("offset", mu=0.03, sigma=0.001, seed=103, rng_name="stream")
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00002, seed=107, rng_name="stream")
    Mechanism.ASSIGNED("i_inj")
    Mechanism.CARRY("seen")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def inject(self, waveform, *, index=None, model_shape=None, model=None, **kwargs):
        del kwargs
        return self.register_waveform_injection(
            waveform, index=index, model_shape=model_shape, model=model
        )

    def initial_values(self, v, values):
        del values
        return {"seen": torch.zeros_like(v)}

    def assigned_values(self, v, values):
        del values
        return {"i_inj": self.injection_values(v, t=self.t)}

    def advance(self, v, dt, values):
        del dt, values
        return {"seen": self.injection_values(v, t=self.t)}

    def i_affine_terms(self, v):
        del v
        return (
            self.g,
            self.g * self.e
            + self.eta.clone()
            + self.gain * (self.i_inj + 0.5 * self.seen + self.x + self.offset),
        )


class _ReseededStochasticLeak(_SharedStochasticLeak):
    Mechanism.GLOBALRAND(
        "offset",
        mu=0.03,
        sigma=0.001,
        seed=103,
        rng_name="stream",
        reseed_on_initialize=True,
    )


class _TransitionGuardLeak(_SharedStochasticLeak):
    FAIL_ON_ADVANCE = False

    def advance(self, v, dt, values):
        if _TransitionGuardLeak.FAIL_ON_ADVANCE:
            raise AssertionError(
                "Numerical transition entered before packet validation"
            )
        return _SharedStochasticLeak.advance(self, v, dt, values)


class _BankStochasticLeak(_SharedStochasticLeak):
    inject = Mechanism.inject


class _BankReseededStochasticLeak(_ReseededStochasticLeak):
    inject = Mechanism.inject


class _CachedStochasticLeak(_SharedStochasticLeak):
    Mechanism.GLOBALRAND(
        "offset",
        mu=0.03,
        sigma=0.001,
        seed=103,
        rng_name="stream",
        resample_on_initialize=False,
    )


class _RandOnlyLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.GLOBALRAND("offset", mu=0.00001, sigma=0.000001, seed=811)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.offset


class _SDEOnlyGate(State):
    State.STATE("x")
    State.RANGE(theta=0.7, sigma=0.04)
    State.DERIVATIVE("x' = theta * (0.3 - x)")
    State.DIFFUSION("x = sigma")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


class _SDEOnlyInjectedLeak(_InjectedLeak):
    Mechanism.STATE_BUNDLE(_SDEOnlyGate)

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.gain * (self.i_inj + self.x)


def _population(
    role="mechanism",
    *,
    repeated=True,
    reseed=False,
    retained=False,
    transition_guard=False,
    seed=401,
):
    cell = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-45.0,
        dtype=DTYPE,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    kind = (
        (_ReseededStochasticLeak if reseed else _SharedStochasticLeak)
        if role == "mechanism"
        else (_BankReseededStochasticLeak if reseed else _BankStochasticLeak)
    )
    if retained:
        assert role == "mechanism" and not reseed
        kind = _CachedStochasticLeak
    if transition_guard:
        assert role == "mechanism" and not reseed and not retained
        kind = _TransitionGuardLeak
    cell.insert(kind.rename("probe"))
    cell.train()
    cell.initialize()
    owner = cell.mech.probe
    gate = owner.DE._SharedStochasticGate
    gate.stream = owner.stream
    gate.x_dW_rng = owner.stream
    owner.stream.reseed(seed)
    pulse = dn.mono_rect(amp=0.2 * nA if role == "intra" else 0.2, pw=0.045).poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=owner.stream.rng,
    )
    waveform = pulse + pulse if repeated else pulse
    if role == "mechanism":
        cell[:, 1].inject(waveform)
    elif role == "intra":
        cell[:, 1].inject(waveform)
    field = torch.linspace(-1.0e6, 1.5e6, cell.v.numel(), dtype=DTYPE).reshape(
        cell.shape
    )
    extra = (field, waveform) if role == "extra" else None
    cell.initialize()
    cell.integrator._initialize(cell, cell.t.new_tensor(DT), force=True)
    cell.detach()
    return cell, extra, pulse


def _initialize_native(model):
    if isinstance(model, dn.Network):
        model.initialize(DT)
    else:
        model.initialize()
        model.integrator._initialize(model, model.t.new_tensor(DT), force=True)


def _lower(model, extra=None):
    if isinstance(model, dn.Network):
        return dn.func.make_functional_network(model, extra=extra)
    return dn.func.make_functional(model, dt=DT, extra=extra)


def _step_native(model, extra=None):
    if isinstance(model, dn.Network):
        model.step(extra=extra)
    else:
        model.step(DT, extra=extra)


def _draw_input(functional, state, steps=None):
    if hasattr(functional, "populations"):
        noises = functional.sample_population_noise_tapes(state, steps)
        stimuli = functional.sample_population_stimulation_tapes(state, steps)
        return dn.func.NetworkRolloutInput(
            populations={
                name: dn.func.RolloutInput(
                    noise=noises[name], stimulation=stimuli[name]
                )
                for name in functional.populations
            }
        )
    if steps is None:
        return dn.func.StepInput(
            noise=functional.sample_noise_draw(state),
            stimulation=functional.sample_stimulation_draw(state),
        )
    return dn.func.RolloutInput(
        noise=functional.sample_noise_draw_tape(state, steps),
        stimulation=functional.sample_stimulation_draw_tape(state, steps),
    )


def _slice(value, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda tensor: tensor[start:stop], value)


def _checkpoint_close(actual, expected):
    left, left_spec = pytree.tree_flatten(actual)
    right, right_spec = pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for value, reference in zip(left, right, strict=True):
        if torch.is_tensor(reference):
            _close(value, reference)
        else:
            assert value == reference


def _network(*, reseed=False, retained=False, seed=601):
    left, _, _ = _population("mechanism", reseed=reseed, retained=retained, seed=seed)
    right, _, _ = _population(
        "mechanism", reseed=reseed, retained=retained, seed=seed + 1
    )
    stream = left.mech.probe.stream
    right.mech.probe.stream = stream
    right.mech.probe.DE._SharedStochasticGate.stream = stream
    right.mech.probe.DE._SharedStochasticGate.x_dW_rng = stream
    pulses = []
    for cell in (left, right):
        for waveform in cell.mech.probe.injected_waveforms:
            for module in waveform.modules():
                if hasattr(module, "generator"):
                    module.generator = stream.rng
                    pulses.append(module)
    model = dn.Network({"left": left, "right": right})
    model.train()
    model.initialize(DT)
    for cell in model.populations.values():
        cell.detach()
    return model, pulses


def _mixed_network():
    populations, extra, pulses = {}, {}, []
    for index, role in enumerate(("extra", "intra", "mechanism", "extra")):
        cell, binding, pulse = _population(role, seed=701 + index)
        name = f"cell{index}"
        populations[name] = cell
        pulses.append(pulse)
        if binding is not None:
            extra[name] = binding
    stream = populations["cell0"].mech.probe.stream
    for cell, pulse in zip(populations.values(), pulses, strict=True):
        owner = cell.mech.probe
        owner.stream = stream
        owner.DE._SharedStochasticGate.stream = stream
        owner.DE._SharedStochasticGate.x_dW_rng = stream
        pulse.generator = stream.rng
    model = dn.Network(populations)
    model.train()
    model.initialize(DT)
    for cell in populations.values():
        cell.detach()
    # Native extra preprocessing follows the supplied mapping, independently of
    # the Population solve order and subsequent intracellular preparation.
    return model, dict(reversed(tuple(extra.items()))), pulses


def _one_stream(state):
    if "populations" in state:
        assert len(state["rng"]) == 1
        for local in state["populations"].values():
            assert not local.get("random", {}).get("rng", {})
            assert not local.get("stimulation", {}).get("rng", {})
        assert not state.get("stimulation_rng", {})
        return next(iter(state["rng"].values()))["cpu"]
    assert len(state["random"]["rng"]) == 1
    assert not state["stimulation"].get("rng", {})
    return next(iter(state["random"]["rng"].values()))["cpu"]


def _host_initialize(functional, tensors):
    if hasattr(functional, "populations"):
        return functional._initialize_host(*tensors, functional.initialization_inputs)
    return functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )


def _gradient_leaves(model):
    cells = model.populations.values() if isinstance(model, dn.Network) else (model,)
    leaves = []
    for cell in cells:
        owner = cell.mech.probe
        pulse = owner.injected_waveforms[0].waveforms[0]
        leaves.extend((pulse.waveform.amp, owner.offset_mu_param))
    return tuple(leaves)


def _functional_gradient_leaves(tensors):
    populations = tensors.parameters.get("populations")
    parameters = (tensors.parameters,) if populations is None else populations.values()
    leaves = []
    for values in parameters:
        for suffix in ("waveform.amp", "offset_mu_param"):
            leaves.append(
                next(value for name, value in values.items() if name.endswith(suffix))
            )
    return tuple(leaves)


@pytest.mark.parametrize("role", ("intra", "extra", "mechanism"))
@pytest.mark.parametrize("repeated", (False, True))
def test_joint_population_packets_follow_native_interleaved_draw_order(role, repeated):
    source, extra, pulse = _population(role, repeated=repeated)
    native, native_extra, native_pulse = _population(role, repeated=repeated)
    source_rng = pulse.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    functional, tensors = _lower(source, extra)
    frozen = _clone(tensors)
    state = tensors.state
    _close(_one_stream(state), source_rng)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    for _ in range(STEPS):
        inputs = _draw_input(functional, state)
        frozen_inputs = _clone(inputs)
        state, _ = functional.step(tensors.parameters, prepared, state, inputs)
        _step_native(native, native_extra)
        expected = _lower(native, native_extra)[1].state
        _close(state, expected)
        _close(_one_stream(state), native_pulse.generator.get_state())
        _close(inputs, frozen_inputs)
    _close(tensors, frozen)
    _close(pulse.generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


def test_joint_runtime_tapes_replay_chunks_and_detach_base_draw_gradients():
    source, _, pulse = _population()
    native, _, native_pulse = _population()
    functional, tensors = _lower(source)
    frozen = _clone(tensors)
    source_rng = pulse.generator.get_state().clone()
    tape = _draw_input(functional, tensors.state, 2 * STEPS)
    _close(tape, _draw_input(functional, tensors.state, 2 * STEPS))
    tape = tape._replace(
        noise=tape.noise._replace(
            draws={
                key: value.detach().clone().requires_grad_()
                for key, value in tape.noise.draws.items()
            }
        ),
        stimulation=tape.stimulation._replace(
            draws={
                key: value.detach().clone().requires_grad_()
                for key, value in tape.stimulation.draws.items()
            }
        ),
    )
    frozen_tape = _clone(tape)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    whole, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, tape, steps=2 * STEPS
    )
    prefix, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, _slice(tape, 0, STEPS), steps=STEPS
    )
    continuation = _draw_input(functional, prefix, STEPS)
    _close(continuation, _slice(tape, STEPS, 2 * STEPS))
    chunked, _ = functional.rollout(
        tensors.parameters, prepared, prefix, continuation, steps=STEPS
    )
    for _ in range(2 * STEPS):
        native.step(DT)
    _close(whole, chunked)
    _close(whole, functional.extract(native).state)
    _close(_one_stream(whole), native_pulse.generator.get_state())
    gradients = torch.autograd.grad(
        whole["integrator"]["v"].square().mean(),
        (*tape.noise.draws.values(), *tape.stimulation.draws.values()),
        allow_unused=True,
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradients)
    _close(tape, frozen_tape)
    _close(tensors, frozen)
    _close(pulse.generator.get_state(), source_rng)


def test_joint_network_tapes_preserve_extra_intra_and_population_kernel_order():
    source, extra, pulses = _mixed_network()
    native, native_extra, native_pulses = _mixed_network()
    functional, tensors = _lower(source, extra)
    frozen = _clone(tensors)
    source_rng = pulses[0].generator.get_state().clone()
    inputs = _draw_input(functional, tensors.state, STEPS)
    final, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        inputs,
        steps=STEPS,
    )
    for _ in range(STEPS):
        _step_native(native, native_extra)
    _close(final, _lower(native, native_extra)[1].state)
    _close(_one_stream(final), native_pulses[0].generator.get_state())
    _close(tensors, frozen)
    _close(pulses[0].generator.get_state(), source_rng)


def test_cross_population_borrower_without_local_rng_keeps_owner_suffix_on_commit():
    def case():
        donor = _population()[0]
        borrower = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-45.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        borrower.insert(_InjectedLeak.rename("probe"))
        pulse = dn.mono_rect(amp=0.2, pw=0.045).poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=donor.mech.probe.stream.rng,
        )
        borrower[:, 1].inject(pulse)
        network = dn.Network({"donor": donor, "borrower": borrower})
        network.train()
        network.initialize(DT)
        for cell in network.populations.values():
            cell.detach()
        return network, pulse

    source, pulse = case()
    native, native_pulse = case()
    functional, tensors = _lower(source)
    frozen = _clone(tensors)
    source_rng = pulse.generator.get_state().clone()
    inputs = _draw_input(functional, tensors.state, STEPS)
    assert inputs.populations["borrower"].noise is None
    final, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        inputs,
        steps=STEPS,
    )
    for _ in range(STEPS):
        native.step()
    _close(final, functional.extract(native).state)
    _close(_one_stream(final), native_pulse.generator.get_state())
    committed, committed_pulse = case()
    functional.commit_state_(committed, final)
    assert committed_pulse.generator is committed.donor.mech.probe.stream.rng
    _close(committed_pulse.generator.get_state(), _one_stream(final))
    checkpoint = _clone(committed.state_dict_for_checkpoint())
    restored, restored_pulse = case()
    restored.restore_dict_from_checkpoint(checkpoint)
    _close(restored_pulse.generator.get_state(), _one_stream(final))
    continued, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        final,
        _draw_input(functional, final, STEPS),
        steps=STEPS,
    )
    for _ in range(STEPS):
        restored.step()
    _close(continued, functional.extract(restored).state)
    _close(tensors, frozen)
    _close(pulse.generator.get_state(), source_rng)


@pytest.mark.parametrize("operation", ("extract", "commit"))
def test_compatible_network_rejects_poisson_detached_from_shared_model_owner(operation):
    source, _ = _network()
    target, pulses = _network()
    functional, tensors = _lower(source)
    independent = torch.Generator().manual_seed(901)
    for pulse in pulses:
        pulse.generator = independent
    before = _clone(target.state_dict_for_checkpoint())
    owner_rng = target.left.mech.probe.stream.rng.get_state().clone()
    independent_rng = independent.get_state().clone()
    frozen = _clone(tensors)
    with pytest.raises(
        dn.func.FunctionalizationError, match="sharing|borrow|RNG|generator"
    ):
        if operation == "extract":
            functional.extract(target)
        else:
            functional.commit_state_(target, tensors.state)
    _checkpoint_close(target.state_dict_for_checkpoint(), before)
    _close(target.left.mech.probe.stream.rng.get_state(), owner_rng)
    _close(independent.get_state(), independent_rng)
    _close(tensors, frozen)


@pytest.mark.parametrize("network", (False, True))
def test_joint_fresh_rollout_keeps_waveform_and_rand_parameter_gradients(network):
    source = _network()[0] if network else _population()[0]
    native = _network()[0] if network else _population()[0]
    functional, tensors = _lower(source)
    initialized = _host_initialize(functional, tensors)
    _initialize_native(native)
    inputs = _draw_input(functional, initialized.state, STEPS)
    final, _ = functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        inputs,
        steps=STEPS,
    )
    for _ in range(STEPS):
        _step_native(native)
    _close(final, functional.extract(native).state)
    actual_loss = (
        sum(
            value["integrator"]["v"].square().mean()
            for value in final["populations"].values()
        )
        if network
        else final["integrator"]["v"].square().mean()
    )
    expected_loss = (
        sum(cell.v.square().mean() for cell in native.populations.values())
        if network
        else native.v.square().mean()
    )
    actual = torch.autograd.grad(actual_loss, _functional_gradient_leaves(tensors))
    expected = torch.autograd.grad(expected_loss, _gradient_leaves(native))
    for left, right in zip(actual, expected, strict=True):
        assert right.abs().sum() > 0
        _close(left, right)


@pytest.mark.parametrize("missing", ("noise", "stimulation"))
def test_shared_stream_runner_rejects_partial_explicit_packet_pairs(missing):
    source, _, pulse = _population()
    runner = dn.compile(
        source, dt=DT, steps=STEPS, inference=False, backend="aot_eager"
    )
    functional = runner.functional
    tensors = functional.extract()
    inputs = _draw_input(functional, tensors.state, STEPS)
    before = _clone(tensors)
    source_rng = pulse.generator.get_state().clone()
    with pytest.raises(
        ValueError, match="shared|sharing|coupled|both|together|complete|explicit"
    ):
        runner(inputs=inputs._replace(**{missing: None}))
    _close(functional.extract(), before)
    _close(pulse.generator.get_state(), source_rng)


def test_rand_only_borrowed_stream_accepts_explicit_stimulation_without_noise():
    def case():
        cell = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-45.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        cell.insert(_RandOnlyLeak.rename("probe"))
        cell.train()
        cell.initialize()
        waveform = dn.mono_rect(amp=0.2 * nA, pw=0.045).poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=cell.mech.probe.offset_rng.rng,
        )
        cell[:, 1].inject(waveform)
        cell.initialize()
        cell.detach()
        return cell, waveform

    source, pulse = case()
    native, native_pulse = case()
    runner = dn.compile(
        source, dt=DT, steps=STEPS, inference=False, backend="aot_eager"
    )
    functional = runner.functional
    state = runner.initial_state
    source_rng = pulse.generator.get_state().clone()
    assert functional.sample_noise_draw_tape(state, STEPS) is None
    tape = functional.sample_stimulation_draw_tape(state, STEPS)
    final, _ = runner(inputs=dn.func.RolloutInput(stimulation=tape))
    for _ in range(STEPS):
        native.step(DT)
    _close(final, functional.extract(native).state)
    _close(_one_stream(final), native_pulse.generator.get_state())
    _close(pulse.generator.get_state(), source_rng)


def test_sde_only_borrowed_stream_preserves_empty_initialization_draws_and_runtime():
    def case():
        cell = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-45.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        cell.insert(_SDEOnlyInjectedLeak.rename("probe"))
        cell.train()
        cell.initialize()
        stream = cell.mech.probe.DE._SDEOnlyGate.x_dW_rng
        stream.reseed(812)
        pulse = dn.mono_rect(amp=0.2, pw=0.045).poisson(
            interval=0.055,
            n=5,
            start=-0.035,
            noise=0.7,
            randomize_every_call=True,
            generator=stream.rng,
        )
        cell[:, 1].inject(pulse)
        cell.initialize()
        cell.integrator._initialize(cell, cell.t.new_tensor(DT), force=True)
        cell.detach()
        return cell, pulse

    source, pulse = case()
    native, native_pulse = case()
    _close(pulse.generator.get_state(), native_pulse.generator.get_state())
    functional, tensors = _lower(source)
    frozen = _clone(tensors)
    source_rng = pulse.generator.get_state().clone()
    random_draw = functional.sample_initialization_draw(
        tensors.parameters, tensors.state
    )
    assert random_draw.draws == {}
    stimulation = functional.sample_stimulation_initialization_draw(tensors.state)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(
            random_draw=random_draw, stimulation=stimulation
        ),
        state=tensors.state,
    )
    _initialize_native(native)
    _close(initialized.state, functional.extract(native).state)
    _close(_one_stream(initialized.state), native_pulse.generator.get_state())
    final, _ = functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        _draw_input(functional, initialized.state, STEPS),
        steps=STEPS,
    )
    for _ in range(STEPS):
        native.step(DT)
    _close(final, functional.extract(native).state)
    _close(_one_stream(final), native_pulse.generator.get_state())
    _close(tensors, frozen)
    _close(pulse.generator.get_state(), source_rng)


@pytest.mark.parametrize("position", (0, STEPS - 1))
def test_joint_runtime_tapes_reject_inconsistent_intermediate_rng_frames(position):
    source, _, pulse = _population()
    functional, tensors = _lower(source)
    inputs = _draw_input(functional, tensors.state, STEPS)
    changed = _clone(inputs.stimulation.rng)
    frame = next(iter(changed.values()))
    frame[position, 0] ^= 1
    inputs = inputs._replace(stimulation=inputs.stimulation._replace(rng=changed))
    frozen = _clone(tensors)
    source_rng = pulse.generator.get_state().clone()
    with pytest.raises(ValueError, match="different end frames"):
        functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            inputs,
            steps=STEPS,
        )
    _close(tensors, frozen)
    _close(pulse.generator.get_state(), source_rng)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_runner_rejects_early_frame_mismatch_before_entering_transition(
    execution, monkeypatch
):
    source, _, pulse = _population(transition_guard=True)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        execution=execution,
        inference=execution == "while",
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
    )
    functional = runner.functional
    tensors = functional.extract()
    frozen = _clone(tensors)
    source_rng = pulse.generator.get_state().clone()
    inputs = _draw_input(functional, tensors.state, STEPS)
    changed = _clone(inputs.stimulation.rng)
    next(iter(changed.values()))[0, 0] ^= 1
    inputs = inputs._replace(stimulation=inputs.stimulation._replace(rng=changed))
    monkeypatch.setattr(_TransitionGuardLeak, "FAIL_ON_ADVANCE", True)
    with pytest.raises(ValueError, match="different end frames"):
        runner(inputs=inputs)
    _close(functional.extract(), frozen)
    _close(pulse.generator.get_state(), source_rng)


def test_joint_explicit_initialization_packets_replay_and_require_both_families():
    source, _, pulse = _population()
    native, _, native_pulse = _population()
    functional, tensors = _lower(source)
    frozen = _clone(tensors)
    source_rng = pulse.generator.get_state().clone()
    random_draw = functional.sample_initialization_draw(
        tensors.parameters, tensors.state
    )
    stimulation = functional.sample_stimulation_initialization_draw(tensors.state)
    inputs = tensors.initialization._replace(
        random_draw=random_draw, stimulation=stimulation
    )
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, inputs, state=tensors.state
    )
    _close(
        initialized,
        functional.initialize(
            tensors.parameters, tensors.constants, inputs, state=tensors.state
        ),
    )
    _initialize_native(native)
    _close(initialized.state, functional.extract(native).state)
    _close(_one_stream(initialized.state), native_pulse.generator.get_state())
    for missing in ("random_draw", "stimulation"):
        with pytest.raises(ValueError, match="both|together|shared|Shared"):
            functional.initialize_host(
                tensors.parameters,
                tensors.constants,
                inputs._replace(**{missing: None}),
                state=tensors.state,
            )
    _close(tensors, frozen)
    _close(pulse.generator.get_state(), source_rng)


@pytest.mark.parametrize("reseed", (False, True))
@pytest.mark.parametrize("network", (False, True))
def test_joint_fresh_initialization_preserves_reseeding_and_native_rng_order(
    reseed, network
):
    if network:
        source, _ = _network(reseed=reseed)
        native, _ = _network(reseed=reseed)
    else:
        source = _population(reseed=reseed)[0]
        native = _population(reseed=reseed)[0]
    functional, tensors = _lower(source)
    frozen = _clone(tensors)
    if network:
        initialized = functional._initialize_host(
            *tensors, functional.initialization_inputs
        )
    else:
        initialized = functional.initialize_host(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
        )
    _initialize_native(native)
    _close(initialized.state, functional.extract(native).state)
    _one_stream(initialized.state)
    _close(tensors, frozen)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_joint_compiled_rollout_records_restarts_and_resumes_checkpoint(
    network, execution
):
    source = _network()[0] if network else _population()[0]
    native = _network()[0] if network else _population()[0]
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        execution=execution,
        inference=execution == "while",
        backend="aot_eager",
        callbacks={
            "trace": dn.func.Recorder(
                ("left.v", "right.v", "t") if network else ("v", "t")
            )
        },
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    frozen = _clone(runner.functional.extract())
    first, auxiliary = runner()
    restarted, _ = runner()
    _close(first, restarted)
    saved = _clone(first)
    final, final_auxiliary = runner(
        saved,
        parameters=auxiliary["parameters"],
        constants=auxiliary["constants"],
        callback_state=auxiliary["callbacks"].state,
    )
    _initialize_native(native)
    for _ in range(2 * STEPS):
        _step_native(native)
    _close(final, runner.functional.extract(native).state)
    _one_stream(final)
    committed = _network()[0] if network else _population()[0]
    runner.functional.commit_state_(committed, final)
    checkpoint = _clone(committed.state_dict_for_checkpoint())
    restored = _network()[0] if network else _population()[0]
    restored.restore_dict_from_checkpoint(checkpoint)
    _close(runner.functional.extract(restored).state, final)
    for _ in range(STEPS):
        _step_native(restored)
    continued, _ = runner(
        final,
        parameters=final_auxiliary["parameters"],
        constants=final_auxiliary["constants"],
    )
    _close(continued, runner.functional.extract(restored).state)
    _close(saved, first)
    _close(runner.functional.extract(), frozen)


@pytest.mark.parametrize("ignore", (False, True))
@pytest.mark.parametrize("network", (False, True))
def test_joint_cached_initialization_preserves_rng_load_policy(ignore, network):
    def case():
        model = _network(retained=True)[0] if network else _population(retained=True)[0]
        for _ in range(2):
            _step_native(model)
        owners = model.populations.values() if network else (model,)
        for cell in owners:
            cell.mech.probe.stream.ignore_rng_on_load(ignore)
        if network:
            model.cache_state()
        else:
            model.cache("_steady_state")
        for _ in range(2):
            _step_native(model)
        return model

    source, native = case(), case()
    functional, tensors = _lower(source)
    if not network:
        assert functional._owned_cache_bound
    frozen = _clone(tensors)
    if network:
        initialized = functional._initialize_host(
            *tensors, functional.initialization_inputs
        )
    else:
        initialized = functional.initialize_host(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
        )
    _initialize_native(native)
    _close(initialized.state, functional.extract(native).state)
    _one_stream(initialized.state)
    _close(tensors, frozen)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_joint_stimulation_compiles_with_inductor_and_preserves_training_gradients(
    execution,
):
    source, _, pulse = _population()
    native, _, native_pulse = _population()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        execution=execution,
        inference=execution == "while",
        backend="inductor",
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        **(
            {"trainable": ("*waveform.amp",), "checkpoint_every": 2}
            if execution == "scan"
            else {}
        ),
    )
    source_rng = pulse.generator.get_state().clone()
    final, _ = runner()
    for _ in range(STEPS):
        native.step(DT)
    _close(final, runner.functional.extract(native).state)
    _close(_one_stream(final), native_pulse.generator.get_state())
    if execution == "scan":
        amplitude = next(
            value
            for name, value in runner.trainable_parameters.items()
            if name.endswith("waveform.amp")
        )
        actual = torch.autograd.grad(
            final["integrator"]["v"].square().mean(), amplitude
        )[0]
        expected = torch.autograd.grad(
            native.v.square().mean(), native_pulse.waveform.amp
        )[0]
        assert expected.abs().sum() > 0
        _close(actual, expected)
    _close(pulse.generator.get_state(), source_rng)
