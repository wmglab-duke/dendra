"""Explicit State SDE tapes preserve paths, gradients, phases, and RNG suffixes."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import graded_syn
from tests.func.test_network_cache_initialization import (
    _cache_initialized_native_packet,
)

DT = 0.125
STEPS = 4
DTYPE = torch.float64
pytestmark = pytest.mark.cpu


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


class _OU(State):
    State.STATE("x")
    State.RANGE(theta=0.7, mu=0.2, sigma=0.15)
    State.DERIVATIVE("x' = theta * (mu - x)")
    State.DIFFUSION("x = sigma")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


class _MultiplicativeHeun(State):
    State.STATE("x")
    State.RANGE(theta=0.7, mu=0.2, sigma=0.15)
    State.ASSIGNED("diffusion")
    State.DERIVATIVE("x' = theta * (mu - x)")
    State.DIFFUSION("x = diffusion")
    State.METHOD("euler_heun", average_drift=True)

    def assigned_values(self, v, values):
        del v
        return {"diffusion": self.sigma * values["x"]}

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


class _SharedNoiseOU(State):
    State.STATE("x")
    State.RANGE(theta=0.7, mu=0.2, sigma=0.15)
    State.RANGENOISE("eta", mu=0.0, sigma=0.01, seed=31, rng_name="x_dW_rng")
    State.DERIVATIVE("x' = theta * (mu - x) + eta")
    State.DIFFUSION("x = sigma")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


class _DrivenLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + 0.002 * self.x


class _OULeak(_DrivenLeak):
    Mechanism.STATE_BUNDLE(_OU)


class _HeunLeak(_DrivenLeak):
    Mechanism.STATE_BUNDLE(_MultiplicativeHeun)


class _SharedLeak(_DrivenLeak):
    Mechanism.STATE_BUNDLE(_SharedNoiseOU)


def _clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)


def _population(mechanism=_OULeak, *, seed=171):
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(mechanism.rename("sde"))
    population.train()
    population.initialize()
    for state in population.mech.sde.DE.values():
        for offset, name in enumerate(state._sde_rng_names):
            getattr(state, name).reseed(seed + offset)
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def _network(mechanism=_OULeak, *, connected=True):
    left, right = _population(mechanism), _population(mechanism, seed=271)
    if connected:
        right.insert(graded_syn.rename("graded"), e=-10.0)
    network = dn.Network({"left": left, "right": right})
    if connected:
        network.connect_continuous_one_to_one(
            left[:],
            right[:],
            right.mech.graded,
            pre_var="mech.sde.x",
            input="g_pre",
            weight=0.001,
            delay=0.0,
            transform=torch.sigmoid,
        )
    network.train()
    network.initialize(DT)
    for population, seed in ((left, 171), (right, 271)):
        for state in population.mech.sde.DE.values():
            for offset, name in enumerate(state._sde_rng_names):
                getattr(state, name).reseed(seed + offset)
    return network


def _slice(tape, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], tape)


def _network_inputs(functional, state, steps):
    return dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(noise=tape)
            for name, tape in functional.sample_population_noise_tapes(
                state, steps
            ).items()
            if tape is not None
        }
    )


@pytest.mark.parametrize("mechanism", (_OULeak, _HeunLeak, _SharedLeak))
@pytest.mark.parametrize("execution", ("eager", "scan"))
def test_population_sde_explicit_tapes_match_native_and_checkpoint_suffix(
    mechanism, execution
):
    source, native = _population(mechanism), _population(mechanism)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    before = _clone((tensors, functional.extract()))
    global_rng = torch.get_rng_state().clone()
    tape = functional.sample_noise_draw_tape(tensors.state, 2 * STEPS)
    _close(tape, functional.sample_noise_draw_tape(tensors.state, 2 * STEPS))
    saved_tape = _clone(tape)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("v", "sde.x", "t"))}
    )
    executable = (
        functional
        if execution == "eager"
        else functional.compile_rollout_chunk(
            STEPS, execution="scan", backend="aot_eager"
        )
    )
    bound = executable.bind(tensors.parameters, prepared)
    prefix, first = dn.func.run(
        bound,
        tensors.state,
        steps=STEPS,
        inputs=dn.func.RolloutInput(noise=_slice(tape, 0, STEPS)),
        callbacks=callbacks,
    )
    suffix = functional.sample_noise_draw_tape(prefix, STEPS)
    _close(suffix, _slice(tape, STEPS, 2 * STEPS))
    final, second = dn.func.run(
        bound,
        prefix,
        steps=STEPS,
        inputs=dn.func.RolloutInput(noise=suffix),
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
    )
    expected_v, expected_x = [native.v.clone()], [native.mech.sde.x.clone()]
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step(DT)
            expected_v.append(native.v.clone())
            expected_x.append(native.mech.sde.x.clone())
    for name, expected in (("v", expected_v), ("sde.x", expected_x)):
        recorded = torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        _close(recorded, torch.stack(expected))
    _close(final, functional.extract(native).state)
    _close((tensors, functional.extract()), before)
    _close(tape, saved_tape)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    committed = _population(mechanism)
    functional.commit_state_(committed, final)
    _close(functional.extract(committed).state, final)
    restored = _population(mechanism)
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = dn.func.run(
        bound,
        final,
        steps=STEPS,
        inputs=dn.func.RolloutInput(
            noise=functional.sample_noise_draw_tape(final, STEPS)
        ),
    )
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step(DT)
    _close(continued, functional.extract(restored).state)


def _oracle(initial, theta, mu, sigma, dt, draws, *, heun):
    value = initial
    for draw in draws.detach():
        dw = torch.sqrt(dt) * draw
        drift = theta * (mu - value)
        diffusion = sigma * value if heun else sigma
        predictor = value + drift * dt + diffusion * dw
        if heun:
            value = (
                value
                + 0.5 * (drift + theta * (mu - predictor)) * dt
                + 0.5 * (diffusion + sigma * predictor) * dw
            )
        else:
            value = predictor
    return value


@pytest.mark.parametrize("mechanism", (_OULeak, _HeunLeak))
def test_population_sde_transient_draw_slots_restore_without_retaining_graph(mechanism):
    population = _population(mechanism)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    before = _clone(functional.extract())
    rng_before = torch.get_rng_state().clone()
    owned_mechanism = functional._transition.population.mech.sde
    slots = [
        (state, slot, state._buffers[slot], state._buffers[slot].clone())
        for state in owned_mechanism.DE.values()
        for _name, slot in state._functional_sde_slots
    ]
    assert slots
    packet = functional.sample_noise_draw(tensors.state)
    packet = packet._replace(
        draws={
            name: value.detach().clone().requires_grad_()
            for name, value in packet.draws.items()
        }
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    outputs = []
    for _ in range(2):
        final, _ = functional.step(
            tensors.parameters, prepared, tensors.state, dn.func.StepInput(noise=packet)
        )
        value = final["mechanisms"]["sde"]["x"]
        assert value.grad_fn is not None
        outputs.append(value)
        for state, name, original, saved in slots:
            assert state._buffers[name] is original
            torch.testing.assert_close(state._buffers[name], saved, rtol=0, atol=0)
            assert state._buffers[name].grad_fn is None
            assert not state._buffers[name].requires_grad
        _close(functional.extract(), before)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)


@pytest.mark.parametrize("mechanism", (_OULeak, _HeunLeak))
@pytest.mark.parametrize("checkpoint_every", (None, 2))
def test_scan_sde_pathwise_gradients_match_independent_equations(
    mechanism, checkpoint_every
):
    runner = dn.compile(
        _population(mechanism),
        dt=DT,
        steps=STEPS,
        inference=False,
        trainable=("*theta_param", "*mu_param", "*sigma_param"),
        execution="scan",
        backend="aot_eager",
        checkpoint_every=checkpoint_every,
    )
    initial_snapshot = _clone(runner.initial_state)
    state = _clone(runner.initial_state)
    initial = state["mechanisms"]["sde"]["x"].requires_grad_()
    constants = _clone(dict(runner.constants))
    dt = constants["dt"].requires_grad_()
    tape = runner.functional.sample_noise_draw_tape(state, STEPS)
    assert len(tape.draws) == 1
    tape = tape._replace(
        draws={
            name: draw.detach().clone().requires_grad_()
            for name, draw in tape.draws.items()
        }
    )
    draw = next(iter(tape.draws.values()))
    theta, mu, sigma = (
        runner.get_parameter(f"*{name}_param") for name in ("theta", "mu", "sigma")
    )
    final, _ = runner(
        state, constants=constants, inputs=dn.func.RolloutInput(noise=tape)
    )
    actual = final["mechanisms"]["sde"]["x"]
    expected = _oracle(initial, theta, mu, sigma, dt, draw, heun=mechanism is _HeunLeak)
    _close(actual, expected)
    operands = (theta, mu, sigma, dt, initial, draw)
    actual_gradients = torch.autograd.grad(
        actual.square().mean(), operands, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected.square().mean(), operands, allow_unused=True
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients[:-1], expected_gradients[:-1], strict=True
    ):
        assert expected_gradient is not None and expected_gradient.abs().sum() > 0
        _close(actual_gradient, expected_gradient)
    assert expected_gradients[-1] is None
    assert (
        actual_gradients[-1] is None or torch.count_nonzero(actual_gradients[-1]) == 0
    )
    assert not torch.equal(initial, actual)
    _close(runner.initial_state, initial_snapshot)


@pytest.mark.parametrize("mechanism", (_OULeak, _HeunLeak))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_network_sde_host_sampling_recording_continuation_and_native_commit(
    mechanism, execution
):
    source, native = _network(mechanism), _network(mechanism)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        execution=execution,
        inference=execution == "while",
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("left.v", "left.sde.x", "right.v", "t"))},
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    before = _clone(runner.functional.extract())
    global_rng = torch.get_rng_state().clone()
    prefix, first = runner()
    replay, repeated = runner()
    _close(prefix, replay)
    _close(dict(first["callbacks"]["trace"]), dict(repeated["callbacks"]["trace"]))
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    expected = {
        "left.v": [native.left.v.clone()],
        "right.v": [native.right.v.clone()],
        "left.sde.x": [native.left.mech.sde.x.clone()],
    }
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step()
            expected["left.v"].append(native.left.v.clone())
            expected["right.v"].append(native.right.v.clone())
            expected["left.sde.x"].append(native.left.mech.sde.x.clone())
    for name, values in expected.items():
        observed = torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        _close(observed, torch.stack(values))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    committed = _network(mechanism)
    runner.functional.commit_state_(committed, final)
    _close(runner.functional.extract(committed).state, final)
    restored = _network(mechanism)
    restored.restore_dict_from_checkpoint(committed.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step()
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("cache", (False, True))
def test_network_sde_fresh_initialization_and_cache_restore_rng_suffix(cache):
    source, native = (
        _network(_SharedLeak, connected=False),
        _network(_SharedLeak, connected=False),
    )
    for network in (source, native):
        for _ in range(2):
            network.step()
        if cache:
            for population in network.populations.values():
                population.cache("_steady_state")
            network.cache_state()
        for _ in range(2):
            network.step()
    functional, tensors = dn.func.make_functional_network(source)
    inputs = functional.initialization_inputs
    packets = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    inputs = inputs._replace(
        populations={
            name: values._replace(random_draw=packets[name])
            for name, values in inputs.populations.items()
        }
    )
    initialized = functional.initialize(*tensors, inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    expected = (
        _cache_initialized_native_packet(functional, native)
        if cache
        else functional.extract(native)
    )
    _close(initialized.state, expected.state)
    state, _ = functional.prepare_and_step(
        *initialized,
        inputs={
            name: dn.func.StepInput(
                noise=plan.sample_noise_draw(initialized.state["populations"][name])
            )
            for name, plan in functional.populations.items()
        },
    )
    native.step()
    _close(state, functional.extract(native).state)


def test_network_sde_pure_rollout_requires_matching_combined_tapes():
    functional, tensors = dn.func.make_functional_network(_network(_SharedLeak))
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    before = _clone(tensors)
    with pytest.raises(ValueError, match="noise|NOISE|SDE|draw"):
        functional.rollout(tensors.parameters, prepared, tensors.state, steps=STEPS)
    inputs = _network_inputs(functional, tensors.state, STEPS)
    expected, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, inputs, steps=STEPS
    )
    chunk = functional.compile_rollout_chunk(2, execution="scan", backend="aot_eager")
    prefix, _ = chunk(tensors.parameters, prepared, tensors.state, _slice(inputs, 0, 2))
    final, _ = chunk(tensors.parameters, prepared, prefix, _slice(inputs, 2, STEPS))
    _close(final, expected)
    with pytest.raises(ValueError, match="steps|samples|leading|shape"):
        functional.rollout(
            tensors.parameters, prepared, tensors.state, inputs, steps=STEPS + 1
        )
    _close(tensors, before)
