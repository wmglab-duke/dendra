"""Network NOISE tapes preserve native phases, detached values, and RNG carry."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import expsyn, graded_syn
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_bitpacked_inference import _sync_stimulus
from tests.func.test_network_cache_initialization import (
    _cache_initialized_native_packet,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
                yield
    finally:
        torch.set_num_threads(previous)


class _NoisyGate(State):
    State.STATE("m")
    State.RANGE(tau=1.1)
    State.RANGENOISE(
        "drive", distribution="normal", mu=0.2, sigma=0.03, scale="dW", seed=13
    )
    State.DERIVATIVE("m' = (drive - m) / tau")

    def state_defaults(self, v, values):
        del values
        return {"m": torch.full_like(v, 0.5)}


class _NoisyLeak(Mechanism):
    Mechanism.STATE_BUNDLE(_NoisyGate)
    Mechanism.RANGE(g=0.001)
    Mechanism.GLOBALNOISE("e", distribution="uniform", low=-68.0, high=-62.0, seed=10)
    Mechanism.BATCHNOISE("scale", distribution="lognormal", mu=0.0, sigma=0.1, seed=11)
    Mechanism.RANGENOISE(
        "eta", distribution="normal", mu=0.0, sigma=0.00002, scale="white", seed=12
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        # Retain each native timestep's detached coefficient for backward;
        # runtime sampling keeps native buffer identity stable across steps.
        conductance = self.g * self.scale.clone() * self.m
        return conductance, conductance * self.e.clone() - self.eta.clone()


class _NoiseSource(Mechanism):
    Mechanism.RANGENOISE("level", distribution="normal", mu=0.2, sigma=0.5, seed=31)


class _SharedRandomNoiseLeak(Mechanism):
    Mechanism.GLOBALRAND(
        "g", distribution="lognormal", mu=-7.0, sigma=0.1, seed=15, rng_name="stream"
    )
    Mechanism.GLOBALNOISE(
        "e", distribution="uniform", low=-68.0, high=-62.0, seed=16, rng_name="stream"
    )
    Mechanism.RANGENOISE(
        "eta", distribution="normal", mu=0.0, sigma=0.00002, seed=17, rng_name="stream"
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e - self.eta


def _network(*, mechanism=_NoisyLeak, stimulus=False):
    cells = {}
    for name in ("left", "right"):
        cell = dn.SingleCompartment(
            N=2, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(mechanism)
        if stimulus:
            cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        cells[name] = cell
    source = (
        dn.NetStim(N=4, interval=0.5, start=0.0, noise=0.6, max_spikes=20, seed=812)
        if stimulus
        else None
    )
    network = dn.Network(
        cells,
        netstim=source,
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    if source is not None:
        for cell in cells.values():
            network.connect_one_to_one(
                source[:], cell[:], cell.mech.syn, weight=0.003, delay=DT
            )
    network.train()
    network.initialize(DT)
    if source is not None:
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=False,
            train_delay_backend="dense",
        )
    return network


def _recording():
    return {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}


def _noise_inputs(functional, tensors, steps):
    packets = functional.sample_population_noise_tapes(tensors.state, steps)
    return dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(noise=packet)
            for name, packet in packets.items()
            if packet is not None
        },
        netstim=functional.sample_draw_tape(tensors.state, steps),
    )


def _slice(inputs, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], inputs)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_network_noise_host_restart_continuation_recording_and_native_checkpoint(
    execution,
):
    source, native = _network(stimulus=True), _network(stimulus=True)
    _sync_stimulus(source, native)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        execution=execution,
        inference=execution == "while",
        callbacks=_recording(),
        backend="aot_eager",
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    before = _clone(runner.functional.extract(), detached=True)
    global_rng = torch.get_rng_state().clone()
    prefix, first = runner()
    replay, replay_auxiliary = runner()
    _close(prefix, replay)
    _close(first["callbacks"]["trace"], replay_auxiliary["callbacks"]["trace"])
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    with torch.no_grad():
        expected = {name: [cell.v.clone()] for name, cell in native.populations.items()}
        for _ in range(8):
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
        torch.testing.assert_close(observed, torch.stack(expected[name]))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)
    torch.testing.assert_close(torch.get_rng_state(), global_rng)
    runner.functional.commit_state_(native, final)
    restored = _network(stimulus=True)
    restored.restore_dict_from_checkpoint(native.state_dict_for_checkpoint())
    continued, _ = runner(final)
    with torch.no_grad():
        for _ in range(4):
            restored.step()
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("execution", ("eager", "scan"))
def test_network_noise_explicit_tapes_replay_native_and_split_chunks(execution):
    source, native = _network(stimulus=True), _network(stimulus=True)
    _sync_stimulus(source, native)
    functional, tensors = dn.func.make_functional_network(source)
    before = _clone(tensors, detached=True)
    inputs = _noise_inputs(functional, tensors, 6)
    _close(inputs, _noise_inputs(functional, tensors, 6))
    frozen_inputs = _clone(inputs, detached=True)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks(_recording())
    execute = (
        functional.rollout
        if execution == "eager"
        else functional.compile_rollout_chunk(3, execution="scan", backend="aot_eager")
    )
    prefix, first = execute(
        tensors.parameters,
        prepared,
        tensors.state,
        _slice(inputs, 0, 3),
        **({"steps": 3} if execution == "eager" else {}),
        callbacks=callbacks,
    )
    final, second = execute(
        tensors.parameters,
        prepared,
        prefix,
        _slice(inputs, 3, 6),
        **({"steps": 3} if execution == "eager" else {}),
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
    )
    with torch.no_grad():
        expected = [native.left.v.clone()]
        for _ in range(6):
            native.step()
            expected.append(native.left.v.clone())
    recorded = torch.cat(
        (first["callbacks"]["trace"]["left.v"], second["callbacks"]["trace"]["left.v"])
    )
    torch.testing.assert_close(recorded, torch.stack(expected))
    _close(final, functional.extract(native).state)
    _close(tensors, before)
    _close(inputs, frozen_inputs)


@pytest.mark.parametrize("mechanism", (_NoisyLeak, _SharedRandomNoiseLeak))
def test_network_noise_fresh_initialization_preserves_rand_noise_draw_order(mechanism):
    source, native = _network(mechanism=mechanism), _network(mechanism=mechanism)
    functional, tensors = dn.func.make_functional_network(source)
    packets = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    inputs = functional.initialization_inputs._replace(
        populations={
            name: values._replace(random_draw=packets[name])
            for name, values in functional.initialization_inputs.populations.items()
        }
    )
    initialized = functional.initialize(*tensors, inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized.state, functional.extract(native).state)
    step_inputs = {
        name: dn.func.StepInput(
            noise=plan.sample_noise_draw(initialized.state["populations"][name])
        )
        for name, plan in functional.populations.items()
    }
    actual, _ = functional.prepare_and_step(*initialized, inputs=step_inputs)
    native.step()
    _close(actual, functional.extract(native).state)


def test_network_noise_only_fresh_runner_continues_without_effective_constants():
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=2,
        initialization="fresh",
        inference=False,
        checkpoint_every=1,
        backend="aot_eager",
    )
    before = _clone(runner.functional.extract(), detached=True)
    prefix, _ = runner()
    final, _ = runner(prefix)
    # Fresh preparation owns active materializations; native extraction is a
    # snapshot with inactive graph anchors. The source itself stays unchanged.
    _close(runner.functional.extract(), before)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    for _ in range(4):
        native.step()
    _close(final, runner.functional.extract(native).state)


@pytest.mark.parametrize("global_cache", (False, True))
def test_network_noise_fresh_cache_restores_saved_buffers_and_rng(global_cache):
    source, native = _network(), _network()
    for network in (source, native):
        for _ in range(2):
            network.step()
        for cell in network.populations.values():
            cell.cache("_steady_state")
        if global_cache:
            network.cache_state()
        for _ in range(2):
            network.step()
    functional, tensors = dn.func.make_functional_network(source)
    before = _clone(functional.extract(), detached=True)
    draws = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    assert all(value is None for value in draws.values())
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    expected = _cache_initialized_native_packet(functional, native)
    _close(initialized.state, expected.state)
    _close(functional.extract(), before)
    step_inputs = {
        name: dn.func.StepInput(
            noise=plan.sample_noise_draw(initialized.state["populations"][name])
        )
        for name, plan in functional.populations.items()
    }
    actual, _ = functional.prepare_and_step(*initialized, inputs=step_inputs)
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("checkpoint_every", (None, 2))
def test_network_noise_gradients_match_native_and_exclude_distribution_and_draws(
    checkpoint_every,
):
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        inference=False,
        trainable=("*g_param", "*eta_mu_param"),
        callbacks=_recording(),
        backend="aot_eager",
        checkpoint_every=checkpoint_every,
    )
    inputs = _noise_inputs(runner.functional, runner.functional.extract(), 4)
    noise_draw = next(iter(inputs.populations["left"].noise.draws.values()))
    noise_draw.requires_grad_()
    g = runner.get_parameter("populations.left.*g_param")
    noise_mu = runner.get_parameter("populations.left.*eta_mu_param")
    final, auxiliary = runner(inputs=inputs)
    trace = auxiliary["callbacks"]["trace"]["left.v"]
    loss = trace.square().mean()
    gradients = torch.autograd.grad(loss, (g, noise_mu, noise_draw), allow_unused=True)
    assert all(
        value is None or torch.count_nonzero(value) == 0 for value in gradients[1:]
    )
    assert all(
        not value.requires_grad
        for value in final["populations"]["left"]["runtime_noise"].values()
    )
    assert torch.isfinite(gradients[0]).all() and gradients[0].abs().sum() > 0
    expected = [native.left.v.clone()]
    for _ in range(4):
        native.step()
        expected.append(native.left.v.clone())
    native_loss = torch.stack(expected).square().mean()
    expected_gradient = torch.autograd.grad(
        native_loss, native.left.mech._NoisyLeak.g_param
    )[0]
    torch.testing.assert_close(gradients[0], expected_gradient)
    _close(final, runner.functional.extract(native).state)


def test_network_connections_read_retained_noise_before_population_refresh():
    def connected():
        source, target = _network(), _network()
        left, right = source.left, target.right
        left.insert(_NoiseSource.rename("noise"))
        right.insert(graded_syn.rename("analog"), e=-10.0)
        network = dn.Network({"left": left, "right": right})
        network.connect_continuous_one_to_one(
            left[:],
            right[:],
            right.mech.analog,
            pre_var="mech.noise.level",
            input="g_pre",
            weight=0.001,
            delay=0.0,
            transform=torch.sigmoid,
        )
        network.train()
        network.initialize(DT)
        return network

    source, native = connected(), connected()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        inference=False,
        callbacks=_recording(),
        backend="aot_eager",
    )
    final, auxiliary = runner()
    expected = [native.right.v.clone()]
    for _ in range(4):
        native.step()
        expected.append(native.right.v.clone())
    torch.testing.assert_close(
        auxiliary["callbacks"]["trace"]["right.v"], torch.stack(expected)
    )
    _close(final, runner.functional.extract(native).state)


def test_network_noise_pure_rollout_requires_complete_matching_tapes():
    functional, tensors = dn.func.make_functional_network(_network())
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    before = _clone(tensors, detached=True)
    with pytest.raises(ValueError, match="noise|NOISE|draw"):
        functional.rollout(tensors.parameters, prepared, tensors.state, steps=2)
    inputs = _noise_inputs(functional, tensors, 2)
    with pytest.raises(ValueError, match="steps|samples|leading|shape"):
        functional.rollout(tensors.parameters, prepared, tensors.state, inputs, steps=3)
    _close(tensors, before)


@pytest.mark.slow
def test_network_noise_while_inductor_cpp_matches_native():
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=5,
        execution="while",
        inference=True,
        callbacks=_recording(),
        backend="inductor",
        options={"cpp_wrapper": True, "compile_threads": 1},
    )
    final, auxiliary = runner()
    with torch.no_grad():
        expected = [native.left.v.clone()]
        for _ in range(5):
            native.step()
            expected.append(native.left.v.clone())
    torch.testing.assert_close(
        auxiliary["callbacks"]["trace"]["left.v"], torch.stack(expected)
    )
    _close(final, runner.functional.extract(native).state)
    assert runner.execution_report().strategy == "compiled_while_loop"
