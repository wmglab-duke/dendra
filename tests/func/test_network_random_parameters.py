"""Network RAND heterogeneity shares native draw, restart, and replay semantics."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import expsyn
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


class _RandomGate(State):
    State.STATE("m")
    State.RANGERAND("tau", distribution="normal", mu=1.0, sigma=0.1, seed=23)
    State.DERIVATIVE("m' = (0.25 - m) / tau")

    def state_defaults(self, v, values):
        del values
        return {"m": torch.full_like(v, 0.9)}


class _RandomLeak(Mechanism):
    Mechanism.STATE_BUNDLE(_RandomGate)
    Mechanism.RANGERAND("g", distribution="lognormal", mu=-7.0, sigma=0.2, seed=20)
    Mechanism.GLOBALRAND("e", distribution="uniform", low=-70.0, high=-55.0, seed=21)
    Mechanism.BATCHRAND(
        "scale",
        distribution="truncated_normal",
        mu=1.0,
        sigma=0.1,
        low=0.7,
        high=1.3,
        seed=22,
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        conductance = self.g * self.scale * self.m
        return conductance, conductance * self.e


class _SharedStreamLeak(Mechanism):
    Mechanism.GLOBALRAND(
        "offset", distribution="uniform", low=-1.0, high=1.0, rng_name="stream", seed=40
    )
    Mechanism.RANGERAND(
        "g",
        distribution="lognormal",
        mu=-7.0,
        sigma=0.2,
        rng_name="stream",
        seed=999,
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * (-65.0 + self.offset)


class _StickyLeak(Mechanism):
    Mechanism.RANGERAND(
        "g",
        distribution="lognormal",
        mu=-7.0,
        sigma=0.2,
        seed=80,
        resample_on_initialize=False,
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * -65.0


class _ReseededLeak(Mechanism):
    Mechanism.RANGERAND(
        "g",
        distribution="lognormal",
        mu=-7.0,
        sigma=0.2,
        seed=37,
        reseed_on_initialize=True,
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * -65.0


class _ReseededStickyLeak(Mechanism):
    Mechanism.RANGERAND(
        "g",
        distribution="lognormal",
        mu=-7.0,
        sigma=0.2,
        seed=38,
        resample_on_initialize=False,
        reseed_on_initialize=True,
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * -65.0


class _SharedReseededStreamLeak(Mechanism):
    Mechanism.GLOBALRAND(
        "offset",
        distribution="uniform",
        low=-1.0,
        high=1.0,
        rng_name="stream",
        seed=40,
        reseed_on_initialize=True,
    )
    Mechanism.RANGERAND(
        "g",
        distribution="lognormal",
        mu=-7.0,
        sigma=0.2,
        rng_name="stream",
        seed=999,
        reseed_on_initialize=True,
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * (-65.0 + self.offset)


def _network(*, mechanism=_RandomLeak, stimulus=False):
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


def _initialization_draws(functional, tensors):
    defaults = functional.initialization_inputs
    draws = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    populations = {
        name: inputs._replace(random_draw=draws[name])
        for name, inputs in defaults.populations.items()
    }
    return defaults._replace(populations=populations)


def _recording():
    return {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_network_random_snapshot_preserves_samples_and_matches_native(execution):
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        execution=execution,
        inference=True,
        callbacks=_recording(),
        backend="aot_eager",
    )
    before = _clone(runner.functional.extract(), detached=True)
    final, auxiliary = runner()
    repeated, repeated_auxiliary = runner()
    _close(final, repeated)
    _close(auxiliary["callbacks"]["trace"], repeated_auxiliary["callbacks"]["trace"])
    with torch.no_grad():
        for _ in range(4):
            native.step()
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)


@pytest.mark.parametrize("mechanism", (_RandomLeak, _SharedStreamLeak))
def test_network_random_explicit_fresh_draws_match_next_native_initialization(
    mechanism,
):
    source, native = _network(mechanism=mechanism), _network(mechanism=mechanism)
    functional, tensors = dn.func.make_functional_network(source)
    before = _clone((tensors, functional.initialization_inputs), detached=True)
    inputs = _initialization_draws(functional, tensors)
    _close(inputs, _initialization_draws(functional, tensors))
    initialized = functional.initialize(*tensors, inputs)
    # Both models start at the captured RNG suffix. RAND seeds do not reseed
    # initialization; the native constructor and first initialize already drew.
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized.state, functional.extract(native).state)
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)
    _close((tensors, functional.initialization_inputs), before)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_network_random_fresh_restart_continuation_and_checkpoint(execution):
    source, native = _network(stimulus=True), _network(stimulus=True)
    _sync_stimulus(source, native)
    options = {"checkpoint_every": 2} if execution == "scan" else {}
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        execution=execution,
        inference=execution == "while",
        callbacks=_recording(),
        backend="aot_eager",
        **options,
    )
    before = _clone(runner.functional.extract(), detached=True)
    prefix, first = runner()
    replay, replay_auxiliary = runner()
    _close(prefix, replay)
    _close(first["callbacks"]["trace"], replay_auxiliary["callbacks"]["trace"])
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    with torch.no_grad():
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
        expected = {name: [cell.v.clone()] for name, cell in native.populations.items()}
        for _ in range(8):
            native.step()
            for name, cell in native.populations.items():
                expected[name].append(cell.v.clone())
    for name in native.populations:
        recorded = torch.cat(
            (
                first["callbacks"]["trace"][f"{name}.v"],
                second["callbacks"]["trace"][f"{name}.v"],
            )
        )
        torch.testing.assert_close(recorded, torch.stack(expected[name]))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), before)
    runner.functional.commit_state_(native, final)
    restored = _network(stimulus=True)
    restored.restore_dict_from_checkpoint(native.state_dict_for_checkpoint())
    continued, _ = runner.functional.prepare_and_step(
        second["parameters"],
        second["constants"],
        final,
        netstim_draw=runner.functional.sample_step_draw(final),
    )
    with torch.no_grad():
        restored.step()
    _close(continued, runner.functional.extract(restored).state)


def test_network_random_draw_initialization_supports_fullgraph_capture():
    functional, tensors = dn.func.make_functional_network(_network())
    inputs = _initialization_draws(functional, tensors)
    eager = functional.initialize(*tensors, inputs)
    compiled = torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
    _close(compiled(*tensors, inputs), eager)


def test_network_random_checkpoint_gradients_retain_shared_distribution_parameters():
    parameter = "integrator.mech.mechanisms._RandomLeak.g_mu_param"
    tau = "integrator.mech.mechanisms._RandomLeak.DE._RandomGate.tau_mu_param"
    source = _network()
    source.right.mech._RandomLeak.g_mu_param = source.left.mech._RandomLeak.g_mu_param
    source.initialize(DT, reinit_weights=False, reinit_delays=False)
    results = []
    for checkpoint_every in (None, 2):
        runner = dn.compile(
            source,
            dt=DT,
            steps=4,
            initialization="fresh",
            inference=False,
            trainable=("*g_mu_param", "*tau_mu_param"),
            callbacks=_recording(),
            backend="aot_eager",
            checkpoint_every=checkpoint_every,
        )
        shared = runner.get_parameter(f"populations.left.{parameter}")
        assert shared is runner.get_parameter(f"populations.right.{parameter}")
        timescale = runner.get_parameter(f"populations.left.{tau}")
        initial = runner.initialization_inputs
        draws = runner.functional.sample_population_initialization_draws(
            runner.parameters, runner.initial_state
        )
        initial = initial._replace(
            populations={
                name: values._replace(random_draw=draws[name])
                for name, values in initial.populations.items()
            }
        )
        base_draw = next(
            value
            for name, value in draws["left"].draws.items()
            if name.endswith(".g.0")
        ).requires_grad_()
        initial.populations["left"].v_init.requires_grad_()
        initial_voltage = initial.populations["left"].v_init
        drive = torch.linspace(0, 0.0002, 4, dtype=DTYPE).reshape(4, 1, 1)
        drive.requires_grad_()
        inputs = dn.func.NetworkRolloutInput(
            populations={"left": dn.func.RolloutInput(intra=drive)}
        )
        prefix, first = runner(inputs=inputs, initialization=initial)
        final, second = runner(
            prefix,
            inputs=inputs,
            parameters=first["parameters"],
            constants=first["constants"],
            callback_state=first["callbacks"].state,
        )
        voltage = torch.cat(
            (
                first["callbacks"]["trace"]["left.v"],
                second["callbacks"]["trace"]["left.v"],
            )
        )
        loss = (
            voltage.square().mean()
            + final["populations"]["right"]["integrator"]["v"].square().mean()
        )
        gradients = torch.autograd.grad(
            loss, (shared, timescale, initial_voltage, drive, base_draw)
        )
        assert all(
            torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
        )
        results.append((final, voltage, gradients))
    _close(results[0], results[1])


@pytest.mark.parametrize("global_cache", (False, True))
def test_network_random_resampling_invalidates_owned_cache_but_global_cache_wins(
    global_cache,
):
    source, native = _network(), _network()
    for network in (source, native):
        for cell in network.populations.values():
            cell.cache("_steady_state")
        if global_cache:
            network.cache_state()
    functional, tensors = dn.func.make_functional_network(source)
    inputs = _initialization_draws(functional, tensors)
    initialized = functional.initialize(*tensors, inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    expected = (
        _cache_initialized_native_packet(functional, native)
        if global_cache
        else functional.extract(native)
    )
    _close(initialized.state, expected.state)
    if global_cache:
        assert all(value.random_draw is None for value in inputs.populations.values())
        assert all(
            "_steady_state" in cell._caches for cell in native.populations.values()
        )
    else:
        assert all(
            value.random_draw is not None for value in inputs.populations.values()
        )
        assert all(
            "_steady_state" not in cell._caches for cell in native.populations.values()
        )
    # The policy operates on explicit packets and does not clear the source.
    assert all("_steady_state" in cell._caches for cell in source.populations.values())


def test_network_random_sticky_owned_cache_restores_sample_without_drawing():
    source, native = _network(mechanism=_StickyLeak), _network(mechanism=_StickyLeak)
    for network in (source, native):
        for cell in network.populations.values():
            cell.cache("_steady_state")
    functional, tensors = dn.func.make_functional_network(source)
    inputs = _initialization_draws(functional, tensors)
    assert all(value.random_draw is None for value in inputs.populations.values())
    initialized = functional.initialize(*tensors, inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(
        initialized.state, _cache_initialized_native_packet(functional, native).state
    )
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize(
    "mechanism", (_RandomLeak, _ReseededLeak, _SharedReseededStreamLeak)
)
def test_network_random_reseed_policy_uses_selected_successive_rng_boundary(mechanism):
    source, native = _network(mechanism=mechanism), _network(mechanism=mechanism)
    functional, tensors = dn.func.make_functional_network(source)
    first_inputs = _initialization_draws(functional, tensors)
    first = functional.initialize(*tensors, first_inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(first.state, functional.extract(native).state)
    second_inputs = _initialization_draws(functional, first)
    first_draws = first_inputs.populations["left"].random_draw.draws
    second_draws = second_inputs.populations["left"].random_draw.draws
    if mechanism is _RandomLeak:
        assert any(
            not torch.equal(first_draws[name], second_draws[name])
            for name in first_draws
        )
    else:
        _close(first_draws, second_draws)
    parameters = _clone(first.parameters)
    if mechanism is _ReseededLeak:
        parameter = "integrator.mech.mechanisms._ReseededLeak.g_mu_param"
        parameters["populations"]["left"][parameter] = (
            parameters["populations"]["left"][parameter] + 0.2
        )
        with torch.no_grad():
            native.left.mech._ReseededLeak.g_mu_param.add_(0.2)
    second = functional.initialize(
        parameters, first.constants, first.state, second_inputs
    )
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(second.state, functional.extract(native).state)
    if mechanism is _ReseededLeak:
        assert any(
            not torch.equal(
                value, second.state["populations"]["left"]["random"]["values"][name]
            )
            for name, value in first.state["populations"]["left"]["random"][
                "values"
            ].items()
            if name.startswith("random.values.")
        )


def test_network_random_sticky_reseed_does_not_reset_or_advance_stream():
    source, native = (
        _network(mechanism=_ReseededStickyLeak),
        _network(mechanism=_ReseededStickyLeak),
    )
    functional, tensors = dn.func.make_functional_network(source)
    inputs = _initialization_draws(functional, tensors)
    initialized = functional.initialize(*tensors, inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized.state, functional.extract(native).state)
    for name in functional.populations:
        _close(
            initialized.state["populations"][name]["random"],
            tensors.state["populations"][name]["random"],
        )


@pytest.mark.slow
def test_network_random_fresh_while_inductor_cpp_matches_native():
    source, native = (
        _network(mechanism=_SharedReseededStreamLeak),
        _network(mechanism=_SharedReseededStreamLeak),
    )
    runner = dn.compile(
        source,
        dt=DT,
        steps=5,
        initialization="fresh",
        execution="while",
        inference=True,
        callbacks=_recording(),
        backend="inductor",
        options={"cpp_wrapper": True, "compile_threads": 1},
    )
    actual, auxiliary = runner()
    with torch.no_grad():
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
        expected = [native.left.v.clone()]
        for _ in range(5):
            native.step()
            expected.append(native.left.v.clone())
    torch.testing.assert_close(
        auxiliary["callbacks"]["trace"]["left.v"], torch.stack(expected)
    )
    _close(actual, runner.functional.extract(native).state)
    assert runner.execution_report().strategy == "compiled_while_loop"
