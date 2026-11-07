"""Mixed cable/point execution preserves stochastic lifecycle and shared VJPs."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn, graded_syn
from tests.func.test_functional_initialization import (
    _assert_tensor_snapshot,
    _tensor_snapshot,
)
from tests.func.test_native_cable import _morphology
from tests.func.test_network import _clone, _close
from tests.func.test_network_rng_sharing import PREFIX, RAW, _SharedLeak
from tests.func.test_stochastic_initialization_ordering import _native_close

DT = 0.001
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


class _Configured:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.insert(_SharedLeak.rename("probe"))
        self.insert(expsyn.rename("events"), tau=0.02, e=0.0)


class _Cable(_Configured, dn.Cable):
    pass


class _Cell(_Configured, dn.SingleCompartment):
    pass


def _release(voltage):
    return torch.sigmoid(0.03 * (voltage + 65.0))


def _network(*, training=False, cache_kind=None):
    cable = _Cable.from_morphology(
        _morphology(),
        N=1,
        v_init=-64.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    cell = _Cell(N=1, C=1, v_init=-62.0, integrator=dn.rk4(imem=False))
    cell.insert(graded_syn.rename("analog"), e=-10.0)
    cable.initialize()
    cell.initialize()
    # Interleave different consumer shapes on the same two physical streams:
    # the cable has four compartments, whereas the RK4 cell has only one.
    for population in (cable, cell):
        leak = population.mech.probe
        leak.g_mu_param = cable.mech.probe.g_mu_param
        leak.primary = cable.mech.probe.primary
        leak.secondary = cable.mech.probe.secondary
        leak.DE._SharedGate.primary = leak.primary
        leak.DE._SharedGate.secondary = leak.secondary
        leak.DE._SharedGate.m_dW_rng = leak.primary
    cable.mech.probe.primary.reseed(1203)
    cable.mech.probe.secondary.reseed(1207)
    stimulus = dn.NetStim(
        N=1, interval=2 * DT, start=0.0, noise=0.6, max_spikes=40, seed=812
    )
    network = dn.Network(
        {"cable": cable, "cell": cell},
        netstim=stimulus,
        seed=813,
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    for population in (cable, cell):
        network.connect_one_to_one(
            stimulus[:],
            population[:, 0],
            population.mech.events,
            weight=0.0001,
            delay=DT,
        )
    network.connect_continuous_one_to_one(
        cable[:, -1],
        cell[:],
        cell.mech.analog,
        pre_var="v",
        input="g_pre",
        weight=0.0002,
        delay=DT,
        transform=_release,
    )
    network.train(training)
    network.initialize(DT)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    if cache_kind is not None:
        network._enable_synapse_state_cache_recording()
        with torch.no_grad():
            network.step()
            network.step()
            if cache_kind == "owned":
                cable.cache("_steady_state")
                cell.cache("_steady_state")
            else:
                network.cache_state()
            network.step()
            network.step()
        network._disable_synapse_state_cache_recording(release=True)
    return network


def _assert_aliases(network):
    cable, cell = network.cable.mech.probe, network.cell.mech.probe
    assert cable.g_mu_param is cell.g_mu_param
    assert cable.primary is cell.primary
    assert cable.secondary is cell.secondary
    assert cable.primary is not cable.secondary
    assert cell.DE._SharedGate.m_dW_rng is cable.primary


def _initialization(functional, parameters, state):
    packets = functional.sample_population_initialization_draws(parameters, state)
    defaults = functional.initialization_inputs
    return defaults._replace(
        populations={
            name: values._replace(random_draw=packets[name])
            for name, values in defaults.populations.items()
        },
        netstim_draw=functional.sample_initialization_draw(parameters, state),
    )


def _inputs(functional, state, steps):
    return dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(noise=packet)
            for name, packet in functional.sample_population_noise_tapes(
                state, steps
            ).items()
        },
        netstim=functional.sample_draw_tape(state, steps),
    )


def _slice(inputs, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], inputs)


def _recording():
    return {"trace": dn.func.Recorder(("cable.v", "cell.v", "t"))}


def _module_snapshot(network):
    cache_values, cache_spec = pytree.tree_flatten(_caches(network))
    return {
        "tensors": {
            name: _tensor_snapshot(value)
            for name, value in (
                *network.named_parameters(remove_duplicate=False),
                *network.named_buffers(remove_duplicate=False),
            )
        },
        "cache_spec": cache_spec,
        "cache_values": tuple(
            _tensor_snapshot(value) if torch.is_tensor(value) else copy.deepcopy(value)
            for value in cache_values
        ),
    }


def _caches(network):
    return {
        "network": network._state_cache,
        "populations": {
            name: population._caches for name, population in network.populations.items()
        },
    }


def _assert_module_snapshot(network, snapshot):
    values = dict(
        (
            *network.named_parameters(remove_duplicate=False),
            *network.named_buffers(remove_duplicate=False),
        )
    )
    assert values.keys() == snapshot["tensors"].keys()
    for name, value in values.items():
        _assert_tensor_snapshot(value, snapshot["tensors"][name])
    cache_values, cache_spec = pytree.tree_flatten(_caches(network))
    assert cache_spec == snapshot["cache_spec"]
    for value, saved in zip(cache_values, snapshot["cache_values"], strict=True):
        if torch.is_tensor(value):
            _assert_tensor_snapshot(value, saved)
        else:
            assert value == saved


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize(
    "initialization,cache_kind",
    (("snapshot", None), ("fresh", None), ("fresh", "owned"), ("fresh", "network")),
)
def test_mixed_stochastic_lifecycle_recording_restart_and_native_checkpoint(
    execution, initialization, cache_kind
):
    source = _network(cache_kind=cache_kind)
    native = _network(cache_kind=cache_kind)
    native.restore_dict_from_checkpoint(
        _clone(source.state_dict_for_checkpoint(), detached=True)
    )
    snapshot = _module_snapshot(source)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization=initialization,
        inference=True,
        execution=execution,
        callbacks=_recording(),
        backend="aot_eager",
        fullgraph=True,
    )
    assert len(runner.initial_state["rng"]) == 2
    assert all(
        "rng" not in population["random"]
        for population in runner.initial_state["populations"].values()
    )
    frozen = _clone(runner.functional.extract(), detached=True)
    global_rng = torch.get_rng_state().clone()
    if initialization == "fresh":
        initial = _initialization(
            runner.functional, runner.parameters, runner.initial_state
        )
        prepared = runner.functional.initialize(
            runner.parameters, runner.constants, runner.initial_state, initial
        )
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
        _native_close(prepared.state, runner.functional.extract(native).state)
        inputs = _inputs(runner.functional, prepared.state, 2 * STEPS)
        options = {"initialization": initial}
    else:
        inputs = _inputs(runner.functional, runner.initial_state, 2 * STEPS)
        options = {}
    frozen_inputs = _clone(inputs, detached=True)
    prefix, first = runner(inputs=_slice(inputs, 0, STEPS), **options)
    replay, replay_aux = runner(inputs=_slice(inputs, 0, STEPS), **options)
    _close(prefix, replay)
    _close(first["callbacks"]["trace"], replay_aux["callbacks"]["trace"])
    _close(_inputs(runner.functional, prefix, STEPS), _slice(inputs, STEPS, 2 * STEPS))
    parameters = first.get("parameters", runner.parameters)
    constants = first.get("constants", runner.constants)
    final, second = runner(
        prefix,
        inputs=_slice(inputs, STEPS, 2 * STEPS),
        parameters=parameters,
        constants=constants,
        callback_state=first["callbacks"].state,
    )
    with torch.no_grad():
        expected = {
            name: [population.v.clone()]
            for name, population in native.populations.items()
        }
        for _ in range(2 * STEPS):
            native.step()
            for name, population in native.populations.items():
                expected[name].append(population.v.clone())
    for name, trace in expected.items():
        observed = torch.cat(
            (
                first["callbacks"]["trace"][f"{name}.v"],
                second["callbacks"]["trace"][f"{name}.v"],
            )
        )
        torch.testing.assert_close(observed, torch.stack(trace))
    _native_close(final, runner.functional.extract(native).state)
    _close(inputs, frozen_inputs)
    _close(runner.functional.extract(), frozen)
    _assert_module_snapshot(source, snapshot)
    _assert_aliases(source)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    runner.functional.commit_state_(native, final)
    restored = copy.deepcopy(native)
    checkpoint = _clone(native.state_dict_for_checkpoint(), detached=True)
    restored.step()
    restored.restore_dict_from_checkpoint(checkpoint)
    _assert_aliases(restored)
    continued, _ = runner(final, parameters=parameters, constants=constants)
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step()
    _native_close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("checkpoint_every", (None, 1))
def test_mixed_stochastic_training_shared_parameter_and_detached_draw_vjps(
    checkpoint_every,
):
    source, native = _network(training=True), _network(training=True)
    native.restore_dict_from_checkpoint(
        _clone(source.state_dict_for_checkpoint(), detached=True)
    )
    snapshot = _module_snapshot(source)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("*g_mu_param", "*eta_mu_param", "*sigma_param"),
        callbacks=_recording(),
        backend="aot_eager",
        fullgraph=True,
        checkpoint_every=checkpoint_every,
    )
    frozen = _clone(runner.functional.extract(), detached=True)
    global_rng = torch.get_rng_state().clone()
    shared = runner.get_parameter(f"populations.cable.{RAW}")
    assert shared is runner.get_parameter(f"populations.cell.{RAW}")
    sigma = runner.get_parameter(
        f"populations.cable.{PREFIX}DE._SharedGate.sigma_param"
    )
    eta = runner.get_parameter("populations.cable.*eta_mu_param")
    initial = _initialization(
        runner.functional, runner.parameters, runner.initial_state
    )
    rand_draw = next(
        value
        for name, value in initial.populations["cable"].random_draw.draws.items()
        if name.endswith(".g.0")
    ).requires_grad_()
    initialized = runner.functional.initialize(
        runner.parameters, runner.constants, runner.initial_state, initial
    )
    inputs = _inputs(runner.functional, initialized.state, STEPS)
    packet = inputs.populations["cable"].noise
    noise = next(
        value for name, value in packet.draws.items() if name.startswith("noise.draws.")
    ).requires_grad_()
    sde = next(
        value for name, value in packet.draws.items() if name.startswith("sde.draws.")
    ).requires_grad_()
    frozen_inputs = _clone(inputs, detached=True)
    final, auxiliary = runner(inputs=inputs, initialization=initial)
    loss = sum(
        auxiliary["callbacks"]["trace"][f"{name}.v"].square().mean()
        for name in ("cable", "cell")
    )
    actual = torch.autograd.grad(
        loss, (shared, sigma, rand_draw, eta, noise, sde), allow_unused=True
    )
    assert all(
        value is not None and value.isfinite().all() and value.abs().sum() > 0
        for value in actual[:3]
    )
    assert all(value is None or value.count_nonzero() == 0 for value in actual[3:])
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    traces = {
        name: [population.v.clone()] for name, population in native.populations.items()
    }
    for _ in range(STEPS):
        native.step()
        for name, population in native.populations.items():
            traces[name].append(population.v.clone())
    native_loss = sum(torch.stack(trace).square().mean() for trace in traces.values())
    expected = torch.autograd.grad(
        native_loss,
        (
            native.cable.mech.probe.g_mu_param,
            native.cable.mech.probe.DE._SharedGate.sigma_param,
        ),
    )
    _close(actual[:2], expected)
    _native_close(final, runner.functional.extract(native).state)
    _close(inputs, frozen_inputs)
    _close(runner.functional.extract(), frozen)
    _assert_module_snapshot(source, snapshot)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
