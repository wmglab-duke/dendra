"""Point containers retain native spatial stages, lifecycle, and explicit RNG."""

from __future__ import annotations

import copy
from contextlib import nullcontext

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn, graded_syn, pas
from dendra.units import nA
from tests.func.test_euler import _AcceptedCurrent, _CurvedCurrent
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_network import _RAW_WEIGHT
from tests.func.test_network_cache_initialization import _warm_cache
from tests.func.test_state_sde_execution import _HeunLeak, _SharedLeak
from tests.func.test_stimulation import (
    _assert_module_unchanged as _assert_waveform_snapshot,
)
from tests.func.test_stimulation import _module_snapshot as _waveform_snapshot

DT = 0.0001
STEPS = 4
GAIN = "integrator.mech.mechanisms.pas.g_param"
METHODS = ("euler", "rk1", "eulerv1", "rk2", "rk4", "df", "dfh", "dfh-conv")
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=torch.float64, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected, *, dtype=torch.float64):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=3e-5 if dtype == torch.float32 else 3e-10,
        atol=3e-6 if dtype == torch.float32 else 3e-11,
        equal_nan=True,
    )


def _release(voltage):
    return torch.sigmoid(0.03 * (voltage + 65.0))


def _factory(method):
    if method == "dfh-conv":
        return dn.dfh(conv=True, beta=0.8, imem=True)
    return getattr(dn, method)(
        **({"beta": 0.8} if method.startswith("df") else {}), imem=True
    )


def _population(
    method,
    container=dn.Population,
    *,
    count=3,
    batch=None,
    dtype=torch.float64,
    shared=None,
    stochastic=False,
    simple=False,
):
    population = container(
        N=1 if simple else 2,
        C=count,
        v_init=torch.linspace(-66.0, -60.0, count, dtype=dtype),
        dtype=dtype,
        integrator=_factory(method),
    )
    population.insert(pas, g=0.0004 if shared is None else shared, e=-71.0)
    if not simple:
        population.insert(_CurvedCurrent.rename("curved"))
        population.insert(_AcceptedCurrent.rename("probe"))
    if stochastic:
        population.insert(_SharedLeak.rename("sde"))
        population.insert(_HeunLeak.rename("heun"))
    if batch is not None:
        population.batch(batch)
    population.train()
    population.initialize()
    _seed_noise(population, 171)
    return population


def _seed_noise(population, seed):
    for name in ("sde", "heun"):
        mechanism = getattr(population.mech, name, None)
        if mechanism is None:
            continue
        for state in mechanism.DE.values():
            for offset, key in enumerate(state._sde_rng_names):
                getattr(state, key).reseed(seed + offset)


def _native_step(population, inputs=None, *, dt=DT):
    value = population.t.new_tensor(dt)
    population.integrator._initialize(population, value, force=True)
    population.integrator.step(
        population,
        value,
        None if inputs is None else inputs.ve,
        None if inputs is None else inputs.intra,
    )
    population.t = population.t + value


def _network(
    method, *, training=False, cached=False, stochastic=False, simple=False, count=3
):
    shared = torch.nn.Parameter(torch.tensor(0.0004, dtype=torch.float64))
    left = _population(
        method, shared=shared, stochastic=stochastic, simple=simple, count=count
    )
    right = _population(
        method,
        dn.SingleCompartment,
        shared=shared,
        stochastic=stochastic,
        simple=simple,
        count=count,
    )
    right.insert(graded_syn.rename("syn"), e=0.0)
    if not simple:
        right.insert(expsyn.rename("events"), tau=0.002, e=0.0)
    left[:, min(1, count - 1)].inject(dn.sin(amp=0.2 * nA, freq=0.7, phase=0.3))
    network = dn.Network({"left": left, "right": right})
    network.connect_continuous_one_to_one(
        left[:],
        right[:],
        right.mech.syn,
        pre_var="v",
        input="g_pre",
        weight=0.0002,
        delay=DT,
        transform=_release,
    )
    if not simple:
        network.connect_one_to_one(
            left[:],
            right[:],
            right.mech.events,
            pre_var="v",
            threshold=0.0,
            weight=0.0001,
            delay=DT,
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
    if not simple:
        next(iter(network.synapses.values())).schedule(
            con_indices=[0, 5],
            times_ms=[0.0, 2 * DT],
            weight=1.0,
        )
    _seed_noise(left, 171)
    _seed_noise(right, 271)
    if cached:
        _warm_cache(network)
        network.initialize(DT)
    assert left.mech.pas.g_param is right.mech.pas.g_param
    return network


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("container", (dn.Population, dn.SingleCompartment))
@pytest.mark.parametrize("count", (1, 3))
def test_point_steps_preserve_native_state_accepted_currents_and_tensor_ownership(
    method, container, count
):
    dtype = torch.float32 if count == 1 else torch.float64
    source = _population(method, container, count=count, batch=2, dtype=dtype)
    native = _population(method, container, count=count, batch=2, dtype=dtype)
    before = _module_snapshot(source)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state
    for index in range(STEPS):
        old_voltage = state["integrator"]["v"].clone()
        drive = torch.linspace(-2e-9, 3e-9, source.v.numel(), dtype=dtype).reshape(
            source.shape
        )
        ve = torch.linspace(-0.2, 0.3, source.v.numel(), dtype=dtype).reshape(
            source.shape
        )
        step_input = dn.func.StepInput(intra=(index + 1) * drive, ve=ve)
        incoming, frozen_incoming = state, _clone(state)
        state, _ = functional.step(tensors.parameters, prepared, incoming, step_input)
        _close(incoming, frozen_incoming, dtype=dtype)
        _native_step(native, step_input)
        _close(state, functional.extract(native).state, dtype=dtype)
        _close(state["mechanism_buffers"]["probe"]["voltage"], old_voltage, dtype=dtype)
        _close(
            state["mechanism_buffers"]["probe"]["ticks"],
            torch.full_like(old_voltage, index + 1),
            dtype=dtype,
        )
        assert state["integrator"]["v"].dtype == dtype
        if method.startswith("df"):
            _close(state["integrator"]["v_prev"], old_voltage, dtype=dtype)
            assert state["integrator"]["_df_history_valid"].all()
    _close(tensors, frozen, dtype=dtype)
    _assert_module_snapshot(source, before)


@pytest.mark.parametrize(
    "method,execution,initialization,cached,backend",
    [
        ("euler", "scan", "snapshot", False, "aot_eager"),
        ("rk2", "while", "snapshot", False, "aot_eager"),
        ("rk4", "scan", "fresh", False, "aot_eager"),
        ("df", "while", "fresh", False, "aot_eager"),
        ("dfh", "scan", "fresh", True, "aot_eager"),
        ("dfh-conv", "while", "fresh", True, "aot_eager"),
        ("rk4", "scan", "snapshot", False, "inductor"),
        ("dfh", "while", "snapshot", False, "inductor"),
    ],
)
def test_point_network_recording_initialization_restart_and_native_checkpoint(
    method, execution, initialization, cached, backend
):
    def make():
        return _network(
            method,
            cached=cached,
            simple=backend == "inductor",
            count=1 if backend == "inductor" else 3,
        )

    source, native = make(), make()
    before = {name: _module_snapshot(pop) for name, pop in source.populations.items()}
    waveform_before = _waveform_snapshot(source.left.intra.stims[0])
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization=initialization,
        inference=True,
        execution=execution,
        backend=backend,
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
    )
    frozen = _clone(runner.initial_state)
    if initialization == "fresh":
        native.initialize(DT)
    frames = {
        "left.v": [native.left.v.clone()],
        "right.v": [native.right.v.clone()],
        "t": [native.t.clone()],
    }
    with torch.no_grad():
        for _ in range(2 * STEPS):
            native.step()
            for name, value in (
                ("left.v", native.left.v),
                ("right.v", native.right.v),
                ("t", native.t),
            ):
                frames[name].append(value.clone())
    prefix, first = runner()
    assert runner.execution_report().strategy == (
        "compiled_while_loop" if execution == "while" else "compiled_scan"
    )
    repeated, _ = runner()
    _close(prefix, repeated)
    parameters = first.get("parameters", runner.parameters)
    constants = first.get("constants", runner.constants)
    final, second = runner(
        prefix,
        parameters=parameters,
        constants=constants,
        callback_state=first["callbacks"].state,
    )
    _close(final, runner.functional.extract(native).state)
    for name, expected in frames.items():
        actual = torch.cat(
            (first["callbacks"]["trace"][name], second["callbacks"]["trace"][name])
        )
        _close(actual, torch.stack(expected))
    receiver = make()
    runner.functional.commit_state_(receiver, prefix)
    restored = make()
    restored.restore_dict_from_checkpoint(receiver.state_dict_for_checkpoint())
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step()
    _close(final, runner.functional.extract(restored).state)
    _close(runner.initial_state, frozen)
    for name, population in source.populations.items():
        _assert_module_snapshot(population, before[name])
    _assert_waveform_snapshot(source.left.intra.stims[0], waveform_before)


def _network_loss(state):
    return sum(
        population["integrator"]["v"].square().mean()
        for population in state["populations"].values()
    )


@pytest.mark.parametrize("method", ("euler", "rk4", "dfh-conv"))
@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_point_network_shared_physical_stimulus_and_connection_gradients(
    method, backend
):
    source, native = (
        _network(method, training=True, simple=backend == "inductor"),
        _network(method, training=True, simple=backend == "inductor"),
    )
    before = {name: _module_snapshot(pop) for name, pop in source.populations.items()}
    waveform_before = _waveform_snapshot(source.left.intra.stims[0])
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        inference=False,
        backend=backend,
        checkpoint_every=2 if backend == "inductor" else None,
        trainable=(
            "*pas.g_param",
            "*cm_param*",
            "*rhoa_param*",
            "*stimulation.intra*.amp",
            f"continuous_synapses.*.{_RAW_WEIGHT}",
        ),
    )
    actual, _ = runner()
    assert runner.execution_report().strategy == "compiled_scan"
    for _ in range(STEPS):
        native.step()
    _close(actual, runner.functional.extract(native).state)
    connection = next(iter(source.continuous_synapses))
    amplitude = next(
        name
        for name in runner.parameters["populations"]["left"]
        if name.startswith("stimulation.intra.") and name.endswith(".amp")
    )
    names = (
        f"populations.left.{GAIN}",
        "populations.left.cm_param",
        "populations.left.rhoa_param",
        "populations.right.cm_param",
        f"populations.left.{amplitude}",
        f"continuous_synapses.{connection}.{_RAW_WEIGHT}",
    )
    targets = tuple(runner.get_parameter(name) for name in names)
    assert targets[0] is runner.get_parameter(f"populations.right.{GAIN}")
    expected_targets = (
        native.left.mech.pas.g_param,
        native.left.cm_param.rho,
        native.left.rhoa_param.rho,
        native.right.cm_param.rho,
        native.left.intra.stims[0].amp,
        dict(native.continuous_synapses[connection].named_parameters())[_RAW_WEIGHT],
    )
    actual_gradients = torch.autograd.grad(_network_loss(actual), targets)
    expected_gradients = torch.autograd.grad(
        native.left.v.square().mean() + native.right.v.square().mean(), expected_targets
    )
    _close(actual_gradients, expected_gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0
        for value in actual_gradients
    )
    for name, population in source.populations.items():
        _assert_module_snapshot(population, before[name])
    _assert_waveform_snapshot(source.left.intra.stims[0], waveform_before)


@pytest.mark.parametrize("method", ("df", "dfh", "dfh-conv"))
@pytest.mark.parametrize("container", (dn.Population, dn.SingleCompartment))
def test_point_dufort_cache_changed_dt_and_history_restarts_match_native(
    method, container
):
    source = _population(method, container)
    for _ in range(2):
        _native_step(source)
    source.cache("_steady_state")
    native = copy.deepcopy(source)
    functional, tensors = dn.func.make_functional(source, dt=2 * DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native.initialize()
    _close(initialized.state, functional.extract(native).state)
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    _native_step(native, dt=2 * DT)
    _close(actual, functional.extract(native).state)
    _close(actual["integrator"]["_df_history_dt"], source.t.new_tensor(2 * DT))
    _close(actual["integrator"]["v_prev"], initialized.state["integrator"]["v"])
    assert actual["integrator"]["_df_history_valid"].all()


@pytest.mark.parametrize(
    "method,network,execution",
    [("euler", False, "scan"), ("rk4", True, "scan"), ("dfh-conv", True, "while")],
)
def test_point_explicit_noise_sde_tapes_and_rng_checkpoint_suffix(
    method, network, execution
):
    make = (
        (lambda: _network(method, stochastic=True))
        if network
        else (lambda: _population(method, dn.SingleCompartment, stochastic=True))
    )
    source, native = make(), make()
    functional, tensors = (
        dn.func.make_functional_network(source)
        if network
        else dn.func.make_functional(source, dt=DT)
    )
    frozen, global_rng = _clone(tensors), torch.get_rng_state().clone()
    if network:
        tapes = functional.sample_population_noise_tapes(tensors.state, 2 * STEPS)
        inputs = dn.func.NetworkRolloutInput(
            populations={
                name: dn.func.RolloutInput(noise=tape) for name, tape in tapes.items()
            }
        )
    else:
        inputs = dn.func.RolloutInput(
            noise=functional.sample_noise_draw_tape(tensors.state, 2 * STEPS)
        )
    saved_inputs = _clone(inputs)

    def select(start, stop):
        return pytree.tree_map_only(
            torch.Tensor, lambda value: value[start:stop], inputs
        )

    kernel = functional.compile_rollout_chunk(
        STEPS, execution=execution, backend="aot_eager"
    )
    context = torch.no_grad() if execution == "while" else nullcontext()
    with context:
        if network:
            prefix, _ = functional.run(
                tensors.parameters,
                tensors.constants,
                tensors.state,
                select(0, STEPS),
                steps=STEPS,
                compiled_chunk=kernel,
            )
            final, _ = functional.run(
                tensors.parameters,
                tensors.constants,
                prefix,
                select(STEPS, 2 * STEPS),
                steps=STEPS,
                compiled_chunk=kernel,
            )
            suffix = functional.sample_population_noise_tapes(prefix, STEPS)
            _close(
                suffix,
                {
                    name: value.noise
                    for name, value in select(STEPS, 2 * STEPS).populations.items()
                },
            )
        else:
            prepared = functional.prepare(tensors.parameters, tensors.constants)
            bound = kernel.bind(tensors.parameters, prepared)
            prefix, _ = dn.func.run(
                bound, tensors.state, inputs=select(0, STEPS), steps=STEPS
            )
            final, _ = dn.func.run(
                bound, prefix, inputs=select(STEPS, 2 * STEPS), steps=STEPS
            )
            _close(
                functional.sample_noise_draw_tape(prefix, STEPS),
                select(STEPS, 2 * STEPS).noise,
            )
        for _ in range(2 * STEPS):
            native.step() if network else native.step(DT)
    _close(final, functional.extract(native).state)
    _close(torch.get_rng_state(), global_rng)
    functional.commit_state_(source, prefix)
    restored = make()
    restored.restore_dict_from_checkpoint(source.state_dict_for_checkpoint())
    with torch.no_grad():
        for _ in range(STEPS):
            restored.step() if network else restored.step(DT)
    _close(final, functional.extract(restored).state)
    _close(inputs, saved_inputs)
    _close(tensors, frozen)
