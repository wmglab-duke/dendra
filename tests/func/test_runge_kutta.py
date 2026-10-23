"""Cable RK stages share native currents, lifecycle phases, and RNG ownership."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn, graded_syn
from tests.func.test_euler import (
    DT,
    STEPS,
    _clone,
    _close,
    _drives,
    _native_step,
)
from tests.func.test_euler import _population as _euler_population
from tests.func.test_euler import (
    _slice,
)
from tests.func.test_network_rollout import _native_driven_step
from tests.func.test_state_sde_execution import _HeunLeak, _SharedLeak

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _population(method, kind="path", *, sde=None, **kwargs):
    population = _euler_population(
        kind,
        integrator=getattr(dn, method)(imem=kwargs.get("imem", True)),
        **kwargs,
    )
    if sde is not None:
        population.insert(sde.rename("sde"))
        population.initialize()
        for state in population.mech.sde.DE.values():
            for offset, name in enumerate(state._sde_rng_names):
                getattr(state, name).reseed(171 + offset)
    return population


@pytest.mark.parametrize("method", ("rk2", "rk4"))
@pytest.mark.parametrize("kind", ("path", "short", "myelinated"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_runge_kutta_scalar_cables_match_native_state_and_accepted_currents(
    method, kind, dtype
):
    source = _population(method, kind, dtype=dtype, batch=(2, 3))
    native = _population(method, kind, dtype=dtype, batch=(2, 3))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    inputs, state = _drives(source), tensors.state
    for index in range(STEPS):
        old_voltage = state["integrator"]["v"].clone()
        step_input = dn.func.StepInput(ve=inputs.ve[index], intra=inputs.intra[index])
        state, _ = functional.step(tensors.parameters, prepared, state, step_input)
        _native_step(native, step_input)
        _close(state, functional.extract(native).state, dtype=dtype)
        _close(state["mechanism_buffers"]["probe"]["voltage"], old_voltage, dtype=dtype)
        _close(
            state["mechanism_buffers"]["probe"]["seen"],
            state["ions"]["na"]["ina"],
            dtype=dtype,
        )
        _close(
            state["mechanism_buffers"]["probe"]["ticks"],
            torch.full_like(old_voltage, index + 1),
            dtype=dtype,
        )
        assert state["integrator"]["v"].dtype == dtype
        assert torch.isfinite(state["integrator"]["i_membrane"]).all()
    _close(tensors, frozen, dtype=dtype)
    _close(functional.extract(source), frozen, dtype=dtype)


@pytest.mark.parametrize("method", ("rk2", "rk4"))
def test_runge_kutta_nonlinear_stage_formula_and_accepted_ion_frame(method):
    source = _population(method, "short")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    initial = tensors.state["integrator"]["v"]
    gain = 1.0 + DT * (initial + 70.0)

    def ionic(voltage):
        return gain * 0.0003 * (voltage + 68.0) + 0.00002 * (voltage + 65.0) ** 2

    def slope(voltage):
        return -(ionic(voltage) + 0.0004 * (voltage + 71.0)) * 1000 / source.cm

    k1 = slope(initial)
    v2 = initial + k1 * DT / 2.0
    k2 = slope(v2)
    if method == "rk2":
        expected_voltage = initial + DT * k2
        accepted_current = ionic(v2)
    else:
        v3 = initial + k2 * DT / 2.0
        k3 = slope(v3)
        v4 = initial + k3 * DT
        k4 = slope(v4)
        expected_voltage = initial + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        accepted_current = (
            ionic(initial) + 2 * ionic(v2) + 2 * ionic(v3) + ionic(v4)
        ) / 6.0
    actual, _ = functional.step(tensors.parameters, prepared, tensors.state)
    _close(actual["integrator"]["v"], expected_voltage)
    _close(actual["ions"]["na"]["ina"], accepted_current)
    _close(actual["mechanism_buffers"]["probe"]["seen"], accepted_current)
    assert not torch.allclose(accepted_current, ionic(initial), rtol=1e-7, atol=0)


@pytest.mark.parametrize("method", ("rk2", "rk4"))
@pytest.mark.parametrize("training", (False, True), ids=("eval", "train"))
def test_runge_kutta_stencil_geometry_and_drive_gradients_match_native(
    method, training
):
    source, native = (
        _population(method, training=training),
        _population(method, training=training),
    )
    raw_weight = source.v.new_tensor([[[0.8, -1.7, 1.1], [1.2, -2.1, 0.7]]])
    for population in (source, native):
        with torch.no_grad():
            population.integrator.ssd.weight.copy_(raw_weight)
        population.integrator.ssd.weight.requires_grad_(True)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    parameters, constants, state = map(
        _clone, (tensors.parameters, tensors.constants, tensors.state)
    )
    names = ["integrator.ssd.weight", "cm_param", "rhoa_param"]
    names.append(next(name for name in parameters if name.endswith(".pas.g_param")))
    targets = [parameters[name].requires_grad_() for name in names]
    targets.extend(constants[name].requires_grad_() for name in ("diam", "dx"))
    targets.append(state["integrator"]["v"].requires_grad_())
    inputs = _drives(source)
    inputs = inputs._replace(
        ve=inputs.ve.requires_grad_(), intra=inputs.intra.requires_grad_()
    )
    targets.extend((inputs.ve, inputs.intra))
    final, _ = functional.prepare_and_rollout(
        parameters, constants, state, inputs, steps=STEPS
    )
    loss = (
        final["integrator"]["v"].square().mean()
        + (1e6 * final["integrator"]["i_membrane"]).square().mean()
    )
    gradients = torch.autograd.grad(loss, targets)
    native_parameters = dict(native.named_parameters())
    native_targets = [native_parameters[name].requires_grad_() for name in names]
    for name in ("diam", "dx"):
        value = getattr(native, name).detach().clone().requires_grad_()
        setattr(native, name, value)
        native_targets.append(value)
    native.v = native.v.detach().clone().requires_grad_()
    native_targets.append(native.v)
    native_inputs = _clone(inputs)
    native_inputs = native_inputs._replace(
        ve=native_inputs.ve.requires_grad_(), intra=native_inputs.intra.requires_grad_()
    )
    native_targets.extend((native_inputs.ve, native_inputs.intra))
    for index in range(STEPS):
        _native_step(
            native,
            dn.func.StepInput(
                ve=native_inputs.ve[index], intra=native_inputs.intra[index]
            ),
        )
    expected_loss = native.v.square().mean() + (1e6 * native.i_membrane).square().mean()
    expected_gradients = torch.autograd.grad(expected_loss, native_targets)
    _close(final, functional.extract(native).state)
    _close(gradients, expected_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
    if training:
        _close(gradients[0], gradients[0].flip(-1))
    else:
        assert not torch.allclose(gradients[0], gradients[0].flip(-1))


@pytest.mark.parametrize("method", ("rk2", "rk4"))
@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("network", (False, True))
def test_runge_kutta_fresh_cache_checkpoint_and_snapshot_restart(
    method, cached, network
):
    def make():
        source = _population(method, imem=False)
        if network:
            source = dn.Network({"axon": source})
            source.train()
            source.initialize(DT)
        if cached:
            for _ in range(2):
                source.step() if network else source.step(DT)
            (source.axon if network else source).cache("_steady_state")
        for name, value in source.named_parameters():
            if name.endswith(".pas.g_param"):
                value.requires_grad_(True)
        return source

    source, native = make(), make()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("*.pas.g_param",),
        backend="aot_eager",
        checkpoint_every=2,
    )
    frozen = _clone(runner.functional.extract())
    actual, auxiliary = runner()
    replay, _ = runner()
    _close(actual, replay)
    native.initialize(DT) if network else native.initialize()
    for _ in range(STEPS):
        native.step() if network else native.step(DT)
    _close(actual, runner.functional.extract(native).state)
    _close(runner.functional.extract(), frozen)
    continued, _ = runner(
        actual, parameters=auxiliary["parameters"], constants=auxiliary["constants"]
    )
    for _ in range(STEPS):
        native.step() if network else native.step(DT)
    _close(continued, runner.functional.extract(native).state)
    # A continued state remains connected through both checkpointed chunks.
    parameters = auxiliary["parameters"]
    population_parameters = parameters["populations"]["axon"] if network else parameters
    conductance = next(
        value
        for name, value in population_parameters.items()
        if name.endswith(".pas.g_param")
    )
    voltage = (
        continued["populations"]["axon"]["integrator"]["v"]
        if network
        else continued["integrator"]["v"]
    )
    gradient = torch.autograd.grad(voltage.sum(), conductance)[0]
    assert torch.isfinite(gradient).all()
    native_population = native.axon if network else native
    native_conductance = next(
        value
        for name, value in native_population.named_parameters()
        if name.endswith(".pas.g_param")
    )
    if cached:
        # Native cache restoration keeps its saved effective mechanism values.
        # Their detached materializations do not depend on the live raw g leaf.
        assert not native_population.v.requires_grad
        _close(gradient, torch.zeros_like(conductance))
    else:
        expected_gradient = torch.autograd.grad(
            native_population.v.sum(), native_conductance
        )[0]
        _close(gradient, expected_gradient)
        assert gradient.abs().sum() > 0
    runner.functional.commit_state_(source, actual)
    checkpoint = source.state_dict_for_checkpoint()
    restored = make()
    restored.restore_dict_from_checkpoint(checkpoint)
    _close(runner.functional.extract(restored).state, actual)


def _network(method, *, stochastic=False, sde=None):
    left = _population(method, imem=False, stochastic=stochastic, sde=sde)
    right = _euler_population(
        imem=False, integrator=dn.bwd_euler_ub(method="pcr", imem=False)
    )
    right.insert(graded_syn.rename("syn"), e=0.0)
    right.insert(expsyn.rename("events"), tau=0.002, e=0.0)
    network = dn.Network({"explicit": left, "implicit": right})
    network.connect_continuous_one_to_one(
        left[:],
        right[:],
        right.mech.syn,
        pre_var="v",
        input="g_pre",
        weight=0.0002,
        delay=DT,
        transform=torch.sigmoid,
    )
    network.connect_one_to_one(
        left[:],
        right[:],
        right.mech.events,
        pre_var="v",
        threshold=0.0,
        weight=0.0001,
        delay=DT,
    )
    network.train()
    network.initialize(DT)
    if sde is not None:
        for state in left.mech.sde.DE.values():
            for offset, name in enumerate(state._sde_rng_names):
                getattr(state, name).reseed(171 + offset)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    next(iter(network.synapses.values())).schedule(
        con_indices=[0, 5], times_ms=[0.0, 2 * DT], weight=1.0
    )
    return network


@pytest.mark.parametrize("method", ("rk2", "rk4"))
@pytest.mark.parametrize("execution", ("eager", "scan", "while"))
def test_runge_kutta_mixed_network_delivery_recording_and_continuation(
    method, execution
):
    source, native = _network(method), _network(method)
    functional, tensors = dn.func.make_functional_network(source)
    inputs = dn.func.NetworkRolloutInput(
        populations={
            name: _drives(population) for name, population in source.populations.items()
        }
    )
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("explicit.v", "implicit.v", "t"))}
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    kernel = (
        None
        if execution == "eager"
        else functional.compile_rollout_chunk(
            STEPS, execution=execution, backend="aot_eager"
        )
    )
    with torch.no_grad() if execution == "while" else nullcontext():
        actual, auxiliary = functional.run(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            inputs,
            steps=STEPS,
            callbacks=callbacks,
            compiled_chunk=kernel,
        )
        first, _ = functional.rollout(
            tensors.parameters, prepared, tensors.state, _slice(inputs, 0, 2), steps=2
        )
        continued, _ = functional.rollout(
            tensors.parameters, prepared, first, _slice(inputs, 2, STEPS), steps=2
        )
        for index in range(STEPS):
            _native_driven_step(
                native,
                {
                    name: dn.func.StepInput(
                        ve=value.ve[index], intra=value.intra[index]
                    )
                    for name, value in inputs.populations.items()
                },
            )
    _close(actual, functional.extract(native).state)
    _close(actual, continued)
    trace = auxiliary["callbacks"]["trace"]
    _close(trace["explicit.v"][-1], native.explicit.v)
    _close(trace["implicit.v"][-1], native.implicit.v)
    assert trace["t"].shape[0] == STEPS + 1
    if execution != "while":
        weight = next(iter(tensors.parameters["continuous_synapses"].values()))[
            "weight.weights.0.param.rho"
        ]
        gradient = torch.autograd.grad(
            actual["populations"]["implicit"]["integrator"]["v"].sum(), weight
        )[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("method", ("rk2", "rk4"))
@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("sde", (_SharedLeak, _HeunLeak), ids=("em", "heun"))
def test_runge_kutta_noise_sde_poisson_rng_checkpoint_continuation(
    method, execution, network, sde
):
    def make():
        return (
            _network(method, stochastic=True, sde=sde)
            if network
            else _population(method, stochastic=True, imem=False, sde=sde)
        )

    source, native = make(), make()
    functional, tensors = (
        dn.func.make_functional_network(source)
        if network
        else dn.func.make_functional(source, dt=DT)
    )
    frozen = _clone(tensors)
    global_rng = torch.get_rng_state().clone()
    if network:
        noise = functional.sample_population_noise_tapes(tensors.state, STEPS)
        stimulation = functional.sample_population_stimulation_tapes(
            tensors.state, STEPS
        )
        inputs = dn.func.NetworkRolloutInput(
            populations={
                name: dn.func.RolloutInput(
                    noise=noise.get(name), stimulation=stimulation.get(name)
                )
                for name in source.populations
            }
        )
    else:
        inputs = dn.func.RolloutInput(
            noise=functional.sample_noise_draw_tape(tensors.state, STEPS),
            stimulation=functional.sample_stimulation_draw_tape(tensors.state, STEPS),
        )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    kernel = functional.compile_rollout_chunk(
        STEPS,
        execution="default" if execution == "while" and not network else execution,
        backend="aot_eager",
    )
    with torch.no_grad() if execution == "while" else nullcontext():
        if network:
            actual, _ = functional.run(
                tensors.parameters,
                tensors.constants,
                tensors.state,
                inputs,
                steps=STEPS,
                compiled_chunk=kernel,
            )
        else:
            actual, _ = dn.func.run(
                kernel.bind(tensors.parameters, prepared),
                tensors.state,
                inputs=inputs,
                steps=STEPS,
            )
        first, _ = functional.rollout(
            tensors.parameters, prepared, tensors.state, _slice(inputs, 0, 2), steps=2
        )
        continued, _ = functional.rollout(
            tensors.parameters, prepared, first, _slice(inputs, 2, STEPS), steps=2
        )
        for _ in range(STEPS):
            native.step() if network else native.step(DT)
    _close(actual, continued)
    assert pytree.tree_structure(actual) == pytree.tree_structure(tensors.state)
    if execution == "while":
        assert kernel.execution_report().strategy == "compiled_while_loop"
    _close(actual, functional.extract(native).state)
    _close(torch.get_rng_state(), global_rng)
    _close(tensors, frozen)
    functional.commit_state_(source, first)
    checkpoint = source.state_dict_for_checkpoint()
    restored = make()
    restored.restore_dict_from_checkpoint(checkpoint)
    for _ in range(2):
        restored.step() if network else restored.step(DT)
    _close(functional.extract(restored).state, actual)
