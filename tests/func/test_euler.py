"""Single-stage explicit cable solves share native phases and tensor ownership."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import expsyn, graded_syn, pas
from dendra.units import nA
from tests.func.test_network_rollout import _native_driven_step

DT = 0.0001
STEPS = 4
pytestmark = pytest.mark.cpu


class _CurvedCurrent(Mechanism):
    Mechanism.RANGE(g=0.0003, e=-68.0, curvature=0.00002)
    Mechanism.CARRY("gain")
    Mechanism.USEION("na", write=["ina"])

    def initial_values(self, v, values):
        del values
        return {"gain": torch.ones_like(v)}

    def advance(self, v, dt, values):
        return {"gain": values["gain"] + dt * (v + 70.0)}

    def ina(self, v):
        return self.gain * self.g * (v - self.e) + self.curvature * (v + 65.0) ** 2

    def ina_with_conductance(self, v):
        return self.ina(v), self.gain * self.g + 2 * self.curvature * (v + 65.0)


class _AcceptedCurrent(Mechanism):
    Mechanism.CARRY("seen", "voltage", "ticks")
    Mechanism.USEION("na", read=["ina"])

    def initial_values(self, v, values):
        del values
        return {name: torch.zeros_like(v) for name in ("seen", "voltage", "ticks")}

    def advance(self, v, dt, values):
        del dt
        return {"seen": values["ina"], "voltage": v, "ticks": values["ticks"] + 1}


class _NoisyLeak(Mechanism):
    Mechanism.RANGE(g=0.0003, e=-68.0)
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00003, seed=31)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.eta


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


def _clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def _close(actual, expected, *, dtype=torch.float64):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=3e-5 if dtype == torch.float32 else 2e-10,
        atol=2e-6 if dtype == torch.float32 else 2e-11,
    )


def _population(kind="path", **kwargs):
    with dn.ctx(IMEM=kwargs.get("imem", True)):
        return _make_population(kind, **kwargs)


def _make_population(
    kind="path",
    *,
    dtype=torch.float64,
    batch=(),
    imem=True,
    alias="euler",
    training=True,
    stochastic=False,
    integrator=None,
):
    method = {"euler": dn.euler, "rk1": dn.rk1, "eulerv1": dn.eulerv1}[alias]
    integrator = method(imem=imem) if integrator is None else integrator
    common = {"dtype": dtype, "v_init": -63.0, "integrator": integrator}
    if kind == "myelinated":
        population = dn.Myelinated([8.0, 12.0], n_node=3, **common)
    else:
        population = dn.Unmyelinated(
            [2.0, 2.5], L=0.0 if kind == "short" else 30.0, dx=10.0, **common
        )
    population.insert(pas, g=0.0004, e=-71.0)
    if stochastic:
        population.insert(_NoisyLeak.rename("noise"))
        waveform = dn.mono_rect(amp=0.2 * nA, pw=0.00015).poisson(
            interval=0.00017,
            n=5,
            start=-0.0001,
            noise=0.7,
            randomize_every_call=True,
            generator=torch.Generator().manual_seed(51),
        )
        population[:, 1].inject(waveform)
    else:
        population.insert(_CurvedCurrent.rename("curved"))
        population.insert(_AcceptedCurrent.rename("probe"))
    for size in batch:
        population.batch(size)
    population.train(training)
    population.initialize()
    population.v = torch.linspace(
        -66.0, -59.0, population.v.numel(), dtype=dtype
    ).reshape(population.shape)
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def _drives(population, steps=STEPS):
    count = steps * population.v.numel()
    shape = (steps, *population.shape)
    return dn.func.RolloutInput(
        ve=torch.linspace(-0.3, 0.4, count, dtype=population.dtype()).reshape(shape),
        intra=torch.linspace(-2e-9, 3e-9, count, dtype=population.dtype()).reshape(
            shape
        ),
    )


def _native_step(population, inputs):
    dt = population.t.new_tensor(DT)
    population.integrator._initialize(population, dt, force=True)
    population.integrator.step(population, dt, inputs.ve, inputs.intra)
    population.t = population.t + dt


def _slice(inputs, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], inputs)


@pytest.mark.parametrize("kind", ("path", "short", "myelinated"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("alias", ("euler", "rk1", "eulerv1"))
def test_euler_scalar_cables_match_native_all_state_and_accepted_current(
    kind, dtype, alias
):
    source = _population(kind, dtype=dtype, alias=alias, batch=(2, 3))
    native = _population(kind, dtype=dtype, alias=alias, batch=(2, 3))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    inputs = _drives(source)
    state = tensors.state
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
        assert state["integrator"]["v"].dtype == dtype
        assert torch.isfinite(state["integrator"]["i_membrane"]).all()
    _close(tensors, frozen, dtype=dtype)
    _close(functional.extract(source), frozen, dtype=dtype)


@pytest.mark.parametrize("training", (False, True), ids=("eval", "train"))
def test_euler_raw_stencil_geometry_and_drive_gradients_match_native(training):
    source = _population(training=training)
    native = _population(training=training)
    raw_weight = torch.tensor(
        [[[0.8, -1.7, 1.1], [1.2, -2.1, 0.7]]], dtype=source.dtype()
    )
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
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    if training:
        _close(gradients[0], gradients[0].flip(-1))
    else:
        assert not torch.allclose(gradients[0], gradients[0].flip(-1))


@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("network", (False, True))
def test_euler_fresh_initialization_cache_checkpoint_and_snapshot_restart(
    cached, network
):
    def make():
        source = _population(imem=False)
        if network:
            source = dn.Network({"axon": source})
            source.train()
            source.initialize(DT)
        if cached:
            for _ in range(2):
                source.step() if network else source.step(DT)
            (source.axon if network else source).cache("_steady_state")
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
    runner.functional.commit_state_(source, actual)
    checkpoint = source.state_dict_for_checkpoint()
    restored = make()
    restored.restore_dict_from_checkpoint(checkpoint)
    _close(runner.functional.extract(restored).state, actual)


def _network(*, stochastic=False):
    left = _population(imem=False, stochastic=stochastic)
    right = _population(
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
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    next(iter(network.synapses.values())).schedule(
        con_indices=[0, 5],
        times_ms=[0.0, 2 * DT],
        weight=1.0,
    )
    return network


@pytest.mark.parametrize("execution", ("eager", "scan", "while"))
def test_mixed_euler_backward_euler_network_native_order_recording_and_continuation(
    execution,
):
    source, native = _network(), _network()
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
    context = torch.no_grad() if execution == "while" else nullcontext()
    with context:
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


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("network", (False, True))
def test_euler_explicit_noise_poisson_tapes_rng_checkpoint_continuation(
    execution, network
):
    source = (
        _network(stochastic=True)
        if network
        else _population(stochastic=True, imem=False)
    )
    native = (
        _network(stochastic=True)
        if network
        else _population(stochastic=True, imem=False)
    )
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
    context = torch.no_grad() if execution == "while" else nullcontext()
    with context:
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
    restored = (
        _network(stochastic=True)
        if network
        else _population(stochastic=True, imem=False)
    )
    restored.restore_dict_from_checkpoint(checkpoint)
    for _ in range(2):
        restored.step() if network else restored.step(DT)
    _close(functional.extract(restored).state, actual)
