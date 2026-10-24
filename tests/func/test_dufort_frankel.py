"""Two-level cable solves preserve native histories, lifecycle phases, and RNG."""

from __future__ import annotations

import copy
from contextlib import nullcontext

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_euler import DT, STEPS, _clone, _close, _drives
from tests.func.test_euler import _population as _euler_population
from tests.func.test_euler import _slice
from tests.func.test_runge_kutta import _network as _stochastic_network
from tests.func.test_runge_kutta import _population as _stochastic_population
from tests.func.test_runge_kutta import (
    test_runge_kutta_fresh_cache_checkpoint_and_snapshot_restart as _assert_fresh_lifecycle,
)
from tests.func.test_runge_kutta import (
    test_runge_kutta_mixed_network_delivery_recording_and_continuation as _assert_network_lifecycle,
)
from tests.func.test_state_sde_execution import _HeunLeak, _SharedLeak

pytestmark = pytest.mark.cpu
VARIANTS = ("df", "dfh", "dfh-conv")


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


def _population(method, kind="path", *, beta=1.0, **kwargs):
    options = {"beta": beta, "imem": kwargs.get("imem", True)}
    if method == "dfh-conv":
        options["conv"] = True
        method = "dfh"
    return _euler_population(kind, integrator=getattr(dn, method)(**options), **kwargs)


def _native_step(population, inputs=None, *, dt=DT):
    dt_tensor = population.t.new_tensor(dt)
    population.integrator._initialize(population, dt_tensor, force=True)
    population.integrator.step(
        population,
        dt_tensor,
        None if inputs is None else inputs.ve,
        None if inputs is None else inputs.intra,
    )
    population.t = population.t + dt_tensor


@pytest.mark.parametrize("method", VARIANTS)
@pytest.mark.parametrize(
    "kind,beta", (("path", 1.0), ("short", 0.7), ("myelinated", 0.7))
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_dufort_scalar_cables_match_native_histories_and_accepted_currents(
    method, kind, beta, dtype
):
    source = _population(method, kind, beta=beta, dtype=dtype, batch=(2, 3))
    native = _population(method, kind, beta=beta, dtype=dtype, batch=(2, 3))
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
        _close(state["integrator"]["v_prev"], old_voltage, dtype=dtype)
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
        assert state["integrator"]["_df_history_valid"].dtype == torch.bool
        assert state["integrator"]["_df_history_valid"].all()
        assert torch.isfinite(state["integrator"]["i_membrane"]).all()
    _close(tensors, frozen, dtype=dtype)
    _close(functional.extract(source), frozen, dtype=dtype)


@pytest.mark.parametrize("method", VARIANTS)
def test_dufort_geometry_smoothing_stencil_and_drive_gradients_match_native(method):
    source, native = (_population(method, beta=0.6), _population(method, beta=0.6))
    names = ["cm_param", "rhoa_param"]
    names.append(
        next(
            name
            for name, _ in source.named_parameters()
            if name.endswith(".pas.g_param")
        )
    )
    names.append("integrator.filter.weight")
    if method == "dfh-conv":
        names.append("integrator.ssd.weight")
        weight = source.v.new_tensor(
            [[[0.8, 0.1, 1.1], [0.1, -1.2, 0.2], [1.2, -2.1, 0.7]]]
        )
        for population in (source, native):
            with torch.no_grad():
                population.integrator.ssd.weight.copy_(weight)
    for population in (source, native):
        for name, parameter in population.named_parameters():
            if name in names:
                parameter.requires_grad_()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    parameters, constants, state = map(
        _clone, (tensors.parameters, tensors.constants, tensors.state)
    )
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
    native_targets = [native_parameters[name] for name in names]
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


@pytest.mark.parametrize("method", VARIANTS)
def test_dufort_sealed_affine_compartment_matches_independent_two_level_formula(method):
    factory = dn.df() if method == "df" else dn.dfh(conv=method == "dfh-conv")
    with dn.ctx(IMEM=False):
        source = dn.Unmyelinated(
            [2.0], L=0.0, dx=10.0, cm=1.0, v_init=-65.0, integrator=factory
        )
    source.insert(pas, g=0.001, e=-70.0)
    source.initialize(DT)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    previous = tensors.state["integrator"]["v"]
    rate = DT * 1000.0 * source.mech.pas.g / source.cm
    first, _ = functional.step(tensors.parameters, prepared, tensors.state)
    expected_first = previous - rate * (previous + 70.0)
    _close(first["integrator"]["v"], expected_first)
    second, _ = functional.step(tensors.parameters, prepared, first)
    expected_second = (previous - 2.0 * rate * (0.5 * previous + 70.0)) / (1.0 + rate)
    _close(second["integrator"]["v"], expected_second)
    _close(second["integrator"]["v_prev"], expected_first)


@pytest.mark.parametrize("method", VARIANTS)
def test_dufort_nan_history_clock_uses_single_dt_euler_starter(method):
    source = _population(method, beta=0.7)
    source._df_history_dt.fill_(torch.nan)
    source._df_history_valid.fill_(False)
    source.v_prev = torch.full_like(source.v_prev, 500.0)
    native = copy.deepcopy(source)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert tensors.state["integrator"]["_df_history_dt"].isnan().all()
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    actual, _ = functional.step(tensors.parameters, prepared, tensors.state)
    _native_step(native)
    _close(actual, functional.extract(native).state)
    _close(actual["integrator"]["v_prev"], source.v)
    assert actual["integrator"]["_df_history_valid"].all()
    assert source._df_history_dt.isnan().all()
    assert not source._df_history_valid.any()


@pytest.mark.parametrize("method", VARIANTS)
def test_dufort_same_dt_reprepare_and_changed_dt_restart_match_native(method):
    source = _population(method, beta=0.7)
    for _ in range(2):
        _native_step(source)
    old_voltage, old_history = source.v.clone(), source.v_prev.clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    repeated = functional.prepare(tensors.parameters, tensors.constants)
    first, _ = functional.step(tensors.parameters, prepared, tensors.state)
    second, _ = functional.step(tensors.parameters, repeated, tensors.state)
    _close(first, second)
    _close(tensors.state["integrator"]["v"], old_voltage)
    _close(tensors.state["integrator"]["v_prev"], old_history)
    changed, changed_tensors = dn.func.make_functional(source, dt=2 * DT)
    changed_prepared = changed.prepare(
        changed_tensors.parameters, changed_tensors.constants
    )
    actual, _ = changed.step(
        changed_tensors.parameters, changed_prepared, changed_tensors.state
    )
    native = copy.deepcopy(source)
    _native_step(native, dt=2 * DT)
    _close(actual, changed.extract(native).state)
    _close(actual["integrator"]["v_prev"], old_voltage)
    _close(actual["integrator"]["_df_history_dt"], old_voltage.new_tensor(2 * DT))


@pytest.mark.parametrize("method", ("df", "dfh-conv"))
@pytest.mark.parametrize("network", (False, True))
def test_dufort_pure_initial_state_matches_native_before_first_step(method, network):
    def make():
        population = _population(method, imem=False)
        if not network:
            return population
        result = dn.Network({"axon": population})
        result.train()
        result.initialize(DT)
        return result

    source, native = make(), make()
    functional, tensors = (
        dn.func.make_functional_network(source)
        if network
        else dn.func.make_functional(source, dt=DT)
    )
    frozen = _clone(tensors)
    initialized = (
        functional.initialize(*tensors, functional.initialization_inputs)
        if network
        else functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
    )
    native.initialize(DT) if network else native.initialize()
    torch.testing.assert_close(
        initialized.state,
        functional.extract(native).state,
        rtol=2e-10,
        atol=2e-11,
        equal_nan=True,
    )
    state = initialized.state["populations"]["axon"] if network else initialized.state
    history = state["integrator"]
    _close(history["v_prev"], history["v"])
    assert not history["_df_history_valid"].any()
    if network:
        _close(history["_df_history_dt"], history["v"].new_tensor(DT))
    else:
        assert history["_df_history_dt"].isnan().all()
    _close(functional.extract(source), frozen)


@pytest.mark.parametrize(
    "method,cached,network",
    (
        ("df", False, False),
        ("dfh", True, False),
        ("dfh", False, True),
        ("df", True, True),
    ),
)
def test_dufort_fresh_cache_checkpoint_and_snapshot_restart(method, cached, network):
    _assert_fresh_lifecycle(method, cached, network)


@pytest.mark.parametrize(
    "method,execution", (("df", "eager"), ("dfh", "scan"), ("df", "while"))
)
def test_dufort_mixed_network_delivery_recording_and_continuation(method, execution):
    _assert_network_lifecycle(method, execution)


def _assert_stochastic_lifecycle(method, execution, network, sde):
    def make():
        return (
            _stochastic_network(method, stochastic=True, sde=sde)
            if network
            else _stochastic_population(method, stochastic=True, imem=False, sde=sde)
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
    torch.testing.assert_close(tensors, frozen, equal_nan=True)
    functional.commit_state_(source, first)
    checkpoint = source.state_dict_for_checkpoint()
    restored = make()
    restored.restore_dict_from_checkpoint(checkpoint)
    for _ in range(2):
        restored.step() if network else restored.step(DT)
    _close(functional.extract(restored).state, actual)


@pytest.mark.parametrize(
    "method,execution,network,sde",
    (
        ("df", "scan", False, _SharedLeak),
        ("dfh", "while", False, _HeunLeak),
        ("dfh", "scan", True, _HeunLeak),
        ("df", "while", True, _SharedLeak),
    ),
)
def test_dufort_noise_sde_poisson_rng_checkpoint_continuation(
    method, execution, network, sde
):
    _assert_stochastic_lifecycle(method, execution, network, sde)


@pytest.mark.parametrize("method", VARIANTS)
@pytest.mark.parametrize("changed_dt", [False, True])
def test_dufort_owned_cache_preserves_history_until_solver_configuration(
    method, changed_dt
):
    source = _population(method, beta=0.7)
    for _ in range(2):
        _native_step(source)
    source.cache("_steady_state")
    native = copy.deepcopy(source)
    dt = DT * (2 if changed_dt else 1)
    functional, tensors = dn.func.make_functional(source, dt=dt)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native.initialize()
    # Population-owned cache restoration returns before solver configuration.
    # The cached two-level clock is retained until the first accepted step.
    _close(initialized.state, functional.extract(native).state)
    assert initialized.state["integrator"]["_df_history_valid"].all()
    _close(initialized.state["integrator"]["_df_history_dt"], source._df_history_dt)
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    _native_step(native, dt=dt)
    _close(actual, functional.extract(native).state)
    _close(actual["integrator"]["_df_history_dt"], source.v.new_tensor(dt))
