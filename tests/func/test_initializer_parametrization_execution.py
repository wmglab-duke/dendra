"""Parametrized Ion/Material recipes retain lifecycle and raw-source gradients."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn
from dendra.models.parametric import PositiveParam
from tests.func.test_initialization_source_execution import (
    _ConcentrationWriter,
    _CurrentReader,
    _InitializerLeak,
)
from tests.func.test_network import _clone
from tests.func.test_network_cache_initialization import (
    _cache_initialized_native_packet,
)
from tests.func.test_shared_field_initialization import _assert_tree_close

pytestmark = pytest.mark.cpu
DT = 0.125
STEPS = 4
DTYPE = torch.float64


def _close(actual, expected, **tolerance):
    _assert_tree_close(actual, expected, **({"rtol": 2e-8, "atol": 2e-9} | tolerance))


class _Affine(torch.nn.Module):
    def __init__(self, dtype, factor=1.1, offset=0.03):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(factor, dtype=dtype))
        self.register_buffer("offset", torch.tensor(offset, dtype=dtype))

    def forward(self, value):
        return value * self.factor + self.offset


class _Split(torch.nn.Module):
    def forward(self, left, right):
        return left + right

    def right_inverse(self, value):
        return value * 0.4, value * 0.6


class _Initializer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.tensor(140.0, dtype=dtype))

    def forward(self):
        return self.raw


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


def _recipe(kind="positive", dtype=DTYPE, *, small=False):
    if kind == "positive":
        source = PositiveParam(
            torch.tensor(0.5 if small else 140.0, dtype=dtype), requires_grad=True
        )
        name = "rho"
    else:
        source = _Initializer(dtype)
        name = "raw"
    register_parametrization(source, name, _Split())
    register_parametrization(source, name, _Affine(dtype))
    return source


def _population(source, dtype=DTYPE, *, synapse=False):
    population = dn.SingleCompartment(
        N=1,
        C=2,
        v_init=-60.0,
        dtype=dtype,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    population.concentrations(nao0=source)
    population.equilibria(ena=source)
    population.material(
        "pool", fields={"amount": source, "reserve": source}, conserved=False
    )
    # Both prescribed reversal and concentration recipes use the same module.
    population.ion_style("na", 0, 0)
    population.insert(_ConcentrationWriter.rename("writer"))
    population.insert(_InitializerLeak.rename("leak"))
    population.insert(_CurrentReader.rename("reader"))
    if synapse:
        population.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
    population.train()
    with torch.no_grad():
        population.initialize()
    population.detach()
    return population


def _network(*, connected=False):
    source = _recipe(small=connected)
    populations = {
        "left": _population(source, synapse=connected),
        "right": _population(source, synapse=connected),
    }
    stimulus = None
    if connected:
        stimulus = dn.NetStim(
            N=1, interval=0.5, start=0.0, noise=0.0, max_spikes=20, seed=317
        )
        stimulus.interval = source
    network = dn.Network(
        populations,
        netstim=stimulus,
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    if connected:
        network.connect_one_to_one(
            stimulus[:],
            network.left[:, 0],
            network.left.mech.syn,
            weight=source,
            delay=DT,
        )
    network.train()
    network.initialize(DT)
    if connected:
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=False,
            train_delay_backend="dense",
        )
    for population in network.populations.values():
        population.detach()
    return network, source


def _leaves(source):
    return tuple(source.parameters())


def _source_names(parameters):
    return tuple(
        name for name in parameters if ".ions.na.e_init.parametrizations." in name
    )


def _field_loss(state):
    return (
        state["integrator"]["v"].square().sum()
        + 0.01 * state["ions"]["na"]["nao"].square().sum()
        + 0.01 * state["materials"]["pool"]["reserve"].square().sum()
    )


def _native_field_loss(population):
    return (
        population.v.square().sum()
        + 0.01 * population.mech.ions["na"].nao.square().sum()
        + 0.01 * population.mech.materials["pool"].reserve.square().sum()
    )


@pytest.mark.parametrize("kind", ("positive", "custom"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("check_purity", (False, True))
def test_fresh_initializer_chains_keep_raw_sources_dtype_and_native_vjp(
    kind, dtype, check_purity
):
    source = _recipe(kind, dtype)
    population = _population(source, dtype)
    native = copy.deepcopy(population)
    original = _clone(population.state_dict(), detached=True)
    rng = torch.random.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional(population, dt=DT)
    names = _source_names(tensors.parameters)
    assert len(names) == 3
    assert {name.rsplit(".", 1)[-1] for name in names} == {
        "original0",
        "original1",
        "factor",
    }
    assert all(tensors.parameters[name].is_leaf for name in names)
    assert any(name.endswith(".1.offset") for name in tensors.constants)
    frozen = _clone(tensors, detached=True)

    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    actual, _ = functional.prepare_and_rollout(
        initialized.parameters, initialized.constants, initialized.state, steps=STEPS
    )
    native.initialize()
    for _ in range(STEPS):
        native.step(DT)
    tolerance = {"rtol": 3e-5, "atol": 3e-5} if dtype == torch.float32 else {}
    _close(actual, functional.extract(native).state, **tolerance)
    assert actual["integrator"]["v"].dtype == dtype
    for field in ("nao", "ena"):
        assert actual["ions"]["na"][field].dtype == dtype
    expected = torch.autograd.grad(
        _native_field_loss(native), _leaves(native.mech.ions["na"].e_init)
    )
    observed = torch.autograd.grad(
        _field_loss(actual), tuple(tensors.parameters[name] for name in names)
    )
    _close(observed, expected, **tolerance)
    assert all(value.abs().sum() > 0 for value in observed)
    _close(population.state_dict(), original)
    _close(tensors, frozen)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("execution", ("scan", "while"))
@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
def test_network_shared_initializer_connection_and_netstim_sources_continue(
    execution, initialization
):
    source, recipe = _network(connected=True)
    native, _ = _network(connected=True)
    native.netstim.restore_dict_from_checkpoint(
        _clone(source.netstim.state_dict_for_checkpoint(), detached=True)
    )
    with torch.no_grad():
        source.step()
        native.step()
    options = {"checkpoint_every": 1} if execution == "scan" else {}
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization=initialization,
        execution=execution,
        inference=execution == "while",
        backend="aot_eager",
        trainable=("*ions.na.e_init.parametrizations.*",),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        **options,
    )
    names = _source_names(runner.parameters["populations"]["left"])
    assert len(names) == 3
    for name in names:
        leaf = runner.parameters["populations"]["left"][name]
        assert leaf is runner.parameters["populations"]["right"][name]
        suffix = name.split("e_init.", 1)[1]
        assert leaf is runner.parameters["netstim"]["parameters"][f"interval.{suffix}"]
        assert any(
            leaf is value
            for connection in runner.parameters["synapses"].values()
            for value in connection.values()
        )
        assert all(leaf is not value for value in _leaves(recipe))
    frozen = _clone(runner.initial_state, detached=True)
    prefix, first = runner()
    repeated, _ = runner()
    _close(prefix, repeated)
    final, second = runner(
        prefix,
        parameters=first.get("parameters", runner.parameters),
        constants=first.get("constants", runner.constants),
        callback_state=first["callbacks"].state,
    )
    if initialization == "fresh":
        native.initialize(DT)
    for _ in range(2 * STEPS):
        native.step()
    _close(final, runner.functional.extract(native).state)
    assert (
        torch.cat(
            (
                first["callbacks"]["trace"]["left.v"],
                second["callbacks"]["trace"]["left.v"],
            )
        ).shape[0]
        == 2 * STEPS + 1
    )
    if execution == "scan":
        observed = torch.autograd.grad(
            sum(_field_loss(state) for state in final["populations"].values()),
            tuple(runner.parameters["populations"]["left"][name] for name in names),
        )
        expected = torch.autograd.grad(
            sum(
                _native_field_loss(population)
                for population in native.populations.values()
            ),
            _leaves(native.left.mech.ions["na"].e_init),
        )
        _close(observed, expected)
        assert all(value.abs().sum() > 0 for value in observed)
    else:
        assert runner.execution_report().strategy == "compiled_while_loop"
    _close(runner.initial_state, frozen)

    checkpoint = _clone(native.state_dict_for_checkpoint(), detached=True)
    resumed, _ = _network(connected=True)
    resumed.restore_dict_from_checkpoint(checkpoint)
    continued, _ = runner.functional.prepare_and_step(
        second.get("parameters", runner.parameters),
        second.get("constants", runner.constants),
        final,
        netstim_draw=runner.functional.sample_step_draw(final),
    )
    resumed.step()
    _close(continued, runner.functional.extract(resumed).state)


@pytest.mark.parametrize("cache_kind", ("population", "network"))
def test_initializer_cache_restores_raw_originals_and_transform_configuration(
    cache_kind,
):
    source, recipe = _network()
    with torch.no_grad():
        for _ in range(STEPS):
            source.step()
        if cache_kind == "population":
            source.left.cache("_steady_state")
        else:
            source.cache_state()
        for leaf in _leaves(recipe):
            leaf.add_(0.1)
        recipe.parametrizations.rho[1].offset.add_(0.2)
    # Keep saved initialization inputs separate from the lowered live snapshot.
    saved_network = source._state_cache
    saved_population = source.left._caches.pop("_steady_state", None)
    source._state_cache = {}
    with torch.no_grad():
        source.initialize(DT)
    source._state_cache = saved_network
    if saved_population is not None:
        source.left._caches["_steady_state"] = saved_population
    native = copy.deepcopy(source)
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors, detached=True)
    initialize = torch.compile(
        functional.initialize, backend="aot_eager", fullgraph=True
    )
    initialized = initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT)
    _close(initialized, _cache_initialized_native_packet(functional, native))
    names = _source_names(tensors.parameters["populations"]["left"])
    for name in names:
        value = initialized.parameters["populations"]["left"][name]
        assert value is initialized.parameters["populations"]["right"][name]
        upstream = tensors.parameters["populations"]["left"][name]
        derivative = torch.autograd.grad(value.sum(), upstream, retain_graph=True)[0]
        torch.testing.assert_close(derivative, torch.ones_like(upstream))
    config = next(
        name
        for name in tensors.constants["populations"]["left"]
        if "e_init" in name and name.endswith(".1.offset")
    )
    assert (
        initialized.constants["populations"]["left"][config]
        is initialized.constants["populations"]["right"][config]
    )
    actual, _ = functional.prepare_and_rollout(*initialized, steps=STEPS)
    for _ in range(STEPS):
        native.step()
    _close(actual, functional.extract(native).state)
    observed = torch.autograd.grad(
        sum(_field_loss(state) for state in actual["populations"].values()),
        tuple(tensors.parameters["populations"]["left"][name] for name in names),
        allow_unused=True,
    )
    native_loss = sum(
        _native_field_loss(population) for population in native.populations.values()
    )
    expected = (
        torch.autograd.grad(
            native_loss,
            _leaves(native.left.mech.ions["na"].e_init),
            allow_unused=True,
        )
        if native_loss.requires_grad
        else (None,) * len(names)
    )
    for value, oracle in zip(observed, expected, strict=True):
        if oracle is None:
            assert value is None or not torch.count_nonzero(value)
        else:
            torch.testing.assert_close(value, oracle)
    _close(tensors, frozen)


def test_fresh_parametrized_initializer_supports_function_transforms_and_fullgraph():
    population = _population(_recipe())
    functional, tensors = dn.func.make_functional(population, dt=DT)
    names = _source_names(tensors.parameters)
    raw = tuple(tensors.parameters[name].detach() for name in names)

    def evaluate(*values):
        parameters = {**tensors.parameters, **dict(zip(names, values, strict=True))}
        initialized = functional.initialize(
            parameters, tensors.constants, tensors.initialization
        )
        return torch.cat(
            (
                initialized.state["ions"]["na"]["nao"].flatten(),
                initialized.state["ions"]["na"]["ena"].flatten(),
                initialized.state["materials"]["pool"]["amount"].flatten(),
            )
        )

    arguments = tuple(range(len(raw)))
    reverse = torch.func.jacrev(evaluate, argnums=arguments)(*raw)
    forward = torch.func.jacfwd(evaluate, argnums=arguments)(*raw)
    _close(reverse, forward)
    assert all(value.abs().sum() > 0 for value in reverse)
    _, tangent = torch.func.jvp(
        evaluate, raw, tuple(torch.ones_like(value) for value in raw)
    )
    torch.testing.assert_close(tangent, sum(reverse))
    batched = tuple(torch.stack((value - 0.01, value + 0.01)) for value in raw)
    torch.testing.assert_close(
        torch.vmap(evaluate)(*batched),
        torch.stack(
            tuple(evaluate(*(value[lane] for value in batched)) for lane in range(2))
        ),
    )
    compiled = torch.compile(
        torch.func.jacrev(evaluate, argnums=arguments),
        backend="aot_eager",
        fullgraph=True,
    )
    _close(compiled(*raw), reverse)


@pytest.mark.parametrize("check_purity", (False, True))
def test_packed_population_initializers_keep_parametrized_source_aliases_and_vjp(
    check_purity,
):
    recipe = _recipe()
    population = dn.concat_models(
        {"left": _population(recipe), "right": _population(recipe)}
    )
    population.train()
    with torch.no_grad():
        population.initialize()
    population.detach()
    native = copy.deepcopy(population)
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional(population, dt=DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    actual, _ = functional.prepare_and_rollout(
        initialized.parameters, initialized.constants, initialized.state, steps=STEPS
    )
    native.initialize()
    for _ in range(STEPS):
        native.step(DT)
    _close(actual, functional.extract(native).state)
    names = tuple(
        name
        for name in tensors.parameters
        if ".ions.na.e_init." in name and ".parametrizations." in name
    )
    assert len(names) == 3
    observed = torch.autograd.grad(
        _field_loss(actual), tuple(tensors.parameters[name] for name in names)
    )
    expected = torch.autograd.grad(
        _native_field_loss(native), _leaves(native.mech.ions["na"].e_init)
    )
    _close(observed, expected)
    assert all(value.abs().sum() > 0 for value in observed)


def test_inductor_fresh_parametrized_initializer_keeps_native_training_gradient():
    source, _ = _network()
    native, _ = _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        backend="inductor",
        trainable=("*ions.na.e_init.parametrizations.*",),
    )
    actual, auxiliary = runner()
    native.initialize(DT)
    for _ in range(STEPS):
        native.step()
    _close(actual, runner.functional.extract(native).state)
    names = _source_names(runner.parameters["populations"]["left"])
    observed = torch.autograd.grad(
        sum(_field_loss(state) for state in actual["populations"].values()),
        tuple(runner.parameters["populations"]["left"][name] for name in names),
    )
    expected = torch.autograd.grad(
        sum(
            _native_field_loss(population) for population in native.populations.values()
        ),
        _leaves(native.left.mech.ions["na"].e_init),
    )
    _close(observed, expected)
    assert all(value.abs().sum() > 0 for value in observed)
    assert runner.execution_report().strategy == "compiled_scan"
    for name in names:
        assert (
            auxiliary["parameters"]["populations"]["left"][name]
            is runner.parameters["populations"]["left"][name]
        )
