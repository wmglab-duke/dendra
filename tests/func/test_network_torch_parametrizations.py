"""Ordinary torch parametrizations retain Network aliases and lifecycle state."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrize import cached, register_parametrization

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import graded_syn, hh
from dendra.units import nA
from examples.scaled_waveform_frequency import FrequencyFromCycles
from tests.func.test_extcell_multi_population import _seed_circuit
from tests.func.test_mechanism_injection_plan import _population as _injected_population
from tests.func.test_membrane_current import _inputs as _membrane_inputs
from tests.func.test_membrane_current import _model as _membrane_model
from tests.func.test_network import DTYPE, _clone, _close
from tests.func.test_network_cache_initialization import (
    _cache_initialized_native_packet,
)
from tests.func.test_population_owned_cache import DT as CACHE_DT
from tests.func.test_population_owned_cache import _step as _native_population_step
from tests.func.test_torch_parametrizations import _population
from tests.test_scaled_waveform_frequency import _cable, _wave

pytestmark = pytest.mark.cpu
DT = 0.025
ORIGINAL = "parametrizations.celsius_param.original"
FACTOR = "parametrizations.celsius_param.0.factor"


class _TemperatureScale(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.register_buffer("offset", factor.detach().new_tensor(0.5))

    def forward(self, value):
        return value * self.factor + self.offset


def _release(value):
    return torch.sigmoid((value + 55.0) * 0.1)


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


def _network(*, cache=None, target="celsius_param", shared_transform=False):
    original = torch.nn.Parameter(torch.tensor(30.0, dtype=DTYPE))
    factor = torch.nn.Parameter(torch.tensor(0.9, dtype=DTYPE))
    shared_recipe = _TemperatureScale(factor) if shared_transform else None
    populations = {}
    transforms = []
    for name, voltage in (("left", -45.0), ("right", -55.0)):
        cell = dn.SingleCompartment(
            N=1,
            C=1,
            celsius=original,
            v_init=voltage,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        cell.insert(hh)
        cell.insert(graded_syn.rename("analog"), e=-10.0)
        transform = shared_recipe if shared_transform else _TemperatureScale(factor)
        transforms.append(transform)
        cell.register_parametrization(target, transform)
        populations[name] = cell
    network = dn.Network(populations)
    network.connect_continuous_one_to_one(
        network.left[:],
        network.right[:],
        network.right.mech.analog,
        pre_var="v",
        input="g_pre",
        transform=_release,
        weight=0.002,
        delay=2 * DT,
    )
    network.train()
    network.initialize(DT)
    if cache is not None:
        if cache == "network":
            network._enable_synapse_state_cache_recording()
        with torch.no_grad():
            for _ in range(3):
                network.step()
            if cache == "network":
                network.cache_state()
            elif cache == "population":
                network.left.cache("_steady_state")
            else:
                raise AssertionError(cache)
            original.add_(4.0)
            factor.add_(0.15)
            for transform in dict.fromkeys(transforms):
                transform.offset.add_(0.25)
        if cache == "network":
            network._disable_synapse_state_cache_recording(release=True)
        # Lower a coherent snapshot with different current raw inputs, while
        # retaining the earlier cache as a separate fresh-initialization input.
        saved_network = network._state_cache
        saved_population = network.left._caches.pop("_steady_state", None)
        network._state_cache = {}
        network.initialize(DT, reinit_weights=False, reinit_delays=False)
        network._state_cache = saved_network
        if saved_population is not None:
            network.left._caches["_steady_state"] = saved_population
    return network


def _targets(packet):
    left = packet["populations"]["left"]
    return left[ORIGINAL], left[FACTOR]


def _native_targets(network):
    transforms = network.left.parametrizations.celsius_param
    return transforms.original, transforms[0].factor


def _loss(state):
    return sum(
        population["integrator"]["v"].square().sum()
        for population in state["populations"].values()
    )


def _callbacks():
    return {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}


def _trainable():
    return tuple(f"populations.*.{name}" for name in (ORIGINAL, FACTOR))


def test_network_shared_originals_and_transform_leaves_sum_native_gradients():
    source, native = _network(), _network()
    functional, tensors = dn.func.make_functional_network(source)
    for name in (ORIGINAL, FACTOR):
        assert (
            tensors.parameters["populations"]["left"][name]
            is tensors.parameters["populations"]["right"][name]
        )
    independent = tensors.independent_parameters(_trainable())
    for name in (ORIGINAL, FACTOR):
        assert (
            independent["populations"]["left"][name]
            is independent["populations"]["right"][name]
        )
    before = _clone((tensors.constants, tensors.state), detached=True)
    actual, _ = functional.prepare_and_rollout(
        independent, tensors.constants, tensors.state, steps=5
    )
    for _ in range(5):
        native.step()
    _close(actual, functional.extract(native).state)
    observed = torch.autograd.grad(_loss(actual), _targets(independent))
    oracle = torch.autograd.grad(
        native.left.v.square().sum() + native.right.v.square().sum(),
        _native_targets(native),
    )
    _close(observed, oracle)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)
    _close((tensors.constants, tensors.state), before)


def test_one_transform_module_shared_across_populations_keeps_fresh_native_gradient():
    source, native = _network(shared_transform=True), _network(shared_transform=True)
    assert (
        source.left.parametrizations.celsius_param[0]
        is source.right.parametrizations.celsius_param[0]
    )
    functional, tensors = dn.func.make_functional_network(source)
    parameters = tensors.independent_parameters(_trainable())
    initialized = functional.initialize(
        parameters, tensors.constants, tensors.state, functional.initialization_inputs
    )
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    for name in (ORIGINAL, FACTOR):
        left = initialized.parameters["populations"]["left"][name]
        assert left is initialized.parameters["populations"]["right"][name]
        assert (
            len(
                {
                    id(value)
                    for key, value in tensors.named_parameters.items()
                    if key.endswith(name)
                }
            )
            == 1
        )
    constants = initialized.constants["populations"]
    key = next(name for name in constants["left"] if name.endswith(".0.offset"))
    assert constants["left"][key] is constants["right"][key]
    prefix, _ = functional.prepare_and_rollout(*initialized, steps=2)
    actual, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, prefix
    )
    for _ in range(3):
        native.step()
    _close(actual, functional.extract(native).state)
    observed = torch.autograd.grad(_loss(actual), _targets(parameters))
    oracle = torch.autograd.grad(
        native.left.v.square().sum() + native.right.v.square().sum(),
        _native_targets(native),
    )
    _close(observed, oracle)
    assert all(value.abs().sum() > 0 for value in observed)


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_parametrized_network_restart_recording_continuation_and_checkpoint(
    initialization, execution
):
    source, native = _network(), _network()
    for network in (source, native):
        with torch.no_grad():
            for _ in range(2):
                network.step()
    options = {"checkpoint_every": 2} if execution == "scan" else {}
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization=initialization,
        execution=execution,
        inference=execution == "while",
        trainable=_trainable(),
        callbacks=_callbacks(),
        backend="aot_eager",
        **options,
    )
    for name in (ORIGINAL, FACTOR):
        assert runner.get_parameter(f"populations.left.{name}") is runner.get_parameter(
            f"populations.right.{name}"
        )
    frozen = _clone(runner.initial_state, detached=True)
    prefix, first = runner()
    repeated, again = runner()
    _close(prefix, repeated)
    _close(first["callbacks"].outputs, again["callbacks"].outputs)
    final, second = runner(
        prefix,
        parameters=first.get("parameters", runner.parameters),
        constants=first.get("constants", runner.constants),
        callback_state=first["callbacks"].state,
    )
    if initialization == "fresh":
        native.initialize(DT, reinit_weights=False, reinit_delays=False)
    frames = {
        name: [population.v.detach().clone()]
        for name, population in native.populations.items()
    }
    for _ in range(8):
        native.step()
        for name, population in native.populations.items():
            frames[name].append(population.v.detach().clone())
    _close(final, runner.functional.extract(native).state)
    for name in native.populations:
        field = f"{name}.v"
        trace = torch.cat(
            (first["callbacks"]["trace"][field], second["callbacks"]["trace"][field])
        )
        torch.testing.assert_close(trace, torch.stack(frames[name]))
        assert trace.shape[0] == 9
    if execution == "scan":
        gradients = torch.autograd.grad(_loss(final), _targets(runner.parameters))
        oracle = torch.autograd.grad(
            native.left.v.square().sum() + native.right.v.square().sum(),
            _native_targets(native),
        )
        _close(gradients, oracle)
        assert all(value.abs().sum() > 0 for value in gradients)
    else:
        assert runner.execution_report().strategy == "compiled_while_loop"
    _close(runner.initial_state, frozen)

    target = _network()
    runner.functional.commit_state_(target, final)
    checkpoint = target.state_dict_for_checkpoint()
    resumed = _network()
    resumed.restore_dict_from_checkpoint(checkpoint)
    for name in (ORIGINAL, FACTOR):
        assert (
            dict(resumed.left.named_parameters())[name]
            is dict(resumed.right.named_parameters())[name]
        )
    continued, _ = runner.functional.prepare_and_step(
        second.get("parameters", runner.parameters),
        second.get("constants", runner.constants),
        final,
    )
    resumed.step()
    _close(continued, runner.functional.extract(resumed).state)


@pytest.mark.parametrize("cache", ("population", "network"))
def test_parametrized_network_cache_restore_preserves_aliases_and_native_buffers(cache):
    source, native = _network(cache=cache), _network(cache=cache)
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors, detached=True)
    initialize = torch.compile(
        functional.initialize, backend="aot_eager", fullgraph=True
    )
    initialized = initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized, _cache_initialized_native_packet(functional, native))
    for name in (ORIGINAL, FACTOR):
        left = initialized.parameters["populations"]["left"][name]
        assert left is initialized.parameters["populations"]["right"][name]
        torch.testing.assert_close(left, dict(native.left.named_parameters())[name])
    # Native cache restoration copies saved values into the existing original
    # and transform Parameters. The explicit replacement keeps the same VJP.
    for name, value in zip(
        (ORIGINAL, FACTOR), _targets(initialized.parameters), strict=True
    ):
        upstream = tensors.parameters["populations"]["left"][name]
        gradient = torch.autograd.grad(value.sum(), upstream, retain_graph=True)[0]
        torch.testing.assert_close(gradient, torch.ones_like(upstream))
    actual, _ = functional.prepare_and_rollout(*initialized, steps=3)
    for _ in range(3):
        native.step()
    _close(actual, functional.extract(native).state)
    observed = torch.autograd.grad(
        _loss(actual), _targets(tensors.parameters), allow_unused=True
    )
    oracle = torch.autograd.grad(
        native.left.v.square().sum() + native.right.v.square().sum(),
        _native_targets(native),
        allow_unused=True,
    )
    if cache == "network":
        # Saved expanded temperature fields are detached native cache values.
        # The pure cache selector retains an explicit edge with a zero VJP.
        assert oracle == (None, None)
        assert all(
            value is None or torch.count_nonzero(value) == 0 for value in observed
        )
    else:
        # The uncached right Population still materializes from the restored
        # shared Parameters, so its runtime sensitivity remains differentiable.
        _close(observed, oracle)
        assert all(value is not None and value.abs().sum() > 0 for value in observed)
    _close(tensors, frozen)


@pytest.mark.parametrize("cache", ("population", "network"))
def test_cached_parametrized_effective_buffer_retains_native_transform_gradients(cache):
    source, native = (
        _network(cache=cache, target="cm"),
        _network(cache=cache, target="cm"),
    )
    functional, tensors = dn.func.make_functional_network(source)
    key = "parametrizations.cm.0.factor"
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized, _cache_initialized_native_packet(functional, native))
    factor = initialized.parameters["populations"]["left"][key]
    assert factor is initialized.parameters["populations"]["right"][key]
    # The composed program differentiates its raw input. A later cache write
    # returns a new effective source tensor; earlier owner materializations
    # retain their path to the original input before that transaction.
    input_factor = tensors.parameters["populations"]["left"][key]
    actual, _ = functional.prepare_and_rollout(*initialized, steps=3)
    for _ in range(3):
        native.step()
    _close(actual, functional.extract(native).state)
    observed = torch.autograd.grad(_loss(actual), input_factor)[0]
    oracle = torch.autograd.grad(
        native.left.v.square().sum() + native.right.v.square().sum(),
        native.left.parametrizations.cm[0].factor,
    )[0]
    torch.testing.assert_close(observed, oracle)
    assert observed.isfinite().all() and observed.abs().sum() > 0


def test_parametrized_network_inductor_scan_native_state_and_gradients():
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=2,
        inference=False,
        trainable=_trainable(),
        backend="inductor",
    )
    final, _ = runner()
    for _ in range(2):
        native.step()
    _close(final, runner.functional.extract(native).state)
    gradients = torch.autograd.grad(_loss(final), _targets(runner.parameters))
    oracle = torch.autograd.grad(
        native.left.v.square().sum() + native.right.v.square().sum(),
        _native_targets(native),
    )
    _close(gradients, oracle)
    assert all(value.abs().sum() > 0 for value in gradients)


def test_nonpersistent_cross_population_transform_config_cache_fails_before_mutation():
    def network():
        model = _network(target="cm")
        shared = model.left.parametrizations.cm[0].offset
        reader = model.right.parametrizations.cm[0]
        reader._buffers["offset"] = shared
        reader._non_persistent_buffers_set.add("offset")
        with torch.no_grad():
            for _ in range(2):
                model.step()
            model.left.cache("_steady_state")
            shared.add_(0.25)
        saved = model.left._caches.pop("_steady_state")
        model.initialize(DT, reinit_weights=False, reinit_delays=False)
        model.left._caches["_steady_state"] = saved
        return model

    source, native = network(), network()
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors, detached=True)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    torch.testing.assert_close(
        native.right.parametrizations.cm[0].offset,
        native.right.v.new_tensor(0.5),
    )
    # The writer's owned cache changes a different Population's nonpersistent
    # reader. This unsupported replay must fail before publishing any state.
    with pytest.raises(dn.func.FunctionalizationError, match="nonpersistent.*config"):
        functional.initialize(*tensors, functional.initialization_inputs)
    _close(tensors, frozen)
    # The same fixed configuration remains a valid snapshot execution graph.
    actual, _ = functional.prepare_and_step(*tensors)
    source.step()
    _close(actual, functional.extract(source).state)


def test_parametrized_diameter_population_owned_cache_keeps_native_geometry_gradient():
    def population():
        model = _population("ub")
        model.v_init = torch.linspace(
            -65.0, -50.0, model.v.numel(), dtype=DTYPE
        ).reshape(model.shape)
        transform = _TemperatureScale(
            torch.nn.Parameter(torch.tensor(0.9, dtype=DTYPE))
        )
        model.register_parametrization("diam", transform)
        model.initialize()
        with torch.no_grad():
            for _ in range(2):
                _native_population_step(model)
            model.cache("_steady_state")
            model.parametrizations.diam.original.mul_(1.1)
            transform.factor.add_(0.2)
            transform.offset.add_(0.15)
        saved = model._caches.pop("_steady_state")
        model.initialize()
        model._caches["_steady_state"] = saved
        return model

    source, native = population(), population()
    functional, tensors = dn.func.make_functional(source, dt=CACHE_DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native.initialize()
    original_key = next(
        entry.key
        for entry in functional._parametrization_constants
        if entry.module_path == "parametrizations.diam"
        and entry.buffer_name == "original"
    )
    torch.testing.assert_close(
        initialized.constants[original_key], native.parametrizations.diam.original
    )
    key = "parametrizations.diam.0.factor"
    torch.testing.assert_close(
        initialized.parameters[key], native.parametrizations.diam[0].factor
    )
    actual = initialized.state
    for _ in range(2):
        actual, _ = functional.prepare_and_step(
            initialized.parameters, initialized.constants, actual
        )
        _native_population_step(native)
    _close(actual, functional.extract(native).state)
    observed = torch.autograd.grad(
        actual["integrator"]["v"].square().sum(), initialized.parameters[key]
    )[0]
    oracle = torch.autograd.grad(
        native.v.square().sum(), native.parametrizations.diam[0].factor
    )[0]
    torch.testing.assert_close(observed, oracle)
    assert observed.isfinite().all() and observed.abs().sum() > 0


@pytest.mark.parametrize("check_purity", (False, True))
@pytest.mark.parametrize("role", ("intra", "extra", "mechanism"))
def test_documented_waveform_frequency_parametrization_keeps_native_stimulation_gradient(
    role, check_purity
):
    def waveform():
        wave = _wave("sin", amp_scale=0.2 * nA if role != "extra" else 0.2)
        register_parametrization(wave, "freq", FrequencyFromCycles(3.5).to(wave.freq))
        return wave

    wave, native_wave = waveform(), waveform()
    if role == "mechanism":
        model, native = _injected_population(wave), _injected_population(native_wave)
        for population in (model, native):
            population.train()
            population.initialize()
    else:
        model = _cable(wave if role == "intra" else _wave("sin", amp_scale=0.0))
        native = _cable(native_wave if role == "intra" else _wave("sin", amp_scale=0.0))
    field = torch.linspace(-0.04, 0.06, model.v.numel(), dtype=DTYPE).reshape(
        model.shape
    )
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional(
            model, dt=0.01, extra=(field, wave) if role == "extra" else None
        )
    actual, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=8
    )
    native.run(
        tstop=0.08,
        dt=0.01,
        extra=(field, native_wave) if role == "extra" else None,
    )
    torch.testing.assert_close(
        actual["integrator"]["v"], native.v, rtol=2e-10, atol=2e-10
    )
    name = next(
        name
        for name in tensors.parameters
        if name.endswith("parametrizations.freq.original")
    )
    observed = torch.autograd.grad(
        actual["integrator"]["v"].square().sum(), tensors.parameters[name]
    )[0]
    oracle = torch.autograd.grad(
        native.v.square().sum(), native_wave.parametrizations.freq.original
    )[0]
    torch.testing.assert_close(observed, oracle, rtol=2e-8, atol=2e-9)
    assert observed.isfinite().all() and observed.abs().sum() > 0

    def frequency_update(multiplier):
        parameters = {
            **tensors.parameters,
            name: tensors.parameters[name] * multiplier,
        }
        state, _ = functional.prepare_and_rollout(
            parameters, tensors.constants, tensors.state, steps=8
        )
        return state["integrator"]["v"]

    # An ambient native getter cache cannot override the explicit frequency
    # operands supplied to functional stimulation.
    with torch.no_grad():
        expected = tuple(frequency_update(value) for value in (0.8, 1.3))
        assert torch.count_nonzero(expected[0] - expected[1])
        with cached():
            for multiplier, reference in zip((0.8, 1.3), expected, strict=True):
                torch.testing.assert_close(frequency_update(multiplier), reference)


@pytest.mark.parametrize("field", ("xc", "xraxial"))
@pytest.mark.parametrize("cached", (False, True))
def test_packed_extracellular_parametrized_fields_preserve_native_cached_vjp(
    field, cached
):
    def population():
        model = _membrane_model("multi_block", batch=None)
        component = model.populations["block_path"]
        transform = _TemperatureScale(
            torch.nn.Parameter(torch.tensor(0.9, dtype=DTYPE))
        )
        transform.offset.zero_()
        component.register_parametrization(field, transform)
        model.initialize()
        _seed_circuit(model)
        if cached:
            with torch.no_grad():
                for _ in range(2):
                    _native_population_step(model)
                model.cache("_steady_state")
                component.parametrizations[field].original.mul_(1.15)
                transform.factor.add_(0.2)
                transform.offset.add_(0.02)
            saved = model._caches.pop("_steady_state")
            model.initialize()
            model._caches["_steady_state"] = saved
        return model

    source, native = population(), population()
    functional, tensors = dn.func.make_functional(source, dt=CACHE_DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native.initialize()
    registry = native.populations["block_path"].parametrizations[field]
    prefix = f"populations.block_path.parametrizations.{field}"
    key = prefix + ".0.factor"
    torch.testing.assert_close(initialized.parameters[key], registry[0].factor)
    for relative, expected in (
        ("original", registry.original),
        ("0.offset", registry[0].offset),
    ):
        constant = next(
            value
            for name, value in initialized.constants.items()
            if name.startswith("populations.block_path.")
            and name.endswith(f"parametrizations.{field}.{relative}")
        )
        torch.testing.assert_close(constant, expected)
    ve, intra = _membrane_inputs(source, steps=2)
    actual, _ = functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
        steps=2,
    )
    dt = native.v.new_tensor(CACHE_DT)
    native.integrator._initialize(native, dt, force=True, compile_scope="population")
    for index in range(2):
        native.integrator.step(native, dt, ve[index], intra[index])
        native.t = native.t + dt
    _close(actual, functional.extract(native).state)
    voltage = actual["integrator"]["vc"]
    observed = torch.autograd.grad(
        voltage.square().sum(), initialized.parameters[key], allow_unused=True
    )[0]
    oracle = torch.autograd.grad(
        native.vc.square().sum(), registry[0].factor, allow_unused=True
    )[0]
    if cached and field == "xc":
        # The packed cache restores the assembled parent shell capacitance.
        assert oracle is None or torch.count_nonzero(oracle) == 0
        assert observed is None or torch.count_nonzero(observed) == 0
    else:
        # Extracellular axial edges read the live child getter even when the
        # packed parent's assembled physical fields came from its cache.
        torch.testing.assert_close(observed, oracle, rtol=2e-8, atol=2e-8)
        assert observed.isfinite().all() and observed.abs().sum() > 1e-6
