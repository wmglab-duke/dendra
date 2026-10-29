"""One cold initialization boundary publishes effective explicit sources."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import (
    _assign,
    _callbacks,
    _close,
    _joined,
    _plain,
    _snapshot,
)
from tests.func.test_continuation_initialization import _Scale
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_membrane_current import DT as CACHE_DT
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network import _stim_network
from tests.func.test_network_cache_initialization import _cached_network
from tests.func.test_network_initialization_bindings import _cross_population_binding
from tests.func.test_network_mapping_boundaries import _OnItems
from tests.func.test_population_owned_cache import _cached as _cached_population
from tests.func.test_stimulation_rng_sharing import DT as SHARED_DT
from tests.func.test_stimulation_rng_sharing import _network as _shared_network
from tests.func.test_stimulation_rng_sharing import _one_stream
from tests.func.test_stimulation_rng_sharing import _population as _shared_population

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _case(network, *, scaled=False):
    model = _network() if network else _population()
    if scaled:
        owner = model.left if network else model
        owner.register_pre_initialize_transform(
            "scale",
            _Scale(),
            reads=(f"parameters.{G}",),
            writes=(f"parameters.{G}",),
            inputs={"factor": owner.v.new_tensor(1.5)},
        )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    return functional, tensors


def _defaults(functional, tensors, network):
    return functional.initialization_inputs if network else tensors.initialization


def _pure(functional, tensors, inputs, network):
    if network:
        return functional.initialize(
            tensors.parameters, tensors.constants, tensors.state, inputs
        )
    return functional.initialize(
        tensors.parameters, tensors.constants, inputs, state=tensors.state
    )


def _frames(tensors):
    return tensors.parameters, tensors.constants, tensors.state


def _training_case(network, compiled, monkeypatch):
    functional, tensors = _case(network, scaled=True)
    native = _snapshot(functional.extract().state)
    parameters = _plain(tensors.parameters)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    _assign(parameters, network, upstream * 1.2)
    tensors = tensors._replace(parameters=parameters)
    factor = upstream.new_tensor(1.7, requires_grad=True)
    defaults = _defaults(functional, tensors, network)
    if network:
        populations = dict(defaults.populations)
        population = populations["left"]
        populations["left"] = population._replace(
            transforms={**population.transforms, "pre.scale.factor": factor}
        )
        inputs = defaults._replace(populations=populations)
    else:
        inputs = defaults._replace(
            transforms={**defaults.transforms, "pre.scale.factor": factor}
        )
    initialized = dn.func.initialize(functional, tensors, initialization=inputs)
    assert type(initialized) is dn.func.Continuation
    assert initialized.callback_state is None
    effective = _target(initialized.parameters, network)
    assert effective is not _target(parameters, network)
    assert effective.requires_grad and effective.grad_fn is not None
    torch.testing.assert_close(effective, upstream * 1.2 * factor)
    if network:
        assert effective is initialized.parameters["populations"]["right"][G]
    reference = _pure(functional, tensors, inputs, network)
    _close(_frames(initialized), _frames(reference))
    callbacks = functional.make_callbacks(_callbacks(network))
    expected, whole = functional.prepare_and_rollout(
        reference.parameters,
        reference.constants,
        reference.state,
        steps=5,
        callbacks=callbacks,
    )
    owner = (
        functional.compile_rollout_chunk(2, backend="aot_eager")
        if compiled
        else functional
    )
    bound = owner.bind(initialized.parameters, constants=initialized.constants)
    callback_initializations = []
    initialize_callbacks = callbacks._initialize

    def counted(*args, **kwargs):
        callback_initializations.append(None)
        return initialize_callbacks(*args, **kwargs)

    monkeypatch.setattr(callbacks, "_initialize", counted)
    _, first = dn.func.run(
        bound,
        initialized,
        steps=2,
        checkpoint_every=2,
        callbacks=callbacks,
    )
    actual, last = dn.func.run(
        bound,
        first["continuation"],
        steps=3,
        checkpoint_every=2,
        callbacks=callbacks,
    )
    assert len(callback_initializations) == 1
    trace = "left.v" if network else "v"
    assert first["callbacks"]["trace"][trace].shape[0] == 3
    assert last["callbacks"]["trace"][trace].shape[0] == 3
    _close(actual, expected)
    _close(_joined(first, last), whole["callbacks"]["trace"])
    _close(last["callbacks"]["count"], whole["callbacks"]["count"])
    observed = torch.autograd.grad(
        _voltage(actual, network).square().sum(),
        (upstream, factor),
        retain_graph=True,
    )
    gradients = torch.autograd.grad(
        _voltage(expected, network).square().sum(), (upstream, factor)
    )
    _close(observed, gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert _target(last["continuation"].parameters, network) is effective
    _close(functional.extract().state, native)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("compiled", (False, True))
def test_initialized_effective_sources_span_binding_callbacks_and_checkpointed_gradients(
    network, compiled, monkeypatch
):
    _training_case(network, compiled, monkeypatch)


@pytest.mark.parametrize("network", (False, True))
def test_default_initialization_uses_bundle_or_plan_inputs_without_extracting_native_state(
    network, monkeypatch
):
    functional, tensors = _case(network)
    native = _snapshot(functional.extract().state)
    if network:
        functional.initialization_inputs.populations["left"].v_init.add_(3.0)
        functional.initialization_inputs.t.fill_(0.75)
    else:
        tensors = tensors._replace(
            initialization=tensors.initialization._replace(
                v_init=tensors.initialization.v_init + 3.0
            )
        )
    before = _snapshot(tensors)
    expected = _pure(
        functional, tensors, _defaults(functional, tensors, network), network
    )

    def reject(*args, **kwargs):
        raise AssertionError("explicit initialization must use the bundle state")

    with monkeypatch.context() as guarded:
        guarded.setattr(functional, "extract", reject)
        if network:
            guarded.setattr(functional, "_extract", reject)
            populations = functional.populations.values()
        else:
            populations = (functional,)
        for population in populations:
            guarded.setattr(population, "extract", reject)
            guarded.setattr(population, "_extract_native_state", reject)
        actual = dn.func.initialize(functional, tensors)
    _close(_frames(actual), _frames(expected))
    _close(tensors, before)
    _close(functional.extract().state, native)


@pytest.mark.parametrize("network", (False, True))
def test_shared_model_and_poisson_initialization_draws_match_automatic_sampling_and_replay(
    network, monkeypatch
):
    if network:
        model, pulses = _shared_network()
    else:
        model, _, pulse = _shared_population()
        pulses = [pulse]
    functional, tensors = dn.func.make_functional(model, dt=SHARED_DT)
    native = _snapshot(functional.extract())
    native_rng = pulses[0].generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    defaults = _defaults(functional, tensors, network)
    if network:
        random = functional.sample_population_initialization_draws(
            tensors.parameters, tensors.state
        )
        stimulation = functional.sample_population_stimulation_initialization_draws(
            tensors.parameters, tensors.state
        )
        inputs = defaults._replace(
            populations={
                name: value._replace(
                    random_draw=random[name], stimulation=stimulation[name]
                )
                for name, value in defaults.populations.items()
            }
        )
    else:
        inputs = defaults._replace(
            random_draw=functional.sample_initialization_draw(
                tensors.parameters, tensors.state
            ),
            stimulation=functional.sample_stimulation_initialization_draw(
                tensors.state
            ),
        )
    expected = _pure(functional, tensors, inputs, network)
    automatic = dn.func.initialize(functional, tensors)
    _close(_frames(automatic), _frames(expected))

    def reject(*args, **kwargs):
        raise AssertionError("supplied initialization draws must be replayed")

    with monkeypatch.context() as guarded:
        for name in (
            "sample_initialization_draw",
            "sample_stimulation_initialization_draw",
            "sample_population_initialization_draws",
            "sample_population_stimulation_initialization_draws",
        ):
            if hasattr(functional, name):
                guarded.setattr(functional, name, reject)
        replayed = dn.func.initialize(functional, tensors, initialization=inputs)
    _close(_frames(replayed), _frames(automatic))
    continued = dn.func.initialize(
        functional,
        tensors._replace(
            parameters=automatic.parameters,
            constants=automatic.constants,
            state=automatic.state,
        ),
    )
    assert not torch.equal(_one_stream(automatic.state), _one_stream(continued.state))
    _close(functional.extract(), native)
    torch.testing.assert_close(pulses[0].generator.get_state(), native_rng)
    torch.testing.assert_close(torch.get_rng_state(), global_rng)


def test_initialized_population_bundle_preserves_its_consumed_explicit_draws_by_default(
    monkeypatch,
):
    model, _, _ = _shared_population()
    functional, tensors = dn.func.make_functional(model, dt=SHARED_DT)
    inputs = tensors.initialization._replace(
        random_draw=functional.sample_initialization_draw(
            tensors.parameters, tensors.state
        ),
        stimulation=functional.sample_stimulation_initialization_draw(tensors.state),
    )
    previous = _pure(functional, tensors, inputs, False)
    expected = _pure(functional, previous, previous.initialization, False)

    def reject(*args, **kwargs):
        raise AssertionError("bundle initialization inputs are explicit replay data")

    monkeypatch.setattr(functional, "sample_initialization_draw", reject)
    monkeypatch.setattr(functional, "sample_stimulation_initialization_draw", reject)
    actual = dn.func.initialize(functional, previous)
    _close(_frames(actual), _frames(expected))


def test_netstim_initialization_draw_override_preserves_seed_clock_and_native_rng():
    model = _stim_network()
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    defaults = functional.initialization_inputs
    draw = functional.sample_initialization_draw(
        tensors.parameters, tensors.state, constants=tensors.constants
    )
    inputs = defaults._replace(netstim_draw=draw)
    native_rng = model.netstim._rng.get_state().clone()
    automatic = dn.func.initialize(functional, tensors)
    explicit = dn.func.initialize(functional, tensors, initialization=inputs)
    expected = _pure(functional, tensors, inputs, True)
    _close(_frames(automatic), _frames(expected))
    _close(_frames(explicit), _frames(expected))
    torch.testing.assert_close(model.netstim._rng.get_state(), native_rng)


@pytest.mark.parametrize("network", (False, True))
def test_fresh_initialization_preserves_active_cache_frames_and_native_sources(network):
    model = (
        _cached_network(training=False, stimulus=True)
        if network
        else _cached_population("point")
    )
    functional, tensors = dn.func.make_functional(
        model, dt=NETWORK_DT if network else CACHE_DT
    )
    before = _snapshot(tensors)
    inputs = _defaults(functional, tensors, network)
    if network:
        inputs = inputs._replace(
            netstim_draw=functional.sample_initialization_draw(
                tensors.parameters, tensors.state, constants=tensors.constants
            )
        )
    expected = _pure(functional, tensors, inputs, network)
    actual = dn.func.initialize(functional, tensors)
    _close(_frames(actual), _frames(expected))
    constants = actual.constants["populations"]["cell"] if network else actual.constants
    assert bool(constants["cache.active"])
    _close(tensors, before)
    _close(functional.extract(), before)


@pytest.mark.parametrize("reader_first", (False, True))
def test_network_initialization_publishes_canonical_parameter_constant_aliases(
    reader_first,
):
    model, key = _cross_population_binding(reader_first=reader_first)
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    parameters = tensors.independent_parameters(trainable=f"populations.left.{key}")
    upstream = parameters["populations"]["left"][key]
    initialized = dn.func.initialize(
        functional, tensors._replace(parameters=parameters)
    )
    selected = initialized.parameters["populations"]["left"][key]
    assert (
        selected
        is initialized.constants["populations"]["right"]["parametrizations.factor"]
    )
    assert selected.requires_grad
    state, _ = dn.func.run(
        functional.bind(initialized.parameters, constants=initialized.constants),
        initialized,
        steps=2,
    )
    gradient = torch.autograd.grad(_voltage(state, True).square().sum(), upstream)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("mode", ("no_grad", "inference"))
def test_initialization_preserves_ambient_grad_mode_and_constructs_ordinary_tensors(
    network, mode
):
    functional, tensors = _case(network, scaled=True)
    parameters = _plain(tensors.parameters)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    _assign(parameters, network, upstream * 1.2)
    with torch.no_grad() if mode == "no_grad" else torch.inference_mode():
        assert not torch.is_grad_enabled()
        result = dn.func.initialize(functional, tensors._replace(parameters=parameters))
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled() == (mode == "inference")
    target = _target(result.parameters, network)
    assert not target.requires_grad and target.grad_fn is None
    assert all(
        not value.is_inference()
        for value in pytree.tree_leaves(_frames(result))
        if isinstance(value, torch.Tensor)
    )


@pytest.mark.parametrize("network", (False, True))
def test_initialization_rejects_wrong_bundle_and_plan_families_before_host_sampling(
    network, monkeypatch
):
    functional, tensors = _case(network)
    other, incompatible = _case(not network)

    def reject(*args, **kwargs):
        raise AssertionError("invalid initialization request reached host sampling")

    class AuthoredBundle(type(tensors)):
        @property
        def state(self):
            raise AssertionError("bundle subclasses must reject before reading state")

    class AuthoredPlan(type(functional)):
        pass

    monkeypatch.setattr(
        functional, "_initialize_host" if network else "initialize_host", reject
    )
    for invalid in (incompatible, AuthoredBundle(*tensors), object(), None):
        with pytest.raises(TypeError, match="tensors must be a .*Tensors"):
            dn.func.initialize(functional, invalid)
    with pytest.raises(
        ValueError, match="initialization requires explicit non-None state"
    ):
        dn.func.initialize(functional, tensors._replace(state=None))
    for invalid in (
        other.bind(incompatible.parameters, constants=incompatible.constants),
        functional._source if network else functional._population,
        object.__new__(AuthoredPlan),
        object(),
    ):
        with pytest.raises(
            TypeError,
            match="functional must be a FunctionalPopulation or FunctionalNetwork",
        ):
            dn.func.initialize(invalid, tensors)


@pytest.mark.parametrize("network", (False, True))
def test_initialization_rejects_foreign_input_types_without_mutating_bundle(network):
    functional, tensors = _case(network)
    before = _snapshot(tensors)
    with pytest.raises(TypeError, match="InitializationInput"):
        dn.func.initialize(functional, tensors, initialization=object())
    _close(tensors, before)


@pytest.mark.parametrize("network", (False, True))
def test_selected_initialization_mappings_finish_before_native_source_validation(
    network,
):
    functional, tensors = _case(network)
    inputs = _defaults(functional, tensors, network)
    source = functional._source if network else functional._population
    wrapped = _OnItems(
        inputs.populations if network else inputs.transforms,
        lambda: setattr(source, "_step", object()),
    )
    inputs = (
        inputs._replace(populations=wrapped)
        if network
        else inputs._replace(transforms=wrapped)
    )
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.initialize(functional, tensors, initialization=inputs)
    assert wrapped.accessed


@pytest.mark.parametrize("network", (False, True))
def test_explicit_override_does_not_traverse_unused_default_initialization_mappings(
    network,
):
    functional, tensors = _case(network)
    override = _defaults(functional, tensors, network)
    expected = _pure(functional, tensors, override, network)

    def reject():
        raise AssertionError("unused default initialization must not be traversed")

    wrapped = _OnItems(override.populations if network else override.transforms, reject)
    if network:
        functional._initialization_inputs = override._replace(populations=wrapped)
    else:
        tensors = tensors._replace(initialization=override._replace(transforms=wrapped))
    actual = dn.func.initialize(functional, tensors, initialization=override)
    _close(_frames(actual), _frames(expected))
    assert not wrapped.accessed


def test_initializer_has_no_competing_source_or_callback_keywords():
    for keyword in (
        "state",
        "parameters",
        "constants",
        "random_draw",
        "callback_state",
    ):
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{keyword}'"):
            dn.func.initialize(object(), object(), **{keyword: object()})


@pytest.mark.parametrize("context", ("compile", "grad", "forward_ad"))
def test_cold_initialization_rejects_transforms_before_entering_host_sampler(
    context, monkeypatch
):
    functional, tensors = _case(False)

    def reject(*args, **kwargs):
        raise AssertionError("cold-boundary rejection must precede host initialization")

    monkeypatch.setattr(functional, "initialize_host", reject)

    def work(value):
        dn.func.initialize(functional, tensors)
        return value.square()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="initialize must run outside torch.compile",
    ):
        if context == "compile":
            monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
            work(torch.tensor(0.5))
        elif context == "grad":
            torch.func.grad(work)(torch.tensor(0.5))
        else:
            with torch.autograd.forward_ad.dual_level():
                work(torch.tensor(0.5))


@pytest.mark.mps
def test_mps_initialization_preserves_effective_network_sources_and_aot_gradients(
    monkeypatch,
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    from tests.func.test_runtime_extra_backends import (
        _environment as backend_environment,
    )

    with backend_environment("mps"):
        _training_case(True, True, monkeypatch)
