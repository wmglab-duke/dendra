"""Population initialization consumes one replayable initialization packet."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._initialization_draws import InitializationDraws
from tests.func.test_network import _clone, _close
from tests.func.test_network_random_parameters import (
    _RandomLeak,
    _ReseededLeak,
    _StickyLeak,
)
from tests.func.test_random_parameters import DT, _population
from tests.func.test_stimulation_rng_sharing import (
    _initialize_native,
    _lower,
    _one_stream,
)
from tests.func.test_stimulation_rng_sharing import _population as _shared_population
from tests.func.test_stochastic_initialization_ordering import (
    _population as _transformed_population,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                IMEM=False,
                DTYPE=torch.float64,
                DEVICE="cpu",
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _fail_sampling(*args, **kwargs):
    raise AssertionError("an explicit packet must not trigger automatic sampling")


def _same_tensor_leaves(actual, expected):
    left, left_spec = _pytree.tree_flatten(actual)
    right, right_spec = _pytree.tree_flatten(expected)
    assert left_spec == right_spec
    assert all(a is b for a, b in zip(left, right, strict=True))


@pytest.mark.parametrize("mechanism", (_RandomLeak, _StickyLeak, _ReseededLeak))
def test_host_capture_executes_once_and_packet_replays_retained_or_resampled_draws(
    mechanism, monkeypatch
):
    source = _population(mechanism)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors, detached=True)
    initialize = functional.initialize
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return initialize(*args, **kwargs)

    monkeypatch.setattr(functional, "initialize", counted)
    captured = {}
    expected = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        _draws_capture=captured,
    )
    assert calls == [None]
    assert captured["random"] is expected.initialization.random_draw
    assert captured["stimulation"] is expected.initialization.stimulation is None
    packet = InitializationDraws(**captured)
    monkeypatch.setattr(functional, "sample_initialization_draw", _fail_sampling)
    actual = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        draws=packet,
    )
    assert calls == [None, None]
    _close(actual, expected)
    _same_tensor_leaves(actual.initialization.random_draw, packet.random)
    assert actual.initialization.v_init is tensors.initialization.v_init
    assert all(
        actual.initialization.states[name] is value
        for name, value in tensors.initialization.states.items()
    )
    if mechanism is _StickyLeak:
        _close(actual.state["random"], tensors.state["random"])
    _close(tensors, frozen)
    _close(functional.extract().state, tensors.state)


def test_canonical_initialization_compiles_and_preserves_distribution_gradients():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    packet = InitializationDraws(
        random=functional.sample_initialization_draw(tensors.parameters, tensors.state)
    )
    compiled = torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
    actual = compiled(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        draws=packet,
    )
    expected = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        random_draw=packet.random,
    )
    _close(actual, expected)

    def loss(initialized):
        state, _ = functional.prepare_and_step(
            initialized.parameters, initialized.constants, initialized.state
        )
        return state["integrator"]["v"].square().sum()

    name = next(name for name in tensors.parameters if name.endswith("g_mu_param"))
    target = tensors.parameters[name]
    gradient = torch.autograd.grad(loss(actual), target)[0]
    reference = torch.autograd.grad(loss(expected), target)[0]
    _close(gradient, reference)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("method", ("initialize", "initialize_host"))
def test_explicit_empty_packet_does_not_fill_missing_random_draws(method, monkeypatch):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    monkeypatch.setattr(functional, "sample_initialization_draw", _fail_sampling)
    with pytest.raises(ValueError, match="draw packet"):
        getattr(functional, method)(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
            draws=InitializationDraws(),
        )


@pytest.mark.parametrize("legacy", ("random_draw", "stimulation", "keyword"))
def test_canonical_packet_rejects_any_legacy_draw_before_source_validation(
    legacy, monkeypatch
):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    inputs = tensors.initialization
    kwargs = {}
    if legacy == "keyword":
        kwargs["random_draw"] = draw
    else:
        inputs = inputs._replace(
            **{
                legacy: draw
                if legacy == "random_draw"
                else dn.func.StimulationDraw({}, {})
            }
        )
    monkeypatch.setattr(functional, "_validate_source", _fail_sampling)
    with pytest.raises(ValueError, match="legacy initialization draw packets"):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            inputs,
            state=tensors.state,
            draws=InitializationDraws(random=draw),
            **kwargs,
        )
    if legacy != "keyword":
        with pytest.raises(ValueError, match="legacy initialization draw packets"):
            functional.initialize_host(
                tensors.parameters,
                tensors.constants,
                inputs,
                state=tensors.state,
                draws=InitializationDraws(random=draw),
            )


@pytest.mark.parametrize("field", ("netstim", "connections"))
def test_population_packet_rejects_network_families_before_source_validation(
    field, monkeypatch
):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    monkeypatch.setattr(functional, "_validate_source", _fail_sampling)
    packet = InitializationDraws(**{field: object()})
    with pytest.raises(ValueError, match="does not accept NetStim or connection draws"):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            draws=packet,
        )


@pytest.mark.parametrize("reseed", (False, True))
def test_shared_random_and_poisson_packets_capture_one_native_ordered_rng_frame(
    reseed, monkeypatch
):
    source, _, pulse = _shared_population(reseed=reseed)
    native, _, native_pulse = _shared_population(reseed=reseed)
    functional, tensors = _lower(source)
    frozen = _clone(tensors, detached=True)
    source_rng = pulse.generator.get_state().clone()
    captured = {}
    expected = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        _draws_capture=captured,
    )
    assert captured["random"] is expected.initialization.random_draw
    assert captured["stimulation"] is expected.initialization.stimulation
    assert captured["random"] is not None and captured["stimulation"] is not None
    packet = InitializationDraws(**captured)
    monkeypatch.setattr(
        functional._stimulation_rng_plan, "sample_initialization", _fail_sampling
    )
    actual = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        draws=packet,
    )
    _close(actual, expected)
    _initialize_native(native)
    _close(actual.state, functional.extract(native).state)
    _close(_one_stream(actual.state), native_pulse.generator.get_state())
    _close(pulse.generator.get_state(), source_rng)
    _close(tensors, frozen)
    for missing in ("random", "stimulation"):
        with pytest.raises(ValueError, match="draw|packet"):
            functional.initialize_host(
                tensors.parameters,
                tensors.constants,
                tensors.initialization,
                state=tensors.state,
                draws=packet._replace(**{missing: None}),
            )


def test_host_capture_omits_unused_packets_when_owned_cache_skips_initialization(
    monkeypatch,
):
    source, _, pulse = _shared_population(retained=True)
    source.cache("_steady_state")
    functional, tensors = _lower(source)
    assert functional._owned_cache_bound
    frozen = _clone(tensors, detached=True)
    source_rng = pulse.generator.get_state().clone()
    monkeypatch.setattr(functional, "sample_initialization_draw", _fail_sampling)
    monkeypatch.setattr(
        functional, "sample_stimulation_initialization_draw", _fail_sampling
    )
    monkeypatch.setattr(
        functional._stimulation_rng_plan, "sample_initialization", _fail_sampling
    )
    captured = {}
    initialized = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        draws=InitializationDraws(
            random=dn.func.RandomParameterDraw({}, {}),
            stimulation=dn.func.StimulationDraw({}, {}),
        ),
        _draws_capture=captured,
    )
    assert captured == {"random": None, "stimulation": None}
    assert initialized.initialization.cache is tensors.initialization.cache
    assert initialized.initialization.random_draw is not None
    replay = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        draws=InitializationDraws(**captured),
    )
    _close(replay.state, initialized.state)
    _close(replay.parameters, initialized.parameters)
    _close(replay.constants, initialized.constants)
    _close(tensors, frozen)
    _close(pulse.generator.get_state(), source_rng)


def test_canonical_packet_preserves_sampling_around_pre_initialization_transforms():
    source = _transformed_population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors, detached=True)
    captured = {}
    expected = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        _draws_capture=captured,
    )
    actual = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        draws=InitializationDraws(**captured),
    )
    _close(actual, expected)
    _same_tensor_leaves(
        actual.initialization.transforms, tensors.initialization.transforms
    )
    target = tensors.parameters["trial_mu_param"]
    gradient = torch.autograd.grad(actual.parameters["trial_mu_param"].sum(), target)[0]
    _close(gradient, torch.full_like(target, 1.5))
    _close(tensors, frozen)
