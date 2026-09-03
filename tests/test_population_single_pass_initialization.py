"""Population-level contracts for the single-pass initialization lifecycle."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism

DTYPE = torch.float64


class _Add(torch.nn.Module):
    def forward(self, value, increment):
        return (value + increment,)


def _only_mechanism(model, mechanism_type):
    return next(
        mechanism
        for mechanism in model.mech.mechanisms.values()
        if isinstance(mechanism, mechanism_type)
    )


def test_population_calls_handler_once_and_post_hook_does_not_reinitialize(
    monkeypatch,
):
    events = []

    class InitialVoltageProbe(Mechanism):
        Mechanism.CARRY("initial_voltage")

        def initial_values(self, v, values):
            del values
            events.append("initial")
            return {"initial_voltage": v.clone()}

    model = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    model.insert(InitialVoltageProbe)
    model.build()

    handler = model.mech
    original_initialize = handler.initialize
    handler_calls = 0

    def counted_initialize(*args, **kwargs):
        nonlocal handler_calls
        handler_calls += 1
        events.append("handler")
        return original_initialize(*args, **kwargs)

    monkeypatch.setattr(handler, "initialize", counted_initialize)

    def pre_initialize(population):
        events.append("pre")
        population.v.fill_(-55.0)

    def post_initialize(population):
        events.append("post")
        population.v.fill_(-40.0)

    model.register_pre_initialize_hook(pre_initialize)
    model.register_post_initialize_hook(post_initialize)
    model.initialize()

    probe = _only_mechanism(model, InitialVoltageProbe)
    assert handler_calls == 1
    assert events == ["pre", "handler", "initial", "post"]
    torch.testing.assert_close(
        probe.initial_voltage,
        torch.full_like(probe.initial_voltage, -55.0),
    )
    torch.testing.assert_close(model.v, torch.full_like(model.v, -40.0))


def test_pre_transform_overrides_initial_condition_and_invalidates_steady_cache():
    initial_calls = []

    class InitialVoltageProbe(Mechanism):
        Mechanism.CARRY("initial_voltage")

        def initial_values(self, v, values):
            del values
            initial_calls.append(v.detach().clone())
            return {"initial_voltage": v.clone()}

    class ReplaceVoltage(torch.nn.Module):
        def forward(self, value):
            return (value,)

    model = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    model.insert(InitialVoltageProbe)
    model.initialize()
    model.cache("_steady_state")
    assert "_steady_state" in model._caches

    override = torch.tensor([[-51.0, -49.0]], dtype=DTYPE)
    model.register_pre_initialize_transform(
        "replace_voltage",
        ReplaceVoltage(),
        writes=("state.integrator.v",),
        inputs={"value": override},
    )
    assert "_steady_state" not in model._caches

    model.initialize()
    probe = _only_mechanism(model, InitialVoltageProbe)
    torch.testing.assert_close(probe.initial_voltage, override)
    torch.testing.assert_close(model.v, override)
    assert len(initial_calls) == 2

    # A steady-state cache is already a fully initialized snapshot. Restoring
    # it must not rerun either pre-initialization overrides or INITIAL.
    cached_voltage = torch.tensor([[-47.0, -46.0]], dtype=DTYPE)
    model.v.copy_(cached_voltage)
    probe.initial_voltage.copy_(cached_voltage)
    model.cache("_steady_state")
    model.v.fill_(20.0)
    probe.initial_voltage.fill_(30.0)

    model.initialize()

    assert model.initializing_from_state_cache
    assert len(initial_calls) == 2
    torch.testing.assert_close(model.v, cached_voltage)
    torch.testing.assert_close(probe.initial_voltage, cached_voltage)


def test_steady_restore_skips_fresh_initialization_work(monkeypatch):
    events = []

    class InitialProbe(Mechanism):
        def initial_values(self, v, values):
            del v, values
            events.append("initial")
            return {}

    model = dn.Population(N=1, C=2, dtype=DTYPE)
    model.insert(InitialProbe)
    model.register_pre_initialize_hook(lambda _model: events.append("pre"))
    model.register_post_initialize_hook(lambda _model: events.append("post"))
    model.initialize()
    model.cache("_steady_state")
    assert events == ["pre", "initial", "post"]

    population_populates = 0
    handler_initializes = 0
    original_population_populate = model.populate_parameter_buffers
    original_handler_initialize = model.mech.initialize

    def counted_population_populate(*args, **kwargs):
        nonlocal population_populates
        population_populates += 1
        return original_population_populate(*args, **kwargs)

    def counted_handler_initialize(*args, **kwargs):
        nonlocal handler_initializes
        handler_initializes += 1
        return original_handler_initialize(*args, **kwargs)

    monkeypatch.setattr(
        model,
        "populate_parameter_buffers",
        counted_population_populate,
    )
    monkeypatch.setattr(model.mech, "initialize", counted_handler_initialize)

    model.initialize()

    assert model.initializing_from_state_cache
    assert population_populates == 0
    assert handler_initializes == 0
    assert events == ["pre", "initial", "post", "post"]

    integrator_initializes = 0
    original_integrator_initialize = model.integrator.initialize

    def counted_integrator_initialize(*args, **kwargs):
        nonlocal integrator_initializes
        integrator_initializes += 1
        return original_integrator_initialize(*args, **kwargs)

    monkeypatch.setattr(
        model.integrator,
        "initialize",
        counted_integrator_initialize,
    )
    model.step(dt=0.01)
    model.step(dt=0.01)

    assert integrator_initializes == 1
    assert model.initializing_from_state_cache
    assert not model._integrator_reinit_pending


def test_steady_restore_preserves_structured_post_transform_outputs_exactly():
    observed = []
    model = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    model.register_post_initialize_transform(
        "raise_voltage",
        _Add(),
        reads=("state.integrator.v",),
        writes=("state.integrator.v",),
        inputs={"increment": torch.tensor(5.0, dtype=DTYPE)},
    )
    model.register_post_initialize_hook(
        lambda population: observed.append(population.v.detach().clone())
    )
    model.initialize()
    expected = model.v.detach().clone()
    model.cache("_steady_state")
    model.v.fill_(20.0)

    model.initialize()

    assert model.initializing_from_state_cache
    torch.testing.assert_close(model.v, expected, rtol=0.0, atol=0.0)
    assert len(observed) == 2
    for value in observed:
        torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)


def test_registering_post_transform_invalidates_steady_cache():
    model = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    model.initialize()
    model.cache("_steady_state")

    model.register_post_initialize_transform(
        "raise_voltage",
        _Add(),
        reads=("state.integrator.v",),
        writes=("state.integrator.v",),
        inputs={"increment": torch.tensor(5.0, dtype=DTYPE)},
    )

    assert "_steady_state" not in model._caches
    model.initialize()
    assert not model.initializing_from_state_cache
    torch.testing.assert_close(
        model.v,
        torch.full_like(model.v, -60.0),
        rtol=0.0,
        atol=0.0,
    )


def test_set_v_init_invalidates_steady_cache_before_the_next_initialize():
    model = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    model.initialize()
    model.v.fill_(-42.0)
    model.cache("_steady_state")

    model.set_v_init(-51.0)

    assert "_steady_state" not in model._caches
    model.initialize()
    assert not model.initializing_from_state_cache
    torch.testing.assert_close(model.v, torch.full_like(model.v, -51.0))


def test_failed_post_hook_is_fail_closed_without_a_hidden_second_initialization():
    initial_calls = 0

    class InitialCountProbe(Mechanism):
        def initial_values(self, v, values):
            nonlocal initial_calls
            del v, values
            initial_calls += 1
            return {}

    model = dn.Population(N=1, C=1, dtype=DTYPE)
    model.insert(InitialCountProbe)

    def fail(_population):
        raise RuntimeError("injected post-initialize failure")

    model.register_post_initialize_hook(fail)
    with pytest.raises(RuntimeError, match="post-initialize failure"):
        model.initialize()

    assert initial_calls == 1
    assert not model.initialized
    assert not model.initializing_from_state_cache
    assert not model.integrator.initialized
    with pytest.raises(ValueError, match="initialized"):
        model.step(dt=0.01)

    model.post_initialize_hooks.clear()
    model.initialize()
    assert initial_calls == 2
    assert model.initialized


def test_fresh_initialize_samples_each_mechanism_stochastic_stream_once():
    class StochasticProbe(Mechanism):
        Mechanism.RANGERAND(
            "quenched",
            distribution="normal",
            mu=0.0,
            sigma=1.0,
            seed=123,
        )
        Mechanism.RANGENOISE(
            "runtime",
            distribution="normal",
            mu=0.0,
            sigma=1.0,
            seed=456,
        )

    shape = (1, 4)
    model = dn.Population(N=shape[0], C=shape[1], dtype=DTYPE)
    model.insert(StochasticProbe)
    model.build()
    probe = _only_mechanism(model, StochasticProbe)

    quenched_generator = torch.Generator(device="cpu").manual_seed(123)
    runtime_generator = torch.Generator(device="cpu").manual_seed(456)
    constructor_quenched = torch.randn(shape, generator=quenched_generator, dtype=DTYPE)
    constructor_runtime = torch.randn(shape, generator=runtime_generator, dtype=DTYPE)
    expected_quenched = torch.randn(shape, generator=quenched_generator, dtype=DTYPE)
    expected_runtime = torch.randn(shape, generator=runtime_generator, dtype=DTYPE)
    expected_next_quenched = torch.randn(
        shape, generator=quenched_generator, dtype=DTYPE
    )
    expected_next_runtime = torch.randn(shape, generator=runtime_generator, dtype=DTYPE)

    # Construction samples each declaration once. A fresh Population
    # initialization must consume exactly one additional sample from each
    # stream, including detached runtime NOISE.
    torch.testing.assert_close(probe.quenched, constructor_quenched)
    torch.testing.assert_close(probe.runtime, constructor_runtime)
    model.initialize()
    torch.testing.assert_close(probe.quenched, expected_quenched)
    torch.testing.assert_close(probe.runtime, expected_runtime)

    actual_next_quenched = probe.quenched_rng.randn(shape, device="cpu", dtype=DTYPE)
    actual_next_runtime = probe.runtime_rng.randn(shape, device="cpu", dtype=DTYPE)
    torch.testing.assert_close(actual_next_quenched, expected_next_quenched)
    torch.testing.assert_close(actual_next_runtime, expected_next_runtime)
