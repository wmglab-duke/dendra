"""Native observation sessions own frames without owning evolving carry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from dendra.func import FunctionalCallback, FunctionalCallbacks
from dendra.func._native_callbacks import _bind_native_callbacks
from dendra.func._types import FunctionalizationError

pytestmark = pytest.mark.cpu


class _ViewSquares(FunctionalCallback):
    def initialize(self, state, auxiliary):
        assert auxiliary is None
        value = state["integrator"]["v"]
        return {"value": value, "alias": state["aliases"]["v"]}, value.square()

    def update(self, carry, state, auxiliary):
        del carry
        value = state["integrator"]["v"]
        assert value is auxiliary["v"] and value is state["aliases"]["v"]
        return {"value": value, "alias": state["aliases"]["v"]}, value.square()

    def finalize(self, carry, emissions):
        del carry
        return emissions


def _case(callback=None, *, requires_grad=False):
    population = torch.nn.Module()
    population.register_buffer(
        "v",
        torch.tensor([[1.0, 2.0]], dtype=torch.float64, requires_grad=requires_grad),
    )
    population.register_buffer("v_alias", population.v)
    population.register_buffer("t", torch.tensor(0.0, dtype=torch.float64))
    events = []

    def state():
        return {
            "integrator": {"v": population.v},
            "aliases": {"v": population.v_alias},
            "clock": {"t": population.t},
        }

    functional = SimpleNamespace(
        _population=population,
        _validate_source=lambda: events.append("source"),
        _state_layout=tuple(
            SimpleNamespace(public_path=path, module_path="", buffer_name=name)
            for path, name in (
                (("integrator", "v"), "v"),
                (("aliases", "v"), "v_alias"),
                (("clock", "t"), "t"),
            )
        ),
        extract=lambda: SimpleNamespace(state=state()),
    )
    callbacks = FunctionalCallbacks(
        functional, {"view": _ViewSquares() if callback is None else callback}
    )
    return SimpleNamespace(
        population=population,
        functional=functional,
        callbacks=callbacks,
        events=events,
    )


def test_disabled_session_performs_no_population_access_and_rejects_orphan_state():
    class Untouched:
        def __getattribute__(self, name):
            raise AssertionError("disabled observation must not access its Population")

    assert _bind_native_callbacks(Untouched(), None, None) is None
    with pytest.raises(
        ValueError, match="^functional_callback_state requires functional_callbacks$"
    ):
        _bind_native_callbacks(Untouched(), None, object())
    with pytest.raises(
        TypeError,
        match="^functional_callbacks must be a FunctionalCallbacks plan created by functional.make_callbacks",
    ):
        _bind_native_callbacks(Untouched(), object(), None)


def test_session_reuses_exact_source_and_native_hdf5_validation():
    case = _case()
    with pytest.raises(FunctionalizationError, match="exact source Population"):
        _bind_native_callbacks(torch.nn.Module(), case.callbacks, None)
    assert case.events == []
    _bind_native_callbacks(case.population, case.callbacks, None)
    assert case.events == ["source"]
    case.callbacks._storage = {"view": object()}
    with pytest.raises(FunctionalizationError, match="HDF5 recording"):
        _bind_native_callbacks(case.population, case.callbacks, None)
    assert case.events == ["source"]


def test_session_owns_native_frames_preserves_aliases_and_survives_buffer_mutation():
    case = _case(requires_grad=True)
    raw = case.population.v
    session = _bind_native_callbacks(case.population, case.callbacks, None)
    initial, parts = session.start()
    assert initial["view"]["value"] is initial["view"]["alias"]
    assert initial["view"]["value"] is not raw
    with torch.no_grad():
        raw.add_(1.0)
    current, emitted = session.update(initial)
    assert current["view"]["value"] is current["view"]["alias"]
    assert current["view"]["value"] is not raw
    assert current["view"]["value"] is not initial["view"]["value"]
    session.collect(current, parts, session.stack([emitted]))
    with torch.no_grad():
        raw.fill_(100.0)
    result = session.finish(current, parts)
    torch.testing.assert_close(
        result["view"], torch.tensor([[[1.0, 4.0]], [[4.0, 9.0]]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        initial["view"]["value"], torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.state["view"]["value"], torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    )
    assert result.state["view"]["value"] is result.state["view"]["alias"]
    gradient = torch.autograd.grad(result["view"].sum(), raw)[0]
    torch.testing.assert_close(
        gradient, torch.tensor([[6.0, 10.0]], dtype=torch.float64)
    )


def test_resume_setup_is_lazy_and_emits_no_duplicate_initial_boundary(monkeypatch):
    case = _case()
    first = _bind_native_callbacks(case.population, case.callbacks, None)
    state, parts = first.start()
    prefix = first.finish(state, parts)
    resumed = _bind_native_callbacks(case.population, case.callbacks, prefix.state)

    def fail():
        raise AssertionError("resume setup must not read a new native frame")

    with monkeypatch.context() as context:
        context.setattr(type(resumed.binding), "state", lambda self: fail())
        state, parts = resumed.start()
    assert state is prefix.state
    empty = resumed.finish(state, parts)
    assert empty["view"].shape == (0, 1, 2)
    with torch.no_grad():
        case.population.v.add_(1.0)
    updated, emitted = resumed.update(state)
    resumed.collect(updated, parts, resumed.stack([emitted]))
    tail = resumed.finish(updated, parts)
    assert tail["view"].shape == (1, 1, 2)
    torch.testing.assert_close(
        tail["view"], torch.tensor([[[4.0, 9.0]]], dtype=torch.float64)
    )
    assert resumed.initial_callback_state is prefix.state


def test_session_does_not_store_evolving_carry_and_repeated_starts_are_fresh():
    case = _case()
    session = _bind_native_callbacks(case.population, case.callbacks, None)
    first, _ = session.start()
    second, _ = session.start()
    assert first is not second
    assert first["view"]["value"] is not second["view"]["value"]
    assert session.initial_callback_state is None
    assert set(vars(session)) == {"callbacks", "binding", "initial_callback_state"}
    with pytest.raises(FrozenInstanceError):
        session.initial_callback_state = first


def test_resume_retains_existing_plan_and_carry_validation():
    case = _case()
    session = _bind_native_callbacks(case.population, case.callbacks, object())
    with pytest.raises(
        TypeError, match="callback_state must be the FunctionalCallbackState"
    ):
        session.start()
    other = _case()
    foreign, parts = _bind_native_callbacks(
        other.population, other.callbacks, None
    ).start()
    session = _bind_native_callbacks(case.population, case.callbacks, foreign)
    with pytest.raises(FunctionalizationError, match="different callback plan"):
        session.start()


def test_session_timestep_validation_forwards_identity_and_exception(monkeypatch):
    case = _case()
    session = _bind_native_callbacks(case.population, case.callbacks, None)
    timestep = object()
    failure = ValueError("prepared timestep mismatch")
    values = []

    def validate(value):
        values.append(value)
        raise failure

    monkeypatch.setattr(case.callbacks, "_validate_runtime_dt", validate)
    with pytest.raises(ValueError) as caught:
        session.validate_dt(timestep)
    assert caught.value is failure and values == [timestep]


def test_session_updates_use_current_buffer_owners_and_share_auxiliary_snapshot():
    case = _case()
    session = _bind_native_callbacks(case.population, case.callbacks, None)
    state, _ = session.start()
    replacement = torch.tensor([[7.0, 8.0]], dtype=torch.float64)
    case.population.v = case.population.v_alias = replacement
    updated, emitted = session.update(state)
    assert updated["view"]["value"] is updated["view"]["alias"]
    assert updated["view"]["value"] is not replacement
    torch.testing.assert_close(updated["view"]["value"], replacement)
    torch.testing.assert_close(emitted["view"], replacement.square())


class _NetworkViews(FunctionalCallback):
    def initialize(self, state, auxiliary):
        assert auxiliary is None
        values = {
            name: child["integrator"]["v"]
            for name, child in state["populations"].items()
        }
        return values, values["left"].square()

    def update(self, carry, state, auxiliary):
        del carry
        assert set(auxiliary) == {"populations"}
        values = {}
        for name, child in state["populations"].items():
            values[name] = child["integrator"]["v"]
            assert auxiliary["populations"][name]["v"] is values[name]
        return values, values["left"].square()

    def finalize(self, carry, emissions):
        del carry
        return emissions


def _network_case():
    from dendra.func._network import FunctionalNetwork

    left, right = _case(requires_grad=True), _case(requires_grad=True)
    right.population.v = right.population.v_alias = left.population.v
    network = SimpleNamespace(
        populations={"left": left.population, "right": right.population},
        synapses={},
        continuous_synapses={},
        netstim=None,
        t=left.population.t,
        _clock_origin=left.population.t,
        _clock_step=torch.tensor(0, dtype=torch.int64),
        _duration_remainder=torch.tensor(0.0, dtype=torch.float64),
    )
    functional = object.__new__(FunctionalNetwork)
    functional._source = network
    functional.populations = {"left": left.functional, "right": right.functional}
    for plan in functional.populations.values():
        plan._random_plan = plan._poisson_plan = None
    functional.synapses = functional.continuous_synapses = {}
    functional.netstim = None
    functional._rng_plan = SimpleNamespace(
        enabled=False, streams={}, strip_population=lambda value: value
    )
    functional._stimulation_plan = SimpleNamespace(
        enabled=False, plans={}, strip=lambda value: value
    )
    functional._connection_sampling_plan = SimpleNamespace(extract=lambda source: None)
    events, frames = [], []
    functional._validate_source = lambda: events.append("source")
    functional._validate_state = frames.append
    functional.extract = lambda: SimpleNamespace(
        state={
            "populations": {
                "left": left.functional.extract().state,
                "right": right.functional.extract().state,
            },
            "synapses": {},
            "continuous_synapses": {},
            "netstim": None,
            "clock": {
                "t": network.t,
                "origin": network._clock_origin,
                "step": network._clock_step,
                "duration_remainder": network._duration_remainder,
            },
        }
    )
    callbacks = FunctionalCallbacks(functional, {"view": _NetworkViews()})
    return SimpleNamespace(
        population=left.population,
        network=network,
        functional=functional,
        callbacks=callbacks,
        events=events,
        frames=frames,
    )


def test_network_session_preserves_cross_population_aliases_and_local_auxiliary():
    case = _network_case()
    session = _bind_native_callbacks(case.network, case.callbacks, None)
    assert case.events == ["source"] and len(case.frames) == 1
    snapshot = session._snapshot()
    assert snapshot["clock"]["t"] is snapshot["populations"]["left"]["clock"]["t"]
    assert snapshot["clock"]["t"] is not case.network.t
    state, parts = session.start()
    assert state["view"]["left"] is state["view"]["right"]
    assert state["view"]["left"] is not case.population.v
    with torch.no_grad():
        case.population.v.add_(1.0)
    updated, emitted = session.update(state)
    assert updated["view"]["left"] is updated["view"]["right"]
    session.collect(updated, parts, session.stack([emitted]))
    result = session.finish(updated, parts)
    assert result["view"].shape == (2, 1, 2)
    assert case.events == ["source"] and len(case.frames) == 1
    gradient = torch.autograd.grad(result["view"].sum(), case.population.v)[0]
    torch.testing.assert_close(
        gradient, torch.tensor([[6.0, 10.0]], dtype=torch.float64)
    )


def test_network_session_rejects_wrong_owner_and_lost_native_buffer_before_hooks():
    case = _network_case()
    for wrong_owner in (case.population, SimpleNamespace()):
        with pytest.raises(FunctionalizationError, match="exact source Network"):
            _bind_native_callbacks(wrong_owner, case.callbacks, None)
    assert case.events == [] and case.frames == []
    del case.population.v_alias
    with pytest.raises(
        FunctionalizationError, match="state binding lost runtime tensor.*v_alias"
    ):
        _bind_native_callbacks(case.network, case.callbacks, None)
    assert case.events == ["source"] and case.frames == []


def test_network_metadata_order_matches_extraction_without_copying_tensor_leaves():
    case = _network_case()
    expected = case.functional.extract().state
    for name, plan in case.functional.populations.items():
        value = case.network.populations[name].v
        random = {"rng": {}, "values": {"sample": value}}
        plan._poisson_plan = SimpleNamespace(entries=())
        plan._random_plan = SimpleNamespace(
            extract=lambda population, random=random: ({}, random)
        )
        expected["populations"][name]["stimulation"] = {"schedules": {}}
        expected["populations"][name]["random"] = random
    session = _bind_native_callbacks(case.network, case.callbacks, None)
    actual = session.binding.state()
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    assert all(
        observed is wanted
        for observed, wanted in zip(actual_leaves, expected_leaves, strict=True)
    )
    for values in actual["populations"].values():
        assert tuple(values)[-2:] == ("stimulation", "random")
