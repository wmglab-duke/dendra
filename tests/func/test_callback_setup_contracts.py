"""Host and fixed rollouts share fresh and resumed callback setup semantics."""

from types import SimpleNamespace

import pytest
import torch

from dendra.func import _callbacks, _runners
from dendra.func._population import FunctionalPopulation

pytestmark = pytest.mark.cpu

_ENDPOINTS = ("host", "rollout")
_FRESH_PHASES = ("owner", "dt", "initialize", "stack", "has")
_ROLLOUT_SUFFIX = ("assemble", "transition", "stack-empty", "has-empty", "complete")


def _case(endpoint, monkeypatch, *, mode="emitting", failures=()):
    events = []
    errors = {phase: RuntimeError(phase) for phase in failures}
    raw = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    carry = {"first": raw, "alias": raw}
    state = {"integrator": {"v": raw}}
    if mode in ("emitting", "empty_tensor"):
        emission = raw if mode == "emitting" else raw[:0]
        initial = {"trace": emission}
        initial_part = {"trace": emission.unsqueeze(0)}
    else:
        initial = None if mode == "none" else {}
        initial_part = {"trace": None} if mode == "none" else {}
    empty_part = {"trace": None}
    initial_lists = []
    dt = raw.new_tensor(0.25)

    def phase(name):
        events.append(name)
        if name in errors:
            raise errors[name]

    def assemble(*args, sampling_times=None):
        assert sampling_times is None
        phase("assemble")
        return None, None

    def transition(*args):
        phase("transition")
        return state, {}

    functional = SimpleNamespace(
        _assemble_bound_inputs=assemble,
        _transition_rollout_values=transition,
    )

    class Callbacks(_callbacks.FunctionalCallbacks):
        def _validate_owner(self, owner):
            phase("owner")
            assert owner is functional

        def _validate_runtime_dt(self, value):
            phase("dt")
            assert value == 0.25

        def _validate_runtime_dt_tensor(self, value):
            phase("dt")
            assert value is dt

        def _initialize(self, value):
            phase("initialize")
            assert value is state
            return carry, initial

        def _stack(self, emissions):
            if not emissions:
                phase("stack-empty")
                return empty_part
            phase("stack")
            assert len(emissions) == 1 and emissions[0] is initial
            initial_lists.append(emissions)
            return initial_part

        def _has_emissions(self, part):
            if part is empty_part:
                phase("has-empty")
                return False
            phase("has")
            assert part is initial_part
            return _callbacks.FunctionalCallbacks._has_emissions(part)

        def _validate_state(self, value):
            phase("validate")
            assert value is case.supplied
            return carry

    callbacks = object.__new__(Callbacks)
    callbacks._storage = {}
    case = SimpleNamespace(
        events=events,
        errors=errors,
        raw=raw,
        carry=carry,
        initial_part=initial_part,
        initial_lists=initial_lists,
        supplied=None,
    )

    def complete(actual_callbacks, actual_carry, parts):
        phase("complete")
        assert actual_callbacks is callbacks
        return actual_carry, parts

    monkeypatch.setattr(_callbacks, "_complete_callback_results", complete)
    monkeypatch.setattr(_runners, "_complete_callback_results", complete)

    def invoke(supplied=None):
        case.supplied = supplied
        if endpoint == "host":
            selected, actual_carry, parts = _runners._prepare_callback_collection(
                functional, callbacks, supplied, state, 0.25
            )
            assert selected is callbacks
            return actual_carry, parts
        final, result = FunctionalPopulation._execute_rollout_values_with_callbacks(
            functional,
            {},
            {"integrator": {"dt": dt}},
            state,
            None,
            None,
            0,
            callbacks,
            supplied,
        )
        assert final is state
        return result["callbacks"]

    case.invoke = invoke
    return case


def _expected(endpoint, phases):
    return list(phases + (_ROLLOUT_SUFFIX if endpoint == "rollout" else ()))


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("mode", ("emitting", "none", "empty", "empty_tensor"))
def test_fresh_setup_keeps_initial_parts_and_gradients_without_reusing_lists(
    endpoint, mode, monkeypatch
):
    case = _case(endpoint, monkeypatch, mode=mode)
    version = case.raw._version
    carry, parts = case.invoke()
    assert case.events == _expected(endpoint, _FRESH_PHASES)
    assert carry is case.carry
    assert carry["first"] is carry["alias"] is case.raw
    assert len(parts) == int(mode in ("emitting", "empty_tensor"))
    if parts:
        assert parts[0] is case.initial_part
        assert parts[0]["trace"].requires_grad

    case.events.clear()
    repeated_carry, repeated_parts = case.invoke()
    assert case.events == _expected(endpoint, _FRESH_PHASES)
    assert repeated_carry is carry and repeated_parts is not parts
    assert case.initial_lists[0] is not case.initial_lists[1]
    if repeated_parts:
        assert repeated_parts[0] is parts[0]
    loss = carry["first"].sum() + carry["alias"].sum()
    if parts:
        loss = loss + parts[0]["trace"].sum()
    gradient = torch.autograd.grad(loss, case.raw)[0]
    torch.testing.assert_close(
        gradient, torch.full_like(case.raw, 2 + (mode == "emitting"))
    )
    assert case.raw._version == version


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_falsey_non_none_resume_uses_validated_carry_and_fresh_empty_parts(
    endpoint, monkeypatch
):
    case = _case(endpoint, monkeypatch)
    supplied = {}
    assert not supplied
    carry, parts = case.invoke(supplied)
    assert case.events == _expected(endpoint, ("owner", "dt", "validate"))
    assert carry is case.carry and carry is not supplied
    assert parts == [] and case.initial_lists == []
    case.events.clear()
    repeated_carry, repeated_parts = case.invoke(supplied)
    assert case.events == _expected(endpoint, ("owner", "dt", "validate"))
    assert (
        repeated_carry is carry and repeated_parts == [] and repeated_parts is not parts
    )


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("failure", _FRESH_PHASES)
def test_caller_checks_and_first_setup_failure_precede_later_hooks(
    endpoint, failure, monkeypatch
):
    index = _FRESH_PHASES.index(failure)
    case = _case(endpoint, monkeypatch, failures=_FRESH_PHASES[index:])
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.errors[failure]
    assert case.events == list(_FRESH_PHASES[: index + 1])


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_resume_validation_failure_does_not_initialize_or_execute(
    endpoint, monkeypatch
):
    case = _case(endpoint, monkeypatch, failures=("validate", "initialize"))
    with pytest.raises(RuntimeError) as caught:
        case.invoke({})
    assert caught.value is case.errors["validate"]
    assert case.events == ["owner", "dt", "validate"]


def test_host_absent_callbacks_returns_independent_empty_parts():
    first = _runners._prepare_callback_collection(None, None, None, None, None)
    second = _runners._prepare_callback_collection(None, None, None, None, None)
    assert first == second == (None, None, [])
    assert first[2] is not second[2]


@pytest.mark.parametrize(
    "callbacks,carry,error,message",
    (
        (None, {}, ValueError, "callback_state requires callbacks"),
        (
            object(),
            None,
            TypeError,
            "callbacks must be a FunctionalCallbacks plan created by functional.make_callbacks(...)",
        ),
    ),
)
def test_host_none_and_type_checks_keep_their_original_errors(
    callbacks, carry, error, message
):
    with pytest.raises(error) as caught:
        _runners._prepare_callback_collection(None, callbacks, carry, None, None)
    assert str(caught.value) == message
