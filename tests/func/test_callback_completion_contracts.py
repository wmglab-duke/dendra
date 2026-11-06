"""Host and fixed rollouts preserve callback completion boundaries."""

from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from dendra.func import _callbacks, _runners
from dendra.func._callbacks import FunctionalCallbackResults
from dendra.func._population import FunctionalPopulation
from dendra.func._types import FunctionalizationError

pytestmark = pytest.mark.cpu

_ENDPOINTS = ("host", "rollout")
_PHASES = ("concatenate", "capture", "finalize", "materialize", "envelope")


def _case(endpoint, monkeypatch, *, auxiliary="mapping", first_failure=None):
    events = []
    raw = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    carry = {"first": raw, "alias": raw}
    captured = {**carry, "schema": object()}
    initial_part = {"trace": raw.unsqueeze(0)}
    parts = [initial_part]
    errors = {name: RuntimeError(name) for name in _PHASES}
    stop = len(_PHASES) if first_failure is None else _PHASES.index(first_failure)
    case = SimpleNamespace(
        raw=raw,
        carry=carry,
        captured=captured,
        events=events,
        errors=errors,
        stacked=None,
        outputs=None,
        materialized=None,
        received_parts=None,
    )

    def phase(name):
        events.append(name)
        # Later stages would also fail; the first failure must win.
        if _PHASES.index(name) >= stop:
            raise errors[name]

    def concatenate(actual_parts):
        phase("concatenate")
        assert len(actual_parts) == 1 and actual_parts[0] is initial_part
        case.received_parts = actual_parts
        case.stacked = {"trace": initial_part["trace"].square()}
        return case.stacked

    def capture(actual_carry, stacked):
        phase("capture")
        assert actual_carry is carry and stacked is case.stacked
        return captured

    def finalize(actual_carry, stacked):
        phase("finalize")
        assert actual_carry is captured and stacked is case.stacked
        case.outputs = {"trace": stacked["trace"], "state_alias": captured["first"]}
        return case.outputs

    def materialize(actual_carry):
        phase("materialize")
        assert actual_carry is captured
        value = captured["first"].clone()
        case.materialized = {"first": value, "alias": value}
        return case.materialized

    def envelope(*, state, outputs):
        phase("envelope")
        assert state is case.materialized and outputs is case.outputs
        return FunctionalCallbackResults(state=state, outputs=outputs)

    callbacks = SimpleNamespace(
        _concatenate=concatenate,
        _capture_emission_schemas=capture,
        _finalize=finalize,
        _materialize_state=materialize,
        _validate_owner=lambda owner: None,
        _validate_runtime_dt_tensor=lambda dt: None,
        _initialize=lambda state: (carry, initial_part),
        _stack=lambda emissions: initial_part if emissions else None,
        _has_emissions=lambda emitted: emitted is not None,
    )
    # Old host code keeps an imported constructor; fixed rollouts import it
    # at invocation. Patch both locations so baseline bodies observe it too.
    monkeypatch.setattr(_callbacks, "FunctionalCallbackResults", envelope)
    monkeypatch.setattr(_runners, "FunctionalCallbackResults", envelope, raising=False)
    if auxiliary == "mapping":
        case.auxiliary = MappingProxyType({"external": raw})
    elif auxiliary == "none":
        case.auxiliary = None
    elif auxiliary == "reserved":
        case.auxiliary = {"callbacks": object()}
    else:
        assert auxiliary == "invalid"
        case.auxiliary = object()

    state = {"integrator": {"v": raw}}

    def invoke():
        if endpoint == "host":
            return _runners._attach_callback_results(
                case.auxiliary, callbacks, carry, parts
            )
        population = SimpleNamespace(
            _assemble_bound_inputs=lambda parameters,
            state,
            ve,
            intra,
            steps,
            dt,
            *,
            sampling_times=None: (
                ve,
                intra,
            ),
            _transition_rollout_values=lambda *args: (state, case.auxiliary),
        )
        final, result = FunctionalPopulation._execute_rollout_values_with_callbacks(
            population,
            {},
            {"integrator": {"dt": raw.new_tensor(0.25)}},
            state,
            None,
            None,
            0,
            callbacks,
            None,
        )
        assert final is state
        return result

    case.invoke, case.parts = invoke, parts
    return case


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("auxiliary", ("none", "mapping"))
def test_completion_preserves_phase_order_initial_parts_forwarding_aliases_and_gradients(
    endpoint, auxiliary, monkeypatch
):
    case = _case(endpoint, monkeypatch, auxiliary=auxiliary)
    result = case.invoke()
    assert case.events == list(_PHASES)
    completed = result["callbacks"]
    assert isinstance(completed, FunctionalCallbackResults)
    assert completed.state is case.materialized
    assert completed.outputs is case.outputs
    assert completed.state["first"] is completed.state["alias"]
    assert completed["state_alias"] is case.raw
    assert completed["trace"] is case.stacked["trace"]
    if endpoint == "host":
        assert case.received_parts is case.parts
    if auxiliary == "mapping":
        assert result["external"] is case.raw
        assert "callbacks" not in case.auxiliary
        assert result is not case.auxiliary
    else:
        assert tuple(result) == ("callbacks",)
    loss = (
        completed["trace"].sum()
        + completed["state_alias"].sum()
        + completed.state["first"].sum()
    )
    gradient = torch.autograd.grad(loss, case.raw)[0]
    torch.testing.assert_close(gradient, 2 * case.raw + 2)
    torch.testing.assert_close(case.raw, torch.tensor([2.0, 3.0], dtype=torch.float64))


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("first_failure", _PHASES)
def test_first_completion_failure_stops_later_phases(
    endpoint, first_failure, monkeypatch
):
    case = _case(endpoint, monkeypatch, first_failure=first_failure)
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.errors[first_failure]
    assert case.events == list(_PHASES[: _PHASES.index(first_failure) + 1])


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
@pytest.mark.parametrize("auxiliary", ("invalid", "reserved"))
def test_each_callers_auxiliary_errors_precede_all_completion_work(
    endpoint, auxiliary, monkeypatch
):
    case = _case(
        endpoint, monkeypatch, auxiliary=auxiliary, first_failure="concatenate"
    )
    error = TypeError if auxiliary == "invalid" else FunctionalizationError
    with pytest.raises(error) as caught:
        case.invoke()
    if auxiliary == "invalid":
        expected = (
            "callback-enabled steps must return a Mapping or None as auxiliary"
            if endpoint == "host"
            else "callback-enabled rollouts require a Mapping or None auxiliary"
        )
    else:
        expected = (
            "step auxiliary output uses the reserved 'callbacks' key"
            if endpoint == "host"
            else "rollout auxiliary output uses the reserved 'callbacks' key"
        )
    assert str(caught.value) == expected
    assert case.events == []


def test_host_without_callbacks_returns_arbitrary_auxiliary_unchanged():
    auxiliary = object()
    assert _runners._attach_callback_results(auxiliary, None, None, None) is auxiliary
