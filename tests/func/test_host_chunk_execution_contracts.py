"""Ordinary host schedules retain sampling, recording, and failure boundaries."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils import _pytree as pytree

from dendra.func import _runners
from dendra.func._types import (
    FunctionalizationError,
    RolloutInput,
    RuntimeNoiseDraw,
    StimulationDraw,
)

pytestmark = pytest.mark.cpu

_DT = 0.25
_PHASES = (
    "capture",
    "prepare",
    "autodraw",
    "noise",
    "stimulation",
    "assemble",
    "execute",
    "collect",
    "validate",
)


class _Callbacks:
    def __init__(self, record):
        self.record = record

    def _initialize(self, state):
        self.record("initialize")
        return 0, self._emission(state)

    @staticmethod
    def _emission(state):
        return {"value": state["value"], "time": state["clock"]["t"]}

    def _update(self, carry, state, auxiliary):
        self.record("update", float(state["clock"]["t"]))
        return carry + 1, self._emission(state)

    def _stack(self, emissions):
        self.record("stack", len(emissions))
        return {
            key: torch.stack([item[key] for item in emissions])
            for key in ("value", "time")
        }

    def _has_emissions(self, emitted):
        self.record("collect")
        return emitted["value"].numel() != 0

    def _concatenate(self, parts):
        self.record("concatenate", len(parts))
        return {
            key: torch.cat([part[key] for part in parts])
            if parts
            else torch.empty(0, dtype=torch.float64)
            for key in ("value", "time")
        }

    def _capture_emission_schemas(self, carry, emitted):
        self.record("schemas")
        return carry

    def _finalize(self, carry, emitted):
        self.record("finalize", carry)
        return {"trace": emitted}

    def _materialize_state(self, carry):
        self.record("materialize", carry)
        return carry


def _case(monkeypatch, *, callbacks, steps=5, extra="tensor", failure=None):
    """Enter the validated scheduler with inert plans and real eager execution.

    Admission/plan binding is outside this boundary. Sampling, input slicing,
    extra assembly, eager step/callback execution, state checks, and final result
    assembly retain their production implementations.
    """
    events = []
    sentinel = RuntimeError(f"{failure} failed")

    def record(phase, *details):
        events.append((phase, *details))
        if phase == failure:
            raise sentinel

    source = {
        "value": torch.tensor(2.0, dtype=torch.float64, requires_grad=True),
        "clock": {"t": torch.tensor(8.0, dtype=torch.float64)},
        "control": {"duration_remainder": torch.tensor(0.0, dtype=torch.float64)},
    }
    case = SimpleNamespace(
        events=events,
        failure=sentinel,
        state=source,
        steps=steps,
        span=2,
        aux={},
        bad_result=False,
        clock_multiplier=1,
        change_remainder=False,
        gain=torch.tensor(0.5, dtype=torch.float64, requires_grad=True),
        inputs=RolloutInput(
            intra=torch.arange(steps, dtype=torch.float64).requires_grad_()
        ),
    )

    def times(state, count, dt):
        return state["clock"]["t"] + torch.arange(count, dtype=dt.dtype) * dt

    def autodraw(state, inputs, count):
        record("autodraw", float(state["clock"]["t"]), count)
        return inputs

    def noise(state, count):
        record("noise", float(state["clock"]["t"]), count)
        return RuntimeNoiseDraw(
            {"eta": times(state, count, torch.tensor(_DT, dtype=torch.float64))}, {}
        )

    def stimulation(state, count):
        record("stimulation", float(state["clock"]["t"]), count)
        return StimulationDraw(
            {"uniform": times(state, count, torch.tensor(_DT, dtype=torch.float64))}, {}
        )

    def assemble(parameters, constants, accepted, *, temporal_slice, total_steps):
        record(
            "assemble",
            temporal_slice.start,
            temporal_slice.stop,
            total_steps,
            tuple(accepted.tolist()),
        )
        return parameters["gain"] * accepted

    def assemble_waveform(tensors, accepted):
        record("assemble", tuple(accepted.tolist()))
        return tensors.parameters["gain"] * accepted

    extra_plan = SimpleNamespace(
        uses_waveforms=extra == "waveform",
        assemble=assemble,
        assemble_tensors=assemble_waveform,
    )
    extra_tensors = SimpleNamespace(parameters={"gain": case.gain}, constants={})
    functional = SimpleNamespace(
        dt=_DT,
        device=torch.device("cpu"),
        dtype=torch.float64,
        intra=SimpleNamespace(enabled=False),
        extra=SimpleNamespace(enabled=False),
        _noise_plan=SimpleNamespace(runtime=(object(),)),
        _poisson_plan=object(),
        _sample_missing_runtime_draws=autodraw,
        sample_noise_draw_tape=noise,
        sample_stimulation_draw_tape=stimulation,
        _stimulation_times=times,
        _stimulation_rng_plan=SimpleNamespace(validate_packets=lambda *args: None),
        _validate_noise_draw=lambda *args, **kwargs: None,
        _validate_stimulation_draw=lambda *args, **kwargs: None,
        _resolve_rollout_inputs=lambda inputs, count: (inputs.ve, inputs.intra, count),
        _validate_state=lambda state: record("validate", float(state["clock"]["t"])),
        _clone_state_tree=lambda state: pytree.tree_map_only(
            torch.Tensor, torch.clone, state
        ),
    )

    def step(state, inputs):
        record("step", float(state["clock"]["t"]))
        if case.bad_result:
            return state
        torch.testing.assert_close(inputs.noise.draws["eta"], state["clock"]["t"])
        torch.testing.assert_close(
            inputs.stimulation.draws["uniform"], state["clock"]["t"]
        )
        result = {
            "value": state["value"]
            + inputs.intra
            + (0 if inputs.ve is None else inputs.ve),
            "clock": {"t": state["clock"]["t"] + _DT * case.clock_multiplier},
            "control": {
                "duration_remainder": state["control"]["duration_remainder"]
                + (1.0 if case.change_remainder else 0.0)
            },
        }
        return result, case.aux

    def prepared_extra(functional, supplied, inputs):
        return (None, None) if supplied is None else (extra_plan, extra_tensors)

    def prepared_callbacks(functional, selected, carry, state, dt):
        if selected is None:
            return None, None, []
        if carry is None:
            carry, emitted = selected._initialize(state)
            return selected, carry, [selected._stack([emitted])]
        return selected, carry, []

    monkeypatch.setattr(_runners, "_prepare_extra", prepared_extra)
    monkeypatch.setattr(_runners, "_prepare_callback_collection", prepared_callbacks)
    for name, phase in (
        ("_capture_clock", "capture"),
        ("_prepare_chunk_inputs", "prepare"),
        ("_execute_steps", "execute"),
        ("_execute_steps_with_callbacks", "execute"),
    ):
        original = getattr(_runners, name)

        def observed(*args, _original=original, _phase=phase, **kwargs):
            state = args[0] if _phase == "capture" else args[1]
            record(_phase, float(state["clock"]["t"]))
            return _original(*args, **kwargs)

        monkeypatch.setattr(_runners, name, observed)
    selected_callbacks = _Callbacks(record) if callbacks else None

    def invoke(*, callback_state=None):
        return _runners._run_validated_schedule(
            functional,
            step,
            case.state,
            case.inputs,
            expected_remainder=0.0,
            steps=case.steps,
            span=case.span,
            dt=_DT,
            extra=None if extra is None else extra_plan,
            callbacks=selected_callbacks,
            callback_state=callback_state,
        )

    case.invoke = invoke
    return case


@pytest.mark.parametrize("callbacks", (False, True))
def test_short_tail_preserves_clock_sampling_extra_horizon_and_auxiliary_ownership(
    callbacks, monkeypatch
):
    case = _case(monkeypatch, callbacks=callbacks)
    frozen = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), case.state
    )
    final, auxiliary = case.invoke()
    phases = [event[0] for event in case.events if event[0] in _PHASES]
    per_chunk = list(_PHASES)
    if not callbacks:
        per_chunk.remove("collect")
    assert phases == per_chunk * 3
    for name in ("autodraw", "noise", "stimulation"):
        assert [event[1:] for event in case.events if event[0] == name] == [
            (8.0, 2),
            (8.5, 2),
            (9.0, 1),
        ]
    assert [event[1:] for event in case.events if event[0] == "assemble"] == [
        (0, 2, 5, (8.0, 8.25)),
        (2, 4, 5, (8.5, 8.75)),
        (4, 5, 5, (9.0,)),
    ]
    assert final["clock"]["t"].item() == 9.25
    pytree.tree_map(torch.testing.assert_close, case.state, frozen)
    if callbacks:
        assert auxiliary is not case.aux
        trace = auxiliary["callbacks"]["trace"]
        torch.testing.assert_close(trace["time"], torch.arange(6).double() * _DT + 8.0)
        assert trace["value"].shape == (6,)
        assert auxiliary["callbacks"].state == 5
        assert case.events[-4:] == [
            ("concatenate", 4),
            ("schemas",),
            ("finalize", 5),
            ("materialize", 5),
        ]
    else:
        assert auxiliary is case.aux


@pytest.mark.parametrize("callbacks", (False, True))
@pytest.mark.parametrize(
    "phase",
    (
        "capture",
        "prepare",
        "autodraw",
        "noise",
        "stimulation",
        "assemble",
        "execute",
        "validate",
    ),
)
def test_ordinary_chunk_failures_stop_before_later_phases(
    callbacks, phase, monkeypatch
):
    case = _case(monkeypatch, callbacks=callbacks, steps=2, failure=phase)
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.failure
    phases = [event[0] for event in case.events if event[0] in _PHASES]
    expected = list(_PHASES)
    if not callbacks:
        expected.remove("collect")
    assert phases == expected[: expected.index(phase) + 1]
    assert not any(event[0] == "finalize" for event in case.events)


@pytest.mark.parametrize("phase", ("collect", "finalize"))
def test_callback_collection_and_finalization_errors_keep_their_boundaries(
    phase, monkeypatch
):
    case = _case(monkeypatch, callbacks=True, steps=2, failure=phase)
    with pytest.raises(RuntimeError) as caught:
        case.invoke()
    assert caught.value is case.failure
    names = [event[0] for event in case.events]
    assert ("validate" in names) is (phase == "finalize")
    assert "materialize" not in names


@pytest.mark.parametrize("callbacks", (False, True))
@pytest.mark.parametrize("extra", (None, "tensor", "waveform"))
def test_zero_work_keeps_initial_recording_and_lazy_waveforms(
    callbacks, extra, monkeypatch
):
    case = _case(monkeypatch, callbacks=callbacks, steps=0, extra=extra)
    final, auxiliary = case.invoke()
    assert not any(
        event[0] in _PHASES and event[0] != "assemble" for event in case.events
    )
    assemblies = [event for event in case.events if event[0] == "assemble"]
    assert assemblies == ([("assemble", 0, 0, 0, ())] if extra == "tensor" else [])
    assert final is not case.state
    for result, source in zip(
        pytree.tree_leaves(final), pytree.tree_leaves(case.state), strict=True
    ):
        torch.testing.assert_close(result, source)
        assert result.data_ptr() != source.data_ptr()
    if callbacks:
        assert auxiliary["callbacks"]["trace"]["time"].tolist() == [8.0]
        assert auxiliary["callbacks"].state == 0
    else:
        assert auxiliary is None


@pytest.mark.parametrize("callbacks", (False, True))
@pytest.mark.parametrize("fault", ("result", "clock", "remainder"))
def test_step_and_host_clock_validation_messages_are_preserved(
    callbacks, fault, monkeypatch
):
    case = _case(monkeypatch, callbacks=callbacks, steps=1)
    if fault == "result":
        case.bad_result = True
        error, message = TypeError, "step must return exactly (next_state, auxiliary)"
    elif fault == "clock":
        case.clock_multiplier = 2
        error, message = (
            FunctionalizationError,
            "step clock advance does not match runner dt:",
        )
    else:
        case.change_remainder = True
        error, message = (
            FunctionalizationError,
            "step changed host-owned state control.duration_remainder",
        )
    with pytest.raises(error) as caught:
        case.invoke()
    assert str(caught.value).startswith(message)


def test_callback_trace_retains_gradients_across_short_tail_and_zero_step_resume(
    monkeypatch,
):
    case = _case(monkeypatch, callbacks=True)
    final, auxiliary = case.invoke()
    trace = auxiliary["callbacks"]["trace"]
    gradients = torch.autograd.grad(
        final["value"] + trace["value"].sum(),
        (case.state["value"], case.gain, case.inputs.intra),
    )
    weights = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0], dtype=torch.float64)
    accepted = torch.arange(5, dtype=torch.float64) * _DT + 8.0
    torch.testing.assert_close(gradients[0], torch.tensor(7.0, dtype=torch.float64))
    torch.testing.assert_close(gradients[1], (weights * accepted).sum())
    torch.testing.assert_close(gradients[2], weights)
    case.state, case.steps = final, 0
    case.inputs = RolloutInput(intra=case.inputs.intra[:0])
    resumed, outputs = case.invoke(callback_state=auxiliary["callbacks"].state)
    assert resumed["clock"]["t"].item() == 9.25
    assert outputs["callbacks"]["trace"]["time"].shape == (0,)
    assert outputs["callbacks"].state == 5


def test_plain_schedule_returns_opaque_auxiliary_without_copying_or_validation(
    monkeypatch,
):
    case = _case(monkeypatch, callbacks=False, steps=1)
    case.aux = object()
    _final, auxiliary = case.invoke()
    assert auxiliary is case.aux


@pytest.mark.parametrize(
    "auxiliary, message",
    (
        (object(), "callback-enabled steps must return a Mapping or None as auxiliary"),
        (
            {"callbacks": object()},
            "step auxiliary output uses the reserved 'callbacks' key",
        ),
    ),
)
def test_callback_auxiliary_rejection_follows_chunk_validation(
    auxiliary, message, monkeypatch
):
    case = _case(monkeypatch, callbacks=True, steps=1)
    case.aux = auxiliary
    with pytest.raises((TypeError, FunctionalizationError)) as caught:
        case.invoke()
    assert str(caught.value) == message
    names = [event[0] for event in case.events]
    assert names[-1] == "validate"
    assert "finalize" not in names
