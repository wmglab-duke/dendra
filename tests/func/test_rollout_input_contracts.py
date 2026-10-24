"""Strict rollout entry points reject invalid packets before numerical work."""

import pytest
import torch

from dendra.func._lowered import _LoweredPopulationChunk
from tests.func.test_poisson_stimulation import STEPS, _clone, _close
from tests.func.test_stimulation_rng_sharing import (
    _draw_input,
)
from tests.func.test_stimulation_rng_sharing import _environment as _environment
from tests.func.test_stimulation_rng_sharing import (
    _lower,
    _population,
    _TransitionGuardLeak,
)

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("entry", ("lowered", "compiled", "atomic"))
@pytest.mark.parametrize(
    "fault,message",
    (
        ("drive", "ve must use cpu/torch.float64"),
        ("noise", "Runtime NOISE/SDE draw names do not match"),
        ("stimulation", "Poisson stimulation uniforms must be finite"),
        ("shared_frame", "different end frames"),
    ),
)
def test_strict_rollout_rejection_order_precedes_numerical_execution(
    entry, fault, message, monkeypatch
):
    source, _, pulse = _population(transition_guard=True)
    functional, tensors = _lower(source)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    inputs = _draw_input(functional, tensors.state, STEPS)

    changed_rng = _clone(inputs.stimulation.rng)
    # The final frame still agrees. Every timestep's shared frame belongs to
    # the packet contract, including an earlier conflicting suffix.
    next(iter(changed_rng.values()))[0, 0] ^= 1
    inputs = inputs._replace(stimulation=inputs.stimulation._replace(rng=changed_rng))
    if fault in {"drive", "noise"}:
        inputs = inputs._replace(
            noise=inputs.noise._replace(draws={}),
            stimulation=inputs.stimulation._replace(draws={}),
        )
    if fault == "drive":
        inputs = inputs._replace(
            ve=torch.zeros((STEPS, *functional.shape), dtype=torch.float32)
        )
    elif fault == "stimulation":
        draws = _clone(inputs.stimulation.draws)
        next(iter(draws.values())).reshape(-1)[0] = -0.1
        inputs = inputs._replace(stimulation=inputs.stimulation._replace(draws=draws))

    if entry == "lowered":
        kernel = _LoweredPopulationChunk(functional, steps=STEPS)

        def execute():
            return kernel(
                *kernel.bind(tensors.parameters, prepared, tensors.state, inputs)
            )

    elif entry == "compiled":
        kernel = functional.compile_rollout_chunk(STEPS, disable=True)

        def execute():
            return kernel(tensors.parameters, prepared, tensors.state, inputs)

    else:

        def execute():
            return functional.prepare_and_rollout(
                tensors.parameters,
                tensors.constants,
                tensors.state,
                inputs,
                steps=STEPS,
            )

    frozen_tensors, frozen_inputs = _clone(tensors), _clone(inputs)
    source_rng = pulse.generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    monkeypatch.setattr(_TransitionGuardLeak, "FAIL_ON_ADVANCE", True)
    with pytest.raises(ValueError, match=message):
        execute()
    _close(tensors, frozen_tensors)
    _close(inputs, frozen_inputs)
    _close(functional.extract(), frozen_tensors)
    _close(pulse.generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


def test_compiled_packet_rejection_retains_the_prepared_weak_cache(monkeypatch):
    source, _, _ = _population(transition_guard=True)
    functional, tensors = _lower(source)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    inputs = _draw_input(functional, tensors.state, STEPS)
    expected, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, inputs, steps=STEPS
    )
    kernel = functional.compile_rollout_chunk(STEPS, disable=True)
    calls = {"full": 0, "freshness": 0}
    validate_prepared = functional._validate_prepared
    validate_freshness = functional._validate_prepared_freshness

    def count_prepared(candidate, parameters):
        calls["full"] += 1
        return validate_prepared(candidate, parameters)

    def count_freshness(candidate, parameters):
        calls["freshness"] += 1
        return validate_freshness(candidate, parameters)

    monkeypatch.setattr(functional, "_validate_prepared", count_prepared)
    monkeypatch.setattr(functional, "_validate_prepared_freshness", count_freshness)
    malformed = inputs._replace(noise=inputs.noise._replace(draws={}))
    monkeypatch.setattr(_TransitionGuardLeak, "FAIL_ON_ADVANCE", True)
    with pytest.raises(ValueError, match="Runtime NOISE/SDE draw names do not match"):
        kernel(tensors.parameters, prepared, tensors.state, malformed)
    assert calls == {"full": 1, "freshness": 1}

    monkeypatch.setattr(_TransitionGuardLeak, "FAIL_ON_ADVANCE", False)
    actual, _ = kernel(tensors.parameters, prepared, tensors.state, inputs)
    # The failed input call already qualified this live preparation. Reusing
    # it checks freshness without repeating full prepared-schema validation.
    assert calls == {"full": 1, "freshness": 2}
    _close(actual, expected)
