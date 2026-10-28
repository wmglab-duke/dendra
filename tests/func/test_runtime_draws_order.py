"""Canonical requests retain native order even for independent generators."""

from types import MethodType

import pytest
import torch

from dendra.func import _stimulation_rng
from dendra.models.rng import RNGModule
from tests.func.test_stimulation_rng_sharing import _environment as _environment
from tests.func.test_stimulation_rng_sharing import (
    _lower,
    _population,
    _step_native,
)

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("role", ("mechanism", "extra"))
def test_independent_runtime_streams_follow_native_request_order(monkeypatch, role):
    source, extra, pulse = _population(role, repeated=True)
    native, native_extra, native_pulse = _population(role, repeated=True)
    pulse.generator = torch.Generator().manual_seed(777)
    native_pulse.generator = torch.Generator().manual_seed(777)
    functional, tensors = _lower(source, extra)
    assert not functional._stimulation_rng_plan.borrowed
    expected, actual = [], []
    original_normal = RNGModule.randn
    original_poisson = native_pulse.fn
    owned_fn = "fn" in vars(native_pulse)

    def normal(self, *args, **kwargs):
        expected.append("normal")
        return original_normal(self, *args, **kwargs)

    def poisson(self, times):
        del self
        expected.append("poisson")
        return original_poisson(times)

    with monkeypatch.context() as observed:
        observed.setattr(RNGModule, "randn", normal)
        observed.setattr(native_pulse, "fn", MethodType(poisson, native_pulse))
        _step_native(native, native_extra)
    if not owned_fn:
        delattr(native_pulse, "fn")

    draw = _stimulation_rng.sample_random_parameter_draw
    sample_entry = functional._poisson_plan._sample_entry

    def sampled_noise(*args, **kwargs):
        actual.append("normal")
        return draw(*args, **kwargs)

    def sampled_poisson(*args, **kwargs):
        actual.append("poisson")
        return sample_entry(*args, **kwargs)

    with monkeypatch.context() as observed:
        observed.setattr(
            _stimulation_rng, "sample_random_parameter_draw", sampled_noise
        )
        observed.setattr(functional._poisson_plan, "_sample_entry", sampled_poisson)
        packet = functional.sample_runtime_draws(tensors.state)
    assert actual == expected
    assert "poisson" in expected and "normal" in expected
    actual_state, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    expected_state = _lower(native, native_extra)[1].state
    from tests.func.test_poisson_stimulation import _close

    _close(actual_state, expected_state)
