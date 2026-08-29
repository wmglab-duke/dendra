"""Imperative contracts for structured population initialization transforms."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.initialization import _InitializationTransformHook
from dendra.models.mechanisms import Mechanism, State

DTYPE = torch.float64


class _Add(torch.nn.Module):
    def forward(self, value, increment):
        return (value + increment,)


class _Identity(torch.nn.Module):
    def forward(self, value):
        return (value,)


class _TwoOutputsWithInvalidClock(torch.nn.Module):
    def forward(self, voltage):
        return (torch.full_like(voltage, -41.0), torch.ones(2))


class _InitialVoltageProbe(Mechanism):
    Mechanism.CARRY("initial_voltage")

    def initial_values(self, v, values):
        del values
        return {"initial_voltage": v.clone()}


class _DerivedParameterState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0.0 * x")
    State.GLOBAL(scale=2.0)
    State.DERIVED_BUFFER("q10", "state_workspace")

    def derive_buffers(self):
        return {
            "q10": self.celsius / 10.0,
            "state_workspace": self.scale + self.celsius,
        }

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}


class _DerivedParameterMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_DerivedParameterState)
    Mechanism.GLOBAL(gain=1.0)
    Mechanism.DERIVED_BUFFER("mechanism_workspace")

    def derive_buffers(self):
        return {"mechanism_workspace": self.gain + self.celsius}


class _UpdateRawParameters(torch.nn.Module):
    def forward(self, celsius, gain, scale):
        return (celsius - 20.7, gain * 2.0, scale * 3.0)


def _probe(model):
    return next(
        mechanism
        for mechanism in model.mech.mechanisms.values()
        if isinstance(mechanism, _InitialVoltageProbe)
    )


def test_structured_transforms_preserve_order_with_legacy_hooks():
    model = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    model.insert(_InitialVoltageProbe)
    observed = []
    model.register_pre_initialize_hook(
        lambda population: observed.append(("pre-before", population.v.clone()))
    )
    pre = model.register_pre_initialize_transform(
        "raise_voltage",
        _Add(),
        reads=("state.integrator.v",),
        writes=("state.integrator.v",),
        inputs={"increment": torch.tensor(10.0)},
    )
    model.register_pre_initialize_hook(
        lambda population: observed.append(("pre-after", population.v.clone()))
    )
    model.register_post_initialize_hook(
        lambda population: observed.append(("post-before", population.v.clone()))
    )
    post = model.register_post_initialize_transform(
        "raise_voltage_again",
        _Add(),
        reads=("state.integrator.v",),
        writes=("state.integrator.v",),
        inputs={"increment": torch.tensor(5.0)},
    )
    model.register_post_initialize_hook(
        lambda population: observed.append(("post-after", population.v.clone()))
    )

    model.initialize()

    assert pre.phase == "pre"
    assert post.phase == "post"
    assert pre.reads == post.reads == ("state.integrator.v",)
    assert pre.writes == post.writes == ("state.integrator.v",)
    assert pre.input_names == post.input_names == ("increment",)
    assert isinstance(model.pre_initialize_hooks[1], _InitializationTransformHook)
    assert isinstance(model.post_initialize_hooks[1], _InitializationTransformHook)
    assert [name for name, _ in observed] == [
        "pre-before",
        "pre-after",
        "post-before",
        "post-after",
    ]
    torch.testing.assert_close(observed[0][1], torch.full_like(model.v, -65.0))
    torch.testing.assert_close(observed[1][1], torch.full_like(model.v, -55.0))
    torch.testing.assert_close(observed[2][1], torch.full_like(model.v, -55.0))
    torch.testing.assert_close(observed[3][1], torch.full_like(model.v, -50.0))
    torch.testing.assert_close(
        _probe(model).initial_voltage,
        torch.full_like(model.v, -55.0),
    )
    torch.testing.assert_close(model.v, torch.full_like(model.v, -50.0))
    assert not model.integrator.initialized


def test_transform_output_validation_is_atomic():
    model = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    model.register_pre_initialize_transform(
        "invalid_clock",
        _TwoOutputsWithInvalidClock(),
        reads=("state.integrator.v",),
        writes=("state.integrator.v", "state.clock.t"),
    )

    with pytest.raises(ValueError, match="not broadcastable"):
        model.initialize()

    torch.testing.assert_close(model.v, torch.full_like(model.v, -65.0))
    torch.testing.assert_close(model.t, torch.zeros_like(model.t))
    assert not model.initialized
    assert not model.integrator.initialized


def test_transform_owns_an_independent_explicit_input():
    model = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    model.insert(_InitialVoltageProbe)
    override = torch.tensor([[-51.0, -49.0]], dtype=DTYPE, requires_grad=True)

    action = model.register_pre_initialize_transform(
        "replace_voltage",
        _Identity(),
        writes=("state.integrator.v",),
        inputs={"value": override},
    )
    assert action.phase == "pre"
    assert action.reads == ()
    assert action.writes == ("state.integrator.v",)
    assert action.input_names == ("value",)
    assert action.input_values()[0].data_ptr() != override.data_ptr()
    assert not action.input_values()[0].requires_grad

    with torch.no_grad():
        override.fill_(20.0)
    model.initialize()

    expected = torch.tensor([[-51.0, -49.0]], dtype=DTYPE)
    torch.testing.assert_close(model.v, expected)
    torch.testing.assert_close(_probe(model).initial_voltage, expected)


def test_structured_transform_runs_under_inference_mode():
    model = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    model.register_pre_initialize_transform(
        "raise_voltage",
        _Add(),
        reads=("state.integrator.v",),
        writes=("state.integrator.v",),
        inputs={"increment": torch.tensor(3.0, dtype=DTYPE)},
    )

    with torch.inference_mode():
        model.initialize()

    torch.testing.assert_close(model.v, torch.full_like(model.v, -62.0))
    assert model.initialized
    assert not model.integrator.initialized


def test_registration_rejects_hidden_module_state_and_duplicate_names():
    class Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("hidden", torch.tensor(1.0))

        def forward(self, value):
            return (value + self.hidden,)

    model = dn.Population(N=1, C=1, dtype=DTYPE)
    with pytest.raises(ValueError, match="must be stateless"):
        model.register_pre_initialize_transform(
            "stateful",
            Stateful(),
            reads=("state.integrator.v",),
            writes=("state.integrator.v",),
        )

    model.register_pre_initialize_transform(
        "unique",
        _Add(),
        reads=("state.integrator.v",),
        writes=("state.integrator.v",),
        inputs={"increment": torch.tensor(1.0)},
    )
    with pytest.raises(ValueError, match="unique across both phases"):
        model.register_post_initialize_transform(
            "unique",
            _Add(),
            reads=("state.integrator.v",),
            writes=("state.integrator.v",),
            inputs={"increment": torch.tensor(1.0)},
        )


def test_raw_parameter_outputs_rematerialize_effective_q10_and_derived_buffers():
    model = dn.Population(N=1, C=2, dtype=DTYPE)
    model.insert(_DerivedParameterMechanism)
    model.build()
    prefix = "integrator.mech.mechanisms._DerivedParameterMechanism"
    references = (
        "parameters.celsius_param",
        f"parameters.{prefix}.gain_param",
        f"parameters.{prefix}.DE._DerivedParameterState.scale_param",
    )
    model.register_post_initialize_transform(
        "update_raw_parameters",
        _UpdateRawParameters(),
        reads=references,
        writes=references,
    )

    model.initialize()

    mechanism = model.mech.mechanisms["_DerivedParameterMechanism"]
    state = mechanism.DE["_DerivedParameterState"]
    torch.testing.assert_close(model.celsius, torch.tensor(16.3, dtype=DTYPE))
    torch.testing.assert_close(mechanism.gain, torch.tensor(2.0, dtype=DTYPE))
    torch.testing.assert_close(state.scale, torch.tensor(6.0, dtype=DTYPE))
    torch.testing.assert_close(
        state.q10,
        torch.full_like(state.q10, 1.63),
    )
    torch.testing.assert_close(
        mechanism.mechanism_workspace,
        torch.full_like(mechanism.mechanism_workspace, 18.3),
    )
    torch.testing.assert_close(
        state.state_workspace,
        torch.full_like(state.state_workspace, 22.3),
    )
    assert not model.integrator.initialized
