"""Hard waveform gates retain their intended sigmoid surrogate derivatives."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.stim.waveform.implementations import _rect_gate

DTYPE = torch.float64
PARAMETERS = {"amp", "freq", "phase", "delay", "off", "off_after", "tau"}


def _waveform(kind, **kwargs):
    with dn.ctx(REQUIRE_GRAD=1):
        return kind(**kwargs).to(dtype=DTYPE)


def _clone_parameters(waveform):
    return {
        name: value.detach().clone().requires_grad_()
        for name, value in waveform.named_parameters()
    }


def _reference_component(t, parameters, trig, stop_kind=None):
    """Independent scalar component with an explicitly selected end edge.

    These are straight-through derivatives, so finite differences of the
    intentionally hard forward waveform are not an appropriate oracle.
    """
    delay = parameters["delay"]
    tau = parameters["tau"].clamp(min=1e-6)
    soft = torch.sigmoid((t - delay) / tau)
    hard = t >= delay
    if stop_kind is not None:
        stop = (
            parameters["off"]
            if stop_kind == "absolute"
            else delay + parameters["off_after"]
        )
        soft = soft * torch.sigmoid((stop - t) / tau)
        hard = hard & (t < stop)
    gate = hard.to(t.dtype) + (soft - soft.detach())
    carrier = trig(
        2 * torch.pi * parameters["freq"] * (t - delay) + parameters["phase"]
    )
    return parameters["amp"] * gate * carrier


def _gradients(value, parameters, weights):
    values = torch.autograd.grad(
        (value * weights).sum(), tuple(parameters.values()), allow_unused=True
    )
    return {
        name: torch.zeros_like(parameter) if gradient is None else gradient
        for (name, parameter), gradient in zip(parameters.items(), values, strict=True)
    }


def _assert_finite_close(actual, expected):
    actual_leaves, actual_spec = pytree.tree_flatten(actual)
    expected_leaves, expected_spec = pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for observed, reference in zip(actual_leaves, expected_leaves, strict=True):
        assert torch.isfinite(observed).all()
        assert torch.isfinite(reference).all()
        torch.testing.assert_close(observed, reference, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("kind,trig", [(dn.sin, torch.sin), (dn.cos, torch.cos)])
def test_default_infinite_stop_has_finite_named_parameter_gradients(kind, trig):
    waveform = _waveform(kind, amp=1.3, freq=0.7, phase=0.4, delay=0.2, tau=0.07)
    parameters = dict(waveform.named_parameters())
    assert parameters.keys() == PARAMETERS
    assert torch.isposinf(parameters["off"]) and torch.isposinf(parameters["off_after"])
    reference_parameters = _clone_parameters(waveform)
    t = torch.tensor([0.11, 0.18, 0.2, 0.23, 0.33], dtype=DTYPE)
    weights = torch.tensor([0.3, -0.2, 0.7, 1.1, 0.4], dtype=DTYPE)
    actual = waveform(t)
    reference = _reference_component(t, reference_parameters, trig)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    observed = _gradients(actual, parameters, weights)
    expected = _gradients(reference, reference_parameters, weights)
    _assert_finite_close(observed, expected)
    assert observed["delay"].abs() > 0.1
    assert observed["tau"].abs() > 0.1
    assert observed["off"].item() == observed["off_after"].item() == 0


@pytest.mark.parametrize("kind,trig", [(dn.sin, torch.sin), (dn.cos, torch.cos)])
def test_mixed_batched_stops_preserve_each_components_reference_gradients(kind, trig):
    waveform = _waveform(
        kind,
        amp=torch.tensor([[1.3, 0.7, -0.4], [0.8, -0.6, 1.1]], dtype=DTYPE),
        freq=torch.tensor([[0.7, 0.4, 1.2], [0.9, 0.5, 0.3]], dtype=DTYPE),
        phase=torch.tensor([[0.4, -0.1, 0.3], [-0.2, 0.7, 0.1]], dtype=DTYPE),
        delay=torch.tensor([[0.2, 0.1, 0.12], [0.08, 0.11, 0.15]], dtype=DTYPE),
        off=torch.tensor(
            [[torch.inf, 0.31, torch.inf], [0.51, 0.37, torch.inf]], dtype=DTYPE
        ),
        off_after=torch.tensor(
            [[torch.inf, torch.inf, 0.19], [0.22, 0.75, torch.inf]], dtype=DTYPE
        ),
        tau=torch.tensor([[0.07, 0.04, 0.06], [0.05, 0.08, 0.09]], dtype=DTYPE),
    )
    parameters = dict(waveform.named_parameters())
    reference_parameters = _clone_parameters(waveform)
    # The finite active constraint is chosen explicitly; the reference never
    # evaluates an infinite division, even in a branch that is later masked.
    stop_kinds = ((None, "absolute", "relative"), ("relative", "absolute", None))
    t = torch.tensor([0.06, 0.12, 0.2, 0.27, 0.31, 0.39, 0.5], dtype=DTYPE)
    reference = torch.stack(
        [
            sum(
                _reference_component(
                    t,
                    {
                        name: value[batch, component]
                        for name, value in reference_parameters.items()
                    },
                    trig,
                    stop_kinds[batch][component],
                )
                for component in range(3)
            )
            for batch in range(2)
        ]
    )
    actual = waveform(t)
    torch.testing.assert_close(actual, reference, rtol=2e-15, atol=2e-15)
    weights = torch.arange(1, 15, dtype=DTYPE).reshape(2, 7) / 10
    observed = _gradients(actual, parameters, weights)
    expected = _gradients(reference, reference_parameters, weights)
    _assert_finite_close(observed, expected)
    for name in ("delay", "tau"):
        assert torch.all(observed[name].abs() > 1e-4)
    for batch, component in ((0, 0), (1, 2)):
        assert observed["off"][batch, component] == 0
        assert observed["off_after"][batch, component] == 0
    assert observed["off"][0, 1].abs() > 1e-4
    assert observed["off_after"][0, 2].abs() > 1e-4


@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize("tau_value", [0.04, 1e-8])
def test_finite_gate_boundaries_and_surrogate_gradients_are_unchanged(
    inclusive, tau_value
):
    t = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], dtype=DTYPE)
    parameters = {
        "start": torch.tensor(0.1, dtype=DTYPE, requires_grad=True),
        "stop": torch.tensor(0.3, dtype=DTYPE, requires_grad=True),
        "tau": torch.tensor(tau_value, dtype=DTYPE, requires_grad=True),
    }
    actual = _rect_gate(t, **parameters, inclusive_stop=inclusive)
    torch.testing.assert_close(
        actual, torch.tensor([0, 1, 1, int(inclusive), 0], dtype=DTYPE), rtol=0, atol=0
    )
    original = {
        name: value.detach().clone().requires_grad_()
        for name, value in parameters.items()
    }
    tau = original["tau"].clamp(min=1e-6)
    soft = torch.sigmoid((t - original["start"]) / tau) * torch.sigmoid(
        (original["stop"] - t) / tau
    )
    weights = torch.tensor([0.4, 0.8, -0.3, 1.2, 0.6], dtype=DTYPE)
    _assert_finite_close(
        _gradients(actual, parameters, weights), _gradients(soft, original, weights)
    )


@pytest.mark.parametrize("kind,trig", [(dn.sin, torch.sin), (dn.cos, torch.cos)])
@torch_compiler_warning_context()
def test_infinite_stop_supports_forward_reverse_jacobians_and_second_derivatives(
    kind, trig
):
    waveform = _waveform(kind, amp=1.3, freq=0.7, phase=0.4, delay=0.2, tau=0.07)
    parameters = _clone_parameters(waveform)
    t = torch.tensor([0.16, 0.2, 0.23], dtype=DTYPE)

    def actual(p):
        return torch.func.functional_call(waveform, p, (t,))

    def reference(p):
        return _reference_component(t, p, trig)

    expected_jacobian = torch.func.jacrev(reference)(parameters)
    _assert_finite_close(torch.func.jacrev(actual)(parameters), expected_jacobian)
    _assert_finite_close(torch.func.jacfwd(actual)(parameters), expected_jacobian)
    weights = torch.tensor([0.3, 0.7, 1.1], dtype=DTYPE)
    actual_hessian = torch.func.jacfwd(
        torch.func.jacrev(lambda p: (actual(p) * weights).sum())
    )(parameters)
    expected_hessian = torch.func.jacfwd(
        torch.func.jacrev(lambda p: (reference(p) * weights).sum())
    )(parameters)
    _assert_finite_close(actual_hessian, expected_hessian)
    assert actual_hessian["tau"]["tau"].abs() > 0.1


@pytest.mark.parametrize(
    "start_value,stop_value,limit",
    [
        (-torch.inf, torch.inf, 1.0),
        (torch.inf, torch.inf, 0.0),
        (0.1, -torch.inf, 0.0),
        (-torch.inf, -torch.inf, 0.0),
    ],
)
def test_infinite_endpoint_limits_have_zero_finite_gradients(
    start_value, stop_value, limit
):
    t = torch.tensor([0.0, 0.1, 0.3], dtype=DTYPE)
    parameters = {
        "start": torch.tensor(start_value, dtype=DTYPE, requires_grad=True),
        "stop": torch.tensor(stop_value, dtype=DTYPE, requires_grad=True),
        "tau": torch.tensor(0.07, dtype=DTYPE, requires_grad=True),
    }
    actual = _rect_gate(t, **parameters)
    torch.testing.assert_close(actual, torch.full_like(t, limit), rtol=0, atol=0)
    observed = _gradients(actual, parameters, torch.ones_like(t))
    _assert_finite_close(
        observed, {name: torch.zeros_like(value) for name, value in parameters.items()}
    )
