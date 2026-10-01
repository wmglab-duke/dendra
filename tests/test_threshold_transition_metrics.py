"""Gaussian composition of hard transition roots, including signed jumps."""

import math

import pytest
import torch

from dendra.models.analysis.hard_transition_table import discover_hard_transitions
from dendra.models.analysis.threshold_trace_tangent import trace_tangent_threshold_proxy
from dendra.models.analysis.threshold_transition_metrics import (
    gaussian_expected_activation,
    gaussian_expected_hard_from_transition_roots,
)


def _cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2.0)))


def _pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x.square()) / math.sqrt(2 * math.pi)


def test_one_recruitment_root_gives_probability_and_all_parameter_slopes():
    q = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    mu = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)
    root = 0.3 + 0.2 * q[0] - 0.1 * q[1]
    sigma = 0.15
    value = gaussian_expected_hard_from_transition_roots(
        mu,
        root.reshape(1),
        torch.ones(1, dtype=q.dtype),
        0.0,
        sigma,
    )
    z = (mu.detach() - root.detach()) / sigma
    torch.testing.assert_close(value.detach(), _cdf(z), atol=1e-15, rtol=0)

    dmu, dq = torch.autograd.grad(value, (mu, q))
    density = _pdf(z) / sigma
    torch.testing.assert_close(dmu, density, atol=1e-14, rtol=0)
    torch.testing.assert_close(
        dq,
        -density * torch.tensor([0.2, -0.1], dtype=q.dtype),
        atol=1e-14,
        rtol=0,
    )


def test_activation_convenience_wrapper_matches_one_transition_table():
    threshold = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    center = torch.tensor([0.31, 0.42], dtype=torch.float64, requires_grad=True)
    sigma = 0.08
    probability = gaussian_expected_activation(center, threshold, sigma)
    expected = gaussian_expected_hard_from_transition_roots(
        center,
        threshold.reshape(1),
        torch.ones(1, dtype=threshold.dtype),
        0.0,
        sigma,
    )
    torch.testing.assert_close(probability, expected, atol=0, rtol=0)
    dcenter, dthreshold = torch.autograd.grad(
        probability.sum(),
        (center, threshold),
    )
    z = (center.detach() - threshold.detach()) / sigma
    density = _pdf(z) / sigma
    torch.testing.assert_close(dcenter, density, atol=1e-14, rtol=0)
    torch.testing.assert_close(dthreshold, -density.sum(), atol=1e-14, rtol=0)


def test_activation_convenience_wrapper_validates_threshold():
    with pytest.raises(TypeError, match="threshold must be a tensor"):
        gaussian_expected_activation(0.2, 0.3, 0.1)
    with pytest.raises(ValueError, match="threshold must be scalar"):
        gaussian_expected_activation(
            0.2,
            torch.tensor([0.3, 0.4], dtype=torch.float64),
            0.1,
        )


def test_hard_forward_trace_tangent_root_composes_into_expected_hard_gradient():
    q = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True)
    hard_root = 0.35 + 0.1 * q[0] - 0.04 * q[1]
    probe = torch.tensor(0.3, dtype=q.dtype)
    amplitude_tangent = torch.tensor([2.0, -3.0], dtype=q.dtype)
    voltage = amplitude_tangent * (probe - hard_root)
    root_result = trace_tangent_threshold_proxy(
        hard_root.detach(),
        probe,
        voltage,
        amplitude_tangent,
        min_tangent_norm=1e-9,
    )
    assert root_result.valid
    mu = torch.tensor(0.4, dtype=q.dtype)
    sigma = 0.08
    expected = gaussian_expected_hard_from_transition_roots(
        mu,
        root_result.proxy.reshape(1),
        torch.ones(1, dtype=q.dtype),
        0.0,
        sigma,
    )
    z = (mu - hard_root.detach()) / sigma
    torch.testing.assert_close(expected.detach(), _cdf(z), atol=1e-15, rtol=0)
    (slope,) = torch.autograd.grad(expected, q)
    torch.testing.assert_close(
        slope,
        -_pdf(z) / sigma * torch.tensor([0.1, -0.04], dtype=q.dtype),
        atol=1e-14,
        rtol=0,
    )


def test_signed_multiple_roots_match_complete_hard_step_quadrature_and_gradients():
    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    mu = torch.tensor([0.1, 0.3], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)
    roots = torch.stack((-0.4 + 0.05 * q, 0.2 - 0.1 * q, 0.7 + 0.03 * q))
    jumps = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)
    left = 0.2
    values = gaussian_expected_hard_from_transition_roots(
        mu,
        roots,
        jumps,
        left,
        sigma,
    )

    z = (mu.detach()[:, None] - roots.detach()[None, :]) / sigma.detach()
    expected = left + (_cdf(z) * jumps).sum(-1)
    torch.testing.assert_close(values.detach(), expected, atol=1e-15, rtol=0)

    # Independently evaluate the complete step protocol over normal quantiles.
    n = 20001
    quantile = (torch.arange(n, dtype=mu.dtype) + 0.5) / n
    noise = math.sqrt(2.0) * torch.erfinv(2 * quantile - 1)
    amplitudes = mu.detach()[:, None] + sigma.detach() * noise[None, :]
    hard = left + ((amplitudes[..., None] >= roots.detach()) * jumps).sum(-1)
    torch.testing.assert_close(values.detach(), hard.mean(-1), atol=8e-5, rtol=0)

    dmu, dq, dsigma = torch.autograd.grad(values.sum(), (mu, q, sigma))
    density = _pdf(z) / sigma.detach()
    torch.testing.assert_close(dmu, (density * jumps).sum(-1), atol=1e-14, rtol=0)
    root_slopes = torch.tensor([0.05, -0.1, 0.03], dtype=q.dtype)
    torch.testing.assert_close(
        dq,
        -(density * jumps * root_slopes).sum(),
        atol=1e-14,
        rtol=0,
    )
    torch.testing.assert_close(
        dsigma,
        -(density * jumps * z).sum(),
        atol=1e-14,
        rtol=0,
    )


def test_discovered_hard_transition_table_composes_to_analytic_count_expectation():
    def hard_count(amplitude: float) -> float:
        if amplitude < 1.0:
            return 0.0
        if amplitude < 2.0:
            return 2.0
        if amplitude < 2.5:
            return 1.0
        return 3.0

    table = discover_hard_transitions(
        hard_count,
        [0.0, 1.5, 2.25, 3.0],
        tolerance=1e-8,
    )
    assert table.valid
    assert [transition.jump for transition in table.transitions] == [2.0, -1.0, 2.0]
    roots = torch.tensor(
        [transition.midpoint for transition in table.transitions],
        dtype=torch.float64,
    )
    jumps = torch.tensor(
        [transition.jump for transition in table.transitions],
        dtype=roots.dtype,
    )
    mu = torch.tensor(1.8, dtype=roots.dtype, requires_grad=True)
    sigma = 0.35
    value = gaussian_expected_hard_from_transition_roots(
        mu,
        roots,
        jumps,
        table.scan_values[0],
        sigma,
    )
    exact_roots = torch.tensor([1.0, 2.0, 2.5], dtype=roots.dtype)
    z = (mu.detach() - exact_roots) / sigma
    exact_value = (torch.tensor([2.0, -1.0, 2.0], dtype=roots.dtype) * _cdf(z)).sum()
    exact_slope = (
        torch.tensor([2.0, -1.0, 2.0], dtype=roots.dtype) * _pdf(z)
    ).sum() / sigma
    (slope,) = torch.autograd.grad(value, mu)
    torch.testing.assert_close(value.detach(), exact_value, atol=2e-8, rtol=0)
    torch.testing.assert_close(slope, exact_slope, atol=5e-8, rtol=0)


def test_empty_table_has_constant_value_and_zero_center_gradient():
    mu = torch.tensor([-0.2, 0.4], dtype=torch.float64, requires_grad=True)
    empty = torch.empty(0, dtype=mu.dtype)
    result = gaussian_expected_hard_from_transition_roots(
        mu,
        empty,
        empty,
        0.375,
        0.2,
    )
    torch.testing.assert_close(result, torch.full_like(mu, 0.375), rtol=0, atol=0)
    (slope,) = torch.autograd.grad(result.sum(), mu)
    torch.testing.assert_close(slope, torch.zeros_like(mu), rtol=0, atol=0)


def test_python_scalars_preserve_root_dtype_and_tensor_inputs_remain_differentiable():
    root = torch.tensor([0.37], dtype=torch.float64, requires_grad=True)
    jump = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    left = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    out = gaussian_expected_hard_from_transition_roots(
        0.4,
        root,
        jump,
        left,
        0.2,
    )
    assert out.dtype == torch.float64
    assert out.device == root.device
    dr, dj, dl = torch.autograd.grad(out, (root, jump, left))
    z = torch.tensor((0.4 - 0.37) / 0.2, dtype=root.dtype)
    torch.testing.assert_close(dr, (-2.0 * _pdf(z) / 0.2).reshape(1))
    torch.testing.assert_close(dj, _cdf(z).reshape(1))
    assert dl.item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("roots", "jumps", "reason"),
    [
        ([0.3, 0.1], [1.0, -1.0], "strictly increasing"),
        ([0.2, 0.2], [1.0, -1.0], "strictly increasing"),
        ([float("nan")], [1.0], "roots must be finite"),
        ([0.2], [float("inf")], "jumps must be finite"),
    ],
)
def test_invalid_transition_tables_are_rejected(roots, jumps, reason):
    r = torch.tensor(roots, dtype=torch.float64)
    j = torch.tensor(jumps, dtype=torch.float64)
    with pytest.raises(ValueError, match=reason):
        gaussian_expected_hard_from_transition_roots(0.0, r, j, 0.0, 0.1)


def test_shapes_dtypes_and_noise_scale_are_validated():
    roots = torch.tensor([0.1, 0.4], dtype=torch.float64)
    jumps = torch.tensor([1.0, -0.5], dtype=torch.float64)

    def call(r=roots, j=jumps, c=0.0, l=0.0, s=0.2):
        return gaussian_expected_hard_from_transition_roots(c, r, j, l, s)

    with pytest.raises(TypeError, match="must be tensors"):
        call(r=[0.1, 0.4])
    with pytest.raises(ValueError, match="equal-length vectors"):
        call(j=jumps[:1])
    with pytest.raises(ValueError, match="equal-length vectors"):
        call(r=roots.reshape(1, 2), j=jumps.reshape(1, 2))
    with pytest.raises(ValueError, match="floating point"):
        call(j=torch.tensor([1, -1]))
    with pytest.raises(ValueError, match="share device and dtype"):
        call(j=jumps.float())
    with pytest.raises(ValueError, match="share the roots' device and dtype"):
        call(c=torch.tensor(0.0, dtype=torch.float32))
    with pytest.raises(ValueError, match="scalar"):
        call(l=torch.tensor([0.0, 0.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="scalar"):
        call(s=torch.tensor([0.2], dtype=torch.float64))
    for sigma in (0.0, -0.1, float("inf")):
        with pytest.raises(ValueError, match="noise_std"):
            call(s=sigma)
    with pytest.raises(ValueError, match="center must be finite"):
        call(c=float("nan"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tensor_device_mismatch_is_rejected():
    roots = torch.tensor([0.2], dtype=torch.float64, device="cuda")
    jumps = torch.tensor([1.0], dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="share the roots' device and dtype"):
        gaussian_expected_hard_from_transition_roots(
            torch.tensor(0.0, dtype=torch.float64),
            roots,
            jumps,
            0.0,
            0.1,
        )
