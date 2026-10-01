"""Exact threshold-noise expectations of hard spike detection protocols."""

import torch

from dendra.models.analysis import firing_rate, hard_active
from dendra.models.analysis.threshold_noise_metrics import (
    expected_hard_active,
    gaussian_smoothed_hard_measure,
)


def _logistic_midpoint_thresholds(n: int, scale: float) -> torch.Tensor:
    u = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    return scale * torch.logit(u)


def test_active_matches_logistic_quadrature_for_disjoint_rising_intervals():
    # Eligible thresholds are the disjoint union (0, 1] U (2, 3].
    V = torch.tensor([2.0, 3.0, 0.0, 1.0], dtype=torch.float64)[:, None, None]
    result = expected_hard_active(V, 0.1, threshold_noise_scale_mV=1.0)["active"]
    sigmoid = torch.sigmoid
    analytic = sigmoid(torch.tensor(3.0, dtype=V.dtype)) - sigmoid(
        torch.tensor(2.0, dtype=V.dtype)
    )
    analytic += sigmoid(torch.tensor(1.0, dtype=V.dtype)) - sigmoid(
        torch.tensor(0.0, dtype=V.dtype)
    )
    assert torch.allclose(result, analytic[None], atol=1e-14)

    thresholds = _logistic_midpoint_thresholds(10000, 1.0)
    low = V[:-1, 0, 0, None]
    high = V[1:, 0, 0, None]
    hard = ((low < thresholds) & (high >= thresholds)).any(dim=0)
    assert abs(result.item() - hard.double().mean().item()) < 1e-4

    # The same deterministic rule is implemented by the hard descriptor.
    for threshold in (-0.1, 0.5, 1.5, 2.5, 3.5):
        expected = any(a < threshold <= b for a, b in zip(V[:-1, 0, 0], V[1:, 0, 0]))
        assert (
            bool(hard_active(V, 0.1, V_spk=threshold, use_dv_gate=False)["active"][0])
            == expected
        )


def test_active_mask_gate_window_and_voltage_gradient():
    V = torch.tensor(
        [
            [[-2.0, -3.0]],
            [[-1.0, -2.0]],
            [[0.5, -1.0]],
            [[-2.0, -3.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    masked = expected_hard_active(
        V, 0.1, node_mask=torch.tensor([0, 1]), threshold_noise_scale_mV=1.0
    )["active"]
    expected = torch.sigmoid(torch.tensor(-1.0, dtype=V.dtype)) - torch.sigmoid(
        torch.tensor(-3.0, dtype=V.dtype)
    )
    assert torch.allclose(masked, expected[None], atol=1e-14)
    masked.sum().backward()
    assert torch.isfinite(V.grad).all()
    assert V.grad[:, 0, 0].abs().sum() == 0
    assert V.grad[:, 0, 1].abs().sum() > 0

    window = expected_hard_active(
        V.detach(), 0.1, time_window=(2, 4), threshold_noise_scale_mV=1.0
    )["active"]
    assert window.item() == 0.0
    gated = expected_hard_active(
        V.detach(),
        0.1,
        threshold_noise_scale_mV=1.0,
        use_dv_gate=True,
        dv_spk=16.0,
    )["active"]
    assert gated.item() == 0.0


def test_firing_count_is_expected_hard_count_under_one_logistic_threshold():
    # The no-dV-gate firing surrogate is already an exact expected hard count
    # per compartment under a shared logistic detection threshold.
    V = torch.tensor([-3.0, 2.0, -1.0, 3.0, -2.0], dtype=torch.float64)[:, None, None]
    scale = 0.7
    soft = firing_rate(V, 0.1, gate_V_scale=scale, use_dv_gate=False)
    thresholds = _logistic_midpoint_thresholds(20000, scale)
    hard_count = (
        ((V[:-1, 0, 0, None] < thresholds) & (V[1:, 0, 0, None] >= thresholds))
        .double()
        .sum(dim=0)
    )
    assert abs(soft["count_comp"].item() - hard_count.mean().item()) < 1e-4

    occupancies = torch.sigmoid(V[:, 0, 0] / scale)
    formula = torch.relu(occupancies[1:] - occupancies[:-1]).sum()
    assert torch.allclose(soft["count_comp"].squeeze(), formula, atol=1e-14)


def test_gaussian_smoothed_hard_step_has_correct_value_and_score_gradient():
    center = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    n = 20000
    u = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    offsets = 2.0**0.5 * torch.erfinv(2 * u - 1)
    hard = ((center.detach() + offsets) >= 0.3).to(torch.float64)
    out = gaussian_smoothed_hard_measure(center, hard, offsets, 1.0, baseline=0.5)
    out["surrogate"].backward()
    normal_cdf = 0.5 * (1 + torch.erf((center.detach() - 0.3) / 2.0**0.5))
    normal_pdf = torch.exp(-0.5 * (center.detach() - 0.3) ** 2) / (2 * torch.pi) ** 0.5
    assert abs(out["value"].item() - normal_cdf.item()) < 1e-4
    assert abs(out["gradient_estimate"].item() - normal_pdf.item()) < 3e-4
    assert torch.allclose(center.grad, out["gradient_estimate"])
