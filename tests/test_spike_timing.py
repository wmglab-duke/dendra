"""Synthetic checks for experimental crossing-time and interval descriptors."""

import math

import pytest
import torch

from dendra.models.analysis import hard_firing_rate
from dendra.models.analysis.spike_timing import (
    branch_conditioned_spike_timing,
    hard_spike_timing,
)


def _trace(values):
    return torch.tensor(values, dtype=torch.float64)[:, None, None]


def test_interpolated_irregular_spikes_and_undefined_sites():
    V = torch.full((15, 2, 2), -2.0, dtype=torch.float64)
    V[3, 0, 0] = 2.0  # first root at sample 2.5
    V[8, 0, 0] = 3.0  # second root at sample 7.4
    V[13, 0, 0] = 1.0  # third root at sample 12 + 2/3
    V[5, 0, 1] = 2.0  # one spike: interval is undefined
    V[4, 1, 0] = 2.0  # another fiber, also undefined

    out = hard_spike_timing(V, 0.1, node_mask=torch.tensor([1, 0]))
    expected_times = [0.25, 0.74, (12 + 2 / 3) * 0.1]
    assert out["event_times_ms"][:3, 0, 0].tolist() == pytest.approx(expected_times)
    assert out["interspike_intervals_ms"][:2, 0, 0].tolist() == pytest.approx(
        [expected_times[1] - expected_times[0], expected_times[2] - expected_times[1]]
    )
    assert torch.isnan(out["interspike_intervals_ms"][:, 0, 1]).all()
    assert torch.isnan(out["event_times_ms"][1, 0, 1])
    expected_isi = (expected_times[-1] - expected_times[0]) / 2
    assert out["mean_isi_ms_comp"][0, 0].item() == pytest.approx(expected_isi)
    assert out["span_frequency_hz_comp"][0, 0].item() == pytest.approx(
        1000 / expected_isi
    )
    assert out["span_frequency_hz"][0].item() == pytest.approx(1000 / expected_isi)
    assert out["valid"].tolist() == [True, False]
    assert out["valid_comp"].tolist() == [[True, False], [False, False]]
    assert not out["selected_valid_comp"][0, 1]
    assert torch.isnan(out["mean_isi_ms_comp"][0, 1])
    assert torch.isnan(out["span_frequency_hz"][1])
    assert out["branch_signature"] == (((2, 7, 12), None), ((3,), None))
    assert not out["span_frequency_hz"].requires_grad


def test_window_gate_and_refractory_match_hard_firing_rate_count_semantics():
    V = _trace([-2, 2, -2, 2, -2, 2, -2, -1, 3, -2])
    choices = [
        dict(time_window=(2, 9)),
        dict(time_window_ms=(0.2, 0.8)),
        dict(use_dv_gate=True, dv_spk=35.0),
        dict(refractory_ms=0.3),
    ]
    for kwargs in choices:
        timing = hard_spike_timing(V, 0.1, **kwargs)
        count = hard_firing_rate(V, 0.1, **kwargs)["count_comp"].long()
        assert torch.equal(timing["count_comp"], count)


def test_non_grid_ms_window_filters_interpolated_crossing_times():
    V = _trace([-1, 1, -1, 1])

    excluded = hard_spike_timing(V, 1.0, time_window_ms=(0.6, 2.4))
    assert excluded["count_comp"].item() == 0
    assert excluded["branch_signature"] == (((),),)
    assert torch.isnan(excluded["event_times_ms"]).all()
    assert excluded["window_ms"].item() == pytest.approx(1.8)
    hard_count = hard_firing_rate(V, 1.0, time_window_ms=(0.6, 2.4))
    assert hard_count["count_comp"].item() == 0
    assert hard_count["window_ms"].item() == pytest.approx(1.8)

    included = hard_spike_timing(V, 1.0, time_window_ms=(0.5, 2.5))
    assert included["count_comp"].item() == 2
    assert included["branch_signature"] == (((0, 2),),)
    assert included["event_times_ms"][:, 0, 0].tolist() == pytest.approx([0.5, 2.5])


def test_ms_window_includes_crossing_on_sample_aligned_lower_bound():
    V = _trace([-1, 0, -1])
    timing = hard_spike_timing(V, 1.0, time_window_ms=(1.0, 1.5))
    count = hard_firing_rate(V, 1.0, time_window_ms=(1.0, 1.5))

    assert timing["count_comp"].item() == 1
    assert timing["branch_signature"] == (((0,),),)
    assert timing["event_times_ms"][0, 0, 0].item() == pytest.approx(1.0)
    assert count["count_comp"].item() == 1

    decimal = _trace([-1] * 10)
    decimal[7, 0, 0] = 0
    decimal_timing = hard_spike_timing(
        decimal,
        0.01,
        time_window_ms=(0.07, 0.075),
    )
    decimal_count = hard_firing_rate(
        decimal,
        0.01,
        time_window_ms=(0.07, 0.075),
    )
    assert decimal_timing["count_comp"].item() == 1
    assert decimal_timing["event_times_ms"][0, 0, 0].item() == pytest.approx(0.07)
    assert decimal_count["count_comp"].item() == 1


def test_hard_firing_rate_rejects_zero_recorded_window_duration():
    with pytest.raises(ValueError, match="positive duration"):
        hard_firing_rate(
            _trace([-1, 0]),
            1.0,
            time_window_ms=(1.0, 2.0),
        )


def test_branch_conditioned_frequency_gradient_matches_stable_finite_difference():
    base = torch.tensor(
        [-2.0, -2.0, -2.0, 2.0, -2.0, -2.0, -2.0, 3.0, -2.0, -2.0, -2.0, 1.0, -2.0],
        dtype=torch.float64,
    )
    change = torch.zeros_like(base)
    change[2] = 0.7
    change[3] = 0.3
    change[10] = -0.4
    change[11] = 0.9

    q = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    V = (base + q * change)[:, None, None]
    out = branch_conditioned_spike_timing(V, 0.2)
    hard_forward = hard_spike_timing(V, 0.2)
    for key in (
        "event_times_ms",
        "interspike_intervals_ms",
        "mean_isi_ms_comp",
        "span_frequency_hz_comp",
        "span_frequency_hz",
    ):
        assert torch.allclose(
            out[key], hard_forward[key], rtol=0, atol=0, equal_nan=True
        )
    (slope,) = torch.autograd.grad(out["span_frequency_hz"][0], q)

    def hard_at(value):
        return hard_spike_timing((base + value * change)[:, None, None], 0.2)

    step = 1e-5
    lower = hard_at(float(q.detach()) - step)
    upper = hard_at(float(q.detach()) + step)
    assert (
        out["branch_signature"]
        == lower["branch_signature"]
        == upper["branch_signature"]
    )
    finite_difference = (
        upper["span_frequency_hz"][0] - lower["span_frequency_hz"][0]
    ) / (2 * step)
    assert math.isfinite(slope.item())
    assert slope.item() == pytest.approx(finite_difference.item(), rel=1e-7)


def test_branch_signature_exposes_birth_and_loss_without_extrapolating_gradient():
    base = _trace([-1, 2, -1, -2, -1, 2, -1])
    two = branch_conditioned_spike_timing(base, 1.0)
    born = branch_conditioned_spike_timing(base + _trace([0, 0, 0, 3, 0, 0, 0]), 1.0)
    lost = branch_conditioned_spike_timing(base + _trace([0, 0, 0, 0, 0, -3, 0]), 1.0)
    assert two["count_signature"] == ((2,),)
    assert born["count_signature"] == ((3,),)
    assert lost["count_signature"] == ((1,),)
    assert two["branch_signature"] != born["branch_signature"]
    assert two["branch_signature"] != lost["branch_signature"]
    assert two["valid"].item() and born["valid"].item()
    assert not lost["valid"].item()
    assert torch.isnan(lost["span_frequency_hz"]).all()


def test_spike_spacing_changes_with_same_event_count():
    base = _trace([-2, 2, -2, -2, -2, -2, 2, -2])
    shifted = base.clone()
    shifted[6, 0, 0] = -2
    shifted[7, 0, 0] = 2
    first = hard_spike_timing(base, 1.0)
    second = hard_spike_timing(shifted, 1.0)
    assert first["count_signature"] == second["count_signature"] == ((2,),)
    assert second["mean_isi_ms_comp"][0, 0] > first["mean_isi_ms_comp"][0, 0]
    assert second["span_frequency_hz"][0] < first["span_frequency_hz"][0]


def test_interior_interval_gradients_survive_when_span_frequency_is_unchanged():
    base = _trace([-2, -2, 2, -2, -2, -2, 2, -2, -2, -2, 2, -2])
    q = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    change = torch.zeros_like(base)
    change[6, 0, 0] = 0.5  # Shift the middle interpolated crossing only.
    out = branch_conditioned_spike_timing(base + q * change, 1.0)
    intervals = out["interspike_intervals_ms"][:2, 0, 0]
    (interval_slope,) = torch.autograd.grad(intervals[0], q, retain_graph=True)
    (span_slope,) = torch.autograd.grad(out["span_frequency_hz"][0], q)
    assert interval_slope.item() != 0.0
    assert span_slope.item() == pytest.approx(0.0, abs=1e-12)
    assert out["branch_signature"] == (((1, 5, 9),),)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_bad_dt_rejected(bad):
    with pytest.raises(ValueError, match="dt_ms"):
        hard_spike_timing(_trace([-1, 1]), bad)


@pytest.mark.parametrize(
    "bad", [(0.0, 0.0), (1.0, 0.0), (0.0, float("inf")), (float("nan"), 1.0)]
)
def test_bad_ms_window_rejected(bad):
    with pytest.raises(ValueError, match="time_window_ms"):
        hard_spike_timing(_trace([-1, 1]), 1.0, time_window_ms=bad)
