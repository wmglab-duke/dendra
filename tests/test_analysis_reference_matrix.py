import pytest
import torch
import torch.nn.functional as F

from dendra.models.analysis import (
    _as_F_vector,
    _coerce_compartment_weights,
    _coerce_lengths_um,
    _coerce_pulse_times_ms,
    _coerce_train_frequency_hz,
    _finite_mean_over_indices,
    _frequency_group_ids,
    _gather_time_windows_FNKC,
    _grouped_mean_by_frequency,
    _weighted_mean_over_time_indices,
    activity_dependent_slowing,
    frequency_following,
    hard_activity_dependent_slowing,
    hard_frequency_following,
    hard_paired_pulse_recovery_from_trials,
    hard_spike_arrival_times,
    paired_pulse_recovery_from_trials,
)

DT_MS = 0.1
DTYPE = torch.float64


def _resting_trace(time_steps, fibers, compartments):
    return torch.full((time_steps, fibers, compartments), -70.0, dtype=DTYPE)


def _add_square_spike_ms(voltage, *, fiber, compartment, time_ms, width_samples=2):
    sample = int(round(float(time_ms) / DT_MS))
    voltage[sample : sample + width_samples, fiber, compartment] = 30.0


def _train_trace():
    """Two trains; the second train fails only at the distal final pulse."""
    pulse_times = torch.tensor([2.0, 6.0, 10.0, 14.0], dtype=DTYPE)
    voltage = _resting_trace(180, 2, 3)
    for fiber in range(2):
        for pulse_index, pulse_time in enumerate(pulse_times):
            for compartment in range(3):
                if fiber == 1 and pulse_index == 3 and compartment == 2:
                    continue
                latency_ms = 0.3 + 0.2 * compartment + 0.1 * pulse_index
                _add_square_spike_ms(
                    voltage,
                    fiber=fiber,
                    compartment=compartment,
                    time_ms=pulse_time + latency_ms,
                )
    return voltage, pulse_times


def _paired_pulse_trials():
    """Two ISIs with empirical thresholds 3.5 and 2.5, plus block at 5."""
    amplitudes = torch.tensor([1, 2, 3, 4, 5] * 2, dtype=DTYPE)
    isis_ms = torch.tensor([3.0] * 5 + [6.0] * 5, dtype=DTYPE)
    test_times_ms = 1.0 + isis_ms
    voltage = _resting_trace(100, amplitudes.numel(), 3)

    for trial, (amplitude, isi, test_time) in enumerate(
        zip(amplitudes, isis_ms, test_times_ms)
    ):
        threshold = 4.0 if float(isi) == 3.0 else 3.0
        if threshold <= float(amplitude) < 5.0:
            for compartment in range(3):
                _add_square_spike_ms(
                    voltage,
                    fiber=trial,
                    compartment=compartment,
                    time_ms=test_time + 0.3 + 0.2 * compartment,
                )
    return voltage, amplitudes, isis_ms


def _soft_detection_kwargs():
    return {
        "kappa_V": 1.0,
        "gate_V_scale": 1.0,
        "use_dv_gate": True,
        "dv_spk": 100.0,
        "kappa_dv": 0.1,
        "gate_dv_scale": 10.0,
        "beta": 2.0,
    }


def test_analysis_coercion_helpers_broadcast_and_validate_design_inputs():
    device = torch.device("cpu")
    shared = _coerce_pulse_times_ms([1.0, 2.0, 3.0], Fibs=2, device=device, dtype=DTYPE)
    assert shared.shape == (2, 3)
    assert torch.equal(shared[0], shared[1])
    assert torch.equal(
        _coerce_pulse_times_ms(shared[:1], Fibs=2, device=device, dtype=DTYPE),
        shared,
    )
    assert torch.equal(
        _coerce_pulse_times_ms(shared, Fibs=2, device=device, dtype=DTYPE),
        shared,
    )

    for bad, message in [
        ([], "at least one"),
        ([1.0, float("nan")], "finite"),
        ([1.0, 1.0], "strictly increasing"),
        ([2.0, 1.0], "strictly increasing"),
        (torch.ones((2, 2, 2)), "one- or two-dimensional"),
        (torch.ones((3, 2)), "must have shape"),
    ]:
        with pytest.raises(ValueError, match=message):
            _coerce_pulse_times_ms(bad, Fibs=2, device=device, dtype=DTYPE)

    assert torch.equal(
        _coerce_compartment_weights(None, Fibs=2, C=3, device=device, dtype=DTYPE),
        torch.ones((2, 3), dtype=DTYPE),
    )
    weights = _coerce_compartment_weights(
        torch.tensor([1.0, -2.0, 0.5]),
        Fibs=2,
        C=3,
        device=device,
        dtype=DTYPE,
    )
    assert torch.equal(weights[0], torch.tensor([1.0, 0.0, 0.5], dtype=DTYPE))
    assert torch.equal(
        _coerce_compartment_weights(weights, Fibs=2, C=3, device=device, dtype=DTYPE),
        weights,
    )
    with pytest.raises(ValueError, match="node_mask"):
        _coerce_compartment_weights(
            torch.ones(2), Fibs=2, C=3, device=device, dtype=DTYPE
        )

    for lengths in (100.0, [100.0], torch.full((3,), 100.0)):
        assert _coerce_lengths_um(
            lengths, Fibs=2, C=3, device=device, dtype=DTYPE
        ).shape == (2, 3)
    matrix_lengths = torch.arange(6, dtype=DTYPE).reshape(2, 3) + 1
    assert torch.equal(
        _coerce_lengths_um(matrix_lengths, Fibs=2, C=3, device=device, dtype=DTYPE),
        matrix_lengths,
    )
    with pytest.raises(ValueError, match="lengths_um"):
        _coerce_lengths_um(torch.ones(2), Fibs=2, C=3, device=device, dtype=DTYPE)

    assert (
        _as_F_vector(None, Fibs=2, device=device, dtype=DTYPE, name="reference") is None
    )
    assert torch.equal(
        _as_F_vector(2.0, Fibs=2, device=device, dtype=DTYPE, name="reference"),
        torch.full((2,), 2.0, dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="reference"):
        _as_F_vector(
            torch.ones(3),
            Fibs=2,
            device=device,
            dtype=DTYPE,
            name="reference",
        )


def test_local_window_gathering_preserves_indices_and_rejects_bad_windows():
    data = torch.arange(12, dtype=DTYPE).reshape(6, 2, 1)
    starts = torch.tensor([[-0.5, 1.0], [0.0, 2.0]], dtype=DTYPE)
    ends = starts + 1.0
    gathered, times, valid = _gather_time_windows_FNKC(
        data,
        torch.tensor(0.5, dtype=DTYPE),
        starts,
        ends,
        sample_offset=0.5,
    )

    assert gathered.shape == (2, 2, 6, 1)
    assert times.shape == valid.shape == (2, 2, 6)
    assert not valid[0, 0, 0]
    assert gathered[0, 1, 0, 0].item() == data[2, 0, 0].item()
    assert times[0, 1, 0].item() == pytest.approx(1.25)

    cases = [
        (data[:, 0], starts, ends, 0.5, 0.0, "shape"),
        (data, starts[:, :1], ends, 0.5, 0.0, "both have shape"),
        (data, starts[:1], ends[:1], 0.5, 0.0, "fiber dimension"),
        (data, starts, ends, torch.ones(2), 0.0, "scalar"),
        (data, starts, ends, 0.0, 0.0, "positive"),
        (data, starts, ends, 0.5, -0.1, "nonnegative"),
        (data, starts, starts, 0.5, 0.0, "end > start"),
    ]
    mixed_ends = ends.clone()
    mixed_ends[0, 0] = starts[0, 0] - 0.1
    cases.append((data, starts, mixed_ends, 0.5, 0.0, "end > start"))
    for tensor, win_start, win_end, dt, margin, message in cases:
        with pytest.raises(ValueError, match=message):
            _gather_time_windows_FNKC(
                tensor,
                dt,
                win_start,
                win_end,
                margin_ms=margin,
            )


def test_frequency_grouping_and_finite_mean_helpers_have_known_answers():
    pulses = torch.tensor([[0.0, 10.0, 20.0], [0.0, 20.0, 40.0]], dtype=DTYPE)
    inferred = _coerce_train_frequency_hz(None, pulses, eps=1e-12)
    assert torch.allclose(inferred, torch.tensor([100.0, 50.0], dtype=DTYPE))
    for value in (50.0, [50.0], torch.tensor([[50.0, 60.0]])):
        result = _coerce_train_frequency_hz(value, pulses, eps=1e-12)
        assert result.shape == (2,)
    with pytest.raises(ValueError, match="frequency_hz"):
        _coerce_train_frequency_hz(torch.ones(3), pulses, eps=1e-12)
    for invalid in (0.0, -10.0, float("nan")):
        with pytest.raises(ValueError, match="positive and finite"):
            _coerce_train_frequency_hz(invalid, pulses, eps=1e-12)

    unique, group_id, counts = _frequency_group_ids(
        torch.tensor([10.001, 20.0, 10.002], dtype=DTYPE),
        round_decimals=2,
    )
    assert torch.equal(unique, torch.tensor([10.0, 20.0], dtype=DTYPE))
    assert torch.equal(group_id, torch.tensor([0, 1, 0]))
    assert torch.equal(counts, torch.tensor([2, 1]))
    with pytest.raises(ValueError, match="shape"):
        _frequency_group_ids(torch.ones((1, 2)), round_decimals=None)
    with pytest.raises(ValueError, match="Rounded frequency_hz"):
        _frequency_group_ids(torch.tensor([0.1], dtype=DTYPE), round_decimals=0)

    values = torch.tensor([1.0, float("nan"), 3.0], dtype=DTYPE)
    ids = torch.tensor([0, 0, 1])
    assert torch.allclose(
        _grouped_mean_by_frequency(values, ids, 2, eps=1e-12),
        torch.tensor([1.0, 3.0], dtype=DTYPE),
    )
    assert torch.allclose(
        _grouped_mean_by_frequency(
            values,
            ids,
            2,
            eps=1e-12,
            weights=torch.tensor([1.0, 5.0, 2.0], dtype=DTYPE),
        ),
        torch.tensor([1.0, 3.0], dtype=DTYPE),
    )
    assert _grouped_mean_by_frequency(None, ids, 2, eps=1e-12) is None
    with pytest.raises(ValueError, match="values"):
        _grouped_mean_by_frequency(torch.ones((1, 3)), ids, 2, eps=1e-12)
    with pytest.raises(ValueError, match="weights"):
        _grouped_mean_by_frequency(
            torch.ones(3), ids, 2, eps=1e-12, weights=torch.ones(2)
        )

    matrix = torch.tensor([[1.0, float("nan")], [2.0, 4.0]], dtype=DTYPE)
    idx = torch.tensor([0, 1])
    assert torch.equal(
        _finite_mean_over_indices(matrix, idx, eps=1e-12),
        torch.tensor([1.0, 3.0], dtype=DTYPE),
    )
    weighted = _weighted_mean_over_time_indices(
        matrix,
        idx,
        weights=torch.tensor([[1.0, 1.0], [1.0, 3.0]], dtype=DTYPE),
        eps=1e-12,
    )
    assert torch.allclose(weighted, torch.tensor([1.0, 3.5], dtype=DTYPE))
    empty = _weighted_mean_over_time_indices(
        matrix, torch.empty(0, dtype=torch.long), eps=1e-12
    )
    assert torch.isnan(empty).all()


def test_hard_ads_recovers_latency_velocity_and_dropped_distal_spikes():
    voltage, pulse_times = _train_trace()
    distal = hard_activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        node_mask=torch.tensor([0, 0, 1]),
        response_window_ms=(0.1, 1.5),
        dv_th=100.0,
        tail_n_pulses=2,
    )

    assert torch.equal(
        distal["p_success"],
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=DTYPE),
    )
    assert torch.allclose(
        distal["latency_ms"][0],
        torch.tensor([0.67, 0.77, 0.87, 0.97], dtype=DTYPE),
        atol=1e-12,
    )
    assert torch.isnan(distal["latency_ms"][1, -1])
    assert torch.all(distal["ads_percent"][0, 1:] > 0)
    assert distal["input_isi_ms"].shape == (2, 3)

    all_sites = hard_activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        lengths_um=torch.tensor([100.0, 100.0, 100.0]),
        response_window_ms=(0.1, 1.5),
        dv_th=100.0,
        baseline_pulse_indices=[0, 1],
        reference_latency_ms=torch.tensor([0.3, 0.3]),
        tail_n_pulses=2,
        interpolate=False,
        window_margin_ms=0.2,
    )
    assert torch.allclose(all_sites["v_m_per_s"][0], torch.full((4,), 0.5, dtype=DTYPE))
    assert torch.allclose(
        all_sites["velocity_change_percent"][0],
        torch.zeros(4, dtype=DTYPE),
        atol=1e-8,
    )

    single = hard_activity_dependent_slowing(
        voltage,
        pulse_times[:1],
        DT_MS,
        response_window_ms=(0.1, 1.5),
        dv_th=None,
    )
    assert single["input_isi_ms"].shape == (2, 0)
    assert single["p_success"].shape == (2, 1)


def test_ads_dimensioned_stabilizers_are_independent_and_eps_is_compatible():
    voltage, pulse_times = _train_trace()
    voltage = voltage[:175, :1]
    common = dict(
        lengths_um=100.0,
        response_window_ms=(0.1, 1.5),
        dv_th=100.0,
        baseline_n_pulses=2,
        tail_n_pulses=2,
        reference_latency_ms=0.4,
        reference_velocity_m_per_s=0.4,
    )

    legacy = hard_activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        eps=1e-4,
        **common,
    )
    explicit_legacy = hard_activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        eps=1e-4,
        eps_time_ms2=1e-4,
        eps_latency_ms=1e-4,
        eps_velocity_m_per_s=1e-4,
        **common,
    )
    for key in (
        "v_m_per_s",
        "ads_percent",
        "instantaneous_frequency_hz",
        "velocity_change_percent",
    ):
        torch.testing.assert_close(legacy[key], explicit_legacy[key])

    soft_common = dict(
        lengths_um=100.0,
        response_window_ms=(0.1, 1.5),
        reference_latency_ms=0.4,
        reference_velocity_m_per_s=0.4,
        **_soft_detection_kwargs(),
    )
    soft_legacy = activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        eps=1e-4,
        **soft_common,
    )
    soft_explicit_legacy = activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        eps=1e-4,
        eps_time_ms2=1e-4,
        eps_latency_ms=1e-4,
        eps_velocity_m_per_s=1e-4,
        **soft_common,
    )
    for key in (
        "v_m_per_s",
        "ads_percent",
        "instantaneous_frequency_hz",
        "velocity_change_percent",
    ):
        torch.testing.assert_close(soft_legacy[key], soft_explicit_legacy[key])

    baseline = hard_activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        eps=1e-12,
        **common,
    )
    time_regularized = hard_activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        eps=1e-12,
        eps_time_ms2=0.005,
        **common,
    )
    latency_regularized = hard_activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        eps=1e-12,
        eps_latency_ms=0.2,
        **common,
    )
    velocity_regularized = hard_activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        eps=1e-12,
        eps_velocity_m_per_s=0.2,
        **common,
    )

    assert not torch.equal(time_regularized["v_m_per_s"], baseline["v_m_per_s"])
    torch.testing.assert_close(time_regularized["ads_percent"], baseline["ads_percent"])
    torch.testing.assert_close(latency_regularized["v_m_per_s"], baseline["v_m_per_s"])
    assert not torch.equal(latency_regularized["ads_percent"], baseline["ads_percent"])
    assert not torch.equal(
        latency_regularized["instantaneous_frequency_hz"],
        baseline["instantaneous_frequency_hz"],
    )
    torch.testing.assert_close(velocity_regularized["v_m_per_s"], baseline["v_m_per_s"])
    torch.testing.assert_close(
        velocity_regularized["ads_percent"], baseline["ads_percent"]
    )
    assert not torch.equal(
        velocity_regularized["velocity_change_percent"],
        baseline["velocity_change_percent"],
    )


@pytest.mark.parametrize(
    "descriptor",
    [activity_dependent_slowing, hard_activity_dependent_slowing],
)
@pytest.mark.parametrize(
    "stabilizer", ["eps", "eps_time_ms2", "eps_latency_ms", "eps_velocity_m_per_s"]
)
@pytest.mark.parametrize("invalid", [-1e-6, float("nan"), float("inf")])
def test_ads_rejects_invalid_stabilizers(descriptor, stabilizer, invalid):
    voltage = torch.zeros((2, 1, 1), dtype=DTYPE)
    pulse_times = torch.zeros(1, dtype=DTYPE)

    with pytest.raises(ValueError, match=stabilizer):
        descriptor(voltage, pulse_times, DT_MS, **{stabilizer: invalid})


def test_hard_ads_excludes_spikes_in_gather_padding_outside_response_window():
    voltage = _resting_trace(50, 3, 1)
    # Crossing times are 2.47 (inside), 2.87 (after), and 2.07 ms (before).
    for fiber, time_ms in enumerate((2.5, 2.9, 2.1)):
        _add_square_spike_ms(voltage, fiber=fiber, compartment=0, time_ms=time_ms)

    out = hard_activity_dependent_slowing(
        voltage,
        [2.0],
        DT_MS,
        response_window_ms=(0.1, 0.5),
        dv_th=None,
    )
    assert torch.equal(
        out["p_success"], torch.tensor([[1.0], [0.0], [0.0]], dtype=DTYPE)
    )


def test_hard_arrival_and_paired_pulse_enforce_exact_non_grid_ms_window():
    voltage = _resting_trace(50, 3, 1)
    # The interpolated crossings are 2.11 ms (before), 2.27 ms (inside),
    # and 2.47 ms (after) the exact [2.15, 2.45] ms response window.
    voltage[22, 0, 0] = 630.0
    voltage[23, 1, 0] = 30.0
    voltage[25, 2, 0] = 30.0

    arrival = hard_spike_arrival_times(
        voltage,
        DT_MS,
        time_window_ms=(2.15, 2.45),
    )
    assert torch.equal(
        arrival["has_crossing"].flatten(),
        torch.tensor([False, True, False]),
    )
    assert arrival["t_cross_ms"][1, 0].item() == pytest.approx(2.27)
    assert torch.isnan(arrival["t_cross_ms"][[0, 2]]).all()

    paired = hard_paired_pulse_recovery_from_trials(
        voltage,
        torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE),
        torch.full((3,), 2.0, dtype=DTYPE),
        DT_MS,
        test_pulse_times_ms=torch.full((3,), 2.0, dtype=DTYPE),
        response_window_ms=(0.15, 0.45),
        dv_th=None,
    )
    assert torch.equal(
        paired["p_response_trial"],
        torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE),
    )


@pytest.mark.parametrize("use_dv_gate", [False, True])
def test_soft_ads_matches_independent_dense_gate_formula(use_dv_gate):
    voltage = torch.full((40, 1, 1), -2.0, dtype=DTYPE)
    voltage[10:12, 0, 0] = 2.0
    voltage[15:17, 0, 0] = 1.0

    pulse_time = 1.0
    window_start = 1.2
    window_end = 1.8
    gate_t_scale = 0.1
    kappa_v = 2.0
    gate_v_scale = 1.0
    kappa_dv = 0.2
    gate_dv_scale = 5.0
    dv_spk = 10.0
    beta = 2.0
    dv_scale = 10.0
    lambda_early = 0.3

    out = activity_dependent_slowing(
        voltage,
        [pulse_time],
        DT_MS,
        response_window_ms=(0.2, 0.8),
        gate_t_scale_ms=gate_t_scale,
        window_margin_ms=10.0,
        V_spk=0.0,
        kappa_V=kappa_v,
        gate_V_scale=gate_v_scale,
        use_dv_gate=use_dv_gate,
        dv_spk=dv_spk,
        kappa_dv=kappa_dv,
        gate_dv_scale=gate_dv_scale,
        beta=beta,
        dv0=0.0,
        dv_scale=dv_scale,
        lambda_early=lambda_early,
        eps=1e-12,
    )

    time = torch.arange(voltage.shape[0], dtype=DTYPE) * DT_MS
    gate = torch.sigmoid((time - window_start) / gate_t_scale) * torch.sigmoid(
        (window_end - time) / gate_t_scale
    )
    voltage_score = (
        torch.logsumexp(kappa_v * voltage[:, 0, 0] + torch.log(gate), dim=0) / kappa_v
    )
    expected_success = torch.sigmoid(voltage_score / gate_v_scale)

    derivative = (voltage[1:, 0, 0] - voltage[:-1, 0, 0]) / DT_MS
    time_mid = 0.5 * (time[1:] + time[:-1])
    gate_mid = torch.sigmoid((time_mid - window_start) / gate_t_scale) * torch.sigmoid(
        (window_end - time_mid) / gate_t_scale
    )
    if use_dv_gate:
        derivative_score = (
            torch.logsumexp(
                kappa_dv * (derivative - dv_spk) + torch.log(gate_mid), dim=0
            )
            / kappa_dv
        )
        expected_success = expected_success * torch.sigmoid(
            derivative_score / gate_dv_scale
        )

    arrival_logits = (
        beta * F.softplus(derivative / dv_scale)
        + torch.log(gate_mid)
        - lambda_early * (time_mid - window_start)
    )
    expected_arrival = (torch.softmax(arrival_logits, dim=0) * time_mid).sum()

    assert out["p_success"][0, 0].item() == pytest.approx(
        expected_success.item(), abs=1e-10
    )
    assert out["t_hat_ms_comp"][0, 0, 0].item() == pytest.approx(
        expected_arrival.item(), abs=1e-10
    )
    assert out["latency_ms_comp"][0, 0, 0].item() == pytest.approx(
        expected_arrival.item() - pulse_time, abs=1e-10
    )


def test_soft_ads_is_chunk_invariant_matches_hard_events_and_has_finite_gradients():
    base, pulse_times = _train_trace()
    voltage = base.clone().requires_grad_()
    kwargs = {
        **_soft_detection_kwargs(),
        "node_mask": torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE),
        "response_window_ms": (0.1, 1.5),
        "tail_n_pulses": 2,
        "baseline_pulse_indices": [0, 1],
        "reference_latency_ms": torch.tensor([0.67, 0.67], dtype=DTYPE),
        "reg_ess_weight": 0.01,
        "reg_var_weight": 0.01,
        "reg_success_weight": 0.01,
    }
    whole = activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        return_compartment_metrics=True,
        return_time_traces=True,
        **kwargs,
    )
    chunked = activity_dependent_slowing(
        voltage,
        pulse_times[None, :],
        torch.tensor(DT_MS, dtype=DTYPE),
        chunk_pulses=2,
        return_compartment_metrics=False,
        **kwargs,
    )

    for key in ("p_success", "latency_ms", "ads_percent", "follow_fraction"):
        assert torch.allclose(
            whole[key], chunked[key], atol=1e-12, rtol=1e-12, equal_nan=True
        )
    assert chunked["p_success_comp"] is None
    assert chunked["latency_ms_comp"] is None
    assert "intentionally not stored" in whole["return_time_traces_note"]

    hard = hard_activity_dependent_slowing(
        base,
        pulse_times,
        DT_MS,
        node_mask=torch.tensor([0, 0, 1]),
        response_window_ms=(0.1, 1.5),
        dv_th=100.0,
        tail_n_pulses=2,
    )
    assert torch.equal(whole["p_success"] > 0.5, hard["p_success"].bool())
    assert torch.allclose(
        whole["latency_ms"][:, :3], hard["latency_ms"][:, :3], atol=0.03
    )

    loss = whole["p_success"].sum() + whole["latency_ms"].sum() + whole["reg"]
    loss.backward()
    assert voltage.grad is not None
    assert torch.isfinite(voltage.grad).all()


def test_soft_ads_velocity_references_and_no_derivative_gate_are_finite():
    voltage, pulse_times = _train_trace()
    out = activity_dependent_slowing(
        voltage,
        pulse_times,
        DT_MS,
        lengths_um=100.0,
        node_mask=torch.ones((2, 3), dtype=DTYPE),
        response_window_ms=(0.1, 1.5),
        window_margin_ms=0.2,
        use_dv_gate=False,
        beta=2.0,
        reference_latency_ms=0.45,
        reference_velocity_m_per_s=torch.tensor([0.5, 0.5], dtype=DTYPE),
        tail_n_pulses=2,
        reg_ess_weight=0.01,
        reg_var_weight=0.01,
        reg_success_weight=0.01,
    )
    assert out["p_dv_comp"] is None
    assert out["v_m_per_s"].shape == (2, 4)
    assert torch.isfinite(out["v_m_per_s"]).all()
    assert torch.isfinite(out["velocity_change_percent"]).all()
    assert torch.isfinite(out["reg"])


def test_soft_train_and_paired_empty_readouts_have_nan_measurements():
    voltage = _resting_trace(80, 2, 2)
    for fiber in range(2):
        _add_square_spike_ms(voltage, fiber=fiber, compartment=0, time_ms=2.5)
        _add_square_spike_ms(voltage, fiber=fiber, compartment=1, time_ms=2.7)
    node_mask = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=DTYPE)
    common = dict(
        response_window_ms=(0.1, 1.0),
        node_mask=node_mask,
        lengths_um=100.0,
        use_dv_gate=False,
        kappa_V=1.0,
        gate_V_scale=1.0,
        beta=2.0,
    )

    train = activity_dependent_slowing(voltage, [2.0], DT_MS, **common)
    assert train["p_success"][0, 0].item() == 0.0
    assert train["p_success"][1, 0] > 0.0
    for key in (
        "latency_ms",
        "arrival_time_ms",
        "ads_percent",
        "var_t_ms2",
        "v_m_per_s",
        "speed_m_per_s",
        "velocity_change_percent",
    ):
        assert torch.isnan(train[key][0]).all()
        assert torch.isfinite(train[key][1]).all()

    amplitudes = torch.tensor([1.0, 2.0], dtype=DTYPE)
    isis_ms = torch.full((2,), 2.0, dtype=DTYPE)
    paired = paired_pulse_recovery_from_trials(
        voltage,
        amplitudes,
        isis_ms,
        DT_MS,
        test_pulse_times_ms=torch.full((2,), 2.0, dtype=DTYPE),
        **common,
    )
    assert paired["p_response_trial"][0].item() == 0.0
    assert paired["p_response_trial"][1] > 0.0
    for key in (
        "latency_ms_trial",
        "var_t_ms2",
        "v_m_per_s_trial",
        "speed_m_per_s_trial",
    ):
        assert torch.isnan(paired[key][0])
        assert torch.isfinite(paired[key][1])
    assert torch.isfinite(paired["latency_ms_by_isi"]).all()
    assert torch.isfinite(paired["v_m_per_s_by_isi"]).all()

    all_empty = paired_pulse_recovery_from_trials(
        voltage,
        amplitudes,
        isis_ms,
        DT_MS,
        test_pulse_times_ms=torch.full((2,), 2.0, dtype=DTYPE),
        node_mask=torch.zeros_like(node_mask),
        lengths_um=100.0,
        response_window_ms=(0.1, 1.0),
        use_dv_gate=False,
        kappa_V=1.0,
        gate_V_scale=1.0,
        beta=2.0,
    )
    assert torch.equal(all_empty["p_response_trial"], torch.zeros(2, dtype=DTYPE))
    assert torch.isnan(all_empty["latency_ms_trial"]).all()
    assert torch.isnan(all_empty["latency_ms_by_isi"]).all()
    assert torch.isnan(all_empty["v_m_per_s_by_isi"]).all()


def test_frequency_following_separates_initiation_from_propagation_failure():
    voltage, pulse_times = _train_trace()
    kwargs = dict(
        frequency_hz=torch.tensor([50.0, 100.0]),
        node_mask=torch.tensor([0, 0, 1]),
        initiation_node_mask=torch.tensor([1, 0, 0]),
        response_window_ms=(0.1, 1.5),
        tail_n_pulses=2,
    )
    hard = hard_frequency_following(voltage, pulse_times, DT_MS, dv_th=100.0, **kwargs)
    assert torch.equal(
        hard["p_success"],
        torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=DTYPE),
    )
    assert torch.equal(hard["p_initiated"], torch.ones((2, 4), dtype=DTYPE))
    assert torch.equal(
        hard["final_conduction_failure"], torch.tensor([0.0, 1.0], dtype=DTYPE)
    )
    assert torch.allclose(
        hard["block_fraction"], torch.tensor([0.0, 0.25], dtype=DTYPE)
    )
    assert torch.equal(
        hard["frequency_unique_hz"], torch.tensor([50.0, 100.0], dtype=DTYPE)
    )
    assert torch.allclose(
        hard["follow_fraction_by_frequency"],
        torch.tensor([1.0, 0.75], dtype=DTYPE),
    )

    permutation = torch.tensor([1, 0])
    permuted = hard_frequency_following(
        voltage[:, permutation],
        pulse_times,
        DT_MS,
        dv_th=100.0,
        frequency_hz=kwargs["frequency_hz"][permutation],
        node_mask=kwargs["node_mask"],
        initiation_node_mask=kwargs["initiation_node_mask"],
        response_window_ms=kwargs["response_window_ms"],
        tail_n_pulses=2,
    )
    assert torch.equal(permuted["p_success"], hard["p_success"][permutation])
    assert torch.equal(
        permuted["follow_fraction_by_frequency"],
        hard["follow_fraction_by_frequency"],
    )

    soft = frequency_following(
        voltage,
        pulse_times,
        DT_MS,
        **kwargs,
        **_soft_detection_kwargs(),
    )
    assert soft["final_conduction_failure"][1] > soft["final_conduction_failure"][0]
    assert torch.equal(soft["p_success"] > 0.5, hard["p_success"].bool())
    assert soft["initiation_ads"] is not None
    assert soft["distal_ads"] is not None


def test_frequency_following_single_pulse_and_summary_validation():
    voltage, pulse_times = _train_trace()
    single = hard_frequency_following(
        voltage,
        pulse_times[:1],
        DT_MS,
        frequency_hz=20.0,
        response_window_ms=(0.1, 1.5),
        dv_th=None,
    )
    assert single["pair_success"] is None
    assert single["p_entrained_pair"] is None
    assert torch.isnan(single["entrained_fraction"]).all()
    assert single["soft_max_following_frequency_hz"].item() == pytest.approx(20.0)

    inferred = hard_frequency_following(
        voltage,
        pulse_times,
        DT_MS,
        response_window_ms=(0.1, 1.5),
        dv_th=None,
    )
    assert inferred["frequency_unique_hz"].item() == pytest.approx(250.0)

    per_fiber_tolerance = hard_frequency_following(
        voltage,
        pulse_times,
        DT_MS,
        frequency_hz=[50.0, 100.0],
        response_window_ms=(0.1, 1.5),
        dv_th=None,
        max_isi_error_ms=torch.tensor([0.2, 0.3], dtype=DTYPE),
    )
    assert per_fiber_tolerance["p_isi_close"].shape == (2, 3)
    matrix_tolerance = hard_frequency_following(
        voltage,
        pulse_times,
        DT_MS,
        frequency_hz=[50.0, 100.0],
        response_window_ms=(0.1, 1.5),
        dv_th=None,
        max_isi_error_ms=torch.full((2, 3), 0.2, dtype=DTYPE),
    )
    assert matrix_tolerance["p_isi_close"].shape == (2, 3)

    with pytest.raises(ValueError, match="smaller than"):
        hard_frequency_following(
            voltage,
            pulse_times,
            DT_MS,
            frequency_hz=20.0,
            accommodation_skip_pulses=len(pulse_times),
        )
    with pytest.raises(ValueError, match="max_isi_error_ms"):
        hard_frequency_following(
            voltage,
            pulse_times,
            DT_MS,
            frequency_hz=20.0,
            max_isi_error_ms=torch.ones(3),
        )


def test_paired_pulse_hard_reference_recovers_threshold_block_and_velocity():
    voltage, amplitudes, isis_ms = _paired_pulse_trials()
    hard = hard_paired_pulse_recovery_from_trials(
        voltage,
        amplitudes,
        isis_ms,
        DT_MS,
        condition_pulse_time_ms=1.0,
        response_window_ms=(0.1, 1.3),
        node_mask=torch.ones(3),
        lengths_um=100.0,
        baseline_threshold=torch.tensor([2.5, 2.5]),
        reference_latency_ms=0.27,
        reference_velocity_m_per_s=0.5,
        dv_th=100.0,
        isi_round_decimals=3,
    )

    assert torch.equal(hard["isi_unique_ms"], torch.tensor([3.0, 6.0], dtype=DTYPE))
    assert torch.equal(hard["I_th_test"], torch.tensor([3.5, 2.5], dtype=DTYPE))
    assert torch.allclose(
        hard["threshold_ratio"], torch.tensor([1.4, 1.0], dtype=DTYPE)
    )
    assert torch.equal(
        hard["boundaries"]["I_post_inactive"],
        torch.tensor([5.0, 5.0], dtype=DTYPE),
    )
    assert torch.allclose(
        hard["latency_ms_by_isi"], torch.full((2,), 0.27, dtype=DTYPE)
    )
    assert torch.allclose(hard["v_m_per_s_by_isi"], torch.full((2,), 0.5, dtype=DTYPE))
    assert torch.allclose(
        hard["velocity_percent_change"], torch.zeros(2, dtype=DTYPE), atol=1e-10
    )

    alternate = hard_paired_pulse_recovery_from_trials(
        voltage,
        -amplitudes,
        isis_ms,
        torch.tensor(DT_MS, dtype=DTYPE),
        test_pulse_times_ms=1.0 + isis_ms,
        response_window_ms=(0.1, 1.3),
        node_mask=torch.ones((amplitudes.numel(), 3)),
        strength=-amplitudes,
        use_abs_strength=True,
        threshold_method="onset",
        interpolate=False,
        dv_th=None,
    )
    assert torch.equal(alternate["I_th_test"], torch.tensor([4.0, 3.0], dtype=DTYPE))
    assert alternate["threshold_ratio"] is None
    assert alternate["latency_shift_ms"] is None
    assert alternate["v_m_per_s_trial"] is None

    quiet = hard_paired_pulse_recovery_from_trials(
        torch.full_like(voltage, -70.0),
        amplitudes,
        isis_ms,
        DT_MS,
        condition_pulse_time_ms=1.0,
        response_window_ms=(0.1, 1.3),
        dv_th=None,
    )
    assert torch.isnan(quiet["I_th_test"]).all()


def test_soft_paired_pulse_matches_hard_threshold_and_is_trial_permutation_invariant():
    voltage, amplitudes, isis_ms = _paired_pulse_trials()
    common = dict(
        condition_pulse_time_ms=1.0,
        response_window_ms=(0.1, 1.3),
        node_mask=torch.ones(3),
        lengths_um=100.0,
        baseline_threshold=torch.tensor([2.5, 2.5]),
        reference_latency_ms=0.45,
        reference_velocity_m_per_s=0.5,
        isi_round_decimals=3,
    )
    hard = hard_paired_pulse_recovery_from_trials(
        voltage,
        amplitudes,
        isis_ms,
        DT_MS,
        dv_th=100.0,
        **common,
    )

    differentiable_voltage = voltage.clone().requires_grad_()
    soft = paired_pulse_recovery_from_trials(
        differentiable_voltage,
        amplitudes,
        isis_ms,
        DT_MS,
        threshold_method="onset_midpoint",
        reg_bracket_weight=0.01,
        reg_monotone_weight=0.01,
        reg_unimodal_weight=0.01,
        return_time_traces=True,
        **common,
        **_soft_detection_kwargs(),
    )
    assert torch.allclose(soft["I_th_test"], hard["I_th_test"], atol=1e-8)
    assert torch.equal(
        soft["p_response_trial"] > 0.5,
        hard["p_response_trial"].bool(),
    )
    assert torch.allclose(soft["v_m_per_s_by_isi"], hard["v_m_per_s_by_isi"], atol=1e-8)
    assert soft["g_response"].shape == (*voltage.shape[:2], 1)
    assert soft["g_response_mid"].shape == (
        voltage.shape[0] - 1,
        voltage.shape[1],
        1,
    )

    permutation = torch.tensor([7, 0, 9, 3, 5, 1, 8, 4, 2, 6])
    permuted = paired_pulse_recovery_from_trials(
        voltage[:, permutation],
        amplitudes[permutation],
        isis_ms[permutation],
        DT_MS,
        threshold_method="onset_midpoint",
        **common,
        **_soft_detection_kwargs(),
    )
    for key in ("I_th_test", "p_response_by_isi", "latency_ms_by_isi"):
        assert torch.allclose(soft[key], permuted[key], atol=1e-10)

    (soft["I_th_test"].sum() + soft["reg"]).backward()
    assert differentiable_voltage.grad is not None
    assert torch.isfinite(differentiable_voltage.grad).all()


@pytest.mark.parametrize(
    "threshold_method,success_aggregate",
    [
        ("onset", "mean"),
        ("onset_midpoint", "max"),
        ("ptarget", "soft_or"),
        ("bracket_midpoint", "mean"),
    ],
)
def test_soft_paired_pulse_threshold_and_aggregate_modes(
    threshold_method, success_aggregate
):
    voltage, amplitudes, isis_ms = _paired_pulse_trials()
    out = paired_pulse_recovery_from_trials(
        voltage,
        -amplitudes,
        isis_ms,
        DT_MS,
        condition_pulse_time_ms=1.0,
        response_window_ms=(0.1, 1.3),
        use_abs_strength=True,
        threshold_method=threshold_method,
        success_aggregate=success_aggregate,
        compute_block=threshold_method != "ptarget",
        compute_latency=threshold_method != "bracket_midpoint",
        enforce_min_trials_per_isi=True,
        **_soft_detection_kwargs(),
    )
    assert torch.isfinite(out["I_th_test"]).all()
    if threshold_method == "ptarget":
        assert torch.isnan(out["boundaries"]["I_post_inactive"]).all()
    if threshold_method == "bracket_midpoint":
        assert out["latency_ms_trial"] is None


def test_train_and_paired_descriptors_reject_invalid_experimental_designs():
    voltage, pulse_times = _train_trace()
    for fn in (activity_dependent_slowing, hard_activity_dependent_slowing):
        with pytest.raises(ValueError, match="at least one"):
            fn(voltage, [], DT_MS)
        with pytest.raises(ValueError, match="strictly increasing"):
            fn(voltage, [2.0, 2.0], DT_MS)
        with pytest.raises(ValueError, match="positive"):
            fn(voltage, pulse_times, 0.0)
        with pytest.raises(ValueError, match="baseline_n_pulses"):
            fn(voltage, pulse_times, DT_MS, baseline_n_pulses=0)
        with pytest.raises(ValueError, match="tail_n_pulses"):
            fn(voltage, pulse_times, DT_MS, tail_n_pulses=0)
        with pytest.raises(ValueError, match="out-of-range"):
            fn(voltage, pulse_times, DT_MS, baseline_pulse_indices=[99])

    with pytest.raises(ValueError, match="shape"):
        activity_dependent_slowing(voltage[:, 0], pulse_times, DT_MS)
    with pytest.raises(ValueError, match="at least 2"):
        activity_dependent_slowing(voltage[:1], pulse_times, DT_MS)
    with pytest.raises(ValueError, match="end > start"):
        activity_dependent_slowing(
            voltage, pulse_times, DT_MS, response_window_ms=(1.0, 1.0)
        )
    with pytest.raises(ValueError, match="positive integer"):
        activity_dependent_slowing(voltage, pulse_times, DT_MS, chunk_pulses=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        activity_dependent_slowing(voltage, pulse_times, DT_MS, baseline_n_pulses=99)
    with pytest.raises(ValueError, match="at least one index"):
        activity_dependent_slowing(
            voltage, pulse_times, DT_MS, baseline_pulse_indices=[]
        )
    with pytest.raises(ValueError, match="node_mask"):
        hard_activity_dependent_slowing(
            voltage, pulse_times, DT_MS, node_mask=torch.ones(2)
        )
    with pytest.raises(ValueError, match="lengths_um"):
        hard_activity_dependent_slowing(
            voltage, pulse_times, DT_MS, lengths_um=torch.ones(2)
        )
    with pytest.raises(ValueError, match="nonnegative"):
        hard_activity_dependent_slowing(
            voltage, pulse_times, DT_MS, window_margin_ms=-0.1
        )

    with pytest.raises(ValueError, match="positive and finite"):
        frequency_following(voltage, pulse_times, DT_MS, frequency_hz=-10.0)

    paired_voltage, amplitudes, isis_ms = _paired_pulse_trials()
    for fn in (
        paired_pulse_recovery_from_trials,
        hard_paired_pulse_recovery_from_trials,
    ):
        with pytest.raises(ValueError, match="positive"):
            fn(paired_voltage, amplitudes, isis_ms, 0.0)
        with pytest.raises(ValueError, match="end > start"):
            fn(
                paired_voltage,
                amplitudes,
                isis_ms,
                DT_MS,
                response_window_ms=(1.0, 1.0),
            )
        with pytest.raises(ValueError, match="threshold_method"):
            fn(
                paired_voltage,
                amplitudes,
                isis_ms,
                DT_MS,
                threshold_method="invalid",
            )

    with pytest.raises(ValueError, match="shape"):
        paired_pulse_recovery_from_trials(
            paired_voltage[:, :, 0], amplitudes, isis_ms, DT_MS
        )
    with pytest.raises(ValueError, match="at least 2"):
        paired_pulse_recovery_from_trials(
            paired_voltage[:1], amplitudes, isis_ms, DT_MS
        )
    with pytest.raises(ValueError, match=r"shape \(P,\)"):
        paired_pulse_recovery_from_trials(
            paired_voltage, amplitudes[:-1], isis_ms, DT_MS
        )
    with pytest.raises(ValueError, match="test_pulse_times_ms"):
        paired_pulse_recovery_from_trials(
            paired_voltage,
            amplitudes,
            isis_ms,
            DT_MS,
            test_pulse_times_ms=torch.ones(2),
        )
    with pytest.raises(ValueError, match="strength"):
        paired_pulse_recovery_from_trials(
            paired_voltage,
            amplitudes,
            isis_ms,
            DT_MS,
            strength=torch.ones(2),
        )
    with pytest.raises(ValueError, match="node_mask"):
        paired_pulse_recovery_from_trials(
            paired_voltage,
            amplitudes,
            isis_ms,
            DT_MS,
            node_mask=torch.ones(2),
        )
    with pytest.raises(ValueError, match="success_aggregate"):
        paired_pulse_recovery_from_trials(
            paired_voltage,
            amplitudes,
            isis_ms,
            DT_MS,
            success_aggregate="invalid",
        )
    with pytest.raises(ValueError, match="baseline_threshold"):
        paired_pulse_recovery_from_trials(
            paired_voltage,
            amplitudes,
            isis_ms,
            DT_MS,
            baseline_threshold=torch.ones(3),
        )

    one_per_group = torch.tensor([0, 5])
    with pytest.raises(ValueError, match="< 2 trials"):
        paired_pulse_recovery_from_trials(
            paired_voltage[:, one_per_group],
            amplitudes[one_per_group],
            isis_ms[one_per_group],
            DT_MS,
        )
