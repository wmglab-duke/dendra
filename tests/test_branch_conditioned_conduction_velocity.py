"""Hard-forward, selected-branch tests for conduction velocity."""

from __future__ import annotations

import pytest
import torch

import dendra.models.analysis as analysis
from dendra.models.analysis import hard_conduction_velocity
from dendra.models.analysis.branch_conditioned_metrics import (
    branch_conditioned_conduction_velocity,
    compare_branch_signatures,
)

DT = 0.05


def _linear_traveling_trace(
    q: torch.Tensor,
    *,
    fibers: int = 2,
    compartments: int = 4,
    samples: int = 64,
) -> torch.Tensor:
    """A sampled linear upstroke whose interpolated roots are known exactly."""
    time = torch.arange(samples, device=q.device, dtype=q.dtype) * DT
    fiber = torch.arange(fibers, device=q.device, dtype=q.dtype)
    site = torch.arange(compartments, device=q.device, dtype=q.dtype)
    # Keep the roots away from sample times at q=0 so a symmetric perturbation
    # exercises one fixed interpolation branch.
    base = 0.473 + 0.137 * fiber
    delay = (0.231 + 0.043 * fiber) * torch.exp(q)
    arrival = base[:, None] + delay[:, None] * site[None, :]
    return 80.0 * (time[:, None, None] - arrival[None, :, :])


def _lengths_and_mask(
    kind: str,
    *,
    dtype: torch.dtype,
) -> tuple[float | torch.Tensor, torch.Tensor | None]:
    if kind == "scalar-none":
        return 100.0, None
    if kind == "vector-vector":
        lengths = torch.tensor([80.0, 90.0, 110.0, 120.0], dtype=dtype)
        mask = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
        return lengths, mask
    if kind == "matrix-matrix":
        lengths = torch.tensor(
            [[80.0, 90.0, 110.0, 120.0], [120.0, 105.0, 95.0, 75.0]],
            dtype=dtype,
        )
        mask = torch.tensor(
            [[1, 1, 0, 1], [0, 1, 1, 1]],
            dtype=torch.bool,
        )
        return lengths, mask
    raise AssertionError(f"Unknown test case {kind!r}.")


@pytest.mark.parametrize(
    ("dtype", "input_kind"),
    [
        (torch.float32, "scalar-none"),
        (torch.float64, "vector-vector"),
        (torch.float32, "matrix-matrix"),
    ],
)
def test_exact_hard_forward_for_supported_lengths_masks_and_dtypes(dtype, input_kind):
    q = torch.zeros((), dtype=dtype, requires_grad=True)
    voltage = _linear_traveling_trace(q)
    lengths, node_mask = _lengths_and_mask(input_kind, dtype=dtype)
    options = dict(V_th=0.0, dv_th=20.0, time_window_ms=(0.2, 2.0))

    hard = hard_conduction_velocity(
        voltage,
        lengths,
        DT,
        node_mask=node_mask,
        **options,
    )
    selected = branch_conditioned_conduction_velocity(
        voltage,
        lengths,
        DT,
        node_mask=node_mask,
        **options,
    )

    for key, expected in hard.items():
        torch.testing.assert_close(
            selected[key], expected, rtol=0, atol=0, equal_nan=True
        )

    assert selected["branch_matches_hard"].shape == hard["has_crossing"].shape
    assert selected["branch_matches_hard"].all()
    signature = selected["branch_signature"]
    assert set(signature) == {
        "crossing_pair",
        "crossing_present",
        "site_used",
        "velocity_valid",
        "velocity_sign",
    }
    assert all(
        value.shape == hard["has_crossing"].shape for value in signature.values()
    )
    assert torch.equal(signature["crossing_present"].bool(), hard["has_crossing"])
    assert torch.equal(signature["site_used"].bool(), hard["weights"])
    expected_valid = torch.isfinite(hard["v_m_per_s"])[:, None].expand_as(
        hard["has_crossing"]
    )
    assert torch.equal(signature["velocity_valid"].bool(), expected_valid)
    expected_sign = torch.sign(hard["v_m_per_s"].nan_to_num()).long()
    expected_sign = expected_sign[:, None].expand_as(hard["has_crossing"])
    assert torch.equal(signature["velocity_sign"], expected_sign)


def test_gradient_matches_complete_hard_finite_difference_on_stable_branch():
    q = torch.zeros((), dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor([85.0, 95.0, 110.0, 125.0], dtype=q.dtype)
    selected = branch_conditioned_conduction_velocity(
        _linear_traveling_trace(q, fibers=1),
        lengths,
        DT,
        V_th=0.0,
        dv_th=20.0,
    )
    (automatic,) = torch.autograd.grad(selected["v_m_per_s"].sum(), q)

    step = 1e-4
    hard_values = []
    signatures = []
    for value in (-step, step):
        q_side = torch.tensor(value, dtype=q.dtype)
        voltage = _linear_traveling_trace(q_side, fibers=1)
        hard_values.append(
            hard_conduction_velocity(
                voltage,
                lengths,
                DT,
                V_th=0.0,
                dv_th=20.0,
            )["v_m_per_s"]
        )
        signatures.append(
            branch_conditioned_conduction_velocity(
                voltage,
                lengths,
                DT,
                V_th=0.0,
                dv_th=20.0,
            )["branch_signature"]
        )

    assert compare_branch_signatures(signatures[0], signatures[1])["all_stable"].all()
    finite_difference = ((hard_values[1] - hard_values[0]) / (2 * step)).sum()
    assert torch.isfinite(automatic)
    assert automatic.abs() > 1e-4
    torch.testing.assert_close(automatic, finite_difference, rtol=2e-5, atol=1e-8)


@pytest.mark.filterwarnings(
    r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
)
def test_voltage_parameter_jvp_matches_reverse_mode_and_complete_hard_secant():
    q = torch.zeros((), dtype=torch.float64)
    direction = torch.ones_like(q)

    def selected_velocity(parameter):
        return branch_conditioned_conduction_velocity(
            _linear_traveling_trace(parameter, fibers=1), 100.0, DT
        )["v_m_per_s"].sum()

    primal, tangent = torch.func.jvp(selected_velocity, (q,), (direction,))
    reverse_q = q.clone().requires_grad_()
    reverse_value = selected_velocity(reverse_q)
    (reverse_gradient,) = torch.autograd.grad(reverse_value, reverse_q)

    step = 1e-4
    hard_sides = []
    signatures = []
    for sign in (-1.0, 1.0):
        parameter = q + sign * step * direction
        voltage = _linear_traveling_trace(parameter, fibers=1)
        hard_sides.append(
            hard_conduction_velocity(voltage, 100.0, DT)["v_m_per_s"].sum()
        )
        signatures.append(
            branch_conditioned_conduction_velocity(voltage, 100.0, DT)[
                "branch_signature"
            ]
        )
    assert compare_branch_signatures(signatures[0], signatures[1])["all_stable"].all()
    hard_secant = (hard_sides[1] - hard_sides[0]) / (2 * step)
    hard_center = hard_conduction_velocity(
        _linear_traveling_trace(q, fibers=1), 100.0, DT
    )["v_m_per_s"].sum()

    torch.testing.assert_close(primal, hard_center, rtol=0, atol=0)
    torch.testing.assert_close(tangent, reverse_gradient, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(tangent, hard_secant, rtol=2e-5, atol=1e-8)


@pytest.mark.filterwarnings(
    r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
)
def test_length_vector_jvp_matches_reverse_direction_and_complete_hard_secant():
    q = torch.zeros(4, dtype=torch.float64)
    direction = torch.tensor([0.3, -0.2, 0.4, -0.1], dtype=q.dtype)
    base_lengths = torch.tensor([85.0, 95.0, 110.0, 125.0], dtype=q.dtype)
    voltage = _linear_traveling_trace(torch.zeros((), dtype=q.dtype), fibers=1)

    def selected_velocity(log_length_scale):
        lengths = base_lengths * torch.exp(log_length_scale)
        return branch_conditioned_conduction_velocity(voltage, lengths, DT)[
            "v_m_per_s"
        ].sum()

    primal, tangent = torch.func.jvp(selected_velocity, (q,), (direction,))
    reverse_q = q.clone().requires_grad_()
    reverse_value = selected_velocity(reverse_q)
    (reverse_gradient,) = torch.autograd.grad(reverse_value, reverse_q)
    reverse_direction = (reverse_gradient * direction).sum()

    step = 1e-4
    hard_sides = []
    for sign in (-1.0, 1.0):
        lengths = base_lengths * torch.exp(q + sign * step * direction)
        hard_sides.append(
            hard_conduction_velocity(voltage, lengths, DT)["v_m_per_s"].sum()
        )
    hard_secant = (hard_sides[1] - hard_sides[0]) / (2 * step)
    hard_center = hard_conduction_velocity(voltage, base_lengths, DT)["v_m_per_s"].sum()

    assert tangent.abs() > 1e-4
    torch.testing.assert_close(primal, hard_center, rtol=0, atol=0)
    torch.testing.assert_close(tangent, reverse_direction, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(tangent, hard_secant, rtol=2e-5, atol=1e-8)


def _two_spike_trace() -> torch.Tensor:
    voltage = torch.full((25, 1, 3), -1.0, dtype=torch.float64)
    for site in range(3):
        # The first event crosses slowly; the second event crosses steeply.
        voltage[3 + site, 0, site] = 0.1
        voltage[10 + site, 0, site] = 2.0
    return voltage


def test_first_spike_time_windows_and_dv_gate_select_declared_crossing():
    voltage = _two_spike_trace()
    common = dict(lengths_um=100.0, dt_ms=0.1, V_th=0.0)

    first = branch_conditioned_conduction_velocity(voltage, **common)
    sample_window = branch_conditioned_conduction_velocity(
        voltage, time_window=(8, 16), **common
    )
    time_window = branch_conditioned_conduction_velocity(
        voltage, time_window_ms=(0.8, 1.3), **common
    )
    steep = branch_conditioned_conduction_velocity(voltage, dv_th=20.0, **common)

    expected_first = torch.tensor([[2, 3, 4]])
    expected_second = torch.tensor([[9, 10, 11]])
    assert torch.equal(first["branch_signature"]["crossing_pair"], expected_first)
    for result in (sample_window, time_window, steep):
        assert torch.equal(result["branch_signature"]["crossing_pair"], expected_second)
        assert result["branch_matches_hard"].all()

    assert torch.all(first["t_cross_ms"] < time_window["t_cross_ms"])
    torch.testing.assert_close(
        sample_window["t_cross_ms"], time_window["t_cross_ms"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        steep["t_cross_ms"], time_window["t_cross_ms"], rtol=0, atol=0
    )


def _mixed_validity_trace(q: torch.Tensor) -> torch.Tensor:
    time = torch.arange(64, dtype=q.dtype, device=q.device) * DT
    site = torch.arange(3, dtype=q.dtype, device=q.device)

    valid_arrival = 0.47 + 0.23 * torch.exp(q) * site
    valid = 80.0 * (time[:, None] - valid_arrival[None, :])
    absent = torch.full_like(valid, -70.0) + 0.0 * q
    one_site = (
        torch.stack(
            (
                80.0 * (time - 0.63),
                torch.full_like(time, -70.0),
                torch.full_like(time, -70.0),
            ),
            dim=-1,
        )
        + 0.0 * q
    )
    simultaneous = 80.0 * (time[:, None] - torch.full_like(site[None, :], 0.71))
    simultaneous = simultaneous + 0.0 * q
    return torch.stack((valid, absent, one_site, simultaneous), dim=1)


def test_invalid_fibers_are_explicit_without_poisoning_valid_gradient():
    q = torch.zeros((), dtype=torch.float64, requires_grad=True)
    voltage = _mixed_validity_trace(q)
    selected = branch_conditioned_conduction_velocity(voltage, 100.0, DT)
    hard = hard_conduction_velocity(voltage, 100.0, DT)

    for key, expected in hard.items():
        torch.testing.assert_close(
            selected[key], expected, rtol=0, atol=0, equal_nan=True
        )
    assert torch.isfinite(selected["v_m_per_s"][0])
    assert torch.isnan(selected["v_m_per_s"][1:]).all()
    assert selected["ess"].tolist() == [3.0, 0.0, 1.0, 3.0]
    assert selected["var_t_ms2"][3].item() == pytest.approx(0.0, abs=1e-14)
    valid_signature = selected["branch_signature"]["velocity_valid"].bool()
    assert valid_signature[0].all()
    assert not valid_signature[1:].any()
    assert selected["branch_matches_hard"].all()

    (gradient,) = torch.autograd.grad(selected["v_m_per_s"][0], q)
    assert torch.isfinite(gradient)
    assert gradient.abs() > 1e-4


def test_noninterpolated_crossings_have_zero_voltage_gradient():
    q = torch.zeros((), dtype=torch.float64, requires_grad=True)
    selected = branch_conditioned_conduction_velocity(
        _linear_traveling_trace(q, fibers=1),
        100.0,
        DT,
        interpolate=False,
    )
    hard = hard_conduction_velocity(
        _linear_traveling_trace(q.detach(), fibers=1),
        100.0,
        DT,
        interpolate=False,
    )
    torch.testing.assert_close(selected["v_m_per_s"], hard["v_m_per_s"], rtol=0, atol=0)

    objective = selected["v_m_per_s"].sum()
    assert objective.requires_grad
    (gradient,) = torch.autograd.grad(objective, q)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize(
    ("samples", "window_options"),
    [
        (0, {}),
        (1, {}),
        (5, {"time_window": (2, 2)}),
        (5, {"time_window": (2, 3)}),
        (5, {"time_window": (20, 25)}),
        (5, {"time_window_ms": (10.0, 11.0)}),
    ],
    ids=(
        "empty-recording",
        "one-sample-recording",
        "empty-sample-window",
        "one-sample-window",
        "outside-sample-window",
        "outside-ms-window",
    ),
)
def test_empty_and_short_windows_match_hard_and_mask_to_finite_zero_gradient(
    samples, window_options
):
    q = torch.zeros((), dtype=torch.float64, requires_grad=True)
    voltage = torch.full((samples, 1, 2), -70.0, dtype=q.dtype) + 0.0 * q
    selected = branch_conditioned_conduction_velocity(
        voltage, 100.0, DT, **window_options
    )
    hard = hard_conduction_velocity(voltage, 100.0, DT, **window_options)

    for key, expected in hard.items():
        torch.testing.assert_close(
            selected[key], expected, rtol=0, atol=0, equal_nan=True
        )
    assert not selected["velocity_valid"].any()
    assert selected["branch_matches_hard"].all()
    assert (selected["branch_signature"]["crossing_pair"] == -1).all()

    masked = torch.nan_to_num(selected["v_m_per_s"], nan=0.0).sum()
    assert masked.requires_grad
    (gradient,) = torch.autograd.grad(masked, q)
    assert torch.isfinite(gradient)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_ms_window_includes_crossings_exactly_on_both_boundaries():
    voltage = torch.tensor(
        [[[-1.0, -1.0]], [[0.0, -1.0]], [[1.0, 1.0]], [[1.0, 1.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    # Binary-exact values isolate inclusive-boundary behavior from decimal
    # representation roundoff in the hard reference.
    dt = 0.125
    options = dict(V_th=0.0, time_window_ms=(0.125, 0.1875))
    selected = branch_conditioned_conduction_velocity(voltage, 100.0, dt, **options)
    hard = hard_conduction_velocity(voltage, 100.0, dt, **options)

    assert torch.equal(
        selected["branch_signature"]["crossing_pair"], torch.tensor([[0, 1]])
    )
    assert selected["has_crossing"].all()
    torch.testing.assert_close(
        selected["t_cross_ms"],
        torch.tensor([[0.125, 0.1875]], dtype=voltage.dtype),
        rtol=0,
        atol=1e-15,
    )
    for key, expected in hard.items():
        torch.testing.assert_close(
            selected[key], expected, rtol=0, atol=0, equal_nan=True
        )
    assert selected["velocity_valid"].all()
    assert selected["branch_matches_hard"].all()


@pytest.mark.parametrize("excluded_length", [float("nan"), float("inf")])
def test_excluded_nonfinite_trailing_geometry_does_not_poison_valid_fit(
    excluded_length,
):
    q = torch.zeros((), dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor([100.0, 100.0, excluded_length], dtype=q.dtype)
    mask = torch.tensor([1, 1, 0], dtype=torch.bool)
    voltage = _linear_traveling_trace(q, fibers=1, compartments=3)
    selected = branch_conditioned_conduction_velocity(
        voltage, lengths, DT, node_mask=mask
    )
    hard = hard_conduction_velocity(voltage, lengths, DT, node_mask=mask)

    for key, expected in hard.items():
        torch.testing.assert_close(
            selected[key], expected, rtol=0, atol=0, equal_nan=True
        )
    assert selected["velocity_valid"].all()
    assert torch.isfinite(selected["v_m_per_s"]).all()
    assert selected["branch_matches_hard"].all()
    assert not selected["branch_signature"]["site_used"][0, 2]
    (automatic,) = torch.autograd.grad(selected["v_m_per_s"].sum(), q)
    assert torch.isfinite(automatic)
    assert automatic.abs() > 1e-4

    step = 1e-4
    sides = []
    for value in (-step, step):
        sides.append(
            hard_conduction_velocity(
                _linear_traveling_trace(
                    torch.tensor(value, dtype=q.dtype), fibers=1, compartments=3
                ),
                lengths,
                DT,
                node_mask=mask,
            )["v_m_per_s"]
        )
    finite_difference = ((sides[1] - sides[0]) / (2 * step)).sum()
    torch.testing.assert_close(automatic, finite_difference, rtol=2e-5, atol=1e-8)


def test_eps_is_only_a_variance_cutoff_and_does_not_bias_the_derivative():
    q = torch.zeros((), dtype=torch.float64, requires_grad=True)
    eps = 0.01
    selected = branch_conditioned_conduction_velocity(
        _linear_traveling_trace(q, fibers=1),
        100.0,
        DT,
        eps=eps,
    )
    hard = hard_conduction_velocity(
        _linear_traveling_trace(q.detach(), fibers=1),
        100.0,
        DT,
        eps=eps,
    )
    assert hard["var_t_ms2"].item() > eps
    assert selected["velocity_valid"].all()
    torch.testing.assert_close(selected["v_m_per_s"], hard["v_m_per_s"], rtol=0, atol=0)
    (automatic,) = torch.autograd.grad(selected["v_m_per_s"].sum(), q)

    step = 1e-4
    sides = []
    for value in (-step, step):
        sides.append(
            hard_conduction_velocity(
                _linear_traveling_trace(torch.tensor(value, dtype=q.dtype), fibers=1),
                100.0,
                DT,
                eps=eps,
            )["v_m_per_s"]
        )
    finite_difference = ((sides[1] - sides[0]) / (2 * step)).sum()
    torch.testing.assert_close(automatic, finite_difference, rtol=2e-5, atol=1e-8)


def _reverse_traveling_trace(q: torch.Tensor) -> torch.Tensor:
    time = torch.arange(64, dtype=q.dtype, device=q.device) * DT
    site = torch.arange(4, dtype=q.dtype, device=q.device)
    arrival = 1.483 - 0.217 * torch.exp(q) * site
    return 80.0 * (time[:, None, None] - arrival[None, None, :])


def test_reverse_propagation_preserves_signed_velocity_and_speed_derivative():
    q = torch.zeros((), dtype=torch.float64, requires_grad=True)
    selected = branch_conditioned_conduction_velocity(
        _reverse_traveling_trace(q), 100.0, DT
    )
    assert selected["v_m_per_s"].item() < 0
    assert selected["speed_m_per_s"].item() > 0
    assert (selected["branch_signature"]["velocity_sign"] == -1).all()
    (signed_gradient,) = torch.autograd.grad(
        selected["v_m_per_s"].sum(), q, retain_graph=True
    )
    (speed_gradient,) = torch.autograd.grad(selected["speed_m_per_s"].sum(), q)
    assert signed_gradient > 0
    assert speed_gradient < 0
    torch.testing.assert_close(speed_gradient, -signed_gradient, rtol=1e-12, atol=1e-12)

    step = 1e-4
    signed_sides = []
    speed_sides = []
    signatures = []
    for value in (-step, step):
        voltage = _reverse_traveling_trace(torch.tensor(value, dtype=q.dtype))
        hard = hard_conduction_velocity(voltage, 100.0, DT)
        signed_sides.append(hard["v_m_per_s"])
        speed_sides.append(hard["speed_m_per_s"])
        signatures.append(
            branch_conditioned_conduction_velocity(voltage, 100.0, DT)[
                "branch_signature"
            ]
        )
    assert compare_branch_signatures(signatures[0], signatures[1])["all_stable"].all()
    signed_fd = ((signed_sides[1] - signed_sides[0]) / (2 * step)).sum()
    speed_fd = ((speed_sides[1] - speed_sides[0]) / (2 * step)).sum()
    torch.testing.assert_close(signed_gradient, signed_fd, rtol=2e-5, atol=1e-8)
    torch.testing.assert_close(speed_gradient, speed_fd, rtol=2e-5, atol=1e-8)


def test_gradient_step_improves_fresh_hard_velocity_objective():
    q = torch.zeros((), dtype=torch.float64, requires_grad=True)
    lengths = 100.0
    selected = branch_conditioned_conduction_velocity(
        _linear_traveling_trace(q, fibers=1), lengths, DT
    )
    baseline = selected["v_m_per_s"].detach()
    target = 0.8 * baseline
    loss = (selected["v_m_per_s"] - target).square().sum()
    loss.backward()
    assert q.grad is not None and torch.isfinite(q.grad)
    assert q.grad < 0  # increasing q increases delay and lowers velocity

    updated_q = (q - 0.25 * q.grad).detach()
    updated = hard_conduction_velocity(
        _linear_traveling_trace(updated_q, fibers=1), lengths, DT
    )["v_m_per_s"]
    assert (updated - target).square().sum() < (baseline - target).square().sum()


def test_public_exports_include_branch_conditioned_and_descriptive_aliases():
    assert (
        analysis.branch_conditioned_conduction_velocity
        is branch_conditioned_conduction_velocity
    )
    assert (
        analysis.differentiable_conduction_velocity
        is branch_conditioned_conduction_velocity
    )
    assert "branch_conditioned_conduction_velocity" in analysis.__all__
    assert "differentiable_conduction_velocity" in analysis.__all__
