"""A hard-forward threshold proxy from a differentiable voltage trace.

The hard spike protocol supplies the threshold value. A voltage trace at a
fixed probe amplitude and its amplitude tangent supply a *candidate* local
parameter direction. This direction need not equal the derivative of the hard
threshold: validation against complete hard searches remains necessary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TraceTangentDiagnostics:
    """Detached checks in the caller's trace and amplitude units."""

    selected_samples: int
    amplitude_tangent_l2: float
    relative_linearization_error: float | None
    secant_gain: float | None


@dataclass(frozen=True)
class TraceTangentThresholdResult:
    """Hard value with a trace-projection backward, or a rejection reason."""

    proxy: torch.Tensor | None
    diagnostics: TraceTangentDiagnostics
    rejection_reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.proxy is not None and not self.rejection_reasons


@dataclass(frozen=True)
class TraceTangentRecoveryResult:
    """Conditioned-to-baseline threshold ratio, or a rejection reason."""

    ratio: torch.Tensor | None
    rejection_reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.ratio is not None and not self.rejection_reasons


@dataclass(frozen=True)
class TraceTangentChronaxieResult:
    """Validated current-domain strength-duration fit, or rejection reasons."""

    rheobase: torch.Tensor | None
    strength_duration_slope: torch.Tensor | None
    chronaxie_ms: torch.Tensor | None
    thresholds: torch.Tensor | None
    rejection_reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            self.rheobase is not None
            and self.strength_duration_slope is not None
            and self.chronaxie_ms is not None
            and self.thresholds is not None
            and not self.rejection_reasons
        )


def chronaxie_from_threshold_proxies(
    pulse_widths_ms: Sequence[float] | torch.Tensor,
    threshold_proxies: Sequence[TraceTangentThresholdResult],
    *,
    weights: Sequence[float] | torch.Tensor | None = None,
    require_positive_chronaxie: bool = True,
) -> TraceTangentChronaxieResult:
    r"""Fit rheobase and chronaxie from validated hard-forward thresholds.

    Each input result should represent the complete hard activation threshold
    at one fixed pulse width.  The function stacks their hard-forward proxies
    and fits the current-domain Weiss relation

    ``T(d) = r + b/d = r * (1 + c/d)``,

    where ``d`` is pulse width in milliseconds, ``T`` is threshold stimulus
    strength, ``r`` is rheobase, ``b`` is the strength-duration slope, and
    ``c = b/r`` is chronaxie in milliseconds.  Ordinary or weighted least
    squares is used without a denominator regularizer, so a valid result has
    the exact fit and gradient for the supplied hard-forward thresholds.

    Pulse widths and weights define the fixed protocol and are detached.
    Gradients propagate through every threshold proxy to all connected model
    parameters.  The fit is rejected when a proxy is invalid or disconnected,
    when the design lacks two distinct positive pulse widths with positive
    weight, or when the fitted physiological quantities are invalid.  Rejected
    results contain explicit ``rejection_reasons`` rather than a regularized
    numerical answer.

    Parameters
    ----------
    pulse_widths_ms
        One positive, finite pulse width per threshold proxy.
    threshold_proxies
        Valid results returned by :func:`trace_tangent_threshold_proxy`.
    weights
        Optional fixed nonnegative fit weights. Zero excludes a point.
    require_positive_chronaxie
        Reject nonpositive fitted chronaxie values when true (the default).
    """
    if isinstance(threshold_proxies, (str, bytes)) or not isinstance(
        threshold_proxies, Sequence
    ):
        raise TypeError("threshold_proxies must be a sequence of results.")
    results = tuple(threshold_proxies)
    if not results:
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("no_threshold_proxies",),
        )

    reasons: list[str] = []
    scalar_proxies: list[torch.Tensor] = []
    reference: torch.Tensor | None = None
    for index, result in enumerate(results):
        prefix = f"threshold_{index}"
        if not isinstance(result, TraceTangentThresholdResult):
            raise TypeError(
                "threshold_proxies must contain TraceTangentThresholdResult objects."
            )
        if not result.valid:
            details = result.rejection_reasons or ("proxy_unavailable",)
            reasons.extend(f"{prefix}:{reason}" for reason in details)
            continue
        proxy = result.proxy
        if not isinstance(proxy, torch.Tensor) or proxy.numel() != 1:
            reasons.append(f"{prefix}:proxy_not_scalar_tensor")
            continue
        if not proxy.is_floating_point() or not bool(
            torch.isfinite(proxy.detach()).all()
        ):
            reasons.append(f"{prefix}:proxy_not_finite_float")
            continue
        if not proxy.requires_grad:
            reasons.append(f"{prefix}:proxy_has_no_autograd_connection")
            continue
        scalar = proxy.reshape(())
        if reference is None:
            reference = scalar
        elif scalar.device != reference.device or scalar.dtype != reference.dtype:
            reasons.append("proxy_device_or_dtype_mismatch")
            continue
        scalar_proxies.append(scalar)
    if reasons:
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            tuple(dict.fromkeys(reasons)),
        )
    assert reference is not None

    try:
        widths = torch.as_tensor(
            pulse_widths_ms,
            dtype=reference.dtype,
            device=reference.device,
        ).detach()
    except (TypeError, ValueError) as error:
        raise TypeError("pulse_widths_ms must be numeric.") from error
    if widths.ndim != 1 or widths.numel() != len(scalar_proxies):
        raise ValueError("pulse_widths_ms must be a vector matching threshold_proxies.")
    if not bool(torch.isfinite(widths).all()):
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("pulse_width_nonfinite",),
        )
    if not bool((widths > 0).all()):
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("pulse_width_nonpositive",),
        )

    if weights is None:
        fit_weights = torch.ones_like(widths)
    else:
        try:
            fit_weights = torch.as_tensor(
                weights,
                dtype=reference.dtype,
                device=reference.device,
            ).detach()
        except (TypeError, ValueError) as error:
            raise TypeError("weights must be numeric.") from error
        if fit_weights.ndim != 1 or fit_weights.shape != widths.shape:
            raise ValueError("weights must be a vector matching pulse_widths_ms.")
        if not bool(torch.isfinite(fit_weights).all()):
            return TraceTangentChronaxieResult(
                None,
                None,
                None,
                None,
                ("weights_nonfinite",),
            )
        if not bool((fit_weights >= 0).all()):
            return TraceTangentChronaxieResult(
                None,
                None,
                None,
                None,
                ("weights_negative",),
            )

    included = fit_weights > 0
    if int(included.count_nonzero()) < 2:
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("insufficient_positive_weight_points",),
        )
    if torch.unique(widths[included]).numel() < 2:
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("insufficient_distinct_pulse_widths",),
        )

    thresholds = torch.stack(scalar_proxies)
    x = widths.reciprocal()
    total_weight = fit_weights.sum()
    x_mean = (fit_weights * x).sum() / total_weight
    threshold_mean = (fit_weights * thresholds).sum() / total_weight
    centered_x = x - x_mean
    design_variance = (fit_weights * centered_x.square()).sum()
    if not bool(torch.isfinite(design_variance).item()) or bool(
        (design_variance <= 0).item()
    ):
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("strength_duration_design_singular",),
        )

    slope = (
        fit_weights * centered_x * (thresholds - threshold_mean)
    ).sum() / design_variance
    rheobase = threshold_mean - slope * x_mean
    if not bool(torch.isfinite(rheobase.detach()).item()):
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("rheobase_nonfinite",),
        )
    if bool((rheobase.detach() <= 0).item()):
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("rheobase_nonpositive",),
        )
    if not bool(torch.isfinite(slope.detach()).item()):
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("strength_duration_slope_nonfinite",),
        )

    chronaxie_ms = slope / rheobase
    if not bool(torch.isfinite(chronaxie_ms.detach()).item()):
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("chronaxie_nonfinite",),
        )
    if require_positive_chronaxie and bool((chronaxie_ms.detach() <= 0).item()):
        return TraceTangentChronaxieResult(
            None,
            None,
            None,
            None,
            ("chronaxie_nonpositive",),
        )
    return TraceTangentChronaxieResult(
        rheobase,
        slope,
        chronaxie_ms,
        thresholds,
        (),
    )


def paired_pulse_recovery_ratio_from_trace_tangents(
    conditioned: TraceTangentThresholdResult,
    baseline: TraceTangentThresholdResult,
) -> TraceTangentRecoveryResult:
    r"""Compose two matched hard-forward threshold proxies as ``C / B``.

    ``conditioned`` and ``baseline`` must come from the same chosen recovery
    protocol and parameter coordinate. This function supplies no simulation
    or matching policy: the caller must run the full conditioned and
    time-matched unconditioned hard searches, and validate each trace tangent
    against hard parameter secants. When both inputs are valid, ordinary Torch
    division gives the quotient gradient through *both* searches,

    ``d(C/B)/dq = (B dC/dq - C dB/dq) / B**2``.

    The forward ratio is exactly the ratio of the two reported hard thresholds.
    A nonpositive baseline or a malformed proxy returns ``ratio=None`` with
    an explicit reason. No denominator regularizer is added, since that would
    change the hard protocol's ratio.
    """
    if not isinstance(conditioned, TraceTangentThresholdResult):
        raise TypeError("conditioned must be a TraceTangentThresholdResult.")
    if not isinstance(baseline, TraceTangentThresholdResult):
        raise TypeError("baseline must be a TraceTangentThresholdResult.")
    reasons: list[str] = []
    for name, result in (("conditioned", conditioned), ("baseline", baseline)):
        if not result.valid:
            details = result.rejection_reasons or ("proxy_unavailable",)
            reasons.extend(f"{name}:{reason}" for reason in details)
            continue
        proxy = result.proxy
        if not isinstance(proxy, torch.Tensor) or proxy.numel() != 1:
            reasons.append(f"{name}:proxy_not_scalar_tensor")
        elif not proxy.is_floating_point() or not bool(
            torch.isfinite(proxy.detach()).all()
        ):
            reasons.append(f"{name}:proxy_not_finite_float")
        elif not proxy.requires_grad:
            reasons.append(f"{name}:proxy_has_no_autograd_connection")
    if reasons:
        return TraceTangentRecoveryResult(None, tuple(reasons))

    numerator = conditioned.proxy.reshape(())
    denominator = baseline.proxy.reshape(())
    if numerator.device != denominator.device or numerator.dtype != denominator.dtype:
        return TraceTangentRecoveryResult(None, ("proxy_device_or_dtype_mismatch",))
    if bool((denominator.detach() <= 0).item()):
        return TraceTangentRecoveryResult(None, ("baseline_threshold_nonpositive",))
    ratio = numerator / denominator
    if not bool(torch.isfinite(ratio.detach()).item()):
        return TraceTangentRecoveryResult(None, ("ratio_nonfinite",))
    return TraceTangentRecoveryResult(ratio, ())


def trace_tangent_threshold_proxy(
    hard_threshold: float | torch.Tensor,
    probe_amplitude: float | torch.Tensor,
    voltage: torch.Tensor,
    amplitude_tangent: torch.Tensor,
    *,
    min_tangent_norm: float,
    mask: torch.Tensor | None = None,
    check_amplitude: float | torch.Tensor | None = None,
    check_voltage: torch.Tensor | None = None,
    max_relative_linearization_error: float | None = None,
) -> TraceTangentThresholdResult:
    r"""Attach a trace-based direction to a hard protocol threshold.

    The caller obtains ``hard_threshold`` using its complete hard activation
    protocol. It then runs a differentiable trial at the fixed, normally
    subthreshold ``probe_amplitude`` and supplies its voltage trace ``V`` and
    amplitude tangent ``J_A = partial V / partial A``. The tangent can be
    computed with a simulator JVP or approximated by a same-side finite
    difference in stimulus amplitude. Only the latter scalar input is
    differenced; no model parameter coordinate is differenced. The tangent
    is detached here so no second-order simulator derivative is needed. One
    reverse pass through the returned scalar proxy reaches every parameter
    connected to ``voltage``.

    For parameter ``q``, the backward slope is the weighted least-squares
    amplitude shift that offsets the trace perturbation,

    ``dT_proxy/dq = -<J_A, J_q>_w / <J_A, J_A>_w``.

    ``mask`` is a broadcastable, nonnegative sample weight; omitted means all
    samples. Units of the returned slope are amplitude units per unit of
    ``q``. The probe amplitude is an experimental choice. If its tensor has
    an autograd connection, its direct derivative is cancelled; model
    parameter gradients through the trace remain. The forward value is the
    detached hard threshold, irrespective of the trace's numerical value.

    Optionally supply a second same-side voltage trace and its amplitude to
    measure ``||V(A+delta)-V(A)-delta J_A||_w / ||delta J_A||_w``.
    A limit on this error rejects a locally nonlinear amplitude response.
    This check cannot establish that the trace direction follows the hard
    activation boundary; compare it against complete hard threshold searches
    when selecting probe amplitudes and trace windows.
    """
    if not isinstance(voltage, torch.Tensor) or not isinstance(
        amplitude_tangent, torch.Tensor
    ):
        raise TypeError("Voltage and amplitude tangent must be tensors.")
    if not voltage.is_floating_point() or not voltage.requires_grad:
        raise ValueError(
            "Voltage must be a floating-point tensor with an autograd connection."
        )
    if voltage.shape != amplitude_tangent.shape:
        raise ValueError("Voltage and amplitude tangent shapes must match.")
    if (
        voltage.device != amplitude_tangent.device
        or voltage.dtype != amplitude_tangent.dtype
    ):
        raise ValueError("Voltage and amplitude tangent must share device and dtype.")
    if not math.isfinite(min_tangent_norm) or min_tangent_norm <= 0:
        raise ValueError("Minimum tangent norm must be finite and positive.")
    if max_relative_linearization_error is not None and (
        not math.isfinite(max_relative_linearization_error)
        or max_relative_linearization_error < 0
    ):
        raise ValueError("Linearization error limit must be finite and nonnegative.")
    if (check_amplitude is None) != (check_voltage is None):
        raise ValueError("Check amplitude and voltage must be supplied together.")
    if max_relative_linearization_error is not None and check_voltage is None:
        raise ValueError("A linearization limit requires a second voltage trace.")

    if isinstance(hard_threshold, torch.Tensor):
        if hard_threshold.numel() != 1:
            raise ValueError("Hard threshold must be a scalar.")
        hard_value = float(hard_threshold.detach().item())
    else:
        # torch.as_tensor(Python float) follows the global default dtype and
        # can silently round a float64 threshold through float32.
        hard_value = float(hard_threshold)
    if not math.isfinite(hard_value):
        raise ValueError("Hard threshold must be finite.")
    amplitude = torch.as_tensor(
        probe_amplitude, dtype=voltage.dtype, device=voltage.device
    )
    if amplitude.numel() != 1 or not bool(torch.isfinite(amplitude.detach()).all()):
        raise ValueError("Probe amplitude must be a finite scalar.")

    if mask is None:
        weight = torch.ones_like(voltage)
    else:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Mask must be a tensor.")
        try:
            weight = torch.broadcast_to(mask.detach().to(voltage), voltage.shape)
        except RuntimeError as error:
            raise ValueError("Mask must broadcast to the voltage shape.") from error
        if not bool((torch.isfinite(weight) & (weight >= 0)).all()):
            raise ValueError("Mask weights must be finite and nonnegative.")
    selected = weight > 0
    count = int(selected.count_nonzero())
    if count == 0:
        raise ValueError("Mask selects no voltage samples.")
    if not bool(torch.isfinite(voltage.detach()).all()):
        raise ValueError("Voltage samples must be finite.")
    if not bool(torch.isfinite(amplitude_tangent.detach()[selected]).all()):
        raise ValueError("Selected amplitude-tangent samples must be finite.")

    tangent = torch.where(selected, amplitude_tangent.detach(), 0)
    tangent_square = (weight * tangent.square()).sum()
    tangent_norm = float(tangent_square.sqrt())
    relative_error = None
    secant_gain = None
    if check_voltage is not None:
        if (
            not isinstance(check_voltage, torch.Tensor)
            or check_voltage.shape != voltage.shape
            or check_voltage.device != voltage.device
            or check_voltage.dtype != voltage.dtype
        ):
            raise ValueError(
                "Check voltage must match the voltage shape, device, and dtype."
            )
        if not bool(torch.isfinite(check_voltage.detach()).all()):
            raise ValueError("Check-voltage samples must be finite.")
        second_amplitude = torch.as_tensor(
            check_amplitude,
            dtype=voltage.dtype,
            device=voltage.device,
        )
        if second_amplitude.numel() != 1 or not bool(
            torch.isfinite(second_amplitude.detach()).all()
        ):
            raise ValueError("Check amplitude must be a finite scalar.")
        delta = float(second_amplitude.detach() - amplitude.detach())
        if delta == 0:
            raise ValueError("Check and probe amplitudes must differ.")
        secant = torch.where(selected, check_voltage.detach() - voltage.detach(), 0)
        if tangent_norm > 0:
            secant_gain = float(
                (weight * tangent * secant).sum() / (delta * tangent_square)
            )
            residual = secant - delta * tangent
            relative_error = float((weight * residual.square()).sum().sqrt()) / (
                abs(delta) * tangent_norm
            )

    diagnostics = TraceTangentDiagnostics(
        count,
        tangent_norm,
        relative_error,
        secant_gain,
    )
    reasons = []
    if not math.isfinite(tangent_norm) or tangent_norm <= min_tangent_norm:
        reasons.append("amplitude_tangent_too_small")
    if (
        max_relative_linearization_error is not None
        and relative_error is not None
        and (
            not math.isfinite(relative_error)
            or relative_error > max_relative_linearization_error
        )
    ):
        reasons.append("amplitude_tangent_nonlinear")
    if reasons:
        return TraceTangentThresholdResult(None, diagnostics, tuple(reasons))

    # Stop tangent and weights. The scalar score uses one reverse pass for all
    # model parameters; +dA cancels the collocation amplitude's direct slope.
    trace_shift = (
        weight * tangent * (voltage - voltage.detach())
    ).sum() / tangent_square
    proxy = (
        voltage.new_tensor(hard_value)
        - trace_shift
        + (amplitude - amplitude.detach()).reshape(())
    )
    return TraceTangentThresholdResult(proxy, diagnostics, ())
