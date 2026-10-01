#!/usr/bin/env python3
"""Differentiate activation threshold and chronaxie in a real HH axon.

This example keeps the experimental protocol explicit:

* Dendra's :class:`~dendra.models.instruments.Thresholder` runs the complete
  hard activation search at four pulse widths;
* a batched voltage replay and a stimulus-amplitude JVP provide local trace
  directions without differencing any model parameter;
* the hard-forward threshold proxies are fitted to the Weiss
  strength-duration relation to obtain chronaxie; and
* fresh hard threshold searches at perturbed sodium conductances provide an
  independent finite-difference check of both derivatives.

The trainable coordinate is the log of the physical HH sodium conductance.
The model and protocol choices belong to this example. The analysis helpers
accept any compatible differentiable voltage trace and hard protocol value.

Run from the Dendra repository root with::

    python examples/threshold_descriptor_gradients.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

import dendra as dn
from dendra.models.analysis import (
    chronaxie_from_threshold_proxies,
    compute_rheobase_chronaxie,
    trace_tangent_threshold_proxy,
)
from dendra.models.callbacks import Active, Recorder
from dendra.models.instruments import Thresholder
from dendra.models.mod import hh

DT_MS = 0.01
TSTOP_MS = 6.0
PULSE_DELAY_MS = 0.5
PULSE_WIDTHS_MS = (0.05, 0.1, 0.2, 0.5)
MONITORED_NODES = (5, 36)
BASE_GNABAR_S_PER_CM2 = 0.12


@dataclass(frozen=True)
class HardStrengthDurationMeasurement:
    """One complete hard strength-duration protocol."""

    lower_mA: torch.Tensor
    upper_mA: torch.Tensor
    threshold_mA: torch.Tensor
    rheobase_mA: float
    chronaxie_ms: float


@dataclass(frozen=True)
class TangentReplay:
    """Voltage data needed by ``trace_tangent_threshold_proxy``."""

    voltage_mV: torch.Tensor
    amplitude_tangent_mV_per_mA: torch.Tensor
    check_voltage_mV: torch.Tensor


class HHStrengthDurationProtocol:
    """A concrete hard and differentiable protocol for one HH axon family.

    One population lane is used for each pulse width. The lanes have identical
    geometry and share the scalar ``hh.gnabar`` parameter, while their waveform
    widths differ. This lets one simulator replay provide all amplitude
    tangents and retain one graph for the chronaxie derivative.
    """

    def __init__(self, *, hard_rtol: float) -> None:
        self.widths_ms = torch.tensor(PULSE_WIDTHS_MS, dtype=torch.float64)
        self.hard_rtol = hard_rtol

        with dn.ctx(JIT=0, REQUIRE_GRAD=1):
            self.model = dn.Myelinated(
                diameters=[6.0] * len(self.widths_ms),
                n_node=41,
                node_length=1.0,
                celsius=6.3,
                v_init=-65.0,
                dtype=torch.float64,
            )
            self.model.insert(hh)
            self.model.build()
            self.model.train()
            self.model.freeze()
            self.model.unfreeze("hh.gnabar")

        self.parameter = self.model.mech.hh.gnabar_param
        if self.parameter.numel() != 1 or not self.parameter.requires_grad:
            raise RuntimeError("Expected one differentiable shared HH gnabar.")

        self.field_mV_per_mA = dn.isotropic_point(z=200.0)(self.model)
        self.waveform = dn.mono_rect(
            amp=-1.0,
            pw=self.widths_ms[:, None],
            delay=PULSE_DELAY_MS,
        )
        criterion = Active(
            threshold=0.0,
            node_check=list(MONITORED_NODES),
            dt=DT_MS,
        )
        self.thresholder = Thresholder(
            self.model,
            criterion,
            space=self.field_mV_per_mA,
            time=self.waveform,
            lb=0.0,
            ub=0.2,
            rtol=hard_rtol,
            max_tries_thresh=40,
        )

    def set_log_sodium_conductance(self, log_scale: float) -> None:
        """Set ``gnabar = 0.12 exp(log_scale)`` in physical units."""

        if not math.isfinite(log_scale):
            raise ValueError("The log-conductance coordinate must be finite.")
        value = BASE_GNABAR_S_PER_CM2 * math.exp(log_scale)
        with torch.no_grad():
            self.parameter.fill_(value)

    def hard_strength_duration(
        self,
        log_sodium_conductance: float,
    ) -> HardStrengthDurationMeasurement:
        """Run a fresh complete hard search and fit its hard thresholds."""

        self.set_log_sodium_conductance(log_sodium_conductance)
        upper, lower = self.thresholder.calculate_thresholds(
            tstop=TSTOP_MS,
            dt=DT_MS,
            block_possible=True,
        )
        threshold = 0.5 * (lower + upper)
        if not bool(torch.isfinite(threshold).all()):
            raise RuntimeError("The complete hard threshold search failed.")
        fit = compute_rheobase_chronaxie(
            self.widths_ms,
            threshold,
            fit_domain="current",
            return_dict=True,
            eps=0.0,
        )
        return HardStrengthDurationMeasurement(
            lower_mA=lower,
            upper_mA=upper,
            threshold_mA=threshold,
            rheobase_mA=float(fit["rheobase"]),
            chronaxie_ms=float(fit["chronaxie_ms"]),
        )

    def hard_activity(self, amplitudes_mA: torch.Tensor) -> torch.Tensor:
        """Replay the same hard spike rule at specified amplitudes."""

        detector = Active(
            threshold=0.0,
            node_check=list(MONITORED_NODES),
            dt=DT_MS,
        )
        with torch.no_grad():
            self.model.initialize()
            self.model.run(
                extra=(self.field_mV_per_mA * amplitudes_mA[:, None], self.waveform),
                tstop=TSTOP_MS,
                dt=DT_MS,
                callbacks=[detector],
                progressbar=False,
            )
        return detector.is_active().reshape(-1)

    def run_voltage(self, amplitudes_mA: torch.Tensor) -> torch.Tensor:
        """Replay the protocol and record both sites used by ``Active``."""

        recorder = Recorder(["v"], node_indices=list(MONITORED_NODES))
        self.model.initialize()
        self.model.run(
            extra=(self.field_mV_per_mA * amplitudes_mA[:, None], self.waveform),
            tstop=TSTOP_MS,
            dt=DT_MS,
            callbacks=[recorder],
            progressbar=False,
        )
        return recorder.stack("v")

    def run_voltage_and_amplitude_tangent(
        self,
        probe_amplitudes_mA: torch.Tensor,
        check_amplitudes_mA: torch.Tensor,
    ) -> TangentReplay:
        r"""Return a gradient-bearing trace and ``partial V / partial A``.

        ``torch.autograd.functional.jvp`` differentiates the voltage replay
        with respect to the vector of stimulus amplitudes. Population lanes are
        independent, so a vector of ones yields the individual amplitude
        tangent in every lane. ``create_graph=False`` detaches this tangent:
        the threshold proxy needs only first derivatives of the simulator.

        Dendra's stateful runner can reuse internal buffers. The amplitude JVP
        and the detached linearity-check replay therefore run first. A fresh
        gradient-bearing replay runs last, and its graph is consumed before
        any subsequent simulation.
        """

        _, amplitude_tangent = torch.autograd.functional.jvp(
            self.run_voltage,
            (probe_amplitudes_mA,),
            (torch.ones_like(probe_amplitudes_mA),),
            create_graph=False,
            strict=True,
        )
        with torch.no_grad():
            check_voltage = self.run_voltage(check_amplitudes_mA).detach().clone()
        voltage = self.run_voltage(probe_amplitudes_mA)
        return TangentReplay(
            voltage_mV=voltage,
            amplitude_tangent_mV_per_mA=amplitude_tangent.detach(),
            check_voltage_mV=check_voltage,
        )


def relative_error(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Elementwise relative error with an explicit zero-reference fallback."""

    scale = torch.where(
        reference.abs() > 0,
        reference.abs(),
        torch.ones_like(reference),
    )
    return (estimate - reference).abs() / scale


def run_example(
    *,
    hard_rtol: float = 1e-5,
    finite_difference_step: float = 0.01,
    max_relative_error: float = 0.02,
) -> dict[str, object]:
    """Run the example and return values used by its console summary."""

    protocol = HHStrengthDurationProtocol(hard_rtol=hard_rtol)

    # These are independent complete hard protocols. They run before the
    # differentiable replay so they cannot overwrite its saved backward data.
    baseline = protocol.hard_strength_duration(0.0)
    minus = protocol.hard_strength_duration(-finite_difference_step)
    plus = protocol.hard_strength_duration(+finite_difference_step)
    hard_threshold_slopes = (plus.threshold_mA - minus.threshold_mA) / (
        2.0 * finite_difference_step
    )
    hard_chronaxie_slope = (plus.chronaxie_ms - minus.chronaxie_ms) / (
        2.0 * finite_difference_step
    )

    protocol.set_log_sodium_conductance(0.0)

    # Thresholder's returned bounds have operational meaning: the lower value
    # must remain inactive and the upper value must satisfy the complete hard
    # event rule. Check that contract before constructing a local direction.
    if bool(protocol.hard_activity(baseline.lower_mA).any()) or not bool(
        protocol.hard_activity(baseline.upper_mA).all()
    ):
        raise RuntimeError("The final hard threshold brackets are inconsistent.")

    # Probe safely below each certified inactive bound. A second, slightly
    # lower amplitude checks that the voltage response is locally linear in A.
    bracket_width = baseline.upper_mA - baseline.lower_mA
    probe_distance = torch.maximum(
        5.0 * bracket_width,
        1e-4 * baseline.threshold_mA,
    )
    probe_amplitudes = baseline.lower_mA - probe_distance
    check_amplitudes = probe_amplitudes - 0.2 * probe_distance
    if bool(protocol.hard_activity(probe_amplitudes).any()) or bool(
        protocol.hard_activity(check_amplitudes).any()
    ):
        raise RuntimeError("A supposedly subthreshold trace probe activated the axon.")

    replay = protocol.run_voltage_and_amplitude_tangent(
        probe_amplitudes,
        check_amplitudes,
    )
    sample_times_ms = (
        torch.arange(
            replay.voltage_mV.shape[0],
            dtype=replay.voltage_mV.dtype,
            device=replay.voltage_mV.device,
        )
        * DT_MS
    )

    threshold_results = []
    for lane, pulse_width_ms in enumerate(protocol.widths_ms):
        # The fixed mask uses both sites from the hard Active rule and samples
        # from pulse offset onward. It selects protocol data; the analysis
        # helper does not assume a model, stimulus, site, or time window.
        post_pulse = (sample_times_ms >= PULSE_DELAY_MS + float(pulse_width_ms))[
            :, None
        ]
        result = trace_tangent_threshold_proxy(
            baseline.threshold_mA[lane],
            probe_amplitudes[lane],
            replay.voltage_mV[:, lane, :],
            replay.amplitude_tangent_mV_per_mA[:, lane, :],
            min_tangent_norm=1e-9,
            mask=post_pulse,
            check_amplitude=check_amplitudes[lane],
            check_voltage=replay.check_voltage_mV[:, lane, :],
            max_relative_linearization_error=0.05,
        )
        if not result.valid:
            raise RuntimeError(
                f"Trace proxy at {float(pulse_width_ms):g} ms was rejected: "
                f"{result.rejection_reasons}"
            )
        threshold_results.append(result)

    chronaxie = chronaxie_from_threshold_proxies(
        protocol.widths_ms,
        threshold_results,
    )
    if not chronaxie.valid:
        raise RuntimeError(f"Chronaxie fit was rejected: {chronaxie.rejection_reasons}")
    torch.testing.assert_close(
        chronaxie.thresholds.detach(),
        baseline.threshold_mA,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        chronaxie.chronaxie_ms.detach(),
        torch.as_tensor(
            baseline.chronaxie_ms,
            dtype=chronaxie.chronaxie_ms.dtype,
            device=chronaxie.chronaxie_ms.device,
        ),
        rtol=1e-12,
        atol=1e-12,
    )

    # The simulator parameter is physical gnabar. Multiplication by its
    # baseline value applies d/d(log gnabar) = gnabar * d/d(gnabar).
    autograd_threshold_slopes = []
    for result in threshold_results:
        (physical_slope,) = torch.autograd.grad(
            result.proxy,
            protocol.parameter,
            retain_graph=True,
        )
        autograd_threshold_slopes.append(
            BASE_GNABAR_S_PER_CM2 * physical_slope.detach()
        )
    autograd_threshold_slopes = torch.stack(autograd_threshold_slopes)
    (physical_chronaxie_slope,) = torch.autograd.grad(
        chronaxie.chronaxie_ms,
        protocol.parameter,
    )
    autograd_chronaxie_slope = float(
        BASE_GNABAR_S_PER_CM2 * physical_chronaxie_slope.detach()
    )

    threshold_errors = relative_error(
        autograd_threshold_slopes,
        hard_threshold_slopes,
    )
    chronaxie_difference = abs(autograd_chronaxie_slope - hard_chronaxie_slope)
    chronaxie_error = (
        chronaxie_difference / abs(hard_chronaxie_slope)
        if hard_chronaxie_slope != 0.0
        else chronaxie_difference
    )
    if bool((threshold_errors > max_relative_error).any()) or (
        chronaxie_error > max_relative_error
    ):
        raise RuntimeError(
            "The trace-derived slopes did not match the independent hard "
            "finite differences at the requested tolerance."
        )

    return {
        "pulse_widths_ms": protocol.widths_ms,
        "hard_thresholds_mA": baseline.threshold_mA,
        "proxy_thresholds_mA": chronaxie.thresholds.detach(),
        "hard_chronaxie_ms": baseline.chronaxie_ms,
        "proxy_chronaxie_ms": float(chronaxie.chronaxie_ms.detach()),
        "autograd_threshold_slopes": autograd_threshold_slopes,
        "hard_fd_threshold_slopes": hard_threshold_slopes,
        "threshold_relative_errors": threshold_errors,
        "autograd_chronaxie_slope": autograd_chronaxie_slope,
        "hard_fd_chronaxie_slope": hard_chronaxie_slope,
        "chronaxie_relative_error": chronaxie_error,
        "linearity_errors": torch.tensor(
            [
                result.diagnostics.relative_linearization_error
                for result in threshold_results
            ],
            dtype=torch.float64,
        ),
    }


def print_summary(result: dict[str, object]) -> None:
    """Print a compact comparison in the parameter coordinate being trained."""

    print("\nDerivatives with respect to log physical HH gnabar")
    print(
        "pulse width   threshold    autograd slope    hard FD slope    rel. error\n"
        "                              (mA/log-unit)    (mA/log-unit)"
    )
    for width, threshold, automatic, finite, error in zip(
        result["pulse_widths_ms"],
        result["hard_thresholds_mA"],
        result["autograd_threshold_slopes"],
        result["hard_fd_threshold_slopes"],
        result["threshold_relative_errors"],
        strict=True,
    ):
        print(
            f"{float(width):8.3f} ms  {float(threshold):9.6f} mA  "
            f"{float(automatic):+13.6f}  {float(finite):+13.6f}  "
            f"{100 * float(error):8.4f}%"
        )
    print(
        f"\nchronaxie: {result['hard_chronaxie_ms']:.6f} ms "
        "(hard and proxy fits agree to numerical precision)"
    )
    print(
        "chronaxie slope: "
        f"autograd {result['autograd_chronaxie_slope']:+.6f} ms/log-unit, "
        f"hard FD {result['hard_fd_chronaxie_slope']:+.6f} ms/log-unit, "
        f"relative error {100 * result['chronaxie_relative_error']:.4f}%"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hard-rtol",
        type=float,
        default=1e-5,
        help="relative width of each complete hard threshold bracket",
    )
    parser.add_argument(
        "--finite-difference-step",
        type=float,
        default=0.01,
        help="central step in log physical HH gnabar",
    )
    parser.add_argument(
        "--max-relative-error",
        type=float,
        default=0.02,
        help="fail if an autograd/finite-difference slope error exceeds this",
    )
    args = parser.parse_args()
    if args.hard_rtol <= 0 or args.finite_difference_step <= 0:
        parser.error("Search tolerance and finite-difference step must be positive.")
    if args.max_relative_error <= 0:
        parser.error("Maximum relative error must be positive.")
    return args


def main() -> None:
    args = parse_args()
    result = run_example(
        hard_rtol=args.hard_rtol,
        finite_difference_step=args.finite_difference_step,
        max_relative_error=args.max_relative_error,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
