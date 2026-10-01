#!/usr/bin/env python3
"""Tune repeated HH firing with a macroscopic descriptor, without a target trace.

The example increases late, repeated firing under a fixed current by changing
two physical log coordinates: the speeds of the HH sodium activation and
inactivation gates.  A fresh differentiable simulation supplies one VJP per
accepted update.  Complete hard simulations decide whether proposed updates
improve the measured firing frequency without reducing the spike count.

Run the full example from the Dendra repository root with::

    python examples/macroscopic_descriptor_training.py

For a short smoke run, use::

    python examples/macroscopic_descriptor_training.py \
        --updates 1 --tstop-ms 60 --objective-start-ms 20 --skip-holdout --no-jit

The HH parameter choices belong to this example.  The descriptor accepts any
graph-connected voltage tensor with shape ``(time, neuron, compartment)``.
"""

from __future__ import annotations

import argparse
import gc
import math

import torch

import dendra as dn
from dendra.models.analysis import differentiable_spike_timing, hard_spike_timing
from dendra.models.mod import hh
from dendra.units import nA

DTYPE = torch.float64
DIAMETER_UM = 20.0
LENGTH_UM = 20.0
CURRENT_DENSITY_UA_CM2 = 10.0
AREA_CM2 = math.pi * DIAMETER_UM * LENGTH_UM * 1e-8
CURRENT_NA = CURRENT_DENSITY_UA_CM2 * AREA_CM2 * 1000.0
RATE_DEFAULTS = {
    "am1": 0.1,
    "am2": 4.0,
    "ah1": 0.07,
    "ah2": 1.0,
}
RATE_PAIRS = (("am1", "am2"), ("ah1", "ah2"))
COORDINATE_NAMES = (
    "log sodium activation speed",
    "log sodium inactivation speed",
)


def scalar(value: torch.Tensor) -> float:
    """Convert a finite scalar tensor to a Python float."""

    result = float(value.detach())
    if not math.isfinite(result):
        raise RuntimeError("The descriptor produced a nonfinite scalar.")
    return result


class HHFiringProtocol:
    """One model-specific protocol around model-agnostic analysis functions."""

    def __init__(
        self,
        *,
        dt_ms: float,
        tstop_ms: float,
        objective_start_ms: float,
        holdout_ms: float,
        chunk_steps: int,
        jit: bool,
    ) -> None:
        self.dt_ms = dt_ms
        self.tstop_ms = tstop_ms
        self.objective_start_ms = objective_start_ms
        self.chunk_steps = chunk_steps
        stimulus_duration_ms = max(tstop_ms, holdout_ms) + 10.0

        with dn.ctx(JIT=int(jit)):
            self.model = dn.SingleCompartment(
                N=1,
                C=1,
                celsius=6.3,
                cm=1.0,
                v_init=-65.0,
                dtype=DTYPE,
            )
            self.model.diam.fill_(DIAMETER_UM)
            self.model.dx.fill_(LENGTH_UM)
            self.model.insert(hh)
            self.model[..., 0].inject(
                dn.mono_rect(
                    amp=CURRENT_NA * nA,
                    delay=10.0,
                    pw=stimulus_duration_ms,
                )
            )
            self.model.build()
            self.model.train()

        mechanism = self.model.mech.hh
        self.gates = mechanism.DE["mhn"]
        with torch.no_grad():
            for name, value in RATE_DEFAULTS.items():
                getattr(self.gates, f"{name}_param").fill_(value)
            for name, value in {
                "gnabar": 0.12,
                "gkbar": 0.036,
                "gl": 0.0003,
                "ena": 50.0,
                "ek": -77.0,
                "el": -54.3,
            }.items():
                getattr(mechanism, f"{name}_param").fill_(value)

        for names in RATE_PAIRS:
            for name in names:
                self.model.unfreeze(f"hh.mhn.{name}")

        self.log_coordinates = torch.zeros(2, dtype=DTYPE)
        self.set_log_coordinates(self.log_coordinates)

    def set_log_coordinates(self, coordinates: torch.Tensor) -> None:
        """Set paired rate coefficients from physical log coordinates."""

        coordinates = torch.as_tensor(coordinates, dtype=DTYPE)
        if coordinates.shape != (2,) or not bool(torch.isfinite(coordinates).all()):
            raise ValueError("Expected two finite physical log coordinates.")
        self.log_coordinates = coordinates.detach().clone()
        with torch.no_grad():
            for coordinate, names in zip(
                self.log_coordinates,
                RATE_PAIRS,
                strict=True,
            ):
                scale = math.exp(float(coordinate))
                for name in names:
                    getattr(self.gates, f"{name}_param").fill_(
                        RATE_DEFAULTS[name] * scale
                    )

    def _timing(
        self,
        voltage: torch.Tensor,
        *,
        start_ms: float,
        stop_ms: float,
        hard: bool,
    ):
        timing_function = hard_spike_timing if hard else differentiable_spike_timing
        timing = timing_function(
            voltage,
            self.dt_ms,
            V_spk=0.0,
            time_window_ms=(start_ms, stop_ms),
        )
        if not bool(timing["valid"][0]):
            raise RuntimeError(
                "The selected timing window needs at least two complete spikes."
            )
        return timing

    def measure(
        self,
        *,
        tstop_ms: float | None = None,
        objective_start_ms: float | None = None,
    ) -> dict[str, object]:
        """Run the complete hard protocol at the current parameters."""

        stop = self.tstop_ms if tstop_ms is None else tstop_ms
        start = (
            self.objective_start_ms
            if objective_start_ms is None
            else objective_start_ms
        )
        recorder = dn.callbacks.Recorder(["v"], node_indices=[0])
        counter = dn.callbacks.APCount(
            threshold=0.0,
            node_check=[0],
            dt=self.dt_ms,
        )
        with torch.no_grad():
            self.model.initialize()
            self.model.run(
                tstop=stop,
                dt=self.dt_ms,
                callbacks=[counter, recorder],
                progressbar=False,
            )
            timing = self._timing(
                recorder.stack("v"),
                start_ms=start,
                stop_ms=stop,
                hard=True,
            )
        return {
            "frequency_hz": scalar(timing["span_frequency_hz"][0]),
            "count": int(counter.n.item()),
            "branch_signature": timing["branch_signature"],
        }

    def frequency_gradient(self) -> tuple[torch.Tensor, dict[str, object]]:
        """Return one simulator VJP in the two physical log coordinates."""

        self.model.initialize()
        functional, _ = dn.func.make_functional(self.model, dt=self.dt_ms)
        callbacks = functional.make_callbacks(
            {
                "trace": dn.func.Recorder(["v"], node_indices=[0]),
                "hard_count": dn.func.APCount(
                    threshold=0.0,
                    node_check=[0],
                    dt=self.dt_ms,
                ),
            }
        )
        _, results = self.model.longrun_checkpointed(
            tstop=self.tstop_ms,
            chunklength=self.chunk_steps,
            dt=self.dt_ms,
            safe_checkpoint=True,
            restore_state_after_backward=True,
            functional_callbacks=callbacks,
            progressbar=False,
        )
        timing = self._timing(
            results["trace"]["v"],
            start_ms=self.objective_start_ms,
            stop_ms=self.tstop_ms,
            hard=False,
        )
        frequency = timing["span_frequency_hz"][0]
        parameters = tuple(
            getattr(self.gates, f"{name}_param")
            for names in RATE_PAIRS
            for name in names
        )
        physical_gradients = torch.autograd.grad(frequency, parameters)
        log_gradients = torch.stack(
            tuple(
                sum(
                    parameter.detach() * gradient.detach()
                    for parameter, gradient in zip(
                        parameters[2 * index : 2 * index + 2],
                        physical_gradients[2 * index : 2 * index + 2],
                        strict=True,
                    )
                )
                for index in range(2)
            )
        )
        if not bool(torch.isfinite(log_gradients).all()) or not bool(
            torch.linalg.vector_norm(log_gradients) > 0
        ):
            raise RuntimeError("The selected descriptor has no usable gradient.")
        diagnostics = {
            "frequency_hz": scalar(frequency),
            "count": int(results["hard_count"].item()),
            "branch_signature": timing["branch_signature"],
        }
        del (
            callbacks,
            frequency,
            functional,
            parameters,
            physical_gradients,
            results,
            timing,
        )
        gc.collect()
        return log_gradients, diagnostics


def optimize(protocol: HHFiringProtocol, *, updates: int, step_norm: float):
    """Use normalized gradient ascent with hard-protocol backtracking."""

    current = protocol.measure()
    initial = dict(current)
    print(
        "initial: "
        f"frequency={current['frequency_hz']:.6f} Hz, "
        f"APCount={current['count']}"
    )

    for update in range(1, updates + 1):
        gradient, replay = protocol.frequency_gradient()
        if (
            not math.isclose(
                float(replay["frequency_hz"]),
                float(current["frequency_hz"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or int(replay["count"]) != int(current["count"])
            or replay["branch_signature"] != current["branch_signature"]
        ):
            raise RuntimeError("Differentiable replay differs from the hard rerun.")
        direction = gradient / torch.linalg.vector_norm(gradient)
        previous_coordinates = protocol.log_coordinates.clone()
        accepted = None

        for candidate_norm in (
            step_norm,
            step_norm / 2,
            step_norm / 4,
            step_norm / 8,
        ):
            proposal = previous_coordinates + candidate_norm * direction
            protocol.set_log_coordinates(proposal)
            candidate = protocol.measure()
            if float(candidate["frequency_hz"]) > float(
                current["frequency_hz"]
            ) and int(candidate["count"]) >= int(current["count"]):
                accepted = candidate
                break

        if accepted is None:
            protocol.set_log_coordinates(previous_coordinates)
            print(f"update {update}: no improving step found; stopping")
            break

        pair_changed = accepted["branch_signature"] != current["branch_signature"]
        current = accepted
        print(
            f"update {update}: frequency={current['frequency_hz']:.6f} Hz, "
            f"APCount={current['count']}, step_norm={candidate_norm:.5g}, "
            f"crossing_pair_changed={pair_changed}"
        )

    return initial, current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=9)
    parser.add_argument("--step-norm", type=float, default=0.08)
    parser.add_argument("--dt-ms", type=float, default=0.025)
    parser.add_argument("--tstop-ms", type=float, default=160.0)
    parser.add_argument("--objective-start-ms", type=float, default=40.0)
    parser.add_argument("--holdout-ms", type=float, default=400.0)
    parser.add_argument("--chunk-steps", type=int, default=250)
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args()
    if args.updates < 0:
        parser.error("--updates must be nonnegative")
    if args.step_norm <= 0 or args.dt_ms <= 0 or args.tstop_ms <= 0:
        parser.error("step norm, dt, and tstop must be positive")
    if not 0 <= args.objective_start_ms < args.tstop_ms:
        parser.error("objective start must lie inside the training recording")
    if args.holdout_ms < args.tstop_ms:
        parser.error("holdout must be at least as long as the training recording")
    if args.chunk_steps <= 0:
        parser.error("--chunk-steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    protocol = HHFiringProtocol(
        dt_ms=args.dt_ms,
        tstop_ms=args.tstop_ms,
        objective_start_ms=args.objective_start_ms,
        holdout_ms=args.holdout_ms,
        chunk_steps=args.chunk_steps,
        jit=not args.no_jit,
    )
    baseline_coordinates = protocol.log_coordinates.clone()
    initial, final = optimize(
        protocol,
        updates=args.updates,
        step_norm=args.step_norm,
    )

    relative_change = (
        float(final["frequency_hz"]) / float(initial["frequency_hz"]) - 1.0
    )
    print(f"training frequency change: {100 * relative_change:.3f}%")
    for name, value in zip(
        COORDINATE_NAMES,
        protocol.log_coordinates,
        strict=True,
    ):
        print(f"{name}: exp(q)={math.exp(float(value)):.6f}")

    if not args.skip_holdout:
        holdout_start_ms = max(
            args.objective_start_ms,
            args.holdout_ms - 120.0,
        )
        final_coordinates = protocol.log_coordinates.clone()
        protocol.set_log_coordinates(baseline_coordinates)
        heldout_initial = protocol.measure(
            tstop_ms=args.holdout_ms,
            objective_start_ms=holdout_start_ms,
        )
        protocol.set_log_coordinates(final_coordinates)
        heldout_final = protocol.measure(
            tstop_ms=args.holdout_ms,
            objective_start_ms=holdout_start_ms,
        )
        print(
            f"held-out {holdout_start_ms:g}-{args.holdout_ms:g} ms: "
            f"{heldout_initial['frequency_hz']:.6f} -> "
            f"{heldout_final['frequency_hz']:.6f} Hz; "
            f"full APCount {heldout_initial['count']} -> "
            f"{heldout_final['count']}"
        )


if __name__ == "__main__":
    main()
