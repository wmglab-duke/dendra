"""Infer sinusoid frequency and timing from ordinary Dendra voltage recordings.

    python examples/fit_sinusoid_voltage.py --output-dir /tmp/voltage-fit
    python examples/fit_sinusoid_voltage.py --membrane hh --noise-std-mv 0.5 --output-dir /tmp/voltage-hh

The cable, membrane channels, amplitude, phase, and initial state are known.
Only frequency, delay, and one cutoff parameter are fitted. Gaussian observation
noise is sampled once and shared across all starts. Fitting and selection use
observed voltage alone; clean traces are retained only for evaluation. Abrupt
edges use surrogate gradients. JIT is enabled; use --no-jit for debugging.
This is an exploratory fitting example, including local optimization failures.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import dendra as dn
from dendra.models.callbacks import Recorder
from dendra.models.mod import hh, pas
from dendra.units import Hz, nA

DTYPE = torch.float64
TSTOP = 16.0
V_INIT = -65.0
REFERENCE = {"frequency_hz": 300.0, "delay_ms": 2.08, "stop_ms": 11.44}
MODEL_DEFAULTS = {
    "passive": {"amplitude_na": 0.05, "dt": 0.125, "tau": 0.25, "celsius": 37.0},
    "hh": {"amplitude_na": 0.2, "dt": 0.0625, "tau": 0.5, "celsius": 6.3},
}
MEMBRANE_PARAMETERS = {
    "passive": {"g": 0.001, "e": V_INIT},
    "hh": {
        "gnabar": 0.12,
        "gkbar": 0.036,
        "gl": 0.0003,
        "ena": 50.0,
        "ek": -77.0,
        "el": -54.3,
    },
}


def build_model(
    frequency_hz,
    delay_ms,
    stop_ms,
    *,
    cutoff,
    tau,
    membrane="passive",
    amplitude_na=None,
):
    """Build a fixed passive or Hodgkin-Huxley cable driven at compartment 0."""
    defaults = MODEL_DEFAULTS[membrane]
    if amplitude_na is None:
        amplitude_na = defaults["amplitude_na"]
    end = stop_ms if cutoff == "off" else stop_ms - delay_ms
    waveform = dn.sin(
        amp=amplitude_na * nA,
        freq=frequency_hz * Hz,
        phase=0.4,
        delay=delay_ms,
        tau=tau,
        **{cutoff: end},
    )
    waveform.requires_grad_(False)
    model = dn.Unmyelinated(
        [2.0],
        L=250.0,
        dx=50.0,
        v_init=V_INIT,
        celsius=defaults["celsius"],
        dtype=DTYPE,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(hh if membrane == "hh" else pas, **MEMBRANE_PARAMETERS[membrane])
    model[:, 0].inject(waveform)
    model.requires_grad_(False)
    model.train()
    return model, waveform


def simulate(model, recorder, *, dt):
    """Start each trial from the same rest state and retain the trace's graph."""
    recorder.reset()
    model.initialize()
    model.run(tstop=TSTOP, dt=dt, callbacks=[recorder])
    # numpy() would detach. stack() keeps the path back to waveform parameters.
    return recorder.stack("v")


def physical_parameters(waveform, cutoff):
    delay = waveform.delay.detach().item()
    end = getattr(waveform, cutoff).detach().item()
    return {
        "frequency_hz": waveform.freq.detach().item() / Hz,
        "delay_ms": delay,
        "cutoff_ms": end,
        "stop_ms": end if cutoff == "off" else delay + end,
    }


def noisy_observations(clean, *, std_mv, seed):
    """Draw observation noise once without changing the global random state."""
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = (
        torch.randn(
            clean.shape, generator=generator, device=clean.device, dtype=clean.dtype
        )
        * std_mv
    )
    return (clean.detach() + noise).detach()


def project_parameters(waveform, *, cutoff, dt):
    """Keep guesses within the acquisition window, without reference parameters."""
    with torch.no_grad():
        waveform.freq.clamp_(min=1e-6, max=0.5 / dt)  # kHz, sampling Nyquist bound
        waveform.delay.clamp_(0.0, TSTOP - dt)
        end = getattr(waveform, cutoff)
        if cutoff == "off":
            end.clamp_(max=TSTOP)
            end.copy_(torch.maximum(end, waveform.delay + dt))
        else:
            end.clamp_(min=dt)
            end.copy_(torch.minimum(end, TSTOP - waveform.delay))


def fit_voltage(
    target,
    *,
    initial,
    cutoff,
    dt,
    tau,
    iterations,
    record_nodes,
    membrane="passive",
    amplitude_na=None,
):
    """Fit an observed voltage tensor without access to reference parameters."""
    model, waveform = build_model(
        **initial, cutoff=cutoff, tau=tau, membrane=membrane, amplitude_na=amplitude_na
    )
    names = ("freq", "delay", cutoff)
    parameters = [getattr(waveform, name) for name in names]
    for parameter in parameters:
        parameter.requires_grad_(True)
    # Frequency is in kHz and timing is in ms; use separate learning rates.
    optimizer = torch.optim.Adam(
        [
            {"params": [parameters[0]], "lr": 0.0015},
            {"params": [parameters[1]], "lr": 0.03},
            {"params": [parameters[2]], "lr": 0.06},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    recorder = Recorder(["v"], node_indices=list(record_nodes))
    target = target.detach()
    scale = (target - V_INIT).square().mean()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Target must have a finite, nonzero voltage response.")

    history = []
    best_loss = math.inf
    best_iteration = 0
    best_parameters = None
    for iteration in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        voltage = simulate(model, recorder, dt=dt)
        if voltage.shape != target.shape:
            raise ValueError("Target and candidate recording shapes differ.")
        # The objective sees voltage alone: no waveform or parameter loss.
        loss = (voltage - target).square().mean() / scale
        value = loss.detach().item()
        if not math.isfinite(value):
            raise RuntimeError(f"Nonfinite voltage loss at iteration {iteration}.")
        history.append(
            {
                "iteration": iteration,
                "loss": value,
                **physical_parameters(waveform, cutoff),
            }
        )
        if iteration == 0:
            initial_trace = voltage.detach().clone()
        if value < best_loss:
            best_loss = value
            best_iteration = iteration
            best_parameters = [parameter.detach().clone() for parameter in parameters]
        if iteration == iterations:
            break
        loss.backward()
        for name, parameter in zip(names, parameters, strict=True):
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Missing or nonfinite gradient for {name}.")
        optimizer.step()
        project_parameters(waveform, cutoff=cutoff, dt=dt)
        scheduler.step()

    # Re-evaluate the selected parameters instead of plotting a stale pre-step trace.
    with torch.no_grad():
        for parameter, best in zip(parameters, best_parameters, strict=True):
            parameter.copy_(best)
        fitted_trace = simulate(model, recorder, dt=dt).detach().clone()
    return {
        "initial": dict(initial),
        "fitted": physical_parameters(waveform, cutoff),
        "initial_loss": history[0]["loss"],
        "best_loss": best_loss,
        "best_iteration": best_iteration,
        "rmse_mv": (fitted_trace - target).square().mean().sqrt().item(),
        "history": history,
        "initial_trace": initial_trace,
        "fitted_trace": fitted_trace,
    }


def save_results(directory, summary, clean, observed, runs, *, dt, record_nodes):
    """Save numeric results and ordinary, shareable scientific plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    best = runs[summary["selected_start"]]
    times = np.arange(clean.shape[0]) * dt  # Recorder includes the initial frame.
    reference = clean[:, 0].numpy()
    observations = observed[:, 0].numpy()
    initial = best["initial_trace"][:, 0].numpy()
    fitted = best["fitted_trace"][:, 0].numpy()
    np.savez(
        directory / "traces.npz",
        time_ms=times,
        reference_mv=reference,
        observed_mv=observations,
        initial_mv=initial,
        fitted_mv=fitted,
        residual_mv=observations - fitted,
        record_nodes=np.asarray(record_nodes),
    )

    fig, axes = plt.subplots(
        len(record_nodes) + 2,
        1,
        figsize=(9, 2.25 * (len(record_nodes) + 2)),
        constrained_layout=True,
    )
    for index, node in enumerate(record_nodes):
        axis = axes[index]
        if summary["noise_std_mv"] > 0:
            axis.scatter(
                times,
                observations[:, index],
                color="#7f8b93",
                s=7,
                alpha=0.45,
                label="Noisy observations",
                rasterized=True,
            )
        axis.plot(
            times, reference[:, index], color="black", lw=1.6, label="Clean reference"
        )
        axis.plot(
            times,
            initial[:, index],
            color="#ce822d",
            ls=":",
            label=f"Initial (stop {best['initial']['stop_ms']:g} ms)",
        )
        axis.plot(times, fitted[:, index], color="#16769b", ls="--", label="Fitted")
        axis.set_title(
            f"Compartment {node}" + (" (injection site)" if node == 0 else "")
        )
        axis.set(xlabel="Time (ms)", ylabel="Voltage (mV)")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2, frameon=False)
    for index, run in enumerate(runs):
        axes[-2].semilogy(
            [row["iteration"] for row in run["history"]],
            [max(row["loss"], 1e-18) for row in run["history"]],
            label=f"Initial stop {run['initial']['stop_ms']:g} ms",
        )
    axes[-2].set(
        xlabel="Optimizer updates", ylabel="Observed voltage MSE\n(normalized)"
    )
    axes[-2].legend(frameon=False)
    axes[-2].grid(alpha=0.2)
    for index, node in enumerate(record_nodes):
        axes[-1].plot(
            times,
            observations[:, index] - fitted[:, index],
            lw=0.8,
            alpha=0.7,
            label=f"Compartment {node}",
        )
    if summary["noise_std_mv"] > 0:
        width = 2 * summary["noise_std_mv"]
        axes[-1].axhspan(-width, width, color="grey", alpha=0.15, label="±2 noise std")
    axes[-1].axhline(0, color="black", lw=0.6)
    axes[-1].set(xlabel="Time (ms)", ylabel="Observed − fitted (mV)")
    axes[-1].legend(
        frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0)
    )
    axes[-1].grid(alpha=0.2)
    membrane_label = "Hodgkin–Huxley" if summary["membrane"] == "hh" else "Passive"
    fig.suptitle(
        f"{membrane_label} membrane · observation noise {summary['noise_std_mv']:g} mV"
    )
    fig.savefig(directory / "voltage_fit.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    for axis, key, label in zip(
        axes,
        ("frequency_hz", "delay_ms", "stop_ms"),
        ("Frequency (Hz)", "Delay (ms)", "Effective stop (ms)"),
        strict=True,
    ):
        for run in runs:
            axis.plot(
                [row["iteration"] for row in run["history"]],
                [row[key] for row in run["history"]],
                label=f"Initial stop {run['initial']['stop_ms']:g} ms",
            )
        axis.axhline(
            summary["reference"][key], color="black", ls="--", label="Reference"
        )
        axis.set(xlabel="Optimizer updates", ylabel=label)
        axis.grid(alpha=0.2)
    lower, upper = summary["indistinguishable_stop_interval_ms"]
    axes[-1].axhspan(lower, upper, color="grey", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Parameter recovery; grey band marks one cutoff sampling interval")
    fig.savefig(directory / "parameter_fit.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membrane", choices=tuple(MODEL_DEFAULTS), default="passive")
    parser.add_argument(
        "--noise-std-mv",
        type=float,
        default=0.0,
        help="Independent Gaussian observation noise standard deviation (mV)",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Local observation-noise seed; reuse it for comparisons",
    )
    parser.add_argument(
        "--amplitude-na",
        type=float,
        help="Known current amplitude: passive 0.05 nA, HH 0.2 nA",
    )
    parser.add_argument(
        "--no-jit", action="store_true", help="Disable Dendra JIT for debugging"
    )
    parser.add_argument("--cutoff", choices=("off", "off_after"), default="off_after")
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument(
        "--dt", type=float, help="Simulation timestep: passive 0.125 ms, HH 0.0625 ms"
    )
    parser.add_argument(
        "--tau", type=float, help="Edge surrogate width: passive 0.25 ms, HH 0.5 ms"
    )
    parser.add_argument("--initial-frequency-hz", type=float, default=270.0)
    parser.add_argument("--initial-delay-ms", type=float, default=1.75)
    parser.add_argument(
        "--initial-stop-ms", type=float, nargs="+", default=[10.25, 12.0]
    )
    parser.add_argument("--record-nodes", type=int, nargs="+", default=[0, 2, 3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    defaults = MODEL_DEFAULTS[args.membrane]
    for name in ("dt", "tau", "amplitude_na"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    if not math.isfinite(args.noise_std_mv) or args.noise_std_mv < 0:
        parser.error("--noise-std-mv must be finite and nonnegative")
    if not 0 <= args.noise_seed < 2**63:
        parser.error("--noise-seed must be an integer in [0, 2**63)")
    if not math.isfinite(args.amplitude_na) or args.amplitude_na <= 0:
        parser.error("--amplitude-na must be finite and positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if not math.isfinite(args.dt) or args.dt <= 0 or args.dt > TSTOP:
        parser.error("--dt must be finite, positive, and no larger than 16 ms")
    if not math.isclose(TSTOP / args.dt, round(TSTOP / args.dt), abs_tol=1e-9):
        parser.error("--dt must divide the 16 ms recording interval")
    if REFERENCE["frequency_hz"] >= 500 / args.dt:
        parser.error(
            "--dt must resolve the 300 Hz reference below the sampling Nyquist frequency"
        )
    if not math.isfinite(args.tau) or args.tau <= 0:
        parser.error("--tau must be finite and positive")
    if (
        not math.isfinite(args.initial_frequency_hz)
        or not 0.001 <= args.initial_frequency_hz <= 500 / args.dt
    ):
        parser.error(
            "--initial-frequency-hz must be between 0.001 Hz and the sampling Nyquist frequency"
        )
    if (
        not math.isfinite(args.initial_delay_ms)
        or not 0 <= args.initial_delay_ms <= TSTOP - args.dt
    ):
        parser.error("--initial-delay-ms must leave at least one timestep before 16 ms")
    if any(
        not math.isfinite(stop) or not args.initial_delay_ms + args.dt <= stop <= TSTOP
        for stop in args.initial_stop_ms
    ):
        parser.error(
            "Initial stops must be at least one timestep after delay and at most 16 ms"
        )
    if len(set(args.record_nodes)) != len(args.record_nodes):
        parser.error("--record-nodes must not contain duplicates")

    began = time.perf_counter()
    with dn.ctx(JIT=not args.no_jit, REQUIRE_GRAD=0, DTYPE=DTYPE):
        reference, _ = build_model(
            **REFERENCE,
            cutoff=args.cutoff,
            tau=args.tau,
            membrane=args.membrane,
            amplitude_na=args.amplitude_na,
        )
        if any(node not in range(reference.nc) for node in args.record_nodes):
            parser.error(f"--record-nodes must be between 0 and {reference.nc - 1}")
        recorder = Recorder(["v"], node_indices=args.record_nodes)
        with torch.no_grad():
            clean = simulate(reference, recorder, dt=args.dt).detach().clone()
        observed = noisy_observations(
            clean, std_mv=args.noise_std_mv, seed=args.noise_seed
        )
        print(
            f"{args.membrane} membrane, JIT {'off' if args.no_jit else 'on'}, "
            f"noise std {args.noise_std_mv:g} mV (seed {args.noise_seed})",
            flush=True,
        )
        runs = []
        for stop in args.initial_stop_ms:
            initial = {
                "frequency_hz": args.initial_frequency_hz,
                "delay_ms": args.initial_delay_ms,
                "stop_ms": stop,
            }
            run = fit_voltage(
                observed,
                initial=initial,
                cutoff=args.cutoff,
                dt=args.dt,
                tau=args.tau,
                iterations=args.iterations,
                record_nodes=args.record_nodes,
                membrane=args.membrane,
                amplitude_na=args.amplitude_na,
            )
            runs.append(run)
            print(
                f"Initial stop {stop:g} ms: normalized voltage MSE "
                f"{run['initial_loss']:.6g} -> {run['best_loss']:.6g}",
                flush=True,
            )
    selected = min(range(len(runs)), key=lambda index: runs[index]["best_loss"])
    best = runs[selected]
    # Clean-reference diagnostics are evaluated only after observed-loss selection.
    for run in runs:
        run["clean_rmse_mv"] = (
            (run["fitted_trace"] - clean).square().mean().sqrt().item()
        )
        run["observed_rmse_mv"] = run["rmse_mv"]
    stop_bin = math.ceil(REFERENCE["stop_ms"] / args.dt)
    input_times = torch.arange(round(TSTOP / args.dt), dtype=DTYPE) * args.dt
    summary = {
        "torch_version": torch.__version__,
        "membrane": args.membrane,
        "celsius": defaults["celsius"],
        "membrane_parameters": MEMBRANE_PARAMETERS[args.membrane],
        "amplitude_na": args.amplitude_na,
        "phase_rad": 0.4,
        "v_init_mv": V_INIT,
        "jit_enabled": not args.no_jit,
        "noise_std_mv": args.noise_std_mv,
        "noise_seed": args.noise_seed,
        "realized_noise_rmse_mv": (observed - clean).square().mean().sqrt().item(),
        "reference_peak_mv": clean.amax(dim=0)[0].tolist(),
        "selection_objective": "normalized_observed_voltage_mse",
        "cutoff_parameter": args.cutoff,
        "dt_ms": args.dt,
        "tau_ms": args.tau,
        "iterations_per_start": args.iterations,
        "record_nodes": args.record_nodes,
        "model_compartments": reference.nc,
        "model_length_um": reference.L,
        "reference": dict(REFERENCE),
        "selected_start": selected,
        "indistinguishable_stop_interval_ms": [
            (stop_bin - 1) * args.dt,
            stop_bin * args.dt,
        ],
        "stop_samples_match": torch.equal(
            input_times < REFERENCE["stop_ms"], input_times < best["fitted"]["stop_ms"]
        ),
        "onset_samples_match": torch.equal(
            input_times >= REFERENCE["delay_ms"],
            input_times >= best["fitted"]["delay_ms"],
        ),
        "elapsed_seconds": time.perf_counter() - began,
        "runs": [
            {
                key: value
                for key, value in run.items()
                if key not in {"initial_trace", "fitted_trace"}
            }
            for run in runs
        ],
    }
    print(
        f"Selected start {selected + 1} by observed voltage MSE; "
        f"observed RMSE {best['observed_rmse_mv']:.6g} mV; "
        f"clean-reference RMSE {best['clean_rmse_mv']:.6g} mV"
    )
    for key, label in (
        ("frequency_hz", "Frequency (Hz)"),
        ("delay_ms", "Delay (ms)"),
        ("stop_ms", "Effective stop (ms)"),
    ):
        print(
            f"{label}: reference {REFERENCE[key]:g}, fitted {best['fitted'][key]:.6f}"
        )
    print(f"Fitted {args.cutoff}: {best['fitted']['cutoff_ms']:.6f} ms")
    lower, upper = summary["indistinguishable_stop_interval_ms"]
    print(f"Hard cutoff samples cannot distinguish stops in ({lower:g}, {upper:g}] ms")
    if args.output_dir:
        save_results(
            args.output_dir,
            summary,
            clean,
            observed,
            runs,
            dt=args.dt,
            record_nodes=args.record_nodes,
        )
        print(f"Saved results to {args.output_dir}")
    return summary


if __name__ == "__main__":
    main()
