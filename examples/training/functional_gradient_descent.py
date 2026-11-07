#!/usr/bin/env python3
"""Fit HH conductances with Dendra's explicit functional Population API.

This is the functional counterpart of ``docs/basics/04_gradient_descent.ipynb``.
It uses the same model, stimulus, loss, Adam settings, and default 500-iteration
workload. A custom pure callback accumulates voltage mean-squared error without
retaining the candidate trace. By default, derivatives are produced by
``torch.func.grad_and_value`` over the eager functional rollout.
``--iterations`` and ``--tstop`` make shorter smoke runs convenient.
``--compiled-chunk-steps 4`` compiles four simulation steps and their loss
updates together, using ordinary autograd. Combine it with ``--checkpointed``
to recompute activations during backward; ``--chunklength`` independently sets
the number of steps between checkpoints. ``--compiled-step`` (also spelled
``--compile-step``) remains available for one-step compilation. The scalar
reducer reduces callback output memory in each mode.

Each simulation reconstructs its initial voltage and HH gates through the pure
functional initializer. The objective therefore includes initialization in its
transformable tensor program rather than reusing a detached state snapshot.
"""

from __future__ import annotations

import argparse
from functools import partial

import torch

# Import Dendra before PyTorch so an explicitly enabled Dendra bootstrap can
# configure TorchInductor before torch is initialized.
import dendra as dn
from dendra.models.mod import hh
from dendra.units import nA

GNABAR_QUERY = "gnabar"
GKBAR_QUERY = "gkbar"


class VoltageTraceMSE(dn.func.FunctionalCallback):
    """Accumulate full-trace voltage MSE with constant-size callback carry.

    ``target`` is immutable observed data shaped ``(frames, *model.shape)``.
    The initial boundary consumes frame zero, and every post-step update
    consumes the next frame. Only the sum and frame count evolve.
    """

    def __init__(self, target):
        if not torch.is_tensor(target):
            raise TypeError("target must be a Tensor")
        if target.ndim < 1 or target.shape[0] < 1:
            raise ValueError("target must contain at least one time frame")
        with torch.inference_mode(False):
            self.target = target.detach().clone()
        self.elements_per_frame = self.target[0].numel()

    def _validate_voltage(self, voltage):
        if self.target.shape[1:] != voltage.shape:
            raise ValueError(
                "target trailing shape must match functional voltage; "
                f"got {tuple(self.target.shape[1:])} and {tuple(voltage.shape)}"
            )
        if self.target.dtype != voltage.dtype or self.target.device != voltage.device:
            raise ValueError(
                "target and functional voltage must use the same dtype and device"
            )

    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        self._validate_voltage(voltage)
        carry = {
            "sse": (voltage - self.target[0]).square().sum(),
            "frames": voltage.new_ones((), dtype=torch.int64),
        }
        return carry, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        target = torch.index_select(
            self.target,
            0,
            carry["frames"].reshape(1),
        ).squeeze(0)
        return {
            "sse": carry["sse"] + (voltage - target).square().sum(),
            "frames": carry["frames"] + 1,
        }, None

    def finalize(self, carry, emissions):
        if emissions is not None:
            raise RuntimeError("VoltageTraceMSE does not emit samples")
        denominator = carry["frames"].to(carry["sse"].dtype)
        denominator = denominator * self.elements_per_frame
        return carry["sse"] / denominator


def build_model(*, gnabar: float, gkbar: float):
    model = dn.Unmyelinated(
        [2.0],
        celsius=6.3,
        v_init=-65.0,
        rhoa=100.0,
        integrator=dn.bwd_euler_ub(method="thomas", imem=False),
    ).double()
    model.insert(hh, gnabar=gnabar, gkbar=gkbar)
    model[0, 10].inject(dn.mono_rect(amp=2.0 * nA, delay=0.1, pw=0.2))
    model.train()
    model.initialize()
    return model


def simulate_callbacks(
    functional,
    tensors,
    parameters,
    callbacks,
    *,
    steps: int,
    chunklength: int,
    compiled_chunk=None,
    compiled_one_step=None,
    checkpointed: bool = False,
    host_runner: bool = False,
):
    """Initialize purely, simulate, and return finalized callback results.

    ``compiled_one_step`` is retained as an alias for older example callers.
    """
    if compiled_one_step is not None:
        if compiled_chunk is not None:
            raise ValueError("pass only one compiled chunk")
        compiled_chunk = compiled_one_step
    initialized = functional.initialize(
        parameters,
        tensors.constants,
        tensors.initialization,
    )
    if not host_runner and compiled_chunk is None and not checkpointed:
        _final_state, auxiliary = functional.prepare_and_rollout(
            initialized.parameters,
            initialized.constants,
            initialized.state,
            steps=steps,
            callbacks=callbacks,
        )
        return auxiliary["callbacks"]

    prepared = functional.prepare(
        initialized.parameters,
        initialized.constants,
    )
    step = partial(
        functional.step if compiled_chunk is None else compiled_chunk,
        initialized.parameters,
        prepared,
    )
    runner = dn.func.longrun_checkpointed if checkpointed else dn.func.longrun
    _final_state, auxiliary = runner(
        functional,
        step,
        initialized.state,
        steps * functional.dt,
        chunklength,
        callbacks=callbacks,
    )
    return auxiliary["callbacks"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--tstop", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument(
        "--chunklength",
        type=int,
        default=100,
        help="steps per host span, or steps between checkpoints with --checkpointed",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--report-every", type=int, default=50)
    compilation = parser.add_mutually_exclusive_group()
    compilation.add_argument(
        "--compiled-step",
        "--compile-step",
        dest="compiled_step",
        action="store_true",
        help="compile one simulation step and its callback updates",
    )
    compilation.add_argument(
        "--compiled-chunk-steps",
        type=int,
        help="compile this many simulation steps and callback updates together",
    )
    parser.add_argument("--checkpointed", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if args.iterations < 0:
        raise ValueError("iterations must be non-negative")
    if args.tstop <= 0 or args.dt <= 0:
        raise ValueError("tstop and dt must be positive")
    if args.chunklength <= 0:
        raise ValueError("chunklength must be positive")
    if args.compiled_chunk_steps is not None and args.compiled_chunk_steps <= 0:
        raise ValueError("compiled-chunk-steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.report_every <= 0:
        raise ValueError("report-every must be positive")


def resolve_steps(tstop: float, dt: float) -> int:
    """Resolve the static horizon required by a transform-native rollout."""

    steps = round(tstop / dt)
    if not torch.isclose(
        torch.tensor(steps * dt, dtype=torch.float64),
        torch.tensor(tstop, dtype=torch.float64),
        rtol=0.0,
        atol=max(torch.finfo(torch.float64).eps * abs(tstop) * 8, 1.0e-15),
    ):
        raise ValueError("tstop must be an integer multiple of dt in this example")
    return steps


def main():
    args = parse_args()
    validate_args(args)
    steps = resolve_steps(args.tstop, args.dt)

    reference = build_model(gnabar=0.12, gkbar=0.036)
    reference_functional, reference_tensors = dn.func.make_functional(
        reference,
        dt=args.dt,
    )
    reference_callbacks = reference_functional.make_callbacks(
        {"voltage": dn.func.Recorder(["v"])},
    )
    reference_parameters = reference_tensors.independent_parameters()
    with torch.no_grad():
        target = simulate_callbacks(
            reference_functional,
            reference_tensors,
            reference_parameters,
            reference_callbacks,
            steps=steps,
            chunklength=args.chunklength,
        )["voltage"]["v"]

    candidate = build_model(gnabar=0.05, gkbar=0.05)
    functional, tensors = dn.func.make_functional(candidate, dt=args.dt)
    loss_callbacks = functional.make_callbacks({"mse": VoltageTraceMSE(target)})
    trace_callbacks = functional.make_callbacks({"voltage": dn.func.Recorder(["v"])})
    parameter_names = (
        tensors.parameter_name(GNABAR_QUERY, within="model"),
        tensors.parameter_name(GKBAR_QUERY, within="model"),
    )
    parameters = tensors.independent_parameters(
        trainable=parameter_names,
        within="model",
    )
    trainable = {name: parameters[name] for name in parameter_names}
    fixed_parameters = {
        name: value for name, value in parameters.items() if name not in trainable
    }
    optimizer = torch.optim.Adam(trainable.values(), lr=args.learning_rate)
    compiled_steps = args.compiled_chunk_steps
    if args.compiled_step:
        compiled_steps = 1
    compiled_chunk = (
        functional.compile_rollout_chunk(compiled_steps)
        if compiled_steps is not None
        else None
    )
    use_torch_func = compiled_chunk is None and not args.checkpointed

    def objective(selected_parameters):
        local_parameters = {**fixed_parameters, **selected_parameters}
        results = simulate_callbacks(
            functional,
            tensors,
            local_parameters,
            loss_callbacks,
            steps=steps,
            chunklength=args.chunklength,
        )
        return results["mse"]

    gradient_and_value = torch.func.grad_and_value(objective)

    initial_trace = None
    if args.plot:
        with torch.no_grad():
            initial_parameters = {**fixed_parameters, **trainable}
            initial_trace = simulate_callbacks(
                functional,
                tensors,
                initial_parameters,
                trace_callbacks,
                steps=steps,
                chunklength=args.chunklength,
                compiled_chunk=compiled_chunk,
                checkpointed=args.checkpointed,
                host_runner=compiled_chunk is not None or args.checkpointed,
            )["voltage"]["v"]

    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        if use_torch_func:
            gradients, loss = gradient_and_value(trainable)
            for name, parameter in trainable.items():
                parameter.grad = gradients[name].detach()
        else:
            loss = simulate_callbacks(
                functional,
                tensors,
                parameters,
                loss_callbacks,
                steps=steps,
                chunklength=args.chunklength,
                compiled_chunk=compiled_chunk,
                checkpointed=args.checkpointed,
                host_runner=True,
            )["mse"]
            loss.backward()
        optimizer.step()
        if iteration % args.report_every == 0:
            print(f"Step {iteration:4d}, loss: {loss.item():.8g}")

    # Recompute after the final optimizer update.  This avoids pairing the
    # updated parameter values with a trace from immediately before that step.
    with torch.no_grad():
        final_parameters = {**fixed_parameters, **trainable}
        final_loss = simulate_callbacks(
            functional,
            tensors,
            final_parameters,
            loss_callbacks,
            steps=steps,
            chunklength=args.chunklength,
            compiled_chunk=compiled_chunk,
            checkpointed=args.checkpointed,
            host_runner=compiled_chunk is not None or args.checkpointed,
        )["mse"]
        final_trace = None
        if args.plot:
            final_trace = simulate_callbacks(
                functional,
                tensors,
                final_parameters,
                trace_callbacks,
                steps=steps,
                chunklength=args.chunklength,
                compiled_chunk=compiled_chunk,
                checkpointed=args.checkpointed,
                host_runner=compiled_chunk is not None or args.checkpointed,
            )["voltage"]["v"]

    print(f"Final loss: {final_loss.item():.8g}")
    print("Optimized parameters:")
    print(f"  gkbar:  {trainable[parameter_names[1]].item():.8g}")
    print(f"  gnabar: {trainable[parameter_names[0]].item():.8g}")
    print("Reference parameters:")
    print("  gkbar:  0.036")
    print("  gnabar: 0.12")

    if args.plot:
        import matplotlib.pyplot as plt

        assert initial_trace is not None and final_trace is not None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, compartment in zip(axes, (0, -1), strict=True):
            axis.plot(initial_trace[:, 0, compartment].cpu(), label="Initial")
            axis.plot(final_trace[:, 0, compartment].cpu(), label="Optimized")
            axis.plot(
                target[:, 0, compartment].cpu(),
                linestyle="--",
                label="Reference",
            )
            axis.set_xlabel("Time step")
            axis.set_ylabel("Voltage (mV)")
            axis.set_title(f"Compartment {compartment}")
            axis.legend()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
