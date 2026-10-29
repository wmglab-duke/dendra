#!/usr/bin/env python3
"""Run a compiled functional simulation with explicit state and little setup.

Run from the repository root::

    python examples/functional_compiled_runner.py --execution scan
    python examples/functional_compiled_runner.py --chunk-steps 3
    python examples/functional_compiled_runner.py --execution while --threads 4
    python examples/functional_compiled_runner.py --execution scan --grad
    python examples/functional_compiled_runner.py --grad --checkpoint-every 3
    python examples/functional_compiled_runner.py --grad --chunk-steps 3 --checkpoint-every 5
    python examples/functional_compiled_runner.py --extra --grad --checkpoint-every 4
    python examples/functional_compiled_runner.py --initialization fresh --grad --checkpoint-every 4

Recording is selected explicitly through the callback mapping. Inductor uses
the C++ loop wrapper by default for CPU inference. The first call
includes compilation; reuse the runner for warm execution. ``--backend
aot_eager`` exercises capture without native kernel generation. Both modes use
Dendra's PyTorch 2.14+ baseline and compiler-contract checks. This is a usage example,
not a performance benchmark or a convergence demonstration.
"""

from __future__ import annotations

import argparse

import torch

import dendra as dn
from dendra.models.mod import hh


def make_model(*, gradients: bool):
    with dn.ctx(JIT=0, REQUIRE_GRAD=int(gradients)):
        model = dn.Unmyelinated(
            [2.0],
            L=2.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -58.0, -62.0]),
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(hh)
        model.initialize()
        if gradients:
            model.train()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", choices=("scan", "while"), default="scan")
    parser.add_argument(
        "--backend", choices=("inductor", "aot_eager"), default="inductor"
    )
    parser.add_argument("--threads", choices=(1, 4), type=int, default=1)
    parser.add_argument("--grad", action="store_true", help="differentiate scan twice")
    parser.add_argument(
        "--initialization", choices=("snapshot", "fresh"), default="snapshot"
    )
    parser.add_argument(
        "--extra", action="store_true", help="bind an extracellular field and waveform"
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        help="compiled chunk width; the call still advances eight steps",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        help="recompute training scan spans of this many steps during backward",
    )
    args = parser.parse_args()
    if args.grad and args.execution != "scan":
        parser.error("--grad requires --execution scan; while is inference-only")
    if args.chunk_steps is not None:
        if args.chunk_steps < 1:
            parser.error("--chunk-steps must be a positive integer")
        if args.execution == "while" and min(8, args.chunk_steps) < 4:
            parser.error("--execution while requires a chunk width of at least four")
    if args.checkpoint_every is not None:
        if args.checkpoint_every < 1:
            parser.error("--checkpoint-every must be a positive integer")
        if not args.grad:
            parser.error("--checkpoint-every requires --grad")
    torch.set_num_threads(args.threads)

    model = make_model(gradients=args.grad)
    original_voltage = model.v.detach().clone()
    extra = None
    trainable = ["gnabar"] if args.grad else []
    if args.extra:
        field = torch.linspace(
            -1.0,
            1.0,
            model.v.numel(),
            device=model.v.device,
            dtype=model.v.dtype,
        ).reshape(model.shape)  # Toy unit-current field [mV/mA].
        extra = (field, dn.sin(amp=0.1, freq=0.7, phase=0.2))  # Amplitude [mA].
        if args.grad:
            trainable.extend(
                (
                    "stimulation.extra.contacts.0.field",
                    "stimulation.extra.waveforms.0.amp",
                    "stimulation.extra.waveforms.0.freq",
                    "stimulation.extra.waveforms.0.phase",
                )
            )
    runner = dn.compile(
        model,
        dt=0.01,
        steps=8,
        chunk_steps=args.chunk_steps,
        extra=extra,
        execution=args.execution,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        inference=not args.grad,
        initialization=args.initialization,
        checkpoint_every=args.checkpoint_every,
        trainable=trainable,
        backend=args.backend,
    )

    if args.grad:
        parameters = [p for p in runner.parameters.values() if p.requires_grad]
        if args.initialization == "fresh":
            # Initialization-only leaves are explicit, not runtime parameters.
            parameters.append(runner.initialization_inputs.v_init.requires_grad_())
        optimizer = torch.optim.SGD(parameters, lr=1e-5)
        for iteration in range(2):
            optimizer.zero_grad()
            final, auxiliary = runner()
            loss = auxiliary["callbacks"]["trace"]["v"].square().mean()
            loss.backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in parameters
            )
            optimizer.step()
            print(f"iteration {iteration}: loss={loss.item():.8g}")
    else:
        # No functionalization, prepare/bind calls, or inference context needed.
        final, auxiliary = runner()
        voltage = auxiliary["callbacks"]["trace"]["v"]
        assert voltage.shape[0] == 9  # Initial frame plus eight accepted steps.
        assert torch.isfinite(voltage).all() and not voltage.requires_grad

        # Repeating runner() restarts: snapshot by default, freshly initialized
        # from the same inputs when initialization='fresh'.
        repeated, _ = runner()
        torch.testing.assert_close(
            repeated["integrator"]["v"], final["integrator"]["v"]
        )

        # Explicit continuation leaves the source Population unchanged.
        continued, _ = runner(auxiliary["continuation"])
        assert torch.isfinite(continued["integrator"]["v"]).all()
        print(
            f"recorded {voltage.shape[0]} voltage frames; explicit continuation succeeded"
        )

    torch.testing.assert_close(model.v, original_voltage)
    print(f"steps per call: 8; compiled chunk width: {runner.kernel.steps}")
    print(f"execution strategy: {runner.execution_report().strategy}")


if __name__ == "__main__":
    main()
