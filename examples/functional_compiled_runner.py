#!/usr/bin/env python3
"""Run a compiled functional simulation with explicit state and little setup.

Run from the repository root::

    python examples/functional_compiled_runner.py --execution scan
    python examples/functional_compiled_runner.py --execution while --threads 4
    python examples/functional_compiled_runner.py --execution scan --grad

Recording is selected explicitly through the callback mapping. Inductor uses
the C++ loop wrapper by default for CPU inference. The first call
includes compilation; reuse the runner for warm execution. ``--backend
aot_eager`` exercises capture without native kernel generation. Scan requires
Dendra's verified PyTorch 2.14.0 or 2.14.1 implementation. This is a usage example,
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
    args = parser.parse_args()
    if args.grad and args.execution != "scan":
        parser.error("--grad requires --execution scan; while is inference-only")
    torch.set_num_threads(args.threads)

    model = make_model(gradients=args.grad)
    original_voltage = model.v.detach().clone()
    runner = dn.func.compile(
        model,
        dt=0.01,
        steps=8,
        execution=args.execution,
        callbacks={"trace": dn.func.Recorder(("v", "t"))},
        inference=not args.grad,
        trainable=("gnabar",) if args.grad else (),
        backend=args.backend,
    )

    if args.grad:
        parameters = [p for p in runner.parameters.values() if p.requires_grad]
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

        # Repeating runner() restarts from the same captured initial state.
        repeated, _ = runner()
        torch.testing.assert_close(
            repeated["integrator"]["v"], final["integrator"]["v"]
        )

        # Explicit continuation leaves the source Population unchanged.
        continued, _ = runner(final, callback_state=auxiliary["callbacks"].state)
        assert torch.isfinite(continued["integrator"]["v"]).all()
        print(
            f"recorded {voltage.shape[0]} voltage frames; explicit continuation succeeded"
        )

    torch.testing.assert_close(model.v, original_voltage)
    print(f"execution strategy: {runner.execution_report().strategy}")


if __name__ == "__main__":
    main()
