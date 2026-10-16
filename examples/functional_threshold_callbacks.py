#!/usr/bin/env python3
"""Use named threshold reducers with a compiled functional simulation.

Run from the repository root::

    python examples/functional_threshold_callbacks.py --execution scan
    python examples/functional_threshold_callbacks.py --execution while --threads 4
    python examples/functional_threshold_callbacks.py --execution while --record

The default stores only reducer carry, not a full voltage trace. ``--record``
adds a Recorder to the callback mapping. CPU inference uses the C++ wrapper by
default; the first call includes compilation. ``--backend aot_eager`` exercises capture
without native code generation. Scan requires Dendra's verified PyTorch 2.14.0
or 2.14.1 implementation. This small model illustrates callback semantics, not
physiological excitation thresholds or a performance benchmark.
"""

from __future__ import annotations

import argparse

import torch

import dendra as dn
from dendra.models.mod import hh


def make_model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=0):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=2.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -58.0, -62.0]),
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="thomas", imem=False),
        )
        model.insert(hh)
        model.initialize()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", choices=("scan", "while"), default="scan")
    parser.add_argument(
        "--backend", choices=("inductor", "aot_eager"), default="inductor"
    )
    parser.add_argument("--threads", choices=(1, 4), type=int, default=1)
    parser.add_argument(
        "--record", action="store_true", help="also retain the voltage trace"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = make_model()
    original_voltage = model.v.detach().clone()
    selected = {"threshold": -63.0, "node_check": [0, -1]}
    callbacks = {
        "active": dn.func.Active(**selected),
        "site_quorum": dn.func.ActiveAL(**selected, at_least=2),
        "event_quorum": dn.func.ActiveALCount(**selected, at_least=2),
        "by_site": dn.func.Active(**selected, partition=(1, 1)),
        "counts": dn.func.APCount(**selected),
    }
    if args.record:
        callbacks["trace"] = dn.func.Recorder(("v",))
    runner = dn.func.compile(
        model,
        dt=0.01,
        steps=8,
        execution=args.execution,
        backend=args.backend,
        callbacks=callbacks,
    )

    final, auxiliary = runner()
    results = auxiliary["callbacks"]
    assert results["active"].shape == (2,)
    assert results["by_site"].shape == (2, 2)
    assert results["counts"].shape == (2, 2)
    assert results["active"].dtype == torch.bool
    assert results["counts"].dtype == torch.int64
    if args.record:
        assert results["trace"]["v"].shape[0] == 9  # Initial frame plus eight steps.
    else:
        assert "trace" not in results

    # Count checked sites that have fired, versus all events across those sites.
    torch.testing.assert_close(results["active"], (results["counts"] > 0).any(dim=-1))
    torch.testing.assert_close(
        results["site_quorum"], (results["counts"] > 0).sum(dim=-1) >= 2
    )
    torch.testing.assert_close(
        results["event_quorum"], results["counts"].sum(dim=-1) >= 2
    )

    # Continuing explicitly preserves crossing latches, counts, and callback time.
    continued, continued_auxiliary = runner(final, callback_state=results.state)
    assert torch.isfinite(continued["integrator"]["v"]).all()
    assert (continued_auxiliary["callbacks"]["counts"] >= results["counts"]).all()

    # A fresh runner() restarts both model state and callback detection history.
    repeated, repeated_auxiliary = runner()
    torch.testing.assert_close(repeated["integrator"]["v"], final["integrator"]["v"])
    torch.testing.assert_close(
        repeated_auxiliary["callbacks"]["counts"], results["counts"]
    )
    torch.testing.assert_close(model.v, original_voltage)

    print(f"active: {results['active'].tolist()}")
    print(f"counts per checked site: {results['counts'].tolist()}")
    print(f"execution strategy: {runner.execution_report().strategy}")


if __name__ == "__main__":
    main()
