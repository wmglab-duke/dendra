#!/usr/bin/env python3
"""Find voltage-response thresholds with native and compiled Populations.

Run from the repository root::

    python examples/training/functional_thresholding.py
    python examples/training/functional_thresholding.py --backend aot_eager
    python examples/training/functional_thresholding.py --execution while
    python examples/training/functional_thresholding.py --backend inductor

Two passive cables receive the same temporal pulse with different spatial
gains. The threshold is the amplitude that makes a checked compartment cross
-64 mV. This small example demonstrates the search and callback contracts;
use an excitable model and its activation criterion for physiological spike
thresholds.
"""

from __future__ import annotations

import argparse

import torch

import dendra as dn
from dendra.models.instruments import Thresholder
from dendra.models.mod import pas

DT = 0.025
DURATION_MS = 0.5


def make_model():
    morphology = dn.Morphology(rhoa=35.4, cm=1.0)
    morphology.section("cable", L=60.0, diam=20.0, nseg=3)
    model = dn.Cable.from_morphology(
        morphology,
        N=2,
        v_init=-65.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas, g=0.001, e=-65.0)
    model.eval()
    model.initialize()
    return model


def make_criterion():
    return dn.callbacks.Active(threshold=-64.0, node_check=[0], dt=DT)


def run_demo(*, backend="eager", execution="scan"):
    source, native = make_model(), make_model()
    initial_voltage = source.v.detach().clone()
    initial_time = source.t.detach().clone()
    field = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=source.dtype())
    waveform = dn.mono_rect(amp=1.0, delay=0.05, pw=0.2)
    configured_activity = make_criterion()

    # region compiled_threshold
    program = dn.compile(
        source,
        dt=DT,
        duration=DURATION_MS,
        chunk_steps=4,
        extra=dn.VarInput(),
        callbacks={
            "activation": configured_activity,
            "peak_voltage": dn.Recorder(("v",), max_only=True),
        },
        inference=True,
        initialization="snapshot",
        backend=backend,
        execution=execution,
    )
    compiled_search = Thresholder(
        program,
        active="activation",
        peak="peak_voltage",
        space=field,
        time=waveform,
        ub=4.0,
        rtol=0.01,
    )
    upper, lower = compiled_search.calculate_thresholds(block_possible=True)
    # endregion compiled_threshold

    # Each amplitude trial starts from the program's snapshot. Neither the
    # native source nor the callback configuration accumulates trial history.
    torch.testing.assert_close(source.v, initial_voltage, rtol=0, atol=0)
    torch.testing.assert_close(source.t, initial_time, rtol=0, atol=0)
    assert configured_activity.record is None
    assert configured_activity.state_cache is None
    assert configured_activity.i == 0
    assert torch.isfinite(upper).all() and torch.isfinite(lower).all()
    assert (upper >= lower).all() and ((upper - lower) / upper < 0.01).all()

    native_search = Thresholder(
        native,
        active=make_criterion(),
        space=field,
        time=waveform,
        ub=4.0,
        rtol=0.01,
    )
    native_upper, native_lower = native_search.calculate_thresholds(
        tstop=DURATION_MS, dt=DT, block_possible=True
    )
    torch.testing.assert_close(upper, native_upper, rtol=0, atol=0)
    torch.testing.assert_close(lower, native_lower, rtol=0, atol=0)
    print(f"{execution}: native and compiled threshold brackets agree")
    print(f"lower: {lower.tolist()}")
    print(f"upper: {upper.tolist()}")
    return upper, lower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("eager", "aot_eager", "inductor"), default="eager"
    )
    parser.add_argument("--execution", choices=("scan", "while"), default="scan")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        run_demo(backend=arguments.backend, execution=arguments.execution)


if __name__ == "__main__":
    main()
