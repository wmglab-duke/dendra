"""End-to-end training check for a promoted macroscopic descriptor."""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.analysis import (
    differentiable_action_potential_width,
    hard_action_potential_width,
)
from dendra.models.mod import hh
from dendra.units import nA

DT_MS = 0.005
TSTOP_MS = 8.0
WIDTH_OPTIONS = {
    "baseline_mean_mode": "continuous_time",
    "baseline_margin_mV": 10.0,
    "full_width_post_ms": 5.0,
}


def _run_and_record(model: dn.SingleCompartment, *, track_gradients: bool):
    model.initialize()
    recorder = dn.callbacks.Recorder(["v"], node_indices=[0])
    counter = dn.callbacks.APCount(
        threshold=0.0,
        node_check=[0],
        dt=DT_MS,
    )
    context = torch.enable_grad() if track_gradients else torch.no_grad()
    with context:
        model.run(
            tstop=TSTOP_MS,
            dt=DT_MS,
            callbacks=[recorder, counter],
            progressbar=False,
        )
    return recorder.stack("v"), int(counter.n.item())


def test_hh_ap_width_gradient_improves_complete_hard_measurement():
    """One descriptor VJP supplies a useful update for a real HH simulation."""

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=1,
            celsius=6.3,
            cm=1.0,
            v_init=-65.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.diam.fill_(20.0)
        model.dx.fill_(20.0)
        model.insert(hh)
        model[..., 0].inject(dn.mono_rect(amp=0.5 * nA, delay=1.5, pw=0.2))
        model.train()
        model.unfreeze("hh.gkbar")
        model.initialize()

    voltage, baseline_count = _run_and_record(model, track_gradients=True)
    soft = differentiable_action_potential_width(
        voltage,
        DT_MS,
        **WIDTH_OPTIONS,
    )
    baseline_hard = hard_action_potential_width(
        voltage.detach(),
        DT_MS,
        **WIDTH_OPTIONS,
    )

    assert baseline_count == 1
    assert bool(baseline_hard["has_crossing"][0, 0])
    assert bool(soft["branch_matches_hard"][0, 0])
    assert int(soft["branch_signature"]["half_valid"][0, 0]) == 1
    assert int(soft["branch_signature"]["half_rise_kind"][0, 0]) == 0
    assert int(soft["branch_signature"]["half_fall_kind"][0, 0]) == 0
    torch.testing.assert_close(
        soft["width_ms"].detach(),
        baseline_hard["width_ms"],
        rtol=0.0,
        atol=1e-12,
    )

    parameter = model.mech.hh.gkbar_param
    (gradient,) = torch.autograd.grad(soft["width_ms"].sum(), parameter)
    assert bool(torch.isfinite(gradient))
    assert abs(float(gradient)) > 1e-6

    # Gradient ascent asks for a wider AP.  This step changes gkbar by about 4%,
    # while the hard event guard below ensures that the same single spike remains.
    with torch.no_grad():
        parameter.add_(1e-4 * gradient)

    updated_voltage, updated_count = _run_and_record(
        model,
        track_gradients=False,
    )
    updated_hard = hard_action_potential_width(
        updated_voltage,
        DT_MS,
        **WIDTH_OPTIONS,
    )

    assert updated_count == baseline_count == 1
    assert bool(updated_hard["has_crossing"][0, 0])
    assert bool(torch.isfinite(updated_hard["width_ms"][0]))
    assert updated_hard["width_ms"][0] > baseline_hard["width_ms"][0] + 0.005
