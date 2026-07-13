"""Temperature-sensitive HH state and current oracles against NEURON."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.mod import hh
from dendra.units import nA

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DT = 0.00625
N_STEPS = 800
TSTOP = DT * N_STEPS
CONVERGENCE_TSTOP = 1.5
CONVERGENCE_DTS = (2.0 * DT, DT, 0.5 * DT)
NON_EQUILIBRIUM_DTS = (4.0 * DT, 2.0 * DT, DT)
NON_EQUILIBRIUM_REFERENCE_DT = DT / 8.0
NON_EQUILIBRIUM_GATES = (0.2, 0.25, 0.55)
V_INIT = -65.0
STIMULUS_AMPLITUDE_NA = 8.0
STIMULUS_DELAY = 0.5
STIMULUS_DURATION = 0.75
DTYPE = torch.float64
STATE_NAMES = ("m", "h", "n")
TRACE_NAMES = ("v", "m", "h", "n", "ina", "ik", "il")

# Dendra advances the gates at v[n] before solving v[n + 1], while NEURON's
# secondorder=0 path solves voltage with the old gates and then advances them at
# v[n + 1].  Consequently, the same-time gate traces contain the expected
# first-order Lie-splitting difference even though their voltage-solve phases
# agree (Dendra gate[n + 1] corresponds to NEURON gate[n]).
STATE_ERROR_BOUNDS = {
    "m": (0.04, 0.01),
    "h": (0.015, 0.004),
    "n": (0.015, 0.004),
}

# Lookup-table interpolation is disabled suite-wide, so the analytic HH model
# and the quantities evaluated at the shared voltage-solve phase should agree
# closely.  These bounds retain modest cross-platform numerical headroom.
SOLVE_PHASE_ERROR_BOUNDS = {
    "v": (5.0e-5, 1.0e-5),
    "ina": (1.0e-6, 2.0e-7),
    "ik": (1.0e-6, 2.0e-7),
    "il": (5.0e-8, 1.0e-8),
}

# Bounds for the finest Dendra trajectory in the deliberately non-equilibrium
# convergence oracle below.  Unlike the same-dt comparisons above, these are
# errors against a much finer NEURON reference rather than phase-alignment
# tolerances.
NON_EQUILIBRIUM_FINE_ERROR_BOUNDS = {
    "v": (0.07, 0.02),
    "m": (0.0012, 0.0005),
    "h": (0.0009, 0.00045),
    "n": (0.0006, 0.0003),
    "ina": (0.00025, 0.00007),
    "ik": (0.00025, 0.00007),
    "il": (0.000021, 0.000006),
}


def _number_of_steps(dt: float, tstop: float) -> int:
    n_steps = round(tstop / dt)
    assert tstop == pytest.approx(n_steps * dt, abs=1.0e-14)
    return n_steps


def _hh_currents(
    v: np.ndarray, m: np.ndarray, h_gate: np.ndarray, n: np.ndarray
) -> dict[str, np.ndarray]:
    """Evaluate canonical HH currents from explicitly selected state samples."""
    return {
        "ina": 0.12 * m**3 * h_gate * (v - 50.0),
        "ik": 0.036 * n**4 * (v + 77.0),
        "il": 0.0003 * (v + 54.3),
    }


def _run_neuron(
    celsius: float,
    *,
    dt: float = DT,
    tstop: float = TSTOP,
    initial_gates: tuple[float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    assert h.usetable_hh == 0, "NEURON HH rate-table interpolation must be disabled"
    n_steps = _number_of_steps(dt, tstop)
    section = h.Section(name=f"hh_temperature_{celsius:g}")
    section.L = 100.0
    section.diam = 100.0
    section.Ra = 35.4
    section.cm = 1.0
    section.nseg = 1
    section.insert("hh")
    segment = section(0.5)

    stimulus = h.IClamp(segment)
    stimulus.delay = STIMULUS_DELAY
    stimulus.dur = STIMULUS_DURATION
    stimulus.amp = STIMULUS_AMPLITUDE_NA

    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = celsius
    h.dt = dt
    h.finitialize(V_INIT)
    if initial_gates is not None:
        segment.hh.m, segment.hh.h, segment.hh.n = initial_gates

    rows = [
        (
            float(segment.v),
            float(segment.hh.m),
            float(segment.hh.h),
            float(segment.hh.n),
        )
    ]
    for _ in range(n_steps):
        h.fadvance()
        rows.append(
            (
                float(segment.v),
                float(segment.hh.m),
                float(segment.hh.h),
                float(segment.hh.n),
            )
        )

    values = np.asarray(rows)
    traces = {name: values[:, index] for index, name in enumerate(("v", "m", "h", "n"))}

    # After fadvance(), NEURON's exposed current fields retain its earlier
    # BREAKPOINT evaluation, whereas Dendra's methods evaluate at the current
    # model state.  Compare the voltage-solve phase explicitly instead: for
    # secondorder=0 NEURON solves v[k] with gates[k - 1], while Dendra solves it
    # with its already-advanced gates[k].  The initial sample uses the common
    # initialized gates.
    solve_m = np.concatenate((traces["m"][:1], traces["m"][:-1]))
    solve_h = np.concatenate((traces["h"][:1], traces["h"][:-1]))
    solve_n = np.concatenate((traces["n"][:1], traces["n"][:-1]))
    traces.update(_hh_currents(traces["v"], solve_m, solve_h, solve_n))
    return traces


def _run_dendra(
    celsius: float,
    *,
    dt: float = DT,
    tstop: float = TSTOP,
    initial_gates: tuple[float, float, float] | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    n_steps = _number_of_steps(dt, tstop)
    model = dn.SingleCompartment(
        N=1,
        C=1,
        celsius=celsius,
        v_init=V_INIT,
        rhoa=35.4,
        cm=1.0,
        dtype=DTYPE,
    )
    model.diam.fill_(100.0)
    model.dx.fill_(100.0)
    model.insert(hh)
    model[..., 0].inject(
        dn.mono_rect(
            amp=STIMULUS_AMPLITUDE_NA * nA,
            delay=STIMULUS_DELAY,
            pw=STIMULUS_DURATION,
        )
    )
    model.eval()
    model.initialize()
    mechanism = model.mech.hh
    if initial_gates is not None:
        mechanism.m.fill_(initial_gates[0])
        mechanism.h.fill_(initial_gates[1])
        mechanism.n.fill_(initial_gates[2])

    recorder = dn.callbacks.RecorderLambda(
        {
            "v": lambda population: population.v.detach().clone(),
            "m": lambda _population: mechanism.m.detach().clone(),
            "h": lambda _population: mechanism.h.detach().clone(),
            "n": lambda _population: mechanism.n.detach().clone(),
            "ina": lambda population: mechanism.ina(population.v).detach().clone(),
            "ik": lambda population: mechanism.ik(population.v).detach().clone(),
            "il": lambda population: mechanism.il(population.v).detach().clone(),
        }
    )
    model.run(tstop=tstop, dt=dt, callbacks=[recorder])
    traces = {name: recorder.numpy(name).reshape(n_steps + 1) for name in TRACE_NAMES}
    q10 = float(mechanism.DE["mhn"].q10())
    return traces, q10


@pytest.mark.parametrize("celsius", [6.3, 22.0, 34.0])
def test_hh_temperature_gates_currents_and_voltage_match_neuron(celsius):
    expected = _run_neuron(celsius)
    actual, q10 = _run_dendra(celsius)

    assert q10 == pytest.approx(3.0 ** ((celsius - 6.3) / 10.0), rel=2.0e-15)
    assert np.ptp(expected["v"]) > 10.0
    for name in TRACE_NAMES:
        assert actual[name].shape == expected[name].shape == (N_STEPS + 1,)
        assert np.isfinite(actual[name]).all()
        assert np.isfinite(expected[name]).all()
        difference = actual[name] - expected[name]
        max_abs = float(np.max(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(difference * difference)))
        bounds = STATE_ERROR_BOUNDS if name in STATE_NAMES else SOLVE_PHASE_ERROR_BOUNDS
        max_bound, rmse_bound = bounds[name]
        assert max_abs < max_bound and rmse < rmse_bound, (
            f"HH {name} mismatch at {celsius:g} C: "
            f"max_abs={max_abs:.6g}, rmse={rmse:.6g}"
        )
        if name in STATE_NAMES:
            phase_difference = actual[name][1:] - expected[name][:-1]
            assert float(np.max(np.abs(phase_difference))) < 1.0e-6, (
                f"HH {name} voltage-solve phases do not align at {celsius:g} C"
            )


def test_hh_lie_split_gate_difference_converges_at_first_order():
    errors = {name: [] for name in STATE_NAMES}
    for dt in CONVERGENCE_DTS:
        expected = _run_neuron(22.0, dt=dt, tstop=CONVERGENCE_TSTOP)
        actual, _ = _run_dendra(22.0, dt=dt, tstop=CONVERGENCE_TSTOP)
        for name in STATE_NAMES:
            difference = actual[name] - expected[name]
            errors[name].append(float(np.sqrt(np.mean(difference * difference))))

    # Both simulators use first-order Lie splitting but apply the two coupled
    # substeps in opposite orders.  A timestep halving should therefore halve
    # their same-time state discrepancy.  Check both refinement levels so an
    # accidentally small result at one resolution cannot masquerade as order.
    for name, (coarse, medium, fine) in errors.items():
        assert coarse > medium > fine > 0.0
        for ratio in (coarse / medium, medium / fine):
            assert 1.8 < ratio < 2.2, (
                f"HH {name} splitting did not converge at first order: "
                f"errors={errors[name]!r}, ratio={ratio:.6g}"
            )


def test_hh_nonequilibrium_trajectory_converges_to_fine_neuron_reference():
    """Exercise the coupled solve without steady-state gate phase alignment."""
    reference = _run_neuron(
        22.0,
        dt=NON_EQUILIBRIUM_REFERENCE_DT,
        tstop=CONVERGENCE_TSTOP,
        initial_gates=NON_EQUILIBRIUM_GATES,
    )
    assert tuple(reference[name][0] for name in STATE_NAMES) == pytest.approx(
        NON_EQUILIBRIUM_GATES
    )
    assert np.ptp(reference["v"]) > 10.0

    # Compare currents at the reported state phase, not NEURON's stale
    # BREAKPOINT fields and not the old-gate voltage-solve phase used by the
    # same-dt oracle above.  Dendra's recorder evaluates these same canonical
    # expressions from its reported voltage and gates.
    reference.update(
        _hh_currents(reference["v"], reference["m"], reference["h"], reference["n"])
    )

    rmse_by_trace = {name: [] for name in TRACE_NAMES}
    finest_max_abs: dict[str, float] = {}
    for dt in NON_EQUILIBRIUM_DTS:
        actual, _ = _run_dendra(
            22.0,
            dt=dt,
            tstop=CONVERGENCE_TSTOP,
            initial_gates=NON_EQUILIBRIUM_GATES,
        )
        stride = round(dt / NON_EQUILIBRIUM_REFERENCE_DT)
        assert dt == pytest.approx(stride * NON_EQUILIBRIUM_REFERENCE_DT, abs=1.0e-14)
        for name in TRACE_NAMES:
            expected = reference[name][::stride]
            assert actual[name].shape == expected.shape
            difference = actual[name] - expected
            assert np.isfinite(difference).all()
            rmse_by_trace[name].append(float(np.sqrt(np.mean(difference * difference))))
            if dt == NON_EQUILIBRIUM_DTS[-1]:
                finest_max_abs[name] = float(np.max(np.abs(difference)))

    # With gates displaced well away from their -65 mV steady state, Dendra's
    # state-first and NEURON's voltage-first Lie splittings no longer enjoy the
    # near-exact one-sample phase alignment of the initialized HH oracle.  Both
    # nevertheless converge to the same coupled trajectory at first order.
    for name, (coarse, medium, fine) in rmse_by_trace.items():
        assert coarse > medium > fine > 0.0
        for ratio in (coarse / medium, medium / fine):
            assert 1.7 < ratio < 2.3, (
                f"non-equilibrium HH {name} did not converge at first order: "
                f"errors={rmse_by_trace[name]!r}, ratio={ratio:.6g}"
            )

        max_bound, rmse_bound = NON_EQUILIBRIUM_FINE_ERROR_BOUNDS[name]
        assert finest_max_abs[name] < max_bound
        assert fine < rmse_bound
