"""NEURON oracles for active-HH extracellular stimulation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.io import neuron_to_dendra_graph
from dendra.models.mod import hh

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DT = 0.025
N_STEPS = 320
RESTING_VOLTAGE = -65.0
CELSIUS = 6.3
RHOA = 35.4
CM = 1.0
FIELD_AMPLITUDE = 16.0
DTYPE = torch.float64


def _new_stylized_section(name, *, length, diameter, nseg):
    section = h.Section(name=name)
    section.L = float(length)
    section.diam = float(diameter)
    section.Ra = RHOA
    section.cm = CM
    section.nseg = int(nseg)
    return section


def _set_3d_path(section, start, end):
    h.pt3dclear(sec=section)
    h.pt3dadd(*start, float(section.diam), sec=section)
    h.pt3dadd(*end, float(section.diam), sec=section)


def _insert_neuron_hh(sections):
    """Install matching HH membranes and an ideal prescribed-vext boundary."""
    for section in sections:
        section.insert("hh")
        section.insert("extracellular")
        for segment in section:
            # Assert the oracle has the same canonical HH parametrization as
            # Dendra's ``hh`` implementation instead of relying silently on it.
            assert segment.hh.gnabar == pytest.approx(0.12)
            assert segment.hh.gkbar == pytest.approx(0.036)
            assert segment.hh.gl == pytest.approx(0.0003)
            assert segment.hh.el == pytest.approx(-54.3)
            assert segment.ena == pytest.approx(50.0)
            assert segment.ek == pytest.approx(-77.0)
            for layer in range(int(h.nlayer_extracellular())):
                segment.xraxial[layer] = 1.0e9
                segment.xg[layer] = 1.0e9
                segment.xc[layer] = 0.0


def _initialize_dendra(model):
    model.insert(hh)
    model.eval()
    model.initialize()
    return model


def _branched_tree_case():
    """Build Dendra from the exact active cable tree used by NEURON."""
    soma = _new_stylized_section("active_ext_soma", length=100.0, diameter=3.0, nseg=5)
    dend_a = _new_stylized_section(
        "active_ext_dend_a", length=60.0, diameter=2.0, nseg=3
    )
    dend_b = _new_stylized_section(
        "active_ext_dend_b", length=60.0, diameter=1.5, nseg=3
    )

    _set_3d_path(soma, (0.0, 0.0, 0.0), (100.0, 0.0, 0.0))
    _set_3d_path(dend_a, (30.0, 0.0, 0.0), (90.0, -30.0, 0.0))
    _set_3d_path(dend_b, (70.0, 0.0, 0.0), (70.0, 60.0, 20.0))
    dend_a.connect(soma(0.3), 0.0)
    dend_b.connect(soma(0.7), 0.0)

    sections = (soma, dend_a, dend_b)
    graph, node_to_segment = neuron_to_dendra_graph(soma)
    assert set(node_to_segment) == set(graph)
    assert not any(
        str(attrs["name"]).startswith("branchpoint.")
        for _, attrs in graph.nodes(data=True)
    )

    model = dn.Tree.from_graph(
        graph,
        N=1,
        celsius=CELSIUS,
        v_init=RESTING_VOLTAGE,
        dtype=DTYPE,
    )
    _initialize_dendra(model)
    _insert_neuron_hh(sections)
    return model, node_to_segment


def _axon_case(kind):
    if kind == "unmyelinated":
        model = dn.Unmyelinated(
            diameters=[2.0],
            L=100.0,
            dx=20.0,
            celsius=CELSIUS,
            v_init=RESTING_VOLTAGE,
            rhoa=RHOA,
            cm=CM,
            dtype=DTYPE,
        )
    elif kind == "myelinated":
        model = dn.Myelinated(
            diameters=[10.0],
            n_node=5,
            node_length=2.0,
            celsius=CELSIUS,
            v_init=RESTING_VOLTAGE,
            rhoa=RHOA,
            cm=CM,
            dtype=DTYPE,
        )
    else:  # pragma: no cover - guarded by parametrization
        raise ValueError(kind)

    _initialize_dendra(model)

    if kind == "unmyelinated":
        expected_geometry = {"dx": 20.0, "diam": 2.0, "rhoa": RHOA, "cm": CM}
        expected_x = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0], dtype=DTYPE)
    else:
        # Myelinated is the documented reduced node-only cable: physical node
        # area plus an effective axial resistivity for the internodal path.
        expected_geometry = {"dx": 2.0, "diam": 7.0, "rhoa": 17700.0, "cm": CM}
        expected_x = torch.tensor([-2000.0, -1000.0, 0.0, 1000.0, 2000.0], dtype=DTYPE)

    assert model.nc == 5
    torch.testing.assert_close(model.x[0], expected_x)
    for name, expected in expected_geometry.items():
        value = getattr(model, name)
        torch.testing.assert_close(value, torch.full_like(value, expected))

    section = _new_stylized_section(
        f"active_ext_{kind}",
        length=float(model.dx[0].sum()),
        diameter=float(model.diam[0, 0]),
        nseg=model.nc,
    )
    section.Ra = float(model.rhoa[0, 0])
    section.cm = float(model.cm[0, 0])
    _insert_neuron_hh((section,))
    segments = list(section)
    assert len(segments) == model.nc
    return model, dict(enumerate(segments))


def _spatial_profile(model):
    """Return a curved, gauge-centered prescribed potential in mV."""
    coordinate = model.x + 0.35 * model.y - 0.2 * model.z
    coordinate = coordinate - coordinate.mean(dim=-1, keepdim=True)
    scale = coordinate.abs().amax(dim=-1, keepdim=True)
    assert torch.all(scale > 0)
    normalized = coordinate / scale
    profile = 0.55 * normalized + 0.45 * torch.cos(torch.pi * (normalized - 0.2))
    profile = profile - profile.mean(dim=-1, keepdim=True)
    profile = FIELD_AMPLITUDE * profile / profile.abs().amax(dim=-1, keepdim=True)
    assert torch.any(profile > 0)
    assert torch.any(profile < 0)
    return profile


def _temporal_profile():
    """A finite pulse with unambiguous step timing."""
    time = torch.arange(N_STEPS, dtype=DTYPE) * DT
    return ((time >= 0.5) & (time < 1.5)).to(DTYPE).unsqueeze(0)


def _run_neuron(node_to_segment, spatial, temporal):
    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = CELSIUS
    h.dt = DT

    assignments = [
        (node_to_segment[node], float(spatial[0, node]))
        for node in range(spatial.shape[-1])
    ]
    for segment, _ in assignments:
        segment.e_extracellular = 0.0

    h.finitialize(RESTING_VOLTAGE)
    trace = [[float(node_to_segment[node].v) for node in range(spatial.shape[-1])]]
    for scale in temporal[0].tolist():
        for segment, field_value in assignments:
            segment.e_extracellular = field_value * scale
        h.fadvance()
        trace.append(
            [float(node_to_segment[node].v) for node in range(spatial.shape[-1])]
        )
    return np.asarray(trace)


def _run_dendra(model, spatial, temporal):
    model.initialize()
    recorder = dn.callbacks.Recorder(states=["v"], node_indices=list(range(model.nc)))
    model.run(
        tstop=N_STEPS * DT,
        dt=DT,
        extra=(spatial, temporal),
        callbacks=[recorder],
    )
    return recorder.numpy("v")[:, 0, :]


def _assert_simulators_close(actual, expected, *, case, polarity):
    assert actual.shape == expected.shape == (N_STEPS + 1, expected.shape[1])
    excursion = float(np.max(np.abs(expected - expected[0])))
    assert excursion > 0.25, f"{case} {polarity} oracle response was trivially small"
    assert np.isfinite(actual).all()
    assert np.isfinite(expected).all()

    difference = actual - expected
    max_abs = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    diagnostics = (
        f"Active-HH extracellular {case} {polarity} mismatch: "
        f"max_abs={max_abs:.6g} mV, rmse={rmse:.6g} mV"
    )

    oracle_spiked = float(expected.max()) > 20.0
    actual_spiked = float(actual.max()) > 20.0
    assert actual_spiked == oracle_spiked, diagnostics
    # Disabling NEURON's HH rate interpolation removes the former sub-mV model
    # discrepancy.  Keep both pointwise and trajectory-wide guards tight for
    # every polarity, including the steep spike upstrokes.
    assert np.allclose(actual, expected, rtol=2.0e-6, atol=5.0e-4), diagnostics
    assert max_abs < 2.0e-3, diagnostics
    assert rmse < 4.0e-4, diagnostics
    if not oracle_spiked:
        return

    # Constrain spike timing, polarity-localized peak, amplitude, and
    # after-hyperpolarization independently as semantic checks in addition to
    # the strict waveform comparison above.

    expected_crossings = np.argwhere(expected > 0.0)
    actual_crossings = np.argwhere(actual > 0.0)
    expected_crossing_time = float(expected_crossings[0, 0]) * DT
    actual_crossing_time = float(actual_crossings[0, 0]) * DT
    assert abs(actual_crossing_time - expected_crossing_time) <= DT + 1.0e-12, (
        diagnostics
    )

    expected_peak = np.unravel_index(np.argmax(expected), expected.shape)
    actual_peak = np.unravel_index(np.argmax(actual), actual.shape)
    assert expected_peak[1] == actual_peak[1], diagnostics
    assert abs((actual_peak[0] - expected_peak[0]) * DT) <= DT + 1.0e-12, diagnostics
    assert abs(float(actual.max() - expected.max())) < 1.0e-3, diagnostics
    assert abs(float(actual.min() - expected.min())) < 1.0e-3, diagnostics


@pytest.mark.parametrize("case", ["tree", "unmyelinated", "myelinated"])
def test_active_hh_extracellular_stimulation_matches_neuron(case):
    if case == "tree":
        model, node_to_segment = _branched_tree_case()
    else:
        model, node_to_segment = _axon_case(case)

    spatial = _spatial_profile(model)
    temporal = _temporal_profile()
    oracle_by_polarity = {}
    for sign, polarity in ((1.0, "positive"), (-1.0, "negative")):
        signed_spatial = sign * spatial
        expected = _run_neuron(node_to_segment, signed_spatial, temporal)
        actual = _run_dendra(model, signed_spatial, temporal)
        _assert_simulators_close(actual, expected, case=case, polarity=polarity)
        oracle_by_polarity[polarity] = expected

    assert any(float(trace.max()) > 20.0 for trace in oracle_by_polarity.values()), (
        f"{case} active-HH oracle never crossed spike threshold"
    )

    # Reversing the field must reverse the initial activating response at the
    # most strongly affected compartment. This prevents a quiescent or
    # polarity-insensitive implementation from passing the oracle comparison.
    onset = int(0.5 / DT) + 1
    positive_delta = oracle_by_polarity["positive"][onset] - RESTING_VOLTAGE
    negative_delta = oracle_by_polarity["negative"][onset] - RESTING_VOLTAGE
    strongest = int(np.argmax(np.abs(positive_delta)))
    assert abs(float(positive_delta[strongest])) > 0.05
    assert positive_delta[strongest] * negative_delta[strongest] < 0.0

    positive_peak_node = int(np.argmax(oracle_by_polarity["positive"].max(axis=0)))
    negative_peak_node = int(np.argmax(oracle_by_polarity["negative"].max(axis=0)))
    assert positive_peak_node != negative_peak_node
