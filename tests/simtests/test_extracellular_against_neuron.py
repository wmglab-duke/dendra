"""NEURON oracles for prescribed extracellular stimulation."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.io import neuron_to_dendra_graph
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms.compilers import ast as ast_compiler
from dendra.models.mechanisms.compilers import source as source_compiler
from dendra.models.mechanisms.compilers.ast import factorize_linear_in_v
from dendra.models.mechanisms.compilers.source import SourceUnavailableError
from dendra.models.mod import pas
from dendra.units import nA

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

DT = 0.025
N_STEPS = 120
RESTING_VOLTAGE = -70.0
LEAK_CONDUCTANCE = 0.001
RHOA = 35.4
CM = 1.0
DTYPE = torch.float64


class _MultilinePassive(Mechanism):
    """A deliberately noncanonical spelling of NEURON's ``pas`` current."""

    Mechanism.RANGE(g=LEAK_CONDUCTANCE, e=RESTING_VOLTAGE)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        conductance = self.g
        driving_force = v - self.e
        membrane_current = conductance * driving_force
        return membrane_current


_FirstPassiveAlias = _MultilinePassive.rename("symbolic_pas_first_alias")
_HeadlessNestedPassive = _FirstPassiveAlias.rename("symbolic_pas_nested_headless")


def _insert_neuron_membrane(sections):
    """Install the passive membrane and ideal prescribed-vext boundary."""
    for section in sections:
        section.insert("pas")
        section.insert("extracellular")
        for segment in section:
            segment.pas.g = LEAK_CONDUCTANCE
            segment.pas.e = RESTING_VOLTAGE
            for layer in range(int(h.nlayer_extracellular())):
                segment.xraxial[layer] = 1.0e9
                segment.xg[layer] = 1.0e9
                segment.xc[layer] = 0.0


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


def _initialize_dendra(model, mechanism=pas):
    model.insert(mechanism, g=LEAK_CONDUCTANCE, e=RESTING_VOLTAGE)
    model.eval()
    model.initialize()
    return model


def _branched_tree_case(mechanism=pas):
    """Build Dendra from the exact resistor tree used by NEURON."""
    soma = _new_stylized_section("ext_soma", length=100.0, diameter=3.0, nseg=5)
    dend_a = _new_stylized_section("ext_dend_a", length=60.0, diameter=2.0, nseg=3)
    dend_b = _new_stylized_section("ext_dend_b", length=60.0, diameter=1.5, nseg=3)

    _set_3d_path(soma, (0.0, 0.0, 0.0), (100.0, 0.0, 0.0))
    _set_3d_path(dend_a, (30.0, 0.0, 0.0), (90.0, 0.0, 0.0))
    _set_3d_path(dend_b, (70.0, 0.0, 0.0), (70.0, 60.0, 0.0))
    dend_a.connect(soma(0.3), 0.0)
    dend_b.connect(soma(0.7), 0.0)

    sections = (soma, dend_a, dend_b)
    graph, id_to_segment = neuron_to_dendra_graph(soma)
    assert set(id_to_segment) == set(graph)
    assert not any(
        str(attrs["name"]).startswith("branchpoint.")
        for _, attrs in graph.nodes(data=True)
    )

    model = dn.Tree.from_graph(
        graph,
        N=1,
        celsius=37.0,
        v_init=RESTING_VOLTAGE,
        dtype=DTYPE,
    )
    _initialize_dendra(model, mechanism)
    _insert_neuron_membrane(sections)
    indices = list(range(model.nc))
    return model, id_to_segment, indices


def _axon_case(kind, mechanism=pas):
    if kind == "unmyelinated":
        model = dn.Unmyelinated(
            diameters=[2.0],
            L=100.0,
            dx=20.0,
            celsius=37.0,
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
            celsius=37.0,
            v_init=RESTING_VOLTAGE,
            rhoa=RHOA,
            cm=CM,
            dtype=DTYPE,
        )
    else:  # pragma: no cover - guarded by the parametrization below
        raise ValueError(kind)

    _initialize_dendra(model, mechanism)

    if kind == "unmyelinated":
        expected_geometry = {"dx": 20.0, "diam": 2.0, "rhoa": RHOA, "cm": CM}
        expected_x = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0], dtype=DTYPE)
    else:
        expected_geometry = {"dx": 2.0, "diam": 7.0, "rhoa": 17700.0, "cm": CM}
        expected_x = torch.tensor([-2000.0, -1000.0, 0.0, 1000.0, 2000.0], dtype=DTYPE)
    assert model.nc == 5
    torch.testing.assert_close(model.x[0], expected_x)
    for name, expected in expected_geometry.items():
        value = getattr(model, name)
        torch.testing.assert_close(value, torch.full_like(value, expected))

    # Myelinated is deliberately a reduced node-only cable: its short membrane
    # compartments retain physical node area while an effective Ra represents
    # the long, narrow internodal axial path. Reading these tensors after
    # initialize() is essential because in-graph parametrizations are now applied.
    section = _new_stylized_section(
        f"ext_{kind}",
        length=float(model.dx[0].sum()),
        diameter=float(model.diam[0, 0]),
        nseg=model.nc,
    )
    section.Ra = float(model.rhoa[0, 0])
    section.cm = float(model.cm[0, 0])
    _insert_neuron_membrane((section,))
    segments = list(section)
    assert len(segments) == model.nc
    return model, dict(enumerate(segments)), list(range(model.nc))


def _spatial_profile(model):
    """A nonuniform, gauge-centered extracellular potential in mV."""
    coordinate = model.x + 0.6 * model.y - 0.25 * model.z
    coordinate = coordinate - coordinate.mean(dim=-1, keepdim=True)
    magnitude = coordinate.abs().amax(dim=-1, keepdim=True)
    assert torch.all(magnitude > 0)
    return 8.0 * coordinate / magnitude


def _temporal_profile():
    """A smooth polarity-reversing prescribed-field scale."""
    time = torch.arange(N_STEPS, dtype=DTYPE) * DT
    return torch.sin(2.0 * torch.pi * 0.25 * time).unsqueeze(0)


def _run_neuron(node_to_segment, record_indices, spatial, temporal):
    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = 37.0
    h.dt = DT

    assignments = [
        (node_to_segment[node], float(spatial[0, node]))
        for node in range(spatial.shape[-1])
    ]
    recorded_segments = [node_to_segment[node] for node in record_indices]

    for segment, _ in assignments:
        segment.e_extracellular = 0.0
    h.finitialize(RESTING_VOLTAGE)
    trace = [[float(segment.v) for segment in recorded_segments]]

    for scale in temporal[0].tolist():
        for segment, field_value in assignments:
            segment.e_extracellular = field_value * scale
        h.fadvance()
        trace.append([float(segment.v) for segment in recorded_segments])
    return np.asarray(trace)


def _run_dendra(model, record_indices, spatial, temporal):
    recorder = dn.callbacks.Recorder(states=["v"], node_indices=record_indices)
    model.run(
        tstop=N_STEPS * DT,
        dt=DT,
        extra=(spatial, temporal),
        callbacks=[recorder],
    )
    return recorder.numpy("v")[:, 0, :]


def _assert_simulators_close(actual, expected, *, case):
    assert actual.shape == expected.shape == (N_STEPS + 1, expected.shape[1])
    excursion = float(np.max(np.abs(expected - expected[0])))
    assert excursion > 0.05, f"{case} oracle response was trivially small"

    difference = actual - expected
    max_abs = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    assert np.allclose(actual, expected, rtol=1.0e-7, atol=2.0e-4), (
        f"Extracellular {case} mismatch: max_abs={max_abs:.6g} mV, rmse={rmse:.6g} mV"
    )


def _force_headless_stable_source(monkeypatch):
    """Require analysis to recover the original class, never a dynamic alias."""
    real_getsource = inspect.getsource
    real_safe_source = ast_compiler.safe_source
    source_owners = []

    def getsource_without_dynamic_alias(obj):
        if obj in (_FirstPassiveAlias, _HeadlessNestedPassive):
            raise OSError("dynamic aliases have no discoverable class source")
        return real_getsource(obj)

    def safe_source_from_stable_owner(obj):
        source_owners.append(obj)
        if obj in (_FirstPassiveAlias, _HeadlessNestedPassive):
            raise AssertionError("symbolic analysis inspected a dynamic alias")
        if obj is _MultilinePassive.i:
            raise SourceUnavailableError("force original class-source recovery")
        return real_safe_source(obj)

    monkeypatch.setattr(source_compiler, "IPYTHON_AVAILABLE", False)
    monkeypatch.setattr(
        source_compiler.inspect,
        "getsource",
        getsource_without_dynamic_alias,
    )
    monkeypatch.setattr(ast_compiler, "safe_source", safe_source_from_stable_owner)
    factorize_linear_in_v.cache_clear()
    return source_owners


def _assert_symbolic_passive_provenance(model, source_owners):
    mechanism = model.mech.mechanisms[_HeadlessNestedPassive.__name__]
    assert _HeadlessNestedPassive._dendra_symbolic_source_class is _MultilinePassive
    assert mechanism._current_conductance_mode == {"i": "symbolic"}
    assert mechanism._current_conductance_fallback_reason == {"i": None}
    assert source_owners == [_MultilinePassive.i, _MultilinePassive]


def _run_single_compartment_neuron():
    section = _new_stylized_section(
        "symbolic_pas_single", length=100.0, diameter=100.0, nseg=1
    )
    _insert_neuron_membrane((section,))
    stimulus = h.IClamp(section(0.5))
    stimulus.delay = 0.5
    stimulus.dur = 0.75
    stimulus.amp = 0.1

    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = 37.0
    h.dt = DT
    h.finitialize(RESTING_VOLTAGE)
    trace = [float(section(0.5).v)]
    for _ in range(N_STEPS):
        h.fadvance()
        trace.append(float(section(0.5).v))
    return np.asarray(trace)[:, None]


def _run_single_compartment_dendra(source_owners):
    model = dn.SingleCompartment(
        N=1,
        C=1,
        celsius=37.0,
        v_init=RESTING_VOLTAGE,
        rhoa=RHOA,
        cm=CM,
        dtype=DTYPE,
    )
    model.diam.fill_(100.0)
    model.dx.fill_(100.0)
    model.insert(
        _HeadlessNestedPassive,
        g=LEAK_CONDUCTANCE,
        e=RESTING_VOLTAGE,
    )
    model[:, 0].inject(dn.mono_rect(amp=0.1 * nA, delay=0.5, pw=0.75))
    recorder = dn.callbacks.Recorder(states=["v"], node_indices=[0])
    model.eval()
    model.initialize()
    _assert_symbolic_passive_provenance(model, source_owners)
    model.run(tstop=N_STEPS * DT, dt=DT, callbacks=[recorder])
    return model, recorder.numpy("v")[:, 0, :]


@pytest.mark.parametrize("case", ["tree", "unmyelinated", "myelinated"])
def test_prescribed_extracellular_stimulation_matches_neuron(case):
    if case == "tree":
        model, node_to_segment, record_indices = _branched_tree_case()
    else:
        model, node_to_segment, record_indices = _axon_case(case)

    spatial = _spatial_profile(model)
    temporal = _temporal_profile()
    expected = _run_neuron(node_to_segment, record_indices, spatial, temporal)
    actual = _run_dendra(model, record_indices, spatial, temporal)

    _assert_simulators_close(actual, expected, case=case)


@pytest.mark.parametrize("case", ["tree", "unmyelinated", "myelinated"])
def test_headless_renamed_multiline_passive_trajectory_matches_neuron(
    case, monkeypatch
):
    source_owners = _force_headless_stable_source(monkeypatch)
    if case == "tree":
        model, node_to_segment, record_indices = _branched_tree_case(
            _HeadlessNestedPassive
        )
    else:
        model, node_to_segment, record_indices = _axon_case(
            case, _HeadlessNestedPassive
        )

    _assert_symbolic_passive_provenance(model, source_owners)
    spatial = _spatial_profile(model)
    temporal = _temporal_profile()
    expected = _run_neuron(node_to_segment, record_indices, spatial, temporal)
    actual = _run_dendra(model, record_indices, spatial, temporal)

    _assert_simulators_close(actual, expected, case=f"symbolic {case}")


def test_headless_renamed_multiline_passive_single_compartment_matches_neuron(
    monkeypatch,
):
    source_owners = _force_headless_stable_source(monkeypatch)
    expected = _run_single_compartment_neuron()
    _, actual = _run_single_compartment_dendra(source_owners)

    _assert_simulators_close(actual, expected, case="symbolic single-compartment")
