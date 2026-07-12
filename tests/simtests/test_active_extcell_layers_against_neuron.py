"""Fine-reference oracles for active finite-layer extracellular cables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.extcell import ExtCellAxon, ExtCellTree
from dendra.models.io import neuron_to_dendra_graph
from dendra.models.mod import hh

neuron = pytest.importorskip("neuron")
h = neuron.h

pytestmark = pytest.mark.neuron

# NEURON 9's Python wrapper aliases indexed vext references to vext[0].  HOC
# access remains layer-correct.
h("func dendra_active_ext_vext0() { return vext[0]($1) }")
h("func dendra_active_ext_vext1() { return vext[1]($1) }")

DTYPE = torch.float64
DTS = (0.0125, 0.00625, 0.003125)
REFERENCE_DT = DTS[-1] / 8.0
TSTOP = 3.0
V_INIT = -65.0
CELSIUS = 22.0
RHOA = 35.4
CM = 1.0
FIELD_AMPLITUDE = 18.0
XRAXIAL = (8.0e4, 1.3e5)
XC = (0.18, 0.07)
XG = (3.5e-2, 6.5e-2)
TRACE_NAMES = ("v", "vext0", "vext1", "imem", "m", "h", "n", "ina", "ik", "il")
FINE_ERROR_BOUNDS = {
    "v": (5.0, 0.85),
    "vext0": (2.2, 0.37),
    "vext1": (1.8, 0.15),
    "imem": (2.2e-7, 3.5e-8),
    "m": (0.06, 0.013),
    "h": (0.02, 0.005),
    "n": (0.02, 0.005),
    "ina": (0.11, 0.014),
    "ik": (0.05, 0.012),
    "il": (0.0015, 0.00027),
}


@dataclass
class _Case:
    name: str
    model: ExtCellAxon | ExtCellTree
    node_to_segment: dict[int, object]
    record_nodes: tuple[int, ...]
    bath_nodes: tuple[int, ...]


def _section(name, *, length, diameter, nseg):
    section = h.Section(name=name)
    section.L = float(length)
    section.diam = float(diameter)
    section.Ra = RHOA
    section.cm = CM
    section.nseg = int(nseg)
    return section


def _path(section, start, end):
    h.pt3dclear(sec=section)
    h.pt3dadd(*start, float(section.diam), sec=section)
    h.pt3dadd(*end, float(section.diam), sec=section)


def _configure_active_membrane(sections):
    assert int(h.nlayer_extracellular()) == 2
    for section in sections:
        section.insert("hh")
        section.insert("extracellular")
        for segment in section:
            for layer in range(2):
                segment.xraxial[layer] = XRAXIAL[layer]
                segment.xc[layer] = XC[layer]
                segment.xg[layer] = XG[layer]


def _assert_exact_layer_parameters(model):
    for name, expected_layers in (
        ("xraxial", XRAXIAL),
        ("xc", XC),
        ("xg", XG),
    ):
        values = getattr(model, name)
        assert values.dtype == DTYPE
        for layer, expected in enumerate(expected_layers):
            torch.testing.assert_close(
                values[..., layer],
                torch.full_like(values[..., layer], expected),
                rtol=0.0,
                atol=0.0,
            )


def _axon_case():
    n_comp = 9
    dx = 20.0
    diameter = 4.0
    section = _section(
        "active_finite_ext_axon",
        length=n_comp * dx,
        diameter=diameter,
        nseg=n_comp,
    )
    _configure_active_membrane((section,))
    segments = list(section)

    with dn.ctx(IMEM=1):
        model = ExtCellAxon(
            diameters=[diameter],
            n_comp=n_comp,
            celsius=CELSIUS,
            v_init=V_INIT,
            dtype=DTYPE,
            rhoa=RHOA,
            cm=CM,
        )
    model.dx.fill_(dx)
    model.x.copy_(model._x())
    for layer in range(2):
        model.xraxial[..., layer].fill_(XRAXIAL[layer])
        model.xc[..., layer].fill_(XC[layer])
        model.xg[..., layer].fill_(XG[layer])
    _assert_exact_layer_parameters(model)
    model.insert(hh)
    model.eval()
    model.initialize()
    return _Case(
        "axon",
        model,
        dict(enumerate(segments)),
        (0, n_comp // 2, n_comp - 1),
        tuple(range(n_comp)),
    )


def _tree_case():
    trunk = _section("active_finite_ext_trunk", length=60.0, diameter=7.0, nseg=3)
    branch_a = _section("active_finite_ext_branch_a", length=80.0, diameter=4.0, nseg=3)
    branch_b = _section(
        "active_finite_ext_branch_b", length=100.0, diameter=3.0, nseg=3
    )
    branch_a.connect(trunk(1.0), 0.0)
    branch_b.connect(trunk(1.0), 0.0)
    _path(trunk, (0.0, 0.0, 0.0), (60.0, 0.0, 0.0))
    _path(branch_a, (60.0, 0.0, 0.0), (60.0, 80.0, 10.0))
    _path(branch_b, (60.0, 0.0, 0.0), (60.0, -100.0, -15.0))
    sections = (trunk, branch_a, branch_b)
    _configure_active_membrane(sections)

    graph, node_to_segment = neuron_to_dendra_graph(trunk, extcell=2)
    with dn.ctx(IMEM=1):
        model = ExtCellTree.from_graph(
            graph,
            N=1,
            celsius=CELSIUS,
            v_init=V_INIT,
            dtype=DTYPE,
        )
    # This model-level equality catches accidental Python-double -> default
    # float32 -> float64 widening anywhere in graph gathering/loading.
    _assert_exact_layer_parameters(model)
    model.insert(hh)
    model.eval()
    model.initialize()

    branchpoints = tuple(
        node
        for node, attrs in graph.nodes(data=True)
        if str(attrs["name"]).startswith("branchpoint.")
    )
    assert len(branchpoints) == 1
    bath_nodes = tuple(node for node in graph if node not in branchpoints)

    def node_for(section, segment_index):
        target = list(section)[segment_index]
        matches = [
            node
            for node, segment in node_to_segment.items()
            if node in bath_nodes
            and segment.sec == target.sec
            and float(segment.x) == pytest.approx(float(target.x))
        ]
        assert len(matches) == 1
        return matches[0]

    return _Case(
        "tree",
        model,
        node_to_segment,
        (node_for(trunk, 0), node_for(branch_a, -1), node_for(branch_b, -1)),
        bath_nodes,
    )


def _spatial_profile(model):
    coordinate = model.x + 0.4 * model.y - 0.25 * model.z
    coordinate = coordinate - coordinate.mean(dim=-1, keepdim=True)
    scale = coordinate.abs().amax(dim=-1, keepdim=True)
    assert torch.all(scale > 0.0)
    normalized = coordinate / scale
    profile = normalized + 0.35 * torch.cos(torch.pi * (normalized - 0.15))
    profile = profile - profile.mean(dim=-1, keepdim=True)
    return FIELD_AMPLITUDE * profile / profile.abs().amax(dim=-1, keepdim=True)


def _temporal_profile(dt):
    n_steps = round(TSTOP / dt)
    assert TSTOP == pytest.approx(n_steps * dt, abs=1.0e-14)
    time = torch.arange(n_steps, dtype=DTYPE) * dt
    # A finite, smoothly modulated pulse.  Both discontinuities lie on every
    # nested grid, so refinement measures the coupled solver rather than a
    # changing stimulus boundary.
    active = ((time >= 0.25) & (time < 1.25)).to(DTYPE)
    carrier = 0.8 + 0.2 * torch.sin(2.0 * torch.pi * 0.5 * time)
    return (active * carrier).unsqueeze(0)


def _canonical_currents(v, m, h_gate, n):
    return {
        "ina": 0.12 * m**3 * h_gate * (v - 50.0),
        "ik": 0.036 * n**4 * (v + 77.0),
        "il": 0.0003 * (v + 54.3),
    }


def _run_neuron(case, spatial, dt):
    assert h.usetable_hh == 0
    temporal = _temporal_profile(dt)
    recorded = [case.node_to_segment[node] for node in case.record_nodes]
    assignments = [
        (case.node_to_segment[node], float(spatial[0, node]))
        for node in case.bath_nodes
    ]
    for segment, _ in assignments:
        segment.e_extracellular = 0.0

    cvode = h.CVode()
    cvode.active(0)
    h.secondorder = 0
    h.celsius = CELSIUS
    h.dt = dt
    h.finitialize(V_INIT)

    def sample():
        return np.asarray(
            [
                [float(segment.v) for segment in recorded],
                [
                    float(h.dendra_active_ext_vext0(segment.x, sec=segment.sec))
                    for segment in recorded
                ],
                [
                    float(h.dendra_active_ext_vext1(segment.x, sec=segment.sec))
                    for segment in recorded
                ],
                [
                    float(segment.i_membrane) * float(segment.area()) * 1.0e-8
                    for segment in recorded
                ],
                [float(segment.hh.m) for segment in recorded],
                [float(segment.hh.h) for segment in recorded],
                [float(segment.hh.n) for segment in recorded],
            ]
        )

    rows = [sample()]
    for scale in temporal[0].tolist():
        for segment, bath in assignments:
            segment.e_extracellular = bath * scale
        h.fadvance()
        rows.append(sample())
    values = np.asarray(rows)
    result = {
        name: values[:, index, :]
        for index, name in enumerate(("v", "vext0", "vext1", "imem", "m", "h", "n"))
    }
    # NEURON's ionic-current fields retain its earlier BREAKPOINT evaluation
    # after fadvance(). Reconstruct from simultaneously sampled post-step
    # voltage and gates so both simulators use one explicit physical phase.
    result.update(
        _canonical_currents(result["v"], result["m"], result["h"], result["n"])
    )
    return result


def _run_dendra(case, spatial, dt):
    # Each resolution starts from a clean state, including both extracellular
    # layers and HH gates.
    case.model.initialize()
    nodes = torch.as_tensor(case.record_nodes, dtype=torch.long)
    mechanism = case.model.mech.hh
    recorder = dn.callbacks.RecorderLambda(
        {
            "v": lambda model: model.v.index_select(-1, nodes),
            "vext0": lambda model: model.vc[..., 1].index_select(-1, nodes),
            "vext1": lambda model: model.vc[..., 2].index_select(-1, nodes),
            "imem": lambda model: model.i_membrane.index_select(-1, nodes),
            "m": lambda _model: mechanism.m.index_select(-1, nodes),
            "h": lambda _model: mechanism.h.index_select(-1, nodes),
            "n": lambda _model: mechanism.n.index_select(-1, nodes),
            "ina": lambda model: mechanism.ina(model.v).index_select(-1, nodes),
            "ik": lambda model: mechanism.ik(model.v).index_select(-1, nodes),
            "il": lambda model: mechanism.il(model.v).index_select(-1, nodes),
            "all_vext1": lambda model: model.vc[..., 2],
            "all_imem": lambda model: model.i_membrane,
        }
    )
    temporal = _temporal_profile(dt)
    case.model.run(
        tstop=TSTOP,
        dt=dt,
        extra=(spatial, temporal),
        callbacks=[recorder],
    )
    n_samples = round(TSTOP / dt) + 1
    result = {
        name: recorder.numpy(name).reshape(n_samples, len(case.record_nodes))
        for name in TRACE_NAMES
    }
    all_vext1 = recorder.numpy("all_vext1").reshape(n_samples, case.model.nc)
    all_imem = recorder.numpy("all_imem").reshape(n_samples, case.model.nc)
    bath = temporal.numpy()[0, :, None] * spatial.numpy()[0, None, :]
    area = case.model.area.detach().cpu().numpy().reshape(1, case.model.nc)
    xg = case.model.xg[..., 1].detach().cpu().numpy().reshape(1, case.model.nc)
    xc = case.model.xc[..., 1].detach().cpu().numpy().reshape(1, case.model.nc)
    outer_current = area * (
        xg * (all_vext1[1:] - bath)
        + (1.0e-3 * xc / dt) * (all_vext1[1:] - all_vext1[:-1])
    )
    # Axial shell currents cancel pairwise over a sealed morphology, so total
    # outward membrane current must equal current through the outer radial
    # conductance/capacitance to the bath.
    result["kcl_residual"] = all_imem[1:].sum(axis=1) - outer_current.sum(axis=1)
    return result


def _spike_features(voltage, dt):
    crossings = np.asarray(
        [
            np.flatnonzero(voltage[:, site] > 0.0)[0] * dt
            for site in range(voltage.shape[1])
        ]
    )
    peaks = np.argmax(voltage, axis=0) * dt
    return crossings, peaks


@pytest.mark.parametrize("case_name", ["axon", "tree"])
def test_active_finite_layer_model_converges_to_neuron(case_name):
    case = _axon_case() if case_name == "axon" else _tree_case()
    spatial = _spatial_profile(case.model)
    reference = _run_neuron(case, spatial, REFERENCE_DT)
    actual_runs = [_run_dendra(case, spatial, dt) for dt in DTS]

    assert float(np.max(reference["v"])) > 20.0
    assert float(np.max(np.abs(reference["vext0"] - reference["vext1"]))) > 0.05
    assert float(np.max(np.abs(reference["imem"]))) > 1.0e-10

    errors = {name: [] for name in TRACE_NAMES}
    max_errors = {name: [] for name in TRACE_NAMES}
    for dt, actual in zip(DTS, actual_runs):
        assert float(np.max(np.abs(actual["kcl_residual"]))) < 1.0e-15
        stride = round(dt / REFERENCE_DT)
        assert dt == pytest.approx(stride * REFERENCE_DT, abs=1.0e-15)
        for name in TRACE_NAMES:
            expected = reference[name][::stride]
            assert actual[name].shape == expected.shape
            difference = actual[name] - expected
            assert np.isfinite(difference).all()
            errors[name].append(float(np.sqrt(np.mean(difference * difference))))
            max_errors[name].append(float(np.max(np.abs(difference))))

    for name, (coarse, medium, fine) in errors.items():
        assert coarse > medium > fine > 0.0, (
            f"{case_name} {name} errors were not monotone: {errors[name]!r}; "
            f"max={max_errors[name]!r}"
        )
        for ratio in (coarse / medium, medium / fine):
            assert 1.7 < ratio < 2.4, (
                f"{case_name} {name} did not converge at first order: "
                f"errors={errors[name]!r}, ratio={ratio:.6g}"
            )

        max_bound, rmse_bound = FINE_ERROR_BOUNDS[name]
        assert max_errors[name][-1] < max_bound
        assert fine < rmse_bound

    reference_crossing, reference_peak = _spike_features(reference["v"], REFERENCE_DT)
    crossing_errors = []
    peak_errors = []
    for dt, actual in zip(DTS, actual_runs):
        crossing, peak = _spike_features(actual["v"], dt)
        crossing_errors.append(float(np.max(np.abs(crossing - reference_crossing))))
        peak_errors.append(float(np.max(np.abs(peak - reference_peak))))
    assert crossing_errors[0] > crossing_errors[1] > crossing_errors[2]
    assert peak_errors[0] > peak_errors[1] > peak_errors[2]
    assert crossing_errors[-1] <= 4.0 * DTS[-1]
    assert peak_errors[-1] <= 4.0 * DTS[-1]

    torch.testing.assert_close(
        case.model.v,
        case.model.vc[..., 0] - case.model.vc[..., 1],
        rtol=0.0,
        atol=2.0e-14,
    )
