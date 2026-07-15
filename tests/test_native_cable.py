"""Contracts for compiling native morphologies onto the 1-D Cable fast path."""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch

import dendra as dn
from dendra.models.integrators import bwd_euler_sc, dhs
from dendra.models.integrators.cable import unbranched_edge_conductance
from dendra.models.integrators.implicit import _bwd_euler_ub
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mechanisms._material_process import DiffusionProcess
from dendra.models.mechanisms._spatial import SpatialOperator1D, SpatialOperatorTree
from dendra.models.mod import expsyn, pas
from dendra.units import nA

pytestmark = pytest.mark.cpu

DTYPE = torch.float64


class _ExactGeometryDiffusion(DiffusionProcess):
    """Explicit diffusion makes the selected finite-volume geometry observable."""

    DiffusionProcess.METHOD("explicit", solver="dense")
    DiffusionProcess.DIFFUSE("tracer", field="c", D=1.0)


def _serial_morphology() -> dn.Morphology:
    """Return a tapered, heterogeneous, orientation-reversing physical path."""
    morphology = dn.Morphology(rhoa=90.0, cm=1.0)
    root = morphology.section(
        "root",
        points=[(0.0, 0.0, 0.0, 8.0), (0.0, 30.0, 2.0, 5.0)],
        nseg=2,
        rhoa=82.0,
        cm=0.85,
        labels=("membrane", "stimulus_site"),
    )
    reversed_section = morphology.section(
        "reversed",
        # Authored from the distal end toward the physical root attachment.
        points=[(55.0, 68.0, -4.0, 1.2), (0.0, 30.0, 2.0, 4.5)],
        nseg=3,
        rhoa=137.0,
        cm=1.25,
        labels=("membrane", "reversed_region"),
    )
    tip = morphology.section(
        "tip",
        points=[(55.0, 68.0, -4.0, 1.2), (82.0, 91.0, 7.0, 0.7)],
        nseg=2,
        rhoa=111.0,
        cm=1.1,
        labels=("membrane", "readout"),
    )
    reversed_section.connect(root.at(1.0), child_end=1)
    tip.connect(reversed_section.at(0.0), child_end=0)
    return morphology


def _root_interior_morphology() -> dn.Morphology:
    """Return a physical path whose canonical directed root has two children."""
    morphology = dn.Morphology()
    root = morphology.section(
        "root",
        L=20.0,
        diam=4.0,
        nseg=2,
        labels=("membrane", "root_region"),
    )
    extension = morphology.section(
        "extension",
        points=[(0.0, 0.0, 0.0, 3.0), (-30.0, 4.0, 0.0, 1.0)],
        nseg=3,
        labels=("membrane", "extension_region"),
    )
    # The native compiler roots at root segment 0. Attaching another Section
    # to authored root x=0 therefore yields parent_index (-1, 0, 0, ...), even
    # though the underlying undirected resistor graph is one physical path.
    extension.connect(root.at(0.0), child_end=0)
    return morphology


def _tensor(values, *, device, dtype=DTYPE):
    return torch.as_tensor(values, device=device, dtype=dtype)


def _assert_canonical_geometry(model, graph: dn.CompartmentGraph) -> None:
    """Assert every public/exact geometry tensor follows graph storage order."""
    device = model.device()
    expected_shape = (model.np, graph.n_compartments)

    for name, values in (
        ("dx", graph.geometry.length_um),
        ("diam", graph.geometry.diameter_um),
        ("x", graph.geometry.x_um),
        ("y", graph.geometry.y_um),
        ("z", graph.geometry.z_um),
        ("rhoa", graph.geometry.rhoa_ohm_cm),
        ("cm", graph.geometry.cm_uF_cm2),
        ("volume", graph.geometry.volume_um3),
        ("volume_um3", graph.geometry.volume_um3),
        ("volume_i", graph.geometry.volume_i_um3),
        ("volume_o", graph.geometry.volume_o_um3),
    ):
        actual = getattr(model, name)
        expected = _tensor(values, device=device).expand(expected_shape)
        torch.testing.assert_close(actual, expected)

    expected_area = _tensor(graph.geometry.area_um2, device=device) * 1.0e-8
    torch.testing.assert_close(model.area, expected_area.expand(expected_shape))
    torch.testing.assert_close(
        model._canonical_area_cm2,
        expected_area.expand(expected_shape),
    )
    expected_resistance = _tensor(
        graph.geometry.edge_resistance_ohm, device=device
    ).expand(expected_shape)
    torch.testing.assert_close(
        model._canonical_edge_resistance_ohm,
        expected_resistance,
    )
    torch.testing.assert_close(model.edge_resistance_ohm, expected_resistance)


def test_single_compartment_cable_uses_fast_path_and_exact_native_geometry():
    morphology = dn.Morphology(rhoa=123.0, cm=0.75)
    morphology.section(
        "soma",
        points=[(1.0, 2.0, 3.0, 9.0), (11.0, 17.0, 23.0, 3.0)],
        nseg=1,
        labels=("membrane", "stimulus_site"),
    )
    graph = morphology.compile()

    model = dn.Cable.from_morphology(morphology, N=3, dtype=DTYPE)

    assert type(model) is dn.Cable
    assert model.shape == (3, 1)
    assert model.compartment_graph == graph
    assert model.names == list(graph.metadata.name)
    assert issubclass(model._integrator_class, _bwd_euler_ub)
    assert model.membrane.shape == (3, 1)
    assert model.stimulus_site.shape == (3, 1)
    _assert_canonical_geometry(model, graph)

    # K=1 is a valid degenerate tridiagonal solve, not an Axon-constructor
    # special case. Exercise the public lifecycle and exact-area injection.
    model.membrane.insert(pas, g=3.0e-4, e=-70.0)
    model.stimulus_site.inject(dn.mono_rect(amp=0.04 * nA, delay=0.0, pw=0.05))
    model.initialize()
    model.run(tstop=0.05, dt=0.025)
    assert torch.isfinite(model.v).all()
    assert not torch.equal(model.v, torch.full_like(model.v, -65.0))


def test_two_compartment_float32_cable_converts_exact_area_before_casting():
    morphology = dn.Morphology()
    morphology.section(
        "tiny_taper",
        points=[(0.0, 0.0, 0.0, 0.1), (0.1, 0.0, 0.0, 0.11)],
        nseg=2,
        labels=("membrane",),
    )
    graph = morphology.compile()
    model = dn.Cable.from_morphology(morphology, dtype=torch.float32)

    # This geometry differs by one float32 ULP if area is cast before the
    # um^2 -> cm^2 conversion. Canonical arithmetic is binary64 first.
    expected = (
        torch.as_tensor(graph.geometry.area_um2, dtype=torch.float64) * 1.0e-8
    ).to(torch.float32)
    early_cast = torch.as_tensor(graph.geometry.area_um2, dtype=torch.float32) * 1.0e-8
    assert not torch.equal(expected, early_cast)
    assert torch.equal(model._canonical_area_cm2[0], expected)

    model.membrane.insert(pas)
    model.initialize()
    model.run(tstop=0.025, dt=0.025)
    assert torch.isfinite(model.v).all()


def test_multisection_cable_preserves_oriented_provenance_labels_and_geometry():
    morphology = _serial_morphology()
    graph = morphology.compile()

    model = dn.Cable.from_morphology(morphology, N=2, dtype=DTYPE)

    # This source graph already has endpoint-to-endpoint directed path order,
    # so conversion is an identity snapshot rather than a hidden permutation.
    assert model.compartment_graph == graph
    assert model.compartment_graph.topology.parent_index == tuple(
        [-1, *range(graph.n_compartments - 1)]
    )
    assert model.compartment_graph.metadata.section_name == (
        "root",
        "root",
        "reversed",
        "reversed",
        "reversed",
        "tip",
        "tip",
    )
    # child_end=1 reverses solver traversal but never authored provenance.
    assert model.compartment_graph.metadata.segment_index[2:5] == (2, 1, 0)
    assert model.compartment_graph.metadata.section_x[2:5] == pytest.approx(
        (5 / 6, 1 / 2, 1 / 6)
    )
    _assert_canonical_geometry(model, graph)

    assert sorted(model.membrane.index[-1].tolist()) == list(
        range(graph.n_compartments)
    )
    reversed_indices = model.reversed_region.index[-1].tolist()
    reversed_x = [graph.metadata.section_x[index] for index in reversed_indices]
    assert reversed_x == sorted(reversed_x)
    assert model.readout.index[-1].tolist() == [5, 6]


def test_from_compartment_graph_is_a_snapshot_with_the_same_path_contract():
    morphology = _serial_morphology()
    graph = morphology.compile()

    model = dn.Cable.from_compartment_graph(graph, N=2, dtype=DTYPE)
    morphology.section("unrelated_later_section", L=10.0, diam=1.0)

    assert model.compartment_graph == graph
    assert model.shape == (2, graph.n_compartments)
    assert not hasattr(model, "unrelated_later_section")
    _assert_canonical_geometry(model, graph)


def test_cable_graph_attachment_is_factory_only_and_public_views_are_defensive():
    source = _serial_morphology().compile()
    source_graph = source.to_networkx()

    with pytest.raises(TypeError, match="low-level constructor|from_morphology"):
        dn.Cable(1, source.n_compartments, graph=source_graph, dtype=DTYPE)

    model = dn.Cable.from_compartment_graph(source, N=2, dtype=DTYPE).batch(3)
    graph_view = model.graph
    graph_view.remove_edge(0, 1)
    graph_view.nodes[0]["area"] = -1.0

    # Voltage and material topology share one private validated snapshot;
    # callers cannot mutate it through the interoperability view.
    fresh_graph = model.graph
    assert fresh_graph.has_edge(0, 1)
    assert fresh_graph.nodes[0]["area"] == pytest.approx(source.geometry.area_um2[0])

    canonical = model._canonical_edge_resistance_ohm.clone()
    public_resistance = model.edge_resistance_ohm
    public_resistance[0, 0, 1] = -1.0
    torch.testing.assert_close(model._canonical_edge_resistance_ohm, canonical)
    torch.testing.assert_close(
        model.edge_resistance_ohm,
        canonical.unsqueeze(0).expand(model.shape),
    )


def test_native_factory_is_extensible_but_specialized_axons_opt_out():
    class CustomCable(dn.Cable):
        pass

    model = CustomCable.from_morphology(_serial_morphology(), dtype=DTYPE)
    assert type(model) is CustomCable
    assert model.compartment_graph is not None


def test_native_cable_rejects_integrators_that_discard_path_topology():
    with pytest.raises(TypeError, match="supports_unbranched_cable"):
        dn.Cable.from_morphology(
            _serial_morphology(),
            integrator=bwd_euler_sc(),
            dtype=DTYPE,
        )


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="DHS compatibility requires the CPU dendra_solvers package",
)
def test_native_cable_accepts_dhs_as_an_exact_graph_solver():
    morphology = _serial_morphology()
    fast = dn.Cable.from_morphology(morphology, dtype=DTYPE)
    graph = dn.Cable.from_morphology(morphology, integrator=dhs(), dtype=DTYPE)
    for model in (fast, graph):
        model.membrane.insert(pas, g=3.0e-4, e=-70.0)
        model.stimulus_site.inject(dn.mono_rect(amp=0.12 * nA, delay=0.05, pw=0.15))
        model.initialize()
        model.run(tstop=0.5, dt=0.025)
    torch.testing.assert_close(fast.v, graph.v, rtol=2e-11, atol=2e-11)

    invalid = dn.Cable.from_morphology(morphology, integrator=dhs(), dtype=DTYPE)
    invalid.rhoa_scale = torch.linspace(0.9, 1.1, invalid.nc, dtype=DTYPE)
    invalid.membrane.insert(pas)
    with pytest.raises(ValueError, match="spatially uniform rhoa_scale"):
        invalid.initialize()


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="DHS compatibility requires the CPU dendra_solvers package",
)
def test_native_cable_dhs_full_shaped_rhoa_scale_and_ve_match_fast_path():
    morphology = _serial_morphology()
    fast = dn.Cable.from_morphology(morphology, N=2, dtype=DTYPE)
    graph = dn.Cable.from_morphology(morphology, N=2, integrator=dhs(), dtype=DTYPE)
    row_scale = torch.tensor([[0.5], [2.0]], dtype=DTYPE)
    for model in (fast, graph):
        model.membrane.insert(pas, g=2.5e-4, e=-70.0)
        model.initialize()
        # Initialization populates declared parameters from their specifications.
        # Apply runtime scale overrides afterward, before either integrator has
        # built its timestep-dependent workspace.
        model.rhoa_scale = row_scale.expand(2, model.nc).clone()

    for model in (fast, graph):
        assert torch.equal(model.rhoa_scale, row_scale.expand(2, model.nc))

    n_steps = 20
    spatial = torch.linspace(-3.0, 2.0, fast.nc, dtype=DTYPE)
    ve = spatial.reshape(1, 1, -1).expand(n_steps, 2, -1).clone()
    fast.run(ve=ve, dt=0.025)
    graph.run(ve=ve, dt=0.025)

    assert float((fast.v - fast.v_init).abs().amax()) > 0.01
    torch.testing.assert_close(fast.v, graph.v, rtol=2e-11, atol=2e-11)


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="DHS compatibility requires the CPU dendra_solvers package",
)
def test_native_cable_dhs_full_shaped_rhoa_scale_gradient_matches_fast_path():
    morphology = _serial_morphology()
    fast = dn.Cable.from_morphology(morphology, N=2, dtype=DTYPE)
    graph = dn.Cable.from_morphology(morphology, N=2, integrator=dhs(), dtype=DTYPE)
    dt = 0.031
    scale_inputs = []

    for model in (fast, graph):
        model.membrane.insert(pas, g=2.5e-4, e=-70.0)
        model.initialize()
        scale = (
            torch.tensor([[0.65], [1.7]], dtype=DTYPE)
            .expand(model.shape)
            .clone()
            .requires_grad_()
        )
        model.rhoa_scale = scale
        model.integrator._initialize(model, dt, force=True)
        scale_inputs.append(scale)

    voltage = torch.linspace(-76.0, -48.0, fast.v.numel(), dtype=DTYPE).reshape(
        fast.shape
    )
    intra = torch.linspace(-0.00008, 0.00011, fast.v.numel(), dtype=DTYPE).reshape(
        fast.shape
    )
    ve = torch.linspace(-4.0, 6.0, fast.nc, dtype=DTYPE).expand(fast.shape).clone()
    outputs = [
        model.integrator._step(
            voltage,
            dt,
            model.celsius,
            ve=ve,
            intra=intra,
        )[0]
        for model in (fast, graph)
    ]
    weights = torch.linspace(0.4, 1.3, fast.v.numel(), dtype=DTYPE).reshape(fast.shape)
    gradients = [
        torch.autograd.grad((output.square() * weights).sum(), scale)[0]
        for output, scale in zip(outputs, scale_inputs)
    ]

    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-10, atol=2e-12)
    assert torch.count_nonzero(gradients[0][:, 0]) == fast.np
    assert torch.count_nonzero(gradients[0][:, 1:]) == 0


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="DHS compatibility requires the CPU dendra_solvers package",
)
def test_native_cable_dhs_upcast_uses_same_canonical_edge_source_as_fast_path():
    morphology = dn.Morphology(rhoa=91.234567890123)
    morphology.section(
        "precise_taper",
        points=[
            (0.0, 0.0, 0.0, 8.123456789),
            (17.123456789, 31.987654321, 2.345678901, 0.7123456789),
        ],
        nseg=4,
        labels=("membrane",),
    )
    fast = dn.Cable.from_morphology(morphology, N=2, dtype=torch.float32).to(
        dtype=torch.float64
    )
    graph = dn.Cable.from_morphology(
        morphology, N=2, integrator=dhs(), dtype=torch.float32
    ).to(dtype=torch.float64)

    original_binary64 = torch.as_tensor(
        graph.compartment_graph.geometry.edge_resistance_ohm,
        dtype=torch.float64,
    ).expand(graph.shape)
    # Prove this morphology distinguishes the two candidate sources: an
    # upcast cannot recover precision discarded when the model was float32.
    assert not torch.equal(
        graph._canonical_edge_resistance_ohm,
        original_binary64,
    )

    for model in (fast, graph):
        model.membrane.insert(pas, g=2.5e-4, e=-70.0)
        model.initialize()
        model.integrator._initialize(model, 0.025, force=True)

    edge_conductance = unbranched_edge_conductance(graph)
    node_conductance = torch.cat(
        (
            torch.zeros(
                edge_conductance.shape[0],
                1,
                dtype=edge_conductance.dtype,
            ),
            edge_conductance,
        ),
        dim=1,
    )
    expected_solver = node_conductance.index_select(1, graph.integrator.solver_order)
    expected_original_edges = node_conductance.index_select(
        1, graph.integrator.edge_child_orig
    )
    assert torch.equal(graph.integrator.a_geom, expected_solver)
    assert torch.equal(graph.integrator.edge_gax_orig, expected_original_edges)

    n_steps = 20
    spatial = torch.linspace(-3.0, 4.0, graph.nc, dtype=torch.float64)
    ve = spatial.reshape(1, 1, -1).expand(n_steps, graph.np, -1).clone()
    fast.run(ve=ve, dt=0.025)
    graph.run(ve=ve, dt=0.025)
    torch.testing.assert_close(fast.v, graph.v, rtol=2e-11, atol=2e-11)


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="DHS compatibility requires the CPU dendra_solvers package",
)
def test_batched_native_cable_dhs_full_shaped_scales_match_fast_path():
    morphology = _serial_morphology()
    fast = dn.Cable.from_morphology(morphology, N=2, dtype=DTYPE).batch(3)
    graph = dn.Cable.from_morphology(
        morphology, N=2, integrator=dhs(), dtype=DTYPE
    ).batch(3)
    shape = fast.shape
    batch = torch.arange(3, dtype=DTYPE).reshape(3, 1, 1)
    row = torch.arange(2, dtype=DTYPE).reshape(1, 2, 1)
    compartment = torch.linspace(0.0, 1.0, fast.nc, dtype=DTYPE).reshape(1, 1, -1)
    area_scale = (0.75 + 0.08 * batch + 0.04 * row + 0.03 * compartment).expand(shape)
    cm_scale = (0.9 + 0.05 * batch + 0.07 * row + 0.02 * compartment).expand(shape)
    rhoa_scale = (0.65 + 0.3 * batch + 0.2 * row).expand(shape)

    for model in (fast, graph):
        model.membrane.insert(pas, g=2.5e-4, e=-70.0)
        model.initialize()
        model.area_scale = area_scale.clone()
        model.cm_scale = cm_scale.clone()
        model.rhoa_scale = rhoa_scale.clone()

    for model in (fast, graph):
        assert torch.equal(model.area_scale, area_scale)
        assert torch.equal(model.cm_scale, cm_scale)
        assert torch.equal(model.rhoa_scale, rhoa_scale)

    n_steps = 12
    spatial = torch.linspace(-2.0, 3.0, fast.nc, dtype=DTYPE)
    ve = spatial.reshape(1, 1, 1, -1).expand(n_steps, *shape).clone()
    fast.run(ve=ve, dt=0.025)
    graph.run(ve=ve, dt=0.025)

    torch.testing.assert_close(fast.v, graph.v, rtol=2e-11, atol=2e-11)


def test_root_interior_physical_path_is_deterministically_reordered_end_to_end():
    morphology = _root_interior_morphology()
    source = morphology.compile()
    assert source.topology.parent_index == (-1, 0, 0, 2, 3)

    model = dn.Cable.from_morphology(morphology, dtype=DTYPE)
    reordered = model.compartment_graph

    # When the source root is not a physical endpoint, choose the endpoint with
    # the smallest original canonical ID, then walk the unique physical path.
    # Here original node 1 is that endpoint: 1 -> 0 -> 2 -> 3 -> 4.
    expected_source_order = (1, 0, 2, 3, 4)
    assert reordered.topology.parent_index == (-1, 0, 1, 2, 3)
    assert reordered.metadata.name == tuple(
        source.metadata.name[index] for index in expected_source_order
    )
    assert reordered.metadata.section_name == tuple(
        source.metadata.section_name[index] for index in expected_source_order
    )
    assert reordered.metadata.segment_index == tuple(
        source.metadata.segment_index[index] for index in expected_source_order
    )
    assert reordered.metadata.section_x == tuple(
        source.metadata.section_x[index] for index in expected_source_order
    )

    for field in (
        "length_um",
        "diameter_um",
        "area_um2",
        "volume_um3",
        "volume_i_um3",
        "volume_o_um3",
        "x_um",
        "y_um",
        "z_um",
        "rhoa_ohm_cm",
        "cm_uF_cm2",
    ):
        source_values = getattr(source.geometry, field)
        assert getattr(reordered.geometry, field) == tuple(
            source_values[index] for index in expected_source_order
        )

    # The first old physical edge is traversed in reverse, but resistance is
    # direction-independent; the remaining source child-edge values retain
    # their exact associations.
    assert reordered.geometry.edge_resistance_ohm == pytest.approx(
        (
            0.0,
            source.geometry.edge_resistance_ohm[1],
            source.geometry.edge_resistance_ohm[2],
            source.geometry.edge_resistance_ohm[3],
            source.geometry.edge_resistance_ohm[4],
        )
    )
    _assert_canonical_geometry(model, reordered)

    # Public labels remain authored-coordinate ordered, independently of the
    # endpoint-to-endpoint solver permutation.
    for label in ("root_region", "extension_region"):
        indices = getattr(model, label).index[-1].tolist()
        section_x = [reordered.metadata.section_x[index] for index in indices]
        assert section_x == sorted(section_x)


def _endpoint_branch() -> dn.Morphology:
    morphology = dn.Morphology()
    root = morphology.section("root", L=10.0, diam=5.0)
    left = morphology.section("left", L=20.0, diam=2.0)
    right = morphology.section("right", L=30.0, diam=2.0)
    left.connect(root.at(1.0), child_end=0)
    right.connect(root.at(1.0), child_end=0)
    return morphology


def _interior_branch() -> dn.Morphology:
    morphology = dn.Morphology()
    trunk = morphology.section("trunk", L=30.0, diam=3.0, nseg=3)
    branch = morphology.section("branch", L=15.0, diam=1.5, nseg=2)
    branch.connect(trunk.at(0.5), child_end=0)
    return morphology


@pytest.mark.parametrize(
    "builder",
    [_endpoint_branch, _interior_branch],
    ids=("retained-junction", "degree-three-interior-attachment"),
)
def test_cable_rejects_branched_native_morphologies(builder: Callable[[], object]):
    morphology = builder()
    graph = morphology.compile()

    with pytest.raises(ValueError, match="unbranched|path|junction"):
        dn.Cable.from_morphology(morphology)
    with pytest.raises(ValueError, match="unbranched|path|junction"):
        dn.Cable.from_compartment_graph(graph)


@pytest.mark.parametrize("subclass", [dn.Axon, dn.Unmyelinated, dn.Myelinated])
@pytest.mark.parametrize("factory", ["from_morphology", "from_compartment_graph"])
def test_specialized_cable_subclasses_require_explicit_native_adapters(
    subclass, factory
):
    morphology = _serial_morphology()
    source = morphology if factory == "from_morphology" else morphology.compile()

    with pytest.raises(NotImplementedError, match="explicit.*adapter|specialized"):
        getattr(subclass, factory)(source)


def test_native_cable_material_diffusion_uses_exact_graph_volume_and_edges():
    morphology = _serial_morphology()
    model = dn.Cable.from_morphology(morphology, dtype=DTYPE)
    graph = model.compartment_graph
    initial = torch.tensor([[1.0, 0.15, 0.8, 0.25, 0.65, 0.05, 0.4]], dtype=DTYPE)
    dt = 0.05

    model.material(
        "tracer",
        fields={"c": initial},
        domain="intracellular",
    )
    model.insert(_ExactGeometryDiffusion)
    model.initialize()
    process = next(iter(model.mech.material_processes.values()))

    # A generic Cable has a graph and must therefore use the DHS/tree finite-
    # volume geometry. Falling through to SpatialOperator1D would reconstruct
    # cylindrical volumes and coupling from center diameters and lengths.
    assert process._diffusion_geometry_kind == "tree"
    operator = next(iter(process._spatial_operators.values()))
    assert isinstance(operator, SpatialOperatorTree)
    assert not isinstance(operator, SpatialOperator1D)
    torch.testing.assert_close(process._mp_volume_i, model.volume_i)
    torch.testing.assert_close(process._mp_diff_geom_um, model.diff_geom_um)
    torch.testing.assert_close(
        process._mp_volume_i,
        _tensor(graph.geometry.volume_i_um3, device=model.device()).unsqueeze(0),
    )
    torch.testing.assert_close(
        process._mp_diff_geom_um,
        _tensor(graph.geometry.edge_diff_geom_um, device=model.device()).unsqueeze(0),
    )

    process.set_dt(dt)
    # Timestep reconfiguration stages and transactionally replaces operators.
    operator = next(iter(process._spatial_operators.values()))
    torch.testing.assert_close(operator.volume, model.volume_i)
    # D=1 and path storage order make a_geom equal the exact child-indexed
    # canonical diffusion geometry, including the zero root entry.
    torch.testing.assert_close(operator.a_geom, model.diff_geom_um)

    exact_volume = model.volume_i
    exact_geom = model.diff_geom_um
    exact_net = torch.zeros_like(initial)
    for child in range(1, graph.n_compartments):
        flux = exact_geom[:, child] * (initial[:, child - 1] - initial[:, child])
        exact_net[:, child] += flux
        exact_net[:, child - 1] -= flux
    expected = initial + dt * exact_net / exact_volume

    analytic = SpatialOperator1D(solver="dense")
    analytic_volume, analytic_conductance = analytic.edge_conductance(
        1.0, model.diam, model.dx
    )
    analytic_net = torch.zeros_like(initial)
    for edge in range(graph.n_compartments - 1):
        flux = analytic_conductance[:, edge] * (initial[:, edge] - initial[:, edge + 1])
        analytic_net[:, edge + 1] += flux
        analytic_net[:, edge] -= flux
    cylindrical_fallback = initial + dt * analytic_net / analytic_volume

    # This strongly tapered cable makes both analytic reconstructions visibly
    # different, ensuring the known answer can distinguish backend selection.
    assert not torch.allclose(analytic_volume, exact_volume, rtol=1e-3, atol=0.0)
    assert not torch.allclose(
        analytic_conductance,
        exact_geom[:, 1:],
        rtol=1e-3,
        atol=0.0,
    )

    process.advance_materials(dt)
    actual = model.mech.materials["tracer"].c
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    assert not torch.allclose(actual, cylindrical_fallback, rtol=1e-6, atol=1e-9)


def test_native_cable_exact_edges_accept_only_spatially_uniform_rhoa_scale():
    model = dn.Cable.from_morphology(_serial_morphology(), N=2, dtype=DTYPE)
    C = model.nc
    row_scale = torch.tensor([0.5, 2.0], dtype=DTYPE).reshape(2, 1)
    model.rhoa_scale = row_scale.expand(2, C).clone()

    conductance = unbranched_edge_conductance(model)
    expected = (model._canonical_edge_resistance_ohm[:, 1:] * row_scale).reciprocal()
    torch.testing.assert_close(conductance, expected)

    model.rhoa_scale = torch.linspace(0.75, 1.25, C, dtype=DTYPE).expand(2, C).clone()
    with pytest.raises(ValueError, match="spatially uniform rhoa_scale"):
        unbranched_edge_conductance(model)


@pytest.mark.parametrize(
    "name",
    [
        "diam",
        "dx",
        "volume_i",
        "diff_geom_um",
        "_canonical_area_cm2",
        "_canonical_edge_resistance_ohm",
    ],
)
def test_native_cable_rejects_post_compile_geometry_mutation(name):
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    value = getattr(model, name)
    value.reshape(-1)[-1].add_(1.0)

    with pytest.raises(RuntimeError, match=rf"{name!r}.*modified"):
        model.initialize()


def test_native_cable_rejects_post_compile_rhoa_mutation():
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.rhoa = model.rhoa * 1.01

    with pytest.raises(RuntimeError, match="'rhoa'.*modified"):
        model.initialize()


def test_native_cable_rejects_morphology_fingerprint_mutation():
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.run(tstop=0.025, dt=0.025)

    model._canonical_morphology_fingerprint[0].bitwise_xor_(1)
    time_before = model.t.clone()
    voltage_before = model.v.clone()
    with pytest.raises(RuntimeError, match="fingerprint was modified"):
        model.step(dt=0.025)
    torch.testing.assert_close(model.t, time_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.v, voltage_before, rtol=0.0, atol=0.0)


def test_native_cable_revalidates_after_initialize_at_execution_boundaries():
    morphology = dn.Morphology()
    morphology.section("single", L=10.0, diam=2.0, labels=("membrane",))
    model = dn.Cable.from_morphology(morphology, dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.run(tstop=0.025, dt=0.025)

    model.dx.add_(1.0)
    with pytest.raises(RuntimeError, match="'dx'.*modified"):
        model.run(tstop=0.025, dt=0.025)
    with pytest.raises(RuntimeError, match="'dx'.*modified"):
        model.step(dt=0.025)


def test_network_execution_revalidates_native_cable_geometry():
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    network = dn.Network({"cable": model})
    network.initialize(dt=0.025)
    network.run(tstop=0.025)

    model.volume_i[..., -1].add_(1.0)
    with pytest.raises(RuntimeError, match="'volume_i'.*modified"):
        network.run(tstop=0.025)


@pytest.mark.parametrize("name", ("rhoa_scale", "cm", "cm_scale", "area_scale"))
@pytest.mark.parametrize("mutation", ("inplace", "replacement"))
def test_initialized_solver_dependencies_require_reinitialization_atomically(
    name, mutation
):
    """A same-dt continuation must never bless stale Cable coefficients."""
    dt = 0.025
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.run(tstop=dt, dt=dt)

    current = getattr(model, name)
    changed = current.detach().clone() * 1.125
    if mutation == "inplace":
        current.mul_(1.125)
    else:
        setattr(model, name, changed)

    time_before = model.t.clone()
    voltage_before = model.v.clone()
    workspace_before = {
        field: getattr(model.integrator, field).clone()
        for field in (
            "diag_base",
            "g_edge_Cinv",
            "g_edge_Cinv_right",
            "cm_inv",
            "scale",
            "lower",
            "upper",
        )
    }

    with pytest.raises(RuntimeError, match="initializ"):
        model.step(dt=dt)

    torch.testing.assert_close(model.t, time_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.v, voltage_before, rtol=0.0, atol=0.0)
    for field, expected in workspace_before.items():
        torch.testing.assert_close(
            getattr(model.integrator, field), expected, rtol=0.0, atol=0.0
        )

    # Explicit initialization resolves the stale-workspace state and permits a
    # subsequent same-dt continuation. The opt-out preserves an intentional
    # direct override instead of repopulating it from its parameter source.
    model.initialize(populate_parameter_buffers=False)
    torch.testing.assert_close(getattr(model, name), changed, rtol=0.0, atol=0.0)
    model.step(dt=dt)
    assert float(model.t) == pytest.approx(dt)


@pytest.mark.parametrize("name", ("rhoa_scale", "cm"))
def test_inference_tensor_solver_dependency_mutation_is_rejected_atomically(name):
    """Unversioned inference tensors must not permit stale Cable coefficients."""
    dt = 0.025
    with torch.inference_mode():
        model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
        model.membrane.insert(pas)
        model.initialize()
        model.step(dt=dt)

        dependency = getattr(model, name)
        assert torch.is_inference(dependency)
        time_before = model.t.clone()
        voltage_before = model.v.clone()
        workspace_before = {
            field: getattr(model.integrator, field).clone()
            for field in (
                "diag_base",
                "g_edge_Cinv",
                "g_edge_Cinv_right",
                "cm_inv",
                "scale",
                "lower",
                "upper",
            )
        }

        dependency.mul_(1.125)
        with dn.ctx(RUNTIME_CONTRACT_VALIDATION="versioned"):
            with pytest.raises(RuntimeError, match="initializ"):
                model.step(dt=dt)

        torch.testing.assert_close(model.t, time_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(model.v, voltage_before, rtol=0.0, atol=0.0)
        for field, expected in workspace_before.items():
            torch.testing.assert_close(
                getattr(model.integrator, field), expected, rtol=0.0, atol=0.0
            )


@pytest.mark.parametrize("clone_kind", ("deepcopy", "pickleable"))
def test_valid_initialized_cable_clone_resumes_without_reset(clone_kind):
    dt = 0.025
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.step(dt=dt)

    if clone_kind == "deepcopy":
        cloned = copy.deepcopy(model)
    else:
        cloned = model.pickleable(clone=True)

    source_time = model.t.clone()
    source_voltage = model.v.clone()
    torch.testing.assert_close(cloned.t, source_time, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cloned.v, source_voltage, rtol=0.0, atol=0.0)

    cloned.step(dt=dt)

    assert float(cloned.t) == pytest.approx(float(source_time) + dt)
    torch.testing.assert_close(model.t, source_time, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.v, source_voltage, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("clone_kind", ("deepcopy", "pickleable"))
def test_stale_initialized_cable_clone_remains_rejected(clone_kind):
    dt = 0.025
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.step(dt=dt)
    model.cm_scale.mul_(1.125)

    if clone_kind == "deepcopy":
        cloned = copy.deepcopy(model)
    else:
        cloned = model.pickleable(clone=True)

    time_before = cloned.t.clone()
    voltage_before = cloned.v.clone()
    with pytest.raises(RuntimeError, match="initializ"):
        cloned.step(dt=dt)
    torch.testing.assert_close(cloned.t, time_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cloned.v, voltage_before, rtol=0.0, atol=0.0)


def test_valid_initialized_cable_detach_resumes_without_reset():
    dt = 0.025
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.step(dt=dt)
    time_before = model.t.clone()

    model.detach()
    assert model.initialized
    assert model.integrator.initialized
    model.step(dt=dt)

    assert float(model.t) == pytest.approx(float(time_before) + dt)


def test_stale_initialized_cable_detach_remains_rejected():
    dt = 0.025
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.step(dt=dt)
    model.cm_scale.mul_(1.125)

    model.detach()
    time_before = model.t.clone()
    voltage_before = model.v.clone()
    with pytest.raises(RuntimeError, match="initializ"):
        model.step(dt=dt)
    torch.testing.assert_close(model.t, time_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.v, voltage_before, rtol=0.0, atol=0.0)


def test_network_requires_reinitialization_after_cable_dependency_change():
    dt = 0.025
    cable = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    cable.membrane.insert(pas)
    network = dn.Network({"cable": cable})
    network.initialize(dt=dt)
    network.step()

    cable.cm_scale.mul_(1.125)
    network_time = network.t.clone()
    cable_time = cable.t.clone()
    cable_voltage = cable.v.clone()

    with pytest.raises(RuntimeError, match="initializ"):
        network.step()

    torch.testing.assert_close(network.t, network_time, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cable.t, cable_time, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cable.v, cable_voltage, rtol=0.0, atol=0.0)

    network.initialize(dt=dt)
    network.step()
    assert float(network.t) == pytest.approx(dt)


def test_network_rejects_child_integrator_invalidated_by_state_load():
    dt = 0.025
    source = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    source.membrane.insert(pas)
    source.initialize()
    source.run(tstop=dt, dt=dt)

    cable = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    cable.membrane.insert(pas)
    network = dn.Network({"cable": cable})
    network.initialize(dt=dt)
    network.step()

    cable.load_state_dict(source.state_dict())
    assert not cable.integrator.initialized
    network_time = network.t.clone()
    cable_time = cable.t.clone()
    cable_voltage = cable.v.clone()

    with pytest.raises(RuntimeError, match="initializ"):
        network.step()

    torch.testing.assert_close(network.t, network_time, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cable.t, cable_time, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cable.v, cable_voltage, rtol=0.0, atol=0.0)

    network.initialize(dt=dt)
    network.step()
    assert float(network.t) == pytest.approx(dt)


def test_versioned_runtime_validation_avoids_repeated_full_geometry_scans(monkeypatch):
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.run(tstop=0.025, dt=0.025)
    # Establish the signature immediately before counting full validations.
    model._validate_canonical_geometry()

    calls = 0
    original = model._validate_canonical_geometry

    def counted_validation():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(model, "_validate_canonical_geometry", counted_validation)
    with dn.ctx(RUNTIME_CONTRACT_VALIDATION="versioned"):
        model.step(dt=0.025)
        model.step(dt=0.025)
    assert calls == 0

    model.dx[..., -1].add_(1.0)
    time_before = model.t.clone()
    voltage_before = model.v.clone()
    with dn.ctx(RUNTIME_CONTRACT_VALIDATION="versioned"):
        with pytest.raises(RuntimeError, match="'dx'.*modified"):
            model.step(dt=0.025)
    assert calls == 1
    torch.testing.assert_close(model.t, time_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.v, voltage_before, rtol=0.0, atol=0.0)


def test_strict_runtime_validation_detects_version_counter_bypass():
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.run(tstop=0.025, dt=0.025)

    version = model.dx._version
    model.dx.data[..., -1].add_(1.0)
    assert model.dx._version == version

    # The default fast guard cannot observe unsupported raw/.data writes.
    with dn.ctx(RUNTIME_CONTRACT_VALIDATION="versioned"):
        model.step(dt=0.025)

    time_before = model.t.clone()
    voltage_before = model.v.clone()
    with dn.ctx(RUNTIME_CONTRACT_VALIDATION="strict"):
        with pytest.raises(RuntimeError, match="'dx'.*modified"):
            model.step(dt=0.025)
    torch.testing.assert_close(model.t, time_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.v, voltage_before, rtol=0.0, atol=0.0)


def test_versioned_runtime_validation_tracks_uniformity_constrained_rhoa_scale():
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.run(tstop=0.025, dt=0.025)

    model.rhoa_scale = torch.linspace(0.9, 1.1, model.nc, dtype=DTYPE).unsqueeze(0)
    with dn.ctx(RUNTIME_CONTRACT_VALIDATION="versioned"):
        with pytest.raises(ValueError, match="spatially uniform rhoa_scale"):
            model.step(dt=0.025)


def test_initialize_runtime_validation_skips_boundaries_but_not_rebuild_or_checkpoint():
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
    model.membrane.insert(pas)
    model.initialize()
    model.run(tstop=0.025, dt=0.025)
    cached_edges = model.integrator.g_edge_Cinv.clone()

    model._canonical_edge_resistance_ohm[..., 1].mul_(1.01)
    with dn.ctx(RUNTIME_CONTRACT_VALIDATION="initialize"):
        model.step(dt=0.025)
        torch.testing.assert_close(
            model.integrator.g_edge_Cinv, cached_edges, rtol=0.0, atol=0.0
        )

        # A timestep change rebuilds solver workspaces and therefore retains an
        # unconditional full validation even in the low-overhead mode.
        time_before = model.t.clone()
        voltage_before = model.v.clone()
        with pytest.raises(RuntimeError, match="edge_resistance.*modified"):
            model.step(dt=0.0125)
        torch.testing.assert_close(model.t, time_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(model.v, voltage_before, rtol=0.0, atol=0.0)

        with pytest.raises(RuntimeError, match="edge_resistance.*modified"):
            model.state_dict_for_checkpoint()


def test_versioned_runtime_validation_falls_back_to_full_for_inference_tensors(
    monkeypatch,
):
    with torch.inference_mode():
        model = dn.Cable.from_morphology(_serial_morphology(), dtype=DTYPE)
        model.membrane.insert(pas)
        model.initialize()
        model.run(tstop=0.025, dt=0.025)
    assert torch.is_inference(model.dx)

    calls = 0
    original = model._validate_canonical_geometry

    def counted_validation():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(model, "_validate_canonical_geometry", counted_validation)
    with dn.ctx(RUNTIME_CONTRACT_VALIDATION="versioned"):
        model.step(dt=0.025)
        model.step(dt=0.025)
    assert calls == 2


def test_native_cable_dtype_conversion_preserves_frozen_reference_contract():
    morphology = _serial_morphology()
    model = dn.Cable.from_morphology(morphology, dtype=torch.float32).to(
        dtype=torch.float64
    )
    model.membrane.insert(pas)
    model.initialize()
    model.run(tstop=0.025, dt=0.025)
    assert model.dtype() == torch.float64


def test_post_initialization_dtype_conversion_rebuilds_direct_population_workspace():
    dt = 0.025
    model = dn.Cable.from_morphology(_serial_morphology(), dtype=torch.float64)
    model.membrane.insert(pas)
    # Exercise the PointProcess scaler closure as well as distributed current
    # aggregation scratch across the relocation.
    model.readout.insert(expsyn.rename("relocation_syn"), e=0.0, tau=1.0)
    model.initialize()
    model.run(tstop=dt, dt=dt)

    model.to(dtype=torch.float32)
    converted_workspace = model.integrator.g_edge_Cinv
    time_before = model.t.clone()

    # Module relocation also moves the handler's unregistered aggregation
    # scratch and rebuilds dtype/device-specific point-process scale closures.
    assert model.initialized
    assert not model.integrator.initialized
    assert all(buffer.dtype == torch.float32 for buffer in model.mech._buf_i)
    assert all(buffer.dtype == torch.float32 for buffer in model.mech._buf_g)
    assert model.integrator.g_edge_Cinv is converted_workspace

    # Direct Population execution can now rebuild only its invalidated voltage
    # workspace without resetting voltage, mechanism state, or simulation time.
    model.run(tstop=dt, dt=dt)

    assert model.dtype() == torch.float32
    assert model.integrator.initialized
    assert model.integrator.g_edge_Cinv is not converted_workspace
    assert float(model.t) == pytest.approx(float(time_before) + dt)

    area = torch.broadcast_to(model.area * model.area_scale, model.shape).reshape(
        -1, model.nc
    )
    cm = torch.broadcast_to(model.cm * model.cm_scale, model.shape).reshape(
        -1, model.nc
    )
    capacitance = 1.0e-6 * area * cm
    expected = unbranched_edge_conductance(model) / capacitance[:, :-1]
    torch.testing.assert_close(model.integrator.g_edge_Cinv, expected)


def test_native_cable_upcast_runtime_graph_and_material_share_frozen_buffers():
    model = dn.Cable.from_morphology(_serial_morphology(), N=2, dtype=torch.float32).to(
        dtype=torch.float64
    )
    canonical = model.compartment_graph

    provenance_resistance = torch.as_tensor(
        canonical.geometry.edge_resistance_ohm, dtype=torch.float64
    ).expand_as(model._canonical_edge_resistance_ohm)
    provenance_diffusion = torch.as_tensor(
        canonical.geometry.edge_diff_geom_um, dtype=torch.float64
    ).expand_as(model.diff_geom_um)
    # The immutable CompartmentGraph remains the original binary64 provenance,
    # while the runtime buffers correctly retain the widened float32 values.
    # This proves the regression can distinguish the two candidate sources.
    assert not torch.equal(provenance_resistance, model._canonical_edge_resistance_ohm)
    assert not torch.equal(provenance_diffusion, model.diff_geom_um)

    def graph_edges(graph, name):
        values = [0.0]
        for child, parent in enumerate(canonical.topology.parent_index):
            if parent != -1:
                values.append(graph.edges[parent, child][name])
        return torch.as_tensor(values, dtype=model.dtype()).unsqueeze(0)

    graph = model.graph
    assert torch.equal(
        graph_edges(graph, "R_ohm").expand_as(model._canonical_edge_resistance_ohm),
        model._canonical_edge_resistance_ohm,
    )
    assert torch.equal(
        graph_edges(graph, "diff_geom_um").expand_as(model.diff_geom_um),
        model.diff_geom_um,
    )
    for assembled in model.assemble_graphs():
        assert torch.equal(
            graph_edges(assembled, "R_ohm").expand_as(
                model._canonical_edge_resistance_ohm
            ),
            model._canonical_edge_resistance_ohm,
        )
        assert torch.equal(
            graph_edges(assembled, "diff_geom_um").expand_as(model.diff_geom_um),
            model.diff_geom_um,
        )

    graph.edges[0, 1]["diff_geom_um"] = -1.0
    assert model.graph.edges[0, 1]["diff_geom_um"] > 0.0

    initial = torch.linspace(
        0.1, 0.9, model.nc, dtype=model.dtype(), device=model.device()
    ).expand(model.shape)
    model.material("tracer", fields={"c": initial}, domain="intracellular")
    model.insert(_ExactGeometryDiffusion)
    model.initialize()
    process = next(iter(model.mech.material_processes.values()))
    assert torch.equal(process._mp_diff_geom_um, model.diff_geom_um)

    process.set_dt(0.05)
    operator = next(iter(process._spatial_operators.values()))
    assert isinstance(operator, SpatialOperatorTree)
    assert torch.equal(operator.a_geom, model.diff_geom_um)


def test_native_cable_cross_dtype_load_preserves_target_canonical_geometry():
    morphology = _serial_morphology()
    source = dn.Cable.from_morphology(morphology, dtype=torch.float32)
    target = dn.Cable.from_morphology(morphology, dtype=torch.float64)
    source.v.fill_(-51.25)

    frozen_names = {
        "diam",
        "dx",
        "rhoa",
        "volume",
        "volume_um3",
        "volume_i",
        "volume_o",
        "diff_geom_um",
        "diff_parent_index",
        "_canonical_area_cm2",
        "_canonical_edge_resistance_ohm",
        "_canonical_morphology_fingerprint",
        "_canonical_geometry_reference",
    }
    frozen_before = {
        name: value.clone()
        for name, value in target.state_dict().items()
        if name in frozen_names or name.startswith("rhoa_param.")
    }

    target.load_state_dict(source.state_dict())
    target._validate_canonical_geometry()
    frozen_after = target.state_dict()
    for name, expected in frozen_before.items():
        assert torch.equal(frozen_after[name], expected), name
    torch.testing.assert_close(
        target.v,
        source.v.to(dtype=target.dtype()),
        rtol=0.0,
        atol=0.0,
    )

    graph = target.compartment_graph
    expected_resistance = torch.as_tensor(
        graph.geometry.edge_resistance_ohm, dtype=torch.float64
    ).unsqueeze(0)
    expected_area = (
        torch.as_tensor(graph.geometry.area_um2, dtype=torch.float64) * 1.0e-8
    ).unsqueeze(0)
    expected_volume_i = torch.as_tensor(
        graph.geometry.volume_i_um3, dtype=torch.float64
    ).unsqueeze(0)
    expected_diff_geom = torch.as_tensor(
        graph.geometry.edge_diff_geom_um, dtype=torch.float64
    ).unsqueeze(0)

    # UB consumes the canonical resistance buffer while DHS consumes the graph.
    # Both must remain the same binary64 geometry after a float32 restore.
    assert torch.equal(target._canonical_edge_resistance_ohm, expected_resistance)
    assert torch.equal(target._canonical_area_cm2, expected_area)
    graph_resistance = torch.tensor(
        [
            0.0,
            *(
                target.graph.edges[parent, child]["R_ohm"]
                for parent, child in zip(range(target.nc - 1), range(1, target.nc))
            ),
        ],
        dtype=torch.float64,
    ).unsqueeze(0)
    assert torch.equal(graph_resistance, target._canonical_edge_resistance_ohm)
    assert torch.equal(
        unbranched_edge_conductance(target),
        expected_resistance[:, 1:].reciprocal(),
    )

    # Material processes snapshot exact volume and diffusion geometry from the
    # same canonical target rather than a widened float32 checkpoint payload.
    initial = torch.linspace(0.1, 0.9, target.nc, dtype=torch.float64).unsqueeze(0)
    target.material("tracer", fields={"c": initial}, domain="intracellular")
    target.insert(_ExactGeometryDiffusion)
    target.membrane.insert(pas)
    target.initialize()
    process = next(iter(target.mech.material_processes.values()))
    assert torch.equal(process._mp_volume_i, expected_volume_i)
    assert torch.equal(process._mp_diff_geom_um, expected_diff_geom)
    assert target.dtype() == torch.float64


def test_built_native_cable_cross_dtype_load_preserves_nested_geometry_sources():
    morphology = _serial_morphology()

    def built(dtype, initial):
        model = dn.Cable.from_morphology(morphology, dtype=dtype)
        concentration = torch.full(model.shape, initial, dtype=dtype)
        model.material(
            "tracer",
            fields={"c": concentration},
            domain="intracellular",
        )
        model.insert(_ExactGeometryDiffusion)
        model.membrane.insert(pas, g=2.5e-4, e=-70.0)
        model.initialize()
        model.run(tstop=0.025, dt=0.025)
        return model

    source = built(torch.float32, 0.2)
    target = built(torch.float64, 0.8)

    def is_nested_geometry(name):
        leaf = name.rsplit(".", 1)[-1]
        handler_area = name in {"mech.area", "integrator.mech.area"}
        return "." in name and (
            handler_area
            or leaf == "diam"
            or leaf.startswith("_mp_")
            or "._spatial_operators." in name
        )

    target_state = target.state_dict()
    nested_before = {
        name: value.clone()
        for name, value in target_state.items()
        if is_nested_geometry(name)
    }
    source_concentration = source.mech.materials["tracer"].c.clone()

    target.load_state_dict(source.state_dict())

    assert not target.integrator.initialized
    nested_after = target.state_dict()
    for name, expected in nested_before.items():
        assert torch.equal(nested_after[name], expected), name
    torch.testing.assert_close(
        target.mech.materials["tracer"].c,
        source_concentration.to(dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    target._validate_canonical_geometry()

    # The same-dt continuation must rebuild solver/material workspaces from
    # target-local binary64 geometry rather than reuse widened float32 state.
    target.run(tstop=0.025, dt=0.025)
    assert target.integrator.initialized
    process = next(iter(target.mech.material_processes.values()))
    assert torch.equal(process._mp_volume_i, target.volume_i)
    assert torch.equal(process._mp_diff_geom_um, target.diff_geom_um)
    operator = next(iter(process._spatial_operators.values()))
    assert torch.equal(operator.volume, target.volume_i)
    assert torch.equal(operator.a_geom, target.diff_geom_um)


def test_native_cable_rejects_corrupt_same_morphology_state_atomically():
    morphology = _serial_morphology()
    source = dn.Cable.from_morphology(morphology, dtype=DTYPE)
    target = dn.Cable.from_morphology(morphology, dtype=DTYPE)
    source._canonical_area_cm2[..., -1].add_(1.0)
    before = {name: value.clone() for name, value in target.state_dict().items()}

    with pytest.raises(RuntimeError, match="corrupt canonical geometry"):
        target.load_state_dict(source.state_dict())

    after = target.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(after[name], before[name])
    target._validate_canonical_geometry()


def test_native_cable_checkpoint_rejects_same_shape_different_morphology_atomically():
    source_morphology = _serial_morphology()
    target_morphology = _serial_morphology()
    # Preserve node count and topology while changing exact geometry and
    # provenance, the dangerous case for ordinary shape-only state loading.
    target_graph = target_morphology.compile()
    changed_geometry = target_graph.geometry.__class__(
        **{
            name: (
                tuple(value * 1.01 for value in values)
                if name == "area_um2"
                else values
            )
            for name, values in (
                (field, getattr(target_graph.geometry, field))
                for field in target_graph.geometry.__dataclass_fields__
            )
        }
    )
    changed_graph = target_graph.__class__(
        topology=target_graph.topology,
        geometry=changed_geometry,
        metadata=target_graph.metadata,
        schema_version=target_graph.schema_version,
    )

    source = dn.Cable.from_morphology(source_morphology, dtype=DTYPE)
    target = dn.Cable.from_compartment_graph(changed_graph, dtype=DTYPE)
    source.membrane.insert(pas)
    target.membrane.insert(pas)
    before = {name: value.clone() for name, value in target.state_dict().items()}

    with pytest.raises(RuntimeError, match="different canonical morphology"):
        target.load_state_dict(source.state_dict())

    after = target.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(after[name], before[name])

    source.initialize()
    target.initialize()
    runtime_before = target.v.clone()
    with pytest.raises(RuntimeError, match="different canonical morphology"):
        target.restore_dict_from_checkpoint(source.state_dict_for_checkpoint())
    torch.testing.assert_close(target.v, runtime_before)


def test_native_cable_population_batch_device_dtype_and_state_round_trip():
    morphology = _serial_morphology()
    model = dn.Cable.from_morphology(
        morphology,
        N=2,
        device=torch.device("cpu"),
        dtype=DTYPE,
    )
    template = model.compartment_graph
    model.membrane.insert(pas, g=2.5e-4, e=-72.0)
    model.batch(3)

    assert model.shape == (3, 2, template.n_compartments)
    assert model.device() == torch.device("cpu")
    assert model.dtype() == DTYPE
    assert model.compartment_graph is template
    assert model.membrane.shape == model.shape
    # Immutable morphology geometry has one row per core population and
    # broadcasts over explicit batch dimensions.
    assert model.dx.shape == (2, template.n_compartments)
    assert model._canonical_area_cm2.shape == (2, template.n_compartments)
    assert model._canonical_edge_resistance_ohm.shape == (
        2,
        template.n_compartments,
    )

    model.eval()
    model.initialize()
    model.run(tstop=0.05, dt=0.025)
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}

    restored = dn.Cable.from_morphology(morphology, N=2, dtype=DTYPE)
    restored.membrane.insert(pas, g=2.5e-4, e=-72.0)
    restored.batch(3)
    restored.initialize()
    restored.load_state_dict(snapshot)

    assert restored.compartment_graph == template
    torch.testing.assert_close(restored.v, model.v)
    torch.testing.assert_close(restored.area, model.area)
    torch.testing.assert_close(
        restored._canonical_edge_resistance_ohm,
        model._canonical_edge_resistance_ohm,
    )


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="Tree parity requires the CPU dendra_solvers package",
)
def test_native_cable_passive_trajectory_matches_tree_with_exact_geometry():
    morphology = _serial_morphology()
    cable = dn.Cable.from_morphology(
        morphology,
        celsius=22.0,
        v_init=-65.0,
        dtype=DTYPE,
    )
    tree = dn.Tree.from_morphology(
        morphology,
        celsius=22.0,
        v_init=-65.0,
        dtype=DTYPE,
    )
    for model in (cable, tree):
        model.membrane.insert(pas, g=3.0e-4, e=-71.0)
        model.stimulus_site.inject(dn.mono_rect(amp=0.18 * nA, delay=0.10, pw=0.25))
        model.eval()
        model.initialize()
        model.run(tstop=0.75, dt=0.025)

    # The serial source is already in path order, so public compartment storage
    # aligns exactly. This catches cylindrical area or reconstructed-resistance
    # fallbacks even when all metadata appears correct.
    torch.testing.assert_close(cable.v, tree.v, rtol=2.0e-11, atol=2.0e-11)
