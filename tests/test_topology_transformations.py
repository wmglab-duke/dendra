"""Known-answer tests for topology, slicing, and composite populations."""

from __future__ import annotations

import math

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.graph import (
    get_area_from_graph,
    share_topology_isomorphic,
    share_topology_labeled,
)
from dendra.models.mod import pas
from dendra.models.multi import (
    MultiPopulation,
    _assess_type_and_make_integrator,
    _check_celsius,
    _expand_v_init_like,
    flatten_key,
    indices,
    key_to_flat_index,
    offsets,
)
from dendra.models.slice import (
    _assess_shape_compatibility,
    _merge_indices,
    compose_indices,
    expand_into_shape,
    parse_key,
)
from dendra.models.stim.waveform import constant
from dendra.models.tree import (
    gather_diffusion_edges,
    gather_membrane,
    gather_morphology,
)

DTYPE = torch.float64


def _population(*, n=2, c=4, v_init=-65.0):
    return dn.Population(N=n, C=c, v_init=v_init, dtype=DTYPE)


def _morphology_graph(*, origin_x=0.0, axis="z"):
    graph = nx.DiGraph()
    if axis == "z":
        points = [
            (origin_x, 0.0, 0.0),
            (origin_x, 0.0, 2.0),
            (origin_x + 1.0, 0.0, 0.0),
        ]
    elif axis == "x":
        points = [
            (origin_x, 0.0, 0.0),
            (origin_x + 2.0, 0.0, 0.0),
            (origin_x, 1.0, 0.0),
        ]
    else:  # pragma: no cover - helper guard
        raise ValueError(axis)

    names = ["Cell.soma[0](0.5)", "Cell.dend[0](0.5)", "Cell.axon[0](0.5)"]
    for node, (name, point) in enumerate(zip(names, points)):
        graph.add_node(
            node,
            name=name,
            L=2.0,
            diam=2.0,
            Ra=100.0,
            cm=1.0,
            area=2.0 * math.pi,
            x=point[0],
            y=point[1],
            z=point[2],
        )
    graph.add_edge(0, 1, diff_geom_um=1.5)
    graph.add_edge(0, 2, diff_geom_um=0.75)
    return graph


def test_graph_area_requires_complete_metadata_and_converts_units():
    graph = nx.DiGraph()
    graph.add_node(0, area=2.0)
    graph.add_node(1, area=5.0)
    assert torch.equal(get_area_from_graph(graph), torch.tensor([2.0e-8, 5.0e-8]))

    del graph.nodes[1]["area"]
    assert get_area_from_graph(graph) is None


def test_labeled_topology_reports_each_mismatch_class():
    graph = nx.DiGraph([(0, 1), (1, 2)])
    assert share_topology_labeled([graph, graph.copy()])[0]
    assert share_topology_labeled([]) == (False, "No graphs provided.")

    wrong_kind = nx.Graph(graph)
    ok, reason = share_topology_labeled([graph, wrong_kind])
    assert not ok and "Type mismatch" in reason

    wrong_nodes = graph.copy()
    wrong_nodes.remove_node(2)
    wrong_nodes.add_node(3)
    ok, reason = share_topology_labeled([graph, wrong_nodes])
    assert not ok and "Node set mismatch" in reason

    wrong_edges = graph.copy()
    wrong_edges.remove_edge(1, 2)
    wrong_edges.add_edge(0, 2)
    ok, reason = share_topology_labeled([graph, wrong_edges])
    assert not ok and "Edge set mismatch" in reason


@pytest.mark.parametrize(
    "graph_type",
    [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph],
)
def test_topology_comparison_handles_direction_and_edge_multiplicity(graph_type):
    left = graph_type()
    left.add_nodes_from([0, 1, 2])
    left.add_edge(0, 1)
    left.add_edge(1, 2)
    if left.is_multigraph():
        left.add_edge(0, 1)

    same = left.copy()
    assert share_topology_labeled([left, same])[0]

    changed = left.copy()
    if changed.is_multigraph():
        changed.add_edge(0, 1)
    else:
        changed.remove_edge(1, 2)
    assert not share_topology_labeled([left, changed])[0]


def test_unlabeled_topology_accepts_relabeling_but_not_type_or_shape_changes():
    path = nx.DiGraph([("root", "mid"), ("mid", "tip")])
    relabeled = nx.DiGraph([(10, 20), (20, 30)])
    assert share_topology_isomorphic([path, relabeled])[0]
    assert share_topology_isomorphic([]) == (False, "No graphs provided.")

    assert not share_topology_isomorphic([path, nx.Graph(relabeled)])[0]
    branch = nx.DiGraph([(10, 20), (10, 30), (10, 40)])
    ok, reason = share_topology_isomorphic([path, branch])
    assert not ok and "Not isomorphic" in reason


@pytest.mark.parametrize("graph_type", [nx.Graph, nx.MultiGraph])
def test_labeled_undirected_topology_accepts_heterogeneous_node_labels(graph_type):
    graph = graph_type()
    graph.add_edge(1, "branch")
    if graph.is_multigraph():
        graph.add_edge(1, "branch")
    assert share_topology_labeled([graph, graph.copy()])[0]


def test_index_helpers_match_two_stage_torch_indexing():
    shape = (3, 5)
    values = torch.arange(math.prod(shape)).reshape(shape)
    first = (torch.tensor([0, 2]), slice(1, 5))
    second = (slice(None), torch.tensor([0, 2]))
    composed = compose_indices(shape, first, second)
    assert torch.equal(values[first][second], values[composed])

    spec = parse_key((Ellipsis, torch.tensor([True, False, True, False, True])), shape)
    assert spec.shape == (3, 3)
    assert not spec.is_scalar

    pop = _population(n=3, c=5)
    assert torch.equal(
        spec.to_key(pop),
        torch.arange(15).reshape(shape)[spec.index].reshape(-1),
    )
    scalar = parse_key((1, 2), shape)
    assert scalar.is_scalar and scalar.shape == ()


def test_expand_into_shape_preserves_source_dtype_device_and_fill():
    source = torch.tensor([2.0, 4.0], dtype=DTYPE)
    expanded = expand_into_shape(source, (torch.tensor([0, 2]),), (4,), fill_value=-1)
    assert expanded.tolist() == [2.0, -1.0, 4.0, -1.0]
    assert expanded.dtype == source.dtype


def test_slice_reads_writes_nests_labels_and_represents_selection():
    pop = _population()
    pop.v.copy_(torch.arange(8, dtype=DTYPE).reshape(2, 4))
    middle = pop[:, 1:3]

    assert middle.shape == (2, 2)
    assert middle.numel() == 4
    assert not middle.is_scalar and not middle.is_empty
    assert torch.equal(middle.inspect("v"), pop.v[:, 1:3])
    assert torch.equal(middle.get("v"), pop.v[:, 1:3])
    assert "shape=torch.Size([2, 2])" in repr(middle)

    middle.set("v", torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], dtype=DTYPE))
    assert pop.v.tolist() == [[0.0, -1.0, -2.0, 3.0], [4.0, -3.0, -4.0, 7.0]]
    middle.v = 9.0
    assert torch.equal(pop.v[:, 1:3], torch.full((2, 2), 9.0, dtype=DTYPE))

    last = middle[:, 1]
    assert torch.equal(last.v, pop.v[:, 2])
    last.label("last")
    assert middle.last is last
    assert middle[:, :0].is_empty


def test_slice_injection_is_noop_when_empty_and_batches_registered_labels():
    pop = _population()
    waveform = constant(value=2.0)
    pop[:, :0].inject(waveform)
    assert pop.injections == []

    pop[:, 1].label("target")
    pop.target.inject(waveform)
    assert len(pop.injections) == 1
    assert pop.injections[0][0] is waveform

    pop.batch(3)
    assert pop.target.shape == (3, 2)
    assert torch.equal(pop.target.v, pop.v[..., 1])


def test_slice_sets_dense_and_sparse_mechanism_storage():
    dense = _population(n=2, c=3)
    dense.insert(pas, g=0.1, e=-70.0)
    dense.build()
    dense[:, 1].set("g", torch.tensor([0.4, 0.6], dtype=DTYPE), mechanism="pas")
    assert dense.mech.pas.g[:, 1].tolist() == pytest.approx([0.4, 0.6])
    assert torch.equal(dense[:, 1].mech.pas.g, dense.mech.pas.g[:, 1])

    sparse = _population(n=2, c=3)
    sparse[:, 1:].insert(pas, g=0.2, e=-65.0)
    sparse.build()
    sparse[:, 2].set("g", torch.tensor([0.7, 0.8], dtype=DTYPE), mechanism="pas")
    assert sparse[:, 2].inspect("g", mechanism="pas").tolist() == pytest.approx(
        [0.7, 0.8]
    )
    assert torch.isnan(sparse[:, 0].inspect("g", mechanism="pas")).all()


def test_concat_slices_matches_torch_cat_along_dimension():
    pop = _population()
    pop.v.copy_(torch.arange(8, dtype=DTYPE).reshape(2, 4))
    left = pop[:, :1]
    right = pop[:, 2:]
    merged = dn.concat_slices([left, right], dim=-1)
    assert merged.shape == (2, 3)
    assert torch.equal(merged.v, torch.cat([left.v, right.v], dim=-1))

    repeated = dn.concat_slices([pop[:, [1]], pop[:, [1]]], dim=1)
    assert repeated.shape == (2, 2)
    assert torch.equal(repeated.v, torch.cat([pop[:, [1]].v] * 2, dim=1))


def test_concat_slices_flatten_supports_unequal_lengths_and_arbitrary_regions():
    pop = _population()
    pop.v.copy_(torch.arange(8, dtype=DTYPE).reshape(2, 4))
    first = pop[0, :1]
    second = pop[1, 1:]
    merged = dn.concat_slices([first, second], dim=None)
    assert merged.shape == (4,)
    assert torch.equal(merged.v, torch.cat([first.v.flatten(), second.v.flatten()]))


def test_concat_slices_handles_different_nonconcat_indices_and_raw_keys():
    pop = _population(n=4, c=6)
    pop.v.copy_(torch.arange(24, dtype=DTYPE).reshape(4, 6))

    row_blocks = [pop[:2, [0, 1]], pop[2:, [4, 5]]]
    merged_rows = dn.concat_slices(row_blocks, dim=0)
    assert torch.equal(merged_rows.v, torch.cat([item.v for item in row_blocks], dim=0))

    column_blocks = [pop[[0, 1], :2], pop[[2, 3], 4:]]
    merged_columns = dn.concat_slices(column_blocks, dim=1)
    assert torch.equal(
        merged_columns.v, torch.cat([item.v for item in column_blocks], dim=1)
    )

    duplicated = dn.concat_slices([pop[:], pop[:]], dim=0)
    assert torch.equal(duplicated.v, torch.cat([pop.v, pop.v], dim=0))


def test_concat_slices_handles_rank_reduced_and_nested_selections():
    pop = _population(n=4, c=6)
    pop.v.copy_(torch.arange(24, dtype=DTYPE).reshape(4, 6))

    vectors = [pop[0, :2], pop[1, 2:5]]
    merged_vectors = dn.concat_slices(vectors, dim=0)
    assert torch.equal(merged_vectors.v, torch.cat([item.v for item in vectors], dim=0))

    nested = [pop[:, 1:][1:3, :2], pop[:, 2:][1:3, 2:4]]
    merged_nested = dn.concat_slices(nested, dim=1)
    assert torch.equal(merged_nested.v, torch.cat([item.v for item in nested], dim=1))


def test_concat_slices_rejects_empty_cross_model_and_incompatible_inputs():
    pop = _population()
    other = _population()
    with pytest.raises(ValueError, match="At least one"):
        dn.concat_slices([])
    with pytest.raises(ValueError, match="same underlying model"):
        dn.concat_slices([pop[:, :1], other[:, 1:]])
    with pytest.raises(ValueError, match="Shapes differ"):
        dn.concat_slices([pop[:1, :2], pop[:, 2:]], dim=1)
    with pytest.raises(ValueError, match="out of range"):
        dn.concat_slices([pop[:, :1]], dim=3)

    assert _assess_shape_compatibility([(2,), (3,)], None) == (5,)
    with pytest.raises(ValueError, match="same number of dimensions"):
        _assess_shape_compatibility([(2, 1), (2,)], 0)
    assert torch.equal(
        _merge_indices([torch.tensor([3, 2]), torch.tensor([1])]),
        torch.tensor([3, 2, 1]),
    )


def test_multi_index_helpers_preserve_component_layout():
    populations = {"left": _population(n=2, c=2), "right": _population(n=1, c=3)}
    assert offsets(populations) == [0, 4]
    component_indices = indices(populations)
    assert component_indices[0].tolist() == [[0, 1], [2, 3]]
    assert component_indices[1].tolist() == [[4, 5, 6]]
    assert key_to_flat_index(component_indices[0], (slice(None), 1)).tolist() == [1, 3]
    assert flatten_key(4, (2, 2), (slice(None), 0)).tolist() == [0, 2]


def test_concat_models_preserves_state_geometry_labels_stimuli_and_mechanisms():
    left = _population(n=2, c=2, v_init=[-60.0, -61.0])
    right = _population(n=1, c=1, v_init=-62.0)
    left.x.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE))
    right.x.fill_(5.0)
    left[:, 1].label("tip")
    waveform = constant(value=3.0)
    left[:, 0].inject(waveform)
    left.insert(pas, g=0.1, e=-70.0)
    right[:, 0].insert(pas, g=0.2, e=-60.0)

    multi = dn.concat_models({"left": left, "right": right})
    assert multi.shape == (1, 5)
    assert multi.v_init.tolist() == [-60.0, -61.0, -60.0, -61.0, -62.0]
    assert multi.x.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0]]
    assert len(multi) == 2 and list(multi) == [left, right]
    assert multi.left.shape == (1, 4)
    assert multi.left.tip.shape == (1, 2)
    assert multi.left.tip.index[-1].tolist() == [[1, 3]]
    assert len(multi.injections) == 1 and multi.injections[0][0] is waveform
    assert multi.injections[0][2][-1].tolist() == [0, 2]

    multi.build()
    assert multi.mech.pas.g.flatten().tolist() == pytest.approx(
        [0.1, 0.1, 0.1, 0.1, 0.2]
    )


def test_multi_v_init_overrides_validate_names_shapes_and_preserve_omissions():
    left = _population(n=2, c=2, v_init=[-60.0, -61.0])
    right = _population(n=1, c=2, v_init=[-62.0, -63.0])
    multi = dn.concat_models(
        {"left": left, "right": right}, v_init={"left": [-50.0, -51.0]}
    )
    assert multi.v_init.tolist() == [-50.0, -51.0, -50.0, -51.0, -62.0, -63.0]
    assert multi.set_v_init({"right": -40.0}) is multi
    assert multi.v_init.tolist() == [-60.0, -61.0, -60.0, -61.0, -40.0, -40.0]

    with pytest.raises(KeyError, match="Unknown population"):
        dn.concat_models({"left": left, "right": right}, v_init={"missing": -1.0})
    with pytest.raises(ValueError, match="target_shape"):
        _expand_v_init_like(-65.0, (3,), device="cpu", dtype=DTYPE)
    with pytest.raises(ValueError, match="has length"):
        _expand_v_init_like([1.0, 2.0, 3.0], (2, 2), device="cpu", dtype=DTYPE)
    with pytest.raises(ValueError, match="Unsupported"):
        _expand_v_init_like(torch.zeros(2, 1, 2), (2, 2), device="cpu", dtype=DTYPE)


def test_multi_batch_updates_component_and_nested_composite_labels():
    left = _population(n=2, c=2)
    right = _population(n=1, c=1)
    left[:, 1].label("tip")
    multi = dn.concat_models({"left": left, "right": right})

    assert multi.batch(3) is multi
    assert multi.shape == (3, 1, 5)
    assert left.shape == (3, 2, 2) and right.shape == (3, 1, 1)
    assert multi.left.shape == (3, 1, 4)
    assert multi.left.tip.shape == (3, 1, 2)
    assert torch.equal(multi.left.tip.v, multi.v[..., [1, 3]])


def test_multi_batch_resynchronizes_component_label_lifecycle():
    left = _population(n=1, c=3)
    right = _population(n=1, c=1)
    left[:, 0].label("old")
    multi = dn.concat_models({"left": left, "right": right})
    multi[:, 0].label("composite_only")

    left.clear_labels()
    left[:, 2].label("new")
    multi.batch(2)

    assert "old" not in vars(multi.left)
    assert multi.left.new.shape == (2, 1, 1)
    assert torch.equal(multi.left.new.v, multi.v[..., [2]])
    assert multi.composite_only.shape == (2, 1)


def test_multi_validation_rejects_empty_batched_mixed_and_heterogeneous_inputs():
    with pytest.raises(ValueError, match="At least one"):
        MultiPopulation()
    with pytest.raises(ValueError, match="At least one"):
        dn.concat_models({})
    with pytest.raises(TypeError, match="celsius"):
        _check_celsius("37", {"p": _population()})

    batched = _population().batch(2)
    with pytest.raises(ValueError, match="unbatched"):
        MultiPopulation(p=batched)

    double = _population()
    single = dn.Population(N=1, C=1, dtype=torch.float32)
    with pytest.raises(ValueError, match="same dtype"):
        MultiPopulation(double=double, single=single)

    tree = dn.Tree.from_graph(_morphology_graph(), dtype=DTYPE)
    with pytest.raises(TypeError, match="Incompatible"):
        _assess_type_and_make_integrator({"plain": double, "tree": tree})
    assert callable(_assess_type_and_make_integrator({"plain": double}))
    assert callable(_assess_type_and_make_integrator({"tree": tree}))


def test_gather_morphology_honors_explicit_domains_and_cylinder_fallback():
    graph = _morphology_graph()
    graph.nodes[0].update(volume=12.0, volume_i=8.0, volume_o=4.0)
    graph.nodes[1].update(volume_um3=7.0)
    morphology = gather_morphology(graph)

    assert morphology["volume"].tolist()[0][:2] == pytest.approx([12.0, 7.0])
    assert morphology["volume_i"].tolist()[0][:2] == pytest.approx([8.0, 7.0])
    assert morphology["volume_o"].tolist()[0][:2] == pytest.approx([4.0, 0.0])
    assert morphology["volume"][0, 2].item() == pytest.approx(2.0 * math.pi)
    assert torch.equal(morphology["volume"], morphology["volume_um3"])

    membrane = gather_membrane(graph)
    assert membrane["rhoa"].tolist() == [[100.0, 100.0, 100.0]]
    assert membrane["cm"].tolist() == [[1.0, 1.0, 1.0]]


def test_diffusion_geometry_uses_explicit_resistance_and_stylized_fallbacks():
    explicit = gather_diffusion_edges(_morphology_graph())
    assert explicit["diff_parent_index"].tolist() == [-1, 0, 0]
    assert explicit["diff_geom_um"].tolist() == [[0.0, 1.5, 0.75]]
    assert explicit["diff_edge_parent"].tolist() == [0, 0]
    assert explicit["diff_edge_child"].tolist() == [1, 2]

    resistance = _morphology_graph()
    resistance.remove_edge(0, 2)
    resistance.edges[0, 1].clear()
    resistance.edges[0, 1]["R_ohm"] = 500_000.0
    resistance.nodes[0]["Ra"] = 100.0
    resistance.nodes[1]["Ra"] = 200.0
    assert gather_diffusion_edges(resistance)["diff_geom_um"][
        0, 1
    ].item() == pytest.approx(3.0)

    stylized = _morphology_graph()
    stylized.remove_edge(0, 2)
    stylized.edges[0, 1].clear()
    stylized.edges[0, 1]["L"] = 10.0
    assert gather_diffusion_edges(stylized)["diff_geom_um"][
        0, 1
    ].item() == pytest.approx(math.pi / 10.0)


def test_diffusion_geometry_rejects_ambiguous_parent_and_missing_resistivity():
    graph = _morphology_graph()
    graph.add_edge(2, 1, diff_geom_um=1.0)
    with pytest.raises(ValueError, match="2 parents"):
        gather_diffusion_edges(graph)

    graph = _morphology_graph()
    graph.remove_edge(0, 2)
    graph.edges[0, 1].clear()
    graph.edges[0, 1]["R_ohm"] = 10.0
    graph.nodes[0]["Ra"] = None
    graph.nodes[1]["Ra"] = None
    with pytest.raises(KeyError, match="neither endpoint"):
        gather_diffusion_edges(graph)


@pytest.mark.parametrize(
    "domain,attribute",
    [
        ("i", "volume_i"),
        ("cytosol", "volume_i"),
        ("extracellular", "volume_o"),
        ("total", "volume"),
        ("surface", "area"),
    ],
)
def test_tree_material_volume_domain_aliases(domain, attribute):
    tree = dn.Tree.from_graph(_morphology_graph(), N=2, dtype=DTYPE)
    assert torch.equal(tree.material_volume(domain), getattr(tree, attribute))
    with pytest.raises(ValueError, match="Unsupported material domain"):
        tree.material_volume("nucleus")


def test_tree_recentre_shift_and_move_support_per_cell_coordinates():
    tree = dn.Tree.from_graph(_morphology_graph(), N=2, dtype=DTYPE)
    tree.shift(dx=torch.tensor([1.0, 2.0]), dy=3.0, dz=torch.tensor([-1.0, -2.0]))
    tree.recentre(
        x=torch.tensor([10.0, 20.0]),
        y=4.0,
        z=torch.tensor([5.0, 6.0]),
        origin=0,
    )
    assert tree.x[:, 0].tolist() == [10.0, 20.0]
    assert tree.y[:, 0].tolist() == [4.0, 4.0]
    assert tree.z[:, 0].tolist() == [5.0, 6.0]
    assert tree.move_to(x=1.0, y=2.0, z=3.0, origin=0) is tree
    assert tree.x[:, 0].tolist() == [1.0, 1.0]

    with pytest.raises(AssertionError, match="Expected dx.ndim"):
        tree.shift(dx=torch.zeros(1, 1))


def test_tree_rotation_aligns_identity_antiparallel_and_general_float64_cases():
    tree = dn.Tree.from_graph(_morphology_graph(), N=3, dtype=DTYPE)
    targets = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=DTYPE
    )
    targets = torch.nn.functional.normalize(targets, dim=1)
    tree.rotate_into_direction(targets, origin=0)

    displacement = torch.stack(
        [
            tree.x[:, 1] - tree.x[:, 0],
            tree.y[:, 1] - tree.y[:, 0],
            tree.z[:, 1] - tree.z[:, 0],
        ],
        dim=1,
    )
    assert tree.directions.dtype == DTYPE
    assert torch.allclose(tree.directions, targets, atol=1e-12, rtol=1e-12)
    assert torch.allclose(displacement, 2.0 * targets, atol=1e-12, rtol=1e-12)

    with pytest.raises(ValueError, match="one direction per cell"):
        tree.rotate_into_direction(torch.ones(2, 3), origin=0)
    with pytest.raises(ValueError, match="non-zero"):
        tree.rotate_into_direction(torch.zeros(3), origin=0)


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 2e-6), (torch.float64, 1e-12)])
def test_tree_rotation_preserves_small_nonzero_angles(dtype, atol):
    tree = dn.Tree.from_graph(_morphology_graph(), N=1, dtype=dtype)
    angle = torch.deg2rad(torch.tensor(0.01, dtype=dtype))
    target = torch.stack((torch.sin(angle), angle.new_zeros(()), torch.cos(angle)))
    tree.rotate_into_direction(target, origin=0)

    displacement = torch.stack(
        (
            tree.x[0, 1] - tree.x[0, 0],
            tree.y[0, 1] - tree.y[0, 0],
            tree.z[0, 1] - tree.z[0, 0],
        )
    )
    assert torch.allclose(displacement, 2.0 * target, atol=atol, rtol=atol)
    assert torch.allclose(tree.directions[0], target, atol=atol, rtol=atol)


def test_tree_antiparallel_rotation_handles_axis_parallel_to_fallback_vector():
    tree = dn.Tree.from_graph(
        _morphology_graph(axis="x"),
        N=2,
        principal_axis=[1.0, 0.0, 0.0],
        dtype=DTYPE,
    )
    tree.rotate_into_direction([-1.0, 0.0, 0.0], origin=0)
    displacement = torch.stack(
        [
            tree.x[:, 1] - tree.x[:, 0],
            tree.y[:, 1] - tree.y[:, 0],
            tree.z[:, 1] - tree.z[:, 0],
        ],
        dim=1,
    )
    assert torch.allclose(
        displacement,
        torch.tensor([[-2.0, 0.0, 0.0]], dtype=DTYPE).expand(2, -1),
    )


def test_tree_azimuthal_rotation_broadcasts_scalars_and_validates_batch_angles():
    tree = dn.Tree.from_graph(_morphology_graph(), N=2, dtype=DTYPE)
    tree.rotate_azimuthal(torch.tensor([90.0, -90.0], dtype=DTYPE), origin=0)
    transverse = torch.stack(
        [
            tree.x[:, 2] - tree.x[:, 0],
            tree.y[:, 2] - tree.y[:, 0],
            tree.z[:, 2] - tree.z[:, 0],
        ],
        dim=1,
    )
    expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=DTYPE)
    assert torch.allclose(transverse, expected, atol=1e-12, rtol=1e-12)
    assert tree.azimuthal_rotations.tolist() == pytest.approx([90.0, -90.0])

    tree.reset_azimuthal_rotations(origin=0)
    assert torch.allclose(tree.azimuthal_rotations, torch.zeros(2, dtype=DTYPE))
    tree.rotate_azimuthal(torch.tensor(30.0), origin=0)
    assert tree.azimuthal_rotations.tolist() == pytest.approx([30.0, 30.0])
    with pytest.raises(ValueError, match="one angle per cell"):
        tree.rotate_azimuthal(torch.tensor([1.0, 2.0, 3.0]), origin=0)


def test_tree_reset_rotations_restores_shape_anchored_at_current_nonzero_origin():
    graph = _morphology_graph(origin_x=10.0)
    tree = dn.Tree.from_graph(graph, N=1, dtype=DTYPE)
    base_relative = tree._get_points_as_tensor() - tree._get_points_as_tensor()[:, :1]
    tree.shift(dx=5.0, dy=2.0, dz=-3.0)
    tree.rotate_into_direction([1.0, 0.0, 1.0], origin=0)
    tree.rotate_azimuthal(47.0, origin=0)
    current_origin = tree._get_points_as_tensor()[:, 0].clone()

    assert tree.reset_rotations(origin=0) is tree
    points = tree._get_points_as_tensor()
    assert torch.allclose(points[:, 0], current_origin)
    assert torch.allclose(points - points[:, :1], base_relative)
    assert torch.equal(tree.directions, tree.base_direction)
    assert torch.equal(tree.azimuthal_rotations, tree.base_azimuthal_rotation)

    tree.rotate_into_direction([1.0, 0.0, 0.0], origin=0)
    assert tree.reset_directions(origin=0) is tree


def test_reset_direction_preserves_azimuth_so_it_can_be_reset_separately():
    tree = dn.Tree.from_graph(_morphology_graph(), N=1, dtype=DTYPE)
    base_relative = tree._get_points_as_tensor() - tree._get_points_as_tensor()[:, :1]
    tree.rotate_azimuthal(45.0, origin=0)
    tree.rotate_into_direction([1.0, 0.0, 0.0], origin=0)

    tree.reset_directions(origin=0)
    assert tree.azimuthal_rotations.item() == pytest.approx(45.0)
    tree.reset_azimuthal_rotations(origin=0)
    points = tree._get_points_as_tensor()
    assert torch.allclose(points - points[:, :1], base_relative, atol=1e-12, rtol=1e-12)
