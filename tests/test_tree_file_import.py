"""File constructors select native grids; NEURON capture retains authored grids."""

from __future__ import annotations

import builtins
import gc
from pathlib import Path

import pytest
import torch

import dendra as dn


@pytest.fixture(autouse=True)
def _native_environment():
    with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=torch.float64):
        yield


def _swc(tmp_path):
    path = tmp_path / "branched.swc"
    path.write_text(
        "1 1 0 0 0 5 -1\n"
        "2 3 0 20 0 1 1\n"
        "3 3 0 500 0 1 2\n"
        "4 4 100 0 0 2 1\n"
        "5 4 500 0 0 2 4\n"
        "6 2 0 0 300 0.5 1\n"
    )
    return path


@pytest.mark.parametrize("overrides", ({}, {"rhoa": 87.0, "cm": 1.3}))
def test_swc_tree_uses_native_grid_and_keeps_distinct_anatomical_regions(
    tmp_path, monkeypatch, overrides
):
    path = _swc(tmp_path)
    morphology = dn.Morphology.from_swc(
        path,
        rhoa=overrides.get("rhoa", 35.4),
        cm=overrides.get("cm", 1.0),
        type_labels={3: "dend", 4: "apic"},
    )
    morphology.apply_d_lambda(d_lambda=0.08, freq_hz=173.0)
    original_import = builtins.__import__

    def without_neuron(name, *args, **kwargs):
        if name == "neuron" or name.startswith("neuron."):
            raise AssertionError("SWC Tree construction must use the native parser")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_neuron)
    model = dn.Tree.from_swc(
        path,
        d_lambda=0.08,
        freq=173.0,
        N=2,
        principal_axis=(0.0, 1.0, 0.0),
        **overrides,
    )

    assert model.compartment_graph == morphology.compile()
    assert model.shape[0] == 2
    torch.testing.assert_close(
        model.directions,
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64).expand(2, -1),
    )
    torch.testing.assert_close(model.rhoa, torch.full_like(model.rhoa, morphology.rhoa))
    torch.testing.assert_close(model.cm, torch.full_like(model.cm, morphology.cm))
    for label, prefix in (
        ("soma", "soma_"),
        ("dend", "basal_dendrite_"),
        ("apic", "apical_dendrite_"),
        ("axon", "axon_"),
    ):
        indices = getattr(model, label).index[-1].tolist()
        assert indices
        assert all(
            model.compartment_graph.metadata.section_name[index].startswith(prefix)
            for index in indices
        )
    assert set(model.dend.index[-1].tolist()).isdisjoint(model.apic.index[-1].tolist())


@pytest.mark.parametrize("kwargs", ({"freq": 0.0}, {"d_lambda": 0.0}))
def test_file_tree_reuses_native_discretization_validation(tmp_path, kwargs):
    with pytest.raises(ValueError, match="must be positive"):
        dn.Tree.from_swc(_swc(tmp_path), **kwargs)


def test_swc_segment_callbacks_have_an_explicit_capture_path(tmp_path):
    with pytest.raises(TypeError, match="io.read_swc and Tree.from_graph"):
        dn.Tree.from_swc(_swc(tmp_path), data_func=lambda segment: {})


@pytest.mark.parametrize(
    "constructor",
    (
        "from_swc",
        pytest.param("from_asc", marks=pytest.mark.neuron),
        pytest.param("from_neurolucida", marks=pytest.mark.neuron),
    ),
)
def test_file_tree_preserves_custom_from_graph_delegation(tmp_path, constructor):
    calls = []

    class CustomTree(dn.Tree):
        @classmethod
        def from_graph(cls, graph, **kwargs):
            calls.append((graph, kwargs))
            model = super().from_graph(graph, **kwargs)
            model[:, :1].label("custom_region")
            return model

    path = (
        _swc(tmp_path)
        if constructor == "from_swc"
        else Path(__file__).parent / "data/morphologies/neurolucida_mixed.asc"
    )
    integrator = dn.dhs
    options = {
        "N": 2,
        "integrator": integrator,
        "principal_axis": (0.0, 1.0, 0.0),
        "rhoa": 87.0,
        "cm": 1.3,
    }
    model = getattr(CustomTree, constructor)(path, d_lambda=0.08, freq=173.0, **options)
    reference = getattr(dn.Tree, constructor)(
        path, d_lambda=0.08, freq=173.0, **options
    )

    assert isinstance(model, CustomTree)
    assert len(calls) == 1
    graph, forwarded = calls[0]
    assert forwarded == options
    assert model.compartment_graph == reference.compartment_graph
    assert len(graph) == model.nc
    torch.testing.assert_close(model.v, reference.v)
    torch.testing.assert_close(model.directions, reference.directions)
    nodes = torch.arange(model.nc)
    torch.testing.assert_close(nodes[model.custom_region.index[-1]], nodes[:1])
    for label in ("soma", "dend", "apic", "axon"):
        torch.testing.assert_close(
            nodes[getattr(model, label).index[-1]],
            nodes[getattr(reference, label).index[-1]],
        )


@pytest.mark.neuron
@pytest.mark.parametrize("constructor", ("from_asc", "from_neurolucida"))
def test_asc_tree_uses_native_geometry_and_releases_parser_sections(constructor):
    from neuron import h

    path = Path(__file__).parent / "data/morphologies/neurolucida_mixed.asc"
    before = tuple(sorted(str(section.name()) for section in h.allsec()))
    morphology = dn.Morphology.from_asc(path, rhoa=87.0, cm=1.3)
    morphology.apply_d_lambda(d_lambda=0.08, freq_hz=173.0)

    model = getattr(dn.Tree, constructor)(
        path, d_lambda=0.08, freq=173.0, N=2, rhoa=87.0, cm=1.3
    )

    assert model.compartment_graph == morphology.compile()
    assert model.shape[0] == 2
    gc.collect()
    assert tuple(sorted(str(section.name()) for section in h.allsec())) == before


@pytest.mark.neuron
def test_asc_tree_requires_selection_of_disconnected_components():
    path = Path(__file__).parent / "data/morphologies/neurolucida_two_roots.asc"
    with pytest.raises(ValueError, match="disconnected roots"):
        dn.Tree.from_asc(path)

    model = dn.Tree.from_asc(path, root="dend[0]")

    assert set(model.compartment_graph.metadata.section_name) == {"dend[0]"}


@pytest.mark.neuron
def test_neuron_capture_retains_the_users_compartment_counts():
    from neuron import h

    root = h.Section(name="user_grid_root")
    child = h.Section(name="user_grid_child")
    try:
        root.L, root.diam, root.nseg = 100.0, 5.0, 7
        child.L, child.diam, child.nseg = 40.0, 2.0, 3
        child.connect(root(1.0), 0.0)

        model = dn.Tree.from_NEURON(root)

        assert int(root.nseg) == 7 and int(child.nseg) == 3
        assert (
            sum(kind == "compartment" for kind in model.compartment_graph.metadata.kind)
            == 10
        )
    finally:
        h.delete_section(sec=child)
        h.delete_section(sec=root)
