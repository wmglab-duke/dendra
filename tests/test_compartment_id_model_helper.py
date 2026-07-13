from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dendra.models.core import Unmyelinated
from dendra.models.heterogeneous import CompartmentID
from dendra.models.heterogeneous.compartments import expand_string_list


def test_compartment_id_expands_repeats_wraps_and_supports_lookup():
    cid = CompartmentID(["node", "internode * 2"], n_repeats=2)

    assert cid.names.tolist() == [
        "node",
        "internode",
        "internode",
        "node",
        "internode",
        "internode",
        "node",
    ]
    assert cid.nc() == len(cid) == 7
    assert list(cid) == cid.names.tolist()
    assert cid[3] == "node"
    assert cid.unique() == ["internode", "node"]
    assert cid.loc("node") == [0, 3, 6]
    assert cid.locs(["node", "missing"]) == [0, 3, 6]

    unwrapped = CompartmentID(["node", "internode * 2"], n_repeats=2, wrap=False)
    assert unwrapped.names.tolist() == cid.names[:-1].tolist()

    expanded_boundary = CompartmentID(["node * 2", "internode"], n_repeats=1)
    assert expanded_boundary.names.tolist() == [
        "node",
        "node",
        "internode",
        "node",
    ]


def test_expand_string_list_preserves_order_and_non_patterns():
    assert expand_string_list(["soma", "myelin * 3", "node*0", "terminal"]) == [
        "soma",
        "myelin",
        "myelin",
        "myelin",
        "terminal",
    ]


def test_compartment_id_build_broadcasts_per_axon_values_by_label():
    cid = CompartmentID(["node", "internode * 2"], n_repeats=2)
    model = SimpleNamespace(n_ax=2, n_comp=7)

    values = cid.build(
        {
            "node": lambda _model: np.array([1.0, 2.0]),
            "internode": lambda _model: np.array([10.0, 20.0]),
        },
        model,
    )

    np.testing.assert_array_equal(
        values,
        np.array(
            [
                [1.0, 10.0, 10.0, 1.0, 10.0, 10.0, 1.0],
                [2.0, 20.0, 20.0, 2.0, 20.0, 20.0, 2.0],
            ]
        ),
    )


def test_axon_register_cid_installs_searchable_labelled_slices():
    axon = Unmyelinated(diameters=[8.0, 12.0], L=60.0, dx=10.0)
    cid = CompartmentID(["node", "internode * 2"], n_repeats=2)

    result = axon.register_cid(cid)

    assert result is None
    assert axon.cid is cid
    assert axon.names == cid.names.tolist()
    assert set(axon._labels) == {"node", "internode"}
    assert axon.node is axon._labels["node"]
    assert axon.internode is axon._labels["internode"]
    assert axon.node.shape == (2, 3)
    assert axon.internode.shape == (2, 4)
    assert axon.find("node", exclude=None, as_list=True) == [0, 3, 6]


def test_axon_register_cid_rejects_invalid_tables_without_partial_mutation():
    axon = Unmyelinated(diameters=[8.0], L=60.0, dx=10.0)
    original_v = axon.v
    original_x = axon.x

    with pytest.raises(ValueError, match="7 compartment names"):
        axon.register_cid(CompartmentID(["short"], n_repeats=2, wrap=False))

    assert axon.cid is None
    assert not hasattr(axon, "names")
    assert axon._labels == {}
    assert axon.v is original_v
    assert axon.x is original_x


@pytest.mark.parametrize(
    "names",
    ["abc", np.array("abc"), np.array([["a", "b", "c"]])],
)
def test_axon_register_cid_rejects_scalar_or_nonvector_name_tables(names):
    axon = Unmyelinated(diameters=[8.0], L=20.0, dx=10.0)

    with pytest.raises(TypeError, match="one-dimensional"):
        axon.register_cid(SimpleNamespace(names=names))

    assert axon.cid is None
    assert axon._labels == {}


@pytest.mark.parametrize("reserved", ["x", "v", "batch", "cid", "names"])
def test_axon_register_cid_rejects_reserved_label_collisions_atomically(reserved):
    axon = Unmyelinated(diameters=[8.0], L=60.0, dx=10.0)
    cid = CompartmentID([reserved, "safe * 6"], n_repeats=1, wrap=False)
    buffers_before = {name: value for name, value in axon.named_buffers()}

    with pytest.raises(ValueError, match="conflicts with an existing attribute"):
        axon.register_cid(cid)

    assert axon.cid is None
    assert not hasattr(axon, "names")
    assert axon._labels == {}
    assert all(getattr(axon, name) is value for name, value in buffers_before.items())


def test_axon_register_cid_replaces_only_its_previous_labels():
    axon = Unmyelinated(diameters=[8.0], L=60.0, dx=10.0)
    axon[..., 0].label("manual")
    first = CompartmentID(["node", "internode * 2"], n_repeats=2)
    second = CompartmentID(["proximal * 2", "distal"], n_repeats=2)

    axon.register_cid(first)
    old_node = axon.node
    axon.register_cid(second)

    assert axon.cid is second
    assert set(axon._labels) == {"manual", "proximal", "distal"}
    assert axon.manual.shape == (1,)
    assert not hasattr(axon, "node")
    assert not hasattr(axon, "internode")
    assert axon.proximal.shape == (1, 5)
    assert axon.distal.shape == (1, 2)
    assert old_node is not axon.proximal
    assert torch.equal(axon.proximal.v, axon.v[..., [0, 1, 3, 4, 6]])


def test_axon_register_cid_recovers_after_public_label_clear():
    axon = Unmyelinated(diameters=[8.0], L=60.0, dx=10.0)
    first = CompartmentID(["node", "internode * 2"], n_repeats=2)
    second = CompartmentID(["proximal * 2", "distal"], n_repeats=2)

    axon.register_cid(first)
    axon.clear_labels()
    axon[..., 0].label("node")
    axon.register_cid(second)

    assert axon.cid is second
    assert set(axon._labels) == {"node", "proximal", "distal"}
    assert axon.node.shape == (1,)
    assert not hasattr(axon, "internode")
    assert torch.equal(axon.proximal.v, axon.v[..., [0, 1, 3, 4, 6]])
