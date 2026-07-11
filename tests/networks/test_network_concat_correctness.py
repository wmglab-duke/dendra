"""Correctness contracts for partial network population concatenation."""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import expsyn, graded_syn, sigmoid_release

DT = 0.1
DTYPE = torch.float64


def _populations_with_event_synapses():
    populations = {}
    for name in ("a", "b", "c"):
        population = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
        population.insert(expsyn.rename(f"concat_syn_{name}"), e=0.0, tau=1.0)
        populations[name] = population
    return populations


def _event_spec_edges(network):
    result = {}
    for (source, target, synapse, _pre_var), specs in network.synapse_spec.items():
        result[(source, target, synapse.name)] = (
            torch.cat([spec[0] for spec in specs]),
            torch.cat([spec[1] for spec in specs]),
        )
    return result


def test_partial_concat_preserves_event_endpoints_and_synapse_local_indices():
    populations = _populations_with_event_synapses()
    network = dn.Network(populations)
    connections = (
        ("a", "b", 1.0),
        ("b", "c", 2.0),
        ("c", "b", 3.0),
        ("c", "c", 4.0),
    )
    for source, target, weight in connections:
        network.connect_one_to_one(
            populations[source][:],
            populations[target][:],
            getattr(populations[target].mech, f"concat_syn_{target}"),
            threshold=-30.0,
            weight=weight,
            delay=DT,
            allow_autapses=True,
        )

    concatenated = network.concat(ab=["a", "b"])

    assert set(concatenated.populations) == {"ab", "c"}
    edges = _event_spec_edges(concatenated)
    assert set(edges) == {
        ("ab", "ab", "concat_syn_b"),
        ("ab", "c", "concat_syn_c"),
        ("c", "ab", "concat_syn_b"),
        ("c", "c", "concat_syn_c"),
    }
    assert edges[("ab", "ab", "concat_syn_b")][0].tolist() == [0, 1]
    assert edges[("ab", "c", "concat_syn_c")][0].tolist() == [2, 3]
    for _, post_idx in edges.values():
        # Target ids remain local to the original target synapse after its
        # mechanism is reinserted into the concatenated population.
        assert post_idx.tolist() == [0, 1]

    concatenated.build(DT)
    concatenated.init_synapses()
    assert len(concatenated.synapses) == 4
    for netcon in concatenated.synapses.values():
        assert int(netcon.pre_idx.max()) < netcon.pre.v.numel()
        assert int(netcon.post_idx.max()) < netcon._syn_numel


def test_partial_concat_preserves_netstim_sources_for_mixed_targets():
    populations = _populations_with_event_synapses()
    network = dn.Network(populations, netstim=dn.NetStim(N=2, dtype=DTYPE))
    for target in ("b", "c"):
        network.connect_one_to_one(
            network.netstim[:],
            populations[target][:],
            getattr(populations[target].mech, f"concat_syn_{target}"),
            threshold=None,
            weight=1.0,
            delay=DT,
        )

    concatenated = network.concat(ab=["a", "b"])

    edges = _event_spec_edges(concatenated)
    assert set(edges) == {
        ("netstim", "ab", "concat_syn_b"),
        ("netstim", "c", "concat_syn_c"),
    }
    assert all(pre_idx.tolist() == [0, 1] for pre_idx, _ in edges.values())
    concatenated.build(DT)
    concatenated.init_synapses()
    assert all(
        connection.pre is concatenated.netstim
        for connection in concatenated.synapses.values()
    )


def _populations_with_continuous_synapses():
    populations = {}
    for name in ("a", "b", "c"):
        population = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
        population.insert(
            graded_syn.rename(f"concat_graded_{name}"),
            e=-10.0,
            g_scale=0.02,
        )
        populations[name] = population
    return populations


def test_partial_concat_reapplies_continuous_connections_and_metadata():
    populations = _populations_with_continuous_synapses()
    network = dn.Network(populations)
    transform = sigmoid_release(theta=-30.0, sigma=3.0)
    connections = (("a", "b"), ("b", "c"), ("c", "b"), ("c", "c"))
    for source, target in connections:
        network.connect_continuous_one_to_one(
            populations[source][:],
            populations[target][:],
            getattr(populations[target].mech, f"concat_graded_{target}"),
            pre_var="v",
            input="g_pre",
            weight=torch.tensor([0.4, 0.9], dtype=DTYPE),
            delay=torch.tensor([0.1, 0.2], dtype=DTYPE),
            transform=transform,
            allow_autapses=True,
        )

    concatenated = network.concat(ab=["a", "b"])

    keys = {
        (source, target, synapse.name, input_name, reduce, transform_obj)
        for (
            source,
            target,
            synapse,
            _pre_var,
            input_name,
            reduce,
            transform_obj,
        ) in concatenated.continuous_synapse_spec
    }
    assert {(source, target, synapse) for source, target, synapse, *_ in keys} == {
        ("ab", "ab", "concat_graded_b"),
        ("ab", "c", "concat_graded_c"),
        ("c", "ab", "concat_graded_b"),
        ("c", "c", "concat_graded_c"),
    }
    assert all(input_name == "g_pre" for *_, input_name, _reduce, _transform in keys)
    assert all(reduce == "sum" for *_, reduce, _transform in keys)
    assert all(transform_obj is transform for *_, transform_obj in keys)

    concatenated.build(DT)
    concatenated.init_synapses()
    assert len(concatenated.continuous_synapses) == 4
    for connection in concatenated.continuous_synapses.values():
        assert int(connection.pre_idx.max()) < connection.pre.v.numel()
        assert int(connection.post_idx.max()) < connection._syn_numel
