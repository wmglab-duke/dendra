"""Correctness contracts for partial network population concatenation."""

from __future__ import annotations

import math

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn, graded_syn, pas, sigmoid_release, spikedetect

DT = 0.1
DTYPE = torch.float64


def _branched_tree_graph():
    graph = nx.DiGraph()
    for node in range(3):
        graph.add_node(
            node,
            name=f"section[{node}](0.5)",
            L=10.0,
            diam=2.0,
            Ra=100.0,
            cm=1.0,
            area=2.0e8,
        )
    graph.add_edge(0, 1, R_ohm=75.0)
    graph.add_edge(0, 2, R_ohm=125.0)
    return graph


def _connection_pairs(specs):
    return {
        (int(source), int(target))
        for spec in specs
        for source, target in zip(spec[0], spec[1])
    }


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


def test_mixed_scalar_concat_batch_preserves_shared_event_synapse_slots():
    populations = {
        "single": dn.SingleCompartment(N=1, C=1, dtype=DTYPE),
        "tree": dn.Tree.from_graph(_branched_tree_graph(), N=1, dtype=DTYPE),
        "cable": dn.Unmyelinated(diameters=[1.5], L=30.0, dx=10.0, dtype=DTYPE),
    }
    for population in populations.values():
        population.insert(expsyn, e=0.0, tau=1.0)

    network = dn.Network(populations)
    edges = (("single", "tree"), ("tree", "cable"), ("cable", "single"))
    for source, target in edges:
        network.connect_one_to_one(
            populations[source][0, 0],
            populations[target][0, 0],
            populations[target].mech.expsyn,
            threshold=-30.0,
            weight=1.0,
            delay=DT,
        )

    component_sizes = {
        name: math.prod(population.shape) for name, population in populations.items()
    }
    component_offsets = {}
    offset = 0
    for name, size in component_sizes.items():
        component_offsets[name] = offset
        offset += size

    concatenated = network.concat().batch(2)
    packed = concatenated.all_populations

    assert [type(population) for population in packed.populations.values()] == [
        dn.SingleCompartment,
        dn.Tree,
        dn.Unmyelinated,
    ]
    assert packed._integrator_class.__name__ == "_dhs_multi"
    assert packed.shape == (2, 1, offset)

    specs = next(iter(concatenated.synapse_spec.values()))
    expected_pairs = {
        (
            component_offsets[source] + batch * offset,
            component_offsets[target] + batch * offset,
        )
        for source, target in edges
        for batch in range(2)
    }
    assert _connection_pairs(specs) == expected_pairs

    concatenated.initialize(DT)
    netcon = next(iter(concatenated.synapses.values()))
    assert set(zip(netcon.pre_idx.tolist(), netcon.post_idx.tolist())) == expected_pairs
    assert netcon._syn_numel == 2 * offset
    concatenated.run(2 * DT)
    assert concatenated.t.item() == pytest.approx(2 * DT)
    assert torch.isfinite(packed.v).all()


def test_concat_batch_remaps_shared_continuous_synapse_slots_and_values():
    populations = {name: dn.Population(N=1, C=2, dtype=DTYPE) for name in ("a", "b")}
    for population in populations.values():
        population.insert(graded_syn, e=-10.0, g_scale=0.02)

    network = dn.Network(populations)
    for index, population in enumerate(populations.values()):
        network.connect_continuous_one_to_one(
            population[:],
            population[:],
            population.mech.graded_syn,
            pre_var="v",
            input="g_pre",
            weight=torch.tensor([1.0, 2.0], dtype=DTYPE) + 2 * index,
            delay=0.0,
            allow_autapses=True,
        )

    concatenated = network.concat().batch(2)
    specs = next(iter(concatenated.continuous_synapse_spec.values()))
    assert _connection_pairs(specs) == {(index, index) for index in range(8)}

    concatenated.build(DT)
    concatenated.init_synapses()
    connection = next(iter(concatenated.continuous_synapses.values()))
    assert connection.post_idx.tolist() == connection.pre_idx.tolist()
    assert connection.weight.w.tolist() == pytest.approx(
        [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
    )


def test_concat_maps_union_slots_into_record_ordered_copied_slots():
    copied = dn.Population(N=1, C=3, dtype=DTYPE)
    reordered = dn.Population(N=1, C=3, dtype=DTYPE)
    copied[:, 1].insert(expsyn, copies=2, e=0.0, tau=1.0)
    # Standalone union compilation sorts these placements to physical [0, 2].
    # The copied placement above makes the packed mechanism record ordered,
    # where this component instead appears as physical [2, 0].
    reordered[:, 2].insert(expsyn, e=0.0, tau=1.0)
    reordered[:, 0].insert(expsyn, e=0.0, tau=1.0)

    network = dn.Network(
        {"copied": copied, "reordered": reordered},
        netstim=dn.NetStim(N=2, dtype=DTYPE),
    )
    network.connect_one_to_one_slots(
        network.netstim[:],
        reordered.slots(reordered.mech.expsyn),
        threshold=None,
        weight=torch.tensor([10.0, 20.0], dtype=DTYPE),
        delay=torch.tensor([0.1, 0.2], dtype=DTYPE),
    )

    concatenated = network.concat().batch(2)
    specs = next(iter(concatenated.synapse_spec.values()))
    assert specs[0][1].tolist() == [3, 2, 7, 6]

    concatenated.build(DT)
    concatenated.init_synapses()
    netcon = next(iter(concatenated.synapses.values()))
    assert netcon.post_idx.tolist() == [3, 2, 7, 6]
    assert netcon.weight.w.tolist() == pytest.approx([10.0, 20.0, 10.0, 20.0])


def test_concat_batch_respects_reversed_component_order_and_unequal_sizes():
    populations = {
        "a": dn.Population(N=1, C=2, dtype=DTYPE),
        "b": dn.Population(N=1, C=3, dtype=DTYPE),
        "spare": dn.Population(N=1, C=1, dtype=DTYPE),
    }
    for population in populations.values():
        population.insert(expsyn, e=0.0, tau=1.0)

    network = dn.Network(populations)
    network.connect_one_to_one(
        populations["a"][:, 1],
        populations["b"][:, 2],
        populations["b"].mech.expsyn,
        threshold=-30.0,
        weight=1.0,
        delay=DT,
    )
    network.connect_one_to_one(
        populations["b"][:, 0],
        populations["a"][:, 0],
        populations["a"].mech.expsyn,
        threshold=-30.0,
        weight=2.0,
        delay=DT,
    )

    transformed = network.concat(packed=["b", "a"]).batch(2)

    assert list(transformed.populations) == ["packed", "spare"]
    assert list(transformed.packed.populations) == ["b", "a"]
    specs = next(iter(transformed.synapse_spec.values()))
    assert specs[0][0].tolist() == [4, 9]
    assert specs[0][1].tolist() == [2, 7]
    assert specs[1][0].tolist() == [0, 5]
    assert specs[1][1].tolist() == [3, 8]
    transformed.initialize(DT)


def test_concat_batch_keeps_mechanism_local_source_variables_addressable():
    pre = dn.Population(N=1, C=3, v_init=-65.0, dtype=DTYPE)
    post = dn.Population(N=1, C=3, v_init=-65.0, dtype=DTYPE)
    pre[:, 1].insert(spikedetect, threshold=-30.0)
    post[:, 1].insert(expsyn, e=0.0, tau=1.0)
    network = dn.Network({"pre": pre, "post": post})
    network.connect_one_to_one(
        pre[:, 1],
        post[:, 1],
        post.mech.expsyn,
        threshold=None,
        weight=0.1,
        delay=DT,
        pre_var="mech.spikedetect.spikes",
    )

    transformed = network.concat().batch(3).initialize(DT)
    netcon = next(iter(transformed.synapses.values()))

    assert netcon.pre_idx.tolist() == [0, 1, 2]
    assert netcon.post_idx.tolist() == [0, 1, 2]
    assert tuple(netcon.get_pre_var(transformed.all_populations).shape) == (3, 1, 1)
    transformed.run(2 * DT)


def test_concat_batch_preserves_learnable_weight_and_delay_modules():
    population = dn.Population(N=1, C=2, dtype=DTYPE)
    population.insert(expsyn, e=0.0, tau=1.0)
    network = dn.Network({"population": population})
    weight = torch.nn.Parameter(torch.tensor([0.5, 1.5], dtype=DTYPE))
    delay = torch.nn.Parameter(torch.tensor([0.1, 0.2], dtype=DTYPE))
    network.connect_one_to_one(
        population[:],
        population[:],
        population.mech.expsyn,
        threshold=-30.0,
        weight=weight,
        delay=delay,
        allow_autapses=True,
    )
    original_spec = next(iter(network.synapse_spec.values()))[0]
    original_weight, original_delay = original_spec[4], original_spec[6]

    concatenated = network.concat()
    concatenated_spec = next(iter(concatenated.synapse_spec.values()))[0]
    concatenated_weight, concatenated_delay = concatenated_spec[4], concatenated_spec[6]
    concatenated.batch(3)
    transformed_spec = next(iter(concatenated.synapse_spec.values()))[0]
    assert transformed_spec[4] is concatenated_weight
    assert transformed_spec[6] is concatenated_delay
    assert concatenated_weight is original_weight
    assert concatenated_delay is original_delay
    assert original_weight.rho.requires_grad
    assert original_delay.rho.requires_grad

    concatenated.build(DT)
    concatenated.init_synapses()
    netcon = next(iter(concatenated.synapses.values()))
    netcon.weight.w.sum().backward()
    assert original_weight.rho.grad is not None
    assert torch.isfinite(original_weight.rho.grad).all()


def test_mixed_concat_batch_backpropagates_through_a_runtime_connection():
    single = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    tree = dn.Tree.from_graph(_branched_tree_graph(), N=1, v_init=-65.0, dtype=DTYPE)
    for population in (single, tree):
        population.insert(pas, g=0.001, e=-70.0)
        population.insert(expsyn, e=0.0, tau=1.0)
    netstim = dn.NetStim(
        N=1,
        interval=100.0,
        start=0.0,
        noise=0.0,
        max_spikes=2,
        dtype=DTYPE,
    )
    network = dn.Network(
        {"single": single, "tree": tree},
        netstim=netstim,
        netcon_train_backend="dense",
    )
    weight = torch.nn.Parameter(torch.tensor([0.5], dtype=DTYPE))
    network.connect_one_to_one(
        netstim[:],
        tree[:, 0],
        tree.mech.expsyn,
        threshold=None,
        weight=weight,
        delay=DT,
    )
    original_weight = next(iter(network.synapse_spec.values()))[0][4]

    transformed = network.train().concat().batch(2).initialize(DT)
    transformed.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
    )
    transformed.init_synapses(reinit_weights=False, reinit_delays=False)
    transformed_weight = next(iter(transformed.synapse_spec.values()))[0][4]

    loss = torch.zeros((), dtype=DTYPE)
    for _ in range(4):
        transformed.step()
        loss = loss + transformed.all_populations.mech.expsyn.g.sum()
    loss.backward()

    assert transformed_weight is original_weight
    assert original_weight.rho.grad is not None
    assert torch.isfinite(original_weight.rho.grad).all()
    assert torch.count_nonzero(original_weight.rho.grad).item() > 0


def test_concat_batch_does_not_mutate_the_source_network():
    populations = {
        name: dn.Population(N=1, C=2, dtype=DTYPE) for name in ("a", "b", "c")
    }
    for population in populations.values():
        population.insert(expsyn, e=0.0, tau=1.0)
    network = dn.Network(populations, netstim=dn.NetStim(N=2, dtype=DTYPE)).train()
    network.connect_one_to_one(
        populations["a"][:],
        populations["b"][:],
        populations["b"].mech.expsyn,
        threshold=-30.0,
        weight=1.0,
        delay=DT,
    )
    original_shapes = {
        name: tuple(population.shape)
        for name, population in network.populations.items()
    }
    original_netstim_shape = tuple(network.netstim.shape)

    transformed = network.concat(ab=["a", "b"]).batch(3)

    assert transformed.training
    assert network.training
    assert not network.is_batched
    assert {
        name: tuple(population.shape)
        for name, population in network.populations.items()
    } == original_shapes
    assert tuple(network.netstim.shape) == original_netstim_shape
    assert transformed.ab.populations["a"] is not network.a
    assert transformed.ab.populations["b"] is not network.b
    assert transformed.c is not network.c
    assert transformed.netstim is not network.netstim
    network.build(DT)
    network.init_synapses()


def test_partial_concat_rebinds_cloned_mechanism_clocks_without_batching():
    populations = {
        name: dn.Population(N=1, C=1, dtype=DTYPE) for name in ("a", "b", "spare")
    }
    for population in populations.values():
        population.insert(expsyn, e=0.0, tau=1.0)
    network = dn.Network(populations)
    network.spare.t.fill_(12.0)

    transformed = network.concat(ab=["a", "b"])
    transformed.spare.t.fill_(34.0)

    assert network.spare.mech.expsyn.t.item() == pytest.approx(12.0)
    assert transformed.spare.mech.expsyn.t.item() == pytest.approx(34.0)
    transformed.initialize(DT)
    assert transformed.spare.mech.expsyn.t.item() == pytest.approx(0.0)
    assert network.spare.mech.expsyn.t.item() == pytest.approx(12.0)


def test_concat_detaches_live_netstim_bptt_state_before_batching():
    netstim = dn.NetStim(
        N=1,
        interval=1.0,
        start=0.0,
        noise=0.5,
        max_spikes=3,
        dtype=DTYPE,
    ).set_dt(DT)
    netstim.initialize()
    netstim.interval.rho.requires_grad_(True)
    netstim.forward(10.0, bptt=True, dt=DT)
    assert netstim.next_stoch_time.requires_grad
    assert not netstim.next_stoch_time.is_leaf

    network = dn.Network(
        {"population": dn.Population(N=1, C=1, dtype=DTYPE)},
        netstim=netstim,
    )
    transformed = network.concat().batch(2)

    assert transformed.netstim is not network.netstim
    assert tuple(network.netstim.shape) == (1,)
    assert tuple(transformed.netstim.shape) == (2, 1)
    assert network.netstim.next_stoch_time.requires_grad
    assert transformed.netstim.next_stoch_time.is_leaf
    assert not transformed.netstim.next_stoch_time.requires_grad


def test_concat_validates_population_selection_before_transforming():
    population = dn.Population(N=1, C=1, dtype=DTYPE)
    network = dn.Network({"population": population})

    with pytest.raises(TypeError, match="not a single string"):
        network.concat(group="population")
    with pytest.raises(ValueError, match="At least one"):
        network.concat(group=[])
    with pytest.raises(ValueError, match="duplicates"):
        network.concat(group=["population", "population"])
    with pytest.raises(KeyError, match="Unknown population"):
        network.concat(group=["missing"])
    with pytest.raises(ValueError, match="reserved"):
        network.concat(netstim=["population"])
    with pytest.raises(ValueError, match="reserved"):
        network.concat(batch=["population"])
    with pytest.raises(ValueError, match="reserved"):
        network.concat(synapses=["population"])
    with pytest.raises(ValueError, match="reserved"):
        network.concat(t=["population"])
    with pytest.raises(TypeError, match="ordered iterable"):
        network.concat(group={"population"})
    with pytest.raises(TypeError, match="ordered iterable"):
        network.concat(group={"population": True})
    with pytest.raises(TypeError, match="Named concatenation groups"):
        network.concat(group=None)
    with pytest.raises(ValueError, match="At least one"):
        dn.Network({}).concat()


def test_concat_preflights_disjoint_groups_atomically_and_preserves_order():
    populations = {
        name: dn.Population(N=1, C=1, dtype=DTYPE) for name in ("a", "b", "c", "d")
    }
    network = dn.Network(populations)

    with pytest.raises(ValueError, match="selected by both"):
        network.concat(ab=["a", "b"], bc=["b", "c"])
    assert list(network.populations) == ["a", "b", "c", "d"]
    assert all(
        not population.is_batched() for population in network.populations.values()
    )

    partially_concatenated = network.concat(ab=["a", "b"])
    assert list(partially_concatenated.populations) == ["ab", "c", "d"]

    replacement_named = network.concat(a=["a", "b"])
    assert list(replacement_named.populations) == ["a", "c", "d"]
    assert list(replacement_named.a.populations) == ["a", "b"]

    with pytest.raises(ValueError, match="already in use"):
        network.concat(c=["a", "b"])

    grouped = network.concat(ab=["a", "b"], cd=["c", "d"])
    assert list(grouped.populations) == ["ab", "cd"]
    assert list(grouped.ab.populations) == ["a", "b"]
    assert list(grouped.cd.populations) == ["c", "d"]
    grouped.batch(2)
    assert all(
        not population.is_batched() for population in network.populations.values()
    )
    assert grouped.ab.shape == (2, 1, 2)
    assert grouped.cd.shape == (2, 1, 2)


def test_concat_reports_selected_component_dtype_mismatches_without_mutation():
    populations = {
        "fp64": dn.Population(N=1, C=1, dtype=torch.float64),
        "fp32": dn.Population(N=1, C=1, dtype=torch.float32),
    }
    network = dn.Network(populations)

    with pytest.raises(
        ValueError,
        match=r"same dtype.*fp64=torch\.float64, fp32=torch\.float32",
    ):
        network.concat()

    assert list(network.populations) == ["fp64", "fp32"]
    assert network.fp64.dtype() == torch.float64
    assert network.fp32.dtype() == torch.float32


def test_network_concat_exposes_the_packed_tree_thread_count():
    tree = dn.Tree.from_graph(_branched_tree_graph(), N=1, dtype=DTYPE)
    network = dn.Network({"tree": tree})

    transformed = network.concat(threads=2)
    assert transformed.all_populations.integrator.threads == 2

    with pytest.raises(ValueError, match="threads must divide 32"):
        network.concat(threads=3)
