from __future__ import annotations

import networkx as nx
import pytest

import dendra as dn
from dendra.models.mod import expsyn, spikedetect


def _one_compartment_network(*, with_netstim: bool = False):
    graph = nx.DiGraph()
    graph.add_node(
        0,
        name="Cell.soma[0](0.5)",
        L=10.0,
        diam=10.0,
        Ra=100.0,
        cm=1.0,
        area=314.0,
        x=0.0,
        y=0.0,
        z=0.0,
    )
    cell = dn.Tree.from_graph(graph, N=1)
    cell.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    stim = dn.NetStim(N=1, interval=10.0, start=1.0, max_spikes=1)
    network = dn.Network({"cell": cell}, netstim=stim if with_netstim else None)
    return network, stim


def test_netstim_can_be_attached_after_network_build():
    network, stim = _one_compartment_network()
    network.build(0.1)
    assert network.built
    assert network.netstim is None

    returned = network.attach_netstim(stim)
    assert returned is network
    assert network.netstim is stim
    assert network.netstim.name == "netstim"
    assert not network.built

    network.connect_one_to_one(
        network.netstim[:],
        network.cell[:],
        network.cell.mech.syn,
        weight=0.1,
        delay=0.1,
    )
    assert not network.built

    network.build(0.1)
    assert network.built
    assert len(network.synapses) == 1


def test_any_new_connection_invalidates_materialized_netcons():
    network, _ = _one_compartment_network()
    network.build(0.1)
    assert network.built

    network.connect_one_to_one(
        network.cell[:],
        network.cell[:],
        network.cell.mech.syn,
        threshold=-30.0,
        weight=0.1,
        delay=0.1,
        allow_autapses=True,
    )
    assert not network.built


def test_run_rejects_stale_wiring_after_attachment():
    network, stim = _one_compartment_network()
    network.build(0.1)
    network.attach_netstim(stim)

    with pytest.raises(RuntimeError, match="wiring has changed"):
        network.run(0.1)


def test_constructor_netstim_path_uses_same_registration_contract():
    network, stim = _one_compartment_network(with_netstim=True)
    assert network.netstim is stim
    assert network.netstim.name == "netstim"
    assert not network.built


def test_attach_is_idempotent_for_same_object_and_rejects_implicit_replace():
    network, stim = _one_compartment_network()
    network.attach_netstim(stim)
    assert network.attach_netstim(stim) is network

    replacement = dn.NetStim(N=1)
    with pytest.raises(RuntimeError, match="already has a NetStim"):
        network.attach_netstim(replacement)


def test_replace_with_different_shape_rejected_when_specs_exist():
    network, stim = _one_compartment_network()
    network.attach_netstim(stim)
    network.connect_one_to_one(
        network.netstim[:],
        network.cell[:],
        network.cell.mech.syn,
        weight=0.1,
        delay=0.1,
    )

    with pytest.raises(ValueError, match="different shape"):
        network.attach_netstim(dn.NetStim(N=2), replace=True)


def test_reserved_netstim_population_name_is_rejected():
    graph = nx.DiGraph()
    graph.add_node(
        0,
        name="Cell.soma(0.5)",
        L=10.0,
        diam=10.0,
        Ra=100.0,
        cm=1.0,
        area=314.0,
        x=0.0,
        y=0.0,
        z=0.0,
    )
    cell = dn.Tree.from_graph(graph, N=1)
    with pytest.raises(ValueError, match="reserved"):
        dn.Network({"netstim": cell})


def test_event_pre_var_skips_per_netcon_threshold_history():
    graph = nx.DiGraph()
    graph.add_node(
        0,
        name="Cell.soma(0.5)",
        L=10.0,
        diam=10.0,
        Ra=100.0,
        cm=1.0,
        area=314.0,
        x=0.0,
        y=0.0,
        z=0.0,
    )
    cell = dn.Tree.from_graph(graph, N=1)
    cell.insert(spikedetect, threshold=-30.0)
    cell.insert(expsyn.rename("syn"), e=0.0, tau=1.0)
    network = dn.Network({"cell": cell})
    network.connect_one_to_one(
        network.cell[:],
        network.cell[:],
        network.cell.mech.syn,
        threshold=None,
        pre_var="mech.spikedetect.spikes",
        weight=0.1,
        delay=0.1,
        allow_autapses=True,
    )
    network.build(0.1)
    network.init_synapses()

    netcon = network.synapses["cell:mech_spikedetect_spikes->cell:syn"]
    assert netcon.skip_thresholding
    assert netcon.pre_var == "mech.spikedetect.spikes"
    assert netcon.has_spiked.numel() == 0
