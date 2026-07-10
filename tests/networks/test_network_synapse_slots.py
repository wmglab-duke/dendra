from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn


def _banked_cell():
    cell = dn.Population(N=1, C=3, v_init=-65.0, dtype=torch.float64)
    cell[:, 1].insert(expsyn.rename("syn"), copies=4, e=0.0, tau=1.0)
    net = dn.Network({"cell": cell}, netstim=dn.NetStim(N=4))
    return net, cell


def test_population_and_slice_slots_select_local_synapse_slots():
    net, cell = _banked_cell()
    syn = cell.mech.syn

    all_slots = cell.slots(syn)
    assert isinstance(all_slots, dn.SynapseSlots)
    assert all_slots.local_index.tolist() == [0, 1, 2, 3]
    assert all_slots.flat_index.tolist() == [1, 1, 1, 1]

    selected = cell.slots(syn, local_index=[1, 3])
    assert selected.local_index.tolist() == [1, 3]
    assert selected.flat_index.tolist() == [1, 1]

    sliced = selected[1:]
    assert sliced.local_index.tolist() == [3]

    assert cell[:, 1].slots(syn).local_index.tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="outside the requested population region"):
        cell[:, 0].slots(syn, local_index=[0])


def test_physical_compartment_target_is_rejected_when_slots_are_ambiguous():
    net, cell = _banked_cell()
    syn = cell.mech.syn

    with pytest.raises(ValueError, match="multiple local slots"):
        net.connect_one_to_one(
            net.netstim[:1],
            cell[:, 1],
            syn,
            threshold=None,
            weight=0.1,
            delay=0.1,
        )


def test_connect_one_to_one_slots_materializes_without_threshold_state():
    net, cell = _banked_cell()
    syn = cell.mech.syn
    slots = net.synapse_slots(cell, syn, slots=[0, 1, 2, 3])

    net.connect_one_to_one_slots(
        net.netstim[:],
        slots,
        threshold=None,
        weight=torch.full((4,), 0.1, dtype=torch.float64),
        delay=0.1,
    )

    net.build(0.1)
    net.init_synapses()
    assert len(net.synapses) == 1
    netcon = next(iter(net.synapses.values()))
    assert netcon.skip_thresholding
    assert netcon.has_spiked.numel() == 0
    assert netcon.post_idx.tolist() == [0, 1, 2, 3]


def test_batched_slot_targets_are_rebased_in_synapse_local_frame():
    net, cell = _banked_cell()
    syn = cell.mech.syn
    slots = net.synapse_slots(cell, syn, slots=[0, 1, 2, 3])

    net.connect_one_to_one_slots(
        net.netstim[:],
        slots,
        threshold=None,
        weight=torch.full((4,), 0.1, dtype=torch.float64),
        delay=0.1,
    )

    net.batch(3)
    net.build(0.1)
    net.init_synapses()

    netcon = next(iter(net.synapses.values()))
    assert netcon.syn.shape_f == (3, 4)
    assert netcon._syn_numel == 12
    assert netcon.post_idx.tolist() == list(range(12))
    assert int(netcon.post_idx.max()) < netcon._syn_numel


def test_batched_mechanism_pre_var_indices_remain_local_after_batch():
    from dendra.models.mod import spikedetect

    pre = dn.Population(N=1, C=3, v_init=-65.0, dtype=torch.float64)
    post = dn.Population(N=1, C=3, v_init=-65.0, dtype=torch.float64)
    pre[:, 1].insert(spikedetect, threshold=-30.0)
    post[:, 1].insert(expsyn.rename("syn_prevar_batch"), copies=2, e=0.0, tau=1.0)

    net = dn.Network({"pre": pre, "post": post})
    syn = post.mech.syn_prevar_batch
    net.connect_one_to_one_slots(
        pre[:, 1],
        post.slots(syn, local_index=[0]),
        threshold=None,
        weight=0.1,
        delay=0.1,
        pre_var="mech.spikedetect.spikes",
    )

    net.batch(4)
    net.build(0.1)
    net.init_synapses()

    netcon = next(iter(net.synapses.values()))
    assert netcon.pre_idx.tolist() == [0, 1, 2, 3]
    assert netcon.post_idx.tolist() == [0, 2, 4, 6]
    assert int(netcon.pre_idx.max()) < 4
    assert int(netcon.post_idx.max()) < netcon._syn_numel
