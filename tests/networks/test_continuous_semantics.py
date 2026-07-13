from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import ContinuousSynapse
from dendra.models.mod import graded_syn, pas

DT = 0.1
DTYPE = torch.float64


class _DualContinuousInput(ContinuousSynapse):
    ContinuousSynapse.INPUT("first")
    ContinuousSynapse.INPUT("second", keep_old=False)


def _snapshot(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return copy.deepcopy(value)


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            _assert_nested_equal(got, want)
    else:
        assert actual == expected


def _standalone_graded_synapse(n=2):
    post = dn.SingleCompartment(N=n, C=1, dtype=DTYPE)
    post.insert(graded_syn, e=-10.0, g_scale=0.02)
    net = dn.Network({"post": post})
    net.initialize(DT)
    return post.mech.graded_syn


@pytest.mark.parametrize(
    ("reduce", "expected"),
    [
        ("sum", [-1.0, 1.0]),
        ("add", [-1.0, 1.0]),
        ("set", [-3.0, 5.0]),
        ("replace", [-3.0, 5.0]),
        ("last", [-3.0, 5.0]),
        ("max", [2.0, 5.0]),
        ("min", [-3.0, -4.0]),
    ],
)
def test_continuous_receive_reduction_aliases_and_reset_identity(reduce, expected):
    syn = _standalone_graded_synapse()
    syn.g_pre.fill_(9.0)
    syn.reset_continuous_inputs()

    syn.continuous_receive(
        torch.tensor([[2.0], [-4.0]], dtype=DTYPE), input="g_pre", reduce=reduce
    )
    syn.continuous_receive(
        torch.tensor([[-3.0], [5.0]], dtype=DTYPE), input="g_pre", reduce=reduce
    )

    assert syn.g_pre.flatten().tolist() == pytest.approx(expected)
    assert syn.g_pre_old.flatten().tolist() == pytest.approx([9.0, 9.0])
    assert syn._continuous_reset_count.item() == 1


def test_min_and_max_reduce_only_deliveries_not_zero_reset_sentinel():
    syn = _standalone_graded_synapse(1)

    syn.reset_continuous_inputs()
    syn.continuous_receive(torch.tensor([[2.0]]), input="g_pre", reduce="min")
    assert syn.g_pre.item() == pytest.approx(2.0)

    syn.reset_continuous_inputs()
    syn.continuous_receive(torch.tensor([[-2.0]]), input="g_pre", reduce="max")
    assert syn.g_pre.item() == pytest.approx(-2.0)


def test_continuous_receive_default_reducer_and_input_validation():
    syn = _standalone_graded_synapse(1)
    syn.reset_continuous_inputs()
    syn.continuous_receive(torch.tensor([[2.0]], dtype=torch.float32))
    syn.continuous_receive(
        torch.tensor([[5.0]], dtype=DTYPE), con=SimpleNamespace(reduce="max")
    )
    assert syn.g_pre.dtype == DTYPE
    assert syn.g_pre.item() == pytest.approx(5.0)

    before = syn.g_pre.clone()
    with pytest.raises(ValueError, match="no continuous input"):
        syn.continuous_receive(torch.ones_like(syn.g_pre), input="missing")
    with pytest.raises(ValueError, match="Unsupported continuous reduction"):
        syn.continuous_receive(torch.ones_like(syn.g_pre), reduce="median")
    torch.testing.assert_close(syn.g_pre, before, atol=0.0, rtol=0.0)


def test_network_rejects_invalid_continuous_reducer_before_wiring_mutation():
    pre = dn.SingleCompartment(N=1, C=1, dtype=DTYPE)
    post = dn.SingleCompartment(N=1, C=1, dtype=DTYPE)
    post.insert(graded_syn)
    net = dn.Network({"pre": pre, "post": post})
    before = copy.copy(net.continuous_synapse_spec)

    with pytest.raises(ValueError, match="Unsupported continuous reduction"):
        net.connect_continuous_one_to_one(
            pre[:], post[:], post.mech.graded_syn, reduce="median"
        )

    assert net.continuous_synapse_spec == before


def test_multiple_inputs_require_explicit_name_and_keep_old_is_per_declaration():
    post = dn.SingleCompartment(N=1, C=1, dtype=DTYPE)
    post.insert(_DualContinuousInput)
    net = dn.Network({"post": post})
    net.initialize(DT)
    syn = post.mech._DualContinuousInput

    assert not hasattr(syn, "second_old")
    with pytest.raises(ValueError, match="requires `input=`"):
        syn.continuous_receive(torch.ones_like(syn.first))

    syn.first.fill_(4.0)
    syn.second.fill_(6.0)
    syn.reset_continuous_inputs()
    assert syn.first_old.item() == pytest.approx(4.0)
    assert syn.first.item() == pytest.approx(0.0)
    assert syn.second.item() == pytest.approx(0.0)


def _build_convergent_continuous_network():
    pre_a = dn.SingleCompartment(N=2, C=1, v_init=-20.0, dtype=DTYPE)
    pre_b = dn.SingleCompartment(N=2, C=1, v_init=-40.0, dtype=DTYPE)
    post = dn.SingleCompartment(N=2, C=1, v_init=-65.0, dtype=DTYPE)
    for population in (pre_a, pre_b, post):
        population.insert(pas, g=0.001, e=-65.0)
    post.insert(graded_syn, e=-10.0, g_scale=0.0)

    net = dn.Network({"pre_a": pre_a, "pre_b": pre_b, "post": post})
    net.connect_continuous_one_to_one(
        pre_a[:],
        post[:],
        post.mech.graded_syn,
        pre_var="v",
        input="g_pre",
        weight=torch.tensor([2.0, 1.0], dtype=DTYPE),
        delay=0.0,
    )
    net.connect_continuous_one_to_one(
        pre_b[:],
        post[:],
        post.mech.graded_syn,
        pre_var="v",
        input="g_pre",
        weight=torch.tensor([0.5, 3.0], dtype=DTYPE),
        delay=2 * DT,
    )
    net.initialize(DT)
    return net


def test_multiple_continuous_connections_reset_once_and_sum_delayed_payloads():
    net = _build_convergent_continuous_network()
    values_a = ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0])
    values_b = ([10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0])
    actual = []
    old = []

    for a, b in zip(values_a, values_b):
        net.pre_a.v.copy_(torch.tensor(a, dtype=DTYPE).view_as(net.pre_a.v))
        net.pre_b.v.copy_(torch.tensor(b, dtype=DTYPE).view_as(net.pre_b.v))
        net.step()
        actual.append(net.post.mech.graded_syn.g_pre.flatten().clone())
        old.append(net.post.mech.graded_syn.g_pre_old.flatten().clone())

    expected = torch.tensor(
        [
            [2.0, 2.0],
            [6.0, 4.0],
            [15.0, 66.0],
            [29.0, 128.0],
        ],
        dtype=DTYPE,
    )
    torch.testing.assert_close(torch.stack(actual), expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        torch.stack(old),
        torch.cat([torch.zeros((1, 2), dtype=DTYPE), expected[:-1]], dim=0),
        atol=0.0,
        rtol=0.0,
    )
    assert net.post.mech.graded_syn._continuous_reset_count.item() == len(values_a)


def _build_sparse_reduction_network(reduce, first, second, *, second_delay=0.0):
    pre_a = dn.SingleCompartment(N=1, C=1, v_init=first, dtype=DTYPE)
    pre_b = dn.SingleCompartment(N=1, C=1, v_init=second, dtype=DTYPE)
    post = dn.SingleCompartment(N=2, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(graded_syn, g_scale=0.0)
    net = dn.Network({"pre_a": pre_a, "pre_b": pre_b, "post": post})
    net.connect_continuous_one_to_one(
        pre_a[:], post[0], post.mech.graded_syn, reduce=reduce
    )
    net.connect_continuous_one_to_one(
        pre_b[:],
        post[1],
        post.mech.graded_syn,
        reduce=reduce,
        delay=second_delay,
    )
    net.initialize(DT)
    return net


@pytest.mark.parametrize(
    ("reduce", "first", "second"),
    [
        ("min", 2.0, 3.0),
        ("max", -2.0, -3.0),
        ("set", 2.0, 3.0),
        ("last", 2.0, 3.0),
    ],
)
def test_sparse_nonadditive_reductions_ignore_untargeted_scatter_slots(
    reduce, first, second
):
    net = _build_sparse_reduction_network(reduce, first, second)

    net.step()

    torch.testing.assert_close(
        net.post.mech.graded_syn.g_pre.flatten(),
        torch.tensor([first, second], dtype=DTYPE),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("reduce", "first", "second"),
    [
        ("min", 2.0, 3.0),
        ("max", -2.0, -3.0),
        ("set", 2.0, 3.0),
        ("last", 2.0, 3.0),
    ],
)
def test_sparse_nonadditive_reductions_preserve_delayed_delivery_presence(
    reduce, first, second
):
    net = _build_sparse_reduction_network(reduce, first, second, second_delay=2 * DT)
    actual = []

    for _ in range(3):
        net.step()
        actual.append(net.post.mech.graded_syn.g_pre.flatten().clone())

    torch.testing.assert_close(
        torch.stack(actual),
        torch.tensor([[first, 0.0], [first, 0.0], [first, second]], dtype=DTYPE),
        atol=0.0,
        rtol=0.0,
    )


def _build_zero_payload_checkpoint_network():
    pre_immediate = dn.SingleCompartment(N=1, C=1, v_init=-2.0, dtype=DTYPE)
    pre_delayed = dn.SingleCompartment(N=1, C=1, v_init=0.0, dtype=DTYPE)
    post = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(graded_syn, g_scale=0.0)
    net = dn.Network(
        {"pre_immediate": pre_immediate, "pre_delayed": pre_delayed, "post": post}
    )
    net.connect_continuous_one_to_one(
        pre_immediate[:], post[:], post.mech.graded_syn, reduce="max"
    )
    net.connect_continuous_one_to_one(
        pre_delayed[:],
        post[:],
        post.mech.graded_syn,
        reduce="max",
        delay=2 * DT,
    )
    net.initialize(DT)
    return net


def test_checkpoint_preserves_a_pending_zero_continuous_delivery():
    source = _build_zero_payload_checkpoint_network()
    source.step()
    checkpoint = _snapshot(source.state_dict_for_checkpoint())
    delayed_state = next(
        state
        for name, state in checkpoint["netcons"]["continuous"].items()
        if name.startswith("1:pre_delayed:")
    )
    assert delayed_state["delivery_mask"].any()
    assert not delayed_state["delivery_buffer"].any()

    resumed = _build_zero_payload_checkpoint_network()
    resumed.restore_dict_from_checkpoint(checkpoint)
    resumed.step()
    assert resumed.post.mech.graded_syn.g_pre.item() == pytest.approx(-2.0)
    resumed.step()
    # Zero is a real delayed sample and must participate in max reduction.
    assert resumed.post.mech.graded_syn.g_pre.item() == pytest.approx(0.0)


def test_legacy_checkpoint_without_delivery_mask_uses_nonzero_best_effort():
    source = _build_zero_payload_checkpoint_network()
    source.step()
    checkpoint = _snapshot(source.state_dict_for_checkpoint())
    delayed_state = next(
        state
        for name, state in checkpoint["netcons"]["continuous"].items()
        if name.startswith("1:pre_delayed:")
    )
    delayed_state.pop("delivery_mask")

    resumed = _build_zero_payload_checkpoint_network()
    resumed.restore_dict_from_checkpoint(checkpoint)
    delayed = next(
        con
        for name, con in resumed.continuous_synapses.items()
        if name.startswith("1:pre_delayed:")
    )
    assert torch.equal(delayed.delivery_mask, delayed.delivery_buffer != 0)


def test_delivery_mask_follows_clear_rebuild_and_device_lifecycle():
    net = _build_sparse_reduction_network("min", 2.0, 3.0, second_delay=2 * DT)
    delayed = next(
        con
        for name, con in net.continuous_synapses.items()
        if name.startswith("1:pre_b:")
    )
    delayed.delivery_mask.fill_(True)

    delayed.zero(clear_deliveries=False)
    assert delayed.delivery_mask.all()
    delayed.zero(clear_deliveries=True)
    assert not delayed.delivery_mask.any()

    delayed.delivery_mask.fill_(True)
    delayed.initialize(reinit_weights=False, reinit_delays=True)
    assert not delayed.delivery_mask.any()
    assert delayed.delivery_mask.shape == delayed.delivery_buffer.shape

    delayed.to("cpu")
    assert delayed.delivery_mask.device.type == "cpu"
    assert delayed.post_delivery_mask.device.type == "cpu"
    assert delayed.post_zero_delivery_mask.device.type == "cpu"


@pytest.mark.parametrize(
    ("delays", "advance_name"),
    [
        ([0.0, 0.2, 0.0, 0.2], "_advance_mixed_uniform"),
        ([0.0, 0.1, 0.0, 0.2], "_advance_mixed"),
    ],
)
def test_mixed_immediate_and_delayed_paths_match_manual_edge_queue(
    delays, advance_name
):
    pre = dn.SingleCompartment(N=2, C=1, v_init=0.0, dtype=DTYPE)
    post = dn.SingleCompartment(N=2, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(graded_syn, g_scale=0.0)
    net = dn.Network({"pre": pre, "post": post})
    net.connect_continuous(
        pre[:],
        post[:],
        post.mech.graded_syn,
        conn_spec="all_to_all",
        weight=torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=DTYPE),
        delay=torch.tensor(delays, dtype=DTYPE),
    )
    net.initialize(DT)
    con = next(iter(net.continuous_synapses.values()))
    assert con.advance.__name__ == advance_name

    pending = {}
    source_rows = ([2.0, 3.0], [5.0, 7.0], [11.0, 13.0], [17.0, 19.0])
    for step, values in enumerate(source_rows):
        expected = pending.pop(step, torch.zeros(2, dtype=DTYPE))
        values_t = torch.tensor(values, dtype=DTYPE)
        for edge in range(con.n_connections()):
            payload = con.weight()[edge].detach() * values_t[con.pre_idx[edge]]
            delay_steps = int(con.delay_steps[edge])
            target = int(con.post_idx[edge])
            if delay_steps == 0:
                expected[target] += payload
            else:
                due = pending.setdefault(
                    step + delay_steps, torch.zeros(2, dtype=DTYPE)
                )
                due[target] += payload

        net.pre.v.copy_(values_t.view_as(net.pre.v))
        net.step()
        torch.testing.assert_close(
            net.post.mech.graded_syn.g_pre.flatten(), expected, atol=0.0, rtol=0.0
        )


def _build_autonomous_continuous_network():
    pre = dn.SingleCompartment(N=2, C=1, v_init=-35.0, dtype=DTYPE)
    pre.insert(pas, g=0.003, e=-20.0)
    post = dn.SingleCompartment(N=2, C=1, v_init=-65.0, dtype=DTYPE)
    post.insert(pas, g=0.001, e=-70.0)
    post.insert(graded_syn, e=-10.0, g_scale=0.02)
    net = dn.Network({"pre": pre, "post": post})
    net.connect_continuous_one_to_one(
        pre[:],
        post[:],
        post.mech.graded_syn,
        pre_var="v",
        input="g_pre",
        weight=torch.tensor([0.4, 0.9], dtype=DTYPE),
        delay=torch.tensor([DT, 4 * DT], dtype=DTYPE),
    )
    net.initialize(DT)
    return net


@pytest.mark.parametrize(
    "runner", ["partitioned", "longrun", "checkpointed", "activation_checkpointed"]
)
def test_continuous_delays_are_invariant_to_run_chunk_and_checkpoint_boundaries(runner):
    reference = _build_autonomous_continuous_network()
    reference.run(0.9)
    expected = _snapshot(reference.state_dict_for_checkpoint())

    net = _build_autonomous_continuous_network()
    if runner == "partitioned":
        net.run(0.3)
        net.run(0.6)
    elif runner == "longrun":
        net.longrun(0.9, chunklength=2)
    elif runner == "checkpointed":
        source = _build_autonomous_continuous_network()
        source.run(0.4)
        checkpoint = _snapshot(source.state_dict_for_checkpoint())
        net.restore_dict_from_checkpoint(checkpoint)
        net.run(0.5)
    else:
        net.longrun_checkpointed(0.9, chunklength=2)

    _assert_nested_equal(net.state_dict_for_checkpoint(), expected)
