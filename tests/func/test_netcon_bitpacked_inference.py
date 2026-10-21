"""Packed inference shares transport while preserving native compact carry."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn
from dendra.models.networks.net import make_weight
from dendra.models.networks.netcon import NetCon
from dendra.models.parametric import to_param
from tests.func.test_netcon_source_history import (
    DT,
    DTYPE,
    _clone,
    _close,
    _flat_checkpoint,
    _native_source,
    _tape,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _netcon(*, mode="threshold", batch=None, fanout=False, size=3):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
        pre = (
            dn.NetStim(N=size, noise=0.0, seed=19)
            if mode == "netstim"
            else dn.Population(N=1, C=size, dtype=DTYPE)
        )
        if mode == "raw":
            pre.register_buffer("gate", torch.zeros((1, size), dtype=DTYPE))
        post = dn.Population(N=1, C=size, dtype=DTYPE)
        post.insert(expsyn.rename("syn"))
        network = dn.Network(
            {"post": post} if mode == "netstim" else {"pre": pre, "post": post},
            netstim=pre if mode == "netstim" else None,
            netcon_delay_backend="bitpacked_history",
        )
        edges = size * size if fanout else size
        connect = network.connect_dense if fanout else network.connect_one_to_one
        connect(
            pre[:],
            post[:],
            post.mech.syn,
            pre_var="gate" if mode == "raw" else "v",
            threshold=0.0 if mode == "threshold" else None,
            weight=torch.linspace(-0.3, 0.9, edges, dtype=DTYPE),
            delay=torch.linspace(DT, 3 * DT, edges, dtype=DTYPE),
        )
        if batch is not None:
            network.batch(batch)
            if mode == "raw":
                pre.gate = torch.zeros(pre.shape, dtype=DTYPE)
        network.eval()
        network.initialize(DT)
    native = next(iter(network.synapses.values()))
    assert not native.training
    assert native._checkpoint_runtime_kind() == "bitpacked_history"
    assert native.advance.__name__ == "advance_non_diff_bitpacked_history"
    return network, native


@pytest.mark.parametrize("mode", ("threshold", "raw", "netstim"))
@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("fanout", (False, True))
def test_packed_projection_matches_native_carry_scratch_and_input_purity(
    mode, batch, fanout
):
    _, native = _netcon(mode=mode, batch=batch, fanout=fanout)
    plan, tensors = dn.func.make_functional_netcon(native)
    assert plan.runtime == "bitpacked_history"
    assert tensors.delays is None
    assert tensors.state["delivery_buffer"].shape == (1, native._syn_numel)
    assert "backend_state" not in tensors.state
    before = _clone(tensors.state)
    state = tensors.state
    for row in _tape(plan.source_shape, mode):
        frozen = _clone(state)
        next_state, payload = plan.step(tensors.weights, state, row)
        _close(state, frozen)
        _native_source(native, row, mode, False)
        previous = native.syn.g.clone()
        native.advance()
        _close(next_state, _flat_checkpoint(native))
        torch.testing.assert_close(payload, native.syn.g - previous)
        torch.testing.assert_close(
            next_state["delivery_buffer"], payload.reshape(1, -1)
        )
        state = next_state
    _close(tensors.state, before)


def test_current_weights_reconstruct_already_recorded_activity_and_remain_differentiable():
    _, native = _netcon(mode="raw", fanout=True)
    plan, tensors = dn.func.make_functional_netcon(native)
    state = tensors.state
    active = torch.ones(plan.source_shape, dtype=DTYPE)
    for _ in range(plan._depth):
        state, _ = plan.step(tensors.weights, state, active)
        native.pre.gate = active
        native.advance()
    weights = (tensors.weights.detach() * 2.7).requires_grad_()
    native.weight.w = weights.detach().clone()
    frozen = _clone(state)
    due, payload = plan.read_due(state, weights=weights)
    previous = native.syn.g.clone()
    native.advance()
    torch.testing.assert_close(payload, native.syn.g - previous)
    gradient = torch.autograd.grad(payload.square().sum(), weights)[0]
    torch.testing.assert_close(
        gradient, 2 * payload.detach().flatten()[native.post_idx]
    )
    assert gradient.abs().sum() > 0
    torch.testing.assert_close(due["delivery_buffer"], payload.reshape(1, -1))
    _close(state, frozen)


@pytest.mark.parametrize("mode", ("threshold", "raw", "netstim"))
def test_nonempty_history_roundtrips_native_checkpoint_and_fresh_reset(mode):
    network, native = _netcon(mode=mode)
    plan, tensors = dn.func.make_functional_netcon(native)
    state = tensors.state
    tape = _tape(plan.source_shape, mode)
    for row in tape[:5]:
        state, _ = plan.step(tensors.weights, state, row)
    prefix = state
    frozen = _clone(state)
    plan.commit_state_(native, state)
    _close(plan.extract().state, state)
    for row in tape[5:]:
        state, payload = plan.step(tensors.weights, state, row)
        _native_source(native, row, mode, False)
        previous = native.syn.g.clone()
        native.advance()
        _close(plan.extract().state, state)
        torch.testing.assert_close(payload, native.syn.g - previous)
    reset = plan.initial_state(state, t=2.75)
    network.t.fill_(2.75)
    native.initialize(reinit_weights=False, reinit_delays=False)
    _close(reset, _flat_checkpoint(native))
    assert reset["global_step"].item() == round(2.75 / DT)
    assert not reset["spike_history_packed"].any()
    assert not reset["delivery_buffer"].any()
    _close(prefix, frozen)


@pytest.mark.parametrize("mode", ("threshold", "raw", "netstim"))
def test_fullgraph_packed_projection_and_explicit_weight_gradients(mode):
    _, native = _netcon(mode=mode, fanout=True)
    plan, tensors = dn.func.make_functional_netcon(native)
    compiled = torch.compile(plan.step, backend="aot_eager", fullgraph=True)
    weights = tensors.weights.detach().clone().requires_grad_()
    actual, expected = tensors.state, tensors.state
    loss, reference = 0, 0
    for row in _tape(plan.source_shape, mode):
        actual, payload = compiled(weights, actual, row)
        expected, baseline = plan.step(weights, expected, row)
        _close(actual, expected)
        torch.testing.assert_close(payload, baseline)
        loss = loss + payload.square().sum()
        reference = reference + baseline.square().sum()
    torch.testing.assert_close(
        torch.autograd.grad(loss, weights)[0],
        torch.autograd.grad(reference, weights)[0],
    )


@pytest.mark.parametrize("lanes", (0, 3))
def test_vmap_packed_history_with_independent_history_weights_and_sources(lanes):
    _, native = _netcon(mode="raw")
    plan, tensors = dn.func.make_functional_netcon(native)
    states = pytree.tree_map(
        lambda value: value.unsqueeze(0).repeat(lanes, *([1] * value.ndim)),
        tensors.state,
    )
    packed = states["spike_history_packed"]
    states["spike_history_packed"] = (
        torch.arange(packed.numel()).reshape(packed.shape).remainder(8)
    )
    states["current_time_step"] = (
        torch.arange(lanes).reshape(lanes, 1).remainder(plan._depth)
    )
    weights = tensors.weights.unsqueeze(0) * torch.arange(
        1, lanes + 1, dtype=DTYPE
    ).reshape(lanes, 1)
    values = _tape(plan.source_shape, "raw")[:lanes]
    result = torch.vmap(plan.step)(weights, states, values)
    if lanes:
        expected = [
            plan.step(
                weights[index],
                pytree.tree_map(lambda value: value[index], states),
                values[index],
            )
            for index in range(lanes)
        ]
        reference = pytree.tree_map(lambda *leaves: torch.stack(leaves), *expected)
        _close(result, reference)
    else:
        assert result[1].shape == (0, *plan.shape)
        assert result[0]["delivery_buffer"].shape == (0, 1, native._syn_numel)


def test_packed_word_boundary_and_scratch_clearing_match_native():
    _, native = _netcon(size=65)
    plan, tensors = dn.func.make_functional_netcon(native)
    assert tensors.state["spike_history_packed"].shape[1] == 2
    state = tensors.state
    tape = torch.zeros((10, *plan.source_shape), dtype=DTYPE)
    tape[0, 0, [0, 62, 63, 64]] = 1
    tape[2, 0, [1, 62, 64]] = 1
    for row in tape:
        state, payload = plan.step(tensors.weights, state, row)
        native.pre.v = row
        native.advance()
        _close(state, _flat_checkpoint(native))
    assert not payload.any()
    assert not state["delivery_buffer"].any()


def test_packed_admission_and_frozen_boundary_errors_are_explicit():
    _, native = _netcon()
    plan, tensors = dn.func.make_functional_netcon(native)
    with pytest.raises(TypeError, match="weights"):
        plan.read_due(tensors.state)
    with pytest.raises(ValueError, match="Fixed-delay"):
        plan.read_due(tensors.state, weights=tensors.weights, delays=tensors.weights)
    native.bitpack_source_threshold.add_(0.5)
    with pytest.raises(dn.func.FunctionalizationError, match="routing|detector"):
        plan.extract()
    native.bitpack_source_threshold.sub_(0.5)
    native._sparse_calendar[0] = [(native.post_idx[:1], tensors.weights[:1])]
    with pytest.raises(dn.func.FunctionalizationError, match="side calendars"):
        plan.extract()
    native._sparse_calendar.clear()
    native.schedule(con_indices=[0], times_ms=[0.0], weight=1.0)
    with pytest.raises(dn.func.FunctionalizationError, match="NetCon schedules"):
        dn.func.make_functional_netcon(native)
    with pytest.raises(dn.func.FunctionalizationError, match="configuration changed"):
        plan.read_due(tensors.state, weights=tensors.weights)


def test_packed_canonical_transport_guards_reject_changed_helpers():
    _, native = _netcon()
    plan, tensors = dn.func.make_functional_netcon(native)
    native._bitpack_source_spikes_this_step = lambda: None
    with pytest.raises(dn.func.FunctionalizationError, match="source_spikes_this_step"):
        plan.step(tensors.weights, tensors.state, native.pre.v)
    with pytest.raises(dn.func.FunctionalizationError, match="source_spikes_this_step"):
        dn.func.make_functional_netcon(native)
    del native._bitpack_source_spikes_this_step
    native.advance = native.advance_non_diff_dense_uniform
    with pytest.raises(dn.func.FunctionalizationError, match="transport adapter"):
        dn.func.make_functional_netcon(native)


def test_empty_packed_history_retains_stable_schema_and_native_checkpoint():
    network, template = _netcon(mode="raw")
    empty = torch.empty(0, dtype=DTYPE)
    storage = {"device": "cpu", "dtype": DTYPE}
    native = NetCon(
        pre=template.pre,
        pre_idx=torch.empty(0, dtype=torch.long),
        thresholds=empty,
        post=template.post,
        post_idx=torch.empty(0, dtype=torch.long),
        post_syn=template.syn,
        weight=make_weight([to_param(empty)], [1], **storage),
        delay=make_weight([to_param(empty)], [1], **storage),
        dt=DT,
        pre_var="gate",
        delay_backend="bitpacked_history",
        **storage,
    )
    native.setreference("t", lambda: network.t)
    native.eval()
    native.initialize()
    plan, tensors = dn.func.make_functional_netcon(native)
    assert tensors.state["spike_history_packed"].shape[1] == 1
    state, payload = plan.step(tensors.weights, tensors.state, template.pre.gate)
    assert not payload.any()
    native.advance()
    _close(state, plan.extract().state)
    plan.commit_state_(native, state)
    _close(state, plan.extract().state)
