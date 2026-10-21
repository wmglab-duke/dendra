"""Functional source history preserves native delivery-time transport."""

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

pytestmark = pytest.mark.cpu
DT = 0.125
DTYPE = torch.float64


def _netcon(
    *,
    mode="threshold",
    diff_spiking=True,
    taps=2,
    diff_delays=True,
    batch=None,
    policy="source_history",
    fanout=False,
    diff_weights=True,
    size=3,
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
        pre = (
            dn.NetStim(N=size, interval=1.0, noise=0.0, seed=5)
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
            netcon_train_backend=policy,
        )
        edges = size * size if fanout else size
        connect = network.connect_dense if fanout else network.connect_one_to_one
        connect(
            pre[:],
            post[:],
            post.mech.syn,
            pre_var="gate" if mode == "raw" else "v",
            threshold=None if mode != "threshold" else 0.0,
            weight=torch.linspace(0.3, 0.9, edges, dtype=DTYPE),
            delay=torch.linspace(DT, 3.4 * DT, edges, dtype=DTYPE),
        )
        if batch is not None:
            network.batch(batch)
            if mode == "raw":
                pre.gate = torch.zeros(pre.shape, dtype=DTYPE)
        network.initialize(DT)
        native = next(iter(network.synapses.values()))
        native.set_diff_config(
            diff_weights=diff_weights,
            diff_delays=diff_delays,
            diff_spiking=diff_spiking,
            diff_scheduled_times=False,
            taps=taps,
            sigma=0.43,
            tau=0.17,
            train_delay_backend=policy,
        )
        native.train()
        native.initialize(reinit_weights=False, reinit_delays=False)
    assert native._checkpoint_runtime_kind() == "source_history"
    return network, native


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    actual_values, spec = pytree.tree_flatten(actual)
    expected_values, other = pytree.tree_flatten(expected)
    assert spec == other
    for value, reference in zip(actual_values, expected_values, strict=True):
        torch.testing.assert_close(value, reference)


def _native_source(native, values, mode, diff_spiking):
    if mode == "netstim":
        native.pre.spikes = values > 0
        native.pre.spike_gate = values
    elif mode == "raw":
        native.pre.gate = values
    else:
        native.pre.v = values


def _flat_checkpoint(native):
    # Compact functional carry includes fixed-capacity pending tables in
    # addition to the Tensor fields in native legacy checkpoints.
    return dn.func.make_functional_netcon(native)[1].state


def _tape(shape, mode):
    base = torch.tensor(
        [
            [-0.2, 0.1, -0.1],
            [0.3, 0.4, 0.2],
            [0.2, -0.2, 0.3],
            [-0.1, 0.2, 0.4],
            [0.1, -0.4, -0.1],
            [-0.2, 0.3, 0.4],
            [0.3, 0.2, -0.1],
            [-0.1, -0.3, 0.2],
        ],
        dtype=DTYPE,
    )
    if mode == "netstim":
        base = (base > 0).to(DTYPE)
    return base.reshape(8, *([1] * (len(shape) - 1)), 3).expand(8, *shape).clone()


@pytest.mark.parametrize("mode", ("threshold", "raw", "netstim"))
@pytest.mark.parametrize("diff_spiking", (False, True))
@pytest.mark.parametrize("taps", (None, 2, 3))
@pytest.mark.parametrize("batch", (None, 2))
def test_source_history_native_values_gradients_and_pure_inputs(
    mode, diff_spiking, taps, batch
):
    _, native = _netcon(
        mode=mode,
        diff_spiking=diff_spiking,
        taps=taps or 2,
        diff_delays=taps is not None,
        batch=batch,
    )
    plan, tensors = dn.func.make_functional_netcon(native)
    assert plan.runtime == "source_history"
    assert "backend_state" not in tensors.state
    history_name = "source_gate_history" if diff_spiking else "spike_history_packed"
    assert history_name in tensors.state
    weights = tensors.weights.detach().clone().requires_grad_()
    native_weights = weights.detach().clone().requires_grad_()
    native.weight.w = native_weights
    delays = (
        None
        if tensors.delays is None
        else tensors.delays.detach().clone().requires_grad_()
    )
    native_delays = None if delays is None else delays.detach().clone().requires_grad_()
    if delays is not None:
        native.delay_ms.w = native_delays
    values = _tape(plan.source_shape, mode).requires_grad_()
    native_values = values.detach().clone().requires_grad_()
    initial = _clone(tensors.state)
    state = tensors.state
    observed, expected = [], []
    for index, (row, native_row) in enumerate(zip(values, native_values, strict=True)):
        before = _clone(state)
        next_state, payload = plan.step(weights, state, row, delays=delays)
        _close(state, before)
        state = next_state
        _native_source(native, native_row, mode, diff_spiking)
        previous = native.syn.g
        native.advance()
        _close(state, _flat_checkpoint(native))
        native_payload = native.syn.g - previous
        torch.testing.assert_close(payload, native_payload)
        observed.append((index + 1) * payload.sum())
        expected.append((index + 1) * native_payload.sum())
    targets = (weights, values) + (() if delays is None else (delays,))
    native_targets = (native_weights, native_values) + (
        () if delays is None else (native_delays,)
    )
    actual = torch.autograd.grad(sum(observed), targets, allow_unused=True)
    oracle = torch.autograd.grad(sum(expected), native_targets, allow_unused=True)
    for value, reference in zip(actual, oracle, strict=True):
        if value is None or reference is None:
            assert value is reference
        else:
            torch.testing.assert_close(value, reference)
            assert torch.isfinite(value).all()
    assert actual[0].abs().sum() > 0
    assert (actual[1] is not None) is diff_spiking
    if delays is not None:
        assert actual[2].abs().sum() > 0
    _close(tensors.state, initial)


@pytest.mark.parametrize("diff_spiking", (False, True))
def test_history_projects_current_weights_and_delays_after_recording(diff_spiking):
    _, native = _netcon(mode="raw", diff_spiking=diff_spiking)
    plan, tensors = dn.func.make_functional_netcon(native)
    state = tensors.state
    for row in _tape(plan.source_shape, "raw")[:3]:
        state, _ = plan.step(tensors.weights, state, row, delays=tensors.delays)
        native.pre.gate = row
        native.advance()
    weights = (tensors.weights * 2.3).detach().requires_grad_()
    delays = torch.full_like(tensors.delays, 1.4 * DT).requires_grad_()
    native.weight.w = weights.detach().clone().requires_grad_()
    native.delay_ms.w = delays.detach().clone().requires_grad_()
    frozen = _clone(state)
    _, payload = plan.read_due(state, weights=weights, delays=delays)
    _close(state, frozen)
    previous = native.syn.g
    native.pre.gate = torch.zeros(plan.source_shape, dtype=DTYPE)
    native.advance()
    expected = native.syn.g - previous
    torch.testing.assert_close(payload, expected)
    actual_gradient = torch.autograd.grad(payload.sum(), (weights, delays))
    native_gradient = torch.autograd.grad(
        expected.sum(), (native.weight.w, native.delay_ms.w)
    )
    _close(actual_gradient, native_gradient)
    assert all(value.abs().sum() > 0 for value in actual_gradient)


@pytest.mark.parametrize("diff_spiking", (False, True))
def test_source_history_snapshot_commit_and_fresh_reset(diff_spiking):
    _, native = _netcon(diff_spiking=diff_spiking)
    plan, tensors = dn.func.make_functional_netcon(native)
    state = tensors.state
    for row in _tape(plan.source_shape, "threshold")[:5]:
        state, _ = plan.step(tensors.weights, state, row, delays=tensors.delays)
    before = _clone(state)
    plan.commit_state_(native, state)
    _close(plan.extract().state, state)
    row = _tape(plan.source_shape, "threshold")[5]
    actual, payload = plan.step(tensors.weights, state, row, delays=tensors.delays)
    native.pre.v = row
    previous = native.syn.g
    native.advance()
    _close(actual, plan.extract().state)
    torch.testing.assert_close(payload, native.syn.g - previous)
    _close(state, before)
    reset = plan.initial_state(state, t=2.75)
    native.t.fill_(2.75)
    native.initialize(reinit_weights=False, reinit_delays=False)
    # Reinitialization invalidates the old plan, so compare Tensor carry directly.
    _close(reset, _flat_checkpoint(native))
    assert reset["global_step"].item() == round(2.75 / DT)
    for name, value in reset.items():
        assert not value.requires_grad


@pytest.mark.parametrize("diff_spiking", (False, True))
@pytest.mark.parametrize("taps", (2, 3))
def test_fullgraph_history_projection_and_gradients(diff_spiking, taps):
    with torch_compiler_warning_context():
        _, native = _netcon(diff_spiking=diff_spiking, taps=taps, fanout=True)
        plan, tensors = dn.func.make_functional_netcon(native)
        compiled = torch.compile(plan.step, backend="aot_eager", fullgraph=True)
        weights = tensors.weights.detach().clone().requires_grad_()
        delays = tensors.delays.detach().clone().requires_grad_()
        values = _tape(plan.source_shape, "threshold").requires_grad_()
        actual, expected = tensors.state, tensors.state
        actual_loss, expected_loss = 0, 0
        for row in values:
            actual, payload = compiled(weights, actual, row, delays=delays)
            expected, reference = plan.step(weights, expected, row, delays=delays)
            _close(actual, expected)
            torch.testing.assert_close(payload, reference)
            actual_loss = actual_loss + payload.square().sum()
            expected_loss = expected_loss + reference.square().sum()
        targets = (weights, delays, values)
        actual_gradient = torch.autograd.grad(actual_loss, targets, allow_unused=True)
        expected_gradient = torch.autograd.grad(
            expected_loss, targets, allow_unused=True
        )
        for value, reference in zip(actual_gradient, expected_gradient, strict=True):
            if value is None or reference is None:
                assert value is reference
            else:
                torch.testing.assert_close(value, reference)


@pytest.mark.parametrize("lanes", (0, 3))
@pytest.mark.parametrize("diff_spiking", (False, True))
def test_vmap_history_values_delays_and_parameters(lanes, diff_spiking):
    _, native = _netcon(mode="raw", diff_spiking=diff_spiking)
    plan, tensors = dn.func.make_functional_netcon(native)
    weights = tensors.weights.expand(lanes, -1).clone()
    delays = tensors.delays.expand(lanes, -1).clone()
    values = _tape(plan.source_shape, "raw")[:lanes]
    actual = torch.vmap(
        lambda weight, delay, value: plan.step(
            weight, tensors.state, value, delays=delay
        )
    )(weights, delays, values)
    if lanes:
        references = [
            plan.step(weight, tensors.state, value, delays=delay)
            for weight, delay, value in zip(weights, delays, values, strict=True)
        ]
        expected = pytree.tree_map(lambda *rows: torch.stack(rows), *references)
        _close(actual, expected)
    else:
        assert actual[1].shape == (0, *plan.shape)


def test_source_history_validates_parameters_metadata_and_side_calendar_boundaries():
    _, native = _netcon()
    plan, tensors = dn.func.make_functional_netcon(native)
    with pytest.raises(TypeError, match="weights"):
        plan.read_due(tensors.state)
    with pytest.raises(TypeError, match="delays"):
        plan.read_due(tensors.state, weights=tensors.weights)
    native.bitpack_source_threshold.add_(0.5)
    with pytest.raises(dn.func.FunctionalizationError, match="routing|detector"):
        plan.extract()
    native.bitpack_source_threshold.sub_(0.5)
    native._sparse_calendar[0] = [(native.post_idx[:1], tensors.weights[:1])]
    with pytest.raises(dn.func.FunctionalizationError, match="side calendars"):
        plan.extract()


@pytest.mark.parametrize("diff_delays", (False, True))
def test_packed_source_history_crosses_native_word_boundary(diff_delays):
    _, native = _netcon(size=65, diff_spiking=False, taps=3, diff_delays=diff_delays)
    plan, tensors = dn.func.make_functional_netcon(native)
    assert tensors.state["spike_history_packed"].shape[1] == 2
    state = tensors.state
    values = torch.zeros((7, *plan.source_shape), dtype=DTYPE)
    values[0, 0, [0, 62, 63, 64]] = 1
    values[2, 0, [1, 62, 64]] = 1
    values[4, 0, [0, 63]] = 1
    for row in values:
        state, payload = plan.step(tensors.weights, state, row, delays=tensors.delays)
        native.pre.v = row
        previous = native.syn.g
        native.advance()
        _close(state, _flat_checkpoint(native))
        torch.testing.assert_close(payload, native.syn.g - previous)


def test_source_history_rejects_modified_native_transport_adapters():
    _, native = _netcon()
    plan, tensors = dn.func.make_functional_netcon(native)
    native._source_history_record_current_gate = lambda *args, **kwargs: None
    with pytest.raises(dn.func.FunctionalizationError, match="record_current_gate"):
        plan.step(
            tensors.weights,
            tensors.state,
            torch.zeros(plan.source_shape, dtype=DTYPE),
            delays=tensors.delays,
        )
    with pytest.raises(dn.func.FunctionalizationError, match="record_current_gate"):
        dn.func.make_functional_netcon(native)
    del native._source_history_record_current_gate
    native.advance = native.advance_diff
    with pytest.raises(dn.func.FunctionalizationError, match="transport adapter"):
        dn.func.make_functional_netcon(native)


@pytest.mark.parametrize("diff_spiking", (False, True))
def test_zero_connections_keep_explicit_empty_history_through_commit_and_reset(
    diff_spiking,
):
    network, template = _netcon(mode="raw")
    empty = torch.empty(0, dtype=DTYPE)
    storage = {"device": "cpu", "dtype": DTYPE}
    # Network builders omit zero-edge groups; direct construction admits the
    # same canonical native storage with no participating source positions.
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
        train_delay_backend="source_history",
        **storage,
    )
    native.setreference("t", lambda: network.t)
    native.set_diff_config(
        diff_weights=True,
        diff_delays=True,
        diff_spiking=diff_spiking,
        diff_scheduled_times=False,
        taps=3,
        train_delay_backend="source_history",
    )
    native.train()
    native.initialize()
    if diff_spiking:
        assert (
            "source_gate_history"
            not in native.state_dict_for_checkpoint()["backend_state"]
        )
    plan, tensors = dn.func.make_functional_netcon(native)
    history_name = "source_gate_history" if diff_spiking else "spike_history_packed"
    assert history_name in tensors.state
    assert tensors.state[history_name].shape[1] == (0 if diff_spiking else 1)
    initial = _clone(tensors.state)
    state, payload = plan.step(
        tensors.weights, tensors.state, template.pre.gate, delays=tensors.delays
    )
    torch.testing.assert_close(payload, torch.zeros_like(payload))
    native.advance()
    _close(state, plan.extract().state)
    plan.commit_state_(native, state)
    _close(state, plan.extract().state)
    reset = plan.initial_state(state, t=2.75)
    native.t.fill_(2.75)
    native.initialize(reinit_weights=False, reinit_delays=False)
    _, fresh = dn.func.make_functional_netcon(native)
    _close(reset, fresh.state)
    _close(tensors.state, initial)


def test_default_auto_source_history_and_scheduled_dense_fallback():
    _, native = _netcon(policy="auto")
    plan, _ = dn.func.make_functional_netcon(native)
    assert plan.runtime == "source_history"
    native.schedule(con_indices=[0], times_ms=[DT], weight=1.0)
    native.initialize(reinit_weights=False, reinit_delays=False)
    assert native._checkpoint_runtime_kind() == "dense"
    dense, tensors = dn.func.make_functional_netcon(native)
    assert dense.runtime == "dense"
    dense.step(
        tensors.weights,
        tensors.state,
        torch.zeros(dense.source_shape, dtype=DTYPE),
        delays=tensors.delays,
        schedule=tensors.schedule,
    )


def test_explicit_weight_leaf_controls_gradients_even_if_native_flag_is_disabled():
    _, native = _netcon(mode="raw", diff_weights=False)
    plan, tensors = dn.func.make_functional_netcon(native)
    state, _ = plan.step(
        tensors.weights,
        tensors.state,
        torch.ones(plan.source_shape, dtype=DTYPE),
        delays=tensors.delays,
    )
    weights = tensors.weights.detach().clone().requires_grad_()
    _, payload = plan.read_due(state, weights=weights, delays=tensors.delays)
    gradient = torch.autograd.grad(payload.sum(), weights)[0]
    assert gradient.abs().sum() > 0
