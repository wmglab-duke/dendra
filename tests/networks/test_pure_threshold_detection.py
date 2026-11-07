"""NetCon detector semantics are shared across pure and native execution."""

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn
from dendra.models.networks.netcon import NetCon
from dendra.models.networks.spiking import threshold_transition

DTYPE = torch.float64


@pytest.mark.parametrize("diff_spiking", (False, True))
def test_initial_above_equality_rearming_and_nan_observations(diff_spiking):
    values = torch.tensor(
        [[0.0, 1.0, -1.0], [1.0, 0.0, 0.0], [-1.0, float("nan"), 1.0], [0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    latch = torch.zeros(3, dtype=torch.bool)
    expected_latches = (
        [True, True, False],
        [True, True, True],
        [False, False, True],
        [True, True, True],
    )
    expected_gates = ([1, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 0])
    for row, expected_latch, expected_gate in zip(
        values, expected_latches, expected_gates
    ):
        input_latch = latch
        previous = latch.clone()
        latch, gates = threshold_transition(
            row, latch, torch.zeros(3, dtype=DTYPE), diff_spiking=diff_spiking
        )
        assert latch.tolist() == expected_latch
        assert latch.dtype == torch.bool
        assert not latch.requires_grad
        # The existing sigmoid surrogate propagates an observed NaN even when
        # the hard comparison is false. Preserve that native behavior as well.
        if diff_spiking and torch.isnan(row).any():
            assert torch.isnan(gates[1])
            gates = gates.nan_to_num()
        torch.testing.assert_close(gates.to(DTYPE), row.new_tensor(expected_gate))
        torch.testing.assert_close(input_latch, previous)
        assert gates.dtype == (DTYPE if diff_spiking else torch.bool)


@pytest.mark.parametrize("diff_spiking", (False, True))
def test_mixed_raw_lanes_preserve_values_and_native_latch(diff_spiking):
    values = torch.tensor([-0.2, 0.0, 0.2, 1.5, -2.0], dtype=DTYPE, requires_grad=True)
    previous = torch.tensor([False, False, True, True, False])
    thresholds = torch.zeros_like(values)
    mask = torch.tensor([False, False, False, True, True])
    originals = [x.clone() for x in (values, previous, thresholds, mask)]
    latch, gates = threshold_transition(
        values, previous, thresholds, diff_spiking=diff_spiking, tau=0.5, nan_mask=mask
    )
    assert latch.tolist() == [False, True, True, True, False]
    torch.testing.assert_close(gates, values.new_tensor([0, 1, 0, 1.5, -2.0]))
    gradient = torch.autograd.grad(gates.sum(), values)[0]
    expected = torch.tensor([0, 0, 0, 1, 1], dtype=DTYPE)
    if diff_spiking:
        sigmoid = torch.sigmoid(values.detach()[:3] / 0.5)
        expected[:3] = sigmoid * (1 - sigmoid) / 0.5 * ~previous[:3]
    torch.testing.assert_close(gradient, expected)
    for value, original in zip((values, previous, thresholds, mask), originals):
        torch.testing.assert_close(value, original)


def test_surrogate_gradient_uses_current_value_and_previous_boolean_only():
    earlier = torch.tensor([-0.5, 0.5], dtype=DTYPE, requires_grad=True)
    current = torch.tensor([0.25, 0.25], dtype=DTYPE, requires_grad=True)
    thresholds = torch.tensor([0.0, 0.1], dtype=DTYPE, requires_grad=True)
    first_latch, _ = threshold_transition(
        earlier, torch.zeros(2, dtype=torch.bool), thresholds, diff_spiking=True
    )
    next_latch, gates = threshold_transition(
        current, first_latch, thresholds, diff_spiking=True, tau=0.5
    )
    assert next_latch.tolist() == [True, True]
    assert gates.tolist() == [1.0, 0.0]
    earlier_grad, value_grad, threshold_grad = torch.autograd.grad(
        gates.sum(), (earlier, current, thresholds), allow_unused=True
    )
    assert earlier_grad is None
    sigmoid = torch.sigmoid((current.detach() - thresholds.detach()) / 0.5)
    expected = sigmoid * (1 - sigmoid) / 0.5 * ~first_latch
    torch.testing.assert_close(value_grad, expected)
    torch.testing.assert_close(threshold_grad, -expected)
    jacrev = torch.func.jacrev(
        lambda x: threshold_transition(
            x, first_latch, thresholds, diff_spiking=True, tau=0.5
        )[1]
    )(current)
    jacfwd = torch.func.jacfwd(
        lambda x: threshold_transition(
            x, first_latch, thresholds, diff_spiking=True, tau=0.5
        )[1]
    )(current)
    torch.testing.assert_close(jacrev, jacfwd)
    torch.testing.assert_close(jacrev, torch.diag(expected))


@pytest.mark.parametrize("lanes", (0, 3))
@pytest.mark.parametrize("diff_spiking", (False, True))
@pytest.mark.parametrize("mapped", ("values", "latch", "thresholds"))
def test_vmap_preserves_hidden_axes_and_empty_lanes(lanes, diff_spiking, mapped):
    values = torch.tensor([-0.1, 0.0, 0.5], dtype=DTYPE)
    latch = torch.tensor([False, True, False])
    thresholds = torch.zeros_like(values)
    mask = torch.tensor([False, False, True])
    base = {"values": values, "latch": latch, "thresholds": thresholds}[mapped]
    batch = base.expand(lanes, *base.shape).clone()

    def transition(operand):
        return threshold_transition(
            operand if mapped == "values" else values,
            operand if mapped == "latch" else latch,
            operand if mapped == "thresholds" else thresholds,
            diff_spiking=diff_spiking,
            nan_mask=mask,
        )

    actual = torch.vmap(transition)(batch)
    if lanes:
        expected = tuple(
            torch.stack([transition(row)[part] for row in batch]) for part in (0, 1)
        )
    else:
        expected = (latch.new_empty((0, 3)), values.new_empty((0, 3)))
    for result, oracle in zip(actual, expected):
        torch.testing.assert_close(result, oracle)


@pytest.mark.parametrize("diff_spiking", (False, True))
@pytest.mark.parametrize("mixed", (False, True))
def test_fullgraph_compilation_preserves_values_carry_and_gradients(
    diff_spiking, mixed
):
    values = torch.tensor([-0.2, 0.0, 0.3], dtype=DTYPE, requires_grad=True)
    latch = torch.tensor([False, False, True])
    thresholds = torch.tensor([-0.1, 0.0, 0.2], dtype=DTYPE)
    mask = torch.tensor([False, True, False]) if mixed else None

    def transition(value):
        return threshold_transition(
            value, latch, thresholds, diff_spiking=diff_spiking, nan_mask=mask
        )

    actual = torch.compile(transition, backend="aot_eager", fullgraph=True)(values)
    expected = transition(values)
    for result, oracle in zip(actual, expected):
        torch.testing.assert_close(result, oracle)
    if diff_spiking or mixed:
        actual_grad = torch.autograd.grad(actual[1].sum(), values)[0]
        expected_grad = torch.autograd.grad(expected[1].sum(), values)[0]
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("diff_spiking", (False, True))
def test_native_per_edge_thresholds_and_mixed_raw_fanout(diff_spiking):
    source = SimpleNamespace(
        v=torch.tensor([[0.25], [1.5]], dtype=DTYPE, requires_grad=True)
    )
    connection = SimpleNamespace(
        get_pre_var=lambda pre: pre.v,
        pre_idx=torch.tensor([0, 0, 1, 1]),
        pre_device=torch.device("cpu"),
        pre_dtype=DTYPE,
        skip_thresholding=False,
        apply_masking=True,
        threshold=torch.tensor([0.0, 0.5, 0.0, 2.0], dtype=DTYPE),
        thresh_is_nan=torch.tensor([False, False, True, False]),
        has_spiked=torch.zeros(4, dtype=torch.bool),
    )
    NetCon.determine_spiking_var(connection, source, diff_spiking=diff_spiking, tau=0.5)
    torch.testing.assert_close(
        connection.is_spiking, source.v.new_tensor([1, 0, 1.5, 0])
    )
    assert connection.has_spiked.tolist() == [True, False, True, False]
    NetCon.determine_spiking_var(connection, source, diff_spiking=diff_spiking, tau=0.5)
    torch.testing.assert_close(
        connection.is_spiking, source.v.new_tensor([0, 0, 1.5, 0])
    )
    # A previously above edge suppresses its surrogate derivative; the same
    # source still differentiates through another threshold and its raw edge.
    if diff_spiking:
        gradient = torch.autograd.grad(connection.is_spiking.sum(), source.v)[0]
        sig = torch.sigmoid(source.v.detach().flatten().new_tensor([-0.25, -0.5]) / 0.5)
        expected = sig * (1 - sig) / 0.5
        expected[1] += 1
        torch.testing.assert_close(gradient.flatten(), expected)


@pytest.mark.parametrize(
    "runtime",
    (
        "dense",
        "sparse_calendar",
        "bitpacked_history",
        "source_history_hard",
        "source_history_diff",
    ),
)
def test_native_backends_follow_independent_threshold_delivery_ledger(runtime):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu"):
        pre = dn.Population(N=2, C=1, dtype=DTYPE)
        post = dn.Population(N=2, C=1, dtype=DTYPE)
        post.insert(expsyn.rename("syn"))
        backend = runtime if not runtime.startswith("source_history") else "dense"
        net = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend=backend,
            netcon_train_backend="source_history",
            track_netcon_events=False,
        )
        weights = torch.tensor([1, 2, 3, 4], dtype=DTYPE)
        net.connect_dense(
            pre[:],
            post[:],
            post.mech.syn,
            threshold=torch.tensor([0, 0, 1, 1], dtype=DTYPE),
            weight=weights,
            delay=torch.tensor([0.1, 0.2, 0.2, 0.1], dtype=DTYPE),
        )
        net.build(0.1)
        net.init_synapses()
        connection = next(iter(net.synapses.values()))
        if runtime.startswith("source_history"):
            connection.train()
            connection.set_diff_config(
                diff_weights=False,
                diff_delays=False,
                diff_spiking=runtime.endswith("diff"),
                diff_scheduled_times=False,
                train_delay_backend="source_history",
            )
            connection.initialize(reinit_weights=False, reinit_delays=False)
    tape = (
        (0.0, 2.0),
        (0.5, 1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (-1.0, 0.0),
        (-1.0, 0.0),
    )
    latch = [False, False]
    pending = {}
    state = torch.zeros(2, dtype=DTYPE)
    edge_sources, edge_targets, edge_delays = (0, 0, 1, 1), (0, 1, 0, 1), (1, 2, 2, 1)
    for step, observed in enumerate(tape):
        state = state + pending.pop(step, torch.zeros_like(state))
        net.pre.v.copy_(torch.tensor(observed, dtype=DTYPE).reshape(2, 1))
        connection.advance()
        torch.testing.assert_close(connection.syn.g.flatten(), state)
        above = [observed[0] >= 0.0, observed[1] >= 1.0]
        rising = [current and not previous for current, previous in zip(above, latch)]
        for edge, (source, target, delay) in enumerate(
            zip(edge_sources, edge_targets, edge_delays)
        ):
            if rising[source]:
                due = pending.setdefault(step + delay, torch.zeros_like(state))
                due[target] += weights[edge]
        latch = above
        runtime_latch = (
            connection.bitpack_source_has_spiked
            if runtime
            in ("bitpacked_history", "source_history_hard", "source_history_diff")
            else connection.has_spiked
        )
        assert runtime_latch.tolist() == (
            latch
            if runtime.startswith(("bitpacked", "source_history"))
            else [latch[0], latch[0], latch[1], latch[1]]
        )


def _configured_connection(*, backend="dense", tracked=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu"):
        pre = dn.Population(N=1, C=1, dtype=DTYPE)
        post = dn.Population(N=1, C=1, dtype=DTYPE)
        post.insert(expsyn.rename("syn"))
        network = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend=backend,
            netcon_train_backend="dense",
            track_netcon_events=tracked,
        )
        network.connect_one_to_one(pre[:], post[:], post.mech.syn, threshold=0.0)
        network.build(0.1)
        network.init_synapses()
    return next(iter(network.synapses.values()))


@pytest.mark.parametrize("pending_initialize", (False, True))
def test_successful_config_calls_invalidate_plan_generation_even_for_equal_values(
    pending_initialize,
):
    connection = _configured_connection(
        backend="sparse_calendar" if pending_initialize else "dense"
    )
    connection.train()
    assert connection._mode_requires_initialize is pending_initialize
    kwargs = {
        "diff_weights": True,
        "diff_delays": False,
        "diff_spiking": True,
        "diff_scheduled_times": False,
        "tau": 0.12,
        "train_delay_backend": "dense",
    }
    for _ in range(2):
        generation = connection._event_delivery_generation
        connection.set_diff_config(**kwargs)
        assert connection._event_delivery_generation == generation + 1
        assert connection._mode_requires_initialize is pending_initialize
    generation = connection._event_delivery_generation
    # Native execution permits a Tensor temperature, while the pure facade
    # freezes Python scalar metadata. Equal numeric values must still invalidate
    # a warm plan when the public setter changes their representation.
    connection.set_diff_config(**{**kwargs, "tau": torch.tensor(0.12)})
    assert connection._event_delivery_generation == generation + 1


@pytest.mark.parametrize("failure", ("backend", "tracking", "source_policy", "refresh"))
def test_failed_config_keeps_generation_and_restores_flags(failure, monkeypatch):
    connection = _configured_connection(tracked=failure == "tracking")
    connection.train()
    kwargs = {
        "diff_weights": True,
        "diff_delays": False,
        "diff_spiking": False,
        "diff_scheduled_times": False,
        "train_delay_backend": "dense",
    }
    connection.set_diff_config(**kwargs)
    if failure == "backend":
        kwargs["train_delay_backend"] = "invalid"
    elif failure == "tracking":
        kwargs["diff_spiking"] = True
    elif failure == "source_policy":
        connection.schedule(con_indices=[0], times_ms=[0.0])
        kwargs["train_delay_backend"] = "source_history"
    else:

        def fail_refresh(**kwargs):
            raise RuntimeError("storage refresh failed")

        monkeypatch.setattr(
            connection, "_refresh_training_advance_after_diff_config", fail_refresh
        )
    generation = connection._event_delivery_generation
    previous_flags = connection.train_flags
    previous_backend = connection.train_delay_backend
    with pytest.raises((ValueError, RuntimeError)):
        connection.set_diff_config(**kwargs)
    assert connection._event_delivery_generation == generation
    assert connection.train_flags is previous_flags
    assert connection.train_delay_backend == previous_backend
