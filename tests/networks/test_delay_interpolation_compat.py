"""Freeze dense interpolation against a scalar, independent transport oracle.

The scalar reference deliberately uses neither NetCon helpers nor vectorized
scatter indices. Offset-zero and beyond-horizon writes retain the native
circular-buffer behavior. A separate low-precision oracle freezes the former
indexed accumulation algorithm, including its compiled arithmetic.
"""

from __future__ import annotations

import math

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn
from dendra.models.networks._event_delivery import dense_interpolated_delay_transition

DT = 0.1
SIGMA = 0.43
pytestmark = pytest.mark.cpu


def _native(*, taps, dtype=torch.float64, count=8):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        post = dn.Population(N=1, C=count, dtype=dtype)
        post.insert(expsyn.rename("syn"))
        source = dn.NetStim(N=count, dtype=dtype)
        network = dn.Network(
            {"post": post},
            netstim=source,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
            track_netcon_events=False,
        )
        network.connect_one_to_one(
            source[:],
            post[:],
            post.mech.syn,
            threshold=None,
            weight=torch.ones(count, dtype=dtype),
            delay=torch.full((count,), 0.4, dtype=dtype),
        )
        network.build(DT)
        network.init_synapses()
        connection = next(iter(network.synapses.values()))
        connection.train()
        connection.set_diff_config(
            diff_weights=True,
            diff_delays=True,
            diff_spiking=True,
            diff_scheduled_times=False,
            taps=taps,
            sigma=SIGMA,
            train_delay_backend="dense",
        )
    observed = []
    connection.syn.net_receive = lambda payload, _: observed.append(
        payload.reshape(-1).clone()
    )
    return connection, observed


def _scalar_step(queue, slot, weights, delays, gate, post, *, dt, taps):
    """Old dense algorithm expressed as per-connection, per-tap updates."""
    depth, width = queue.shape
    due = queue[slot]
    cells = [
        [queue[row, column] * (row != slot) for column in range(width)]
        for row in range(depth)
    ]
    for edge in range(weights.numel()):
        distance = (delays[edge] / dt).clamp_min(1)
        lower = math.floor(float(distance.detach()))
        if taps == 2:
            fraction = distance - lower
            offsets = (lower, lower + 1)
            coefficients = (1 - fraction, fraction)
        else:
            offsets = (lower - 1, lower, lower + 1)
            logs = [
                -0.5 * ((distance - offset) / (SIGMA + 1e-6)) ** 2 for offset in offsets
            ]
            coefficients = torch.stack(logs).softmax(0).unbind()
        target = int(post[edge])
        for offset, coefficient in zip(offsets, coefficients, strict=True):
            row = (slot + offset) % depth
            cells[row][target] = (
                cells[row][target] + weights[edge] * gate[edge] * coefficient
            )
    return torch.stack([torch.stack(row) for row in cells]), (slot + 1) % depth, due


def _previous_indexed_step(queue, slot, weights, gates, delays, dt, post, *, taps):
    """Frozen native vectorized algorithm, independent of shared helpers."""
    depth, width = queue.shape
    output = queue.clone()
    output.index_fill_(0, slot, 0)
    flat = output.flatten()
    distance = (delays / dt.to(queue.dtype)).clamp_min(1)
    lower = distance.floor()
    lower_index = lower.long()
    weighted = weights * gates
    if taps == 2:
        fraction = distance - lower
        row0 = (slot + lower_index).remainder(depth)
        row1 = (row0 + 1).remainder(depth)
        flat.index_add_(0, row0 * width + post, weighted * (1 - fraction))
        flat.index_add_(0, row1 * width + post, weighted * fraction)
    else:
        offsets = torch.stack((lower_index - 1, lower_index, lower_index + 1), dim=-1)
        coefficients = (
            -0.5
            * ((distance.unsqueeze(-1) - offsets.to(queue.dtype)) / (SIGMA + 1e-6)) ** 2
        ).softmax(-1)
        destination = (slot + offsets).remainder(depth) * width + post.unsqueeze(-1)
        values = weighted.unsqueeze(-1) * coefficients
        flat.index_add_(0, destination.flatten(), values.flatten())
    return output


@pytest.mark.parametrize("taps", (2, 3))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("backend", (None, "aot_eager", "inductor"))
def test_low_precision_collision_sums_keep_native_indexed_accumulation(
    taps, dtype, backend
):
    connection, _ = _native(taps=taps, dtype=dtype, count=1000)
    connection.post_idx.zero_()
    weights = torch.randn(
        1000, dtype=dtype, generator=torch.Generator().manual_seed(72)
    )
    delays = torch.full_like(weights, 0.17)
    gates = torch.ones_like(weights)
    connection.weight.w = weights
    connection.delay_ms.w = delays
    connection.pre.spike_gate = gates
    initial = connection.delivery_buffer.clone()
    depth, width = initial.shape
    # Freeze the old native index_add algorithm: scatter_add promotes CPU
    # collision sums and gives observably different low-precision results.
    reference_args = (
        initial,
        connection.current_time_step,
        weights,
        gates,
        delays,
        connection.dt,
        connection.post_idx,
    )
    native_expected = _previous_indexed_step(*reference_args, taps=taps)
    expected = native_expected
    transition = dense_interpolated_delay_transition
    with torch_compiler_warning_context():
        if backend is not None:
            options = {"compile_threads": 1} if backend == "inductor" else None
            # Inductor fuses reduced-precision arithmetic differently from
            # eager. Compare the old and new algorithms in the same backend.
            reference = torch.compile(
                _previous_indexed_step, fullgraph=True, backend=backend, options=options
            )
            expected = reference(*reference_args, taps=taps)
            transition = torch.compile(
                transition, fullgraph=True, backend=backend, options=options
            )
        pure = transition(
            initial,
            connection.current_time_step,
            connection.global_step,
            weights,
            gates,
            delays,
            connection.dt,
            post_idx=connection.post_idx,
            depth=depth,
            width=width,
            taps=taps,
            sigma=SIGMA,
        )
    connection.advance()
    torch.testing.assert_close(pure.queue, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        connection.delivery_buffer, native_expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("taps", (2, 3))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_native_interpolation_matches_independent_values_and_all_gradients(taps, dtype):
    connection, observed = _native(taps=taps, dtype=dtype)
    # Include collisions, an initial nonzero ring slot, sub-step boundaries,
    # and effective delays beyond the fixed allocation's horizon.
    connection.post_idx = torch.arange(8) % 3
    connection.current_time_step.fill_(connection.max_delay_steps - 1)
    slot = int(connection.current_time_step)
    delays = torch.tensor(
        [-0.1, 0, 0.05, 0.1, 0.15, 0.2, 0.39, 0.61], dtype=dtype
    ).requires_grad_()
    weights = torch.linspace(-0.7, 1.3, 8, dtype=dtype).requires_grad_()
    gates = torch.linspace(-0.4, 1.2, 8 * 9, dtype=dtype).reshape(9, 8)
    gates.requires_grad_()
    queue = torch.linspace(
        -0.2, 0.3, connection.delivery_buffer.numel(), dtype=dtype
    ).reshape_as(connection.delivery_buffer)
    queue.requires_grad_()
    connection.delivery_buffer = queue
    reference_inputs = [
        value.detach().clone().requires_grad_()
        for value in (weights, delays, gates, queue)
    ]
    ref_weights, ref_delays, ref_gates, reference = reference_inputs
    expected_due = []
    for step in range(gates.shape[0]):
        # Optimizer-like changes affect new entries; pending values stay intact.
        connection.weight.w = weights * (1 + step * 0.13)
        connection.delay_ms.w = delays + step * 0.013
        connection.pre.spike_gate = gates[step]
        connection.advance()
        reference, slot, due = _scalar_step(
            reference,
            slot,
            ref_weights * (1 + step * 0.13),
            ref_delays + step * 0.013,
            ref_gates[step],
            connection.post_idx,
            dt=connection.dt.to(dtype),
            taps=taps,
        )
        expected_due.append(due)
        torch.testing.assert_close(connection.delivery_buffer, reference)
        torch.testing.assert_close(observed[-1], due)
        assert int(connection.current_time_step) == slot
        assert int(connection.global_step) == step + 1
    actual_trace = torch.stack(observed)
    reference_trace = torch.stack(expected_due)
    coefficients = torch.linspace(0.1, 1.9, actual_trace.numel(), dtype=dtype)
    coefficients = coefficients.reshape_as(actual_trace)
    actual_loss = (actual_trace * coefficients).sum() + (
        connection.delivery_buffer.square().sum()
    )
    expected_loss = (reference_trace * coefficients).sum() + reference.square().sum()
    gradients = torch.autograd.grad(actual_loss, (weights, delays, gates, queue))
    expected_gradients = torch.autograd.grad(expected_loss, reference_inputs)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    # This entry remains strictly sub-dt even after all parameter changes.
    torch.testing.assert_close(gradients[1][:1], torch.zeros(1, dtype=dtype))


@pytest.mark.parametrize("taps", (2, 3))
def test_substep_left_tap_retains_native_ring_wrap_policy(taps):
    connection, observed = _native(taps=taps, count=1)
    connection.delay_ms.w = torch.tensor([0.03], dtype=connection.dtype)
    connection.weight.w = torch.tensor([2.0], dtype=connection.dtype)
    for step in range(connection.max_delay_steps + 1):
        connection.pre.spike_gate = torch.tensor([step == 0], dtype=connection.dtype)
        connection.advance()
    trace = torch.cat(observed)
    if taps == 2:
        expected = torch.zeros_like(trace)
        expected[1] = 2
    else:
        # The left tap has offset zero. Native reads before enqueueing, so
        # that component is only observable after the ring wraps completely.
        centers = trace.new_tensor([0, 1, 2])
        coefficients = (-0.5 * ((1 - centers) / (SIGMA + 1e-6)) ** 2).softmax(0)
        expected = torch.zeros_like(trace)
        expected[1] = 2 * coefficients[1]
        expected[2] = 2 * coefficients[2]
        expected[connection.max_delay_steps] = 2 * coefficients[0]
    torch.testing.assert_close(trace, expected)


@pytest.mark.parametrize("taps", (2, 3))
def test_runtime_checkpoint_keeps_pending_snapshot_and_upstream_graph(taps):
    connection, observed = _native(taps=taps, count=2)
    weights = torch.tensor([1.5, 0.7], dtype=connection.dtype, requires_grad=True)
    delays = torch.tensor([0.17, 0.26], dtype=connection.dtype, requires_grad=True)
    gate = torch.tensor([0.3, -0.8], dtype=connection.dtype, requires_grad=True)
    connection.weight.w = weights
    connection.delay_ms.w = delays
    connection.pre.spike_gate = gate
    connection.advance()
    checkpoint = connection.state_dict_for_checkpoint()
    frozen = {name: value.detach().clone() for name, value in checkpoint.items()}
    assert checkpoint["delivery_buffer"] is connection.delivery_buffer

    def suffix(native, output):
        # Only pending payloads remain: current values must not retime or
        # rescale entries created before this optimizer-like change.
        native.weight.w = weights * 100
        native.delay_ms.w = delays + 0.31
        for _ in range(native.max_delay_steps + 1):
            native.pre.spike_gate = torch.zeros_like(gate)
            native.advance()
        return torch.stack(output[-(native.max_delay_steps + 1) :])

    uninterrupted = suffix(connection, observed)
    restored, restored_observed = _native(taps=taps, count=2)
    restored.restore_dict_from_checkpoint(checkpoint)
    assert restored.delivery_buffer is checkpoint["delivery_buffer"]
    resumed = suffix(restored, restored_observed)
    torch.testing.assert_close(resumed, uninterrupted)
    for name, value in checkpoint.items():
        torch.testing.assert_close(value, frozen[name])
    coefficients = resumed.new_tensor([0.2, 1.1, -0.6, 0.8, 1.4, 0.3]).unsqueeze(1)
    actual_gradients = torch.autograd.grad(
        (resumed * coefficients).sum(), (weights, delays, gate), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        (uninterrupted * coefficients).sum(), (weights, delays, gate)
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
