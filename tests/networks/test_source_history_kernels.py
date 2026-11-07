"""Shared source-history kernels retain delivery-time and tape semantics."""

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.models.networks._source_history import (
    read_packed_history,
    source_history_gates,
    source_history_project,
    source_history_record,
)
from dendra.models.networks.netcon import NetCon
from dendra.models.networks.spiking import threshold_transition

pytestmark = pytest.mark.cpu
DTYPE = torch.float64


def _packing(size):
    positions = torch.arange(size)
    return positions.div(63, rounding_mode="floor"), 1 << positions.remainder(63)


@pytest.mark.parametrize("mode", ("netstim", "pre_var", "threshold"))
@pytest.mark.parametrize("differentiable", (False, True))
def test_unique_source_observation_preserves_native_hard_and_surrogate_modes(
    mode, differentiable
):
    values = torch.tensor([-1.0, 0.0, 0.25, 2.0], dtype=DTYPE, requires_grad=True)
    indices = torch.tensor([0, 2, 3])
    latch = torch.tensor([False, False, True])
    thresholds = torch.tensor([0.0, 0.25, 0.5], dtype=DTYPE)
    actual_latch, gates = source_history_gates(
        values,
        latch,
        source_idx=indices,
        mode=mode,
        threshold=thresholds,
        diff_spiking=differentiable,
        tau=0.2,
        dtype=DTYPE,
    )
    selected = values[indices]
    if mode == "threshold":
        expected_latch, expected = threshold_transition(
            selected,
            latch,
            thresholds,
            diff_spiking=differentiable,
            tau=0.2,
        )
    else:
        expected_latch = latch
        expected = (
            selected
            if mode == "netstim" or differentiable
            else (selected > 0).to(DTYPE)
        )
    torch.testing.assert_close(actual_latch, expected_latch)
    torch.testing.assert_close(gates, expected)
    if gates.requires_grad:
        observed = torch.autograd.grad(gates.sum(), values, retain_graph=True)[0]
        reference = torch.autograd.grad(expected.sum(), values)[0]
        torch.testing.assert_close(observed, reference)


@pytest.mark.parametrize("pre_dtype", (torch.float16, torch.float32))
def test_native_netstim_gates_cast_directly_to_delivery_dtype(pre_dtype):
    stimulus = dn.NetStim(N=3, dtype=DTYPE)
    values = torch.tensor(
        [0.123456789, 0.234567891, 0.345678912], dtype=DTYPE, requires_grad=True
    )
    stimulus.spike_gate = values
    indices = torch.tensor([0, 2])
    connection = SimpleNamespace(
        pre=stimulus,
        pre_device=torch.device("cpu"),
        pre_dtype=pre_dtype,
        device=torch.device("cpu"),
        dtype=DTYPE,
        bitpack_source_pre_idx=indices,
        bitpack_source_has_spiked=torch.empty(0, dtype=torch.bool),
        bitpack_source_threshold=torch.empty(0, dtype=pre_dtype),
        _bitpack_mode="netstim",
    )
    observed = NetCon._source_history_source_gate_this_step(
        connection, diff_spiking=True, tau=0.1
    )
    # NetStim outputs bypass the Population detector dtype, matching the native
    # direct source-to-delivery cast even with an explicit pre_dtype override.
    expected = values.index_select(0, indices).to(DTYPE)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(observed.square().sum(), values)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), values)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ("netstim", "pre_var", "threshold"))
@pytest.mark.parametrize("boolean", (False, True))
@pytest.mark.parametrize("pre_dtype", (torch.float16, torch.float32))
def test_native_hard_observation_preserves_source_casts_and_detector_latches(
    mode, boolean, pre_dtype
):
    values = torch.tensor([1e-9, -0.1, 0.3, 1.0], dtype=DTYPE)
    if boolean:
        values = values > 0
    indices = torch.tensor([0, 2, 3])
    previous = torch.tensor([False, True, False])
    thresholds = torch.tensor([0.0, 0.5, 1.0], dtype=pre_dtype)
    connection = SimpleNamespace(
        pre=SimpleNamespace(spikes=values),
        get_pre_var=lambda _: values,
        pre_device=torch.device("cpu"),
        pre_dtype=pre_dtype,
        bitpack_source_pre_idx=indices,
        bitpack_source_has_spiked=previous.clone(),
        bitpack_source_threshold=thresholds,
        _bitpack_mode=mode,
    )
    selected = values.index_select(0, indices)
    expected_latch = previous
    if mode == "netstim":
        # NetStim is an already generated hard event, even if stored numerically.
        expected = selected.to(torch.bool)
    elif mode == "pre_var":
        expected = selected if boolean else selected.to(pre_dtype) > 0
    else:
        expected_latch, expected = threshold_transition(
            selected.to(pre_dtype), previous, thresholds
        )
    actual = NetCon._bitpack_source_spikes_this_step(connection)
    assert actual.dtype == torch.bool
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(connection.bitpack_source_has_spiked, expected_latch)
    torch.testing.assert_close(previous, torch.tensor([False, True, False]))


@pytest.mark.parametrize("mode", ("netstim", "pre_var", "threshold"))
@pytest.mark.parametrize("boolean", (False, True))
def test_binary_observation_preserves_hard_semantics_without_float_event_results(
    mode, boolean
):
    values = torch.tensor([1e-9, -0.25, 0.5, 1.5], dtype=DTYPE)
    if boolean:
        values = values > 0
    indices = torch.tensor([0, 1, 3])
    latch = torch.tensor([False, True, False])
    threshold = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float16)
    updated, gates = source_history_gates(
        values,
        latch,
        source_idx=indices,
        mode=mode,
        threshold=threshold,
        dtype=torch.float16,
        binary=True,
    )
    selected = values[indices]
    expected_latch = latch
    if mode == "netstim":
        expected = selected.to(torch.bool)
    elif mode == "pre_var":
        expected = selected if boolean else selected.to(torch.float16) > 0
    else:
        expected_latch, expected = threshold_transition(
            selected.to(torch.float16), latch, threshold
        )
    assert gates.dtype == torch.bool
    torch.testing.assert_close(gates, expected)
    torch.testing.assert_close(updated, expected_latch)
    torch.testing.assert_close(latch, torch.tensor([False, True, False]))
    with pytest.raises(ValueError, match="binary.*surrogate"):
        source_history_gates(
            values,
            latch,
            source_idx=indices,
            mode=mode,
            threshold=threshold,
            diff_spiking=True,
            binary=True,
        )


def test_packed_record_overwrites_rows_and_crosses_word_boundaries_without_mutation():
    size, depth = 130, 4
    word_idx, bit_mask = _packing(size)
    history = torch.full((depth, 3), 17, dtype=torch.int64)
    gates = torch.zeros(size, dtype=DTYPE)
    active = [0, 61, 62, 63, 64, 125, 126, 129]
    gates[active] = 1
    slot = torch.tensor([2])
    result = source_history_record(
        history,
        slot,
        gates,
        source_word_idx=word_idx,
        source_bit_mask=bit_mask,
    )
    expected = history.clone()
    expected[2] = torch.tensor(
        [
            sum(1 << (value % 63) for value in active if value // 63 == word)
            for word in range(3)
        ]
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(history, torch.full_like(history, 17))
    observed = read_packed_history(
        result,
        slot.expand(size),
        word_idx,
        bit_mask,
        dtype=DTYPE,
    )
    torch.testing.assert_close(observed, gates)
    owned = history.clone()
    address = owned.data_ptr()
    donated = source_history_record(
        owned,
        slot,
        gates,
        source_word_idx=word_idx,
        source_bit_mask=bit_mask,
        _inplace=True,
    )
    assert donated.data_ptr() == address
    torch.testing.assert_close(donated, result)


def test_float_record_preserves_only_unwritten_history_gradients():
    history = torch.arange(15, dtype=DTYPE).reshape(5, 3).requires_grad_()
    gates = torch.tensor([-0.2, 0.0, 0.8], dtype=DTYPE, requires_grad=True)
    slot = torch.tensor([2])
    for ownership in ({}, {"_clone_once": True}):
        updated = source_history_record(history, slot, gates, **ownership)
        expected = history.detach().clone()
        expected[2] = gates.detach()
        torch.testing.assert_close(updated, expected)
        gradients = torch.autograd.grad(updated.sum(), (history, gates))
        mask = torch.ones_like(history)
        mask[2] = 0
        torch.testing.assert_close(gradients[0], mask)
        torch.testing.assert_close(gradients[1], torch.ones_like(gates))
        torch.testing.assert_close(history, torch.arange(15, dtype=DTYPE).reshape(5, 3))


def _fixture(*, packed=False, dtype=DTYPE):
    depth = 5
    source_pos = torch.tensor([0, 1, 2, 0, 2])
    post_idx = torch.tensor([0, 0, 1, 1, 0])
    values = torch.tensor(
        [[0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=dtype
    )
    history = values
    kwargs = {"source_pos": source_pos}
    if packed:
        history = torch.zeros((depth, 1), dtype=torch.int64)
        word_idx, bit_mask = _packing(3)
        for slot in range(depth):
            history = source_history_record(
                history,
                torch.tensor([slot]),
                values[slot],
                source_word_idx=word_idx,
                source_bit_mask=bit_mask,
            )
        kwargs = {"word_idx": word_idx[source_pos], "bit_mask": bit_mask[source_pos]}
    return history, values, source_pos, post_idx, kwargs


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, DTYPE))
@pytest.mark.parametrize("empty", (False, True))
def test_native_packed_projection_reuses_scratch_and_clears_failed_launches(
    dtype, empty
):
    history, _, _, post_idx, metadata = _fixture(packed=True, dtype=dtype)
    weights = torch.tensor([0.7, -0.2, 1.3, 0.5, 2.1], dtype=dtype)
    fixed = torch.tensor([1, 2, 3, 4, 1])
    if empty:
        weights, post_idx, fixed = weights[:0], post_idx[:0], fixed[:0]
        metadata = {name: value[:0] for name, value in metadata.items()}
    scratch = torch.full((2,), 99.0, dtype=dtype)
    address = scratch.data_ptr()
    pristine = history.clone()
    slot = torch.tensor([4])
    rows = (slot - fixed).remainder(history.shape[0])
    gates = (
        torch.bitwise_and(history[rows, metadata["word_idx"]], metadata["bit_mask"])
        != 0
    )
    expected = torch.zeros_like(scratch)
    expected.index_add_(0, post_idx, weights * gates.to(dtype))

    def failed_launch(_, output):
        output.fill_(1000.0)
        return False

    connection = SimpleNamespace(
        _n_conn=weights.numel(),
        _syn_numel=scratch.numel(),
        spike_history_packed=history,
        post_idx=post_idx,
        max_delay_steps=history.shape[0],
        bitpack_conn_word_idx=metadata["word_idx"],
        bitpack_conn_bit_mask=metadata["bit_mask"],
        inference_delay_steps=fixed,
        _try_bitpack_delivery_kernel=failed_launch,
        weight=lambda: weights,
    )
    observed = NetCon._bitpack_build_delivery_from_history(connection, slot, scratch)
    assert observed is scratch and observed.data_ptr() == address
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(history, pristine)
    # The same donated row can be overwritten on the next call without stale data.
    result = source_history_project(
        history,
        slot,
        weights * 2,
        post_idx=post_idx,
        depth=history.shape[0],
        width=scratch.numel(),
        delay_steps=fixed,
        _scratch=scratch,
        **metadata,
    )
    assert result is scratch and result.data_ptr() == address
    expected.zero_().index_add_(0, post_idx, weights * 2 * gates.to(dtype))
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("taps", (None, 2, 3))
def test_delivery_uses_current_weights_delays_and_retains_gradients(packed, taps):
    history, logical, source_pos, post_idx, metadata = _fixture(packed=packed)
    history = history.clone().requires_grad_(not packed)
    weights = torch.tensor([0.7, -0.2, 1.3, 0.5, 2.1], dtype=DTYPE, requires_grad=True)
    delays = torch.tensor(
        [0.15, 0.23, 0.31, 0.42, 0.1], dtype=DTYPE, requires_grad=True
    )
    slot = torch.tensor([4])
    fixed = torch.tensor([1, 2, 3, 4, 1])
    dt = torch.tensor(0.1, dtype=DTYPE)
    result = source_history_project(
        history,
        slot,
        weights,
        post_idx=post_idx,
        depth=5,
        width=2,
        delay_steps=fixed,
        delays=delays if taps else None,
        dt=dt,
        taps=taps or 2,
        sigma=0.4,
        **metadata,
    )
    readable = logical if packed else history
    edge_values = []
    for edge, source in enumerate(source_pos.tolist()):
        if taps is None:
            gate = readable[(4 - fixed[edge]) % 5, source]
        else:
            distance = (delays[edge] / dt).clamp_min(1)
            base = distance.floor()
            if taps == 2:
                fraction = distance - base
                gate = readable[(4 - base.long()) % 5, source] * (1 - fraction)
                gate = gate + readable[(3 - base.long()) % 5, source] * fraction
            else:
                centers = torch.stack((base - 1, base, base + 1))
                interpolation = torch.softmax(
                    -0.5 * ((distance - centers) / 0.400001) ** 2, dim=0
                )
                gate = (
                    readable[(4 - centers.long()) % 5, source] * interpolation
                ).sum()
        edge_values.append(weights[edge] * gate)
    reference = torch.zeros(2, dtype=DTYPE).index_add(
        0, post_idx, torch.stack(edge_values)
    )
    torch.testing.assert_close(result, reference)
    targets = (weights, *(() if packed else (history,)), *((delays,) if taps else ()))
    observed = torch.autograd.grad(result.square().sum(), targets, retain_graph=True)
    expected = torch.autograd.grad(reference.square().sum(), targets)
    for actual, oracle in zip(observed, expected, strict=True):
        torch.testing.assert_close(actual, oracle)
    changed = source_history_project(
        history,
        slot,
        weights * 3,
        post_idx=post_idx,
        depth=5,
        width=2,
        delay_steps=fixed,
        delays=delays if taps else None,
        dt=dt,
        taps=taps or 2,
        sigma=0.4,
        **metadata,
    )
    torch.testing.assert_close(changed, result * 3)


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("taps", (None, 2, 3))
@pytest.mark.parametrize("lanes", (0, 3))
def test_project_record_vmap_including_empty_lanes(packed, taps, lanes):
    history, _logical, source_pos, post_idx, metadata = _fixture(packed=packed)
    word_idx, bit_mask = _packing(3)
    weights = torch.tensor([0.7, -0.2, 1.3, 0.5, 2.1], dtype=DTYPE)
    delays = torch.tensor([0.15, 0.23, 0.31, 0.42, 0.1], dtype=DTYPE)
    dt = torch.tensor(0.1, dtype=DTYPE)
    fixed = torch.tensor([1, 2, 3, 4, 1])

    def execute(h, w, d, slot, gates):
        payload = source_history_project(
            h,
            slot,
            w,
            post_idx=post_idx,
            depth=5,
            width=2,
            delay_steps=fixed,
            delays=d if taps else None,
            dt=dt,
            taps=taps or 2,
            **metadata,
        )
        updated = source_history_record(
            h,
            slot,
            gates,
            source_word_idx=word_idx if packed else None,
            source_bit_mask=bit_mask if packed else None,
        )
        return payload, updated

    histories = history.unsqueeze(0).expand(lanes, *history.shape).clone()
    parameter_lanes = weights.unsqueeze(0).expand(lanes, -1).clone()
    delay_lanes = delays.unsqueeze(0).expand(lanes, -1).clone()
    slots = torch.tensor([1, 3, 4])[:lanes].reshape(lanes, 1)
    gates = torch.tensor([[0.2, -0.4, 0.7]], dtype=DTYPE).expand(lanes, -1).clone()
    actual = torch.vmap(execute)(histories, parameter_lanes, delay_lanes, slots, gates)
    if lanes:
        expected = [
            execute(*values)
            for values in zip(
                histories, parameter_lanes, delay_lanes, slots, gates, strict=True
            )
        ]
        for index in (0, 1):
            torch.testing.assert_close(
                actual[index], torch.stack([value[index] for value in expected])
            )
    else:
        assert actual[0].shape == (0, 2)
        assert actual[1].shape == (0, *history.shape)


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
def test_project_cpu_collision_precision_and_fullgraph(packed, dtype):
    history, logical, source_pos, post_idx, metadata = _fixture(
        packed=packed, dtype=dtype
    )
    weights = torch.tensor([0.3333, -0.2222, 1.4444, 0.5555, 2.1111], dtype=dtype)
    delays = torch.tensor([1, 2, 3, 4, 1])
    slot = torch.tensor([4])

    def execute(h, w):
        return source_history_project(
            h,
            slot,
            w,
            post_idx=post_idx,
            depth=5,
            width=2,
            delay_steps=delays,
            **metadata,
        )

    expected = torch.zeros(2, dtype=dtype)
    expected.index_add_(0, post_idx, weights * logical[(4 - delays) % 5, source_pos])
    torch.testing.assert_close(execute(history, weights), expected, rtol=0, atol=0)
    compiled = torch.compile(execute, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(history, weights), expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed", (False, True))
def test_zero_source_history_and_zero_connections_are_well_defined(packed):
    history = torch.zeros(
        (3, 1 if packed else 0), dtype=torch.int64 if packed else DTYPE
    )
    empty = torch.empty(0, dtype=torch.long)
    metadata = (
        {"word_idx": empty, "bit_mask": empty} if packed else {"source_pos": empty}
    )
    output = source_history_project(
        history,
        torch.tensor([2]),
        torch.empty(0, dtype=DTYPE),
        post_idx=empty,
        depth=3,
        width=4,
        delay_steps=empty,
        **metadata,
    )
    updated = source_history_record(
        history,
        torch.tensor([2]),
        torch.empty(0, dtype=DTYPE),
        source_word_idx=empty if packed else None,
        source_bit_mask=empty if packed else None,
    )
    torch.testing.assert_close(output, torch.zeros(4, dtype=DTYPE))
    torch.testing.assert_close(updated, history)
