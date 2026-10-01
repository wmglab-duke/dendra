"""Signed margins are checked against the callbacks' actual step hooks."""

import pytest
import torch

from dendra.models.analysis.callback_margin import callback_signed_margin
from dendra.models.callbacks import (
    Active,
    ActiveAL,
    ActiveALCount,
    APCount,
    Recorder,
    _Active,
)


class TraceModel:
    def __init__(self, first_frame):
        self.v = first_frame
        self.nc = first_frame.shape[-1]

    @property
    def shape(self):
        return self.v.shape

    def device(self):
        return self.v.device


def run_callback(callback, trace, recorder=None):
    model = TraceModel(trace[0])
    callback.pre_loop_hook(model)
    if recorder is not None:
        recorder.pre_loop_hook(model)
    for frame in trace[1:]:
        model.v = frame
        callback.post_step_hook(model)
        if recorder is not None:
            recorder.post_step_hook(model)
    return callback


def test_active_margin_uses_checked_post_step_window_and_inclusive_threshold():
    # The initial frame, early frames, and first frame after the end may all
    # exceed threshold; only callback steps 2, 3, and 4 are checked.
    trace = torch.tensor(
        [
            [[9.0, 9.0]],
            [[9.0, 9.0]],
            [[9.0, 9.0]],
            [[-2.0, -3.0]],
            [[1.0, -3.0]],
            [[-2.0, -3.0]],
            [[9.0, 9.0]],
        ]
    )
    callback = Active(
        threshold=1.0,
        t_start_check=0.2,
        t_end_check=0.5,
        node_check=[0, -1],
        dt=0.1,
    )
    run_callback(callback, trace)
    result = callback_signed_margin(callback, trace)
    assert torch.equal(result.decision, callback.is_active())
    assert result.margin.item() == 0.0
    assert result.branch_id == (3, 0)

    below = trace.clone()
    below[4, 0, 0] = 0.9
    callback.reset()
    run_callback(callback, below)
    result = callback_signed_margin(callback, below)
    assert torch.equal(result.decision, callback.is_active())
    assert result.margin.item() < 0
    assert result.branch_id == (3, 0)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize(
    "at_least,partition", [(1, None), (2, None), (1, [2, 2]), (2, [2, 2])]
)
def test_active_al_margin_matches_hard_callback_for_batched_partitions(
    batch_shape, at_least, partition
):
    generator = torch.Generator().manual_seed(238)
    trace = torch.randn((8, *batch_shape, 3, 5), generator=generator)
    trace[0] = 10.0  # Initial state is not checked.
    trace[3, ..., 1, 0] = 0.25  # Equality must count as active.
    callback = ActiveAL(
        threshold=0.25,
        t_start_check=0.1,
        t_end_check=0.6,
        node_check=[0, 1, -1, 2],
        dt=0.1,
        at_least=at_least,
    )
    run_callback(callback, trace)
    result = callback_signed_margin(callback, trace, partition=partition)
    hard = callback.is_active(partition=partition)
    assert result.margin.shape == hard.shape
    assert torch.equal(result.decision, hard)
    assert result.step_index.shape == hard.shape
    assert result.checked_site_index.shape == hard.shape
    assert bool(((result.step_index >= 1) & (result.step_index < 6)).all())


def test_margin_has_selected_site_autograd_gradient_and_scalar_branch():
    parameter = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    trace = torch.full((5, 1, 4), -5.0, dtype=torch.double)
    trace[2, 0, 2] = parameter
    callback = Active(threshold=0.0, node_check=[-1, 2], dt=0.1)
    result = callback_signed_margin(callback, trace)
    assert result.branch_id == (1, 1)
    assert torch.autograd.grad(result.margin.sum(), parameter)[0].item() == 1.0


def test_batched_branch_ids_are_flattened_in_row_major_output_order():
    trace = torch.full((3, 2, 1, 2), -1.0)
    trace[1, 0, 0, 1] = 2.0
    trace[2, 1, 0, 0] = 3.0
    callback = Active(node_check=[0, 1])
    result = callback_signed_margin(callback, trace)
    assert result.margin.shape == (2, 1)
    assert result.branch_ids == ((0, 1), (1, 0))
    with pytest.raises(ValueError, match="Select one output lane"):
        _ = result.branch_id


@pytest.mark.parametrize("callback_type", [ActiveALCount, _Active])
def test_other_any_site_callbacks_have_the_same_exact_margin(callback_type):
    trace = torch.tensor(
        [
            [[10.0, -1.0], [-1.0, -1.0]],
            [[-1.0, -1.0], [-1.0, -1.0]],
            [[-1.0, 0.0], [-1.0, -1.0]],
        ]
    )
    callback = callback_type(threshold=0.0, node_check=[0, 1])
    run_callback(callback, trace)
    result = callback_signed_margin(callback, trace)
    assert torch.equal(result.decision, callback.is_active())
    assert result.margin[0].item() == 0.0
    assert result.margin[1].item() < 0

    if callback_type is ActiveALCount:
        partitioned = callback_signed_margin(callback, trace, partition=[1, 1])
        assert torch.equal(partitioned.decision, callback.is_active([1, 1]))
    else:
        with pytest.raises(ValueError, match="does not support partitions"):
            callback_signed_margin(callback, trace, partition=[1, 1])


def test_recorder_adapter_accepts_full_and_matching_indexed_trace():
    trace = torch.tensor(
        [
            [[-1.0, -1.0, -1.0, -1.0]],
            [[-1.0, -1.0, -1.0, 0.2]],
            [[0.5, -1.0, -1.0, -1.0]],
        ]
    )
    full = Recorder(["v"])
    callback = ActiveAL(threshold=0.0, node_check=[-1, 0], at_least=2)
    run_callback(callback, trace, full)
    result = callback_signed_margin(callback, full)
    assert torch.equal(result.decision, callback.is_active())
    assert result.margin.item() == pytest.approx(0.2)

    indexed = Recorder(["v"], node_indices=[3, 0])
    callback = ActiveAL(threshold=0.0, node_check=[-1, 0], at_least=2)
    run_callback(callback, trace, indexed)
    result = callback_signed_margin(callback, indexed)
    assert torch.equal(result.decision, callback.is_active())
    assert result.margin.item() == pytest.approx(0.2)

    # An equivalent callback that has not yet run still has its raw negative
    # index, so the model's compartment count is needed to compare mappings.
    raw_callback = ActiveAL(threshold=0.0, node_check=[-1, 0], at_least=2)
    with pytest.raises(ValueError, match="n_compartments"):
        callback_signed_margin(raw_callback, indexed)
    result = callback_signed_margin(raw_callback, indexed, n_compartments=4)
    assert torch.equal(result.decision, callback.is_active())
    assert result.margin.item() == pytest.approx(0.2)


def test_rejects_recorder_that_cannot_reconstruct_hard_callback_window():
    trace = torch.full((3, 1, 4), -1.0)
    callback = Active(node_check=[0])
    recorder = Recorder(["v"], max_only=True)
    run_callback(callback, trace, recorder)
    with pytest.raises(ValueError, match="reduced or smoothed"):
        callback_signed_margin(callback, recorder)

    recorder = Recorder(["v"], node_indices=[1])
    run_callback(Active(node_check=[0]), trace, recorder)
    with pytest.raises(ValueError, match="must equal"):
        callback_signed_margin(callback, recorder)

    recorder = Recorder(["v"], dt=0.2)
    recorder.dt = 0.1
    with pytest.raises(ValueError, match="every solver step"):
        callback_signed_margin(callback, recorder)


@pytest.mark.parametrize(
    "callback",
    [
        APCount(node_check=[0]),
        ActiveALCount(node_check=[0], at_least=2),
        Active(inv=True, node_check=[0]),
    ],
)
def test_unsupported_callback_semantics_are_explicit(callback):
    trace = torch.zeros((3, 1, 2))
    with pytest.raises(NotImplementedError):
        callback_signed_margin(callback, trace)


def test_rejects_empty_check_window_and_invalid_partition():
    trace = torch.zeros((3, 1, 2))
    callback = Active(node_check=[0], t_start_check=1.0, dt=0.1)
    with pytest.raises(ValueError, match="No post-step"):
        callback_signed_margin(callback, trace)
    callback = ActiveAL(node_check=[0, 1], at_least=2)
    with pytest.raises(ValueError, match="at least"):
        callback_signed_margin(callback, trace, partition=[1, 1])
