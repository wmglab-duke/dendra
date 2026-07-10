import pytest
import torch

from dendra.models.networks.net import dilate
from dendra.models.networks.netcon import (
    _age_rows_to_ring_rows,
    _cache_current_slot,
    _cache_dt_value,
    _clone_calendar_chunks,
    _dilate_history_age_rows,
    _dilate_packed_history_age_rows,
    _dilate_packed_time_rows,
    _dilate_time_rows,
    _restore_calendar_chunks,
    _ring_rows_to_age_rows,
)


def test_cache_scalar_helpers_accept_numbers_and_tensors():
    assert _cache_dt_value(0.1) == pytest.approx(0.1)
    assert _cache_dt_value(torch.tensor(0.2)) == pytest.approx(0.2)
    assert _cache_current_slot(3) == 3
    assert _cache_current_slot(torch.tensor([4])) == 4


@pytest.mark.parametrize("fn", [_dilate_time_rows, _dilate_history_age_rows])
def test_numeric_time_row_dilation_sums_collisions(fn):
    rows = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    out = fn(rows, 0.1, 0.2, n_limit=3)
    assert torch.equal(out, torch.tensor([[1.0], [6.0], [8.0]]))

    clone = fn(rows, 0.1, 0.1, n_limit=4)
    assert torch.equal(clone, rows)
    assert clone.data_ptr() != rows.data_ptr()
    assert fn(rows, 0.1, 0.2, n_limit=0).shape == (0, 1)
    assert fn(rows[:0], 0.1, 0.2, n_limit=2).shape == (2, 1)

    with pytest.raises(ValueError, match="2D"):
        fn(rows[:, 0], 0.1, 0.2, n_limit=2)


@pytest.mark.parametrize(
    "fn", [_dilate_packed_time_rows, _dilate_packed_history_age_rows]
)
def test_packed_time_row_dilation_uses_bitwise_or(fn):
    rows = torch.tensor([[1], [2], [4], [8]], dtype=torch.int64)
    out = fn(rows, 0.1, 0.2, n_limit=3)
    assert torch.equal(out, torch.tensor([[1], [6], [8]], dtype=torch.int64))

    clone = fn(rows, 0.1, 0.1, n_limit=4)
    assert torch.equal(clone, rows)
    assert clone.data_ptr() != rows.data_ptr()
    assert fn(rows, 0.1, 0.2, n_limit=-1).shape == (0, 1)
    assert fn(rows[:0], 0.1, 0.2, n_limit=2).shape == (2, 1)

    with pytest.raises(ValueError, match="2D"):
        fn(rows[:, 0], 0.1, 0.2, n_limit=2)


def test_ring_age_conversion_normalizes_current_slot_and_round_trips():
    ring = torch.tensor([[10], [11], [12], [13]])
    ages = _ring_rows_to_age_rows(ring, cur_slot=2)
    assert torch.equal(ages[:, 0], torch.tensor([12, 11, 10, 13]))

    normalized_ring = _age_rows_to_ring_rows(ages, depth=4)
    assert torch.equal(normalized_ring[:, 0], torch.tensor([12, 13, 10, 11]))
    assert torch.equal(_ring_rows_to_age_rows(normalized_ring, 0), ages)

    assert _ring_rows_to_age_rows(ring[:0], 10).shape == (0, 1)
    assert _age_rows_to_ring_rows(ages, depth=0).shape == (0, 1)
    assert _age_rows_to_ring_rows(ages[:0], depth=3).shape == (3, 1)

    with pytest.raises(ValueError, match="2D"):
        _ring_rows_to_age_rows(ring[:, 0], 0)
    with pytest.raises(ValueError, match="2D"):
        _age_rows_to_ring_rows(ages[:, 0], depth=4)


def test_sparse_calendar_cache_is_relative_cloned_and_dt_adjustable():
    idx = torch.tensor([1, 3])
    values = torch.tensor([0.5, 1.5])
    calendar = {3: [(idx, values)], 0: [(torch.tensor([2]), torch.tensor([2.0]))]}

    cached = _clone_calendar_chunks(calendar, cur_slot=2, depth=4)
    assert set(cached) == {1, 2}
    idx.add_(10)
    values.zero_()
    assert torch.equal(cached[1][0][0], torch.tensor([1, 3]))
    assert torch.equal(cached[1][0][1], torch.tensor([0.5, 1.5]))

    restored = _restore_calendar_chunks(
        cached,
        old_dt=0.1,
        new_dt=0.2,
        depth=3,
        idx_device=torch.device("cpu"),
        idx_dtype=torch.int32,
        val_device=torch.device("cpu"),
        val_dtype=torch.float64,
    )
    # Relative slots 1 and 2 map to new slots 1 and 1, respectively.
    assert set(restored) == {1}
    assert len(restored[1]) == 2
    assert restored[1][0][0].dtype == torch.int32
    assert restored[1][0][1].dtype == torch.float64

    assert _clone_calendar_chunks({0: []}, cur_slot=0, depth=0) == {}
    assert (
        _restore_calendar_chunks(
            {0: []},
            old_dt=0.1,
            new_dt=0.1,
            depth=0,
            idx_device="cpu",
            idx_dtype=torch.long,
            val_device="cpu",
            val_dtype=torch.float32,
        )
        == {}
    )


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("nearest", [[1.0], [6.0], [8.0]]),
        ("floor", [[3.0], [12.0]]),
        ("ceil", [[1.0], [6.0], [8.0]]),
    ],
)
def test_network_dilate_rebins_numeric_rows(mode, expected):
    rows = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    out = dilate(rows, 0.1, 0.2, mode=mode)
    assert torch.equal(out, torch.tensor(expected))


def test_network_dilate_handles_bool_horizon_limits_and_validation():
    rows = torch.tensor([[True], [True], [False]])
    out = dilate(rows, 0.1, 0.2, horizon_ms=0.6)
    assert out.dtype == torch.int64
    assert out.shape == (3, 1)
    assert out[:, 0].tolist() == [1, 1, 0]

    limited = dilate(torch.ones((4, 1)), 0.1, 0.2, n_limit=2)
    assert limited.shape == (2, 1)
    assert limited.sum().item() == 4
    assert dilate(torch.empty((0, 2)), 0.1, 0.2).shape == (0, 2)

    same = torch.ones((2, 1))
    assert dilate(same, 0.1, 0.1) is same
    with pytest.raises(ValueError, match="2D"):
        dilate(torch.ones(2), 0.1, 0.2)
    with pytest.raises(ValueError, match="must be > 0"):
        dilate(torch.ones((2, 1)), 0.0, 0.2)
    with pytest.raises(ValueError, match="Unknown mode"):
        dilate(torch.ones((2, 1)), 0.1, 0.2, mode="bad")
