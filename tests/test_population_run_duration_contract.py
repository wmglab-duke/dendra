"""Cumulative fixed-step duration contracts for Population execution APIs."""

import math

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.1
DTYPE = torch.float64
RUN_VARIANTS = ("run", "longrun", "checkpointed")


def _population():
    pop = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    pop.insert(pas, g=0.001, e=-70.0)
    pop.initialize()
    return pop


def _advance(pop, variant, duration, *, dt=DT):
    if variant == "run":
        return pop.run(tstop=duration, dt=dt)
    if variant == "longrun":
        return pop.longrun(tstop=duration, chunklength=2, dt=dt)
    if variant == "checkpointed":
        return pop.longrun_checkpointed(
            tstop=duration,
            chunklength=2,
            dt=dt,
            safe_checkpoint=True,
        )
    raise AssertionError(f"unknown execution variant {variant!r}")


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def _duration_remainder(pop):
    value = pop.state_dict_for_checkpoint()["duration_remainder"]
    assert value.shape == ()
    return value.item()


@pytest.mark.parametrize("variant", RUN_VARIANTS)
def test_fractional_partition_matches_complete_run_for_each_variant(variant):
    whole = _population()
    split = _population()

    _advance(whole, variant, 0.3)
    _advance(split, variant, 0.15)
    _advance(split, variant, 0.15)

    assert whole.t.item() == pytest.approx(0.3)
    assert split.t.item() == pytest.approx(0.3)
    _assert_nested_equal(
        split.state_dict_for_checkpoint(), whole.state_dict_for_checkpoint()
    )


@pytest.mark.parametrize("variant", RUN_VARIANTS)
def test_tiny_calls_accumulate_without_each_call_rounding_up(variant):
    pop = _population()

    for _ in range(10):
        _advance(pop, variant, 0.01)

    assert pop.t.item() == pytest.approx(DT)


def test_duration_carry_is_shared_when_alternating_execution_variants():
    pop = _population()

    _advance(pop, "run", 0.04)
    assert pop.t.item() == 0.0
    _advance(pop, "longrun", 0.03)
    assert pop.t.item() == 0.0
    _advance(pop, "checkpointed", 0.03)

    assert pop.t.item() == pytest.approx(DT)


@pytest.mark.parametrize("variant", RUN_VARIANTS)
def test_initialize_discards_pending_duration_for_each_variant(variant):
    pop = _population()
    _advance(pop, variant, 0.15)
    assert pop.t.item() == pytest.approx(DT)

    pop.initialize()
    _advance(pop, variant, 0.05)
    assert pop.t.item() == 0.0

    _advance(pop, variant, 0.05)
    assert pop.t.item() == pytest.approx(DT)


def test_checkpoint_restore_recovers_and_replaces_a_pending_duration():
    source = _population()
    _advance(source, "run", 0.04)
    assert source.t.item() == 0.0
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())

    _advance(source, "longrun", 0.06)
    assert source.t.item() == pytest.approx(DT)
    expected = _clone_nested(source.state_dict_for_checkpoint())

    resumed = _population()
    _advance(resumed, "run", 0.02)
    assert resumed.t.item() == 0.0
    resumed.restore_dict_from_checkpoint(checkpoint)
    _advance(resumed, "checkpointed", 0.06)

    _assert_nested_equal(resumed.state_dict_for_checkpoint(), expected)


@pytest.mark.parametrize("variant", RUN_VARIANTS)
@pytest.mark.parametrize(
    ("relation", "expected_steps"),
    [("before", 0), ("exact", 1), ("after", 1)],
)
def test_adjacent_binary_grid_boundaries_remain_distinct(
    variant, relation, expected_steps
):
    dt = 0.125
    if relation == "before":
        duration = math.nextafter(dt, -math.inf)
    elif relation == "after":
        duration = math.nextafter(dt, math.inf)
    else:
        duration = dt

    pop = _population()
    _advance(pop, variant, duration, dt=dt)
    assert pop.t.item() == pytest.approx(expected_steps * dt)

    if relation == "before":
        _advance(pop, variant, dt - duration, dt=dt)
        assert pop.t.item() == pytest.approx(dt)


def test_explicit_step_and_ve_driven_run_preserve_pending_duration():
    pop = _population()
    _advance(pop, "run", 0.05)
    assert _duration_remainder(pop) == pytest.approx(0.05)

    pop.step(dt=DT)
    assert pop.t.item() == pytest.approx(DT)
    assert _duration_remainder(pop) == pytest.approx(0.05)

    ve = torch.zeros((1, *pop.shape), dtype=DTYPE)
    pop.run(ve=ve, dt=DT)
    assert pop.t.item() == pytest.approx(2 * DT)
    assert _duration_remainder(pop) == pytest.approx(0.05)

    _advance(pop, "longrun", 0.05)
    assert pop.t.item() == pytest.approx(3 * DT)
    assert _duration_remainder(pop) == 0.0


def test_pending_duration_is_physical_time_across_timestep_changes():
    pop = _population()
    _advance(pop, "run", 0.05, dt=0.1)
    assert _duration_remainder(pop) == pytest.approx(0.05)

    _advance(pop, "longrun", 0.15, dt=0.2)
    assert pop.t.item() == pytest.approx(0.2)
    assert _duration_remainder(pop) == 0.0

    _advance(pop, "run", 0.05, dt=0.2)
    assert _duration_remainder(pop) == pytest.approx(0.05)
    # A duration-based zero call settles retained physical time on a new grid.
    _advance(pop, "checkpointed", 0.0, dt=0.025)
    assert pop.t.item() == pytest.approx(0.25)
    assert _duration_remainder(pop) == 0.0


def test_legacy_runtime_checkpoint_without_remainder_restores_zero():
    source = _population()
    _advance(source, "run", 0.1)
    legacy = _clone_nested(source.state_dict_for_checkpoint())
    legacy.pop("duration_remainder")

    resumed = _population()
    _advance(resumed, "run", 0.05)
    assert _duration_remainder(resumed) == pytest.approx(0.05)
    resumed.restore_dict_from_checkpoint(legacy)

    assert _duration_remainder(resumed) == 0.0


def test_legacy_normal_state_without_remainder_clears_receiver_carry():
    source = _population()
    legacy = _clone_nested(source.state_dict())
    legacy.pop("_duration_remainder")

    resumed = _population()
    _advance(resumed, "run", 0.05)
    assert _duration_remainder(resumed) == pytest.approx(0.05)
    resumed.load(legacy)

    assert _duration_remainder(resumed) == 0.0


def test_named_cache_restores_pending_duration_for_exact_continuation():
    pop = _population()
    _advance(pop, "run", 0.04)
    pop.cache("pending")

    _advance(pop, "run", 0.06)
    assert pop.t.item() == pytest.approx(DT)
    pop.restore("pending")

    assert pop.t.item() == 0.0
    assert _duration_remainder(pop) == pytest.approx(0.04)
    _advance(pop, "run", 0.06)
    assert pop.t.item() == pytest.approx(DT)


@pytest.mark.parametrize(
    ("bad_value", "error"),
    [
        (1.0, TypeError),
        (torch.zeros(1), ValueError),
        (torch.tensor(0), TypeError),
        (torch.tensor(float("nan")), ValueError),
        (torch.tensor(float("inf")), ValueError),
        (torch.tensor(-0.1), ValueError),
    ],
)
def test_runtime_checkpoint_validates_remainder_atomically(bad_value, error):
    pop = _population()
    _advance(pop, "run", 0.05)
    before = _clone_nested(pop.state_dict_for_checkpoint())
    corrupt = _clone_nested(before)
    corrupt["duration_remainder"] = bad_value

    with pytest.raises(error, match="duration_remainder"):
        pop.restore_dict_from_checkpoint(corrupt)

    _assert_nested_equal(pop.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("variant", RUN_VARIANTS)
@pytest.mark.parametrize("invalid", [-0.1, math.inf, math.nan, True])
def test_invalid_duration_does_not_mutate_pending_time(variant, invalid):
    pop = _population()
    _advance(pop, "run", 0.05)
    before = _clone_nested(pop.state_dict_for_checkpoint())

    with pytest.raises((TypeError, ValueError), match="tstop"):
        _advance(pop, variant, invalid)

    _assert_nested_equal(pop.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("variant", RUN_VARIANTS)
@pytest.mark.parametrize("invalid_dt", [0.0, -0.1, math.inf, math.nan, True])
def test_invalid_timestep_does_not_mutate_pending_time(variant, invalid_dt):
    pop = _population()
    _advance(pop, "run", 0.05)
    before = _clone_nested(pop.state_dict_for_checkpoint())

    with pytest.raises((TypeError, ValueError), match="dt"):
        _advance(pop, variant, 0.05, dt=invalid_dt)

    _assert_nested_equal(pop.state_dict_for_checkpoint(), before)


def test_steady_state_starts_and_leaves_a_fresh_duration_budget():
    pop = _population()
    _advance(pop, "run", 0.05)
    pop.steady_state(dt=0.1, tstop=0.0)

    assert pop.t.item() == 0.0
    assert _duration_remainder(pop) == 0.0
