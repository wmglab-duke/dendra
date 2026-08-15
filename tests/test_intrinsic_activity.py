"""Contracts for the alpha-conductance intrinsic-activity convenience API."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
import dendra.models as models
from dendra.models.intrinsic import (
    _IntrinsicActivity,
    insert_intrinsic_activity,
    remove_intrinsic_activity,
)
from dendra.models.mod import alphasynapse, pas

DTYPE = torch.float64


def _population(*, n=2, c=2):
    return dn.Population(N=n, C=c, v_init=-65.0, dtype=DTYPE)


def _intrinsic_mechanism(population):
    name = _IntrinsicActivity._name or _IntrinsicActivity.__name__
    return population.mech.mechanisms[name]


def test_intrinsic_activity_helpers_are_exported_from_models_and_top_level():
    assert models.insert_intrinsic_activity is insert_intrinsic_activity
    assert models.remove_intrinsic_activity is remove_intrinsic_activity
    assert dn.insert_intrinsic_activity is insert_intrinsic_activity
    assert dn.remove_intrinsic_activity is remove_intrinsic_activity
    assert "insert_intrinsic_activity" in dn.__all__
    assert "remove_intrinsic_activity" in dn.__all__


@pytest.mark.parametrize(
    ("label", "onsets", "expected_key", "expected_onsets"),
    [
        pytest.param(
            "scalar",
            0.25,
            [1, 3],
            [0.25, 0.25],
            id="scalar-event-shared-across-locations",
        ),
        pytest.param(
            "shared_vector",
            [0.1, 0.4, 0.7],
            [1, 3, 1, 3, 1, 3],
            [0.1, 0.1, 0.4, 0.4, 0.7, 0.7],
            id="one-dimensional-shared-train",
        ),
        pytest.param(
            "shared_matrix",
            [[0.1], [0.4]],
            [1, 3, 1, 3],
            [0.1, 0.1, 0.4, 0.4],
            id="explicit-shared-train-matrix",
        ),
        pytest.param(
            "per_location_matrix",
            [[0.1, 0.2]],
            [1, 3],
            [0.1, 0.2],
            id="one-event-per-location",
        ),
        pytest.param(
            "event_by_location_matrix",
            [[0.1, 0.2], [0.4, 0.5]],
            [1, 3, 1, 3],
            [0.1, 0.2, 0.4, 0.5],
            id="event-by-location-matrix",
        ),
    ],
)
def test_onset_layouts_compile_in_copy_major_location_order(
    label, onsets, expected_key, expected_onsets
):
    population = _population()

    dn.insert_intrinsic_activity(population[:, 1], onsets)
    population.build()

    mechanism = _intrinsic_mechanism(population)
    assert mechanism.key.tolist() == expected_key, label
    torch.testing.assert_close(
        mechanism.onset,
        torch.tensor(expected_onsets, dtype=DTYPE),
    )
    torch.testing.assert_close(
        mechanism.tau,
        torch.full_like(mechanism.onset, 0.1),
    )
    torch.testing.assert_close(
        mechanism.gmax,
        torch.full_like(mechanism.onset, 0.1),
    )
    torch.testing.assert_close(
        mechanism.e,
        torch.zeros_like(mechanism.onset),
    )


def test_repeated_single_event_calls_create_additive_colocated_slots():
    population = _population(n=1, c=1)
    target = population[:, 0]

    dn.insert_intrinsic_activity(target, 0.1, gmax=0.02)
    dn.insert_intrinsic_activity(target, 0.3, gmax=0.04)
    population.build()

    mechanism = _intrinsic_mechanism(population)
    assert mechanism.key.tolist() == [0, 0]
    torch.testing.assert_close(
        mechanism.onset,
        torch.tensor([0.1, 0.3], dtype=DTYPE),
    )
    torch.testing.assert_close(
        mechanism.gmax,
        torch.tensor([0.02, 0.04], dtype=DTYPE),
    )


@pytest.mark.parametrize(
    "onsets",
    (
        [],
        torch.empty((0,), dtype=DTYPE),
        torch.empty((0, 2), dtype=DTYPE),
    ),
)
def test_empty_onset_schedules_are_no_ops(onsets):
    population = _population()

    dn.insert_intrinsic_activity(population[:, 1], onsets)

    assert _IntrinsicActivity not in population._mech_data
    population.build()
    assert (_IntrinsicActivity._name or _IntrinsicActivity.__name__) not in (
        population.mech.mechanisms
    )


def test_empty_slice_insert_and_remove_are_no_ops():
    population = _population()
    empty = population[:, 0:0]

    dn.insert_intrinsic_activity(empty, [0.1, 0.2])
    dn.remove_intrinsic_activity(empty)

    assert _IntrinsicActivity not in population._mech_data


@pytest.mark.parametrize(
    ("onsets", "error", "match"),
    [
        pytest.param(
            torch.zeros((1, 1, 1), dtype=DTYPE),
            ValueError,
            "two-dimensional",
            id="three-dimensional",
        ),
        pytest.param(
            torch.empty((2, 0), dtype=DTYPE),
            ValueError,
            "at least one location",
            id="events-with-zero-location-axis",
        ),
        pytest.param("not-a-number", TypeError, "numeric", id="non-numeric"),
    ],
)
def test_invalid_onset_values_fail_at_insertion(onsets, error, match):
    population = _population()

    with pytest.raises(error, match=match):
        dn.insert_intrinsic_activity(population[:, 1], onsets)

    assert _IntrinsicActivity not in population._mech_data


def test_matrix_location_axis_must_match_the_target():
    population = _population()
    dn.insert_intrinsic_activity(
        population[:, 1],
        torch.tensor([[0.1, 0.2, 0.3]], dtype=DTYPE),
    )

    with pytest.raises(ValueError, match="shape"):
        population.build()


@pytest.mark.parametrize(
    "helper", [insert_intrinsic_activity, remove_intrinsic_activity]
)
def test_intrinsic_activity_helpers_require_a_slice(helper):
    population = _population()

    if helper is insert_intrinsic_activity:
        with pytest.raises(TypeError, match="Slice"):
            helper(population, 0.1)
    else:
        with pytest.raises(TypeError, match="Slice"):
            helper(population)


def test_partial_removal_projects_every_event_and_parameter_to_retained_locations():
    population = _population(n=1, c=3)
    dn.insert_intrinsic_activity(
        population[:, :],
        torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=DTYPE),
        tau=torch.tensor([[0.05, 0.06, 0.07]], dtype=DTYPE),
        gmax=torch.tensor([[0.01], [0.02]], dtype=DTYPE),
        e=-5.0,
    )

    dn.remove_intrinsic_activity(population[:, 1])
    population.build()

    mechanism = _intrinsic_mechanism(population)
    assert mechanism.key.tolist() == [0, 2, 0, 2]
    torch.testing.assert_close(
        mechanism.onset,
        torch.tensor([0.1, 0.3, 1.1, 1.3], dtype=DTYPE),
    )
    torch.testing.assert_close(
        mechanism.tau,
        torch.tensor([0.05, 0.07, 0.05, 0.07], dtype=DTYPE),
    )
    torch.testing.assert_close(
        mechanism.gmax,
        torch.tensor([0.01, 0.01, 0.02, 0.02], dtype=DTYPE),
    )
    torch.testing.assert_close(
        mechanism.e,
        torch.full((4,), -5.0, dtype=DTYPE),
    )


def test_removal_is_isolated_from_ordinary_alphasynapse_insertions():
    population = _population(n=1, c=2)
    target = population[:, 0]
    target.insert(
        alphasynapse,
        onset=0.2,
        tau=0.3,
        gmax=0.01,
        e=-10.0,
    )
    dn.insert_intrinsic_activity(target, [0.1, 0.4])

    dn.remove_intrinsic_activity(target)
    population.build()

    assert _IntrinsicActivity not in population._mech_data
    assert alphasynapse in population._mech_data
    assert "alphasynapse" in population.mech.mechanisms
    ordinary = population.mech.mechanisms["alphasynapse"]
    torch.testing.assert_close(
        ordinary.onset.reshape(-1),
        torch.tensor([0.2], dtype=DTYPE),
    )


def test_intrinsic_activity_drives_only_the_selected_compartment_in_a_small_run():
    population = dn.Population(
        N=1,
        C=2,
        v_init=-65.0,
        dt=0.005,
        dtype=DTYPE,
    )
    population.insert(pas, g=0.001, e=-65.0)
    dn.insert_intrinsic_activity(
        population[:, 0],
        0.01,
        tau=0.05,
        gmax=0.001,
        e=0.0,
    )

    population.initialize()
    population.run(tstop=0.1, dt=0.005)

    assert torch.isfinite(population.v).all()
    assert population.v[0, 0] > -65.0
    torch.testing.assert_close(
        population.v[0, 1],
        torch.tensor(-65.0, dtype=DTYPE),
        rtol=0.0,
        atol=1.0e-12,
    )
