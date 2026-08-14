"""Regression tests for colocated point-process bank insertion."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas

DTYPE = torch.float64


def _build_copied_pas(value, *, copies=3, locations=2, name="pas_copy_params"):
    pop = dn.Population(N=locations, C=2, v_init=-65.0, dtype=DTYPE)
    pop[:, 1].insert(
        pas.rename(name),
        copies=copies,
        g=value,
        e=-65.0,
    )
    pop.build()
    return pop, getattr(pop.mech, name)


def test_slice_insert_preserve_multiplicity_keeps_duplicate_slots():
    pop = dn.Population(N=1, C=3, v_init=-65.0, dtype=torch.float64)
    target = pop[(torch.tensor([0, 0, 0]), torch.tensor([1, 1, 1]))]
    target.insert(
        pas.rename("pas_bank"),
        preserve_multiplicity=True,
        g=1.0e-5,
        e=-65.0,
    )

    pop.build()
    mech = pop.mech.pas_bank
    assert tuple(mech.shape_f) == (3,)
    assert mech.key.tolist() == [1, 1, 1]


def test_slice_insert_copies_allocates_independent_colocated_slots():
    pop = dn.Population(N=1, C=3, v_init=-65.0, dtype=torch.float64)
    pop[:, 1].insert(pas.rename("pas_copies"), copies=4, g=1.0e-5, e=-65.0)

    pop.build()
    mech = pop.mech.pas_copies
    assert tuple(mech.shape_f) == (4,)
    assert mech.key.tolist() == [1, 1, 1, 1]


@pytest.mark.parametrize(
    ("label", "value", "expected"),
    [
        (
            "scalar",
            0.25,
            torch.full((6,), 0.25, dtype=DTYPE),
        ),
        (
            "per_copy",
            torch.tensor([[0.1], [0.2], [0.3]], dtype=DTYPE),
            torch.tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3], dtype=DTYPE),
        ),
        (
            "per_location",
            torch.tensor([[0.4, 0.5]], dtype=DTYPE),
            torch.tensor([0.4, 0.5, 0.4, 0.5, 0.4, 0.5], dtype=DTYPE),
        ),
        (
            "copy_by_location",
            torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=DTYPE),
            torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=DTYPE),
        ),
        (
            "full_flat_backward_compatible",
            torch.tensor([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=DTYPE),
            torch.tensor([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=DTYPE),
        ),
    ],
)
def test_copied_insertion_range_parameters_broadcast_over_copy_and_location_axes(
    label, value, expected
):
    copies = 3
    locations = 2
    _, mech = _build_copied_pas(
        value,
        copies=copies,
        locations=locations,
        name=f"pas_copy_params_{label}",
    )

    expected_locations = torch.arange(locations, dtype=torch.long) * 2 + 1
    assert tuple(mech.shape_f) == (copies * locations,)
    torch.testing.assert_close(mech.key.cpu(), expected_locations.repeat(copies))
    torch.testing.assert_close(mech.g, expected)


@pytest.mark.parametrize(
    ("copies", "locations", "value"),
    [
        pytest.param(
            3,
            2,
            torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE),
            id="bare_length_copies",
        ),
        pytest.param(
            3,
            2,
            torch.tensor([0.1, 0.2], dtype=DTYPE),
            id="bare_length_locations",
        ),
        pytest.param(
            3,
            3,
            torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE),
            id="bare_length_ambiguous_when_copies_equal_locations",
        ),
    ],
)
def test_copied_insertion_rejects_ambiguous_bare_vector_override(
    copies, locations, value
):
    with pytest.raises(ValueError) as error:
        _build_copied_pas(
            value,
            copies=copies,
            locations=locations,
            name=f"pas_copy_params_invalid_{copies}_{locations}_{value.numel()}",
        )

    message = str(error.value).lower()
    assert "shape" in message
    assert "cop" in message


def test_ordinary_record_does_not_inherit_copy_axes_from_sibling_record():
    probe = pas.rename("pas_mixed_copy_layouts")
    pop = dn.Population(N=2, C=3, v_init=-65.0, dtype=DTYPE)
    pop[:, 0].insert(
        probe,
        alias="copied",
        copies=2,
        g=1.0e-5,
        e=-65.0,
    )
    pop[:, 1:].insert(
        probe,
        alias="ordinary",
        g=torch.tensor([[2.0e-5]], dtype=DTYPE),
        e=-65.0,
    )

    with pytest.raises(ValueError, match="one-dimensional indexed target"):
        pop.build()
