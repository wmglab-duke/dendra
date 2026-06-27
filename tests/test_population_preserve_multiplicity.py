"""Regression tests for colocated point-process bank insertion."""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import pas


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
