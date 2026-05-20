"""Tests for ``dendra.models.mechanisms._ions``.

These tests aim for high branch & line coverage of the public helpers,
context-managers, and the ``Ion`` module class.  A single automatic fixture
preserves **all** global state so that tests remain isolated and repeatable.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from dendra.models.mechanisms import _ions as ions  # noqa: E402  (after torch)

# -----------------------------------------------------------------------------
# System‑under‑test
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Utility fixtures – keep the *module-level* dictionaries pristine between tests
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _preserve_global_dictionaries():
    """Snapshot & restore the mutable global dicts and the cached ``_last``.

    Because the helpers/contexts mutate *module* globals, we deep‑copy their
    contents at setup and fully restore them afterwards so that every test
    sees the canonical starting state regardless of execution order.
    """

    rev_backup = copy.deepcopy(ions.REVERSAL)
    val_backup = copy.deepcopy(ions.VALENCES)
    cin_backup = copy.deepcopy(ions.CINIT)

    # also reset the cached "last" updates used by the context managers
    ions.equilibria._last = {}
    ions.concentrations._last = {}

    yield

    ions.REVERSAL.clear()
    ions.REVERSAL.update(rev_backup)
    ions.VALENCES.clear()
    ions.VALENCES.update(val_backup)
    ions.CINIT.clear()
    ions.CINIT.update(cin_backup)
    ions.equilibria._last = {}
    ions.concentrations._last = {}


# -----------------------------------------------------------------------------
# Helper / registration functions
# -----------------------------------------------------------------------------


def test_valid_helpers_and_registration():
    """Smoke-test helper accessors and dynamic ``ion_register``."""

    # canonical ions
    assert set(ions.valid_ions()) == {"na", "k", "ca"}
    assert set(ions.reversals()) == {"ena", "ek", "eca"}

    # valid_concentrations drops the trailing "0"
    expected_concs = {key[:-1] for key in ions.CINIT}
    assert set(ions.valid_concentrations()) == expected_concs

    # register a brand‑new ion and verify *all* tables updated coherently
    ions.ion_register("cl", -1.0, -65.0, 4.0, 140.0)
    assert "cl" in ions.valid_ions()
    assert ions.REVERSAL["ecl"] == -65.0
    assert ions.CINIT["cli0"] == 4.0 and ions.CINIT["clo0"] == 140.0


# -----------------------------------------------------------------------------
# Context managers:  equilibria / concentrations
# -----------------------------------------------------------------------------


def test_equilibria_context_restores_values():
    original = copy.deepcopy(ions.REVERSAL)

    with ions.equilibria(ena=55.0, ek=-80.0):
        assert ions.REVERSAL["ena"] == 55.0
        assert ions.REVERSAL["ek"] == -80.0

    # after exit – identical to the original snapshot
    assert ions.REVERSAL == original


def test_equilibria_use_last_chain():
    """Two-stage update via the *use_last* cache."""

    assert ions.equilibria._last == {}
    with ions.equilibria(eca=140.0):
        pass  # first call stores into ``_last`` and applies immediately

    # second call pulls the cached update back out
    with ions.equilibria(use_last=True):
        assert ions.REVERSAL["eca"] == 140.0
    # ``_last`` should now be cleared
    assert ions.equilibria._last == {}


@pytest.mark.parametrize(
    "bad_key, ctx",
    [
        ("efoo", ions.equilibria),
        ("bar0", ions.concentrations),
    ],
)
def test_context_validation_raises(bad_key, ctx):
    with pytest.raises(ValueError):
        with ctx(**{bad_key: 123}):
            pass


# -----------------------------------------------------------------------------
# Ion module – initialization, Einitialisation, advance & detach
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("einit, eadvance", [(0, 0), (1, 0), (1, 1)])
def test_ion_initialize_sets_buffers(einit, eadvance):
    ion = ions.Ion(
        name="na",
        shape=(2, 3),
        cstyle=None,
        estyle=None,
        einit=einit,
        eadvance=eadvance,
        cinit=None,
    )

    # after *ctor* – every buffer must be present with the right shape
    for buf in ("ina", "ena", "nai", "nao"):
        assert getattr(ion, buf).shape == (2, 3)

    # ``initialize`` always resets to default REVERSAL / CINIT values
    ion.initialize(celsius=37.0)

    if einit == 0:
        # No Nernst recomputation – keep default reversal potential
        assert torch.allclose(
            getattr(ion, "ena"),
            torch.full((2, 3), ions.REVERSAL["ena"]),
        )
    else:
        # E initialized using the Nernst equation from default concentrations
        rzf = ion.rzf
        expected = (
            math.log(ions.CINIT["nao0"] / ions.CINIT["nai0"]) * rzf * (273.15 + 37.0)
        )
        assert torch.allclose(
            getattr(ion, "ena"),
            torch.full((2, 3), expected, dtype=getattr(ion, "ena").dtype),
            rtol=1e-5,
            atol=1e-5,
        )

    # for einit≠0 we expect Nernst calculation right after initialize
    if einit:
        rzf = ion.rzf
        expected = (
            math.log(ions.CINIT["nao0"] / ions.CINIT["nai0"]) * rzf * (273.15 + 37.0)
        )
        assert torch.allclose(
            getattr(ion, "ena"), torch.full((2, 3), expected), rtol=0, atol=1e-6
        )


@given(
    extra=st.floats(
        min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False
    ),
    intra=st.floats(
        min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
)
def test_ion_advance_updates_nernst(extra: float, intra: float):
    """Dynamic E-reversal tracking & concentration clamping."""

    ion = ions.Ion("na", (1,), None, None, einit=0, eadvance=1, cinit=None)

    # manually plant arbitrary (positive) concentrations
    getattr(ion, "nao")[0] = torch.tensor(extra)
    getattr(ion, "nai")[0] = torch.tensor(intra)

    ion.advance(celsius=25.0)

    rzf = ion.rzf
    expected = math.log(extra / intra) * rzf * (273.15 + 25.0)
    assert torch.allclose(
        getattr(ion, "ena"),
        torch.tensor([expected], dtype=getattr(ion, "ena").dtype),
        rtol=1e-5,
        atol=1e-5,
    )


def test_ion_advance_clamps_non_positive():
    ion = ions.Ion("na", (3,), None, None, einit=0, eadvance=1, cinit=None)

    getattr(ion, "nao").fill_(0.0)  # zero – should be clamped to 1e‑9
    getattr(ion, "nai").fill_(-2.0)  # negative – should be clamped too

    ion.advance(celsius=0.0)

    assert torch.all(getattr(ion, "nao") > 0)
    assert torch.all(getattr(ion, "nai") > 0)


def test_detach_clears_grad_history():
    ion = ions.Ion("na", (1,), None, None, einit=0, eadvance=0, cinit=None)

    # turn on autograd, then detach
    getattr(ion, "nai").requires_grad_(True)
    ion.detach()
    assert getattr(ion, "nai").grad_fn is None  # leaf, no history
