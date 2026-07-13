"""Tests for ``dendra.models.mechanisms._ions``.

These tests aim for high branch & line coverage of the public helpers,
context-managers, and the ``Ion`` module class.  A single automatic fixture
preserves **all** global state so that tests remain isolated and repeatable.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from dendra.models.mechanisms import _ions as ions  # noqa: E402  (after torch)
from dendra.models.mechanisms._material_process import ExchangeProcess

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


def test_ion_field_metadata_and_exchange_geometry_follow_physical_domains():
    shape = (1, 2)
    ion = ions.Ion("na", shape, einit=0, eadvance=0)

    expected = {
        "ina": ("membrane", "mA/cm²", False, None),
        "ena": ("membrane", "mV", False, None),
        "nai": ("intracellular", "mM", True, ions.MIN_CONCENTRATION["na"]),
        "nao": ("extracellular", "mM", True, ions.MIN_CONCENTRATION["na"]),
    }
    for field, (domain, units, conserved, min_value) in expected.items():
        spec = ion.field_spec(field)
        assert spec.domain == domain
        assert spec.units == units
        assert spec.conserved is conserved
        assert spec.min_value == min_value

    class IonExchange(ExchangeProcess):
        ExchangeProcess.METHOD("exact", require_volumes=True)
        ExchangeProcess.EXCHANGE("na.nai", "na.nao", rate=0.5)

    volume_i = torch.tensor([[1.0, 2.0]])
    volume_o = torch.tensor([[3.0, 4.0]])
    population = SimpleNamespace(
        shape=shape,
        volume_i=volume_i,
        volume_o=volume_o,
    )
    process = IonExchange(
        name="ion_exchange",
        celsius=torch.tensor(37.0),
        diameters=torch.ones(shape),
        shape=shape,
        shape_f=shape,
    ).bind_materials({"na": ion}.__getitem__, population=population)

    ion.nai.copy_(torch.tensor([[10.0, 20.0]]))
    ion.nao.copy_(torch.tensor([[100.0, 200.0]]))
    mass_before = volume_i * ion.nai + volume_o * ion.nao
    process.advance_materials(0.2)

    torch.testing.assert_close(
        volume_i * ion.nai + volume_o * ion.nao,
        mass_before,
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.equal(ion.nai, torch.tensor([[10.0, 20.0]]))
    assert not torch.equal(ion.nao, torch.tensor([[100.0, 200.0]]))


@pytest.mark.parametrize("einit, eadvance", [(0, 0), (1, 0), (1, 1)])
def test_ion_initialize_sets_buffers(einit, eadvance):
    ion = ions.Ion(
        name="na",
        shape=(2, 3),
        einit=einit,
        eadvance=eadvance,
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

    ion = ions.Ion("na", (1,), einit=0, eadvance=1)

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
    ion = ions.Ion("na", (3,), einit=0, eadvance=1)

    getattr(ion, "nao").fill_(0.0)  # zero – should be clamped to 1e‑9
    getattr(ion, "nai").fill_(-2.0)  # negative – should be clamped too

    ion.advance(celsius=0.0)

    assert torch.all(getattr(ion, "nao") > 0)
    assert torch.all(getattr(ion, "nai") > 0)


def test_detach_clears_grad_history():
    ion = ions.Ion("na", (1,), einit=0, eadvance=0)

    # turn on autograd, then detach
    getattr(ion, "nai").requires_grad_(True)
    ion.detach()
    assert getattr(ion, "nai").grad_fn is None  # leaf, no history
