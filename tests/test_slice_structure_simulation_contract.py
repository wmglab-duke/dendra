"""Holistic CPU contracts for Slice-scoped model configuration.

These tests deliberately exercise complete passive simulations rather than
only inspecting queued insertion/parameterization metadata.  A structural
Slice operation is correct only when the resulting voltage trajectory remains
localized and reproducible across batching and rebuild lifecycle transitions.
"""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas

DTYPE = torch.float64
BATCH_SIZE = 3
N_CELLS = 2
N_COMPARTMENTS = 4
V_INIT = -70.0
DT = 0.01
N_STEPS = 2
TSTOP = DT * N_STEPS

CONFIGURATION_ORDERS = (
    "unbatched",
    "batch_before_configuration",
    "batch_after_configuration",
)


def _population() -> dn.Population:
    return dn.Population(
        N=N_CELLS,
        C=N_COMPARTMENTS,
        v_init=V_INIT,
        dtype=DTYPE,
    )


def _batch_before_configuration(population, order: str) -> None:
    if order == "batch_before_configuration":
        population.batch(BATCH_SIZE)


def _batch_after_configuration(population, order: str) -> None:
    if order == "batch_after_configuration":
        population.batch(BATCH_SIZE)


def _expand_core(population, core: torch.Tensor) -> torch.Tensor:
    """Expand an ``[N, C]`` known answer over current leading batch axes."""
    batch_rank = len(population.shape) - 2
    return core.reshape(*(1,) * batch_rank, *core.shape).expand(population.shape)


def _passive_backward_euler(v0, reversal, conductance) -> torch.Tensor:
    """Known answer for Dendra's default passive backward-Euler update."""
    v0 = torch.as_tensor(v0, dtype=DTYPE)
    reversal = torch.as_tensor(reversal, dtype=DTYPE)
    conductance = torch.as_tensor(conductance, dtype=DTYPE)
    decay = (1.0 + 1000.0 * conductance * DT) ** -N_STEPS
    return reversal + (v0 - reversal) * decay


def _run_and_check_reinitialize_and_rebuild(
    population,
    retained_mechanism_slice,
    expected_v: torch.Tensor,
    assert_configuration,
) -> None:
    """Check initialize, reinitialize, force-rebuild, and retained rebinding."""
    population.initialize()
    assert_configuration()
    assert torch.equal(population.v, torch.full_like(population.v, V_INIT))

    population.run(tstop=TSTOP, dt=DT)
    first = population.v.detach().clone()
    torch.testing.assert_close(first, expected_v, rtol=0.0, atol=1e-12)

    # Ordinary reinitialization must reset state without losing structural
    # configuration or changing the deterministic passive trajectory.
    population.initialize()
    assert_configuration()
    assert torch.equal(population.v, torch.full_like(population.v, V_INIT))
    population.run(tstop=TSTOP, dt=DT)
    torch.testing.assert_close(population.v, first, rtol=0.0, atol=0.0)

    # A force rebuild replaces compiled mechanisms.  The retained Slice must
    # bind to that new live object, and structural configuration must survive.
    old_mechanism = retained_mechanism_slice.model
    population.build(force_rebuild=True)
    assert retained_mechanism_slice.model is population.mech.pas
    assert retained_mechanism_slice.model is not old_mechanism

    population.initialize()
    assert_configuration()
    assert torch.equal(population.v, torch.full_like(population.v, V_INIT))
    population.run(tstop=TSTOP, dt=DT)
    torch.testing.assert_close(population.v, first, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("order", CONFIGURATION_ORDERS)
def test_slice_scoped_passive_insertion_is_local_and_lifecycle_stable(order):
    population = _population()

    # Retain the physical selection before any optional batching.  Using it to
    # configure the model afterward verifies that retained Slice rebasing is
    # not merely an inspection-time convenience.
    passive_region = population[:, 1:3]
    _batch_before_configuration(population, order)
    passive_region.insert(pas, g=0.02, e=-40.0)
    _batch_after_configuration(population, order)

    population.build()
    retained_pas = passive_region.mech.pas
    configured_g = retained_pas.g.reshape(-1)[0].item()

    core_expected = torch.full((N_CELLS, N_COMPARTMENTS), V_INIT, dtype=DTYPE)
    core_expected[:, 1:3] = _passive_backward_euler(V_INIT, -40.0, configured_g)
    expected_v = _expand_core(population, core_expected)

    def assert_configuration():
        assert passive_region.shape == population.v[..., 1:3].shape
        torch.testing.assert_close(
            retained_pas.g,
            torch.full(passive_region.shape, configured_g, dtype=DTYPE),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            retained_pas.e,
            torch.full(passive_region.shape, -40.0, dtype=DTYPE),
            rtol=0.0,
            atol=0.0,
        )

    _run_and_check_reinitialize_and_rebuild(
        population,
        retained_pas,
        expected_v,
        assert_configuration,
    )

    # Compartments outside the insertion Slice have no ionic current and must
    # stay at v_init exactly; this protects against dense-scatter leakage.
    assert torch.equal(
        population.v[..., [0, 3]],
        torch.full_like(population.v[..., [0, 3]], V_INIT),
    )
    assert bool(torch.all(population.v[..., 1:3] > V_INIT))


@pytest.mark.parametrize("order", CONFIGURATION_ORDERS)
@pytest.mark.parametrize("insertion_style", ["slice", "population"])
def test_slice_scoped_mechanism_parameterization_controls_only_its_region(
    order, insertion_style
):
    population = _population()
    whole_population = population[:]
    depolarized_region = population[:, 2]

    _batch_before_configuration(population, order)

    # Exercise both mechanism layouts: a full-region Slice insertion compiles
    # through the sparse/keyed path, while ordinary Population insertion uses
    # dense storage with no mapper. Slice parameterization must be structural
    # and rebuild-stable in either representation.
    if insertion_style == "slice":
        whole_population.insert(pas, g=0.01, e=-60.0)
    else:
        population.insert(pas, g=0.01, e=-60.0)
    population.build()
    retained_pas = depolarized_region.mech.pas
    retained_pas.parametrize("e", -30.0, alias="depolarized")
    configured_g = population.mech.pas.g.reshape(-1)[0].item()

    _batch_after_configuration(population, order)

    reversal = torch.full((N_CELLS, N_COMPARTMENTS), -60.0, dtype=DTYPE)
    reversal[:, 2] = -30.0
    core_expected = _passive_backward_euler(V_INIT, reversal, configured_g)
    expected_v = _expand_core(population, core_expected)

    def assert_configuration():
        assert depolarized_region.shape == population.v[..., 2].shape
        torch.testing.assert_close(
            retained_pas.e,
            torch.full(depolarized_region.shape, -30.0, dtype=DTYPE),
            rtol=0.0,
            atol=0.0,
        )
        full_e = whole_population.mech.pas.e
        expected_e = _expand_core(population, reversal)
        torch.testing.assert_close(full_e, expected_e, rtol=0.0, atol=0.0)

    _run_and_check_reinitialize_and_rebuild(
        population,
        retained_pas,
        expected_v,
        assert_configuration,
    )

    # All non-parameterized compartments have identical passive dynamics,
    # while only the selected compartment follows the more depolarized reversal.
    baseline = population.v[..., 0]
    for compartment in (1, 3):
        torch.testing.assert_close(
            population.v[..., compartment], baseline, rtol=0.0, atol=0.0
        )
    assert bool(torch.all(population.v[..., 2] > baseline))
