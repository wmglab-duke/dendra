"""Network and packed-population contracts around mechanism graph rebuilds."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import expsyn, hh, pas

DTYPE = torch.float64
_IdentitySynapse = expsyn.rename("identity_guard_synapse")


class _DuplicateGlobalState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0 * x")
    State.GLOBAL(scale=2.0)


class _DuplicateGlobalMechanism(Mechanism):
    Mechanism.STATE(_DuplicateGlobalState)
    Mechanism.GLOBAL(scale=1.0)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.scale * v

    def i_with_conductance(self, v):
        return self.scale * v, self.scale + 0 * v


def _network_with_synapse():
    population = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    population.insert(_IdentitySynapse, e=0.0, tau=1.0)
    population.insert(pas, g=0.001, e=-70.0)
    netstim = dn.NetStim(N=1, dtype=DTYPE)
    network = dn.Network({"post": population}, netstim=netstim)
    return network, population


def test_live_synapse_identity_check_supports_module_dict_registry():
    network, population = _network_with_synapse()

    network.connect_one_to_one(
        network.netstim[:],
        population[:, 0],
        population.mech.identity_guard_synapse,
        threshold=None,
        weight=0.1,
        delay=0.1,
    )
    network.build(0.1)

    assert len(network.synapses) == 1


def test_structural_rebuild_rejects_stale_synapse_object_and_wiring_spec():
    network, population = _network_with_synapse()
    stale_synapse = population.mech.identity_guard_synapse
    network.connect_one_to_one(
        network.netstim[:],
        population[:, 0],
        stale_synapse,
        threshold=None,
        weight=0.1,
        delay=0.1,
    )
    network.build(0.1)

    # Deleting an unrelated mechanism still replaces the complete compiled
    # mechanism graph, so every previously retained mechanism object is stale.
    population.delete(pas)
    population.initialize()
    assert population.mech.identity_guard_synapse is not stale_synapse

    with pytest.raises(RuntimeError, match="stale or does not belong"):
        network.connect_one_to_one(
            network.netstim[:],
            population[:, 0],
            stale_synapse,
            threshold=None,
            weight=0.1,
            delay=0.1,
        )

    with pytest.raises(RuntimeError, match=r"clear_synapses\(\)"):
        network.build(0.1, force_rebuild=True)


def test_concat_models_preserves_cropped_everywhere_support_and_parameters():
    left = dn.Population(N=1, C=3, v_init=-65.0, dtype=DTYPE)
    left.insert(pas, g=0.012, e=-66.0)
    left[:, 1].delete(pas)

    right = dn.Population(N=1, C=2, v_init=-65.0, dtype=DTYPE)
    right.insert(pas, g=0.034, e=-77.0)

    packed = dn.concat_models({"left": left, "right": right})
    packed.initialize()
    mechanism = packed.mech.pas

    assert mechanism.key.tolist() == [0, 2, 3, 4]
    torch.testing.assert_close(
        mechanism.g,
        torch.tensor([0.012, 0.012, 0.034, 0.034], dtype=DTYPE),
    )
    torch.testing.assert_close(
        mechanism.e,
        torch.tensor([-66.0, -66.0, -77.0, -77.0], dtype=DTYPE),
    )


def test_repeated_everywhere_crop_keeps_projected_tensor_values_flat():
    population = dn.Population(N=2, C=3, v_init=-65.0, dtype=DTYPE)
    population.insert(
        pas,
        g=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE),
        e=-70.0,
    )

    population[:, 1].delete(pas)
    population[:, 2].delete(pas)
    population.initialize()

    # The two survivors form a rectangular (2, 1) region in population
    # coordinates, but projected values use one entry per sparse mechanism
    # slot. Rebuild must therefore retain the flat sparse representation.
    assert population.mech.pas.key.tolist() == [0, 3]
    torch.testing.assert_close(
        population.mech.pas.g,
        torch.tensor([1.0, 4.0], dtype=DTYPE),
    )


def test_repeated_sparse_crop_keeps_projected_initial_conditions_flat():
    population = dn.Population(N=2, C=3, v_init=-65.0, dtype=DTYPE)
    population[:, :].insert(
        hh,
        ic={
            "m": torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                dtype=DTYPE,
            )
        },
    )

    population[:, 1].delete(hh)
    population[:, 2].delete(hh)
    population.initialize()

    assert population.mech.hh.key.tolist() == [0, 3]
    torch.testing.assert_close(
        population.mech.hh.m,
        torch.tensor([0.1, 0.4], dtype=DTYPE),
    )


def test_concat_models_preserves_compatible_global_mechanism_values():
    active = dn.Population(N=1, C=3, v_init=-65.0, dtype=DTYPE)
    active.insert(hh, gnabar=0.222, gkbar=0.111)
    active[:, 1].delete(hh)
    passive = dn.Population(N=1, C=1, v_init=-65.0, dtype=DTYPE)

    packed = dn.concat_models({"active": active, "passive": passive})
    packed.initialize()

    assert packed.mech.hh.key.tolist() == [0, 2]
    torch.testing.assert_close(
        packed.mech.hh.gnabar,
        torch.tensor(0.222, dtype=DTYPE),
    )
    torch.testing.assert_close(
        packed.mech.hh.gkbar,
        torch.tensor(0.111, dtype=DTYPE),
    )


def test_concat_models_rejects_conflicting_global_mechanism_values():
    left = dn.Population(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    right = dn.Population(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    left.insert(hh, gnabar=0.2)
    right.insert(hh, gnabar=0.3)

    with pytest.raises(ValueError, match="different class-wide GLOBAL value"):
        dn.concat_models({"left": left, "right": right})


def test_concat_models_rejects_explicit_global_that_changes_an_omitted_default():
    explicit = dn.Population(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    default = dn.Population(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    explicit.insert(hh, gnabar=0.2)
    default.insert(hh)

    with pytest.raises(ValueError, match="different declared default"):
        dn.concat_models({"explicit": explicit, "default": default})


def test_concat_models_accepts_equal_scalar_global_across_representations():
    explicit = dn.Population(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    default = dn.Population(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    explicit.insert(hh, gnabar=torch.tensor(0.12, dtype=DTYPE))
    default.insert(hh)

    packed = dn.concat_models({"explicit": explicit, "default": default})
    packed.initialize()

    torch.testing.assert_close(
        packed.mech.hh.gnabar,
        torch.tensor(0.12, dtype=DTYPE),
    )


def test_concat_models_preserves_owner_specific_duplicate_global_defaults():
    left = dn.Population(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    right = dn.Population(N=1, C=1, v_init=-65.0, dtype=DTYPE)
    left.insert(_DuplicateGlobalMechanism)
    right.insert(_DuplicateGlobalMechanism)

    packed = dn.concat_models({"left": left, "right": right})
    packed.initialize()

    mechanism = packed.mech._DuplicateGlobalMechanism
    torch.testing.assert_close(
        mechanism.scale,
        torch.tensor(1.0, dtype=DTYPE),
    )
    torch.testing.assert_close(
        mechanism.DE._DuplicateGlobalState.scale,
        torch.tensor(2.0, dtype=DTYPE),
    )
