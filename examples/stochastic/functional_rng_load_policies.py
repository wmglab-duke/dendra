"""Restore cached samples while retaining selected live random streams.

Run: python examples/stochastic/functional_rng_load_policies.py
"""

from __future__ import annotations

import copy

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism

DT = 0.125
STEPS = 4


# region declaration
class CachedNoisyLeak(Mechanism):
    Mechanism.GLOBALRAND(
        "g",
        mu=0.001,
        sigma=0.00005,
        seed=17,
        rng_name="kept",
        resample_on_initialize=False,
    )
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00002, seed=19, rng_name="kept")
    Mechanism.GLOBALNOISE(
        "e",
        distribution="uniform",
        low=-68.0,
        high=-62.0,
        seed=23,
        rng_name="restored",
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e.clone() - self.eta.clone()


# endregion declaration


def make_network():
    cells = {}
    for name in ("left", "right"):
        cell = dn.SingleCompartment(
            N=1, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(CachedNoisyLeak.rename("leak"))
        cell.initialize()
        cells[name] = cell
    left, right = cells["left"].mech.leak, cells["right"].mech.leak
    right.kept, right.restored = left.kept, left.restored
    right.g_mu_param = left.g_mu_param
    model = dn.Network(cells)
    model.train()
    model.initialize(DT)
    # region policy
    left.kept.ignore_rng_on_load(True)
    left.restored.ignore_rng_on_load(False)
    for _ in range(2):
        model.step()
    for cell in model.populations.values():
        cell.cache("_steady_state")
    left.kept.reseed(5101)
    left.restored.reseed(6101)
    for _ in range(2):
        model.step()
    # endregion policy
    return model


def clone_checkpoint(value):
    return torch.utils._pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def assert_same_checkpoint(actual, expected):
    values, layout = torch.utils._pytree.tree_flatten(actual)
    references, other = torch.utils._pytree.tree_flatten(expected)
    assert layout == other
    for value, reference in zip(values, references, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def main():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        source = make_network()
        native = copy.deepcopy(source)
        before = clone_checkpoint(source.state_dict_for_checkpoint())
        # region initialization
        runner = dn.compile(
            source,
            dt=DT,
            steps=STEPS,
            initialization="fresh",
            inference=True,
            backend="aot_eager",
        )
        functional = runner.functional
        tensors = functional.extract()
        packets = functional.sample_population_initialization_draws(
            tensors.parameters, tensors.state
        )
        initialization = runner.initialization_inputs._replace(
            populations={
                name: values._replace(random_draw=packets[name])
                for name, values in runner.initialization_inputs.populations.items()
            }
        )
        initial = functional.initialize(*tensors, initialization)
        native.initialize(DT)
        torch.testing.assert_close(initial.state, functional.extract(native).state)
        assert native.left.mech.leak.kept.get_extra_state()["base_seed"] == 5101
        assert native.left.mech.leak.restored.get_extra_state()["base_seed"] != 6101
        # endregion initialization
        # region continuation
        prefix, auxiliary = runner(initialization=initialization)
        final, _ = runner(
            prefix, parameters=auxiliary["parameters"], constants=auxiliary["constants"]
        )
        with torch.no_grad():
            for _ in range(2 * STEPS):
                native.step()
        torch.testing.assert_close(final, functional.extract(native).state)
        assert_same_checkpoint(source.state_dict_for_checkpoint(), before)
        # endregion continuation
        # region checkpoint
        target = copy.deepcopy(source)
        for stream in (target.left.mech.leak.kept, target.left.mech.leak.restored):
            stream.ignore_rng_on_load(True)
            stream.reseed(9001)
        functional.commit_state_(target, final)
        assert target.left.mech.leak.kept is target.right.mech.leak.kept
        checkpoint = clone_checkpoint(target.state_dict_for_checkpoint())
        target.step()
        target.restore_dict_from_checkpoint(checkpoint)
        torch.testing.assert_close(final, functional.extract(target).state)
        # endregion checkpoint
    print("Cached values, mixed RNG load policies, and exact continuation agree.")


if __name__ == "__main__":
    main()
