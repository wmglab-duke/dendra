import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State


class RandomRangeMechanism(Mechanism):
    Mechanism.RANGERAND(
        "rvar",
        distribution="normal",
        mu=0.0,
        sigma=1.0,
        seed=123,
    )

    def i(self, v):
        return torch.zeros_like(v)


def test_rangerand_creates_sample_buffer_and_distribution_parameters():
    mech = RandomRangeMechanism(
        None,
        torch.tensor(36.0),
        torch.ones(1, 4),
        (1, 4),
        (1, 4),
    )

    assert tuple(mech.rvar.shape) == (1, 4)
    assert tuple(mech.rvar_mu.shape) == (1, 4)
    assert tuple(mech.rvar_sigma.shape) == (1, 4)
    assert hasattr(mech, "rvar_mu_param")
    assert hasattr(mech, "rvar_sigma_param")
    assert hasattr(mech, "rvar_rng")
    assert "rvar_mu" in mech.all_parameter_names()
    assert "rvar_sigma" in mech.all_parameter_names()
    assert mech.random_parameter_names() == ["rvar"]


def test_rangerand_alias_overrides_scatter_to_inserted_region():
    class NearlyDeterministic(Mechanism):
        Mechanism.RANGERAND(
            "rvar",
            distribution="normal",
            mu=0.0,
            sigma=1.0e-8,
            seed=11,
        )

        def i(self, v):
            return torch.zeros_like(v)

    cell = dn.SingleCompartment(N=1, C=4, dtype=torch.float64)
    # Canonical distribution-parameter overrides follow ordinary RANGE override
    # rules, including alias-specific local parameters for restricted regions.
    cell[:, :2].insert(
        NearlyDeterministic, alias="left", rvar_mu=2.0, rvar_sigma=1.0e-8
    )
    cell[:, 2:].insert(
        NearlyDeterministic, alias="right", rvar_mu=3.0, rvar_sigma=1.0e-8
    )
    cell.build()
    mech = cell.mech.NearlyDeterministic
    mech.populate(random_generation=object())

    expected_mu = torch.tensor([[2.0, 2.0, 3.0, 3.0]], dtype=mech.rvar_mu.dtype)
    assert torch.allclose(mech.rvar_mu, expected_mu)
    assert torch.allclose(mech.rvar_sigma, torch.full_like(mech.rvar_sigma, 1.0e-8))
    assert torch.allclose(mech.rvar, expected_mu, atol=1.0e-5)
    assert hasattr(mech, "rvar_mu_left")
    assert hasattr(mech, "rvar_sigma_left")
    assert hasattr(mech, "rvar_mu_right")
    assert hasattr(mech, "rvar_sigma_right")


def test_bare_distribution_override_is_allowed_when_unambiguous():
    class SingleRandom(Mechanism):
        Mechanism.RANGERAND("noise", distribution="normal", mu=0.0, sigma=1e-8)

        def i(self, v):
            return torch.zeros_like(v)

    cell = dn.SingleCompartment(N=1, C=1, dtype=torch.float64)
    cell.insert(SingleRandom, alias="only", mu=3.0, sigma=1e-8)
    cell.build()
    mech = cell.mech.SingleRandom
    mech.populate(random_generation=object())

    assert torch.allclose(mech.noise, torch.full_like(mech.noise, 3.0), atol=1e-5)


def test_bare_distribution_override_raises_when_ambiguous():
    class TwoRandoms(Mechanism):
        Mechanism.RANGERAND("a", distribution="normal", mu=0.0, sigma=1.0)
        Mechanism.RANGERAND("b", distribution="normal", mu=0.0, sigma=1.0)

        def i(self, v):
            return torch.zeros_like(v)

    cell = dn.SingleCompartment(N=1, C=1)
    try:
        cell.insert(TwoRandoms, mu=1.0)
        cell.build()
    except ValueError as exc:
        assert "Ambiguous random distribution override" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ambiguous random distribution override did not raise")


def test_random_generation_prevents_double_resample_within_one_initialize():
    class Seeded(Mechanism):
        Mechanism.RANGERAND("r", distribution="normal", mu=0.0, sigma=1.0, seed=123)

        def i(self, v):
            return torch.zeros_like(v)

    cell = dn.SingleCompartment(N=1, C=1)
    cell.insert(Seeded)
    cell.build()

    # Mechanism construction populates once. A model initialize should then redraw
    # once, not twice, even though the mechanism handler initializes mechanisms twice.
    generator = torch.Generator(device="cpu").manual_seed(123)
    _first_constructor_draw = torch.randn((1, 1), generator=generator)
    expected_one_initialize_draw = torch.randn((1, 1), generator=generator)
    unexpected_double_initialize_draw = torch.randn((1, 1), generator=generator)

    cell.initialize()
    sample = cell.mech.Seeded.r.detach().cpu()
    assert torch.allclose(sample, expected_one_initialize_draw)
    assert not torch.allclose(sample, unexpected_double_initialize_draw)


def test_resample_on_initialize_false_keeps_value_across_initialize():
    class Sticky(Mechanism):
        Mechanism.RANGERAND(
            "r",
            distribution="normal",
            mu=0.0,
            sigma=1.0,
            seed=7,
            resample_on_initialize=False,
        )

        def i(self, v):
            return torch.zeros_like(v)

    cell = dn.SingleCompartment(N=1, C=2)
    cell.insert(Sticky)
    cell.initialize()
    first = cell.mech.Sticky.r.clone()
    cell.initialize()
    second = cell.mech.Sticky.r.clone()
    assert torch.equal(first, second)

    cell.resample_random_parameters("r")
    assert not torch.equal(first, cell.mech.Sticky.r)


def test_globalrand_batchrand_and_lognormal_uniform_shapes():
    class Mixed(Mechanism):
        Mechanism.GLOBALRAND("g", distribution="uniform", low=-1.0, high=1.0, seed=1)
        Mechanism.BATCHRAND("b", distribution="lognormal", mu=0.0, sigma=0.1, seed=2)

        def i(self, v):
            return torch.zeros_like(v)

    mech = Mixed(None, torch.tensor(36.0), torch.ones(3, 5), (3, 5), (3, 5))
    mech.populate(random_generation=object())

    assert tuple(mech.g.shape) == ()
    assert tuple(mech.b.shape) == (3, 1)
    assert torch.all(mech.g >= -1.0)
    assert torch.all(mech.g <= 1.0)
    assert torch.all(mech.b > 0.0)


def test_state_rangerand_works_in_nested_state():
    class Gate(State):
        State.STATE("m")
        State.ASSIGNED("minf", "tau")
        State.RANGERAND("gate_noise", distribution="normal", mu=1.0, sigma=1e-8)
        State.DERIVATIVE("m' = (minf - m) / tau")

        def assigned_values(self, v, values):
            del values
            return {
                "minf": torch.sigmoid(v * 0.0),
                "tau": self.gate_noise.expand_as(v),
            }

    class WithState(Mechanism):
        Mechanism.STATE_BUNDLE(Gate)

        def i(self, v):
            return torch.zeros_like(v)

    mech = WithState(None, torch.tensor(36.0), torch.ones(1, 3), (1, 3), (1, 3))
    mech.populate(random_generation=object())
    state = next(iter(mech.DE.values()))

    assert tuple(state.gate_noise.shape) == (1, 3)
    assert torch.allclose(
        state.gate_noise, torch.ones_like(state.gate_noise), atol=1e-5
    )
