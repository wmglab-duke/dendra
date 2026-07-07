import pytest
import torch

import dendra as dn  # noqa: F401 - ensure Dendra configures Torch before torch import
from dendra.models.parametric import Parameterized
from dendra.models.random_parameters import (
    available_random_distributions,
    get_random_distribution,
    register_random_distribution,
)


def test_truncated_normal_distribution_is_registered():
    spec = get_random_distribution("truncated_normal")
    assert "truncated_normal" in available_random_distributions()
    assert tuple(spec.parameter_defaults) == ("mu", "sigma", "low", "high")
    assert spec.parameter_constraints["sigma"] == "positive"


def test_truncated_normal_samples_respect_tensor_bounds():
    class P(Parameterized):
        Parameterized.RANGERAND(
            "x",
            distribution="truncated_normal",
            mu=0.0,
            sigma=1.0,
            low=-0.25,
            high=0.50,
        )

    p = P(shape=(4, 128), shape_f=(4, 128), dtype=torch.float32)
    p.populate_parameter_buffers(random_generation=object())

    assert torch.all(p.x >= p.x_low)
    assert torch.all(p.x <= p.x_high)
    assert hasattr(p, "x_mu")
    assert hasattr(p, "x_sigma")
    assert hasattr(p, "x_low")
    assert hasattr(p, "x_high")


def test_register_random_distribution_and_declare_parameterized_randvar():
    def constant_sampler(parameters, rng, shape, *, device, dtype):
        return (
            torch.empty(shape, device=device, dtype=dtype).fill_(1.0)
            * parameters["value"]
        )

    register_random_distribution(
        "constant_for_test",
        {"value": 7.0},
        sampler=constant_sampler,
        overwrite=True,
    )

    class P(Parameterized):
        Parameterized.RANGERAND(
            "x",
            distribution="constant_for_test",
            value=3.5,
        )

    p = P(shape=(2, 5), shape_f=(2, 5), dtype=torch.float32)
    p.populate_parameter_buffers(random_generation=object())

    assert torch.allclose(p.x, torch.full_like(p.x, 3.5))
    assert hasattr(p, "x_value")
    assert hasattr(p, "x_value_param")


def test_register_random_distribution_rejects_duplicate_without_overwrite():
    def sampler(parameters, rng, shape, *, device, dtype):
        return torch.zeros(shape, device=device, dtype=dtype)

    register_random_distribution(
        "duplicate_for_test",
        {},
        sampler=sampler,
        overwrite=True,
    )
    with pytest.raises(ValueError, match="already registered"):
        register_random_distribution("duplicate_for_test", {}, sampler=sampler)


def test_register_random_distribution_constraints_are_validated():
    def sampler(parameters, rng, shape, *, device, dtype):
        return torch.zeros(shape, device=device, dtype=dtype)

    with pytest.raises(ValueError, match="unknown parameters"):
        register_random_distribution(
            "bad_constraint_name_for_test",
            {"x": 1.0},
            {"y": "positive"},
            sampler=sampler,
            overwrite=True,
        )

    with pytest.raises(ValueError, match="Invalid constraint"):
        register_random_distribution(
            "bad_constraint_kind_for_test",
            {"x": 1.0},
            {"x": "bounded"},
            sampler=sampler,
            overwrite=True,
        )
