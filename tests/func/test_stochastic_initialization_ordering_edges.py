"""Stochastic initialization keeps source graphs across authored raw writes."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra.models.mod import pas
from tests.func.test_stochastic_initialization_ordering import (
    DT,
    _Affine,
)
from tests.func.test_stochastic_initialization_ordering import (
    _environment as _environment,
)
from tests.func.test_stochastic_initialization_ordering import (
    _RandomPopulation,
)

pytestmark = pytest.mark.cpu


class _SqrtMean(torch.nn.Module):
    def forward(self, value, raw):
        return value * raw.sqrt()


class _LinearCapacitance(torch.nn.Module):
    def forward(self, value, raw):
        return value + 0.1 * raw


class _AffineSource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.7, dtype=torch.float64))

    def forward(self, value):
        return self.scale * value + 0.03


def _base_population():
    source = _RandomPopulation(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
    source.insert(pas)
    return source


def _configure(source):
    source.train()
    source.initialize()
    source.integrator._initialize(source, source.t.new_tensor(DT), force=True)


def _singular_population():
    source = _base_population()
    source.register_parameter(
        "mean_source", torch.nn.Parameter(torch.tensor(4.0, dtype=torch.float64))
    )
    source.register_parametrization_in_graph(
        "trial_mu", _SqrtMean(), args=("mean_source",)
    )
    source.register_parametrization_in_graph(
        "cm", _LinearCapacitance(), args=("mean_source",)
    )
    _configure(source)
    # Register after establishing a valid source snapshot. The first fresh
    # initialization samples sqrt(4) before replacing the incoming raw source.
    source.register_pre_initialize_transform(
        "zero",
        _Affine(),
        reads=("parameters.mean_source",),
        writes=("parameters.mean_source",),
        inputs={
            "factor": torch.tensor(0.0, dtype=torch.float64),
            "offset": torch.tensor(0.0, dtype=torch.float64),
        },
    )
    return source


def test_held_root_sample_avoids_inactive_singular_source_and_keeps_physical_graph():
    source, native = _singular_population(), _singular_population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    offset = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    inputs = tensors.initialization._replace(
        random_draw=draw,
        transforms={
            "pre.zero.factor": torch.tensor(0.0, dtype=torch.float64),
            "pre.zero.offset": offset,
        },
    )
    initial = functional.initialize(
        tensors.parameters, tensors.constants, inputs, state=tensors.state
    )
    prepared = functional.prepare(initial.parameters, initial.constants).values
    native.initialize()
    torch.testing.assert_close(prepared["population"]["trial"], native.trial)
    torch.testing.assert_close(prepared["population"]["cm"], native.cm)
    derivatives = torch.autograd.grad(
        prepared["population"]["trial"] + prepared["population"]["cm"].sum(),
        (tensors.parameters["mean_source"], offset),
    )
    torch.testing.assert_close(derivatives[0], offset.new_tensor(0.05))
    torch.testing.assert_close(derivatives[1], offset.new_tensor(0.1))


def _parametrized_population():
    source = _base_population()
    register_parametrization(source, "trial_mu_param", _AffineSource())
    _configure(source)
    source.register_pre_initialize_transform(
        "scale",
        _Affine(),
        reads=("parameters.parametrizations.trial_mu_param.original",),
        writes=("parameters.parametrizations.trial_mu_param.original",),
        inputs={
            "factor": torch.tensor(2.0, dtype=torch.float64),
            "offset": torch.tensor(0.1, dtype=torch.float64),
        },
    )
    return source


def test_parametrized_root_source_samples_before_original_parameter_write():
    source, native = _parametrized_population(), _parametrized_population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initial = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    prepared = functional.prepare(initial.parameters, initial.constants).values
    native.initialize()
    torch.testing.assert_close(prepared["population"]["trial"], native.trial)
    raw = tensors.parameters["parametrizations.trial_mu_param.original"]
    scale = tensors.parameters["parametrizations.trial_mu_param.0.scale"]
    derivatives = torch.autograd.grad(prepared["population"]["trial"], (raw, scale))
    torch.testing.assert_close(derivatives[0], scale)
    torch.testing.assert_close(derivatives[1], raw)
