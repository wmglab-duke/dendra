"""Compatible checkpoint loads keep each retained RAND realization paired."""

import copy

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=0, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"),
    ):
        yield


def _equal(actual, expected):
    leaves, structure = _pytree.tree_flatten(actual)
    references, expected_structure = _pytree.tree_flatten(expected)
    assert structure == expected_structure
    for value, reference in zip(leaves, references, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        else:
            assert value == reference


class _CompatibilityGate(State):
    State.STATE("m")
    State.RANGERAND("local_random", seed=71, resample_on_initialize=False)
    State.GLOBALRAND("global_random", seed=72, resample_on_initialize=False)
    State.DERIVATIVE("m' = (0.25 - m) / (local_random + global_random)")

    def state_defaults(self, v, values):
        del values
        return {"m": torch.full_like(v, 0.9)}


class _CompatibilityLeak(Mechanism):
    Mechanism.STATE_BUNDLE(_CompatibilityGate)
    Mechanism.RANGERAND("local_random", seed=73, resample_on_initialize=False)
    Mechanism.GLOBALRAND("global_random", seed=74, resample_on_initialize=False)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        conductance = self.local_random * self.m
        return conductance, conductance * self.global_random


class _CompatibilityPopulation(dn.SingleCompartment):
    dn.SingleCompartment.RANGERAND(
        "local_random", seed=75, resample_on_initialize=False
    )
    dn.SingleCompartment.GLOBALRAND(
        "global_random", seed=76, resample_on_initialize=False
    )


class _OtherCompatibilityPopulation(dn.SingleCompartment):
    dn.SingleCompartment.GLOBALRAND(
        "other_random", seed=77, resample_on_initialize=False
    )
    dn.SingleCompartment.GLOBALRAND(
        "global_random", seed=78, resample_on_initialize=False
    )


def _owners(population):
    mechanism = population.mech._CompatibilityLeak
    return population, mechanism, next(iter(mechanism.DE.values()))


def _population(compartments, offset, population_type=_CompatibilityPopulation):
    population = population_type(
        N=1, C=compartments, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(_CompatibilityLeak)
    population.initialize()
    for index, owner in enumerate(_owners(population)):
        for name in owner.random_parameters:
            getattr(owner, f"{name}_mu_param").fill_(offset + index)
        owner.populate_parameter_buffers()
        owner._sample_random_parameters(force=True)
    return population


def _refresh_effective_parameters(population):
    # Compatible loads can replace scalar raw leaves while skipping their
    # shape-dependent effective buffers. Refresh those ordinary views without
    # changing any retained samples.
    for owner in _owners(population):
        owner._install_parameter_buffers(owner._derive_parameter_buffers())


@pytest.mark.parametrize("legacy", (False, True))
def test_shape_compatible_load_keeps_retained_samples_and_provenance(legacy):
    source, target = _population(3, 2.0), _population(5, 8.0)
    checkpoint = copy.deepcopy(source.state_dict())
    if legacy:
        checkpoint = {
            name: value
            for name, value in checkpoint.items()
            if not name.endswith("_random_parameter_state")
        }
    retained = tuple(
        (owner.local_random.clone(), owner._random_parameter_state_for_checkpoint())
        for owner in _owners(target)
    )

    target.load(checkpoint)

    for actual, incoming, (sample, provenance) in zip(
        _owners(target), _owners(source), retained, strict=True
    ):
        _equal(actual.local_random, sample)
        _equal(actual.global_random, incoming.global_random)
        restored = actual._random_parameter_state_for_checkpoint()
        for label in ("draws", "parameters"):
            _equal(restored[label]["local_random"], provenance[label]["local_random"])
            if legacy:
                assert "global_random" not in restored[label]
            else:
                _equal(
                    restored[label]["global_random"],
                    incoming._random_parameter_state_for_checkpoint()[label][
                        "global_random"
                    ],
                )
        # Scalar raw parameters retain the existing compatible-load policy.
        _equal(actual.local_random_mu_param, incoming.local_random_mu_param)

    _refresh_effective_parameters(target)
    if legacy:
        # A loaded legacy sample has no attested draw; explicitly resampling
        # just those GLOBAL samples makes the complete model lowerable again.
        target.resample_random_parameters("global_random")
    target.integrator._initialize(target, target.t.new_tensor(0.125), force=True)
    functional, tensors = dn.func.make_functional(target, dt=0.125)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    actual, _ = functional.step(tensors.parameters, prepared, tensors.state)
    target._step(target.integrator, target, target.t.new_tensor(0.125), None, None)
    _equal(actual["integrator"]["v"], target.v)


@pytest.mark.parametrize("owner_index", (0, 1, 2))
def test_partial_state_dict_keeps_provenance_for_omitted_samples(owner_index):
    source, target = _population(3, 2.0), _population(3, 8.0)
    incoming, owner = _owners(source)[owner_index], _owners(target)[owner_index]
    retained = owner._random_parameter_state_for_checkpoint()
    sample = owner.local_random.clone()
    owner.load_state_dict(
        {
            "global_random": incoming.global_random,
            "_random_parameter_state": incoming._random_parameter_state_for_checkpoint(),
        },
        strict=False,
    )
    _equal(owner.local_random, sample)
    _equal(owner.global_random, incoming.global_random)
    restored = owner._random_parameter_state_for_checkpoint()
    for label in ("draws", "parameters"):
        _equal(restored[label]["local_random"], retained[label]["local_random"])
        _equal(
            restored[label]["global_random"],
            incoming._random_parameter_state_for_checkpoint()[label]["global_random"],
        )


def test_compatible_load_ignores_provenance_for_different_random_declarations():
    source = _population(3, 2.0)
    target = _population(3, 8.0, _OtherCompatibilityPopulation)
    retained = target.other_random.clone()
    provenance = target._random_parameter_state_for_checkpoint()
    target.load(source.state_dict())
    _equal(target.other_random, retained)
    _equal(target.global_random, source.global_random)
    restored = target._random_parameter_state_for_checkpoint()
    for label in ("draws", "parameters"):
        _equal(restored[label]["other_random"], provenance[label]["other_random"])
        _equal(
            restored[label]["global_random"],
            source._random_parameter_state_for_checkpoint()[label]["global_random"],
        )
    _refresh_effective_parameters(target)
    functional, tensors = dn.func.make_functional(target, dt=0.125)
    functional.prepare(tensors.parameters, tensors.constants)


@pytest.mark.parametrize("compatible", (False, True))
def test_loaded_matching_sample_rejects_malformed_provenance(compatible):
    source, target = _population(3, 2.0), _population(3, 8.0)
    before = copy.deepcopy(target.state_dict())
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["_random_parameter_state"]["initialized"]["global_random"] = "bad"
    with pytest.raises(RuntimeError, match="initialized flags"):
        if compatible:
            target.load(checkpoint)
        else:
            target.load_state_dict(checkpoint)
    if compatible:
        _equal(target.state_dict(), before)
