"""Borrowed waveform streams retain unrelated Network packet semantics."""

import copy

import pytest
import torch

import dendra as dn
from dendra.models.rng import RNGModule
from tests.func.test_network_rng_sharing import _network as _noise_network
from tests.func.test_stimulation_rng_sharing import (
    DT,
    STEPS,
    _close,
)
from tests.func.test_stimulation_rng_sharing import _environment as _environment
from tests.func.test_stimulation_rng_sharing import _network as _borrowed_network
from tests.func.test_stimulation_rng_sharing import (
    _population,
)

pytestmark = pytest.mark.cpu


def test_borrowed_network_preserves_unrelated_partial_noise_tapes():
    borrowed, _ = _borrowed_network()
    ordinary = _noise_network()
    source = dn.Network(
        {
            **borrowed.populations,
            **{
                f"ordinary_{name}": population
                for name, population in ordinary.populations.items()
            },
        }
    )
    source.train()
    source.initialize(DT)
    for population in source.populations.values():
        population.detach()
    functional, tensors = dn.func.make_functional_network(source)
    noise, stimulation = functional._stimulation_rng_plan.sample_tapes(
        tensors.state, STEPS
    )
    legacy = functional._rng_plan.sample_noise_tapes(tensors.state, STEPS)
    for name in ordinary.populations:
        _close(noise[f"ordinary_{name}"], legacy[f"ordinary_{name}"])
    partial = functional._stimulation_rng_plan.sample_tapes(
        tensors.state, STEPS, noise={"ordinary_left": noise["ordinary_left"]}
    )
    _close(partial, (noise, stimulation))


def test_network_rejects_untracked_registered_generator_in_compatible_target():
    population, _, pulse = _population()
    pulse.generator = torch.Generator().manual_seed(809)
    source = dn.Network({"cell": population})
    source.train()
    source.initialize(DT)
    functional, _ = dn.func.make_functional_network(source)
    assert not functional._stimulation_rng_plan.enabled
    target = copy.deepcopy(source)
    target.untracked = RNGModule(811, shape_p=(), shape_f=())
    for waveform in target.cell.mech.probe.injected_waveforms:
        for module in waveform.modules():
            if hasattr(module, "generator"):
                module.generator = target.untracked._generator_for("cpu")
    with pytest.raises(dn.func.FunctionalizationError, match="untracked registered"):
        functional.extract(target)


@pytest.mark.parametrize("invalid", ("source", "parameters", "state"))
def test_joint_initialization_validates_before_sampling(monkeypatch, invalid):
    source, _, _ = _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert functional._stimulation_rng_plan.enabled
    parameters, state = tensors.parameters, tensors.state
    if invalid == "source":
        source.mech.probe.stream.ignore_rng_on_load(True)
        error, message = dn.func.FunctionalizationError, "load policy"
    elif invalid == "parameters":
        parameters = dict(parameters)
        parameters.pop(next(iter(parameters)))
        error, message = KeyError, "parameter tree mismatch"
    else:
        state = {key: value for key, value in state.items() if key != "random"}
        error, message = KeyError, "state must contain exactly"
    sampled = []

    def forbidden_sampler(*args, **kwargs):
        sampled.append((args, kwargs))
        raise AssertionError("Initialization sampled before validating its operands")

    monkeypatch.setattr(
        functional._stimulation_rng_plan, "sample_initialization", forbidden_sampler
    )
    with pytest.raises(error, match=message):
        functional.initialize_host(
            parameters,
            tensors.constants,
            tensors.initialization,
            state=state,
        )
    assert not sampled
