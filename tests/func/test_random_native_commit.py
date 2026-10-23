"""Committed RAND carry retains gradients to the destination's Parameters."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_random_parameters import (
    _initialization_draws,
    _network,
    _RandomLeak,
    _StickyLeak,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
                yield
    finally:
        torch.set_num_threads(previous)


def _owners(network, mechanism):
    return tuple(
        getattr(population.mech, mechanism.__name__)
        for population in network.populations.values()
    )


def _share_distribution_parameter(network, mechanism):
    left, right = _owners(network, mechanism)
    right.g_mu_param = left.g_mu_param
    network.initialize(DT, reinit_weights=False, reinit_delays=False)
    # Retained samples must be authored against the shared Parameter, too.
    # Replacing a raw Parameter after sampling leaves its old graph intact.
    for population in network.populations.values():
        population.resample_random_parameters(force=True)


def _next_native_gradients(network, mechanism):
    owners = _owners(network, mechanism)
    parameters = tuple(dict.fromkeys(owner.g_mu_param for owner in owners))
    assert all(owner.g.requires_grad for owner in owners)
    network.step()
    loss = sum(
        population.v.square().sum() for population in network.populations.values()
    )
    gradients = torch.autograd.grad(loss, parameters)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )
    return gradients


@pytest.mark.parametrize("mechanism", (_RandomLeak, _StickyLeak))
@pytest.mark.parametrize("checkpoint", (False, True))
@pytest.mark.parametrize("shared_parameter", (False, True))
def test_native_continuation_after_random_commit_retains_destination_gradients(
    mechanism, checkpoint, shared_parameter
):
    native, target = _network(mechanism=mechanism), _network(mechanism=mechanism)
    if shared_parameter:
        for network in (native, target):
            _share_distribution_parameter(network, mechanism)

    parameter_identities = tuple(
        owner.g_mu_param for owner in _owners(target, mechanism)
    )
    functional, tensors = dn.func.make_functional_network(target)
    parameters = tensors.independent_parameters(("*g_mu_param",))
    assert all(
        value is not owner.g_mu_param
        for value, owner in zip(
            (
                parameters["populations"][name][
                    f"integrator.mech.mechanisms.{mechanism.__name__}.g_mu_param"
                ]
                for name in target.populations
            ),
            _owners(target, mechanism),
            strict=True,
        )
    )
    initialized = functional.initialize(
        parameters,
        tensors.constants,
        tensors.state,
        _initialization_draws(functional, tensors),
    )
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    # Prefix voltages carry no parameter graph; the next-step gradients must
    # therefore pass through the retained sample on the destination model.
    with torch.no_grad():
        final, _ = functional.prepare_and_step(*initialized)
        native.step()
    functional.commit_state_(target, final)
    for actual, expected in zip(
        _owners(target, mechanism), _owners(native, mechanism), strict=True
    ):
        torch.testing.assert_close(actual.g, expected.g)
    _close(target.left.v, native.left.v)
    _close(target.right.v, native.right.v)

    if checkpoint:
        for network in (native, target):
            network.restore_dict_from_checkpoint(network.state_dict_for_checkpoint())

    assert all(
        owner.g_mu_param is parameter
        for owner, parameter in zip(
            _owners(target, mechanism), parameter_identities, strict=True
        )
    )
    if shared_parameter:
        left, right = _owners(target, mechanism)
        assert left.g_mu_param is right.g_mu_param
    _close(
        _next_native_gradients(target, mechanism),
        _next_native_gradients(native, mechanism),
    )


@pytest.mark.parametrize(
    "kind", ("population", "independent_network", "shared_network")
)
def test_runtime_rng_commit_restores_seed_and_suffix_despite_ignore_rng_on_load(kind):
    from dendra.models.rng import RNGModule
    from tests.func.test_network_rng_sharing import _network as shared_network
    from tests.func.test_network_rng_sharing import (
        _rollout_inputs,
    )
    from tests.func.test_network_runtime_noise import _SharedRandomNoiseLeak
    from tests.func.test_runtime_noise import _native_step, _population

    source = (
        _population(_SharedRandomNoiseLeak)
        if kind == "population"
        else shared_network(shared=kind == "shared_network")
    )
    for stream in source.modules():
        if type(stream) is RNGModule:
            stream.ignore_rng_on_load(True)
    native, target = copy.deepcopy(source), copy.deepcopy(source)
    target_streams = tuple(
        stream for stream in target.modules() if type(stream) is RNGModule
    )
    for index, stream in enumerate(target_streams):
        stream.reseed(9401 + index)
        stream.randn((5,), dtype=DTYPE)
    registered_aliases = tuple(
        (path, stream)
        for path, stream in target.named_modules(remove_duplicate=False)
        if type(stream) is RNGModule
    )
    source_before = _clone(source.state_dict_for_checkpoint(), detached=True)
    global_rng = torch.get_rng_state().clone()
    if kind == "population":
        functional, tensors = dn.func.make_functional(source, dt=DT)
        inputs = dn.func.RolloutInput(
            noise=functional.sample_noise_draw_tape(tensors.state, 3)
        )
        step = _native_step
    else:
        functional, tensors = dn.func.make_functional_network(source)
        inputs = _rollout_inputs(functional, tensors.state, 3)
        step = type(source).step
    final, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, inputs
    )
    for _ in range(3):
        step(native)
    functional.commit_state_(target, final)
    _close(functional.extract(target).state, final)
    _close(functional.extract(target).state, functional.extract(native).state)
    assert all(stream._ignore_rng_on_load for stream in target_streams)
    assert all(
        target.get_submodule(path) is stream for path, stream in registered_aliases
    )
    for actual, expected in zip(
        target_streams,
        (stream for stream in native.modules() if type(stream) is RNGModule),
        strict=True,
    ):
        _close(actual.get_extra_state(), expected.get_extra_state())
    # Both the numerical state and the next native draws must continue from
    # the committed suffix, even though ordinary state-dict RNG loads are ignored.
    step(target)
    step(native)
    _close(functional.extract(target).state, functional.extract(native).state)
    _close(source.state_dict_for_checkpoint(), source_before)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
