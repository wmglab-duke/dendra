"""Committed RAND carry retains gradients to the destination's Parameters."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import DT, DTYPE, _close
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
