"""Delay bounds size event projections while preserving delivery and gradients."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import expsyn, pas

pytestmark = pytest.mark.cpu
DT = 0.125
DTYPE = torch.float64
GLOBAL_CAP = 16 * DT


def _network(declarations, *, taps=2, initialize=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        source = dn.SingleCompartment(N=4, C=1, integrator=dn.scnv(imem=False))
        post = dn.SingleCompartment(N=4, C=1, integrator=dn.scnv(imem=False))
        source.insert(pas, g=0.001, e=-65.0)
        post.insert(expsyn.rename("a"), e=0.0, tau=1.0)
        post.insert(expsyn.rename("b"), e=0.0, tau=1.0)
        network = dn.Network(
            {"source": source, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        for i, (name, delay, bound) in enumerate(declarations):
            network.connect_one_to_one(
                source[i : i + 1],
                post[i : i + 1],
                getattr(post.mech, name),
                threshold=None,
                weight=0.05,
                delay=delay,
                delay_upper_bound_ms=bound,
            )
        network.train()
        network.set_synaptic_diff_config(diff_delays=True, taps=taps)
        if initialize:
            network.initialize(DT, max_delay_ms=GLOBAL_CAP)
    return network


def test_bound_sizes_each_projection_and_survives_wiring_transforms():
    network = _network(
        (("a", DT, DT), ("a", 2 * DT, 2 * DT), ("b", DT, None)),
        initialize=False,
    )
    network = network.concat(packed=["source", "post"]).batch(2)
    network.initialize(DT, max_delay_ms=GLOBAL_CAP)
    a, b = network.synapses.values()
    assert a.delay_upper_bound_ms == 2 * DT
    assert a.max_delay_steps == 4
    assert b.delay_upper_bound_ms is None
    assert b.max_delay_steps == 17

    network.initialize(DT / 2, max_delay_ms=GLOBAL_CAP)
    a, b = network.synapses.values()
    assert a.max_delay_steps == 6
    assert b.max_delay_steps == 33


def test_group_with_old_unbounded_spec_uses_legacy_capacity():
    network = _network((("a", DT, DT), ("a", 2 * DT, None)), initialize=False)
    declarations = next(iter(network.synapse_spec.values()))
    declarations[1] = declarations[1][:8]
    network.initialize(DT, max_delay_ms=GLOBAL_CAP)
    (projection,) = network.synapses.values()
    assert projection.delay_upper_bound_ms is None
    assert projection.max_delay_steps == 17


@pytest.mark.parametrize("bound", (-DT, float("nan"), float("inf"), True))
def test_invalid_declared_bound_is_rejected(bound):
    with pytest.raises((TypeError, ValueError), match="delay_upper_bound_ms"):
        _network((("a", DT, bound),), initialize=False)


def test_global_limit_and_initialized_delay_are_validated():
    network = _network((("a", DT, None), ("a", DT, GLOBAL_CAP + DT)), initialize=False)
    with pytest.raises(ValueError, match="max_delay_ms"):
        network.initialize(DT, max_delay_ms=GLOBAL_CAP)
    with pytest.raises(ValueError, match="delay_upper_bound_ms"):
        _network((("a", 2 * DT, DT),))


def _functional_delivery(native, delay):
    plan, tensors = dn.func.make_functional_netcon(native)
    delays = torch.full_like(tensors.delays, delay, requires_grad=True)
    gates = torch.arange(1.0, 9.0, dtype=DTYPE).reshape(8, 1, 1).expand(8, 4, 1)
    state = tensors.state
    payloads = []
    for row in gates:
        state, payload = plan.step(tensors.weights, state, row, delays=delays)
        payloads.append(payload)
    trace = torch.stack(payloads)
    gradient = torch.autograd.grad(trace.square().sum(), delays)[0]
    return trace, gradient


def test_uppermost_two_tap_delay_matches_legacy_delivery_and_gradient():
    delays = 2 * DT
    (bounded,) = _network((("a", delays, delays),)).synapses.values()
    (legacy,) = _network((("a", delays, None),)).synapses.values()
    assert bounded.max_delay_steps == 4
    assert legacy.max_delay_steps == 17
    trace, gradient = _functional_delivery(bounded, delays)
    old_trace, old_gradient = _functional_delivery(legacy, delays)
    torch.testing.assert_close(trace, old_trace, rtol=0, atol=0)
    torch.testing.assert_close(gradient, old_gradient, rtol=0, atol=0)
    assert torch.count_nonzero(gradient)
    with pytest.raises(dn.func.FunctionalizationError, match="delay_upper_bound_ms"):
        _functional_delivery(bounded, delays + DT)


def test_three_taps_keep_the_legacy_depth():
    network = _network((("a", DT, DT),), taps=3)
    (projection,) = network.synapses.values()
    assert projection.max_delay_steps == 17


def test_network_rejects_updated_delays_outside_the_bound():
    network = _network((("a", DT, 2 * DT),))
    name, projection = next(iter(network.synapses.items()))
    functional, tensors = dn.func.make_functional_network(network)
    network.detach()
    with torch.no_grad():
        tensors.parameters["synapses"][name]["delay_ms.weights.0.param.rho"].fill_(0)
        projection.delay_ms.weights[0].param.rho.fill_(0)
    with pytest.raises(dn.func.FunctionalizationError, match="delay_upper_bound_ms"):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
    with pytest.raises(ValueError, match="delay_upper_bound_ms"):
        network.run(steps=2)


def test_switch_to_three_taps_preserves_pending_storage_until_initialize():
    network = _network((("a", DT, DT),))
    (projection,) = network.synapses.values()
    network.step()
    pending = projection.delivery_buffer

    network.set_synaptic_diff_config(diff_delays=True, taps=3)
    assert projection.delivery_buffer is pending
    with pytest.raises(RuntimeError, match="initialize"):
        network.step()

    network.initialize(DT, reinit_delays=False, reinit_weights=False)
    assert projection.max_delay_steps == 17
    network.step()
