#!/usr/bin/env python3
"""Replay one-step runtime draws through native and functional execution.

Run from the repository root::

    python examples/stochastic/shared_runtime_draws.py

The bounded CPU example covers NOISE and State SDE draws for a Population,
then shared Population streams and NetStim in a Network. It checks recordings,
parameter gradients and automatic native continuation after explicit replay.
"""

from __future__ import annotations

import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import expsyn

DT, STEPS = 0.025, 3
G = "integrator.mech.mechanisms.leak.g_param"


class Gate(State):
    State.STATE("x")
    State.RANGE(theta=0.7, mu=0.2, sigma=0.04)
    State.DERIVATIVE("x' = theta * (mu - x)")
    State.DIFFUSION("x = sigma")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.full_like(v, 0.4)}


class NoisyLeak(Mechanism):
    Mechanism.STATE_BUNDLE(Gate)
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00002, seed=19, rng_name="stream")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.eta + 0.002 * self.x


def cell(*, synapse=False):
    population = dn.SingleCompartment(
        N=1, C=2, v_init=-55.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.insert(NoisyLeak.rename("leak"))
    if synapse:
        population.insert(expsyn.rename("syn"), tau=2.0, e=0.0)
    population.train()
    population.initialize()
    population.mech.leak.DE.Gate.x_dW_rng = population.mech.leak.stream
    population.initialize()
    return population


def make_model(network):
    if not network:
        return cell()
    left, right = cell(), cell(synapse=True)
    right.mech.leak.stream = left.mech.leak.stream
    right.mech.leak.DE.Gate.x_dW_rng = left.mech.leak.stream
    stimulus = dn.NetStim(
        N=1, interval=0.075, noise=0.4, start=0.0, max_spikes=8, tau=DT, seed=71
    )
    model = dn.Network({"left": left, "right": right}, netstim=stimulus)
    model.connect_one_to_one(
        stimulus[:], right[:, 0], right.mech.syn, threshold=None, weight=0.002, delay=DT
    )
    model.train()
    model.initialize(DT)
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=True,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    return model


def clone(value):
    return pytree.tree_map_only(
        torch.Tensor, lambda tensor: tensor.detach().clone(), value
    )


def loss(state, network):
    populations = state["populations"].values() if network else (state,)
    return sum(
        population["integrator"]["v"].square().mean() for population in populations
    )


def advance(model, network, index, *, draws=None, callbacks, carry):
    options = {"callbacks": callbacks, "callback_state": carry}
    if draws is not None:
        options["draws"] = draws
    if network:
        if index % 3 == 0:
            return model.step(**options)
        if index % 3 == 1:
            return dn.step(model, **options)
        return dn.step_network(model, **options)
    if index % 3 == 0:
        return model.step(DT, **options)
    if index % 3 == 1:
        return dn.step(model, dt=DT, **options)
    return dn.step_population(model, dt=DT, **options)


def demonstrate(network):
    model = make_model(network)
    # region prepare
    # Lower once; sampling remains on the existing functional plan.
    functional, _ = dn.func.make_functional(model, dt=DT)
    boundary = dn.func.capture_continuation(functional, model)
    bound = functional.bind(boundary)
    fields = ("left.v", "right.v", "t") if network else ("v", "t")
    callbacks = functional.make_callbacks({"trace": dn.Recorder(fields, dt=2 * DT)})
    # endregion prepare
    expected = boundary.state
    native_carry = functional_carry = None
    global_rng = torch.get_rng_state().clone()
    for index in range(STEPS):
        # region replay
        current = dn.func.capture_continuation(functional, model)
        draws = functional.sample_runtime_draws(current.state)
        # Sampling owns no native advancement or RNG publication.
        torch.testing.assert_close(
            dn.func.capture_continuation(functional, model).state, current.state
        )
        frozen = clone(draws)
        results = advance(
            model, network, index, draws=draws, callbacks=callbacks, carry=native_carry
        )
        native_carry = results.state
        expected, auxiliary = bound(
            expected,
            inputs={} if network else None,
            draws=draws,
            callbacks=callbacks,
            callback_state=functional_carry,
        )
        functional_carry = auxiliary["callbacks"].state
        # endregion replay
        torch.testing.assert_close(results["trace"], auxiliary["callbacks"]["trace"])
        torch.testing.assert_close(functional.extract(model).state, expected)
        torch.testing.assert_close(draws, frozen)

    # region automatic
    # The next ordinary native step starts at the last packet's RNG suffix.
    current = dn.func.capture_continuation(functional, model)
    next_draws = functional.sample_runtime_draws(current.state)
    actual = advance(model, network, STEPS, callbacks=callbacks, carry=native_carry)
    expected, auxiliary = bound(
        expected,
        inputs={} if network else None,
        draws=next_draws,
        callbacks=callbacks,
        callback_state=functional_carry,
    )
    torch.testing.assert_close(actual["trace"], auxiliary["callbacks"]["trace"])
    torch.testing.assert_close(functional.extract(model).state, expected)
    # endregion automatic
    torch.testing.assert_close(torch.get_rng_state(), global_rng)
    populations = model.populations.values() if network else (model,)
    native_parameters = tuple(
        population.mech.leak.g_param for population in populations
    )
    functional_parameters = (
        tuple(values[G] for values in boundary.parameters["populations"].values())
        if network
        else (boundary.parameters[G],)
    )
    # Differentiate live native voltage; the captured functional sources own
    # their independent reference graph.
    populations = model.populations.values() if network else (model,)
    native_loss = sum(population.v.square().mean() for population in populations)
    observed = torch.autograd.grad(native_loss, native_parameters)
    wanted = torch.autograd.grad(loss(expected, network), functional_parameters)
    for actual_gradient, expected_gradient in zip(observed, wanted, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert actual_gradient.isfinite().all() and actual_gradient.abs().sum() > 0
    print(
        f"{'Network' if network else 'Population'}: explicit step replay, recordings, "
        "parameter gradients and automatic RNG continuation passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        demonstrate(False)
        demonstrate(True)


if __name__ == "__main__":
    main()
