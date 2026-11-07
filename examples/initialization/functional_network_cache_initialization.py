"""Restart Network and individual Population caches with explicit continuation.

Run from the repository root: python examples/initialization/functional_network_cache_initialization.py
"""

from __future__ import annotations

import torch

import dendra as dn
from dendra.models.mod import expsyn, pas

DT = 0.125
STEPS = 4


def make_network():
    # region cache
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        cell = dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
        cell.insert(pas, g=0.001, e=-65.0)
        cell.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
        stimulus = dn.NetStim(
            N=1, interval=0.25, noise=0.0, start=0.0, max_spikes=100, seed=17
        )
        network = dn.Network(
            {"cell": cell},
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            stimulus[:], cell[:], cell.mech.syn, weight=0.003, delay=3 * DT
        )
        network.steady_state(tstop=1.0, dt=DT, max_delay_ms=1.0)
        # Prepare the selected training runtime after the native warmup.
        network.train()
        network.initialize(DT, max_delay_ms=1.0)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=False,
            train_delay_backend="dense",
        )
    # endregion cache
    return network


def run_demo(*, execution="scan", backend="aot_eager"):
    network = make_network()
    connection = next(iter(network.synapses))
    # region compile
    options = {
        "dt": DT,
        "initialization": "fresh",
        "execution": execution,
        "inference": execution == "while",
        "trainable": (f"synapses.{connection}.weight.weights.0.param.rho",),
        "callbacks": {"trace": dn.func.Recorder(("cell.v", "t"))},
        "backend": backend,
    }
    if execution == "scan":
        options["checkpoint_every"] = 2
    runner = dn.compile(network, steps=STEPS, **options)
    uninterrupted = dn.compile(network, steps=2 * STEPS, **options)
    cache = runner.initialization_inputs.cache
    assert isinstance(cache, dn.func.NetworkInitializationCache)
    # endregion compile
    # region current-weights
    weight = runner.parameters["synapses"][connection]["weight.weights.0.param.rho"]
    with torch.no_grad():
        weight.add_(0.4)
        uninterrupted.parameters["synapses"][connection][
            "weight.weights.0.param.rho"
        ].add_(0.4)
    # Raw source histories are replayed with weights derived from these sources.
    initialized = runner.functional.initialize(
        runner.parameters,
        runner.constants,
        runner.initial_state,
        runner.initialization_inputs,
        netstim_draw=runner.functional.sample_initialization_draw(
            runner.parameters, runner.initial_state
        ),
    )
    pending = initialized.state["synapses"][connection]["delivery_buffer"]
    assert pending.abs().sum() > 0 and pending.grad_fn is None
    # endregion current-weights
    # region continuation
    prefix, first = runner()
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    reference, whole = uninterrupted()
    trace = torch.cat(
        (first["callbacks"]["trace"]["cell.v"], second["callbacks"]["trace"]["cell.v"])
    )
    torch.testing.assert_close(trace, whole["callbacks"]["trace"]["cell.v"])
    torch.testing.assert_close(
        final["populations"]["cell"]["integrator"]["v"],
        reference["populations"]["cell"]["integrator"]["v"],
    )
    assert trace.shape == (2 * STEPS + 1, 1, 1)
    assert final["clock"]["step"].item() == 2 * STEPS
    # endregion continuation
    # region gradients
    if execution == "scan":
        # New connection traffic remains differentiable after the detached
        # cache boundary. Cached Population materializations stay detached.
        gradient = torch.autograd.grad(trace.square().mean(), weight)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    else:
        assert not trace.requires_grad
        assert runner.execution_report().strategy == "compiled_while_loop"
    # endregion gradients
    assert network.t.item() == 0.0
    return trace.detach()


def make_population_cached_network():
    # region population-cache
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        resting = dn.SingleCompartment(
            N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-40.0
        )
        resting.insert(pas, g=0.001, e=-65.0)
        resting.steady_state(dt=DT, tstop=1.0, with_ve=False, with_intra=False)
        fresh = dn.SingleCompartment(
            N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-45.0
        )
        fresh.insert(pas, g=0.001, e=-65.0)
        network = dn.Network({"resting": resting, "fresh": fresh})
        network.train()
        network.initialize(DT)
    # endregion population-cache
    return network


def run_population_cache_demo():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        return _run_population_cache_demo()


def _run_population_cache_demo():
    network = make_population_cached_network()
    native = make_population_cached_network()
    # region population-cache-runner
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=True,
        callbacks={"trace": dn.func.Recorder(("resting.v", "fresh.v", "t"))},
        backend="aot_eager",
    )
    initial = runner.initialization_inputs
    assert initial.cache is None  # No Network-global cache was prepared.
    assert initial.populations["resting"].cache is not None
    assert initial.populations["fresh"].cache is None
    final, auxiliary = runner()
    # The resting Population restores its own cache; fresh uses v_init.
    trace = auxiliary["callbacks"]["trace"]
    torch.testing.assert_close(
        trace["fresh.v"][0], torch.full_like(network.fresh.v, -45)
    )
    assert not torch.allclose(
        trace["resting.v"][0], torch.full_like(network.resting.v, -40)
    )
    # endregion population-cache-runner
    with dn.ctx(JIT=0), torch.no_grad():
        native.initialize(DT)
        resting, fresh = [native.resting.v.clone()], [native.fresh.v.clone()]
        for _ in range(STEPS):
            native.step()
            resting.append(native.resting.v.clone())
            fresh.append(native.fresh.v.clone())
    torch.testing.assert_close(trace["resting.v"], torch.stack(resting))
    torch.testing.assert_close(trace["fresh.v"], torch.stack(fresh))
    torch.testing.assert_close(
        final["populations"]["resting"]["integrator"]["v"], native.resting.v
    )
    return trace["resting.v"]


def main():
    scan = run_demo(execution="scan")
    inference = run_demo(execution="while")
    torch.testing.assert_close(scan, inference)
    print("continued cached-start voltage shape:", tuple(scan.shape))
    owned = run_population_cache_demo()
    print("Population-owned cached-start voltage shape:", tuple(owned.shape))


if __name__ == "__main__":
    main()
