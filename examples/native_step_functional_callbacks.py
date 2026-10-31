#!/usr/bin/env python3
"""Continue pure observations through manually controlled native steps.

Run from the repository root::

    python examples/native_step_functional_callbacks.py

The bounded CPU examples cover a Population and a Network. They compare manual
steps and native run/checkpointed/longrun continuation with functional values,
Recorder cadence and gradients, then restore portable callback carry.
"""

from __future__ import annotations

import io

import torch

import dendra as dn
from dendra.models.mod import pas

DT = 0.025
MANUAL, RUN, CHECKPOINTED, LONG = 2, 2, 2, 2
STEPS = MANUAL + RUN + CHECKPOINTED + LONG
G = "integrator.mech.mechanisms.leak.g_param"


class VoltageSummary(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "initial_voltage": voltage,
            "energy": voltage.new_zeros(()),
            "steps": voltage.new_zeros((), dtype=torch.int64),
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "initial_voltage": carry["initial_voltage"],
            "energy": carry["energy"] + voltage.square().mean(),
            "steps": carry["steps"] + 1,
        }, None

    def finalize(self, carry, emissions):
        del emissions
        return carry

    def checkpoint_configuration(self):
        return {"kind": "example.step_voltage_summary", "version": 1, "fields": {}}


class NativeLoopHooks(dn.callbacks.Callback):
    """Show that loop_hooks controls only ordinary native hooks."""

    def __init__(self):
        super().__init__()
        self.started = self.finished = 0

    def pre_loop_hook(self, model):
        self.started += 1

    def post_loop_hook(self, model):
        self.finished += 1


def make_model(network):
    populations = {}
    for name, compartments, reversal in (
        (("left", 2, -55.0), ("right", 3, -75.0)) if network else (("cell", 2, -55.0),)
    ):
        population = dn.SingleCompartment(
            N=1,
            C=compartments,
            v_init=-65.0,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        population.insert(pas.rename("leak"), g=0.001, e=reversal)
        populations[name] = population
    model = dn.Network(populations) if network else populations["cell"]
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.initialize()
    return model


def make_callbacks(functional, network):
    summary = VoltageSummary()
    if network:
        summary = dn.callbacks.PopulationCallback("right", summary)
    fields = ("left.v", "right.v", "t") if network else ("v", "t")
    return functional.make_callbacks(
        {
            "trace": dn.Recorder(fields),
            "coarse": dn.Recorder(fields, dt=2 * DT),
            "summary": summary,
        }
    )


def run_demo(network):
    model = make_model(network)
    functional, _ = dn.func.make_functional(model, dt=DT)
    observations = make_callbacks(functional, network)
    native_hooks = NativeLoopHooks()
    results = []
    # region manual
    carry = None
    for index in range(MANUAL):
        options = {
            "functional_callbacks": observations,
            "callback_state": carry,
            "callbacks": [native_hooks],
            "loop_hooks": index == MANUAL - 1,
        }
        if network:
            result = model.step(**options)
        else:
            result = dn.step(model, dt=DT, **options)
        carry = result.state
        results.append(result)
    # Native hook lists coexist through functional_callbacks; callback_state
    # keeps pure callback phase independent of the requested native loop hooks.
    assert native_hooks.started == native_hooks.finished == 1
    # endregion manual
    field = "right.v" if network else "v"
    assert results[0]["trace"][field].shape[0] == 2  # Initial + first post-step.
    assert results[1]["trace"][field].shape[0] == 1

    # region continuation
    clock = {} if network else {"dt": DT}
    result = model.run(
        tstop=RUN * DT,
        **clock,
        callbacks=observations,
        callback_state=carry,
    )
    results.append(result)
    hook_loss, result = model.longrun_checkpointed(
        tstop=CHECKPOINTED * DT,
        chunklength=2,
        **clock,
        callbacks=observations,
        callback_state=result.state,
        safe_checkpoint=True,
    )
    assert hook_loss is None
    results.append(result)
    result = model.longrun(
        tstop=LONG * DT,
        chunklength=2,
        **clock,
        callbacks=observations,
        callback_state=result.state,
    )
    results.append(result)
    # endregion continuation
    fields = ("left.v", "right.v", "t") if network else ("v", "t")
    joined = {
        name: {
            selected: torch.cat(tuple(part[name][selected] for part in results))
            for selected in fields
        }
        for name in ("trace", "coarse")
    }
    assert joined["trace"][field].shape[0] == STEPS + 1
    assert joined["coarse"][field].shape[0] == STEPS // 2 + 1
    for selected in fields:
        torch.testing.assert_close(
            joined["coarse"][selected], joined["trace"][selected][::2]
        )
    assert result["summary"]["steps"].item() == STEPS

    reference_model = make_model(network)
    reference, tensors = dn.func.make_functional(reference_model, dt=DT)
    _, auxiliary = reference.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=STEPS,
        callbacks=make_callbacks(reference, network),
    )
    wanted = auxiliary["callbacks"]
    for name, trace in joined.items():
        torch.testing.assert_close(trace, wanted[name])
    torch.testing.assert_close(result["summary"], wanted["summary"])
    targets = (
        tuple(model.populations[name].mech.leak.g_param for name in ("left", "right"))
        if network
        else (model.mech.leak.g_param,)
    )
    reference_targets = (
        tuple(tensors.parameters["populations"][name][G] for name in ("left", "right"))
        if network
        else (tensors.parameters[G],)
    )
    voltage_fields = fields[:-1]
    objective = result["summary"]["energy"] + sum(
        joined["trace"][name].square().mean() for name in voltage_fields
    )
    reference_objective = wanted["summary"]["energy"] + sum(
        wanted["trace"][name].square().mean() for name in voltage_fields
    )
    actual_gradients = torch.autograd.grad(objective, targets)
    expected_gradients = torch.autograd.grad(reference_objective, reference_targets)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0

    # region portable
    buffer = io.BytesIO()
    torch.save(observations.export_state(result.state), buffer)
    buffer.seek(0)
    restored = observations.restore_state(torch.load(buffer, weights_only=True))
    # The native model already remains at this boundary; carry restores no model state.
    with torch.no_grad():
        continued = model.step(
            **clock,
            callbacks=observations,
            callback_state=restored,
        )
    assert continued["summary"]["steps"].item() == STEPS + 1
    assert continued["trace"][field].shape[0] == 1
    # endregion portable
    print(
        f"{'Network' if network else 'Population'}: manual pure observations, "
        "cadence, gradients and callback continuation passed"
    )


def main():
    torch.set_num_threads(1)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
        for network in (False, True):
            run_demo(network)


if __name__ == "__main__":
    main()
