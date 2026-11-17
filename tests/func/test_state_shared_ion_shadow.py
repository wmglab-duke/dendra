"""Private functional owners preserve the setup-time State ion-write seed."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.helpers import DEVICE
from dendra.models.mechanisms import Mechanism, State

pytestmark = pytest.mark.cpu
DT = 0.025
STEPS = 4
WRITER = "writer"


class _DeclaresSodium(State):
    State.STATE("nai")
    State.DERIVATIVE("nai' = 0.125 + 0.0 * nai")


class _SodiumWriter(Mechanism):
    Mechanism.STATE_BUNDLE(_DeclaresSodium)
    Mechanism.USEION("na", write=["nai"])


class _ObservesSodium(State):
    State.STATE("nai")
    State.CARRY("seen")
    State.DERIVATIVE("nai' = 0.0 * nai")

    def advance(self, v, dt, values):
        return {"nai": values["nai"] + dt, "seen": vars(self)["nai"] + 0.0 * v}


class _ObservingSodiumWriter(Mechanism):
    Mechanism.STATE_BUNDLE(_ObservesSodium)
    Mechanism.USEION("na", write=["nai"])


class _ReturnsSodium(State):
    State.STATE("nai")
    State.DERIVATIVE("nai' = 0.0 * nai")

    def advance(self, v, dt, values):
        return {"nai": self.nai}


class _ReturningSodiumWriter(Mechanism):
    Mechanism.STATE_BUNDLE(_ReturnsSodium)
    Mechanism.USEION("na", write=["nai"])


@pytest.fixture(autouse=True)
def _pure_cpu():
    # An existing CPU default needs no additional Torch DeviceContext. The
    # sharing path declines stacks of multiple modes.
    defaults_are_cpu = torch.get_default_device().type == "cpu" and DEVICE.value in (
        "",
        "cpu",
        torch.device("cpu"),
    )
    device = {} if defaults_are_cpu else {"DEVICE": "cpu"}
    with (
        dn.ctx(DTYPE="float64", JIT=0, IMEM=False, CHECK_PURITY=True, **device),
        torch.no_grad(),
    ):
        yield


def _model(mechanism):
    model = dn.SingleCompartment(
        N=2, C=2, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    ).eval()
    model.insert(mechanism.rename(WRITER))
    model.initialize()
    return model


def _runner(model):
    return dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        chunk_steps=STEPS,
        execution="scan",
        inference=True,
        backend="eager",
        initialization="fresh",
    )


def _sodium(state):
    return state["mechanisms"][WRITER]["nai"]


def _rollout(runner):
    plan = runner.functional
    prepared = plan.prepare(runner.parameters, runner.constants)
    output, _ = plan.rollout(
        runner.parameters, prepared, runner.initial_state, steps=STEPS
    )
    return output


def test_canonical_ion_write_shadow_preserves_native_initialization_and_scan():
    model = _model(_SodiumWriter)
    runner = _runner(model)
    reference = _model(_SodiumWriter)
    initial = dn.func.initialize(runner).state
    torch.testing.assert_close(_sodium(initial), reference.mech.mechanisms[WRITER].nai)

    actual, _ = dn.func.run(runner)
    reference.longrun(dt=DT, steps=STEPS, chunklength=STEPS, progressbar=False)
    torch.testing.assert_close(_sodium(actual), reference.mech.mechanisms[WRITER].nai)
    torch.testing.assert_close(actual["integrator"]["v"], reference.v)
    assert runner.execution_report().strategy == "compiled_scan"


def test_authored_ion_write_shadow_uses_setup_value_after_parent_rebinding():
    model = _model(_ObservingSodiumWriter)
    writer = model.mech.mechanisms[WRITER]
    setup_seed = vars(writer.DE["_ObservesSodium"])["nai"].clone()
    writer.nai = writer.nai + 1.0
    runner = _runner(model)

    state = _rollout(runner)
    observed = state["state_buffers"][WRITER]["_ObservesSodium"]["seen"]
    torch.testing.assert_close(observed, setup_seed, rtol=0, atol=0)
    torch.testing.assert_close(_sodium(state), setup_seed + 1.0 + STEPS * DT)


def test_authored_ion_write_shadow_cannot_escape_through_returned_state():
    model = _model(_ReturningSodiumWriter)
    runner = _runner(model)
    plan = runner.functional
    preparation_writer = plan._preparation.population.mech.mechanisms[WRITER]
    prep_state = preparation_writer.DE["_ReturnsSodium"]
    setup_seed = vars(prep_state)["nai"].clone()
    owners = (plan._transition.population, plan._initialization.population)
    transition_seed, initialization_seed = (
        vars(owner.mech.mechanisms[WRITER].DE["_ReturnsSodium"])["nai"]
        for owner in owners
    )
    assert torch._C._is_alias_of(transition_seed, initialization_seed)
    assert not torch._C._is_alias_of(transition_seed, vars(prep_state)["nai"])
    fresh_state = dn.func.initialize(runner).state

    returned = _sodium(_rollout(runner))
    torch.testing.assert_close(returned, setup_seed, rtol=0, atol=0)
    returned.add_(1.5)

    torch.testing.assert_close(vars(prep_state)["nai"], setup_seed, rtol=0, atol=0)
    torch.testing.assert_close(_sodium(_rollout(runner)), setup_seed, rtol=0, atol=0)
    torch.testing.assert_close(
        _sodium(dn.func.initialize(runner).state), _sodium(fresh_state), rtol=0, atol=0
    )
