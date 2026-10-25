"""Network pre transforms precede insertion-time State initialization."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import pas
from tests.func.test_network import DT, DTYPE, _clone, _close

pytestmark = pytest.mark.cpu

RAW_G = "integrator.mech.mechanisms.pas.g_param"
STATE_KEY = "mechanisms.probe.value"
STATE_PATH = f"state.{STATE_KEY}"
SEEDS = {"left": 0.4, "right": 0.8}
OFFSETS = {"left": 1.0, "right": 2.0}


class _PrecedenceState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")


class _PrecedenceProbe(Mechanism):
    Mechanism.STATE_BUNDLE(_PrecedenceState)


class _ShiftState(torch.nn.Module):
    def forward(self, value, offset):
        return (value + offset,)


class _ObserveState(torch.nn.Module):
    def forward(self, voltage, value):
        return (voltage + value,)


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


def _network(*, cached=None):
    shared = torch.nn.Parameter(torch.tensor(0.001, dtype=DTYPE))
    cells = {}
    for name, seed in SEEDS.items():
        cell = dn.SingleCompartment(
            N=1, C=2, integrator=dn.bwd_euler_sc(imem=False), v_init=-60.0
        )
        cell.insert(pas, g=shared, e=-65.0)
        cell.insert(_PrecedenceProbe.rename("probe"), ic={"value": seed})
        cells[name] = cell
    network = dn.Network(cells)
    network.train()
    network.initialize(DT)
    for name, cell in cells.items():
        cell.register_pre_initialize_transform(
            "shift_state",
            _ShiftState(),
            reads=(STATE_PATH,),
            writes=(STATE_PATH,),
            inputs={"offset": torch.tensor(OFFSETS[name], dtype=DTYPE)},
        )
        cell.register_pre_initialize_transform(
            "observe_state",
            _ObserveState(),
            reads=("state.integrator.v", STATE_PATH),
            writes=("state.integrator.v",),
        )
    if cached is not None:
        cells[cached].cache("_steady_state")
    return network


def _values(state, name):
    return state["populations"][name]["mechanisms"]["probe"]["value"]


def _voltage_loss(state):
    return sum(
        population["integrator"]["v"].square().sum()
        for population in state["populations"].values()
    )


@pytest.mark.parametrize("cached", (None, "left"))
def test_network_fresh_state_precedence_matches_native_and_preserves_shared_gradients(
    cached,
):
    source, native = _network(cached=cached), _network(cached=cached)
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    frozen = _clone((tensors, defaults), detached=True)
    initial = functional.initialize(*tensors, defaults)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)

    for name, seed in SEEDS.items():
        torch.testing.assert_close(
            _values(initial.state, name),
            torch.full_like(_values(initial.state, name), seed),
        )
        expected_voltage = -60.0 if name == cached else -60.0 + seed + OFFSETS[name]
        torch.testing.assert_close(
            initial.state["populations"][name]["integrator"]["v"],
            torch.full_like(native.populations[name].v, expected_voltage),
        )
    _close(initial.state, functional.extract(native).state)
    assert (
        initial.parameters["populations"]["left"][RAW_G]
        is initial.parameters["populations"]["right"][RAW_G]
    )
    state, _ = functional.prepare_and_rollout(*initial, steps=3)
    for _ in range(3):
        native.step()
    _close(state, functional.extract(native).state)
    actual = torch.autograd.grad(
        _voltage_loss(state), tensors.parameters["populations"]["left"][RAW_G]
    )[0]
    expected = torch.autograd.grad(
        sum(cell.v.square().sum() for cell in native.populations.values()),
        native.left.get_parameter(RAW_G),
    )[0]
    torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)
    assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    _close((tensors, defaults), frozen)


def test_network_explicit_ic_and_pre_transform_inputs_have_distinct_gradients():
    source, native = _network(), _network()
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    populations = dict(defaults.populations)
    explicit, offsets = {}, {}
    for name in SEEDS:
        incoming = populations[name]
        explicit[name] = incoming.states[STATE_KEY].detach().clone().requires_grad_()
        offsets[name] = incoming.transforms["pre.shift_state.offset"].detach().clone()
        offsets[name].requires_grad_()
        populations[name] = incoming._replace(
            states={STATE_KEY: explicit[name]},
            transforms={"pre.shift_state.offset": offsets[name]},
        )
    initial = functional.initialize(
        *tensors, defaults._replace(populations=populations)
    )
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initial.state, functional.extract(native).state)
    for name in SEEDS:
        loss = (
            _values(initial.state, name)
            + initial.state["populations"][name]["integrator"]["v"]
        ).sum()
        ic_gradient, transform_gradient = torch.autograd.grad(
            loss, (explicit[name], offsets[name])
        )
        torch.testing.assert_close(ic_gradient, torch.ones_like(ic_gradient))
        torch.testing.assert_close(
            transform_gradient,
            transform_gradient.new_tensor(explicit[name].numel()),
        )


def test_compiled_fresh_network_state_precedence_restarts_and_continues_checkpoint():
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=2,
        inference=False,
        initialization="fresh",
        backend="aot_eager",
        checkpoint_every=1,
        trainable=(f"populations.left.{RAW_G}",),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
    )
    frozen = _clone((runner.initial_state, runner.initialization_inputs), detached=True)
    first, prefix = runner()
    repeated, _ = runner()
    _close(first, repeated)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    for name in SEEDS:
        torch.testing.assert_close(
            prefix["callbacks"]["trace"][f"{name}.v"][0], native.populations[name].v
        )
    for _ in range(2):
        native.step()
    _close(first, runner.functional.extract(native).state)
    final, _ = runner(
        first,
        parameters=prefix["parameters"],
        constants=prefix["constants"],
        callback_state=prefix["callbacks"].state,
    )
    for _ in range(2):
        native.step()
    _close(final, runner.functional.extract(native).state)
    _close((runner.initial_state, runner.initialization_inputs), frozen)
