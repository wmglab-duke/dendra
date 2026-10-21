"""Network callbacks reuse Population reducers with qualified observation paths."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.func._network_callbacks import make_network_callbacks
from tests.func.test_network import DTYPE, _mixed_network

pytestmark = pytest.mark.cpu


def _case():
    return dn.func.make_functional_network(_mixed_network())


def _recording(callbacks, state, *, carries=None):
    if carries is None:
        carries, emission = callbacks._initialize(state)
        part = callbacks._stack([emission])
    else:
        carries = callbacks._validate_state(carries)
        part = callbacks._stack([])
    outputs = callbacks._finalize(carries, part)
    return carries, outputs


def test_qualified_and_explicit_network_recorder_paths_share_existing_collection():
    functional, tensors = _case()
    fields = (
        "cell.v",
        "cell.mech.relay.level",
        "cell.relay.level",
        "populations.cell.mechanism_buffers.relay.level",
        "populations.cell.integrator.v",
        "t",
        "cell.t",
    )
    recorder = dn.func.Recorder(fields)
    callbacks = make_network_callbacks(functional, {"trace": recorder})
    assert type(callbacks) is dn.func.FunctionalCallbacks
    callbacks._validate_owner(functional)
    carry, output = _recording(callbacks, tensors.state)
    recorded = output["trace"]
    state = tensors.state["populations"]["cell"]
    torch.testing.assert_close(recorded["cell.v"][0], state["integrator"]["v"])
    for field in fields[1:4]:
        torch.testing.assert_close(
            recorded[field][0], state["mechanism_buffers"]["relay"]["level"]
        )
    torch.testing.assert_close(recorded[fields[4]], recorded[fields[0]])
    torch.testing.assert_close(recorded["t"][0], tensors.state["clock"]["t"])
    torch.testing.assert_close(recorded["cell.t"][0], state["clock"]["t"])
    assert recorder.states == fields
    assert carry["trace"] == ()


def test_network_recorder_qualifies_nested_state_carry_checkpoint_names():
    from tests.func.test_continuous_receive import DT, _model, _Remember

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
        model = dn.Network({"cell": _model(_Remember, batch=None)})
        model.train()
        model.initialize(DT)
    functional, tensors = dn.func.make_functional_network(model)
    fields = (
        "cell.mech.analog.DE._Memory.mark",
        "cell.analog.DE._Memory.mark",
        "populations.cell.state_buffers.analog._Memory.mark",
    )
    callbacks = make_network_callbacks(functional, {"trace": dn.func.Recorder(fields)})
    _, output = _recording(callbacks, tensors.state)
    expected = tensors.state["populations"]["cell"]["state_buffers"]["analog"][
        "_Memory"
    ]["mark"]
    for name in fields:
        torch.testing.assert_close(output["trace"][name][0], expected)


@dataclass(frozen=True)
class _VoltageSum(dn.func.FunctionalCallback):
    def _bind(self, functional):
        assert isinstance(functional, dn.func.FunctionalNetwork)
        return self

    def initialize(self, state, auxiliary):
        del auxiliary
        return state["clock"]["t"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return carry + state["populations"]["cell"]["integrator"]["v"].sum(), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


@pytest.mark.parametrize("compiled", (False, True))
def test_network_callback_updates_keep_recording_and_reducer_gradients(compiled):
    functional, tensors = _case()
    callbacks = make_network_callbacks(
        functional,
        {
            "trace": dn.func.Recorder(("cell.v", "cell.mech.relay.level", "t")),
            "sum": _VoltageSum(),
        },
    )
    state = tensors.state
    initial = (
        state["populations"]["cell"]["integrator"]["v"]
        .detach()
        .clone()
        .requires_grad_()
    )
    state["populations"]["cell"]["integrator"]["v"] = initial
    carry, emission = callbacks._initialize(state)
    parts = [callbacks._stack([emission])]

    def step(parameters, constants, current, callback_state):
        current, auxiliary = functional.prepare_and_step(parameters, constants, current)
        callback_state, emitted = callbacks._update(callback_state, current, auxiliary)
        return current, callback_state, emitted

    execute = (
        torch.compile(step, backend="aot_eager", fullgraph=True) if compiled else step
    )
    expected = initial.sum() * 0
    for _ in range(3):
        state, carry, emitted = execute(
            tensors.parameters, tensors.constants, state, carry
        )
        parts.append(callbacks._stack([emitted]))
        expected = expected + state["populations"]["cell"]["integrator"]["v"].sum()
    stacked = callbacks._concatenate(parts)
    output = callbacks._finalize(carry, stacked)
    assert output["trace"]["cell.v"].shape[0] == 4
    torch.testing.assert_close(output["sum"], expected)
    torch.testing.assert_close(
        output["trace"]["cell.v"][-1], state["populations"]["cell"]["integrator"]["v"]
    )
    observed_grad = torch.autograd.grad(output["sum"], initial, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, initial)[0]
    torch.testing.assert_close(observed_grad, expected_grad)
    assert observed_grad.abs().sum() > 0


def test_network_recorder_continuation_and_empty_segment_do_not_duplicate_initial():
    functional, tensors = _case()
    callbacks = make_network_callbacks(
        functional, {"trace": dn.func.Recorder(("cell.v", "t"))}
    )
    carry, initial = callbacks._initialize(tensors.state)
    first = callbacks._stack([initial])
    empty = callbacks._stack([])
    outputs = callbacks._finalize(callbacks._validate_state(carry), empty)
    assert first["trace"]["cell.v"].shape[0] == 1
    assert outputs["trace"]["cell.v"].shape[0] == 0
    assert outputs["trace"]["t"].shape == (0,)
    other, _ = _case()
    with pytest.raises(dn.func.FunctionalizationError, match="different"):
        callbacks._validate_owner(other)
    other_callbacks = make_network_callbacks(
        functional, {"trace": dn.func.Recorder("cell.v")}
    )
    with pytest.raises(dn.func.FunctionalizationError, match="different callback plan"):
        other_callbacks._validate_state(carry)


@pytest.mark.parametrize(
    "callback",
    (
        dn.func.Recorder("v"),
        dn.func.Recorder("cell.missing"),
        dn.func.Recorder("t", node_indices=[0]),
        dn.func.Recorder("cell.v", dt=0.1),
        dn.func.Recorder("cell.v", sliding_window=3),
        dn.func.Recorder("cell.v", region=object()),
        dn.func.LFP([torch.ones(2, 1, dtype=DTYPE)]),
        dn.func.Raster(),
        dn.func.AnomalyDetector(),
    ),
)
def test_unsupported_network_recording_requests_fail_clearly(callback):
    functional, _ = _case()
    with pytest.raises(dn.func.FunctionalizationError):
        make_network_callbacks(functional, {"trace": callback})


def test_ambiguous_convenience_alias_does_not_block_explicit_state_path():
    tensor = torch.zeros(2, 1, dtype=DTYPE)
    schema = SimpleNamespace(
        shape=tuple(tensor.shape), dtype=tensor.dtype, device=tensor.device
    )
    state = {
        "populations": {
            "cell": {
                "integrator": {"v": tensor},
                "mechanism_buffers": {"integrator": {"v": tensor + 1}},
            }
        },
        "clock": {"t": torch.zeros((), dtype=DTYPE)},
    }
    layout = (
        SimpleNamespace(public_path=("integrator", "v"), checkpoint_key=None),
        SimpleNamespace(
            public_path=("mechanism_buffers", "integrator", "v"),
            checkpoint_key="integrator.v",
        ),
    )
    functional = SimpleNamespace(
        device=tensor.device,
        populations={"cell": SimpleNamespace(_state_layout=layout)},
        _state_schema={
            "populations": {
                "cell": {
                    "integrator": {"v": schema},
                    "mechanism_buffers": {"integrator": {"v": schema}},
                }
            },
            "clock": {
                "t": SimpleNamespace(shape=(), dtype=tensor.dtype, device=tensor.device)
            },
        },
        extract=lambda: SimpleNamespace(state=state),
    )
    with pytest.raises(dn.func.FunctionalizationError, match="ambiguous"):
        make_network_callbacks(
            functional, {"trace": dn.func.Recorder("cell.integrator.v")}
        )
    callbacks = make_network_callbacks(
        functional, {"trace": dn.func.Recorder("populations.cell.integrator.v")}
    )
    _, output = _recording(callbacks, state)
    torch.testing.assert_close(
        output["trace"]["populations.cell.integrator.v"][0], tensor
    )
