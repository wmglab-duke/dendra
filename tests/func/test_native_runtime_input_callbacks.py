"""Native checkpoint operands retain deterministic drives and pure observations."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network_rollout import _driven_network
from tests.test_native_runtime_inputs import DT
from tests.test_native_runtime_inputs import _environment as _environment
from tests.test_native_runtime_inputs import _model, _tapes

pytestmark = pytest.mark.cpu
STEPS = 7


def _packet(drives, network):
    if not network:
        return drives
    return {
        "left": drives,
        "right": dn.func.Inputs(ve=drives.ve, intra=-0.5 * drives.intra),
    }


def _slice(inputs, start, stop):
    if isinstance(inputs, dict):
        return {name: _slice(value, start, stop) for name, value in inputs.items()}
    return dn.func.Inputs(inputs.ve[start:stop], inputs.intra[start:stop])


def _objective(voltages, trace):
    return sum(value.square().mean() for value in voltages) + sum(
        value.square().mean() for name, value in trace.items() if name != "t"
    )


def _check_checkpointed_inputs_and_callbacks(model, source, *, network, dt):
    drives = _tapes(model.left if network else model, STEPS)
    reference_drives = dn.func.Inputs(
        drives.ve.detach().clone().requires_grad_(),
        drives.intra.detach().clone().requires_grad_(),
    )
    inputs = _packet(drives, network)
    original_inputs = _snapshot(inputs)
    functional, tensors = (
        dn.func.make_functional_network(source)
        if network
        else dn.func.make_functional(source, dt=dt)
    )
    fields = ("left.v", "right.v", "t") if network else ("v", "t")
    configurations = {"trace": dn.Recorder(fields, dt=2 * dt)}
    callbacks = dn.func.make_callbacks(model, configurations, dt=dt)
    reference_callbacks = dn.func.make_callbacks(functional, configurations)
    expected, auxiliary = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        inputs=_packet(reference_drives, network),
        steps=STEPS,
        checkpoint_every=3,
        callbacks=reference_callbacks,
    )
    expected_results = auxiliary["callbacks"]
    expected_states = expected["populations"].values() if network else (expected,)
    wanted = torch.autograd.grad(
        _objective(
            (value["integrator"]["v"] for value in expected_states),
            expected_results["trace"],
        ),
        reference_drives,
    )

    segments = []
    callback_state = None
    for start, stop in ((0, 3), (3, STEPS)):
        options = dict(
            tstop=(stop - start) * dt,
            chunklength=3,
            inputs=_slice(inputs, start, stop),
            callbacks=callbacks,
            callback_state=callback_state,
            safe_checkpoint=True,
            restore_state_after_backward=True,
        )
        if not network:
            options["dt"] = dt
        native_loss, result = model.longrun_checkpointed(**options)
        assert native_loss is None
        assert type(result) is dn.func.FunctionalCallbackResults
        assert result["trace"]["t"].shape == (2,)
        assert result.state["trace"].item() == stop
        segments.append(result)
        callback_state = result.state

    trace = {
        name: torch.cat(tuple(result["trace"][name] for result in segments))
        for name in fields
    }
    _close(trace, expected_results["trace"])
    _close(callback_state.carries, expected_results.state.carries)
    torch.testing.assert_close(
        trace["t"],
        torch.arange(0, STEPS, 2, dtype=model.dtype(), device=model.device()) * dt,
    )
    final = functional.extract(model).state
    _close(final, expected)
    saved_model = _snapshot(model.state_dict_for_checkpoint())
    populations = model.populations.values() if network else (model,)
    actual = torch.autograd.grad(
        _objective((population.v for population in populations), trace), drives
    )
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved_model)
    _assert_packet_equal(inputs, original_inputs)
    if network:
        assert inputs["left"].ve is inputs["right"].ve is drives.ve
        assert inputs["left"].intra is drives.intra
        assert inputs["right"].intra.grad_fn is not None


def test_population_checkpointed_inputs_preserve_callback_phase_and_all_drive_vjps():
    model, _ = _model()
    source, _ = _model()
    _check_checkpointed_inputs_and_callbacks(model, source, network=False, dt=DT)


def test_network_checkpointed_inputs_preserve_callback_phase_and_shared_drive_vjps():
    _check_checkpointed_inputs_and_callbacks(
        _driven_network(), _driven_network(), network=True, dt=NETWORK_DT
    )
