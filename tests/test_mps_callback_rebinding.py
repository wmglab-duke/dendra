"""Prepared MPS callbacks follow independent native handoff receivers."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _snapshot
from tests.func.test_continuation_detach import _History
from tests.func.test_lowering_api import DT, _network, _population, _target
from tests.func.test_native_handoff import _capture, _native_parameter, _unchanged

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS is unavailable"
    ),
]


def _specifications(network):
    return {
        "trace": dn.Recorder(
            ("left.v", "right.v", "t") if network else ("v", "t"), dt=2 * DT
        ),
        "history": (
            dn.callbacks.PopulationCallback("left", _History())
            if network
            else _History()
        ),
    }


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


def _loss(result, network):
    return (
        result["history"]["energy"]
        + result["history"]["latest"].square().mean()
        + result["trace"]["left.v" if network else "v"].square().mean()
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", ("run", "longrun_checkpointed"))
def test_mps_prepared_callbacks_rebind_with_cadence_receiver_gradients_and_replay(
    network, method
):
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with (
        torch_compiler_warning_context(),
        dn.ctx(DEVICE="mps", DTYPE=torch.float32, JIT=0, REQUIRE_GRAD=1, IMEM=False),
    ):
        factory = _network if network else _population
        source, receiver, reference = factory(), factory(), factory()
        source_parameter = _native_parameter(source, network)
        receiver_parameter = _native_parameter(receiver, network)
        assert receiver_parameter is not source_parameter
        program = dn.compile(
            source,
            dt=DT,
            steps=3,
            inference=False,
            trainable="*g_param",
            callbacks=_specifications(network),
            backend="eager",
            disable=True,
        )
        callbacks = program.callbacks
        plans = callbacks._plans
        initial = dn.func.Continuation(
            program.initial_state, program.parameters, program.constants
        )
        selected = _target(initial.parameters, network)
        with torch.no_grad():
            selected.mul_(1.4)
        source_before = _capture(source)
        mps_rng_before = torch.mps.get_rng_state().clone()

        _, prefix = dn.func.run(program, initial, steps=3)
        packet = prefix["continuation"]
        carry = callbacks.detach_state(packet.callback_state)
        assert carry.plan_token is packet.callback_state.plan_token
        assert carry["history"]["energy"].grad_fn is None
        assert dn.func.commit_continuation_(program, receiver, packet) is receiver
        assert _native_parameter(receiver, network) is receiver_parameter
        _close(program.functional.extract(receiver).state, packet.state)

        expected_state, expected = dn.func.run(
            program,
            dn.func.Continuation(
                _snapshot(packet.state), packet.parameters, packet.constants, carry
            ),
            steps=4,
        )
        options = {"steps": 4, "callbacks": callbacks, "callback_state": carry}
        if not network:
            options["dt"] = DT
        if method == "longrun_checkpointed":
            options.update(
                chunklength=2,
                safe_checkpoint=True,
                restore_state_after_backward=True,
            )
            native_loss, actual = receiver.longrun_checkpointed(**options)
            assert native_loss is None
        else:
            actual = receiver.run(**options)
        assert callbacks._plans is plans and callbacks._functional is program.functional
        assert actual.state.plan_token is carry.plan_token
        assert actual["history"]["energy"].grad_fn is not None
        assert actual["history"]["latest"].device.type == "mps"
        assert actual["history"]["latest"].dtype == torch.float32
        _close(actual.outputs, expected["callbacks"].outputs)
        _close(program.functional.extract(receiver).state, expected_state)
        assert prefix["callbacks"]["trace"]["t"].shape == (2,)
        assert actual["trace"]["t"].shape == (2,)
        _close(actual["trace"]["t"], actual["trace"]["t"].new_tensor([4 * DT, 6 * DT]))
        assert actual.state["trace"].item() == 7

        receiver_before_backward = _snapshot(receiver.state_dict_for_checkpoint())
        gradient, source_gradient, prefix_gradient = torch.autograd.grad(
            _loss(actual, network),
            (receiver_parameter, source_parameter, selected),
            allow_unused=True,
        )
        wanted = torch.autograd.grad(_loss(expected["callbacks"], network), selected)[0]
        _close(gradient, wanted)
        assert gradient.device.type == "mps" and gradient.isfinite().all()
        assert gradient.abs().sum() > 0
        assert source_gradient is None and prefix_gradient is None
        _assert_packet_equal(
            receiver.state_dict_for_checkpoint(), receiver_before_backward
        )

        with torch.no_grad():
            _, whole = dn.func.run(program, initial, steps=7)
            assert (
                dn.func.commit_continuation_(program, reference, initial) is reference
            )
            native_options = {} if network else {"dt": DT}
            native_whole = reference.run(steps=7, callbacks=callbacks, **native_options)
        _close(native_whole.outputs, whole["callbacks"].outputs)
        for name, trace in whole["callbacks"]["trace"].items():
            joined = torch.cat(
                (prefix["callbacks"]["trace"][name], actual["trace"][name])
            )
            _close(joined, trace)
        _close(actual["history"], whole["callbacks"]["history"])
        _unchanged(source, source_before)
        torch.testing.assert_close(
            torch.mps.get_rng_state(), mps_rng_before, rtol=0, atol=0
        )
