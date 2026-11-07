"""Native MPS callback boundaries retain owned carry, phase and gradients."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation_checkpoints import _disk, _tensors
from tests.func.test_continuation_detach import _History
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage

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
            dn.func.PopulationCallback("left", _History()) if network else _History()
        ),
    }


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_mps_callback_capture_preserves_phase_owned_training_and_checkpoint_replay(
    network, runner, tmp_path
):
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with (
        torch_compiler_warning_context(),
        dn.ctx(DEVICE="mps", DTYPE=torch.float32, JIT=0, REQUIRE_GRAD=1, IMEM=False),
    ):
        native = _network() if network else _population()
        program = dn.compile(
            native,
            dt=DT,
            steps=4,
            inference=False,
            trainable=(),
            callbacks=_specifications(network),
            backend="eager",
            disable=True,
        )
        subject = program if runner else program.functional
        options = {} if runner else {"callbacks": program.callbacks}
        native_options = {} if network else {"dt": DT}
        first = native.run(steps=3, callbacks=program.callbacks, **native_options)
        assert first.state["history"]["energy"].grad_fn is not None

        with torch.inference_mode():
            packet = dn.func.capture_continuation(
                subject, callback_state=first.state, **options
            )
        carry = packet.callback_state
        assert packet._lifecycle == "active"
        assert carry is not first.state
        assert carry.plan_token is first.state.plan_token
        assert carry.emission_schemas is first.state.emission_schemas
        _close(carry.carries, first.state.carries)
        originals = tuple(_tensors(first.state.carries))
        storages = {
            (value.device, value.untyped_storage().data_ptr())
            for value in originals
            if value.numel()
        }
        for value, original in zip(_tensors(carry.carries), originals, strict=True):
            assert type(value) is torch.Tensor and value.device == original.device
            assert value.is_leaf and value.grad_fn is None and not value.is_inference()
            assert isinstance(value._version, int)
            assert value.requires_grad == original.requires_grad
            if value.numel():
                assert (
                    value.device,
                    value.untyped_storage().data_ptr(),
                ) not in storages

        source = native.left.mech.leak.g_param if network else native.mech.leak.g_param
        selected = _target(packet.parameters, network)
        assert type(selected) is torch.Tensor and selected.device.type == "mps"
        assert selected.dtype == torch.float32 and selected.requires_grad
        assert selected.is_leaf and selected.grad_fn is None
        assert not selected.is_inference()
        assert selected.data_ptr() != source.data_ptr()
        if network:
            assert selected is packet.parameters["populations"]["right"][G]

        saved = _disk(
            dn.func.export_continuation(subject, packet, **options),
            tmp_path / "mps-captured-callbacks.pt",
        )
        restored = dn.func.restore_continuation(subject, saved, **options)
        assert restored._lifecycle == "active"
        final, last = dn.func.run(subject, packet, steps=4, **options)
        assert first["trace"]["t"].shape == (2,)
        assert last["callbacks"]["trace"]["t"].shape == (2,)
        times = last["callbacks"]["trace"]["t"]
        _close(times, times.new_tensor([4 * DT, 6 * DT]))
        assert last["callbacks"].state["trace"].item() == 7
        gradient, upstream = torch.autograd.grad(
            _voltage(final, network).square().sum()
            + last["callbacks"]["history"]["energy"],
            (selected, source),
            allow_unused=True,
        )
        assert gradient.device.type == "mps" and gradient.isfinite().all()
        assert gradient.abs().sum() > 0 and upstream is None

        with torch.no_grad():
            replay, replay_auxiliary = dn.func.run(
                subject, restored, steps=4, **options
            )
            native_last = native.run(
                steps=4,
                callbacks=program.callbacks,
                callback_state=first.state,
                **native_options,
            )
            reference = _network() if network else _population()
            callbacks = dn.func.make_callbacks(
                reference, _specifications(network), dt=DT
            )
            whole = reference.run(steps=7, callbacks=callbacks, **native_options)

        _close(replay, final)
        _close(replay_auxiliary["callbacks"].outputs, last["callbacks"].outputs)
        _close(native_last.outputs, last["callbacks"].outputs)
        _close(final, program.functional.extract(native).state)
        for field, expected in whole["trace"].items():
            _close(
                torch.cat((first["trace"][field], last["callbacks"]["trace"][field])),
                expected,
            )
        _close(last["callbacks"]["history"], whole["history"])
        _close(
            _voltage(final, network),
            torch.cat((reference.left.v.flatten(), reference.right.v.flatten()))
            if network
            else reference.v,
        )
