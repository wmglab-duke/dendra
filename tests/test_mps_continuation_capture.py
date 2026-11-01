"""Current native MPS boundaries retain independent training and replay."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import _close
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS is unavailable"
    ),
]


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_mps_capture_replays_current_native_state_and_trains_owned_sources(
    network, runner
):
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with (
        torch_compiler_warning_context(),
        dn.ctx(DEVICE="mps", DTYPE=torch.float32, JIT=0, REQUIRE_GRAD=1, IMEM=False),
    ):
        native = _network() if network else _population()
        program = dn.compile(
            native, dt=DT, steps=3, inference=False, backend="eager", disable=True
        )
        subject = program if runner else program.functional
        options = {} if network else {"dt": DT}
        native.run(steps=2, **options)
        source = native.left.mech.leak.g_param if network else native.mech.leak.g_param
        with torch.inference_mode():
            packet = dn.func.capture_continuation(subject)
        selected = _target(packet.parameters, network)
        assert selected.device.type == "mps"
        assert selected.requires_grad == source.requires_grad
        assert selected.is_leaf and not selected.is_inference()
        assert selected.data_ptr() != source.data_ptr()
        if network:
            assert selected is packet.parameters["populations"]["right"][G]
        saved = dn.func.export_continuation(subject, packet)
        restored = dn.func.restore_continuation(subject, saved)
        first, _ = dn.func.run(subject, packet, steps=3)
        replay, _ = dn.func.run(subject, restored, steps=3)
        _close(first, replay)
        gradient, upstream = torch.autograd.grad(
            _voltage(first, network).square().sum(),
            (selected, source),
            allow_unused=True,
        )
        assert gradient.device.type == "mps" and gradient.isfinite().all()
        assert gradient.abs().sum() > 0 and upstream is None
        with torch.no_grad():
            native.run(steps=3, **options)
        torch.testing.assert_close(
            _voltage(first, network),
            _voltage(program.functional.extract(native).state, network),
            rtol=3e-5,
            atol=3e-5,
        )
