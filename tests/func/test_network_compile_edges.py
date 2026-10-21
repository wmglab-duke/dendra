"""Public Network runner boundaries retain report and snapshot guarantees."""

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import DT, _close, _loss, _mixed_network
from tests.func.test_network_execution_reporting import _RejectFinalize

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


@pytest.mark.parametrize("inference", (True, False))
def test_network_constructor_under_inference_mode_owns_versioned_restart_operands(
    inference,
):
    model = _mixed_network()
    name = f"synapses.{next(iter(model.synapses))}.weights"
    with torch.inference_mode():
        runner = dn.compile(
            model,
            dt=DT,
            steps=5,
            inference=inference,
            trainable=() if inference else (name,),
            backend="aot_eager",
        )
    for value in pytree.tree_leaves(
        (runner._parameters, runner._constants, runner.initial_state)
    ):
        if torch.is_tensor(value):
            assert not value.is_inference()
            assert value._version >= 0
    outputs, gradients = [], []
    for _ in range(2):
        state, _ = runner()
        outputs.append(state)
        voltage = state["populations"]["cell"]["integrator"]["v"]
        if inference:
            assert not voltage.requires_grad
        else:
            gradient = torch.autograd.grad(_loss(state), runner.get_parameter(name))[0]
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            gradients.append(gradient)
    _close(outputs[0], outputs[1])
    if gradients:
        torch.testing.assert_close(*gradients)


def test_network_runner_failed_finalization_preserves_last_successful_report():
    runner = dn.compile(
        _mixed_network(),
        dt=DT,
        steps=2,
        inference=False,
        disable=True,
        callbacks={"reject": _RejectFinalize()},
    )
    # A successful direct dispatch establishes the shared kernel report.
    prepared = runner.functional.prepare(runner.parameters, runner.constants)
    runner.kernel(runner.parameters, prepared, runner.initial_state)
    previous = runner.execution_report()
    assert previous is not None and not previous.callbacks
    with pytest.raises(ValueError, match="Reject callback finalization"):
        runner()
    assert runner.execution_report() is previous
