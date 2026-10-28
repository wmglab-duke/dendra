"""Empty canonical packets retain deterministic compiled loop semantics."""

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_network import _close, _mixed_network

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("training", (False, True))
def test_empty_population_packet_runs_scan_without_a_tensor_input_tape(training):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        model = dn.SingleCompartment(
            N=1,
            C=1,
            v_init=-65.0,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(pas, g=0.001, e=-70.0)
        model.train()
        model.initialize()
    functional, tensors = dn.func.make_functional(model, dt=0.025)
    name = tensors.parameter_name("g_param")
    parameters = tensors.independent_parameters(trainable=(name,))
    packet = functional.sample_runtime_draws(tensors.state, steps=4)
    assert not any(torch.is_tensor(value) for value in pytree.tree_leaves(packet))
    kernel = functional.compile_rollout_chunk(4, execution="scan", backend="aot_eager")
    with torch_compiler_warning_context(), torch.set_grad_enabled(training):
        prepared = functional.prepare(parameters, tensors.constants)
        actual, _ = kernel(parameters, prepared, tensors.state, draws=packet)
        expected, _ = functional.prepare_and_rollout(
            parameters, tensors.constants, tensors.state, steps=4
        )
        _close(actual, expected)
        if training:
            target = parameters[name]
            value = actual["integrator"]["v"].square().sum()
            reference = expected["integrator"]["v"].square().sum()
            (gradient,) = torch.autograd.grad(value, target)
            (expected_gradient,) = torch.autograd.grad(reference, target)
            torch.testing.assert_close(gradient, expected_gradient)
            assert gradient.abs().sum() > 0
    assert kernel.execution_report().strategy == "compiled_scan"


def test_empty_network_packet_runs_scan_without_a_tensor_input_tape():
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    packet = functional.sample_runtime_draws(tensors.state, steps=3)
    assert not any(torch.is_tensor(value) for value in pytree.tree_leaves(packet))
    kernel = functional.compile_rollout_chunk(3, execution="scan", backend="aot_eager")
    with torch_compiler_warning_context(), torch.no_grad():
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        actual, _ = kernel(tensors.parameters, prepared, tensors.state, draws=packet)
        expected, _ = functional.prepare_and_rollout(
            tensors.parameters, tensors.constants, tensors.state, steps=3
        )
    _close(actual, expected)


def test_deterministic_network_automatic_run_does_not_build_runtime_packets(
    monkeypatch,
):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    bound = functional.bind(tensors.parameters, tensors.constants)
    with torch_compiler_warning_context(), torch.no_grad():
        expected, _ = functional.prepare_and_rollout(
            tensors.parameters, tensors.constants, tensors.state, steps=3
        )

        def forbidden_sampler(*_args, **_kwargs):
            pytest.fail("deterministic execution must not sample a runtime packet")

        monkeypatch.setattr(functional, "sample_runtime_draws", forbidden_sampler)
        actual, _ = dn.func.run(bound, tensors.state, steps=3, host_span_steps=2)
    _close(actual, expected)
