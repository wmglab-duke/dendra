"""CUDA dynamic inference operands retain devices and versioned bindings."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.func._inference import _canonicalize_inference_tree
from dendra.models.mod import pas

DT = 0.025
STEPS = 4
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cuda", DTYPE=torch.float32):
        model = dn.SingleCompartment(
            N=2,
            C=3,
            device="cuda",
            dtype=torch.float32,
            v_init=-62.0,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(pas, g=0.0002, e=-71.0)
        model.requires_grad_(False)
        model.eval()
        model.initialize()
    return model


def _versions(tree):
    return tuple(
        (id(value), value._version)
        for value in pytree.tree_leaves(tree)
        if isinstance(value, torch.Tensor)
    )


def _assert_dynamic_devices(actual, original):
    actual_values, actual_spec = pytree.tree_flatten(actual)
    original_values, original_spec = pytree.tree_flatten(original)
    assert actual_spec == original_spec
    for value, expected in zip(actual_values, original_values, strict=True):
        assert isinstance(value, torch.Tensor)
        assert value.shape == expected.shape
        assert value.dtype == expected.dtype
        assert value.device == expected.device
        assert torch.is_inference(value)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize("record", [False, True])
def test_cuda_restart_normalization_preserves_devices_and_versioned_bindings(record):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    with torch.no_grad():
        prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(
        STEPS, execution="unrolled", backend="eager"
    )
    carries = {}
    selected = chunk
    if record:
        callbacks = functional.make_callbacks(
            {
                "trace": dn.func.Recorder(["v", "t"]),
                "spikes": dn.func.APCount(node_check=[0, 2], dt=DT),
            }
        )
        carries = callbacks._initialize(tensors.state)[0].carries
        selected = chunk._runner_callback_chunk(callbacks, STEPS)
    versioned = (tensors.parameters, tensors.constants, prepared.values)
    versions_before = _versions(versioned)
    values_before = pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), versioned
    )
    captured = []

    def capture(*operands):
        captured.append(operands)
        return selected._lowered(*operands)

    # Exercise the production outer dispatch boundary on CUDA, without making
    # this representation/ownership regression depend on expensive codegen.
    selected._compiled = capture
    inputs = dn.func.RolloutInput()

    def invoke():
        if record:
            return selected._rollout_values(
                tensors.parameters, prepared.values, tensors.state, inputs, carries
            )
        return selected(tensors.parameters, prepared, tensors.state, inputs)

    with torch.inference_mode():
        invoke()
        first = captured[0]
        _assert_dynamic_devices(first[2], tensors.state)
        if record:
            _assert_dynamic_devices(first[5], carries)
            assert any(
                value.device.type == "cuda" for value in pytree.tree_leaves(first[5])
            )
        alias_state, alias_carry = _canonicalize_inference_tree(
            (tensors.state, {"voltage": tensors.state["integrator"]["v"]})
        )
        assert alias_state["integrator"]["v"] is alias_carry["voltage"]
        tensors.state["integrator"]["v"].add_(0.25)
        invoke()
        second = captured[1]
        _assert_dynamic_devices(second[2], tensors.state)
        torch.testing.assert_close(
            second[2]["integrator"]["v"],
            first[2]["integrator"]["v"] + 0.25,
            rtol=0,
            atol=0,
        )
    for operands in captured:
        assert operands[0] is tensors.parameters
        assert operands[1] is prepared.values
    assert _versions(versioned) == versions_before
    for value, expected in zip(
        pytree.tree_leaves(versioned), pytree.tree_leaves(values_before), strict=True
    ):
        if not isinstance(value, torch.Tensor):
            assert value is expected
            continue
        assert not torch.is_inference(value)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert all(
        not torch.is_inference(value) for value in pytree.tree_leaves(tensors.state)
    )
