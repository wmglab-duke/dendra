"""Shared scan boundaries preserve optional carry and input leaves."""

import pytest
import torch
from torch.utils import _pytree as pytree

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._carry_packing import _PackedCarryLayout
from dendra.func._scan_tree import _CompiledTensorTreeScan

pytestmark = pytest.mark.cpu


def test_packed_carry_preserves_static_none_without_tensor_slots():
    state = {
        "v": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "optional": None,
        "nested": {"rng": torch.arange(7, dtype=torch.uint8), "absent": None},
    }
    layout = _PackedCarryLayout.from_tree(state)
    assert len(layout.groups) == 2
    packs = layout.pack(state)
    actual = layout.unpack(packs)
    assert actual["optional"] is None and actual["nested"]["absent"] is None
    torch.testing.assert_close(actual["v"], state["v"])
    torch.testing.assert_close(actual["nested"]["rng"], state["nested"]["rng"])
    assert not torch._C._is_alias_of(actual["v"], state["v"])
    with pytest.raises(ValueError, match="static None"):
        layout.pack({**state, "optional": torch.ones(())})
    empty = _PackedCarryLayout.from_tree({"absent": None})
    assert empty.unpack(empty.pack({"absent": None})) == {"absent": None}


@pytest.mark.parametrize("inputs", ("empty", "none", "driven"))
def test_shared_scan_preserves_optional_leaves_and_gradients(inputs):
    def body(parameters, prepared, state, carries, sample):
        assert parameters["unused"] is None and prepared["unused"] is None
        assert state["optional"] is None and carries["unused"] is None
        drive = None if sample is None else sample.get("drive")
        if drive is None:
            drive = torch.zeros_like(state["v"])
        value = state["v"].tanh() + parameters["scale"] * drive + prepared["dt"]
        updated = {
            "v": value,
            "optional": None,
            "rng": state["rng"],
            "count": state["count"] + 1,
            "flag": ~state["flag"],
        }
        carries = {"loss": carries["loss"] + value.square().sum(), "unused": None}
        return (updated, carries), {"v": value}

    parameters = {
        "scale": torch.tensor(0.7, dtype=torch.float64, requires_grad=True),
        "unused": None,
    }
    prepared = {"dt": torch.tensor(0.125, dtype=torch.float64), "unused": None}
    state = {
        "v": torch.tensor([-0.3, 0.2], dtype=torch.float64, requires_grad=True),
        "optional": None,
        "rng": torch.arange(7, dtype=torch.uint8),
        "count": torch.tensor(0),
        "flag": torch.tensor(True),
    }
    carries = {"loss": torch.zeros((), dtype=torch.float64), "unused": None}
    tape = None if inputs == "empty" else {"drive": None}
    if inputs == "driven":
        tape = {
            "drive": torch.tensor(
                [[0.1, 0.2], [-0.2, 0.3], [0.4, -0.1]],
                dtype=torch.float64,
                requires_grad=True,
            ),
            "unused": None,
        }
    operands = parameters, prepared, state, carries, tape
    expected = pytree.tree_map_only(
        torch.Tensor,
        lambda value: value.detach().clone().requires_grad_(value.requires_grad),
        operands,
    )
    engine = _CompiledTensorTreeScan(
        body,
        steps=3,
        compile_options={"backend": "aot_eager", "fullgraph": True},
        device="cpu",
    )
    with torch_compiler_warning_context():
        final, updated, emitted = engine(*operands)
        reference, reference_carry = expected[2:4]
        frames = []
        for index in range(3):
            sample = pytree.tree_map_only(
                torch.Tensor, lambda value: value[index], expected[4]
            )
            (reference, reference_carry), frame = body(
                expected[0], expected[1], reference, reference_carry, sample
            )
            frames.append(frame["v"])
        torch.testing.assert_close(final["v"], reference["v"])
        torch.testing.assert_close(updated["loss"], reference_carry["loss"])
        torch.testing.assert_close(emitted["v"], torch.stack(frames))
        assert final["optional"] is None and updated["unused"] is None
        torch.testing.assert_close(final["rng"], reference["rng"])
        torch.testing.assert_close(final["count"], reference["count"])
        torch.testing.assert_close(final["flag"], reference["flag"])
        actual_targets = [state["v"]]
        expected_targets = [expected[2]["v"]]
        if inputs == "driven":
            actual_targets += [parameters["scale"], tape["drive"]]
            expected_targets += [expected[0]["scale"], expected[4]["drive"]]
        gradients = torch.autograd.grad(
            updated["loss"] + emitted["v"].square().sum(), actual_targets
        )
        oracle = torch.autograd.grad(
            reference_carry["loss"] + torch.stack(frames).square().sum(),
            expected_targets,
        )
        for actual, wanted in zip(gradients, oracle, strict=True):
            torch.testing.assert_close(actual, wanted)
