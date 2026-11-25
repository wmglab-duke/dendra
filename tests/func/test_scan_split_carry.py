"""Oversized scan carry leaves keep independent owned storage and exact gradients."""

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._carry_packing import _PackedCarryLayout
from dendra.func._scan_tree import _CompiledTensorTreeScan

pytestmark = pytest.mark.cpu


def test_packed_carry_isolates_large_leaves_by_size():
    storage = torch.arange(48, dtype=torch.float64).reshape(6, 8)
    wide = storage.T
    values = (
        torch.ones(2, dtype=wide.dtype),
        wide,
        torch.zeros(3, dtype=wide.dtype),
        wide.view_as(wide),
        torch.tensor(True),
        None,
    )
    ordinary = _PackedCarryLayout.from_tree(values)
    layout = _PackedCarryLayout.from_tree(
        values, large_leaf_bytes=wide.numel() * wide.element_size()
    )
    small, first, second_small, second, _, absent = layout.leaves

    assert len(ordinary.groups) == 2
    assert len(layout.groups) == 4
    assert small.group == second_small.group
    assert len({small.group, first.group, second.group}) == 3
    assert first.offset == second.offset == 0
    assert absent is None

    packs = layout.pack(values)
    assert packs[first.group].numel() == wide.numel() + 1
    assert packs[second.group].numel() == wide.numel() + 1
    restored = layout.unpack(packs)
    torch.testing.assert_close(restored, values, rtol=0, atol=0)
    assert not torch._C._is_alias_of(restored[1], wide)
    assert not torch._C._is_alias_of(restored[1], restored[3])


@pytest.mark.slow
def test_large_inductor_training_carry_keeps_gradients_and_owned_continuations():
    def body(parameters, prepared, state, carries, sample):
        del prepared, sample
        gain = parameters["gain"]
        left = state["left"] + gain * state["right"]
        right = state["right"] - 0.125 * left
        updated = {
            "left": left,
            "right": right,
            "count": state["count"] + 1,
            "optional": None,
        }
        return (updated, carries), {"probe": left[0, 0] + right[0, 1]}

    def initial(storage):
        left = storage[:, ::2]
        return {
            "left": left,
            "right": left.view_as(left),
            "count": torch.tensor(0),
            "optional": None,
        }

    def objective(state, frames):
        return (
            state["left"].square().mean()
            + state["right"].square().mean()
            + torch.cat(frames).sum()
        )

    steps = 2
    storage = torch.linspace(-0.3, 0.5, 256 * 1028, dtype=torch.float64)
    storage = storage.reshape(256, 1028).requires_grad_()
    gain = torch.tensor(0.25, dtype=storage.dtype, requires_grad=True)
    reference_storage = storage.detach().clone().requires_grad_()
    reference_gain = gain.detach().clone().requires_grad_()
    state = initial(storage)
    reference = initial(reference_storage)
    assert not state["left"].is_contiguous()
    assert torch._C._is_alias_of(state["left"], state["right"])
    original_version = storage._version

    scan = _CompiledTensorTreeScan(
        body,
        device="cpu",
        steps=steps,
        compile_options={
            "backend": "inductor",
            "fullgraph": True,
            "options": {"compile_threads": 1, "cpp.threads": 1},
        },
    )
    frames = []
    with torch_compiler_warning_context():
        for _ in range(2):
            before = state
            state, _, emissions = scan({"gain": gain}, {}, state, {}, None)
            frames.append(emissions["probe"])
            assert not torch._C._is_alias_of(state["left"], state["right"])
            for source in (storage, before["left"], before["right"]):
                assert not torch._C._is_alias_of(state["left"], source)
                assert not torch._C._is_alias_of(state["right"], source)
        loss = objective(state, frames)
        gradients = torch.autograd.grad(loss, (storage, gain), retain_graph=True)
        repeated = torch.autograd.grad(loss, (storage, gain))

    expected_frames = []
    for _ in range(2 * steps):
        (reference, _), emitted = body(
            {"gain": reference_gain}, {}, reference, {}, None
        )
        expected_frames.append(emitted["probe"])
    expected_loss = objective(reference, [torch.stack(expected_frames)])
    expected_gradients = torch.autograd.grad(
        expected_loss, (reference_storage, reference_gain)
    )
    for name in ("left", "right", "count"):
        torch.testing.assert_close(state[name], reference[name], rtol=0, atol=0)
    assert state["optional"] is None
    torch.testing.assert_close(torch.cat(frames), torch.stack(expected_frames))
    torch.testing.assert_close(gradients, expected_gradients, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(repeated, gradients, rtol=0, atol=0)
    torch.testing.assert_close(storage, reference_storage, rtol=0, atol=0)
    assert storage._version == original_version

    _, layout, _, _ = next(iter(scan._kernels.values()))
    assert sum(len(indices) == 1 for indices in layout.indices) == 3


@pytest.mark.parametrize(
    "device,options,grad_enabled,expected_groups",
    [
        pytest.param("cpu", {"backend": "inductor"}, True, 2, id="cpu-training"),
        pytest.param("cpu", {}, True, 2, id="default-inductor"),
        pytest.param("cpu", {"backend": "inductor"}, False, 1, id="cpu-no-grad"),
        pytest.param("cpu", {"backend": "eager"}, True, 1, id="eager"),
        pytest.param("cpu", {"backend": "aot_eager"}, True, 1, id="aot-eager"),
        pytest.param(
            "cpu",
            {"backend": lambda graph, inputs: graph.forward},
            True,
            1,
            id="custom",
        ),
        pytest.param(
            "cpu", {"backend": "inductor", "disable": True}, True, 1, id="disabled"
        ),
        pytest.param(
            "cpu",
            {"backend": "inductor", "options": {"use_aoti": True}},
            True,
            1,
            id="aot-inductor",
        ),
        pytest.param("cuda", {"backend": "inductor"}, True, 1, id="cuda"),
        pytest.param("mps", {"backend": "inductor"}, True, 1, id="mps"),
    ],
)
def test_large_carry_policy_is_scoped_to_cpu_inductor_training(
    monkeypatch, device, options, grad_enabled, expected_groups
):
    from torch._dynamo.backends import registry

    monkeypatch.setattr(registry, "get_default_backend", lambda: "inductor")
    monkeypatch.setattr(
        "dendra.func._scan_tree.install_scan_compatibility", lambda: None
    )
    scan = _CompiledTensorTreeScan(
        lambda *_: None, steps=1, compile_options=options, device=device
    )
    # Only the declared execution device is inspected by policy.
    state = {
        "large": torch.empty((256, 512), dtype=torch.float64),
        "small": torch.zeros((), dtype=torch.float64),
    }
    layout = scan._carry_layout(state, {}, grad_enabled=grad_enabled)
    assert len(layout.groups) == expected_groups
