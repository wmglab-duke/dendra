"""Prove direct scan carry ownership at an opaque functional operator boundary."""

from dataclasses import replace

import pytest
import torch

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._carry_packing import _PackedCarryLayout
from dendra.func._scan_tree import _CompiledTensorTreeScan

pytestmark = pytest.mark.cpu


@pytest.mark.slow
@pytest.mark.parametrize("case", ("direct", "output_alias", "storage_view"))
def test_large_training_scan_opaque_carry_ownership(case):
    transitions = []

    @torch.library.custom_op(f"dendra_test::advance_carry_{case}", mutates_args=())
    def advance(
        carry: torch.Tensor, gain: torch.Tensor, drive: torch.Tensor
    ) -> torch.Tensor:
        result = carry * gain + drive
        # Keep these allocations live until the assertion so allocator reuse
        # cannot imitate direct carry. The detached handles own no AD graph.
        transitions.append((carry.detach(), result.detach()))
        return result

    @advance.register_fake
    def fake(carry, gain, drive):
        return torch.empty_like(carry, memory_format=torch.contiguous_format)

    def setup_context(ctx, inputs, output):
        carry, gain, drive = inputs
        ctx.save_for_backward(carry, gain)
        ctx.drive_shape = drive.shape

    def backward(ctx, gradient):
        carry, gain = ctx.saved_tensors
        return (
            gradient * gain,
            (gradient * carry).sum_to_size(gain.shape),
            gradient.sum_to_size(ctx.drive_shape),
        )

    advance.register_autograd(backward, setup_context=setup_context)

    def body(parameters, prepared, state, carries, sample):
        prior = state["queue"]
        if case == "storage_view":
            sample = sample + prior.as_strided((1,), (1,), 0)[0]
        queue = advance(prior, parameters["gain"], sample)
        updated = {
            "queue": queue,
            "count": state["count"] + 1,
            "optional": None,
        }
        if case == "output_alias":
            updated["sibling"] = queue.view_as(queue)
        return (updated, carries), {"probe": queue[0, 0]}

    def loss(final, frames):
        value = final["queue"].square().mean() + frames.square().sum()
        if case == "output_alias":
            value = value + final["sibling"].mean()
        return value

    steps = 3
    engine = _CompiledTensorTreeScan(
        body,
        steps=steps,
        compile_options={
            "backend": "inductor",
            "fullgraph": True,
            "options": {"compile_threads": 1, "cpp.threads": 1, "cpp_wrapper": False},
        },
        device="cpu",
    )
    backing = (
        torch.linspace(-0.4, 0.3, 36 * 8192, dtype=torch.float64)
        .reshape(36, 8192)
        .requires_grad_()
    )
    gain = torch.tensor(0.7, dtype=backing.dtype, requires_grad=True)
    drives = torch.tensor([0.4, -0.5, 0.3], dtype=backing.dtype, requires_grad=True)
    original = backing.detach().clone()
    version = backing._version
    state = {
        "queue": backing[:, 1::2],
        "count": torch.tensor(0),
        "optional": None,
    }

    if case == "output_alias":
        state["sibling"] = state["queue"]

    with torch_compiler_warning_context():
        # The first invocation captures authored examples. Only count calls
        # from the already compiled recurrence.
        engine({"gain": gain}, {}, state, {}, drives)
        transitions.clear()
        final, _, emitted = engine({"gain": gain}, {}, state, {}, drives)
        output = final["queue"]
        observed = emitted["probe"]
        actual = torch.autograd.grad(loss(final, observed), (backing, gain, drives))

    eager_backing = original.requires_grad_()
    eager_gain, eager_drives = (
        value.detach().clone().requires_grad_() for value in (gain, drives)
    )
    queue, frames = eager_backing[:, 1::2], []
    for drive in eager_drives:
        if case == "storage_view":
            drive = drive + queue[0, 0]
        queue = queue * eager_gain + drive
        frames.append(queue[0, 0])
    eager_final = {"queue": queue}
    if case == "output_alias":
        eager_final["sibling"] = queue
    expected = torch.autograd.grad(
        loss(eager_final, torch.stack(frames)),
        (eager_backing, eager_gain, eager_drives),
    )

    torch.testing.assert_close(output, queue, rtol=0, atol=0)
    torch.testing.assert_close(observed, torch.stack(frames), rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(backing, original, rtol=0, atol=0)
    assert backing._version == version
    assert int(final["count"]) == steps
    assert final["optional"] is None
    assert not torch._C._is_alias_of(output, backing)
    assert not torch._C._is_alias_of(output, observed)
    if case == "output_alias":
        torch.testing.assert_close(final["sibling"], queue, rtol=0, atol=0)
        assert not torch._C._is_alias_of(output, final["sibling"])

    assert len(transitions) == steps
    assert not torch._C._is_alias_of(transitions[0][0], state["queue"])
    direct = case == "direct"
    for (_, previous_output), (next_input, _) in zip(
        transitions[:-1], transitions[1:], strict=True
    ):
        assert torch._C._is_alias_of(previous_output, next_input) == direct
        assert (previous_output.data_ptr() == next_input.data_ptr()) == direct


def test_direct_scan_ingress_keeps_history_in_inference_context():
    source = torch.ones(3, 4, requires_grad=True)
    layout = _PackedCarryLayout.from_tree(source)
    packed = layout.pack(source)
    direct = replace(layout, direct_groups=frozenset({0}))

    with torch.inference_mode(), torch.enable_grad():
        (carry,) = direct.scan_inputs(packed)
    (gradient,) = torch.autograd.grad(carry.sum(), source)

    torch.testing.assert_close(carry, source)
    torch.testing.assert_close(gradient, torch.ones_like(source))
    assert not torch._C._is_alias_of(carry, source)
