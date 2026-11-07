"""Compiled loop view isolation preserves explicit stochastic tensor programs."""

from __future__ import annotations

import io

import pytest
import torch
from torch.utils import _pytree as pytree

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._scan_tree import _CompiledTensorTreeScan
from dendra.func._tensor_tree_while import _CompiledTensorTreeWhile

pytestmark = pytest.mark.cpu
STEPS = 5  # A while block and a tail both consume the same lifted operands.
DTYPE = torch.float64


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            yield
    finally:
        torch.set_num_threads(previous)


def _packet(steps, rng):
    generator = torch.Generator().set_state(rng)
    draws, states = [], []
    for _ in range(steps):
        draws.append(torch.randn((), generator=generator, dtype=DTYPE))
        states.append(generator.get_state().clone())
    # Zero-stride spatial axes are useful for noise shared across compartments.
    draws = torch.stack(draws).reshape(steps, 1, 1).expand(steps, 2, 2)
    raw_drive = torch.linspace(-0.1, 0.3, steps + 2, dtype=DTYPE)
    drive = raw_drive[1 : steps + 1].reshape(steps, 1, 1).expand(steps, 2, 2)
    raw_gate = torch.arange(steps + 4).remainder(3).ne(0)
    gate = raw_gate[3 : steps + 3].reshape(steps, 1, 1).expand(steps, 2, 2)
    assert gate.storage_offset() == 3
    return {"draw": draws, "drive": drive, "gate": gate, "rng": torch.stack(states)}


def _body(parameters, prepared, state, carries, sample):
    weight = parameters["weight"]
    # Authored tensor programs may use the original view's exact metadata.
    selected = weight.as_strided(weight.shape, weight.stride(), weight.storage_offset())
    value = (
        0.7 * state["value"]
        + selected * prepared["same"]
        + sample["draw"].detach() * prepared["coefficients"]
        + sample["drive"]
    )
    value = torch.where(sample["gate"] ^ state["enabled"], value, -value)
    updated = {
        "value": value,
        "rng": sample["rng"],
        "step": state["step"] + 1,
        "padding": state["padding"],
        "enabled": ~state["enabled"],
    }
    return (updated, {"sum": carries["sum"] + value.square().sum()}), {
        "value": value,
        "step": updated["step"],
        "enabled": updated["enabled"],
    }


def _case(layout, *, train=False):
    source = torch.linspace(0.1, 0.8, 8, dtype=DTYPE, requires_grad=train)
    offset = 0 if layout == "expanded" else 3
    weight = source[offset : offset + 1].expand(2, 2)
    parameters = {"weight": weight}
    prepared = {
        "same": weight,
        "coefficients": torch.tensor([[0.03, 0.07], [0.11, 0.17]], dtype=DTYPE),
    }
    state = {
        "value": torch.linspace(-0.2, 0.1, 4, dtype=DTYPE).reshape(2, 2),
        "rng": torch.Generator().manual_seed(174).get_state(),
        "step": torch.tensor(0),
        # Packed boolean recurrence puts the differentiable branch selector
        # at offset three, rather than an accidentally aligned byte address.
        "padding": torch.tensor([True, False, True]),
        "enabled": torch.tensor(True),
    }
    return source, parameters, prepared, state, {"sum": torch.zeros((), dtype=DTYPE)}


def _slice(packet, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda value: value[start:stop], packet)


def _eager(parameters, prepared, state, carries, tape):
    frames = []
    for index in range(tape["draw"].shape[0]):
        sample = pytree.tree_map_only(torch.Tensor, lambda value: value[index], tape)
        (state, carries), emitted = _body(parameters, prepared, state, carries, sample)
        frames.append(emitted)
    return state, carries, pytree.tree_map(lambda *values: torch.stack(values), *frames)


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


@pytest.mark.parametrize(
    "engine_type", (_CompiledTensorTreeScan, _CompiledTensorTreeWhile)
)
@pytest.mark.parametrize("layout", ("expanded", "offset_expanded"))
def test_cpp_loop_expanded_views_recording_and_rng_checkpoint_continuation(
    engine_type, layout
):
    source, parameters, prepared, state, carries = _case(layout)
    tape = _packet(2 * STEPS, state["rng"])
    operands = parameters, prepared, state, carries, tape
    before = _clone(operands)
    versions = [value._version for value in pytree.tree_leaves(operands)]
    global_rng = torch.get_rng_state().clone()
    engine = engine_type(
        _body,
        steps=STEPS,
        compile_options={
            "backend": "inductor",
            "fullgraph": True,
            "options": {"cpp_wrapper": True},
        },
        device="cpu",
    )
    with torch.no_grad():
        first = engine(parameters, prepared, state, carries, _slice(tape, 0, STEPS))
        replay = engine(parameters, prepared, state, carries, _slice(tape, 0, STEPS))
        torch.testing.assert_close(first, replay)
        checkpoint = io.BytesIO()
        torch.save(first[:2], checkpoint)
        checkpoint.seek(0)
        restored, restored_carries = torch.load(checkpoint, weights_only=True)
        suffix = _slice(tape, STEPS, 2 * STEPS)
        regenerated = _packet(STEPS, restored["rng"])
        torch.testing.assert_close(regenerated["draw"], suffix["draw"])
        torch.testing.assert_close(regenerated["rng"], suffix["rng"], rtol=0, atol=0)
        suffix = {**suffix, "draw": regenerated["draw"], "rng": regenerated["rng"]}
        final = engine(parameters, prepared, restored, restored_carries, suffix)
        expected = _eager(parameters, prepared, state, carries, tape)
    torch.testing.assert_close(final[:2], expected[:2], rtol=2e-12, atol=2e-13)
    trace = pytree.tree_map(
        lambda left, right: torch.cat((left, right)), first[2], final[2]
    )
    torch.testing.assert_close(trace, expected[2], rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(operands, before, rtol=0, atol=0)
    assert [value._version for value in pytree.tree_leaves(operands)] == versions
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    assert parameters["weight"] is prepared["same"]
    assert parameters["weight"].stride() == (0, 0)
    assert torch._C._is_alias_of(parameters["weight"], source)


def test_scan_expanded_views_preserve_logical_raw_and_continuation_gradients():
    source, parameters, prepared, state, carries = _case("offset_expanded", train=True)
    tape = _packet(2 * STEPS, state["rng"])
    # Sampling policy detaches stochastic draws while ordinary drive tapes train.
    tape["draw"].requires_grad_()
    tape["drive"].requires_grad_()
    weight, drive, draws = parameters["weight"], tape["drive"], tape["draw"]
    engine = _CompiledTensorTreeScan(
        _body,
        steps=STEPS,
        compile_options={"backend": "inductor", "fullgraph": True},
        device="cpu",
    )
    first = engine(parameters, prepared, state, carries, _slice(tape, 0, STEPS))
    final = engine(
        parameters, prepared, first[0], first[1], _slice(tape, STEPS, 2 * STEPS)
    )
    expected = _eager(parameters, prepared, state, carries, tape)
    trace = torch.cat((first[2]["value"], final[2]["value"]))
    weights = torch.linspace(0.2, 1.3, trace.numel(), dtype=DTYPE).reshape_as(trace)
    loss = final[1]["sum"] + (trace * weights).sum() + final[0]["value"].square().sum()
    reference = expected[1]["sum"] + (expected[2]["value"] * weights).sum()
    reference = reference + expected[0]["value"].square().sum()
    torch.testing.assert_close(final[:2], expected[:2], rtol=2e-12, atol=2e-13)
    actual = torch.autograd.grad(
        loss, (weight, source, drive, draws), allow_unused=True
    )
    oracle = torch.autograd.grad(
        reference, (weight, source, drive, draws), allow_unused=True
    )
    assert oracle[-1] is None
    if actual[-1] is not None:
        torch.testing.assert_close(actual[-1], torch.zeros_like(draws), rtol=0, atol=0)
    for gradient, expected_gradient in zip(actual[:-1], oracle[:-1], strict=True):
        torch.testing.assert_close(gradient, expected_gradient, rtol=2e-11, atol=2e-12)
    assert weight.stride() == (0, 0) and weight.storage_offset() == 3
    assert tape["gate"].storage_offset() == 3


@pytest.mark.parametrize(
    "engine_type", (_CompiledTensorTreeScan, _CompiledTensorTreeWhile)
)
@pytest.mark.parametrize("shape", ((1, 1), (0, 2)), ids=("singleton", "empty"))
def test_cpp_loop_preserves_degenerate_expanded_metadata(engine_type, shape):
    def body(parameters, prepared, state, carries, sample):
        del prepared, sample
        value = parameters["value"]
        selected = value.as_strided(value.shape, value.stride(), value.storage_offset())
        metadata = sum(value.shape) + sum(value.stride()) + value.storage_offset()
        updated = {"value": state["value"] + selected.sum() + metadata}
        return (updated, carries), updated

    value = torch.tensor(0.7, dtype=DTYPE).expand(shape)
    assert value.is_contiguous() and value.stride() == (0, 0)
    initial = {"value": torch.zeros((), dtype=DTYPE)}
    engine = engine_type(
        body,
        steps=STEPS,
        compile_options={
            "backend": "inductor",
            "fullgraph": True,
            "options": {"cpp_wrapper": True},
        },
        device="cpu",
    )
    with torch.no_grad():
        actual = engine({"value": value}, {}, initial, {}, None)
        repeated = engine({"value": value}, {}, initial, {}, None)
        expected, frames = initial, []
        for _ in range(STEPS):
            (expected, _), frame = body({"value": value}, {}, expected, {}, None)
            frames.append(frame["value"])
    torch.testing.assert_close(actual, repeated)
    torch.testing.assert_close(actual[0], expected)
    torch.testing.assert_close(actual[2]["value"], torch.stack(frames))
    assert value.stride() == (0, 0) and value.storage_offset() == 0


@pytest.mark.parametrize(
    "engine_type", (_CompiledTensorTreeScan, _CompiledTensorTreeWhile)
)
def test_cpp_loop_ignores_unused_views_aliasing_used_dense_operands(engine_type):
    def body(parameters, prepared, state, carries, sample):
        del prepared, sample
        updated = {"value": state["value"] + parameters["used"].sum()}
        return (updated, carries), updated

    source = torch.linspace(0.1, 0.8, 8, dtype=DTYPE)
    expanded = source[:1].expand(2, 2)
    parameters = {
        "used": source,
        "unused": expanded,
        "unused_alias": expanded,
        "unused_slice": source[1::2],
    }
    prepared = {"unused": source[1:2].expand(3)}
    tape = {
        "unused": source[2:3].expand(STEPS, 2),
        "unused_offset": source[3:4].expand(STEPS, 3),
    }
    before = _clone((parameters, prepared, tape))
    initial = {"value": torch.zeros((), dtype=DTYPE)}
    engine = engine_type(
        body,
        steps=STEPS,
        compile_options={
            "backend": "inductor",
            "fullgraph": True,
            "options": {"cpp_wrapper": True},
        },
        device="cpu",
    )
    with torch.no_grad():
        final, _, recorded = engine(parameters, prepared, initial, {}, tape)
    frames = torch.arange(1, STEPS + 1, dtype=DTYPE) * source.sum()
    torch.testing.assert_close(final["value"], frames[-1])
    torch.testing.assert_close(recorded["value"], frames)
    torch.testing.assert_close((parameters, prepared, tape), before, rtol=0, atol=0)
    assert parameters["unused"] is parameters["unused_alias"]
    assert torch._C._is_alias_of(parameters["used"], expanded)


@pytest.mark.parametrize("grad_enabled", (False, True))
def test_cached_population_scan_accepts_reordered_parameter_keys(grad_enabled):
    import dendra as dn
    from tests.func.test_runners import DT, _model

    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    name = next(key for key in tensors.parameters if key.endswith("gnabar_param"))
    raw = tensors.parameters[name]
    with torch.set_grad_enabled(grad_enabled):
        parameters = dict(tensors.parameters)
        # Exercise a copied view as well as ordinary dense parameters: its
        # usage index must still select the same key after mapping reordering.
        parameters[name] = torch.stack((torch.zeros_like(raw), raw))[1]
        assert parameters[name].storage_offset() == 1
        prepared = functional.prepare(parameters, tensors.constants)
        chunk = functional.compile_rollout_chunk(
            2, execution="scan", backend="aot_eager"
        )
        first = chunk(parameters, prepared, tensors.state)
        reordered = dict(reversed(tuple(parameters.items())))
        second = chunk(reordered, prepared, tensors.state)
        torch.testing.assert_close(first, second)
        if grad_enabled:
            expected_gradient = torch.autograd.grad(
                first[0]["integrator"]["v"].square().mean(), raw, retain_graph=True
            )[0]
            actual_gradient = torch.autograd.grad(
                second[0]["integrator"]["v"].square().mean(), raw
            )[0]
            torch.testing.assert_close(actual_gradient, expected_gradient)
            assert actual_gradient.abs().sum() > 0
    kernels = (
        chunk._compiled._training._kernels if grad_enabled else chunk._compiled._kernels
    )
    assert len(kernels) == 1
    assert reordered[name] is parameters[name]
    assert parameters[name].storage_offset() == 1
