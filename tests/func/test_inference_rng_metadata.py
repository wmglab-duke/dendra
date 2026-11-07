"""Owned sampling tapes have stable metadata without changing authored draws."""

from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._inputs import _RuntimeInputs
from dendra.func._runtime_sampling import RuntimeSamplingProgram, _stack_runtime_packets
from tests.func.test_network import _close
from tests.func.test_runtime_noise import DT, _population

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"),
    ):
        yield


def _frames():
    packets = []
    for index in range(2):
        value = torch.arange(6.0).reshape(2, 3).T + index
        rng = torch.arange(8, dtype=torch.uint8) + index
        packets.append(
            dn.func.RuntimeDraws(
                {"noise": {"value": value, "alias": value}, "stimulation": None},
                {"bytes": rng, "alias": rng, "seed": torch.tensor(bool(index))},
            )
        )
    return packets


def _legacy_stack(packets):
    return pytree.tree_map(
        lambda *values: None if values[0] is None else torch.stack(values),
        *packets,
    )


def test_final_owned_stack_is_ordinary_fresh_and_preserves_values_and_sources():
    frames = _frames()
    with torch.inference_mode():
        frames[1] = pytree.tree_map_only(
            torch.Tensor, lambda value: value.clone(), frames[1]
        )
    source = pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), frames)
    versioned = [
        value
        for value in pytree.tree_leaves(frames)
        if torch.is_tensor(value) and not torch.is_inference(value)
    ]
    versions = [value._version for value in versioned]
    with torch.inference_mode():
        first = _stack_runtime_packets(frames)
        second = _stack_runtime_packets(frames)
        expected = _legacy_stack(frames)
    _close(first, expected)
    _close(second, expected)
    leaves = [value for value in pytree.tree_leaves(first) if torch.is_tensor(value)]
    others = [
        value
        for value in pytree.tree_leaves((frames, second))
        if torch.is_tensor(value)
    ]
    for index, value in enumerate(leaves):
        assert not torch.is_inference(value) and not value.requires_grad
        assert value.is_contiguous() and value.storage_offset() == 0
        assert value._version >= 0
        assert all(
            not torch._C._is_alias_of(value, other)
            for other in (*leaves[index + 1 :], *others)
        )
    assert first.draws["stimulation"] is None
    _close(frames, source)
    assert [value._version for value in versioned] == versions
    frames[0].draws["noise"]["value"].add_(0.25)
    with torch.inference_mode():
        updated = _stack_runtime_packets(frames)
    assert not torch.equal(
        first.draws["noise"]["value"], updated.draws["noise"]["value"]
    )
    _close(updated, _legacy_stack(frames))


def test_requires_grad_frames_keep_ambient_stack_autograd_and_inference_behavior():
    root = torch.arange(3.0, requires_grad=True)
    frames = [
        dn.func.RuntimeDraws({"value": root}, {}),
        dn.func.RuntimeDraws({"value": root * 2}, {}),
    ]
    result = _stack_runtime_packets(frames)
    assert result.draws["value"].grad_fn is not None
    gradient = torch.autograd.grad(result.draws["value"].sum(), root)[0]
    torch.testing.assert_close(gradient, torch.full_like(root, 3))
    with torch.inference_mode():
        result = _stack_runtime_packets(frames)
        assert torch.is_inference(result.draws["value"])
    assert frames[0].draws["value"] is root and root.requires_grad


def test_observable_tensors_modes_stack_and_packet_types_keep_ambient_allocation(
    monkeypatch,
):
    class Subclass(torch.Tensor):
        pass

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            return function(*args, **(kwargs or {}))

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            return function(*args, **(kwargs or {}))

    frames = _frames()
    with torch.inference_mode():
        for mode in (FunctionMode(), DispatchMode()):
            with mode:
                result = _stack_runtime_packets(frames)
            assert all(
                torch.is_inference(value)
                for value in pytree.tree_leaves(result)
                if torch.is_tensor(value)
            )
        subclass = [
            packet._replace(
                draws={"value": packet.draws["noise"]["value"].as_subclass(Subclass)}
            )
            for packet in frames
        ]
        result = _stack_runtime_packets(subclass)
        assert type(result.draws["value"]) is Subclass
        assert torch.is_inference(result.draws["value"])
        value = frames[0].rng["bytes"]
        with monkeypatch.context() as patch:
            patch.setattr(value, "authored_attribute", "observable", raising=False)
            result = _stack_runtime_packets(frames)
            assert torch.is_inference(result.rng["bytes"])
        original = torch.stack
        calls = []

        def stack(*args, **kwargs):
            calls.append(torch.is_inference_mode_enabled())
            return original(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch, "stack", stack)
            result = _stack_runtime_packets(frames)
        assert calls and all(calls)
        assert torch.is_inference(result.rng["bytes"])
        custom = namedtuple("CustomPacket", "draws rng")
        for packets in (
            [custom(packet.draws, packet.rng) for packet in frames],
            [dn.func.RuntimeNoiseDraw({"value": torch.ones(2)}, {}) for _ in range(2)],
        ):
            result = _stack_runtime_packets(packets)
            assert type(result) is type(packets[0])
            assert all(
                torch.is_inference(value)
                for value in pytree.tree_leaves(result)
                if torch.is_tensor(value)
            )


def test_runtime_requests_keep_original_context_order_and_frame_empty_contracts():
    for inference in (False, True):
        events = []

        class Spec:
            has_noise = True
            poisson = None
            noise = SimpleNamespace(
                _empty_tape_draws=lambda: {"value": torch.empty((0, 2))}
            )
            index = 0

            def context(self, _model, _stimulation):
                events.append(
                    (
                        "context",
                        torch.is_inference_mode_enabled(),
                        torch.is_grad_enabled(),
                    )
                )
                return {"noise_draws": {}, "stimulation_draws": {}}

            def sample(self, context):
                events.append(
                    (
                        "sample",
                        torch.is_inference_mode_enabled(),
                        torch.is_grad_enabled(),
                    )
                )
                context["noise_draws"]["value"] = torch.full((2,), float(self.index))
                self.index += 1

        program = RuntimeSamplingProgram(
            {"population": Spec()}, {}, {}, {}, {}, {}, (), (), False
        )
        with torch.inference_mode(inference), torch.no_grad():
            tape = program.sample({}, {}, steps=3)
            assert not torch.is_inference(tape.draws["noise"]["value"])
            torch.testing.assert_close(
                tape.draws["noise"]["value"],
                torch.arange(3.0).unsqueeze(1).expand(3, 2),
            )
            assert events == [
                (kind, inference, False)
                for _ in range(3)
                for kind in ("context", "sample")
            ]
            frame = program.sample({}, {})
            assert torch.is_inference(frame.draws["noise"]["value"]) == inference
            count = len(events)
            empty = program.sample({}, {}, steps=0)
            assert empty.draws["noise"]["value"].shape == (0, 2)
            assert torch.is_inference(empty.draws["noise"]["value"]) == inference
            assert len(events) == count


def test_capture_and_transform_contexts_keep_ambient_stack(monkeypatch):
    frames = _frames()
    for owner, name, replacement in (
        (torch.compiler, "is_compiling", lambda: True),
        (torch._C, "_are_functorch_transforms_active", lambda: True),
        (torch.autograd.forward_ad, "_current_level", 0),
    ):
        with monkeypatch.context() as patch, torch.inference_mode():
            patch.setattr(owner, name, replacement)
            actual = _stack_runtime_packets(frames)
            assert torch.is_inference(actual.rng["bytes"])


def test_sampled_no_grad_and_inference_tapes_preserve_native_values_and_rng():
    source = _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    before = pytree.tree_map_only(
        torch.Tensor, lambda value: value.clone(), functional.extract()
    )
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        ordinary = functional.sample_runtime_draws(tensors.state, steps=3)
        native = dn.func.sample_runtime_draws(source, steps=3, dt=DT)
    with torch.inference_mode():
        inferred = functional.sample_runtime_draws(tensors.state, steps=3)
    _close(ordinary, inferred)
    _close(ordinary, native)
    assert all(
        not torch.is_inference(value)
        for value in pytree.tree_leaves((ordinary, inferred))
        if torch.is_tensor(value)
    )
    _close(functional.extract(), before)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def _case(record, *, backend="eager", disable=False):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    with torch.no_grad():
        prepared = functional.prepare(tensors.parameters, tensors.constants)
    packet = functional.sample_runtime_draws(tensors.state, steps=1)
    chunk = functional.compile_rollout_chunk(
        1, execution="unrolled", backend=backend, disable=disable
    )
    callbacks = (
        functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
        if record
        else None
    )
    selected = chunk if not record else chunk._runner_callback_chunk(callbacks, 1)
    carries = None if not record else callbacks._initialize(tensors.state)[0].carries
    return functional, tensors, prepared, packet, selected, carries


def _execute(case, packet):
    _functional, tensors, prepared, _original, selected, carries = case
    args = (
        tensors.parameters,
        prepared.values,
        tensors.state,
        _RuntimeInputs(draws=packet),
    )
    return (
        selected._rollout_values(*args, carries)
        if carries is not None
        else selected._rollout_values(*args)
    )


def _reference(case, packet):
    _functional, tensors, prepared, _original, selected, carries = case
    return selected._lowered(
        tensors.parameters,
        prepared.values,
        tensors.state,
        None,
        None,
        carries,
        None,
        None,
        packet,
    )


@pytest.mark.parametrize("record", (False, True))
def test_short_tail_reuses_backend_for_tapes_sampled_in_different_modes(record):
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    case = _case(record, backend=backend)
    functional, tensors, _prepared, ordinary, _selected, _carries = case
    with torch.inference_mode():
        inferred = functional.sample_runtime_draws(tensors.state, steps=1)
    _close(inferred, ordinary)
    assert all(
        not torch.is_inference(value)
        for value in pytree.tree_leaves((ordinary, inferred))
        if torch.is_tensor(value)
    )
    source = pytree.tree_map_only(
        torch.Tensor, lambda value: value.clone(), (tensors.state, ordinary, inferred)
    )
    versioned = [
        value
        for value in pytree.tree_leaves((tensors.state, ordinary, inferred))
        if torch.is_tensor(value)
    ]
    versions = [value._version for value in versioned]
    global_rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        for packet in (inferred, ordinary, inferred):
            actual = _execute(case, packet)
            _close(actual, _reference(case, packet))
            assert len(graphs) == 1
    _close((tensors.state, ordinary, inferred), source)
    assert [value._version for value in versioned] == versions
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("record", (False, True))
def test_compiled_tail_keeps_authored_packet_and_arithmetic_draw_identity(
    monkeypatch, record
):
    for disable in (False, True):
        case = _case(record, disable=disable)
        _functional, tensors, prepared, packet, selected, _carries = case
        seen = []

        def capture(*operands):
            seen.append(operands)
            return selected._lowered(*operands)

        monkeypatch.setattr(selected, "_compiled", capture)
        with torch.inference_mode():
            actual = _execute(case, packet)
            _close(actual, _reference(case, packet))
        operands = seen[0]
        assert operands[0] is tensors.parameters and operands[1] is prepared.values
        assert operands[-1] is None
        assert operands[-2] is packet
        assert operands[-2].draws is packet.draws
        assert operands[-2].rng is packet.rng


def test_custom_callback_observes_native_published_rng_metadata(monkeypatch):
    class RngMetadata(dn.func.FunctionalCallback):
        @staticmethod
        def observe(state):
            leaves = pytree.tree_leaves(state["random"]["rng"])
            fields = {
                "inference": [torch.is_inference(value) for value in leaves],
                "is_leaf": [value.is_leaf for value in leaves],
                "requires_grad": [value.requires_grad for value in leaves],
                "offset": [value.storage_offset() for value in leaves],
                "dtype": [int(value.dtype is torch.bool) for value in leaves],
                "shape": [size for value in leaves for size in value.shape],
                "stride": [size for value in leaves for size in value.stride()],
            }
            return {
                name: torch.tensor(
                    values, dtype=torch.int64, device=state["integrator"]["v"].device
                )
                for name, values in fields.items()
            }

        def initialize(self, state, auxiliary):
            del auxiliary
            return self.observe(state), None

        def update(self, carry, state, auxiliary):
            del carry, auxiliary
            observed = self.observe(state)
            return observed, observed

        def finalize(self, carry, emissions):
            return (
                {
                    name: value.new_empty((0, *value.shape))
                    for name, value in carry.items()
                }
                if emissions is None
                else emissions
            )

    original = _case(False)
    functional, tensors, prepared, packet, chunk, _carries = original
    callbacks = functional.make_callbacks({"metadata": RngMetadata()})
    selected = chunk._runner_callback_chunk(callbacks, 1)
    carries = callbacks._initialize(tensors.state)[0].carries

    def view(value):
        storage = torch.empty(
            value.numel() * 2 + 3, dtype=value.dtype, device=value.device
        )
        result = storage[3 : 3 + value.numel() * 2 : 2].reshape(value.shape)
        result.copy_(value)
        return result

    packet = packet._replace(rng=pytree.tree_map_only(torch.Tensor, view, packet.rng))
    case = functional, tensors, prepared, packet, selected, carries
    source_rng = pytree.tree_map_only(
        torch.Tensor, lambda value: value.clone(), packet.rng
    )
    leaves = pytree.tree_leaves(packet.rng)
    versions = [value._version for value in leaves]
    observed = []

    def eager(*operands):
        assert operands[-1] is None
        observed.append(operands[-2])
        return selected._lowered(*operands)

    monkeypatch.setattr(selected, "_compiled", eager)
    for inference in (False, True):
        with torch.inference_mode(inference), torch.no_grad():
            actual = _execute(case, packet)
            expected = _reference(case, packet)
            _close(actual, expected)
            metadata = actual[3]["metadata"]
            assert torch.all(metadata["inference"] == int(inference))
            assert torch.all(metadata["is_leaf"] == 1)
            assert torch.all(metadata["requires_grad"] == 0)
            assert torch.all(metadata["offset"] == 0)
            assert observed[-1].draws is packet.draws
            assert observed[-1] is packet
    _close(packet.rng, source_rng)
    assert [value._version for value in leaves] == versions
