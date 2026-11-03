"""While tapes specialize owned samples without changing indexed semantics."""

from __future__ import annotations

import gc
import weakref
from contextlib import nullcontext
from types import MethodType

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._tensor_tree_while import _CompiledTensorTreeWhile
from tests.func.test_while_input_packing import (
    _clone,
    _close,
    _leaves,
    _owned,
    _project_tree,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _warnings():
    with torch_compiler_warning_context():
        yield


def _body(parameters, prepared, state, carries, sample):
    drive = sample["drive"]
    # index_select(...).squeeze(0) materializes logical samples independently
    # of the incoming tape's offsets, strides and conjugate/negative flags.
    layout_gain = (
        1.0
        if drive.is_contiguous()
        and drive.storage_offset() == 0
        and not drive.is_conj()
        and not drive.is_neg()
        else 100.0
    )
    assert sample["nested"][0] is None
    value = state["value"] + layout_gain * (
        parameters["gain"] * drive + 0.3 * sample["alias"] + prepared["bias"]
    )
    return ({"value": value}, {"sum": carries["sum"] + value.sum()}), {"value": value}


def _flat_body(parameters, prepared, state, carries, sample):
    return ({"value": state["value"] + sample}, carries), {}


def _packet(steps, kind="fresh", alias="same", shift=0.0):
    with torch.inference_mode(kind == "inference"):
        value = torch.arange(steps * 6, dtype=torch.float64).reshape(steps, 2, 3)
        value = value * 0.02 + shift
        if kind == "offset":
            storage = torch.empty(value.numel() * 2 + 7, dtype=value.dtype)
            view = storage[3 : 3 + value.numel() * 2 : 2].reshape(value.shape)
            view.copy_(value)
            value = view
        elif kind == "transposed":
            value = value.transpose(1, 2).contiguous().transpose(1, 2)
        elif kind == "negative":
            value = torch._neg_view(-value)
        other = (
            value
            if alias == "same"
            else value.view_as(value)
            if alias == "view"
            else value.clone()
        )
        return {
            "drive": value,
            "alias": other,
            "nested": (None, {"unused": value}),
            "empty": torch.empty(steps, 0, 3),
            "rng": torch.arange(steps * 8).to(torch.uint8).reshape(steps, 8),
        }


def _operands(tape):
    return (
        {"gain": torch.tensor(0.7, dtype=torch.float64)},
        {"bias": torch.tensor(-0.1, dtype=torch.float64)},
        {"value": torch.zeros(2, 3, dtype=torch.float64)},
        {"sum": torch.zeros((), dtype=torch.float64)},
        tape,
    )


def _reference(parameters, prepared, state, carries, tape):
    emissions = []
    for index in range(tape["drive"].shape[0]):
        sample = pytree.tree_map_only(
            torch.Tensor,
            lambda value: value.index_select(
                0, torch.tensor([index], device=value.device, dtype=torch.long)
            ).squeeze(0),
            tape,
        )
        (state, carries), emitted = _body(parameters, prepared, state, carries, sample)
        emissions.append(emitted)
    return (
        state,
        carries,
        pytree.tree_map(lambda *values: torch.stack(values), *emissions),
    )


def test_one_while_capture_handles_sample_layout_inference_and_alias_changes():
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    engine = _CompiledTensorTreeWhile(
        _body,
        steps=5,
        compile_options={"backend": backend, "fullgraph": True},
        device="cpu",
        project_state=_project_tree,
    )
    previous = []
    with torch.no_grad():
        for index, (kind, alias) in enumerate(
            (
                ("fresh", "same"),
                ("offset", "view"),
                ("transposed", "independent"),
                ("inference", "same"),
                ("negative", "view"),
                ("fresh", "independent"),
            )
        ):
            operands = _operands(_packet(5, kind, alias, index * 0.2))
            before = _clone(operands)
            versioned = [
                value for value in _leaves(operands) if not value.is_inference()
            ]
            versions = [value._version for value in versioned]
            actual = engine(*operands)
            _close(actual, _reference(*operands))
            _owned(actual, (operands, previous))
            _close(operands, before)
            assert [value._version for value in versioned] == versions
            assert len(engine._kernels) == len(graphs) == 1, (kind, alias)
            previous.append(actual)
    assert any(
        node.op == "call_function" and node.target is torch.ops.higher_order.while_loop
        for node in graphs[0].graph.nodes
    )


def test_compiled_sample_operands_are_fresh_independent_even_when_unused_or_aliased():
    source_refs, observed = [], []

    def compiler(function, _options):
        def execute(parameters, prepared, initial, invariant, samples):
            assert all(
                type(value) is torch.Tensor
                and value.is_contiguous()
                and value.storage_offset() == 0
                and not value.is_inference()
                and not value.requires_grad
                and value.grad_fn is None
                for value in samples
            )
            for index, value in enumerate(samples):
                assert all(
                    not torch._C._is_alias_of(value, other)
                    for other in samples[index + 1 :]
                )
                assert all(
                    reference() is None or not torch._C._is_alias_of(value, reference())
                    for reference in source_refs
                )
            observed.append(tuple(weakref.ref(value) for value in samples))
            return function(parameters, prepared, initial, invariant, samples)

        return execute

    engine = _CompiledTensorTreeWhile(
        _body,
        steps=1,
        compile_options={},
        device="cpu",
        compiler=compiler,
        project_state=_project_tree,
    )
    with torch.no_grad():
        for kind, alias in (
            ("fresh", "same"),
            ("offset", "view"),
            ("inference", "independent"),
        ):
            operands = _operands(_packet(1, kind, alias, 0.4))
            if kind == "fresh":
                operands[-1]["drive"].requires_grad_()
            source_refs[:] = [weakref.ref(value) for value in _leaves(operands[-1])]
            _close(engine(*operands), _reference(*operands))
            assert len(engine._kernels) == 1
            gc.collect()
            assert all(reference() is None for reference in observed[-1])


def test_sample_admission_retains_no_tape_graph_and_raw_callable_reads_live_updates():
    references = []

    def tracer(function):
        trace = make_fx(function)

        def capture(*examples):
            references.extend(weakref.ref(value) for value in _leaves(examples))
            return trace(*examples)

        return capture

    engine = _CompiledTensorTreeWhile(
        _body,
        steps=1,
        compile_options={},
        device="cpu",
        tracer=tracer,
        compiler=lambda function, _options: function,
        project_state=_project_tree,
    )

    def admit():
        with torch.enable_grad():
            root = torch.arange(6, dtype=torch.float64, requires_grad=True)
            drive = (root * 0.2).reshape(1, 2, 3)
        assert drive.grad_fn is not None
        tape = _packet(1)
        tape["drive"] = drive
        tape["alias"] = tape["drive"].view_as(tape["drive"])
        references.extend(weakref.ref(value) for value in (root, *_leaves(tape)))
        return engine.kernel_for(*_operands(tape))

    with torch.no_grad():
        kernel = admit()
    gc.collect()
    assert references and all(reference() is None for reference in references)
    operands = _operands(_packet(1, shift=0.4))
    with torch.no_grad():
        first = kernel(*operands)
        _close(first, _reference(*operands))
        operands[-1]["drive"].add_(0.25)
        second = kernel(*operands)
        _close(second, _reference(*operands))
    assert not torch.equal(first[0]["value"], second[0]["value"])
    assert len(engine._kernels) == 1


def test_independent_engine_plans_keep_their_own_sample_captures():
    captures = []
    engines = []
    with torch.no_grad():
        for index in range(6):

            def compiler(function, _options, index=index):
                captures.append(index)
                return function

            engine = _CompiledTensorTreeWhile(
                _body,
                steps=1,
                compile_options={},
                device="cpu",
                compiler=compiler,
                project_state=_project_tree,
            )
            engines.append(engine)
            for kind in ("fresh", "offset"):
                operands = _operands(_packet(1, kind, shift=index * 0.1))
                _close(engine(*operands), _reference(*operands))
            assert len(engine._kernels) == 1
    assert captures == list(range(6))
    assert len({id(engine) for engine in engines}) == 6


def test_sample_logical_shapes_dtypes_and_none_positions_still_specialize():
    captures = []

    def body(parameters, prepared, state, carries, sample):
        return ({"value": state["value"] + sample["drive"].sum()}, carries), {}

    def compiler(function, _options):
        captures.append(True)
        return function

    engine = _CompiledTensorTreeWhile(
        body,
        steps=1,
        compile_options={},
        device="cpu",
        compiler=compiler,
        project_state=_project_tree,
    )
    cases = (
        ((2, 3), torch.float64, False, 1),
        ((3, 2), torch.float64, False, 2),
        ((2, 3), torch.float32, False, 3),
        ((2, 3), torch.float64, True, 4),
        ((2, 3), torch.float64, False, 4),
    )
    with torch.no_grad():
        for shape, dtype, optional, expected in cases:
            tape = {
                "drive": torch.arange(6, dtype=dtype).reshape(1, *shape),
                "optional": torch.zeros(1) if optional else None,
            }
            state = {"value": torch.zeros(2, 3, dtype=torch.float64)}
            final, carries, emitted = engine({}, {}, state, {}, tape)
            torch.testing.assert_close(
                final["value"], torch.full_like(state["value"], 15.0)
            )
            assert carries == emitted == {}
            assert len(engine._kernels) == len(captures) == expected


@pytest.mark.parametrize("kind", ("complex", "empty"))
def test_owned_samples_keep_conjugate_values_and_empty_view_semantics(kind):
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    def body(parameters, prepared, state, carries, sample):
        assert sample.is_contiguous() and sample.storage_offset() == 0
        assert not sample.is_conj() and not sample.is_neg()
        return ({"value": state["value"] + sample.sum()}, carries), {}

    if kind == "complex":
        real = torch.arange(24, dtype=torch.float64).reshape(4, 2, 3)
        base = torch.complex(real, real + 1)
        tapes = (base, base.conj(), base.conj().clone(), torch._neg_view(base))
    else:
        base = torch.empty(4, 0, 3, dtype=torch.float64)
        tapes = (
            base,
            torch.empty(64, dtype=base.dtype).as_strided((4, 0, 3), (9, 3, 1), 7),
        )
    engine = _CompiledTensorTreeWhile(
        body,
        steps=4,
        compile_options={"backend": backend, "fullgraph": True},
        device="cpu",
        project_state=_project_tree,
    )
    with torch.no_grad():
        for tape in tapes:
            state = {"value": torch.zeros((), dtype=tape.dtype)}
            final, carries, emitted = engine({}, {}, state, {}, tape)
            torch.testing.assert_close(final["value"], tape.sum())
            assert carries == emitted == {}
            assert len(engine._kernels) == len(graphs) == 1


@pytest.mark.parametrize(
    "kind",
    (
        "subclass",
        "function_mode",
        "dispatch_mode",
        "instance_index_select",
        "class_index_select",
    ),
)
def test_observable_sample_selection_keeps_raw_call_order_and_exception_identity(
    kind, monkeypatch
):
    original = torch.Tensor.index_select
    failure = RuntimeError("authored tape selection")
    events = []
    value = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)

    def selected(function, args):
        operation = (
            torch.ops.aten.index_select.default if kind == "dispatch_mode" else original
        )
        if function is operation and args[0] is authored:
            events.append("selection")
            raise failure

    if kind == "subclass":

        class Value(torch.Tensor):
            @classmethod
            def __torch_function__(cls, function, types, args=(), kwargs=None):
                selected(function, args)
                return super().__torch_function__(function, types, args, kwargs or {})

        authored, context = (
            torch.Tensor._make_subclass(Value, value, False),
            nullcontext,
        )
    elif kind == "function_mode":

        class Mode(TorchFunctionMode):
            def __torch_function__(self, function, types, args=(), kwargs=None):
                selected(function, args)
                return function(*args, **(kwargs or {}))

        authored, context = value, Mode
    elif kind == "dispatch_mode":

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                selected(function, args)
                return function(*args, **(kwargs or {}))

        authored, context = value, Mode
    else:
        authored, context = value, nullcontext

        def select(self, *args, **kwargs):
            selected(original, (self, *args))
            return original(self, *args, **kwargs)

        if kind == "instance_index_select":
            monkeypatch.setattr(authored, "index_select", MethodType(select, authored))
        else:
            monkeypatch.setattr(torch.Tensor, "index_select", select)

    def compiler(function, _options):
        events.append("compiled")
        return function

    engine = _CompiledTensorTreeWhile(
        _flat_body,
        steps=1,
        compile_options={},
        device="cpu",
        compiler=compiler,
        project_state=_project_tree,
    )
    with torch.no_grad(), context(), pytest.raises(RuntimeError) as caught:
        engine.kernel_for({}, {}, {"value": value[0].clone()}, {}, authored)
    assert caught.value is failure
    assert events == ["selection"]
    assert engine._kernels == {}


@pytest.mark.parametrize("method", ("squeeze", "stride"))
def test_class_sample_method_overrides_retain_the_raw_compiled_operand(
    method, monkeypatch
):
    authored = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)
    original = getattr(torch.Tensor, method)
    calls, boundaries = [], []

    def observed(self, *args, **kwargs):
        calls.append(method)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, method, observed)

    def compiler(function, _options):
        def execute(parameters, prepared, initial, invariant, samples):
            assert samples[0] is authored
            boundaries.append(True)
            return function(parameters, prepared, initial, invariant, samples)

        return execute

    engine = _CompiledTensorTreeWhile(
        _flat_body,
        steps=1,
        compile_options={},
        device="cpu",
        compiler=compiler,
        project_state=_project_tree,
    )
    state = {"value": torch.zeros(2, 3, dtype=torch.float64)}
    with torch.no_grad():
        final, carries, emitted = engine({}, {}, state, {}, authored)
    torch.testing.assert_close(final["value"], authored[0])
    assert carries == emitted == {}
    assert calls and boundaries == [True]
