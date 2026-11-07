"""Inference specializes owned recurrent packs while retaining public ownership."""

from __future__ import annotations

import gc
import weakref
from contextlib import nullcontext

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population_while import _PopulationWhileExecutor
from dendra.func._state_projection import _state_projection
from dendra.func._tensor_contracts import _clone_contiguous_tree
from dendra.func._tensor_tree_while import _CompiledTensorTreeWhile

pytestmark = pytest.mark.cpu


@_state_projection(_clone_contiguous_tree)
def _project_tree(state):
    return state


@pytest.fixture(autouse=True)
def _warnings():
    with torch_compiler_warning_context():
        yield


def _leaves(tree):
    return [value for value in pytree.tree_leaves(tree) if torch.is_tensor(value)]


def _clone(tree):
    return pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), tree)


def _close(actual, expected):
    values, spec = pytree.tree_flatten(actual)
    references, reference_spec = pytree.tree_flatten(expected)
    assert spec == reference_spec
    for value, reference in zip(values, references, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


def _owned(actual, originals):
    values = _leaves(actual)
    for index, value in enumerate(values):
        assert all(
            not torch._C._is_alias_of(value, other)
            for other in (*values[index + 1 :], *_leaves(originals))
        )


def _body(parameters, prepared, state, carries, sample):
    value = (
        state["value"]
        + parameters["gain"] * sample
        + prepared["bias"]
        + 0.1 * state["alias"]
        + 0.2 * carries["fixed"]
    )
    flag = ~state["flag"]
    updated = {
        "value": value,
        "alias": value.view_as(value),
        "integer": state["integer"] + 1,
        "flag": flag,
        "none": None,
    }
    return (
        updated,
        {"sum": carries["sum"] + value.sum(), "fixed": carries["fixed"], "none": None},
    ), {"value": value, "flag": flag}


def _eager(parameters, prepared, state, carries, tape, body=_body):
    emissions = []
    for sample in tape:
        (state, carries), emitted = body(parameters, prepared, state, carries, sample)
        emissions.append(emitted)
    return state, carries, pytree.tree_map(lambda *xs: torch.stack(xs), *emissions)


def _recurrent(kind, offset):
    with torch.inference_mode(kind == "inference"):
        value = torch.arange(6, dtype=torch.float64).reshape(2, 3) + offset
        if kind == "offset":
            storage = torch.empty(17, dtype=value.dtype)
            selected = storage[3:15:2].view_as(value)
            selected.copy_(value)
            value = selected
        elif kind == "transposed":
            value = value.T.contiguous().T
        independent = kind == "independent"
        state = {
            "value": value,
            "alias": value.clone() if independent else value.view_as(value),
            "integer": torch.tensor([1, 3]),
            "flag": torch.tensor([False, True]),
            "none": None,
        }
        carries = {
            "sum": torch.tensor(offset, dtype=value.dtype),
            "fixed": value.clone() if independent else value,
            "none": None,
        }
        return state, carries


def test_one_backend_graph_handles_recurrent_alias_layout_and_inference_changes():
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    parameters = {"gain": torch.tensor(0.7, dtype=torch.float64)}
    prepared = {"bias": torch.tensor(-0.2, dtype=torch.float64)}
    tape = torch.linspace(-0.3, 0.4, 30, dtype=torch.float64).reshape(5, 2, 3)
    engine = _CompiledTensorTreeWhile(
        _body,
        steps=5,
        compile_options={"backend": backend, "fullgraph": True},
        device="cpu",
        project_state=_project_tree,
    )
    previous = []
    with torch.no_grad():
        for index, kind in enumerate(
            ("fresh", "independent", "offset", "transposed", "inference")
        ):
            state, carries = _recurrent(kind, index * 0.1)
            operands = parameters, prepared, state, carries, tape
            before = _clone(operands)
            versioned = [
                value for value in _leaves(operands) if not value.is_inference()
            ]
            versions = [value._version for value in versioned]
            expected = _eager(*operands)
            actual = engine(*operands)
            _close(actual, expected)
            _owned(actual, (operands, previous))
            _close(operands, before)
            assert [value._version for value in versioned] == versions
            assert actual[0]["value"].is_contiguous()
            assert actual[1]["fixed"].stride() == carries["fixed"].clone().stride()
            assert len(graphs) == 1, kind
            previous.append(actual)
        saved = _clone(previous)
        state, carries = previous[-1][:2]
        continued = engine(parameters, prepared, state, carries, tape)
        _close(continued, _eager(parameters, prepared, state, carries, tape))
        _owned(continued, (parameters, prepared, state, carries, tape, previous))
        continued[0]["value"].add_(100)
        _close(previous, saved)
    assert len(graphs) == 1
    assert len(engine._layouts) == 1
    assert any(
        node.op == "call_function" and node.target is torch.ops.higher_order.while_loop
        for node in graphs[0].graph.nodes
    )


def test_logical_state_shapes_specialize_even_when_packed_sizes_match():
    def body(parameters, prepared, state, carries, sample):
        updated = {"value": state["value"] + sample}
        return (updated, carries), updated

    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    engine = _CompiledTensorTreeWhile(
        body,
        steps=4,
        compile_options={"backend": backend, "fullgraph": True},
        device="cpu",
        project_state=_project_tree,
    )
    tape = torch.arange(4, dtype=torch.float64)
    with torch.no_grad():
        for shape in ((2, 3), (3, 2), (2, 3)):
            state = {"value": torch.arange(6, dtype=torch.float64).reshape(shape)}
            actual = engine({}, {}, state, {}, tape)
            _close(actual, _eager({}, {}, state, {}, tape, body))
            assert actual[0]["value"].shape == shape
            assert actual[2]["value"].shape == (4, *shape)
    assert len(engine._kernels) == len(graphs) == 2
    assert len(engine._layouts) == 2


def test_kernel_admission_does_not_retain_packed_examples():
    references = []
    configuration = {"backend": "sentinel"}

    def tracer(function):
        trace = make_fx(function)

        def capture(*examples):
            for value in _leaves(examples):
                references.append(weakref.ref(value))
                if value._base is not None:
                    references.append(weakref.ref(value._base))
            return trace(*examples)

        return capture

    def compiler(function, _options):
        function.get_compiler_config = lambda: configuration
        return function

    engine = _CompiledTensorTreeWhile(
        _body,
        steps=1,
        compile_options={},
        device="cpu",
        tracer=tracer,
        compiler=compiler,
        project_state=_project_tree,
    )

    def admit():
        state, carries = _recurrent("offset", 0.3)
        operands = (
            {"gain": torch.tensor(0.7, dtype=torch.float64)},
            {"bias": torch.tensor(-0.2, dtype=torch.float64)},
            state,
            carries,
            torch.ones(1, 2, 3, dtype=torch.float64),
        )
        references.extend(weakref.ref(value) for value in _leaves(operands))
        return engine.kernel_for(*operands)

    with torch.no_grad():
        kernel = admit()
    assert callable(kernel.__wrapped__)
    assert callable(kernel.get_compiler_config)
    assert kernel.get_compiler_config() is configuration
    gc.collect()
    assert references and all(reference() is None for reference in references)
    state, carries = _recurrent("independent", 0.8)
    operands = (
        {"gain": torch.tensor(1.3, dtype=torch.float64)},
        {"bias": torch.tensor(0.1, dtype=torch.float64)},
        state,
        carries,
        torch.full((1, 2, 3), 0.4, dtype=torch.float64),
    )
    with torch.no_grad():
        _close(kernel(*operands), _eager(*operands))


@pytest.mark.parametrize("kind", ("subclass", "function_mode", "dispatch_mode"))
def test_input_cloning_keeps_dispatch_and_failure_before_compiled_execution(kind):
    def body(parameters, prepared, state, carries, sample):
        return ({name: value + sample for name, value in state.items()}, carries), {}

    events = []
    failure = RuntimeError("second input clone")
    fail = False

    def record(function, args, kwargs):
        selected = (
            function is torch.ops.aten.clone.default
            if kind == "dispatch_mode"
            else function is torch.Tensor.clone
        )
        if selected and args[0] is authored:
            assert args[1:] == ()
            assert kwargs == {"memory_format": torch.contiguous_format}
            events.append(id(args[0]))
            if fail and len(events) == 2:
                raise failure

    value = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    if kind == "subclass":

        class Value(torch.Tensor):
            @classmethod
            def __torch_function__(cls, function, types, args=(), kwargs=None):
                record(function, args, kwargs or {})
                return super().__torch_function__(function, types, args, kwargs or {})

        authored = torch.Tensor._make_subclass(Value, value, False)
        context = nullcontext
    elif kind == "function_mode":

        class Mode(TorchFunctionMode):
            def __torch_function__(self, function, types, args=(), kwargs=None):
                record(function, args, kwargs or {})
                return function(*args, **(kwargs or {}))

        authored, context = value, Mode
    else:

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                record(function, args, kwargs or {})
                return function(*args, **(kwargs or {}))

        authored, context = value, Mode

    def compiler(function, _options):
        def execute(*operands):
            events.append("compiled")
            return function(*operands)

        return execute

    engine = _CompiledTensorTreeWhile(
        body,
        steps=1,
        compile_options={},
        device="cpu",
        compiler=compiler,
        project_state=_project_tree,
    )
    tape = torch.ones(1, dtype=value.dtype)
    with torch.no_grad():
        kernel = engine.kernel_for({}, {}, {"first": value, "second": value}, {}, tape)
        events.clear()
        with context():
            actual = kernel({}, {}, {"first": authored, "second": authored}, {}, tape)
        assert events[:2] == [id(authored), id(authored)]
        assert "compiled" in events[2:]
        torch.testing.assert_close(actual[0]["first"], authored + 1)
        events.clear()
        fail = True
        with context(), pytest.raises(RuntimeError) as caught:
            kernel({}, {}, {"first": authored, "second": authored}, {}, tape)
        fail = False
    assert caught.value is failure
    assert events == [id(authored), id(authored)]


def test_population_wrappers_share_one_packed_backend_capture():
    from tests.func.test_while_block_ownership import (
        _assert_owned,
        _case,
    )
    from tests.func.test_while_block_ownership import _clone as clone_population
    from tests.func.test_while_block_ownership import _close as close_population
    from tests.func.test_while_block_ownership import _eager as eager_population

    case = _case(5)
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    executor = _PopulationWhileExecutor(
        case.functional,
        steps=5,
        plans=case.plans,
        compile_options={"backend": backend, "fullgraph": True},
    )
    with torch.no_grad():
        first = None
        for kind in ("original", "cloned", "inference", "continued"):
            if kind == "continued":
                state, carries = first[0], first[2]
            elif kind == "inference":
                with torch.inference_mode():
                    state, carries = clone_population((case.state, case.carries))
            elif kind == "cloned":
                state, carries = clone_population((case.state, case.carries))
            else:
                state, carries = case.state, case.carries
            operands = (
                case.parameters,
                case.prepared,
                state,
                case.ve,
                case.intra,
                carries,
            )
            saved = clone_population(operands)
            kernel, reason = executor.kernel_for(*operands)
            assert reason is None
            actual = kernel(*operands)
            close_population(
                actual, eager_population(case, 5, state=state, carries=carries)
            )
            close_population(operands, saved, exact=True)
            _assert_owned(actual, (operands, first))
            if first is None:
                first = actual
            assert len(graphs) == 1, kind
