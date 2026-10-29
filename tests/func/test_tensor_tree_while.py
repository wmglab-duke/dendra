"""Shared inference loops preserve arbitrary tensor tapes and owned carry."""

import gc
import weakref

import pytest
import torch
from torch.utils import _pytree as pytree

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._tensor_tree_while import _CompiledTensorTreeWhile

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _warnings():
    with torch_compiler_warning_context():
        yield


def _body(parameters, prepared, state, carries, sample):
    drive, clock = sample
    value = state["value"] + parameters["gain"] * drive + prepared["bias"]
    updated = {
        "value": value,
        "alias": value.view_as(value),
        "clock": state["clock"] + clock["delta"],
        "integer": state["integer"] + 1,
        "boolean": ~state["boolean"],
        "none": None,
    }
    return (updated, {"total": carries["total"] + value.sum()}), {
        "trace": {"value": value, "alias": value.view_as(value)},
        "clock": updated["clock"],
    }


def _case(steps):
    value = torch.arange(6, dtype=torch.float64).reshape(3, 2).T
    state = {
        "value": value,
        "alias": value.view_as(value),
        "clock": torch.tensor(0.0, dtype=torch.float64),
        "integer": torch.tensor([2], dtype=torch.long),
        "boolean": torch.tensor([False]),
        "none": None,
    }
    parameters = {"gain": torch.tensor(0.7, dtype=torch.float64)}
    prepared = {"bias": torch.tensor(-0.2, dtype=torch.float64)}
    tape = (
        torch.linspace(-0.5, 0.5, steps * 6, dtype=torch.float64).reshape(steps, 2, 3),
        {"delta": torch.full((steps,), 0.125, dtype=torch.float64), "unused": None},
    )
    return (
        parameters,
        prepared,
        state,
        {"total": torch.tensor(0.0, dtype=torch.float64)},
        tape,
    )


def _eager(arguments, steps, body=_body):
    parameters, prepared, state, carries, tape = arguments
    emissions = []
    for index in range(steps):
        sample = pytree.tree_map_only(torch.Tensor, lambda value: value[index], tape)
        (state, carries), emitted = body(parameters, prepared, state, carries, sample)
        emissions.append(emitted)
    return (
        state,
        carries,
        pytree.tree_map(lambda *leaves: torch.stack(leaves), *emissions),
    )


def _close(actual, expected):
    values, spec = pytree.tree_flatten(actual)
    references, reference_spec = pytree.tree_flatten(expected)
    assert spec == reference_spec
    for value, reference in zip(values, references, strict=True):
        if torch.is_tensor(value):
            torch.testing.assert_close(value, reference)
        else:
            assert value == reference


@pytest.mark.parametrize("steps", [1, 3, 4, 5, 9])
def test_shared_loop_handles_tails_nested_tapes_and_independent_public_leaves(steps):
    arguments = _case(steps)
    originals = pytree.tree_map_only(
        torch.Tensor, lambda value: value.clone(), arguments
    )
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    engine = _CompiledTensorTreeWhile(
        _body,
        steps=steps,
        compile_options={"backend": backend, "fullgraph": True},
        device="cpu",
    )
    with torch.no_grad():
        expected = _eager(arguments, steps)
        actual = engine(*arguments)
        saved = pytree.tree_map_only(torch.Tensor, lambda value: value.clone(), actual)
        repeated = engine(*arguments)
    _close(actual, expected)
    _close(repeated, expected)
    _close(actual, saved)
    _close(arguments, originals)
    has_loop = any(
        node.op == "call_function" and node.target is torch.ops.higher_order.while_loop
        for graph in graphs
        for node in graph.graph.nodes
    )
    assert has_loop == (steps >= 4)
    outputs = [value for value in pytree.tree_leaves(actual) if torch.is_tensor(value)]
    inputs = [
        value for value in pytree.tree_leaves(arguments) if torch.is_tensor(value)
    ]
    for index, value in enumerate(outputs):
        assert not any(
            torch._C._is_alias_of(value, other)
            for other in outputs[index + 1 :] + inputs
        )


def test_shared_loop_without_callbacks_emissions_or_tensor_tapes():
    def body(parameters, prepared, state, carries, sample):
        assert sample == {"input": None}
        return ({"count": state["count"] + parameters["increment"]}, carries), {}

    engine = _CompiledTensorTreeWhile(
        body,
        steps=7,
        compile_options={"backend": "eager", "fullgraph": True},
        device="cpu",
    )
    with torch.no_grad():
        final, carries, emitted = engine(
            {"increment": torch.tensor(2)},
            {},
            {"count": torch.tensor(0)},
            {},
            {"input": None},
        )
    assert final["count"].item() == 14
    assert carries == emitted == {}


@pytest.mark.parametrize("steps", (1, 5))
def test_shared_loop_materializes_optional_aliased_carries_with_invocation_formats(
    steps,
):
    def body(parameters, prepared, state, carries, sample):
        return ({"count": state["count"] + 1}, carries), {}

    source = torch.arange(24, dtype=torch.float64)
    view = source[2:14:2].reshape(2, 3)
    carries = {
        "view": view,
        "alias": view.view_as(view),
        "dense": source[3:9].reshape(3, 2).T,
        "empty": source[5:5],
        "none": None,
    }
    before, version = source.clone(), source._version
    engine = _CompiledTensorTreeWhile(
        body,
        steps=steps,
        compile_options={"backend": "eager", "fullgraph": True},
        device="cpu",
    )
    initial = {"count": torch.tensor(0)}
    with torch.no_grad():
        final, first, emitted = engine({}, {}, initial, carries, None)
        _, second, _ = engine({}, {}, initial, carries, None)
    assert final["count"].item() == steps
    assert emitted == {} and first["none"] is second["none"] is None
    _close(first, carries)
    _close(second, carries)
    outputs = [
        value for value in pytree.tree_leaves((first, second)) if torch.is_tensor(value)
    ]
    for index, value in enumerate(outputs):
        assert not torch._C._is_alias_of(value, source)
        assert all(
            not torch._C._is_alias_of(value, other) for other in outputs[index + 1 :]
        )
    for result in (first, second):
        for name, template in carries.items():
            if torch.is_tensor(template):
                assert result[name].stride() == template.clone().stride()
                assert result[name].storage_offset() == 0
    first["view"].add_(100)
    _close(second, carries)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert source._version == version


@pytest.mark.parametrize("view", (False, True))
def test_shared_loop_isolates_shared_lifted_fields_and_tape_views(view):
    def body(parameters, prepared, state, carries, sample):
        value = (
            state["value"]
            + parameters["scale"] * prepared["left"]
            + prepared["right"].square()
            + sample["drive"]
        )
        return ({"value": value}, carries), {"value": value}

    source = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    parameters = {"scale": source}
    prepared = {
        name: source.view_as(source) if view else source for name in ("left", "right")
    }
    tape = {"drive": source.expand(5)}
    initial = {"value": torch.tensor([-0.3, 0.2], dtype=torch.float64)}
    engine = _CompiledTensorTreeWhile(
        body,
        steps=5,
        compile_options={"backend": "eager", "fullgraph": True},
        device="cpu",
    )
    with torch.no_grad():
        final, _, emitted = engine(parameters, prepared, initial, {}, tape)
        frames = initial["value"] + torch.arange(1, 6, dtype=torch.float64)[:, None] * (
            2 * source.square() + source
        )
    torch.testing.assert_close(final["value"], frames[-1])
    torch.testing.assert_close(emitted["value"], frames)
    assert parameters["scale"] is source and source.is_leaf
    assert all(
        torch._C._is_alias_of(value, source)
        for value in (*prepared.values(), tape["drive"])
    )


@pytest.mark.parametrize("backend", ("eager", "inductor"))
def test_shared_loop_preserves_explicit_strided_views(backend):
    def body(parameters, prepared, state, carries, sample):
        value = parameters["value"]
        selected = value.as_strided((2,), value.stride(), value.storage_offset())
        updated = state["value"] + selected.square().sum() + prepared["same"].sum()
        return ({"value": updated}, carries), {}

    source = torch.nn.Parameter(torch.arange(12.0, dtype=torch.float64))
    value = source[3:9:2]
    parameters, prepared = {"value": value}, {"same": value}
    initial = {"value": torch.tensor(0.0, dtype=value.dtype)}
    engine = _CompiledTensorTreeWhile(
        body,
        steps=4,
        compile_options={"backend": backend, "fullgraph": True},
        device="cpu",
    )
    with torch.no_grad(), torch_compiler_warning_context():
        final, _, _ = engine(parameters, prepared, initial, {}, None)
        expected = initial
        for _ in range(4):
            (expected, _), _ = body(parameters, prepared, expected, {}, None)
    torch.testing.assert_close(final["value"], expected["value"])


def test_shared_capture_keeps_no_example_operands_and_reuses_explicit_values():
    engine = _CompiledTensorTreeWhile(
        _body,
        steps=4,
        compile_options={"backend": "eager", "fullgraph": True},
        device="cpu",
    )
    arguments = _case(4)
    references = [
        weakref.ref(value)
        for value in pytree.tree_leaves(arguments)
        if torch.is_tensor(value)
    ]
    with torch.no_grad():
        kernel = engine.kernel_for(*arguments)
        kernel(*arguments)
    del arguments
    gc.collect()
    assert all(reference() is None for reference in references)
    arguments = list(_case(4))
    arguments[0] = {"gain": torch.tensor(1.3, dtype=torch.float64)}
    with torch.no_grad():
        _close(kernel(*arguments), _eager(arguments, 4))


def test_shared_tape_validation_requires_matching_tensor_time_axes():
    engine = _CompiledTensorTreeWhile(_body, steps=4, compile_options={}, device="cpu")
    arguments = list(_case(4))
    arguments[-1] = (torch.ones(3, 2, 3), {"delta": torch.ones(4)})
    with pytest.raises(ValueError, match="leading time axis"):
        engine.kernel_for(*arguments)


def test_capture_preserves_gradient_flag_metadata_without_retaining_a_graph():
    def body(parameters, prepared, state, carries, sample):
        increment = 1 if parameters["value"].requires_grad else 2
        return ({"value": state["value"] + increment}, carries), {}

    engine = _CompiledTensorTreeWhile(
        body,
        steps=4,
        compile_options={"backend": "eager", "fullgraph": True},
        device="cpu",
    )
    parameters = {"value": torch.tensor(0.0, requires_grad=True)}
    with torch.no_grad():
        first = engine(parameters, {}, {"value": torch.tensor(0)}, {}, None)
        parameters["value"].requires_grad_(False)
        second = engine(parameters, {}, {"value": torch.tensor(0)}, {}, None)
    assert first[0]["value"].item() == 4
    assert second[0]["value"].item() == 8
    assert len(engine._kernels) == 2


def test_population_outer_capture_cache_tracks_parameter_freezing():
    from test_while_block_ownership import (
        _case,
    )
    from test_while_block_ownership import _close as close_population
    from test_while_block_ownership import (
        _eager,
        _ViewFunctional,
    )

    from dendra.func._population_while import _PopulationWhileExecutor

    class FlagFunctional(_ViewFunctional):
        @staticmethod
        def _step_values(parameters, prepared, state, inputs):
            updated, auxiliary = _ViewFunctional._step_values(
                parameters, prepared, state, inputs
            )
            increment = 1 if parameters["gain"].requires_grad else 2
            voltage = updated["integrator"]["v"] + increment
            updated["integrator"]["v"] = voltage
            return updated, {"v": voltage}

    case = _case(4)
    case.functional = FlagFunctional()
    executor = _PopulationWhileExecutor(
        case.functional,
        steps=4,
        plans=case.plans,
        compile_options={"backend": "eager", "fullgraph": True},
    )
    for selected in (True, False):
        case.parameters["gain"].requires_grad_(selected)
        with torch.no_grad():
            expected = _eager(case, 4)
            operands = (
                case.parameters,
                case.prepared,
                case.state,
                case.ve,
                case.intra,
                case.carries,
            )
            kernel, reason = executor.kernel_for(*operands)
            assert reason is None
            close_population(kernel(*operands), expected)
    assert len(executor._kernels) == 2


@pytest.mark.parametrize(
    "kind", ["strided_offset", "expanded", "negative", "conjugate"]
)
def test_capture_preserves_read_only_invariant_view_metadata(kind):
    def body(parameters, prepared, state, carries, sample):
        value = parameters["value"]
        increment = value.sum() + value.stride(0) + value.storage_offset()
        increment = increment + (10 if value.is_contiguous() else 20)
        increment = increment + (30 if value.is_leaf else 40)
        increment = increment + (50 if value.is_conj() else 60)
        increment = increment + (70 if value.is_neg() else 80)
        return ({"value": state["value"] + increment}, carries), {}

    base = torch.arange(12.0, requires_grad=True)
    if kind == "expanded":
        value = base[3:4].expand(3)
    elif kind == "negative":
        value = torch._neg_view(base[3:9:2])
    elif kind == "conjugate":
        value = torch.complex(base, base)[3:9:2].conj()
    else:
        value = base[3:9:2]
    parameters = {"value": value}
    original, version = base.detach().clone(), base._version
    # PyTorch's while HOP has no Negative/Conjugate dispatch. Its bounded
    # compiled tail still exercises capture metadata for those tensor views.
    steps = 3 if kind in {"negative", "conjugate"} else 4
    engine = _CompiledTensorTreeWhile(
        body,
        steps=steps,
        compile_options={"backend": "eager", "fullgraph": True},
        device="cpu",
    )
    state = {"value": value.new_zeros(())}
    with torch.no_grad():
        expected = state
        for _ in range(steps):
            (expected, _), _ = body(parameters, {}, expected, {}, None)
        actual, _, _ = engine(parameters, {}, state, {}, None)
    torch.testing.assert_close(actual["value"], expected["value"])
    torch.testing.assert_close(base, original)
    assert base._version == version and value.requires_grad


def test_capture_cache_distinguishes_invariant_storage_offsets():
    def body(parameters, prepared, state, carries, sample):
        increment = parameters["value"].storage_offset()
        return ({"value": state["value"] + increment}, carries), {}

    engine = _CompiledTensorTreeWhile(
        body,
        steps=4,
        compile_options={"backend": "eager", "fullgraph": True},
        device="cpu",
    )
    base = torch.arange(12.0)
    with torch.no_grad():
        for offset in (0, 2):
            parameters = {"value": base[offset : offset + 6 : 2]}
            final, _, _ = engine(parameters, {}, {"value": torch.tensor(0)}, {}, None)
            assert final["value"].item() == 4 * offset
    assert len(engine._kernels) == 2


def test_capture_cache_distinguishes_leaf_and_nonleaf_invariants():
    def body(parameters, prepared, state, carries, sample):
        increment = 1 if parameters["value"].is_leaf else 2
        return ({"value": state["value"] + increment}, carries), {}

    engine = _CompiledTensorTreeWhile(
        body,
        steps=4,
        compile_options={"backend": "eager", "fullgraph": True},
        device="cpu",
    )
    leaf = torch.ones(3, requires_grad=True)
    nonleaf = leaf * 2
    with torch.no_grad():
        for value, increment in ((leaf, 1), (nonleaf, 2)):
            final, _, _ = engine(
                {"value": value}, {}, {"value": torch.tensor(0)}, {}, None
            )
            assert final["value"].item() == 4 * increment
    assert len(engine._kernels) == 2


def test_capture_cache_distinguishes_inference_and_ordinary_invariants():
    def body(parameters, prepared, state, carries, sample):
        increment = 1 if torch.is_inference(parameters["value"]) else 2
        return ({"value": state["value"] + increment}, carries), {}

    engine = _CompiledTensorTreeWhile(
        body,
        steps=4,
        compile_options={"backend": "eager", "fullgraph": True},
        device="cpu",
    )
    ordinary = torch.ones(3)
    with torch.inference_mode():
        inference = torch.ones(3)
    with torch.no_grad():
        for value, increment in ((ordinary, 2), (inference, 1)):
            final, _, _ = engine(
                {"value": value}, {}, {"value": torch.tensor(0)}, {}, None
            )
            assert final["value"].item() == 4 * increment
    assert len(engine._kernels) == 2


def test_capture_releases_nonleaf_invariants_and_their_upstream_graphs():
    def body(parameters, prepared, state, carries, sample):
        value = (
            state["value"] + parameters["value"].sum() + prepared["bias"] + sample.sum()
        )
        return ({"value": value}, carries), {}

    engine = _CompiledTensorTreeWhile(
        body,
        steps=4,
        compile_options={"backend": "eager", "fullgraph": True},
        device="cpu",
    )

    def execute():
        upstream = torch.arange(6.0, requires_grad=True)
        parameter = (upstream * 2)[1:6:2]
        bias = upstream.sum() * 0.1
        tape = upstream.repeat(2).reshape(4, 3)
        references = tuple(
            weakref.ref(value) for value in (upstream, parameter, bias, tape)
        )
        arguments = (
            {"value": parameter},
            {"bias": bias},
            {"value": torch.tensor(0.0)},
            {},
            tape,
        )
        originals = tuple(
            value.detach().clone() for value in (upstream, parameter, bias, tape)
        )
        versions = tuple(value._version for value in (upstream, parameter, bias, tape))
        with torch.no_grad():
            kernel = engine.kernel_for(*arguments)
            final, _, _ = kernel(*arguments)
            assert not final["value"].requires_grad
        for value, original, version in zip(
            (upstream, parameter, bias, tape), originals, versions, strict=True
        ):
            torch.testing.assert_close(value, original)
            assert value._version == version
        return references

    references = execute()
    gc.collect()
    assert all(reference() is None for reference in references)
