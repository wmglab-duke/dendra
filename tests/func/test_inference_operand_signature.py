"""Initial and recurrent inference chunks share their owned tensor signature."""

from types import MappingProxyType

import pytest
import torch
from test_compiled_chunk import DT, _assert_tree_close, _drives, _model

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._inference import _canonicalize_inference_tree

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script` is deprecated.*"),
    pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated.*"),
]


def test_inference_tree_is_local_and_preserves_duplicate_tensor_references():
    original = torch.arange(12.0).reshape(3, 4).transpose(0, 1)
    original_version = original._version
    state = {"a": original, "nested": MappingProxyType({"b": original})}
    carries = {"same": original}
    with torch.inference_mode():
        existing = torch.ones(2)
        tree = (state, carries, existing)
        result = _canonicalize_inference_tree(tree)
        assert result[0]["a"] is result[0]["nested"]["b"]
        assert result[0]["a"] is result[1]["same"]
        assert result[0]["a"] is not original
        assert result[2] is existing
        assert torch.is_inference(result[0]["a"])
        assert result[0]["a"].stride() == original.stride()
        torch.testing.assert_close(result[0]["a"], original, rtol=0, atol=0)
        assert _canonicalize_inference_tree(result) is result
    assert state["a"] is original
    assert carries["same"] is original
    assert not torch.is_inference(original)
    assert original._version == original_version


def test_inference_tree_observes_restart_edits_without_retaining_a_snapshot():
    value = torch.tensor(1.0)
    tree = {"value": value}
    with torch.inference_mode():
        first = _canonicalize_inference_tree(tree)
        value.add_(2)
        second = _canonicalize_inference_tree(tree)
    assert first["value"].item() == 1
    assert second["value"].item() == 3
    assert first["value"] is not second["value"]


def test_distinct_overlapping_views_preserve_values_schema_and_source_storage():
    source = torch.arange(20.0).reshape(4, 5)
    views = (source[:, 1:4], source.transpose(0, 1))
    before = source.clone()
    version = source._version
    with torch.inference_mode():
        actual = _canonicalize_inference_tree(views)
    for value, expected in zip(actual, views, strict=True):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        assert value.shape == expected.shape
        assert value.dtype == expected.dtype
        assert value.device == expected.device
        assert torch.is_inference(value)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert source._version == version


def test_no_grad_does_not_convert_ordinary_tensors():
    tree = {"value": torch.ones(2, requires_grad=True)}
    with torch.no_grad():
        assert _canonicalize_inference_tree(tree) is tree


def test_training_tensors_retain_identity_and_autograd_history():
    original = torch.ones(2, requires_grad=True)
    tree = {"value": original * 2}
    with torch.enable_grad():
        actual = _canonicalize_inference_tree(tree)
        assert actual is tree
        gradient = torch.autograd.grad(actual["value"].sum(), original)[0]
    torch.testing.assert_close(gradient, torch.full_like(original, 2))


@pytest.mark.parametrize("fallback", ["dynamo", "functorch", "forward_ad"])
def test_inference_canonicalization_bypasses_transform_and_capture_modes(
    monkeypatch, fallback
):
    tree = {"value": torch.ones(2)}
    if fallback == "dynamo":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif fallback == "functorch":
        monkeypatch.setattr(torch._C, "_are_functorch_transforms_active", lambda: True)
    else:
        monkeypatch.setattr(torch.autograd.forward_ad, "_current_level", 0)
    with torch.inference_mode():
        assert _canonicalize_inference_tree(tree) is tree


@pytest.mark.parametrize("record", [False, True])
def test_compiled_boundary_keeps_versioned_bindings_and_authored_inputs(
    monkeypatch, record
):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(functional._population, 4)
    inputs = dn.func.RolloutInput(ve, intra)
    chunk = functional.compile_rollout_chunk(4, execution="unrolled", backend="eager")
    carries = {}
    selected = chunk
    if record:
        callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
        selected = chunk._runner_callback_chunk(callbacks, 4)
        carries = callbacks._initialize(tensors.state)[0].carries
    captured = []
    validated = []
    input_snapshots = [(value, value._version, value.clone()) for value in (ve, intra)]
    execute_operands = chunk._execute_operands

    def capture_validated(operands, *, prewarm_structured):
        validated.append(operands)
        return execute_operands(operands, prewarm_structured=prewarm_structured)

    monkeypatch.setattr(chunk, "_execute_operands", capture_validated)

    def capture(*operands):
        captured.append(operands)
        return selected._lowered(*operands)

    selected._compiled = capture
    with torch.inference_mode():
        if record:
            selected._rollout_values(
                tensors.parameters, prepared.values, tensors.state, inputs, carries
            )
        else:
            selected(tensors.parameters, prepared, tensors.state, inputs)
    actual = captured[0]
    assert actual[0] is tensors.parameters
    assert actual[1] is prepared.values
    # Public rollout validation may broadcast/materialize authored drives
    # before the compiled boundary. Canonicalization must preserve those
    # already-validated operands, not promise a stronger public identity API.
    expected_inputs = (ve, intra) if record else (validated[0].ve, validated[0].intra)
    assert actual[3] is expected_inputs[0]
    assert actual[4] is expected_inputs[1]
    for value, version, original in input_snapshots:
        assert value._version == version
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert all(
        not torch.is_inference(value)
        for value in torch.utils._pytree.tree_leaves((actual[0], actual[1]))
        if isinstance(value, torch.Tensor)
    )
    assert all(
        torch.is_inference(value)
        for value in torch.utils._pytree.tree_leaves(actual[2])
    )
    if record:
        assert all(
            torch.is_inference(value)
            for value in torch.utils._pytree.tree_leaves(actual[5])
        )
    assert all(
        not torch.is_inference(value)
        for value in torch.utils._pytree.tree_leaves(tensors.state)
    )


@pytest.mark.parametrize("record", [False, True])
def test_explicitly_disabled_compilation_skips_canonicalization(monkeypatch, record):
    import dendra.func._compiled as compiled

    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(4, backend="eager", disable=True)
    callbacks = (
        functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
        if record
        else None
    )

    def unexpected(_tree):
        pytest.fail("eager-disabled execution must not normalize its public inputs")

    monkeypatch.setattr(compiled, "_canonicalize_inference_tree", unexpected)
    with torch.inference_mode(), torch_compiler_warning_context():
        dn.func.run(
            chunk.bind(tensors.parameters, prepared),
            tensors.state,
            steps=8,
            callbacks=callbacks,
        )


@pytest.mark.parametrize(
    "execution,record",
    [
        ("unrolled", False),
        ("default", False),
        ("unrolled", True),
        ("default", True),
        ("scan", True),
    ],
)
def test_initial_recurrent_and_resumed_inference_reuse_one_compiled_variant(
    execution, record
):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    with torch.no_grad():
        prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = (
        functional.make_callbacks({"trace": dn.func.Recorder(["v", "hh.m", "t"])})
        if record
        else None
    )
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    kernel = functional.compile_rollout_chunk(4, execution=execution, backend=backend)
    state_before = functional._clone_state_tree(tensors.state)
    with torch.inference_mode(), torch_compiler_warning_context():
        # The fixture retains trainable source parameters. Its no-grad
        # preparation is valid only at this actual inference boundary.
        bound = kernel.bind(tensors.parameters, prepared)
        expected, expected_aux = dn.func.run(
            functional.bind(tensors.parameters, prepared),
            tensors.state,
            steps=12,
            callbacks=callbacks,
        )
        first, auxiliary = dn.func.run(
            bound, tensors.state, steps=12, callbacks=callbacks
        )
        assert len(graphs) == 1
        repeated, repeated_aux = dn.func.run(
            bound, tensors.state, steps=12, callbacks=callbacks
        )
        dn.func.run(
            bound,
            first,
            steps=12,
            callbacks=callbacks,
            callback_state=None if not record else auxiliary["callbacks"].state,
        )
        assert len(graphs) == 1
    _assert_tree_close(first, expected)
    _assert_tree_close(repeated, expected)
    _assert_tree_close(tensors.state, state_before)
    if record:
        _assert_tree_close(
            dict(auxiliary["callbacks"]), dict(expected_aux["callbacks"])
        )
        _assert_tree_close(
            dict(repeated_aux["callbacks"]), dict(expected_aux["callbacks"])
        )
