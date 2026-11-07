"""Shared scan boundaries preserve optional carry and input leaves."""

import pytest
import torch
from torch.utils import _pytree as pytree

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._carry_packing import _PackedCarryLayout
from dendra.func._hop_inputs import (
    _isolate_dense_lifted_inputs,
    _LiftedViewIsolation,
)
from dendra.func._scan_tree import _CompiledTensorTreeScan

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("parameter_first", [True, False])
def test_raw_scan_recaptures_parameter_class_policy_and_gradients(parameter_first):
    def body(parameters, prepared, state, carries, sample):
        scale = parameters["scale"]
        gain = 2.0 if isinstance(scale, torch.nn.Parameter) else 11.0
        value = state["v"] + gain * scale
        return ({"v": value}, carries), {"v": value}

    engine = _CompiledTensorTreeScan(
        body,
        steps=2,
        compile_options={"backend": "aot_eager", "fullgraph": True},
        device="cpu",
    )
    parameter = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    restored = parameter.detach().clone().requires_grad_()
    values = (parameter, restored) if parameter_first else (restored, parameter)
    with torch_compiler_warning_context():
        for scale in values:
            initial = torch.tensor([-0.2, 0.3], dtype=torch.float64, requires_grad=True)
            final, _, emitted = engine({"scale": scale}, {}, {"v": initial}, {}, None)
            gain = 2.0 if isinstance(scale, torch.nn.Parameter) else 11.0
            expected = initial + torch.arange(1, 3, dtype=torch.float64)[:, None] * (
                gain * scale
            )
            torch.testing.assert_close(final["v"], expected[-1])
            torch.testing.assert_close(emitted["v"], expected)
            actual_gradients = torch.autograd.grad(
                final["v"].sum() + emitted["v"].sum(), (scale, initial)
            )
            reference_gradients = torch.autograd.grad(
                expected[-1].sum() + expected.sum(), (scale, initial)
            )
            torch.testing.assert_close(actual_gradients, reference_gradients)
    assert len(engine._kernels) == 2


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


def test_shared_scan_preserves_trainable_leaf_and_nonleaf_parameter_metadata():
    def body(parameters, prepared, state, carries, sample):
        scale = parameters["scale"]
        gain = 2.0 if scale.requires_grad else 11.0
        gain += 3.0 if scale.is_leaf else 7.0
        value = state["v"] + scale * gain + prepared["bias"]
        return ({"v": value}, carries), {"v": value}

    engine = _CompiledTensorTreeScan(
        body,
        steps=3,
        compile_options={"backend": "aot_eager", "fullgraph": True},
        device="cpu",
    )
    for leaf in (True, False):
        upstream = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        scale = upstream if leaf else upstream * 1.25
        initial = torch.tensor([-0.3, 0.2], dtype=torch.float64, requires_grad=True)
        prepared = {"bias": torch.tensor(0.125, dtype=torch.float64)}
        with torch_compiler_warning_context():
            final, _, emitted = engine(
                {"scale": scale}, prepared, {"v": initial}, {}, None
            )
            gain = 5.0 if leaf else 9.0
            frames = initial + torch.arange(1, 4, dtype=torch.float64)[:, None] * (
                scale * gain + prepared["bias"]
            )
            torch.testing.assert_close(final["v"], frames[-1])
            torch.testing.assert_close(emitted["v"], frames)
            actual = torch.autograd.grad(
                final["v"].square().sum() + emitted["v"].square().sum(),
                (upstream, initial),
                retain_graph=True,
            )
            expected = torch.autograd.grad(
                frames[-1].square().sum() + frames.square().sum(),
                (upstream, initial),
            )
        for gradient, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(gradient, reference)
        assert scale.requires_grad and scale.is_leaf is leaf


@pytest.mark.parametrize("view", (False, True))
@pytest.mark.parametrize("grad_enabled", (False, True))
def test_shared_scan_isolates_lifted_aliases_and_accumulates_shared_gradient(
    view, grad_enabled
):
    def body(parameters, prepared, state, carries, sample):
        value = (
            state["v"]
            + parameters["scale"] * prepared["left"]
            + prepared["right"].square()
            + sample["drive"]
        )
        return ({"v": value}, carries), {"v": value}

    scale = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    parameters = {"scale": scale}
    prepared = {
        name: scale.view_as(scale) if view else scale for name in ("left", "right")
    }
    tape = {"drive": scale.expand(3)}
    initial = torch.tensor([-0.3, 0.2], dtype=torch.float64)
    engine = _CompiledTensorTreeScan(
        body,
        steps=3,
        compile_options={"backend": "aot_eager", "fullgraph": True},
        device="cpu",
    )
    with torch.set_grad_enabled(grad_enabled), torch_compiler_warning_context():
        final, _, emitted = engine(parameters, prepared, {"v": initial}, {}, tape)
        frames = initial + torch.arange(1, 4, dtype=torch.float64)[:, None] * (
            2 * scale.square() + scale
        )
        torch.testing.assert_close(final["v"], frames[-1])
        torch.testing.assert_close(emitted["v"], frames)
        if grad_enabled:
            actual = torch.autograd.grad(final["v"].sum(), scale)[0]
            expected = torch.autograd.grad(frames[-1].sum(), scale)[0]
            torch.testing.assert_close(actual, expected)
    assert parameters["scale"] is scale and scale.is_leaf
    assert all(
        torch._C._is_alias_of(value, scale)
        for value in (*prepared.values(), tape["drive"])
    )


@pytest.mark.parametrize(
    "kind", ("slice", "transpose", "expanded", "overlap", "empty", "conj", "neg")
)
def test_lifted_input_isolation_preserves_view_layout_and_logical_vjp(kind):
    source = torch.nn.Parameter(torch.arange(12.0, dtype=torch.float64))
    if kind == "transpose":
        value = source.reshape(3, 4).T
    elif kind == "expanded":
        value = source[3:4].expand(3)
    elif kind == "overlap":
        value = source.as_strided((2, 3), (1, 1), 2)
    elif kind == "empty":
        value = source[5:5]
    elif kind == "conj":
        value = torch.complex(source, source)[3:9:2].conj()
    elif kind == "neg":
        value = torch._neg_view(source[3:9:2])
    else:
        value = source[3:9:2]
    isolate_views = _LiftedViewIsolation.from_usage((True, True))
    first, second = isolate_views((value, value))
    for copied in (first, second):
        torch.testing.assert_close(copied, value)
        assert copied.stride() == value.stride()
        assert copied.storage_offset() == value.storage_offset()
        assert copied.is_conj() == value.is_conj()
        assert copied.is_neg() == value.is_neg()
        assert not torch._C._is_alias_of(copied, value)
    assert not torch._C._is_alias_of(first, second)
    weights = torch.arange(value.numel(), dtype=value.real.dtype).reshape(value.shape)
    weights = (weights + 0.3).to(value.dtype)
    if value.is_complex():
        weights = weights * (1 + 0.5j)
    logical = torch.autograd.grad(first, value, weights, retain_graph=True)[0]
    torch.testing.assert_close(logical, weights)
    actual = torch.autograd.grad(first, source, weights, retain_graph=True)[0]
    expected = torch.autograd.grad(value, source, weights)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ("expanded", "overlap"))
@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_compiled_lifted_view_consumers_keep_logical_and_raw_vjps(kind, backend):
    source = torch.nn.Parameter(torch.arange(12.0, dtype=torch.float64))
    value = (
        source[3:4].expand(2, 3)
        if kind == "expanded"
        else source.as_strided((2, 3), (1, 1), 2)
    )
    weights = torch.arange(6.0, dtype=value.dtype).reshape(2, 3) + 0.3
    isolate_views = _LiftedViewIsolation.from_usage((True, True))

    @torch.compile(backend=backend, fullgraph=True)
    def objective(copies):
        first, second = _isolate_dense_lifted_inputs(copies)
        return (first * weights).sum() + second.sum()

    # The engines isolate authored views at invocation, then isolate dense
    # operands inside compilation. Each logical coordinate retains its own
    # cotangent even when the original view has overlapping storage.
    result = objective(isolate_views((value, value)))
    actual = torch.autograd.grad(result, value, retain_graph=True)[0]
    torch.testing.assert_close(actual, weights + 1)
    raw = torch.autograd.grad(result, source)[0]
    expected = torch.autograd.grad((value * (weights + 1)).sum(), source)[0]
    torch.testing.assert_close(raw, expected)


def test_lifted_input_isolation_keeps_nonuniform_strided_jvp():
    value = torch.arange(12.0, dtype=torch.float64)[3:9:2]
    tangent = torch.tensor([0.3, 0.7, 1.2], dtype=value.dtype)
    isolate_views = _LiftedViewIsolation.from_usage(True)

    def view(value):
        return value.as_strided((2,), value.stride(), value.storage_offset()).square()

    actual = torch.func.jvp(
        lambda value: view(isolate_views(value)), (value,), (tangent,)
    )
    expected = torch.func.jvp(view, (value,), (tangent,))
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference)


def test_lifted_input_usage_keeps_unused_views_and_splits_dense_isolation():
    source = torch.nn.Parameter(torch.arange(12.0, dtype=torch.float64))
    dense = torch.nn.Parameter(torch.tensor([0.2, 0.7], dtype=source.dtype))
    inputs = {
        "used": source[3:9:2],
        "unused": source[1:7:2],
        "dense": dense,
        "absent": None,
    }
    isolate_views = _LiftedViewIsolation.from_usage(
        {"used": True, "unused": False, "dense": True, "absent": False}
    )
    source_before, source_version = source.detach().clone(), source._version
    boundary = isolate_views(inputs)
    assert boundary["unused"] is inputs["unused"]
    assert boundary["dense"] is dense and boundary["absent"] is None
    assert not torch._C._is_alias_of(boundary["used"], inputs["used"])
    assert boundary["used"].stride() == inputs["used"].stride()
    assert boundary["used"].storage_offset() == inputs["used"].storage_offset()

    operands = _isolate_dense_lifted_inputs(boundary)
    assert operands["used"] is boundary["used"]
    assert operands["unused"] is inputs["unused"]
    assert operands["absent"] is None
    assert not torch._C._is_alias_of(operands["dense"], dense)
    objective = operands["used"].square().sum() + operands["dense"].sum()
    actual = torch.autograd.grad(
        objective,
        (source, dense, inputs["unused"]),
        allow_unused=True,
        retain_graph=True,
    )
    expected = torch.autograd.grad(
        inputs["used"].square().sum() + dense.sum(), (source, dense)
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert actual[2] is None
    torch.testing.assert_close(source, source_before)
    assert source._version == source_version


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_shared_scan_preserves_explicit_strided_views_and_shared_vjp(backend):
    def body(parameters, prepared, state, carries, sample):
        value = parameters["value"]
        selected = value.as_strided((2,), value.stride(), value.storage_offset())
        updated = state["value"] + selected.square().sum() + prepared["same"].sum()
        return ({"value": updated}, carries), {}

    source = torch.nn.Parameter(torch.arange(12.0, dtype=torch.float64))
    value = source[3:9:2]
    parameters, prepared = {"value": value}, {"same": value}
    initial = {"value": torch.tensor(0.0, dtype=value.dtype)}
    engine = _CompiledTensorTreeScan(
        body,
        steps=4,
        compile_options={"backend": backend, "fullgraph": True},
        device="cpu",
    )
    with torch_compiler_warning_context():
        final, _, _ = engine(parameters, prepared, initial, {}, None)
    expected = initial
    for _ in range(4):
        (expected, _), _ = body(parameters, prepared, expected, {}, None)
    torch.testing.assert_close(final["value"], expected["value"])
    actual = torch.autograd.grad(final["value"], source)[0]
    reference = torch.autograd.grad(expected["value"], source)[0]
    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("leaf", (False, True))
def test_shared_scan_keeps_weighted_overlapping_parameter_gradients(leaf):
    source = torch.nn.Parameter(torch.arange(12.0, dtype=torch.float64))
    value = source.as_strided((2, 3), (1, 1), 2)
    if leaf:
        value = torch.nn.Parameter(value)
    weights = torch.arange(6.0, dtype=value.dtype).reshape(2, 3) + 0.3

    def body(parameters, prepared, state, carries, sample):
        increment = (parameters["value"] * prepared["weights"]).sum()
        increment = increment + prepared["same"].sum()
        return ({"value": state["value"] + increment}, carries), {}

    engine = _CompiledTensorTreeScan(
        body,
        steps=4,
        compile_options={"backend": "aot_eager", "fullgraph": True},
        device="cpu",
    )
    with torch_compiler_warning_context():
        final, _, _ = engine(
            {"value": value},
            {"same": value, "weights": weights},
            {"value": value.new_zeros(())},
            {},
            None,
        )
    logical = torch.autograd.grad(final["value"], value, retain_graph=True)[0]
    torch.testing.assert_close(logical, 4 * (weights + 1))
    if not leaf:
        raw = torch.autograd.grad(final["value"], source)[0]
        expected = torch.autograd.grad(4 * (value * (weights + 1)).sum(), source)[0]
        torch.testing.assert_close(raw, expected)
