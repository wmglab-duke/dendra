"""Pure pointwise MaterialProcess contracts, independent of functional runners."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import ClampProcess, Material

pytestmark = pytest.mark.cpu
DTYPE = torch.float64
KEY = ("pool", "c")


@pytest.fixture(params=[1, 4], autouse=True)
def _cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    yield request.param
    torch.set_num_threads(previous)


def _process(cls, fields, *, population=None):
    first = next(iter(fields.values()))
    shape = tuple(first.shape)
    materials = {}
    for (name, field), value in fields.items():
        if name not in materials:
            material_fields = {
                other_field: other_value.detach().clone()
                for (other_name, other_field), other_value in fields.items()
                if other_name == name
            }
            materials[name] = Material(name, shape, fields=material_fields).to(
                device=first.device, dtype=first.dtype
            )
        materials[name]._buffers[field] = value
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        process = cls(
            name="process",
            celsius=first.new_tensor(34.0),
            diameters=torch.ones_like(first),
            shape=shape,
            shape_f=shape,
        ).to(device=first.device, dtype=first.dtype)
        process._bind_materials(materials.__getitem__, population=population)
        process.train()
        process.populate()
    return process, materials


@pytest.mark.parametrize(
    ("mode", "canonical"),
    [
        ("set", "set"),
        ("hard", "set"),
        ("assign", "set"),
        ("min", "min"),
        ("lower_bound", "min"),
        ("minimum", "min"),
        ("max", "max"),
        ("upper_bound", "max"),
        ("maximum", "max"),
        ("range", "range"),
        ("bounds", "range"),
        ("clip", "range"),
    ],
)
def test_clamp_pure_and_imperative_paths_preserve_aliases_and_bound_tie_gradients(
    mode, canonical, monkeypatch
):
    lower = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    upper = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    value = (
        (lower, upper)
        if canonical == "range"
        else lower
        if canonical == "min"
        else upper
    )

    class Clamp(ClampProcess):
        ClampProcess.METHOD(mode)
        ClampProcess.CLAMP("pool", field="c", value=value)

    c = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], dtype=DTYPE, requires_grad=True)
    fields = {KEY: c}
    process, materials = _process(Clamp, fields)
    original_resolver = process._get_material
    versions = {name: tensor._version for name, tensor in process.named_buffers()}

    def forbidden_resolver(*args, **kwargs):
        raise AssertionError("Pure process methods must not read bound Materials")

    monkeypatch.setattr(process, "_get_material", forbidden_resolver)
    prepared = process.prepare_materials(fields, c.new_tensor(0.125))
    result = process.material_updates(fields, prepared)[KEY]
    if canonical == "range":
        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=DTYPE)
        expected_gradients = (
            torch.tensor([[0.0, 0.5, 0.5, 0.0]], dtype=DTYPE),
            c.new_tensor(1.5),
            c.new_tensor(1.5),
        )
        targets = (c, lower, upper)
    elif canonical == "min":
        expected = torch.tensor([[0.0, 0.0, 1.0, 2.0]], dtype=DTYPE)
        expected_gradients = (
            torch.tensor([[0.0, 0.5, 1.0, 1.0]], dtype=DTYPE),
            c.new_tensor(1.5),
        )
        targets = (c, lower)
    elif canonical == "max":
        expected = torch.tensor([[-1.0, 0.0, 1.0, 1.0]], dtype=DTYPE)
        expected_gradients = (
            torch.tensor([[1.0, 1.0, 0.5, 0.0]], dtype=DTYPE),
            c.new_tensor(1.5),
        )
        targets = (c, upper)
    else:
        expected = torch.ones_like(c)
        expected_gradients = (torch.zeros_like(c), c.new_tensor(c.numel()))
        targets = (c, upper)
    torch.testing.assert_close(result, expected)
    gradients = torch.autograd.grad(
        result.sum(), targets, allow_unused=True, materialize_grads=True
    )
    for actual, reference in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    assert process.material_fields == (KEY,)
    assert all(
        torch.is_tensor(leaf) for leaf in torch.utils._pytree.tree_leaves(prepared)
    )
    assert versions == {
        name: tensor._version for name, tensor in process.named_buffers()
    }
    assert materials["pool"]._buffers["c"] is c

    monkeypatch.setattr(process, "_get_material", original_resolver)
    process.advance_materials(999.0, prepared=prepared)
    torch.testing.assert_close(materials["pool"]._buffers["c"], expected)
    assert fields[KEY] is c


def test_clamp_masked_range_literals_preserve_autograd_and_unselected_values():
    lower = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    upper = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    mask = torch.tensor([True, False, True, False])

    class Clamp(ClampProcess):
        ClampProcess.BOUNDS("pool", field="c", lower=lower, upper=upper, where=mask)

    c = torch.tensor([[-2.0, -1.0, 3.0, 4.0]], dtype=DTYPE, requires_grad=True)
    process, _materials = _process(Clamp, {KEY: c})
    assert process._buffers["_clamp_literal_0_lower"] is lower
    assert process._buffers["_clamp_literal_0_upper"] is upper
    assert Clamp._clamp_specs[0].value == (lower, upper)
    prepared = process.prepare_materials({KEY: c}, 0.125)
    result = process.material_updates({KEY: c}, prepared)[KEY]
    torch.testing.assert_close(result, c.new_tensor([[0.0, -1.0, 1.0, 4.0]]))
    gradients = torch.autograd.grad(result.sum(), (c, lower, upper))
    expected = (
        c.new_tensor([[0.0, 1.0, 0.0, 1.0]]),
        c.new_tensor(1.0),
        c.new_tensor(1.0),
    )
    for actual, reference in zip(gradients, expected):
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("where", ["terminal", "root", slice(1, 3), [0, 2], False])
def test_clamp_regions_and_sequential_duplicate_specs_use_current_full_fields(where):
    class Clamp(ClampProcess):
        ClampProcess.GLOBAL(lower=0.5)
        ClampProcess.RANGE(upper=2.0)
        ClampProcess.BATCH(prescribed=1.5)
        ClampProcess.MIN("pool", field="c", value="lower")
        ClampProcess.SET("pool", field="c", value="prescribed", where=where)
        ClampProcess.MAX("pool", field="c", value="upper")

    c = torch.tensor([[-1.0, 0.7, 3.0], [0.2, 0.8, 4.0]], dtype=DTYPE)
    process, materials = _process(
        Clamp, {KEY: c}, population=SimpleNamespace(graph=None, shape=tuple(c.shape))
    )
    prepared = process.prepare_materials({KEY: c}, 0.125)
    assert process.material_fields == (KEY,)
    expected = torch.maximum(c, process.lower)
    mask = torch.zeros_like(c, dtype=torch.bool)
    if where == "terminal":
        mask[..., 0] = True
        mask[..., -1] = True
    elif where == "root":
        mask[..., 0] = True
    elif isinstance(where, slice):
        mask[..., where] = True
    elif isinstance(where, list):
        mask[..., where] = True
    expected = torch.minimum(
        torch.where(mask, process.prescribed, expected), process.upper
    )
    process.advance_materials(0.125, prepared=prepared)
    torch.testing.assert_close(materials["pool"]._buffers["c"], expected)


@pytest.mark.parametrize("value", [0.0, (), ("low",), ("low", "high", "extra")])
def test_malformed_clamp_range_preserves_execution_failure_without_publication(value):
    class Clamp(ClampProcess):
        ClampProcess.SET("pool", field="c", value=9.0)
        ClampProcess.CLAMP("other", field="c", value=value, mode="range")

    fields = {
        KEY: torch.tensor([[0.1, 0.7, 1.3]], dtype=DTYPE),
        ("other", "c"): torch.tensor([[0.3, 0.5, 0.9]], dtype=DTYPE),
    }
    process, materials = _process(Clamp, fields)
    with pytest.raises(ValueError, match="requires value"):
        process.advance_materials(0.125)
    for (name, field), original in fields.items():
        assert materials[name]._buffers[field] is original


def test_failed_clamp_preparation_and_prepared_update_are_atomic():
    class Clamp(ClampProcess):
        ClampProcess.SET("pool", field="c", value=9.0)
        ClampProcess.MAX("other", field="c", value=2.0)

    fields = {
        KEY: torch.tensor([[0.1, 0.7, 1.3]], dtype=DTYPE),
        ("other", "c"): torch.tensor([[0.3, 0.5, 0.9]], dtype=DTYPE),
    }
    process, materials = _process(Clamp, fields)
    prepared = list(process.prepare_materials(fields, 0.125))
    prepared[1] = ((torch.ones(4, dtype=DTYPE),),)
    with pytest.raises(RuntimeError, match="size"):
        process.advance_materials(0.125, prepared=tuple(prepared))
    for (name, field), original in fields.items():
        assert materials[name]._buffers[field] is original

    class Invalid(ClampProcess):
        ClampProcess.SET("pool", field="c", value=9.0)
        ClampProcess.SET("other", field="c", value=torch.ones(4, dtype=DTYPE))

    process, materials = _process(Invalid, fields)
    with pytest.raises((ValueError, RuntimeError), match="shape|broadcast|size"):
        process.advance_materials(0.125)
    for (name, field), original in fields.items():
        assert materials[name]._buffers[field] is original


def test_pure_clamp_supports_grad_and_vmap_without_reading_bound_fields(monkeypatch):
    class Clamp(ClampProcess):
        ClampProcess.BOUNDS("pool", field="c", lower=0.0, upper=1.0, where=[0, 2])

    stale = torch.full((1, 3), 9.0, dtype=DTYPE)
    process, materials = _process(Clamp, {KEY: stale})

    def forbidden_resolver(*args, **kwargs):
        raise AssertionError("Pure transform read a bound Material")

    monkeypatch.setattr(process, "_get_material", forbidden_resolver)

    def values(c):
        fields = {KEY: c}
        return process.material_updates(
            fields, process.prepare_materials(fields, 0.125)
        )[KEY]

    def loss(c):
        return values(c).sum()

    c = torch.tensor([[[-1.0, 0.3, 1.7]], [[0.2, 0.4, 0.8]]], dtype=DTYPE)
    actual = torch.vmap(values)(c)
    expected = torch.stack([torch.tensor([[0.0, 0.3, 1.0]], dtype=DTYPE), c[1]])
    torch.testing.assert_close(actual, expected)
    gradients = torch.vmap(torch.func.grad(loss))(c)
    expected_gradients = c.new_tensor([[[0.0, 1.0, 0.0]], [[1.0, 1.0, 1.0]]])
    torch.testing.assert_close(gradients, expected_gradients)
    assert materials["pool"]._buffers["c"] is stale
