"""Direct contracts for pure full-field process preparation and replacements."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import ClearanceProcess, Material, MaterialProcess

pytestmark = pytest.mark.cpu
DTYPE = torch.float64
KEY = ("pool", "c")


def _process(cls, fields):
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
        process._bind_materials(materials.__getitem__)
        process.train()
        process.populate()
    return process, materials


def _oracle(c, dt, rate, target, method):
    if method == "explicit":
        return c - dt * rate * (c - target)
    if method == "implicit":
        return (c + dt * rate * target) / (1.0 + dt * rate)
    return target + (c - target) * torch.exp(-dt * rate)


@pytest.mark.parametrize(
    ("method", "canonical"),
    [
        ("exact", "exact"),
        ("exp", "exact"),
        ("exponential", "exact"),
        ("cnexp", "exact"),
        ("rush_larsen", "exact"),
        ("implicit", "implicit"),
        ("be", "implicit"),
        ("backward_euler", "implicit"),
        ("bwd_euler", "implicit"),
        ("explicit", "explicit"),
        ("forward_euler", "explicit"),
        ("euler", "explicit"),
    ],
)
def test_clearance_pure_update_and_imperative_adapter_preserve_method_aliases(
    method, canonical, monkeypatch
):
    class Clear(ClearanceProcess):
        ClearanceProcess.METHOD(method)
        ClearanceProcess.CLEAR("pool", field="c", rate=0.25, target=0.4)

    c = torch.tensor([[0.1, 0.7, 1.3], [0.2, 0.8, 1.4]], dtype=DTYPE)
    fields = {KEY: c}
    process, materials = _process(Clear, fields)
    versions = {name: value._version for name, value in process.named_buffers()}
    original_resolver = process._get_material

    def forbidden_resolver(*args, **kwargs):
        raise AssertionError("Pure process methods must not resolve bound Materials")

    monkeypatch.setattr(process, "_get_material", forbidden_resolver)
    prepared = process.prepare_materials(fields, c.new_tensor(0.125))
    updates = process.material_updates(fields, prepared)
    expected = _oracle(c, 0.125, c.new_tensor(0.25), c.new_tensor(0.4), canonical)
    torch.testing.assert_close(updates[KEY], expected)
    assert process.material_fields == (KEY,)
    assert all(torch.is_tensor(value) for entry in prepared for value in entry)
    assert fields[KEY] is c and materials["pool"]._buffers["c"] is c
    assert versions == {name: value._version for name, value in process.named_buffers()}

    monkeypatch.setattr(process, "_get_material", original_resolver)
    process.advance_materials(0.125)
    torch.testing.assert_close(materials["pool"]._buffers["c"], expected)
    torch.testing.assert_close(fields[KEY], c)


@pytest.mark.parametrize("method", ["exact", "implicit", "explicit"])
def test_tensor_literal_operands_are_registered_and_preserve_analytic_gradients(method):
    rate = torch.tensor(0.25, dtype=DTYPE, requires_grad=True)
    target = torch.tensor(0.4, dtype=DTYPE, requires_grad=True)

    class Clear(ClearanceProcess):
        ClearanceProcess.METHOD(method)
        ClearanceProcess.CLEAR("pool", field="c", rate=rate, target=target)

    c = torch.tensor([[0.1, 0.7, 1.3]], dtype=DTYPE, requires_grad=True)
    dt = c.new_tensor(0.125, requires_grad=True)
    process, materials = _process(Clear, {KEY: c})
    assert process._buffers["_clearance_literal_0_rate"] is rate
    assert process._buffers["_clearance_literal_0_target"] is target
    assert Clear._clearance_specs[0].rate is rate
    assert Clear._clearance_specs[0].target is target
    process.configure_process()
    assert process._buffers["_clearance_literal_0_rate"] is rate

    prepared = process.prepare_materials({KEY: c}, dt)
    result = process.material_updates({KEY: c}, prepared)[KEY]
    gradients = torch.autograd.grad(result.sum(), (c, rate, target, dt))
    if method == "exact":
        decay = torch.exp(-rate * dt)
        dc = torch.ones_like(c) * decay
        dr = -(c - target) * dt * decay
        dtarget = torch.ones_like(c) * (1.0 - decay)
        ddt = -(c - target) * rate * decay
    elif method == "implicit":
        denom = 1.0 + dt * rate
        dc = torch.ones_like(c) / denom
        dr = dt * (target - c) / denom.square()
        dtarget = torch.ones_like(c) * dt * rate / denom
        ddt = rate * (target - c) / denom.square()
    else:
        dc = torch.ones_like(c) * (1.0 - dt * rate)
        dr = -dt * (c - target)
        dtarget = torch.ones_like(c) * dt * rate
        ddt = -rate * (c - target)
    expected = (dc, dr.sum(), dtarget.sum(), ddt.sum())
    for actual, reference in zip(gradients, expected):
        torch.testing.assert_close(actual, reference, rtol=1.0e-12, atol=1.0e-13)
        assert torch.isfinite(actual).all() and actual.abs().max() > 0
    assert materials["pool"]._buffers["c"] is c


def test_named_operands_and_sequential_duplicate_specs_use_current_full_fields():
    class Clear(ClearanceProcess):
        ClearanceProcess.GLOBAL(rate=0.25)
        ClearanceProcess.RANGE(target=0.4)
        ClearanceProcess.BATCH(second_rate=0.5)
        ClearanceProcess.CLEAR("pool", field="c", rate="rate", target="target")
        ClearanceProcess.CLEAR("pool", field="c", rate="second_rate", target=0.1)

    c = torch.tensor([[0.1, 0.7, 1.3], [0.2, 0.8, 1.4]], dtype=DTYPE)
    process, materials = _process(Clear, {KEY: c})
    prepared = process.prepare_materials({KEY: c}, 0.125)
    expected = _oracle(c, 0.125, process.rate, process.target, "exact")
    expected = _oracle(expected, 0.125, process.second_rate, c.new_tensor(0.1), "exact")
    process.advance_materials(0.125, prepared=prepared)
    assert process.material_fields == (KEY,)
    torch.testing.assert_close(materials["pool"]._buffers["c"], expected)


@pytest.mark.parametrize("allow_negative", [False, True])
def test_clearance_rate_clamp_preserves_existing_policy(allow_negative):
    class Clear(ClearanceProcess):
        ClearanceProcess.METHOD("exact", allow_negative_rate=allow_negative)
        ClearanceProcess.CLEAR("pool", field="c", rate=-0.25, target=0.4)

    c = torch.tensor([[0.1, 0.7, 1.3]], dtype=DTYPE)
    process, materials = _process(Clear, {KEY: c})
    process.advance_materials(0.125)
    rate = c.new_tensor(-0.25 if allow_negative else 0.0)
    torch.testing.assert_close(
        materials["pool"]._buffers["c"],
        _oracle(c, 0.125, rate, c.new_tensor(0.4), "exact"),
    )


def test_failed_multifield_preparation_publishes_no_partial_updates():
    invalid_target = torch.ones(4, dtype=DTYPE)

    class Clear(ClearanceProcess):
        ClearanceProcess.CLEAR("pool", field="c", rate=0.25, target=0.4)
        ClearanceProcess.CLEAR("other", field="c", rate=0.25, target=invalid_target)

    fields = {
        KEY: torch.tensor([[0.1, 0.7, 1.3]], dtype=DTYPE),
        ("other", "c"): torch.tensor([[0.3, 0.5, 0.9]], dtype=DTYPE),
    }
    process, materials = _process(Clear, fields)
    with pytest.raises((RuntimeError, ValueError), match="shape|broadcast|size"):
        process.advance_materials(0.125)
    for (name, field), original in fields.items():
        assert materials[name]._buffers[field] is original


def test_failed_prepared_multifield_update_publishes_no_partial_updates():
    class Clear(ClearanceProcess):
        ClearanceProcess.CLEAR("pool", field="c", rate=0.25, target=0.4)
        ClearanceProcess.CLEAR("other", field="c", rate=0.5, target=0.2)

    fields = {
        KEY: torch.tensor([[0.1, 0.7, 1.3]], dtype=DTYPE),
        ("other", "c"): torch.tensor([[0.3, 0.5, 0.9]], dtype=DTYPE),
    }
    process, materials = _process(Clear, fields)
    prepared = list(process.prepare_materials(fields, 0.125))
    dt, rate, _target = prepared[1]
    prepared[1] = (dt, rate, torch.ones(4, dtype=DTYPE))
    with pytest.raises(RuntimeError, match="size"):
        process.advance_materials(0.125, prepared=tuple(prepared))
    for (name, field), original in fields.items():
        assert materials[name]._buffers[field] is original


@pytest.mark.parametrize("method", ["exact", "implicit", "explicit"])
def test_pure_material_protocol_supports_grad_and_independent_vmap_fields(
    method, monkeypatch
):
    class Clear(ClearanceProcess):
        ClearanceProcess.METHOD(method)
        ClearanceProcess.CLEAR("pool", field="c", rate=0.25, target=0.4)
        ClearanceProcess.CLEAR("other", field="c", rate=0.125, target=0.1)

    other_key = ("other", "c")
    stale = torch.full((1, 3), 9.0, dtype=DTYPE)
    process, materials = _process(Clear, {KEY: stale, other_key: stale + 2.0})

    def forbidden_resolver(*args, **kwargs):
        raise AssertionError(
            "Pure transform read a bound Material instead of its inputs"
        )

    monkeypatch.setattr(process, "_get_material", forbidden_resolver)

    def values(c, other, dt):
        fields = {KEY: c, other_key: other}
        return process.material_updates(fields, process.prepare_materials(fields, dt))

    def loss(c, other, dt):
        updates = values(c, other, dt)
        return updates[KEY].sum() + 2.0 * updates[other_key].sum()

    def reference_values(c, other, dt):
        return {
            KEY: _oracle(c, dt, c.new_tensor(0.25), c.new_tensor(0.4), method),
            other_key: _oracle(
                other, dt, other.new_tensor(0.125), other.new_tensor(0.1), method
            ),
        }

    def reference_loss(c, other, dt):
        updates = reference_values(c, other, dt)
        return updates[KEY].sum() + 2.0 * updates[other_key].sum()

    c = torch.tensor(
        [[[0.1, 0.7, 1.3]], [[0.2, 0.8, 1.4]], [[0.3, 0.9, 1.5]]], dtype=DTYPE
    )
    other = c * 1.5 + 0.2
    dt = c.new_tensor([0.05, 0.125, 0.25])
    original_versions = tuple(value._version for value in (c, other, dt))
    actual_values = torch.vmap(values)(c, other, dt)
    expected_values = torch.vmap(reference_values)(c, other, dt)
    for key in (KEY, other_key):
        torch.testing.assert_close(actual_values[key], expected_values[key])
    gradient = torch.func.grad(loss, argnums=(0, 1, 2))
    reference_gradient = torch.func.grad(reference_loss, argnums=(0, 1, 2))
    actual_gradients = torch.vmap(gradient)(c, other, dt)
    expected_gradients = tuple(
        torch.stack(
            [
                reference_gradient(c[lane], other[lane], dt[lane])[index]
                for lane in range(3)
            ]
        )
        for index in range(3)
    )
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-13)
        assert torch.isfinite(actual).all() and actual.abs().max() > 0
    assert original_versions == tuple(value._version for value in (c, other, dt))
    assert materials["pool"]._buffers["c"] is stale


def test_custom_pure_family_inherits_the_shared_imperative_adapter():
    class PureProcess(MaterialProcess):
        @property
        def material_fields(self):
            return (KEY, ("other", "c"))

        def prepare_materials(self, fields, dt):
            return fields[KEY].new_tensor(dt), fields[KEY].new_tensor(0.5)

        def material_updates(self, fields, prepared):
            dt, rate = prepared
            transfer = dt * rate * fields[KEY]
            return {
                KEY: fields[KEY] - transfer,
                ("other", "c"): fields[("other", "c")] + transfer,
            }

    c = torch.tensor([[0.1, 0.7, 1.3]], dtype=DTYPE)
    other = c * 0.25
    fields = {KEY: c, ("other", "c"): other}
    process, materials = _process(PureProcess, fields)
    assert PureProcess.advance_materials is MaterialProcess.advance_materials
    assert ClearanceProcess.advance_materials is MaterialProcess.advance_materials
    process.advance_materials(0.125)
    first = materials["pool"]._buffers["c"]
    second = materials["other"]._buffers["c"]
    torch.testing.assert_close(first, c * (1.0 - 0.125 * 0.5))
    torch.testing.assert_close(first + second, c + other)
    prepared = (c.new_tensor(0.25), c.new_tensor(0.5))
    process.advance_materials(999.0, prepared=prepared)
    torch.testing.assert_close(
        materials["pool"]._buffers["c"], first * (1.0 - 0.25 * 0.5)
    )
    torch.testing.assert_close(
        materials["pool"]._buffers["c"] + materials["other"]._buffers["c"], c + other
    )


def test_material_process_pure_protocol_is_optional_for_legacy_processes():
    class Legacy(MaterialProcess):
        def advance_materials(self, dt):
            material = self._get_material("pool")
            material._buffers["c"] = material._buffers["c"] + dt

    c = torch.tensor([[0.1, 0.7, 1.3]], dtype=DTYPE)
    process, materials = _process(Legacy, {KEY: c})
    process.advance_materials(0.125)
    torch.testing.assert_close(materials["pool"]._buffers["c"], c + 0.125)
    assert process._functional_material_family is None
    with pytest.raises(NotImplementedError, match="pure material preparation"):
        process.prepare_materials({KEY: c}, 0.125)
    with pytest.raises(NotImplementedError, match="pure material updates"):
        process.material_updates({KEY: c}, ())
