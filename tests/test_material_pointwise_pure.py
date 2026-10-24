"""Pure pointwise MaterialProcess contracts, independent of functional runners."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import ClampProcess, ExchangeProcess, Material

pytestmark = pytest.mark.cpu
DTYPE = torch.float64
KEY = ("pool", "c")


@pytest.fixture(params=[1, 4], autouse=True)
def _cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    yield request.param
    torch.set_num_threads(previous)


def _process(cls, fields, *, population=None, diameters=None):
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
            diameters=torch.ones_like(first) if diameters is None else diameters,
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


def _exchange_reference(a, b, dt, coefficient, Va, Vb, method, *, conductance=False):
    """Independent flux form rather than the implementation's weighted average."""
    g = coefficient if conductance else coefficient * Va
    inverse_mass = Va.reciprocal() + Vb.reciprocal()
    if method == "explicit":
        transfer = dt * g * (a - b)
    elif method == "implicit":
        transfer = dt * g * (a - b) / (1.0 + dt * g * inverse_mass)
    else:
        transfer = (a - b) * (-torch.expm1(-dt * g * inverse_mass)) / inverse_mass
    return a - transfer / Va, b + transfer / Vb


@pytest.mark.parametrize(
    ("method", "canonical"),
    [
        ("exact", "exact"),
        ("exp", "exact"),
        ("exponential", "exact"),
        ("closed_form", "exact"),
        ("analytic", "exact"),
        ("implicit", "implicit"),
        ("be", "implicit"),
        ("backward_euler", "implicit"),
        ("bwd_euler", "implicit"),
        ("explicit", "explicit"),
        ("forward_euler", "explicit"),
        ("euler", "explicit"),
    ],
)
@pytest.mark.parametrize("conductance", [False, True])
def test_exchange_pure_values_gradients_and_conservation_preserve_methods(
    method, canonical, conductance, monkeypatch
):
    coefficient = torch.tensor(0.25, dtype=DTYPE, requires_grad=True)
    Va = torch.tensor(1.2, dtype=DTYPE, requires_grad=True)
    Vb = torch.tensor(2.3, dtype=DTYPE, requires_grad=True)
    kinetic = {"conductance" if conductance else "rate": coefficient}

    class Exchange(ExchangeProcess):
        ExchangeProcess.METHOD(method)
        ExchangeProcess.EXCHANGE("a.c", "b.c", volume_a=Va, volume_b=Vb, **kinetic)

    a = torch.tensor([[1.3, 2.1, 3.5]], dtype=DTYPE, requires_grad=True)
    b = torch.tensor([[0.1, 0.3, 0.8]], dtype=DTYPE, requires_grad=True)
    fields = {("a", "c"): a, ("b", "c"): b}
    dt = a.new_tensor(0.125, requires_grad=True)
    process, materials = _process(Exchange, fields)
    literal_name = "_exchange_literal_0_" + ("conductance" if conductance else "rate")
    assert process._buffers[literal_name] is coefficient
    assert process._buffers["_exchange_literal_0_volume_a"] is Va
    assert process._buffers["_exchange_literal_0_volume_b"] is Vb
    original_resolver = process._get_material
    versions = {name: value._version for name, value in process.named_buffers()}

    def forbidden_resolver(*args, **kwargs):
        raise AssertionError("Pure Exchange must not read bound Material objects")

    monkeypatch.setattr(process, "_get_material", forbidden_resolver)
    prepared = process.prepare_materials(fields, dt)
    updates = process.material_updates(fields, prepared)
    expected = _exchange_reference(
        a, b, dt, coefficient, Va, Vb, canonical, conductance=conductance
    )
    actual = updates[("a", "c")], updates[("b", "c")]
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=2.0e-15, atol=2.0e-15)
    torch.testing.assert_close(
        Va * actual[0] + Vb * actual[1], Va * a + Vb * b, rtol=2.0e-15, atol=2.0e-15
    )
    targets = (a, b, dt, coefficient, Va, Vb)
    gradients = torch.autograd.grad((actual[0] - 2.0 * actual[1]).sum(), targets)
    reference_gradients = torch.autograd.grad(
        (expected[0] - 2.0 * expected[1]).sum(), targets
    )
    for value, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(value, reference, rtol=2.0e-12, atol=2.0e-13)
        assert torch.isfinite(value).all() and torch.count_nonzero(value)
    assert process.material_fields == (("a", "c"), ("b", "c"))
    assert versions == {name: value._version for name, value in process.named_buffers()}
    assert materials["a"]._buffers["c"] is a and materials["b"]._buffers["c"] is b

    monkeypatch.setattr(process, "_get_material", original_resolver)
    process.advance_materials(dt)
    for name, reference in zip(("a", "b"), expected):
        torch.testing.assert_close(materials[name]._buffers["c"], reference)


def test_exchange_sequential_overlapping_pairs_and_named_operands_are_conservative():
    class Exchange(ExchangeProcess):
        ExchangeProcess.GLOBAL(rate=0.25, volume_a=1.0)
        ExchangeProcess.RANGE(conductance=0.15, volume_c=3.0)
        ExchangeProcess.BATCH(second_rate=0.4, volume_b=2.0)
        ExchangeProcess.EXCHANGE(
            "a.c", "b.c", rate="rate", volume_a="volume_a", volume_b="volume_b"
        )
        ExchangeProcess.EXCHANGE(
            "b.c",
            "c.c",
            conductance="conductance",
            volume_a="volume_b",
            volume_b="volume_c",
        )
        ExchangeProcess.EXCHANGE(
            "a.c", "b.c", rate="second_rate", volume_a="volume_a", volume_b="volume_b"
        )

    a = torch.tensor([[1.1, 2.2, 3.3], [1.2, 2.3, 3.4]], dtype=DTYPE)
    b, c = a * 0.2, a * 0.1
    fields = {("a", "c"): a, ("b", "c"): b, ("c", "c"): c}
    process, materials = _process(Exchange, fields)
    expected_a, expected_b = _exchange_reference(
        a, b, 0.125, process.rate, process.volume_a, process.volume_b, "exact"
    )
    expected_b, expected_c = _exchange_reference(
        expected_b,
        c,
        0.125,
        process.conductance,
        process.volume_b,
        process.volume_c,
        "exact",
        conductance=True,
    )
    expected_a, expected_b = _exchange_reference(
        expected_a,
        expected_b,
        0.125,
        process.second_rate,
        process.volume_a,
        process.volume_b,
        "exact",
    )
    process.advance_materials(0.125)
    for name, reference in zip(("a", "b", "c"), (expected_a, expected_b, expected_c)):
        torch.testing.assert_close(materials[name]._buffers["c"], reference)
    total = sum(
        weight * materials[name]._buffers["c"]
        for name, weight in (("a", 1.0), ("b", 2.0), ("c", 3.0))
    )
    torch.testing.assert_close(total, a + 2.0 * b + 3.0 * c, rtol=2.0e-15, atol=2.0e-15)
    assert process.material_fields == (("a", "c"), ("b", "c"), ("c", "c"))


@pytest.mark.parametrize("method", ["exact", "implicit", "explicit"])
def test_exchange_analytic_geometry_is_live_differentiable_and_repreparable(method):
    class Exchange(ExchangeProcess):
        ExchangeProcess.METHOD(method, require_volumes=True)
        ExchangeProcess.EXCHANGE(
            "a.c",
            "b.c",
            conductance=0.25,
            volume_a="intracellular",
            volume_b="membrane",
        )

    diam = torch.tensor([[1.1, 1.7, 2.2]], dtype=DTYPE, requires_grad=True)
    dx = torch.tensor([[1.2, 2.3, 3.4]], dtype=DTYPE, requires_grad=True)
    a = diam.new_tensor([[1.3, 2.1, 3.5]])
    b = diam.new_tensor([[0.1, 0.3, 0.8]])
    fields = {("a", "c"): a, ("b", "c"): b}
    population = SimpleNamespace(diam=diam, dx=dx, graph=None, shape=tuple(a.shape))
    process, _materials = _process(
        Exchange, fields, population=population, diameters=diam
    )
    assert process._exchange_analytic_area
    assert process._buffers["_mp_dx"] is dx
    assert "_mp_area_cm2" not in process._buffers

    def evaluate(diam_value, dx_value, scale):
        process._buffers["diam"] = diam_value * scale
        process._buffers["_mp_dx"] = dx_value
        updates = process.material_updates(
            fields, process.prepare_materials(fields, 0.125)
        )
        return (updates[("a", "c")] - 2.0 * updates[("b", "c")]).sum()

    for scale in (1.0, 1.3):
        actual_loss = evaluate(diam, dx, scale)
        actual = torch.autograd.grad(actual_loss, (diam, dx))
        Va = torch.pi * (0.5 * diam * scale).square() * dx
        Vb = torch.pi * diam * scale * dx
        expected = _exchange_reference(
            a, b, 0.125, a.new_tensor(0.25), Va, Vb, method, conductance=True
        )
        reference = torch.autograd.grad(
            (expected[0] - 2.0 * expected[1]).sum(), (diam, dx)
        )
        for value, target in zip(actual, reference):
            torch.testing.assert_close(value, target, rtol=2.0e-11, atol=2.0e-13)
            assert torch.isfinite(value).all() and torch.count_nonzero(value)
        epsilon = 1.0e-5
        finite_differences = (
            (
                evaluate(diam.detach() + epsilon, dx.detach(), scale)
                - evaluate(diam.detach() - epsilon, dx.detach(), scale)
            )
            / (2.0 * epsilon),
            (
                evaluate(diam.detach(), dx.detach() + epsilon, scale)
                - evaluate(diam.detach(), dx.detach() - epsilon, scale)
            )
            / (2.0 * epsilon),
        )
        for gradient, finite_difference in zip(actual, finite_differences):
            torch.testing.assert_close(
                gradient.sum(), finite_difference, rtol=2.0e-6, atol=2.0e-8
            )


def test_exchange_explicit_geometry_preserves_raw_sources_and_rebinds_without_stale_buffers():
    class Exchange(ExchangeProcess):
        ExchangeProcess.METHOD("exact", require_volumes=True)
        ExchangeProcess.EXCHANGE(
            "a.c",
            "b.c",
            conductance=0.25,
            volume_a="intracellular",
            volume_b="membrane",
        )

    a = torch.tensor([[1.3, 2.1, 3.5]], dtype=DTYPE)
    b = a * 0.2
    fields = {("a", "c"): a, ("b", "c"): b}
    volume = torch.tensor([[1.2, 2.3, 3.4]], dtype=DTYPE, requires_grad=True)
    area = torch.tensor([[1.3e-8, 1.7e-8, 2.4e-8]], dtype=DTYPE, requires_grad=True)
    population = SimpleNamespace(
        volume_i=volume, area=area, graph=object(), shape=tuple(a.shape)
    )
    process, _materials = _process(Exchange, fields, population=population)
    assert process._buffers["_mp_volume_i"] is volume
    assert process._buffers["_mp_area_cm2"] is area
    assert not process._exchange_analytic_area
    assert process._exchange_geometry_sources == (
        ("_mp_volume_i", "volume_i"),
        ("_mp_area_cm2", "canonical_area_cm2"),
    )
    for _repeat in range(2):
        updates = process.material_updates(
            fields, process.prepare_materials(fields, 0.125)
        )
        loss = (updates[("a", "c")] - 2.0 * updates[("b", "c")]).sum()
        gradients = torch.autograd.grad(loss, (volume, area))
        assert all(
            torch.isfinite(value).all() and torch.count_nonzero(value)
            for value in gradients
        )
    replacement = SimpleNamespace(volume_o=torch.ones_like(a), shape=tuple(a.shape))
    process.configure_process(replacement)
    assert (
        "_mp_volume_i" not in process._buffers
        and "_mp_area_cm2" not in process._buffers
    )
    assert process._exchange_geometry_sources == (("_mp_volume_o", "volume_o"),)


@pytest.mark.parametrize("method", ["exact", "implicit", "explicit"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("conductance", [False, True])
def test_exchange_zero_volume_pairs_are_identities_with_finite_gradients(
    method, dtype, conductance
):
    coefficient = torch.tensor(0.25, dtype=dtype, requires_grad=True)
    Va = torch.tensor([[0.0, 1.0, 1.3, 0.0]], dtype=dtype, requires_grad=True)
    Vb = torch.tensor([[1.0, 0.0, 2.3, 0.0]], dtype=dtype, requires_grad=True)
    kinetic = {"conductance" if conductance else "rate": coefficient}

    class Exchange(ExchangeProcess):
        ExchangeProcess.METHOD(method)
        ExchangeProcess.EXCHANGE("a.c", "b.c", volume_a=Va, volume_b=Vb, **kinetic)

    a = torch.tensor([[1.3, 2.1, 3.5, 4.4]], dtype=dtype, requires_grad=True)
    b = torch.tensor([[0.1, 0.3, 0.8, 1.1]], dtype=dtype, requires_grad=True)
    dt = a.new_tensor(0.125, requires_grad=True)
    fields = {("a", "c"): a, ("b", "c"): b}
    process, _materials = _process(Exchange, fields)
    updates = process.material_updates(fields, process.prepare_materials(fields, dt))
    active = (Va > 0.0) & (Vb > 0.0)
    inactive = ~active
    for key, original in fields.items():
        torch.testing.assert_close(
            updates[key][inactive], original[inactive], rtol=0.0, atol=0.0
        )
    targets = (a, b, dt, coefficient, Va, Vb)
    loss = (updates[("a", "c")] + 2.0 * updates[("b", "c")]).sum()
    gradients = torch.autograd.grad(loss, targets, retain_graph=True)
    assert all(torch.isfinite(value).all() for value in gradients)
    for value in gradients[-2:]:
        torch.testing.assert_close(
            value[inactive], torch.zeros_like(value[inactive]), rtol=0.0, atol=0.0
        )
    inactive_loss = (
        updates[("a", "c")][inactive] + 2.0 * updates[("b", "c")][inactive]
    ).sum()
    inactive_gradients = torch.autograd.grad(inactive_loss, targets)
    expected = (
        inactive.to(dtype),
        2.0 * inactive.to(dtype),
        torch.zeros_like(dt),
        torch.zeros_like(coefficient),
        torch.zeros_like(Va),
        torch.zeros_like(Vb),
    )
    for value, target in zip(inactive_gradients, expected):
        torch.testing.assert_close(value, target, rtol=0.0, atol=0.0)
    # The active-set boundary is not differentiable in volume. Check volume
    # finite differences only on the positive active entries elsewhere.


def test_exchange_failures_never_publish_partial_fields():
    class Exchange(ExchangeProcess):
        ExchangeProcess.EXCHANGE("a.c", "b.c", rate=0.25, volume_a=1.0, volume_b=2.0)
        ExchangeProcess.EXCHANGE("c.c", "d.c", rate=0.3, volume_a=1.0, volume_b=2.0)

    fields = {
        (name, "c"): torch.full((1, 3), value, dtype=DTYPE)
        for name, value in (("a", 1.3), ("b", 0.1), ("c", 2.1), ("d", 0.3))
    }
    process, materials = _process(Exchange, fields)
    prepared = list(process.prepare_materials(fields, 0.125))
    prepared[1] = (*prepared[1][:-1], torch.ones(4, dtype=DTYPE))
    with pytest.raises(RuntimeError, match="size"):
        process.advance_materials(0.125, prepared=tuple(prepared))
    for (name, field), original in fields.items():
        assert materials[name]._buffers[field] is original

    class Invalid(ExchangeProcess):
        ExchangeProcess.EXCHANGE("a.c", "b.c", rate=0.25, volume_a=1.0, volume_b=2.0)
        ExchangeProcess.EXCHANGE(
            "c.c", "d.c", rate="missing_rate", volume_a=1.0, volume_b=2.0
        )

    process, materials = _process(Invalid, fields)
    with pytest.raises(AttributeError, match="missing_rate"):
        process.advance_materials(0.125)
    for (name, field), original in fields.items():
        assert materials[name]._buffers[field] is original


@pytest.mark.parametrize("allow_negative", [False, True])
def test_exchange_negative_rate_policy_is_preserved(allow_negative):
    class Exchange(ExchangeProcess):
        ExchangeProcess.METHOD("exact", allow_negative_rate=allow_negative)
        ExchangeProcess.EXCHANGE("a.c", "b.c", rate=-0.25, volume_a=1.0, volume_b=2.0)

    a = torch.tensor([[1.3, 2.1, 3.5]], dtype=DTYPE)
    b = a * 0.1
    fields = {("a", "c"): a, ("b", "c"): b}
    process, materials = _process(Exchange, fields)
    process.advance_materials(0.125)
    expected = _exchange_reference(
        a,
        b,
        0.125,
        a.new_tensor(-0.25 if allow_negative else 0.0),
        a.new_tensor(1.0),
        a.new_tensor(2.0),
        "exact",
    )
    for name, target in zip(("a", "b"), expected):
        torch.testing.assert_close(materials[name]._buffers["c"], target)


@pytest.mark.parametrize("method", ["exact", "implicit", "explicit"])
@pytest.mark.parametrize("batch_size", [0, 3])
def test_pure_exchange_supports_grad_and_empty_vmap_without_bound_field_reads(
    method, batch_size, monkeypatch
):
    class Exchange(ExchangeProcess):
        ExchangeProcess.METHOD(method)
        ExchangeProcess.EXCHANGE("a.c", "b.c", rate=0.25, volume_a=1.2, volume_b=2.3)

    stale = torch.full((1, 3), 9.0, dtype=DTYPE)
    process, materials = _process(
        Exchange, {("a", "c"): stale, ("b", "c"): stale + 2.0}
    )

    def forbidden_resolver(*args, **kwargs):
        raise AssertionError(
            "Pure Exchange read a bound Material instead of its inputs"
        )

    monkeypatch.setattr(process, "_get_material", forbidden_resolver)

    def values(a, b, dt):
        fields = {("a", "c"): a, ("b", "c"): b}
        prepared = process.prepare_materials(fields, dt)
        # The execution adapter owns normalization for PyTorch's empty-vmap
        # scalar-operand limitation. Do not specialize the mechanism math.
        prepared = torch.utils._pytree.tree_map(
            lambda value: value.reshape(
                *((1,) * max(0, a.ndim - value.ndim)), *value.shape
            ),
            prepared,
        )
        updates = process.material_updates(fields, prepared)
        return updates[("a", "c")], updates[("b", "c")]

    def loss(a, b, dt):
        first, second = values(a, b, dt)
        return (first - 2.0 * second).sum()

    def reference_loss(a, b, dt):
        first, second = _exchange_reference(
            a,
            b,
            dt.reshape(1, 1),
            a.new_full((1, 1), 0.25),
            a.new_full((1, 1), 1.2),
            a.new_full((1, 1), 2.3),
            method,
        )
        return (first - 2.0 * second).sum()

    a = torch.arange(batch_size * 3, dtype=DTYPE).reshape(batch_size, 1, 3) / 10.0 + 1.0
    b = a * 0.2
    dt = torch.linspace(0.05, 0.2, batch_size, dtype=DTYPE)
    versions = tuple(value._version for value in (a, b, dt))
    actual = torch.vmap(values)(a, b, dt)
    expected = _exchange_reference(
        a,
        b,
        dt.reshape(batch_size, 1, 1),
        a.new_tensor(0.25),
        a.new_tensor(1.2),
        a.new_tensor(2.3),
        method,
    )
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=2.0e-15, atol=2.0e-15)
    gradients = torch.vmap(torch.func.grad(loss, argnums=(0, 1, 2)))(a, b, dt)
    reference_gradients = torch.vmap(
        torch.func.grad(reference_loss, argnums=(0, 1, 2))
    )(a, b, dt)
    for value, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(value, reference, rtol=2.0e-12, atol=2.0e-13)
        assert torch.isfinite(value).all()
    assert versions == tuple(value._version for value in (a, b, dt))
    assert materials["a"]._buffers["c"] is stale


@pytest.mark.parametrize("mode", ["set", "min", "max", "range"])
@pytest.mark.parametrize("batch_size", [0, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pure_clamp_literals_preserve_empty_vmap_lanes_and_field_gradients(
    mode, batch_size, dtype, monkeypatch
):
    class Clamp(ClampProcess):
        ClampProcess.CLAMP(
            "pool",
            field="c",
            value=(1.2, 1.8) if mode == "range" else 1.2,
            mode=mode,
        )

    stale = torch.full((1, 3), 9.0, dtype=dtype)
    process, materials = _process(Clamp, {KEY: stale})

    def forbidden_resolver(*args, **kwargs):
        raise AssertionError("Pure Clamp read a bound Material instead of its inputs")

    monkeypatch.setattr(process, "_get_material", forbidden_resolver)

    def values(c):
        fields = {KEY: c}
        prepared = process.prepare_materials(fields, c.new_tensor(0.125))
        for operand in torch.utils._pytree.tree_leaves(prepared):
            assert operand.shape == ()
            assert operand.dtype == c.dtype and operand.device == c.device
            assert not operand.requires_grad
        return process.material_updates(fields, prepared)[KEY]

    def reference(c):
        if mode == "set":
            return torch.full_like(c, 1.2)
        if mode == "min":
            return torch.clamp_min(c, 1.2)
        if mode == "max":
            return torch.clamp_max(c, 1.2)
        return torch.clamp(c, 1.2, 1.8)

    c = torch.linspace(-0.7, 2.5, batch_size * 3, dtype=dtype).reshape(batch_size, 1, 3)
    version = c._version
    actual = torch.vmap(values)(c)
    expected = torch.vmap(reference)(c)
    assert actual.shape == (batch_size, 1, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.vmap(torch.func.grad(lambda value: values(value).sum()))(c)
    oracle = torch.vmap(torch.func.grad(lambda value: reference(value).sum()))(c)
    assert gradients.shape == c.shape
    torch.testing.assert_close(gradients, oracle, rtol=0, atol=0)
    assert torch.isfinite(gradients).all()
    assert c._version == version
    assert materials["pool"]._buffers["c"] is stale
