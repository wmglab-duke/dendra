"""Production contracts for Slice state inspection and mutation."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas

DTYPE = torch.float64
BATCH_SIZE = 3


def _population(*, n: int = 2, c: int = 4) -> dn.Population:
    population = dn.Population(N=n, C=c, dtype=DTYPE)
    values = torch.arange(n * c, dtype=DTYPE).reshape(n, c)
    population.v.copy_(values)
    population.register_buffer("spatial_state", values.clone())
    population.register_parameter(
        "spatial_parameter",
        torch.nn.Parameter(values.clone(), requires_grad=True),
    )
    return population


def _readers(
    region, name: str, *, mechanism: str | None = None
) -> tuple[Callable, ...]:
    if mechanism is None:
        return (
            lambda: region.inspect(name),
            lambda: region.get(name),
            lambda: getattr(region, name),
        )
    return (
        lambda: region.inspect(name, mechanism=mechanism),
        lambda: region.get(name, mechanism=mechanism),
        lambda: getattr(getattr(region.mech, mechanism), name),
    )


def _assert_snapshot_readers(
    readers: tuple[Callable, ...],
    expected: torch.Tensor,
    reread: Callable[[], torch.Tensor],
) -> None:
    def assert_equal(actual: torch.Tensor, wanted: torch.Tensor) -> None:
        if actual.is_floating_point() or actual.is_complex():
            torch.testing.assert_close(
                actual, wanted, rtol=0.0, atol=0.0, equal_nan=True
            )
        else:
            torch.testing.assert_close(actual, wanted)

    for read in readers:
        result = read()
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        assert_equal(result, expected)

        # Every public read path returns a snapshot. Mutating that snapshot must
        # never depend on whether the Slice used basic or advanced indexing.
        if result.dtype == torch.bool:
            result.logical_not_()
        else:
            result.fill_(12345)
        assert_equal(reread(), expected)


def _assign(region, name: str, value, *, style: str, mechanism: str | None = None):
    if style == "set":
        region.set(name, value, mechanism=mechanism)
        return
    if style != "attribute":  # pragma: no cover - test helper guard
        raise ValueError(style)
    target = region if mechanism is None else getattr(region.mech, mechanism)
    setattr(target, name, value)


@pytest.mark.parametrize(
    "key",
    [
        (slice(None), slice(1, 3)),
        (torch.tensor([0, 1]), torch.tensor([1, 3])),
    ],
)
@pytest.mark.parametrize("name", ["v", "spatial_state", "spatial_parameter"])
def test_slice_tensor_read_paths_are_equal_non_aliasing_snapshots(key, name):
    population = _population()
    region = population[key]
    expected = getattr(population, name)[key].detach().clone()

    _assert_snapshot_readers(
        _readers(region, name),
        expected,
        lambda: region.inspect(name),
    )


def test_slice_parameter_snapshot_preserves_autograd_connectivity():
    population = _population()
    snapshot = population[:, 1:3].spatial_parameter

    snapshot.square().sum().backward()

    expected = torch.zeros_like(population.spatial_parameter)
    expected[:, 1:3] = 2 * population.spatial_parameter.detach()[:, 1:3]
    assert torch.equal(population.spatial_parameter.grad, expected)


@pytest.mark.parametrize(
    "key",
    [
        (slice(None), slice(1, 3)),
        (torch.tensor([0, 1]), torch.tensor([1, 3])),
    ],
)
def test_slice_reads_derived_spatial_tensor_properties(key):
    population = _population()
    population.diam.copy_(torch.arange(8, dtype=DTYPE).reshape(2, 4) + 1)
    population.dx.copy_(torch.arange(8, dtype=DTYPE).reshape(2, 4) + 2)
    region = population[key]
    expected = population.area[key].clone()

    _assert_snapshot_readers(
        _readers(region, "area"),
        expected,
        lambda: region.inspect("area"),
    )


@pytest.mark.parametrize("name", ["t", "celsius", "nonspatial_metadata"])
def test_slice_rejects_global_scalar_and_nonspatial_model_tensors(name):
    population = _population()
    population.register_buffer("nonspatial_metadata", torch.arange(5, dtype=DTYPE))
    region = population[:, 1:3]
    target = getattr(population, name)
    before = target.clone()
    target_id = id(target)

    for read in _readers(region, name):
        with pytest.raises(ValueError):
            read()

    for style in ("set", "attribute"):
        with pytest.raises(ValueError):
            _assign(region, name, 7.0, style=style)
        assert id(getattr(population, name)) == target_id
        assert torch.equal(getattr(population, name), before)


@pytest.mark.parametrize("style", ["set", "attribute"])
@pytest.mark.parametrize("name", ["v", "spatial_state", "spatial_parameter"])
def test_slice_mutation_preserves_storage_identity_and_parameter_registration(
    style, name
):
    population = _population()
    region = population[:, 1:3]
    target = getattr(population, name)
    target_id = id(target)
    value = torch.tensor([[10.0, 11.0], [12.0, 13.0]], dtype=torch.float32)

    _assign(region, name, value, style=style)

    assert id(getattr(population, name)) == target_id
    assert getattr(population, name).dtype == DTYPE
    assert torch.equal(region.inspect(name), value.to(DTYPE))
    if name == "spatial_parameter":
        assert population._parameters[name] is target
        assert isinstance(target, torch.nn.Parameter)
        assert target.requires_grad
        assert target.is_leaf
    else:
        assert population._buffers[name] is target


@pytest.mark.parametrize("style", ["set", "attribute"])
@pytest.mark.parametrize("name", ["v", "spatial_state", "spatial_parameter"])
def test_slice_mutation_shape_errors_are_atomic(style, name):
    population = _population()
    region = population[:, 1:3]
    target = getattr(population, name)
    target_id = id(target)
    before = target.detach().clone()

    with pytest.raises((ValueError, RuntimeError)):
        _assign(region, name, torch.ones(3, 3, dtype=DTYPE), style=style)

    assert id(getattr(population, name)) == target_id
    assert torch.equal(getattr(population, name), before)
    if name == "spatial_parameter":
        assert population._parameters[name] is target
        assert target.requires_grad


def test_slice_unknown_attribute_assignment_fails_without_shadowing():
    population = _population()
    region = population[:, 1]

    with pytest.raises(AttributeError):
        region.volatge = torch.zeros(region.shape, dtype=DTYPE)

    assert "volatge" not in vars(region)
    assert not hasattr(population, "volatge")
    with pytest.raises(AttributeError):
        getattr(region, "volatge")


@pytest.mark.parametrize("style", ["set", "attribute"])
def test_batched_broadcast_storage_accepts_only_consistent_logical_writes(style):
    population = _population(n=2, c=3).batch(BATCH_SIZE)
    region = population[:, :, 1]
    target = population.cm
    target_id = id(target)
    target_shape = target.shape
    per_neuron = torch.tensor([4.0, 6.0], dtype=DTYPE)
    consistent = per_neuron.expand(BATCH_SIZE, -1).clone()

    assert target_shape == (1, 2, 3)
    assert region.cm.shape == (BATCH_SIZE, 2)
    _assign(region, "cm", consistent, style=style)

    assert id(population.cm) == target_id
    assert population.cm.shape == target_shape
    assert torch.equal(region.cm, consistent)
    assert torch.equal(population.cm[0, :, 1], per_neuron)

    before = population.cm.clone()
    conflicting = consistent + torch.arange(BATCH_SIZE, dtype=DTYPE).unsqueeze(-1)
    with pytest.raises(ValueError):
        _assign(region, "cm", conflicting, style=style)
    assert id(population.cm) == target_id
    assert torch.equal(population.cm, before)


def test_batched_derived_area_expands_to_the_logical_selection():
    population = _population(n=2, c=3).batch(BATCH_SIZE)
    region = population[:, :, 1]
    expected = population.area[:, 1].expand(BATCH_SIZE, -1).clone()

    _assert_snapshot_readers(
        _readers(region, "area"),
        expected,
        lambda: region.inspect("area"),
    )


def _mechanism_population(*, sparse: bool, lifecycle: str) -> dn.Population:
    population = dn.Population(N=2, C=3, dtype=DTYPE)
    insertion_region = population[:, 1:] if sparse else population
    insertion_region.insert(pas, g=0.1, e=-70.0)

    if lifecycle == "batch_before_build":
        population.batch(BATCH_SIZE)
        population.build()
    elif lifecycle == "batch_after_build":
        population.build()
        population.batch(BATCH_SIZE)
    elif lifecycle == "unbatched":
        population.build()
    else:  # pragma: no cover - test helper guard
        raise ValueError(lifecycle)
    return population


def _compartment_region(population: dn.Population, compartment: int):
    batch_axes = (slice(None),) * (len(population.shape) - 2)
    return population[batch_axes + (slice(None), compartment)]


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "lifecycle", ["unbatched", "batch_before_build", "batch_after_build"]
)
def test_mechanism_read_paths_agree_and_return_snapshots(sparse, lifecycle):
    population = _mechanism_population(sparse=sparse, lifecycle=lifecycle)
    region = _compartment_region(population, 2)
    stored = population.mech.pas.g.reshape(-1)[0].item()
    expected = torch.full(region.shape, stored, dtype=DTYPE)

    _assert_snapshot_readers(
        _readers(region, "g", mechanism="pas"),
        expected,
        lambda: region.inspect("g", mechanism="pas"),
    )


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "lifecycle", ["unbatched", "batch_before_build", "batch_after_build"]
)
@pytest.mark.parametrize("style", ["set", "attribute"])
def test_mechanism_mutation_paths_agree_and_preserve_dtype_identity(
    sparse, lifecycle, style
):
    population = _mechanism_population(sparse=sparse, lifecycle=lifecycle)
    mechanism = population.mech.pas
    region = _compartment_region(population, 2)
    target = mechanism.g
    target_id = id(target)
    per_neuron = torch.tensor([0.4, 0.6], dtype=torch.float32)
    value = per_neuron
    if lifecycle != "unbatched":
        value = per_neuron.expand(BATCH_SIZE, -1).clone()

    _assign(region, "g", value, style=style, mechanism="pas")

    assert id(mechanism.g) == target_id
    assert mechanism._buffers["g"] is target
    assert mechanism.g.dtype == DTYPE
    expected = value.to(DTYPE)
    for read in _readers(region, "g", mechanism="pas"):
        assert torch.equal(read(), expected)


@pytest.mark.parametrize(
    "lifecycle", ["unbatched", "batch_before_build", "batch_after_build"]
)
@pytest.mark.parametrize("style", ["set", "attribute"])
def test_sparse_mechanism_out_of_support_writes_fail_atomically(lifecycle, style):
    population = _mechanism_population(sparse=True, lifecycle=lifecycle)
    mechanism = population.mech.pas
    batch_axes = (slice(None),) * (len(population.shape) - 2)
    # This region mixes an unsupported compartment (0) with a supported one (1).
    region = population[batch_axes + (slice(None), slice(0, 2))]
    target = mechanism.g
    target_id = id(target)
    before = target.clone()

    with pytest.raises(ValueError):
        _assign(
            region,
            "g",
            torch.full(region.shape, 0.9, dtype=DTYPE),
            style=style,
            mechanism="pas",
        )

    assert id(mechanism.g) == target_id
    assert mechanism._buffers["g"] is target
    assert torch.equal(mechanism.g, before)


def test_sparse_floating_reads_mark_unsupported_locations_with_nan():
    population = _mechanism_population(sparse=True, lifecycle="unbatched")
    region = population[:, :2]
    stored = population.mech.pas.g.reshape(-1)[0]
    expected = torch.stack(
        (
            torch.full((2,), torch.nan, dtype=DTYPE),
            torch.full((2,), stored, dtype=DTYPE),
        ),
        dim=1,
    )

    for read in _readers(region, "g", mechanism="pas"):
        torch.testing.assert_close(read(), expected, equal_nan=True)


@pytest.mark.parametrize(
    "lifecycle", ["unbatched", "batch_before_build", "batch_after_build"]
)
def test_sparse_mechanism_boolean_fields_preserve_dtype(lifecycle):
    population = _mechanism_population(sparse=True, lifecycle=lifecycle)
    mechanism = population.mech.pas
    mechanism.register_buffer("enabled", torch.ones_like(mechanism.g, dtype=torch.bool))
    region = _compartment_region(population, 2)
    expected = torch.ones(region.shape, dtype=torch.bool)

    _assert_snapshot_readers(
        _readers(region, "enabled", mechanism="pas"),
        expected,
        lambda: region.inspect("enabled", mechanism="pas"),
    )

    enabled_id = id(mechanism.enabled)
    region.set("enabled", torch.zeros(region.shape, dtype=torch.bool), mechanism="pas")
    assert id(mechanism.enabled) == enabled_id
    assert mechanism.enabled.dtype == torch.bool
    assert torch.equal(
        region.inspect("enabled", mechanism="pas"),
        torch.zeros(region.shape, dtype=torch.bool),
    )


def test_sparse_nonfloating_read_rejects_unsupported_locations():
    population = _mechanism_population(sparse=True, lifecycle="unbatched")
    mechanism = population.mech.pas
    mechanism.register_buffer("enabled", torch.ones_like(mechanism.g, dtype=torch.bool))

    for read in _readers(population[:, :2], "enabled", mechanism="pas"):
        with pytest.raises(ValueError, match="no NaN"):
            read()


def test_duplicate_writes_require_equal_values_per_physical_location():
    population = _population(n=1, c=3)
    repeated = population[:, [1, 1]]
    target = population.v
    target_id = id(target)

    repeated.v = torch.tensor([[7.0, 7.0]], dtype=DTYPE)
    assert population.v[0, 1].item() == 7.0
    before = population.v.clone()

    with pytest.raises(ValueError, match="repeated physical locations"):
        repeated.v = torch.tensor([[8.0, 9.0]], dtype=DTYPE)

    assert id(population.v) == target_id
    assert torch.equal(population.v, before)


def test_retained_mechanism_slice_rebinds_after_force_rebuild():
    population = _mechanism_population(sparse=False, lifecycle="unbatched")
    retained = population[:, 1].mech.pas
    old_mechanism = retained.model

    population[:, 2].insert(pas.rename("pas_extra"), g=0.2, e=-60.0)
    population.build()

    assert population.mech.pas is not old_mechanism
    assert retained.model is population.mech.pas
    retained.g = 0.75
    assert torch.equal(population.mech.pas.g[:, 1], torch.full((2,), 0.75, dtype=DTYPE))


def test_insert_after_batch_projects_to_the_shared_structural_core():
    population = _population(n=2, c=3).batch(BATCH_SIZE)
    region = population[1, :, 1]

    region.insert(pas, g=0.2, e=-70.0)
    population.build()

    all_batches = population[:, :, 1]
    expected = torch.full(
        all_batches.shape,
        population.mech.pas.g.flatten()[0],
        dtype=population.mech.pas.g.dtype,
    )
    torch.testing.assert_close(
        all_batches.inspect("g", mechanism="pas"), expected, rtol=0.0, atol=0.0
    )


def test_parametrize_after_batch_uses_shared_parameter_storage():
    population = _population(n=2, c=3).batch(BATCH_SIZE)

    population[1, :, 1].parametrize("rhoa", 91.0, alias="selected")
    population.initialize()

    assert torch.equal(
        population[:, :, 1].rhoa,
        torch.full((BATCH_SIZE, 2), 91.0, dtype=DTYPE),
    )


def test_sparse_mechanism_parametrize_maps_to_local_storage_and_validates_support():
    population = _mechanism_population(sparse=True, lifecycle="unbatched")
    mechanism = population.mech.pas

    population[:, 2].mech.pas.parametrize("g", 0.3, alias="distal")
    with pytest.raises(ValueError, match="outside"):
        population[:, 0].mech.pas.parametrize("g", 0.4, alias="missing")
    population.initialize()

    expected = torch.tensor([[0.1, 0.3], [0.1, 0.3]], dtype=DTYPE)
    torch.testing.assert_close(mechanism.g, expected)


def test_physical_slice_rejects_ambiguous_colocated_slot_state():
    population = dn.Population(N=1, C=2, dtype=DTYPE)
    population[:, 1].insert(pas, copies=2, g=0.1, e=-70.0)
    population.build()

    with pytest.raises(ValueError, match="multiple independent slots"):
        _ = population[:, 1].mech.pas.g
    with pytest.raises(ValueError, match="multiple independent slots"):
        population[:, 1].set("g", 0.2, mechanism="pas")


@pytest.mark.parametrize("sparse", [False, True])
def test_slice_rejects_global_scalar_mechanism_fields(sparse):
    population = _mechanism_population(sparse=sparse, lifecycle="unbatched")
    mechanism = population.mech.pas
    region = population[:, 1]
    target = mechanism.dt
    target_id = id(target)
    before = target.clone()

    for read in _readers(region, "dt", mechanism="pas"):
        with pytest.raises(ValueError):
            read()

    for style in ("set", "attribute"):
        with pytest.raises(ValueError):
            _assign(region, "dt", 0.25, style=style, mechanism="pas")
        assert id(mechanism.dt) == target_id
        assert torch.equal(mechanism.dt, before)
