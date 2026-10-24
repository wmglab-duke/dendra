"""Shared fields and local bindings retain their checkpoint/carry contracts."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _population as func_api
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._ions import Ion
from dendra.models.mechanisms._materials import Material

pytestmark = pytest.mark.cpu
SHAPE = (1, 2)


class _LocalState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0.0 * x")


class _LocalMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_LocalState)


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64),
    ):
        yield


def _mechanism(name, local_names, *, state_locals=True):
    celsius = torch.full(SHAPE, 34.0, dtype=torch.float64)
    owner = _LocalMechanism(name, celsius, torch.ones_like(celsius), SHAPE, SHAPE)
    for index, local_name in enumerate(local_names):
        value = celsius.new_full(SHAPE, index + 1.0)
        owner.register_buffer(local_name, value)
        if state_locals:
            owner.DE["_LocalState"].register_buffer(local_name, value)
    return owner


def _handler():
    names = ("read", "write", "collision", "z", "a", "source")
    zeta, alpha = (_mechanism(name, names) for name in ("zeta", "alpha"))
    ion = Ion("na", SHAPE, einit=0, eadvance=0)
    material = Material("pool", SHAPE, fields={"late": 3.0, "early": 5.0})
    return MechanismHandler(
        zeta.celsius,
        torch.ones_like(zeta.celsius),
        {"zeta": zeta, "alpha": alpha},
        ions={"na": ion},
        materials={"pool": material},
        read_ion={"na": {"alpha": ["read", "collision"]}},
        write_ion_c={
            "na": {"zeta": ["write", "collision"], "alpha": ["collision", "write"]}
        },
        read_material={"pool": {"zeta": ["read"], "alpha": ["read"]}},
        write_material={"pool": {"alpha": ["z", "write"], "zeta": ["a", "collision"]}},
        source_material={
            "pool": {
                "zeta": {"late": "source", "early": "collision"},
                "alpha": {"late": "source", "early": "write"},
            }
        },
    )


def _snapshot(module):
    return tuple(
        (owner, name, value, value.detach().clone())
        for owner in module.modules()
        for name, value in owner._buffers.items()
        if value is not None
    )


def _assert_unchanged(snapshot):
    for owner, name, original, expected in snapshot:
        assert owner._buffers[name] is original
        torch.testing.assert_close(original, expected, rtol=0, atol=0)


def test_checkpoint_and_functional_local_roles_keep_order_priority_and_owners():
    handler = _handler()
    assert tuple(handler._material_local_buffer_names().items()) == (
        ("zeta", ("a", "collision", "read", "source")),
        ("alpha", ("read", "source", "write", "z")),
    )
    local_keys = (
        "alpha.read",
        "alpha.collision",
        "zeta.write",
        "zeta.collision",
        "alpha.write",
        "zeta.a",
        "zeta.read",
        "zeta.source",
        "alpha.source",
        "alpha.z",
    )
    global_paths = (
        ("na_ion.ina", ("ions", "na", "ina")),
        ("na_ion.ena", ("ions", "na", "ena")),
        ("na_ion.nai", ("ions", "na", "nai")),
        ("na_ion.nao", ("ions", "na", "nao")),
        ("pool_material.late", ("materials", "pool", "late")),
        ("pool_material.early", ("materials", "pool", "early")),
    )
    assert tuple(handler.mutable_state_dict()) == (
        "__dendra_mechanism_support_identity__",
        "zeta.x",
        "alpha.x",
        *local_keys,
        *(key for key, _path in global_paths),
    )
    persistent = (
        ("zeta.write", ("ion_write_buffers", "zeta", "write")),
        ("zeta.collision", ("ion_write_buffers", "zeta", "collision")),
        ("alpha.write", ("ion_write_buffers", "alpha", "write")),
        ("alpha.z", ("material_write_buffers", "alpha", "z")),
        ("zeta.a", ("material_write_buffers", "zeta", "a")),
        ("zeta.source", ("material_source_buffers", "zeta", "source")),
        ("alpha.source", ("material_source_buffers", "alpha", "source")),
    )
    assert func_api._local_synchronization_keys(handler) == {
        "alpha.read",
        "alpha.collision",
        "zeta.read",
    }
    assert tuple(func_api._persistent_shared_local_keys(handler).items()) == persistent
    assert tuple(func_api._canonical_handler_keys(handler).items()) == (
        ("zeta.x", ("mechanisms", "zeta", "x")),
        ("alpha.x", ("mechanisms", "alpha", "x")),
        *global_paths,
        *persistent,
    )
    expected = [
        ("zeta.x", handler.mechanisms["zeta"], "x"),
        ("alpha.x", handler.mechanisms["alpha"], "x"),
    ]
    for key in local_keys:
        mechanism_name, name = key.split(".")
        owner = handler.mechanisms[mechanism_name]
        expected.extend(((key, owner, name), (key, owner.DE["_LocalState"], name)))
    expected.extend(
        (
            key,
            handler.ions["na"] if path[0] == "ions" else handler.materials["pool"],
            path[-1],
        )
        for key, path in global_paths
    )
    assert handler._mutable_state_bindings() == tuple(expected)


def test_declared_state_path_wins_over_persistent_local_role_collision():
    handler = _handler()
    handler.source_material["pool"]["zeta"]["extra"] = "x"
    assert func_api._persistent_shared_local_keys(handler)["zeta.x"] == (
        "material_source_buffers",
        "zeta",
        "x",
    )
    assert func_api._canonical_handler_keys(handler)["zeta.x"] == (
        "mechanisms",
        "zeta",
        "x",
    )


@pytest.mark.parametrize("duplicate_state", (False, True))
def test_read_mirror_slots_keep_alias_expansion_deduplication_and_final_sort(
    duplicate_state,
):
    population = torch.nn.Module()
    population.add_module("mech", _handler())
    if duplicate_state:
        for owner in population.mech.mechanisms.values():
            owner.DE["alias"] = owner.DE["_LocalState"]
    suffixes = (
        ("", ".DE._LocalState", ".DE._LocalState")
        if duplicate_state
        else ("", ".DE._LocalState")
    )
    expected = tuple(
        sorted(
            f"population.mech.mechanisms.{mechanism}{suffix}.{name}"
            for mechanism, name in (
                ("alpha", "read"),
                ("alpha", "collision"),
                ("zeta", "read"),
            )
            for suffix in suffixes
        )
    )
    assert func_api._local_synchronization_slots(population) == expected


def test_snapshot_restore_keeps_global_local_state_aliases_and_gradients():
    handler = _handler()
    shared = torch.tensor([[2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    owner = handler.mechanisms["zeta"]
    owner.write = shared
    owner.DE["_LocalState"].write = shared
    handler.ions["na"].nai = shared
    handler.materials["pool"].late = shared
    checkpoint = handler.mutable_state_dict()
    keys = ("zeta.write", "na_ion.nai", "pool_material.late")
    assert all(checkpoint[key] is checkpoint[keys[0]] for key in keys)
    handler.restore_mutable_state_dict(checkpoint)
    restored = (
        owner.write,
        owner.DE["_LocalState"].write,
        handler.ions["na"].nai,
        handler.materials["pool"].late,
    )
    assert all(value is restored[0] for value in restored)
    assert restored[0] is not checkpoint[keys[0]]
    torch.testing.assert_close(
        torch.autograd.grad(sum(value.sum() for value in restored), shared)[0],
        torch.full_like(shared, 4.0),
    )


@pytest.mark.parametrize(
    "role", ("ion_write", "ion_read", "material_write", "material_source")
)
def test_missing_state_local_reference_fallback_remains_specific_to_ion_write(role):
    name = "nai" if role.startswith("ion") else "amount"
    owner = _mechanism("user", (name,), state_locals=False)
    maps = {
        "ion_write": {"write_ion_c": {"na": {"user": [name]}}},
        "ion_read": {"read_ion": {"na": {"user": [name]}}},
        "material_write": {"write_material": {"pool": {"user": [name]}}},
        "material_source": {"source_material": {"pool": {"user": {"amount": name}}}},
    }
    handler = MechanismHandler(
        owner.celsius, torch.ones_like(owner.celsius), {"user": owner}, **maps[role]
    )
    checkpoint = handler.mutable_state_dict()
    before = _snapshot(handler)
    if role == "ion_write":
        handler.restore_mutable_state_dict(checkpoint)
        state = owner.DE["_LocalState"]
        assert getattr(state, name) is owner._buffers[name]
        assert name not in state._buffers
    else:
        with pytest.raises(TypeError) as caught:
            handler.restore_mutable_state_dict(checkpoint)
        assert (
            str(caught.value)
            == f"Live mutable field 'user.{name}' must be a tensor before restore."
        )
        _assert_unchanged(before)


@pytest.mark.parametrize("collection", ("ions", "materials"))
@pytest.mark.parametrize("operation", ("snapshot", "preflight"))
def test_native_field_discovery_evaluates_fallback_before_existing_fields(
    collection, operation, monkeypatch
):
    handler = _handler()
    checkpoint = handler.mutable_state_dict()
    source = next(iter(getattr(handler, collection).values()))
    failure = RuntimeError("authored field fallback failure")
    calls = []

    def fail(*args, **kwargs):
        calls.append("fallback")
        raise failure

    monkeypatch.setattr(source, "named_buffers", fail)
    before = _snapshot(handler)
    assert func_api._canonical_handler_keys(handler)
    assert calls == []
    with pytest.raises(RuntimeError) as caught:
        if operation == "snapshot":
            handler.mutable_state_dict()
        else:
            handler._preflight_mutable_state_dict(checkpoint)
    assert caught.value is failure
    assert calls == ["fallback"]
    _assert_unchanged(before)


@pytest.mark.parametrize(
    "collection,prefix", (("ions", "na_ion"), ("materials", "pool_material"))
)
def test_native_field_fallback_does_not_relax_functional_field_admission(
    collection, prefix
):
    source = torch.nn.Module()
    source.register_buffer("only", torch.ones(SHAPE, dtype=torch.float64))
    handler = MechanismHandler(
        source.only,
        torch.ones_like(source.only),
        {},
        **{collection: {"na" if collection == "ions" else "pool": source}},
    )
    assert tuple(handler.mutable_state_dict()) == (
        "__dendra_mechanism_support_identity__",
        f"{prefix}.only",
    )
    handler.restore_mutable_state_dict(handler.mutable_state_dict())
    with pytest.raises(AttributeError, match="has no attribute 'fields'"):
        func_api._canonical_handler_keys(handler)


class _ObservedMap(dict):
    def __init__(self, values, reads):
        super().__init__(values)
        self.reads = reads

    def items(self):
        self.reads.append("items")
        return super().items()

    def values(self):
        self.reads.append("values")
        return super().values()


@pytest.mark.parametrize("operation", ("snapshot", "preflight", "functional"))
def test_outer_mapping_access_keeps_native_snapshot_and_key_projection_boundaries(
    operation,
):
    handler = _handler()
    checkpoint = handler.mutable_state_dict()
    reads = []
    handler.read_ion = _ObservedMap(handler.read_ion, reads)
    if operation == "snapshot":
        handler.mutable_state_dict()
    elif operation == "preflight":
        handler._preflight_mutable_state_dict(checkpoint)
    else:
        func_api._canonical_handler_keys(handler)
    assert reads == ["items" if operation == "snapshot" else "values"]


class _ObservedFields(list):
    def __init__(self, reads):
        super().__init__()
        self.reads = reads

    def __iter__(self):
        self.reads.append("fields")
        return super().__iter__()


@pytest.mark.parametrize("operation", ("snapshot", "preflight"))
@pytest.mark.parametrize("kind", ("ion", "material"))
def test_empty_unknown_native_owner_keeps_lookup_and_field_iteration_order(
    kind, operation
):
    handler = _handler()
    checkpoint = handler.mutable_state_dict()
    reads = []
    if kind == "ion":
        handler.read_ion = {"na": {"missing": _ObservedFields(reads)}}
    else:
        handler.read_material = {"pool": {"missing": _ObservedFields(reads)}}
    before = _snapshot(handler)
    with pytest.raises(KeyError) as caught:
        if operation == "snapshot":
            handler.mutable_state_dict()
        else:
            handler._preflight_mutable_state_dict(checkpoint)
    assert caught.value.args == ("missing",)
    assert reads == ([] if kind == "ion" else ["fields"])
    _assert_unchanged(before)
    reads.clear()
    assert func_api._canonical_handler_keys(handler)
    assert reads == ["fields"]
