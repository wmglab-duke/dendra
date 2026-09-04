"""Public registry, declaration, and default-context contracts for materials."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import (
    ClearanceProcess,
    Material,
    Mechanism,
)
from dendra.models.mechanisms import _materials as materials
from dendra.models.mechanisms import (
    material_defaults,
    register_material,
)
from dendra.models.mechanisms._materials import MaterialFieldSpec


@pytest.fixture(autouse=True)
def _isolated_material_registry(monkeypatch):
    """Keep global registrations and the use-last cache local to each test."""
    monkeypatch.setattr(materials, "MATERIAL_SPECS", {})
    monkeypatch.setattr(materials, "MATERIAL_ALIASES", {})
    monkeypatch.setattr(material_defaults, "_last", {})


class _PythonScalarSource(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self):
        return self.value


def test_registration_aliases_and_mapping_metadata_drive_material_construction():
    specs = register_material(
        "solute",
        fields={"amount": 1.0, "temperature": 2.0},
        initial_values={"amount": 3.0, "added": 4.0},
        min_values={"amount": 0.25},
        conserved={"amount": False},
        domain={"amount": "intracellular"},
        units={"amount": "mM"},
        aliases=("pool", 7),
    )

    assert materials.material_specs() == {"solute": specs}
    assert materials.valid_materials() == ["solute"]
    assert tuple(specs) == ("amount", "temperature", "added")
    assert specs["amount"] == MaterialFieldSpec(
        "amount",
        initial=3.0,
        min_value=0.25,
        conserved=False,
        domain="intracellular",
        units="mM",
    )
    assert specs["temperature"] == MaterialFieldSpec("temperature", initial=2.0)
    assert specs["added"] == MaterialFieldSpec("added", initial=4.0)

    by_alias = Material("pool", (2, 3))
    by_numeric_alias = Material(7, (1, 3))
    assert by_alias.name == by_numeric_alias.name == "solute"
    assert by_alias.fields == ("amount", "temperature", "added")
    torch.testing.assert_close(by_alias.amount, torch.full((2, 3), 3.0))
    torch.testing.assert_close(by_numeric_alias.added, torch.full((1, 3), 4.0))


def test_registration_treats_one_string_alias_as_one_name():
    register_material("solute", fields={"c": 1.0}, aliases="pool")

    assert materials.MATERIAL_ALIASES == {"pool": "solute"}
    assert Material("pool", (1,)).name == "solute"


def test_scalar_metadata_and_sequence_names_apply_to_every_field():
    specs = register_material(
        "bulk",
        fields=["a", 3],
        initial_values={"3": 2.5},
        conserved=False,
        domain="extracellular",
        units="mM",
    )

    assert tuple(specs) == ("a", "3")
    for spec in specs.values():
        assert not spec.conserved
        assert spec.domain == "extracellular"
        assert spec.units == "mM"
    assert specs["a"].initial == 0.0
    assert specs["3"].initial == 2.5

    material = Material("bulk", (2, 2))
    assert material.has_field("a")
    assert material.has_field(3)
    assert not material.has_field("missing")
    assert material.field_spec(3) is specs["3"]


def test_string_and_mapping_declarations_apply_initial_value_overrides():
    single = Material(
        "single",
        (2,),
        fields="c",
        initial_values={"c": 1.5},
        min_values={"c": 0.0},
    )
    mapping = Material(
        "mapping",
        (1, 2),
        fields={"a": 1.0},
        initial_values={"a": 2.0, "b": 3.0},
    )

    assert single.fields == ("c",)
    torch.testing.assert_close(single.c, torch.full((2,), 1.5))
    assert mapping.fields == ("a", "b")
    torch.testing.assert_close(mapping.a, torch.full((1, 2), 2.0))
    torch.testing.assert_close(mapping.b, torch.full((1, 2), 3.0))


def test_inferred_fields_use_ordered_union_of_initial_and_minimum_keys():
    material = Material(
        "inferred",
        (2,),
        initial_values={"declared": 2.0, "shared": 3.0},
        min_values={"shared": 1.0, "floor_only": 0.5},
    )

    assert material.fields == ("declared", "shared", "floor_only")
    assert material.field_spec("floor_only") == MaterialFieldSpec(
        "floor_only", initial=0.0, min_value=0.5
    )
    material.initialize()
    torch.testing.assert_close(material.declared, torch.full((2,), 2.0))
    torch.testing.assert_close(material.shared, torch.full((2,), 3.0))
    torch.testing.assert_close(material.floor_only, torch.full((2,), 0.5))


def test_python_scalar_parametric_source_is_resolved_with_runtime_dtype():
    source = _PythonScalarSource(1.25)
    material = Material("pool", (2, 2), fields={"c": source}).to(dtype=torch.float64)
    assert material.initial_source("c") is source

    source.value = 3.5
    material.train().initialize()
    assert material.c.dtype == torch.float64
    torch.testing.assert_close(material.c, torch.full((2, 2), 3.5, dtype=torch.float64))


def test_material_defaults_nested_alias_context_restores_after_exception():
    register_material(
        "pool",
        fields={"c": 1.0},
        min_values={"c": 0.1},
        conserved={"c": False},
        domain={"c": "i"},
        units={"c": "mM"},
        aliases=["p"],
    )
    original = materials.material_specs()["pool"]["c"]

    with material_defaults("p", c=2.0):
        outer = materials.material_specs()["pool"]["c"]
        assert outer.initial == 2.0
        assert outer.min_value == original.min_value
        assert outer.conserved == original.conserved
        assert outer.domain == original.domain
        assert outer.units == original.units
        torch.testing.assert_close(Material("pool", (1,)).c, torch.tensor([2.0]))

        with pytest.raises(RuntimeError, match="sentinel"):
            with material_defaults("pool", c=3.0):
                assert materials.material_specs()["pool"]["c"].initial == 3.0
                raise RuntimeError("sentinel")

        assert materials.material_specs()["pool"]["c"] is outer

    assert materials.material_specs()["pool"]["c"] is original


def test_material_defaults_use_last_and_decorator_restore_registry():
    register_material("pool", fields={"c": 1.0}, aliases=["p"])

    with material_defaults("p", c=5.0):
        assert materials.material_specs()["pool"]["c"].initial == 5.0
    assert material_defaults._last == {("pool", "c"): 5.0}

    with material_defaults("pool", use_last=True):
        assert materials.material_specs()["pool"]["c"].initial == 5.0
    assert material_defaults._last == {}
    assert materials.material_specs()["pool"]["c"].initial == 1.0

    @material_defaults("pool", c=7.0)
    def construct_inside_defaults():
        return Material("pool", (1,)).c.clone()

    torch.testing.assert_close(construct_inside_defaults(), torch.tensor([7.0]))
    assert materials.material_specs()["pool"]["c"].initial == 1.0


def test_same_material_defaults_instance_is_reentrant():
    register_material("pool", fields={"c": 1.0})
    defaults = material_defaults("pool", c=2.0)

    with defaults:
        assert materials.material_specs()["pool"]["c"].initial == 2.0
        with defaults:
            assert materials.material_specs()["pool"]["c"].initial == 2.0
        assert materials.material_specs()["pool"]["c"].initial == 2.0

    assert materials.material_specs()["pool"]["c"].initial == 1.0


def test_material_defaults_reject_unknown_materials_and_fields():
    with pytest.raises(ValueError, match="Unknown material 'missing'"):
        material_defaults("missing", c=1.0)

    register_material("pool", fields={"c": 1.0}, aliases=["p"])
    with pytest.raises(ValueError, match=r"Unknown field\(s\).*\['missing'\]"):
        material_defaults("p", missing=2.0)


def test_material_rejects_empty_and_nonbroadcastable_declarations():
    with pytest.raises(ValueError, match="has no fields"):
        Material("empty", (1,))
    with pytest.raises(RuntimeError):
        Material("bad_shape", (2, 2), fields={"c": torch.arange(3.0)})


def test_minimum_guard_has_documented_boundary_gradient():
    source = torch.nn.Parameter(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))
    material = Material(
        "guarded",
        (1, 3),
        fields={"c": source},
        min_values={"c": 0.0},
    ).train()

    material.initialize()
    torch.testing.assert_close(
        material.c, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    )
    material.c.sum().backward()
    torch.testing.assert_close(
        source.grad, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )


def test_late_registered_alias_binds_mechanism_read_write_and_source_end_to_end():
    class AliasReaction(Mechanism):
        Mechanism.USEMATERIAL(
            "late_pool",
            read=["c"],
            write=["c"],
            source={"c": "c_increment"},
        )

        def advance(self, v, dt, values):
            del v
            return {
                "c": values["c"] + dt,
                "c_increment": torch.full_like(values["c_increment"], 2.0 * dt),
            }

    # The alias deliberately does not exist while the class body is evaluated.
    register_material("solute", fields={"c": 1.0}, aliases="late_pool")
    model = dn.Population(N=1, C=2, dtype=torch.float64)
    model.insert(AliasReaction)
    model.initialize()

    assert tuple(model.mech.materials) == ("solute",)
    assert set(model.mech.read_material) == {"solute"}
    assert set(model.mech.write_material) == {"solute"}
    assert set(model.mech.source_material) == {"solute"}
    mechanism = next(iter(model.mech.mechanisms.values()))
    assert mechanism.read_material == {"late_pool": ["c"]}

    model.step(dt=0.1)

    expected = torch.full((1, 2), 1.3, dtype=torch.float64)
    torch.testing.assert_close(model.mech.materials["solute"].c, expected)
    torch.testing.assert_close(mechanism.c, expected)


def test_population_config_and_material_process_resolve_late_registered_alias():
    class AliasClearance(ClearanceProcess):
        ClearanceProcess.CLEAR("late_pool", field="c", rate=1.0, target=0.0)

    model = dn.Population(N=1, C=2, dtype=torch.float64)
    model.material("late_pool", initial_values={"c": 2.0})
    # Both the process declaration and Population config predate the alias.
    register_material(
        "solute",
        fields={"c": 1.0},
        min_values={"c": 0.25},
        domain={"c": "intracellular"},
        aliases="late_pool",
    )
    model.insert(AliasClearance)
    model.initialize()

    material = model.mech.materials["solute"]
    process = next(iter(model.mech.material_processes.values()))
    assert material.field_spec("c").min_value == 0.25
    assert material.field_spec("c").domain == "intracellular"
    assert process._get_material("late_pool") is material

    model.step(dt=0.25)

    expected = torch.full(
        (1, 2), 2.0 * torch.exp(torch.tensor(-0.25)).item(), dtype=torch.float64
    )
    torch.testing.assert_close(material.c, expected)
