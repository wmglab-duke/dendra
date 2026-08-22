"""Gather-once scheduling for shared Ion and Material field reads."""

from __future__ import annotations

import copy

import torch

import dendra as dn  # noqa: F401 - configure Dendra before defining mechanisms
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._ions import Ion
from dendra.models.mechanisms._materials import Material
from dendra.models.mechanisms._support_registry import SupportEntry

DTYPE = torch.float64
SHAPE = (2, 4)
KEY = torch.tensor([0, 2, 4, 6], dtype=torch.long)
LOCAL_SHAPE = (4,)


class _IonReader(Mechanism):
    Mechanism.USEION("na", read=["nai", "ina"])


class _MaterialReader(Mechanism):
    Mechanism.USEMATERIAL("pool", read=["amount"])


class _MaterialReaderWriter(Mechanism):
    Mechanism.USEMATERIAL("pool", read=["amount"], write=["amount"])


class _MaterialWriter(Mechanism):
    Mechanism.USEMATERIAL("pool", write=["amount"])


def _mechanism(cls, name):
    field = torch.ones(SHAPE, dtype=DTYPE)
    return cls(
        name,
        torch.full(SHAPE, 34.0, dtype=DTYPE),
        field,
        LOCAL_SHAPE,
        LOCAL_SHAPE,
        key=KEY.clone(),
    )


def _mechanism_on(cls, name, key):
    field = torch.ones(SHAPE, dtype=DTYPE)
    key = torch.as_tensor(key, dtype=torch.long)
    return cls(
        name,
        torch.full(SHAPE, 34.0, dtype=DTYPE),
        field,
        (key.numel(),),
        (key.numel(),),
        key=key,
    )


def _handler():
    ion = Ion("na", SHAPE, einit=0, eadvance=0).to(dtype=DTYPE)
    material = Material("pool", SHAPE, fields={"amount": 0.0}).to(dtype=DTYPE)
    ion_first = _mechanism(_IonReader, "ion_first")
    ion_second = _mechanism(_IonReader, "ion_second")
    material_writer = _mechanism(_MaterialReaderWriter, "material_writer")
    material_reader = _mechanism(_MaterialReader, "material_reader")
    for mechanism in (ion_first, ion_second):
        mechanism.register_ion(ion)
    for mechanism in (material_writer, material_reader):
        mechanism.register_material(material)

    handler = MechanismHandler(
        torch.full(SHAPE, 34.0, dtype=DTYPE),
        torch.ones(SHAPE, dtype=DTYPE),
        {
            "ion_first": ion_first,
            "ion_second": ion_second,
            "material_writer": material_writer,
            "material_reader": material_reader,
        },
        ions={"na": ion},
        materials={"pool": material},
        read_ion={
            "na": {
                "ion_first": ["nai", "ina"],
                "ion_second": ["nai", "ina"],
            }
        },
        read_material={
            "pool": {
                "material_writer": ["amount"],
                "material_reader": ["amount"],
            }
        },
        write_material={"pool": {"material_writer": ["amount"]}},
    )
    handler.make_maps()
    return handler


def test_shared_field_reads_gather_once_per_field_and_support(monkeypatch):
    handler = _handler()
    assert len(handler._state_support_representatives) == 1
    assert len(handler._ion_read_support_plan) == 2
    assert len(handler._material_read_support_plan) == 1
    assert len(handler._ion_current_read_support_plan) == 1
    assert {
        support_entry.support_id
        for _, _, support_entry, _ in handler._ion_read_support_plan
    } == {0}
    assert {
        support_entry.support_id
        for _, _, support_entry, _ in handler._material_read_support_plan
    } == {0}
    assert {
        support_entry.support_id
        for _, _, support_entry, _ in handler._ion_current_read_support_plan
    } == {0}

    support_entry = handler._state_support_entries[0]
    original_gather = SupportEntry.gather
    calls = 0

    def counted_gather(entry, tensor):
        nonlocal calls
        calls += 1
        return original_gather(entry, tensor)

    monkeypatch.setattr(SupportEntry, "gather", counted_gather)
    nai = torch.arange(8, dtype=DTYPE).reshape(SHAPE)
    ina = 100.0 + nai
    amount = 200.0 + nai
    handler.ions["na"].nai = nai
    handler.ions["na"].ina = ina
    handler.materials["pool"].amount = amount

    def forbidden_get(*args, **kwargs):
        del args, kwargs
        raise AssertionError("handler execution re-entered Mechanism.get")

    for mechanism in handler.mechanisms.values():
        monkeypatch.setattr(mechanism, "get", forbidden_get)

    handler.read_from_ions()
    assert calls == 2
    expected_nai = original_gather(support_entry, nai)
    expected_ina = original_gather(support_entry, ina)
    for name in ("ion_first", "ion_second"):
        torch.testing.assert_close(handler.mechanisms[name].nai, expected_nai)
        torch.testing.assert_close(handler.mechanisms[name].ina, expected_ina)

    calls = 0
    handler.read_from_materials()
    assert calls == 1
    expected_amount = original_gather(support_entry, amount)
    writer = handler.material_writer
    reader = handler.material_reader
    torch.testing.assert_close(writer.amount, expected_amount)
    torch.testing.assert_close(reader.amount, expected_amount)
    assert writer.amount.data_ptr() != reader.amount.data_ptr()
    writer.amount.add_(1.0)
    torch.testing.assert_close(reader.amount, expected_amount)


def test_shared_current_frame_readers_gather_once_per_support(monkeypatch):
    handler = _handler()
    support_entry = handler._state_support_entries[0]
    original_gather = SupportEntry.gather
    calls = 0

    def counted_gather(entry, tensor):
        nonlocal calls
        calls += 1
        return original_gather(entry, tensor)

    monkeypatch.setattr(SupportEntry, "gather", counted_gather)
    current = torch.arange(8, dtype=DTYPE).reshape(SHAPE) + 300.0

    handler._publish_ion_current_frame((current,))

    assert calls == 1
    expected = original_gather(support_entry, current)
    for name in ("ion_first", "ion_second"):
        torch.testing.assert_close(handler.mechanisms[name].ina, expected)


def test_material_replacements_preserve_authored_nonadjacent_support_order(
    monkeypatch,
):
    material = Material("pool", SHAPE, fields={"amount": 0.0}).to(dtype=DTYPE)
    mechanisms = {
        "first": _mechanism_on(_MaterialWriter, "first", [0, 1]),
        "middle": _mechanism_on(_MaterialWriter, "middle", [1]),
        "last": _mechanism_on(_MaterialWriter, "last", [0, 1]),
    }
    for mechanism in mechanisms.values():
        mechanism.register_material(material)
    handler = MechanismHandler(
        torch.full(SHAPE, 34.0, dtype=DTYPE),
        torch.ones(SHAPE, dtype=DTYPE),
        mechanisms,
        materials={"pool": material},
        write_material={
            "pool": {
                "first": ["amount"],
                "middle": ["amount"],
                "last": ["amount"],
            }
        },
    )
    handler.make_maps()
    handler.first.amount = torch.full((2,), 10.0, dtype=DTYPE)
    handler.middle.amount = torch.full((1,), 20.0, dtype=DTYPE)
    handler.last.amount = torch.full((2,), 30.0, dtype=DTYPE)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("handler execution re-entered Mechanism.put")

    for mechanism in mechanisms.values():
        monkeypatch.setattr(mechanism, "put", forbidden)

    scatter_order = []
    original_scatter_set = SupportEntry.scatter_set

    def ordered_scatter_set(entry, *args, **kwargs):
        scatter_order.append(entry.support_id)
        return original_scatter_set(entry, *args, **kwargs)

    monkeypatch.setattr(SupportEntry, "scatter_set", ordered_scatter_set)
    handler.write_material_replacements(torch.zeros(SHAPE, dtype=DTYPE))

    # The final writer shares support 0 with the first writer, but the support-1
    # write between them must not be reordered or bundled across.
    assert scatter_order == [0, 1, 0]
    expected = torch.zeros(SHAPE, dtype=DTYPE)
    expected.reshape(-1)[:2] = 30.0
    torch.testing.assert_close(handler.materials["pool"].amount, expected)


def test_deepcopy_rebinds_sync_wrappers_to_clone_owned_fields():
    source = _handler()
    clone = copy.deepcopy(source)

    for method_name in (
        "read_from_ions",
        "read_from_materials",
        "write_to_materials",
    ):
        assert getattr(clone, method_name) is not getattr(source, method_name)

    def full(value):
        return torch.full(SHAPE, value, dtype=DTYPE)

    def local(value):
        return torch.full(LOCAL_SHAPE, value, dtype=DTYPE)

    source.ions["na"].nai = full(1.0)
    source.ions["na"].ina = full(2.0)
    source.materials["pool"].amount = full(3.0)
    clone.ions["na"].nai = full(11.0)
    clone.ions["na"].ina = full(12.0)
    clone.materials["pool"].amount = full(13.0)

    for mechanism_name in ("ion_first", "ion_second"):
        source.mechanisms[mechanism_name].nai = local(-1.0)
        source.mechanisms[mechanism_name].ina = local(-2.0)
    source.material_writer.amount = local(-3.0)
    source.material_reader.amount = local(-4.0)

    clone.read_from_ions()
    clone.read_from_materials()

    for mechanism_name in ("ion_first", "ion_second"):
        torch.testing.assert_close(clone.mechanisms[mechanism_name].nai, local(11.0))
        torch.testing.assert_close(clone.mechanisms[mechanism_name].ina, local(12.0))
        torch.testing.assert_close(source.mechanisms[mechanism_name].nai, local(-1.0))
        torch.testing.assert_close(source.mechanisms[mechanism_name].ina, local(-2.0))
    torch.testing.assert_close(clone.material_writer.amount, local(13.0))
    torch.testing.assert_close(clone.material_reader.amount, local(13.0))
    torch.testing.assert_close(source.material_writer.amount, local(-3.0))
    torch.testing.assert_close(source.material_reader.amount, local(-4.0))

    clone.material_writer.amount = local(99.0)
    clone.write_to_materials(torch.zeros(SHAPE, dtype=DTYPE))
    expected_clone_material = full(13.0)
    expected_clone_material.reshape(-1).scatter_(0, KEY, local(99.0))
    torch.testing.assert_close(
        clone.materials["pool"].amount,
        expected_clone_material,
    )
    torch.testing.assert_close(source.materials["pool"].amount, full(3.0))
