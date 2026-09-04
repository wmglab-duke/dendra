"""Focused contracts for the handler's single initialization transaction."""

from __future__ import annotations

from types import MethodType

import torch
import torch._inductor.config as inductor_config

import dendra  # noqa: F401  (configure Dendra before constructing mechanisms)
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._ions import Ion
from dendra.models.mechanisms._materials import Material

DTYPE = torch.float64
SHAPE = (1, 2)
VOLTAGE = torch.tensor([[-65.0, -55.0]], dtype=DTYPE)
CELSIUS = torch.full(SHAPE, 34.0, dtype=DTYPE)
DIAMETERS = torch.ones(SHAPE, dtype=DTYPE)


def test_handler_construction_does_not_mutate_global_compiler_configuration():
    """Handler composition must not select process-global compiler policy."""

    for configured_value in (False, True):
        with inductor_config.patch({"cpp_wrapper": configured_value}):
            simple = MechanismHandler(CELSIUS, torch.ones_like(CELSIUS), {})
            assert inductor_config.cpp_wrapper is configured_value
            assert not simple.requires_inductor_python_wrapper

            synchronized = MechanismHandler(
                CELSIUS,
                torch.ones_like(CELSIUS),
                {},
                read_ion={"na": {}},
                read_material={"pool": {}},
            )
            assert inductor_config.cpp_wrapper is configured_value
            assert synchronized.requires_inductor_python_wrapper


class _CountingMaterial(Material):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_calls = 0

    def initialize(self, *args, **kwargs):
        self.initialize_calls += 1
        return super().initialize(*args, **kwargs)


class _StepSource(Mechanism):
    Mechanism.CARRY("delta")
    Mechanism.USEMATERIAL("pool", read=["amount"], source={"amount": "delta"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.populate_calls = 0

    def populate(self, *args, **kwargs):
        self.populate_calls = getattr(self, "populate_calls", 0) + 1
        return super().populate(*args, **kwargs)

    def initial_values(self, v, values):
        del values
        return {"delta": torch.full_like(v, 0.25)}


class _GuardedWriter(Mechanism):
    Mechanism.CARRY("advance_calls")
    Mechanism.USEMATERIAL("pool", read=["amount"], write=["amount"])

    def initial_values(self, v, values):
        del values
        return {
            "amount": torch.full_like(v, -2.0),
            "advance_calls": torch.zeros_like(v),
        }

    def advance(self, v, dt, values):
        del dt
        return {
            "amount": torch.full_like(v, -3.0),
            "advance_calls": values["advance_calls"] + 1,
        }


class _SodiumConcentrationWriter(Mechanism):
    Mechanism.USEION("na", write=["nai"])

    def initial_values(self, v, values):
        del values
        return {"nai": torch.full_like(v, 5.0)}


class _NernstSodiumCurrent(Mechanism):
    Mechanism.RANGE(g=0.01)
    Mechanism.USEION("na", read=["ena"], write=["ina"])

    def initial_values(self, v, values):
        del v, values
        return {}

    def ina(self, v):
        return self.g * (v - self.ena)

    def ina_with_conductance(self, v):
        return self.ina(v), self.g


class _SodiumCurrentReader(Mechanism):
    Mechanism.CARRY("initial_seen")
    Mechanism.USEION("na", read=["ina"])

    def initial_values(self, v, values):
        del v
        return {"initial_seen": values["ina"].clone()}


def _mechanism(cls, name):
    return cls(name, CELSIUS, DIAMETERS, SHAPE, SHAPE).to(DTYPE)


def test_initialization_excludes_step_sources_and_material_process_phase():
    material = _CountingMaterial("pool", SHAPE, fields={"amount": 1.0}).to(DTYPE)
    source = _mechanism(_StepSource, "source")
    source.register_material(material)
    handler = MechanismHandler(
        CELSIUS,
        torch.ones(SHAPE, dtype=DTYPE),
        {"source": source},
        materials={"pool": material},
        read_material={"pool": {"source": ["amount"]}},
        source_material={"pool": {"source": {"amount": "delta"}}},
    ).to(DTYPE)

    process_calls = 0
    advance_processes = handler.advance_material_processes

    def count_process_phase(self, dt):
        nonlocal process_calls
        process_calls += 1
        return advance_processes(dt)

    handler.advance_material_processes = MethodType(count_process_phase, handler)
    handler.initialize(VOLTAGE, CELSIUS, DIAMETERS)

    torch.testing.assert_close(material.amount, torch.ones(SHAPE, dtype=DTYPE))
    assert process_calls == 0
    assert material.initialize_calls == 1
    assert source.populate_calls == 1

    handler.advance(VOLTAGE, torch.as_tensor(0.025, dtype=DTYPE), CELSIUS)

    torch.testing.assert_close(material.amount, torch.full(SHAPE, 1.25, dtype=DTYPE))
    assert process_calls == 1
    assert source.populate_calls == 1


def test_replacement_write_is_guarded_without_a_trailing_overwrite():
    material = Material(
        "pool",
        SHAPE,
        fields={"amount": 1.0},
        min_values={"amount": 0.5},
    ).to(DTYPE)
    writer = _mechanism(_GuardedWriter, "writer")
    writer.register_material(material)
    handler = MechanismHandler(
        CELSIUS,
        torch.ones(SHAPE, dtype=DTYPE),
        {"writer": writer},
        materials={"pool": material},
        read_material={"pool": {"writer": ["amount"]}},
        write_material={"pool": {"writer": ["amount"]}},
    ).to(DTYPE)

    handler.initialize(VOLTAGE, CELSIUS, DIAMETERS)

    expected = torch.full(SHAPE, 0.5, dtype=DTYPE)
    torch.testing.assert_close(material.amount, expected)
    torch.testing.assert_close(writer.amount, expected)
    torch.testing.assert_close(
        writer.advance_calls,
        torch.zeros_like(writer.advance_calls),
    )


def test_final_current_frame_uses_post_commit_nernst_state_without_reinitializing(
    monkeypatch,
):
    sodium = Ion("na", SHAPE, einit=1, eadvance=1).to(DTYPE)
    writer = _mechanism(_SodiumConcentrationWriter, "writer")
    channel = _mechanism(_NernstSodiumCurrent, "channel")
    reader = _mechanism(_SodiumCurrentReader, "reader")
    for mechanism in (writer, channel, reader):
        mechanism.register_ion(sodium)

    handler = MechanismHandler(
        CELSIUS,
        torch.ones(SHAPE, dtype=DTYPE),
        {"writer": writer, "channel": channel, "reader": reader},
        ions={"na": sodium},
        write_ion_c={"na": {"writer": ["nai"]}},
        read_ion={
            "na": {
                "channel": ["ena"],
                "reader": ["ina"],
            }
        },
        currents={"ina": {"channel": ["ina"]}},
    ).to(DTYPE)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("handler execution re-entered a Mechanism mapper")

    for mechanism in handler.mechanisms.values():
        for name in ("get", "add_", "add", "put"):
            monkeypatch.setattr(mechanism, name, forbidden)

    handler.initialize(VOLTAGE, CELSIUS, DIAMETERS)

    expected_nai = torch.full(SHAPE, 5.0, dtype=DTYPE)
    expected_current = channel.ina(VOLTAGE)
    torch.testing.assert_close(sodium.nai, expected_nai)
    torch.testing.assert_close(sodium.ina, expected_current)
    torch.testing.assert_close(reader.ina, expected_current)
    torch.testing.assert_close(handler.capture_ion_current_frame()[0], expected_current)
    assert not torch.allclose(reader.initial_seen, expected_current)
