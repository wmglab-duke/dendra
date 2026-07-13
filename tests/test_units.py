"""Regression tests for Dendra's public scalar unit conversions."""

import pytest
import torch

from dendra import units as U
from dendra.models.mechanisms import Mechanism, PointProcess
from dendra.models.mechanisms._handler import make_scaler


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ms", 1.0),
        ("s", 1.0e3),
        ("minutes", 6.0e4),
        ("hours", 3.6e6),
        ("mV", 1.0),
        ("V", 1.0e3),
        ("mA", 1.0),
        ("A", 1.0e3),
        ("uA", 1.0e-3),
        ("nA", 1.0e-6),
        ("pA", 1.0e-9),
        ("fA", 1.0e-12),
        ("ohm", 1.0),
        ("S", 1.0),
        ("mS", 1.0e-3),
        ("uS", 1.0e-6),
        ("nS", 1.0e-9),
        ("pS", 1.0e-12),
        ("fS", 1.0e-15),
        ("uF", 1.0),
        ("F", 1.0e6),
        ("nF", 1.0e-3),
        ("pF", 1.0e-6),
        ("fF", 1.0e-9),
        ("um", 1.0),
        ("m", 1.0e6),
        ("cm", 1.0e4),
        ("mm", 1.0e3),
        ("nm", 1.0e-3),
        ("kHz", 1.0),
        ("Hz", 1.0e-3),
        ("MHz", 1.0e3),
        ("GHz", 1.0e6),
    ],
)
def test_unit_scalar_converts_to_documented_base_coordinate(name, expected):
    assert getattr(U, name) == pytest.approx(expected)


def test_voltage_conversion_regression():
    assert 0.05 * U.V == pytest.approx(50.0 * U.mV)


def test_conductance_voltage_current_identities():
    assert U.S * U.mV == pytest.approx(U.mA)
    assert U.uS * U.mV == pytest.approx(U.nA)


def test_convert_ns_to_point_process_us_coordinate():
    assert 50.0 * U.nS / U.uS == pytest.approx(0.05)


def test_mechanism_scalers_implement_documented_density_and_point_contracts():
    shape = (1, 2)
    celsius = torch.full(shape, 34.0)
    diameters = torch.ones(shape)
    area_cm2 = torch.tensor([[2.0e-6, 4.0e-6]])

    distributed = Mechanism("distributed", celsius, diameters, shape, shape)
    distributed_scale = make_scaler(distributed, area_cm2)
    current_density = torch.tensor([[4.0, 8.0]])  # mA/cm²
    conductance_density = torch.tensor([[6.0, 12.0]])  # S/cm²
    scaled_i, scaled_g = distributed_scale(current_density, conductance_density)
    assert scaled_i is current_density
    assert scaled_g is conductance_density

    point = PointProcess("point", celsius, diameters, shape, shape)
    point_scale = make_scaler(point, area_cm2)
    current_na = torch.tensor([[4.0, 8.0]])
    conductance_us = torch.tensor([[6.0, 12.0]])
    scaled_i, scaled_g = point_scale(current_na, conductance_us)

    # raw nA/µS divided by 1e6 * area_cm2 -> mA/cm² / S/cm²
    torch.testing.assert_close(scaled_i, torch.tensor([[2.0, 2.0]]))
    torch.testing.assert_close(scaled_g, torch.tensor([[3.0, 3.0]]))
