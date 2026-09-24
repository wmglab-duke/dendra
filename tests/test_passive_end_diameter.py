"""Effective geometry and lifecycle contracts for passive-end diameters."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import hh
from tests.test_passive_end_nodes import (
    DTYPE,
    _assert_tree,
    _mask,
    _mechanism_values,
    _model,
    _snapshot,
)


@pytest.fixture(autouse=True)
def _eager_geometry():
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE=DTYPE):
        yield


def _expanded_diameter(model):
    return torch.broadcast_to(model.diam, model.shape)


@pytest.mark.parametrize("kind", ["cable", "axon"])
def test_diameter_slice_edit_is_physical_and_survives_rebuild_and_added_batch(kind):
    model = _model(kind=kind, batches=(2,))
    expected = _expanded_diameter(model).clone()
    selected = _mask(model, [1, 5], row=1)
    expected[selected] = 0.75
    live_voltage = model.v.clone()

    # The local ends are 5 and 1, despite reordered/duplicate interior entries.
    # Selecting replica 1 still edits the shared physical cable geometry.
    dn.passive_end_nodes_(model[1, 1, [5, 2, 5, 1]], diam=0.75, rhoa=None, cm=None)
    torch.testing.assert_close(model.v, live_voltage, rtol=0, atol=0)
    for force_rebuild in (False, True):
        model.initialize(force_rebuild=force_rebuild)
        torch.testing.assert_close(_expanded_diameter(model), expected, rtol=0, atol=0)
    model.batch(3)
    model.initialize()
    torch.testing.assert_close(
        _expanded_diameter(model),
        expected.unsqueeze(0).expand(model.shape),
        rtol=0,
        atol=0,
    )


def test_myelinated_overrides_preserve_raw_geometry_and_follow_polynomial_updates():
    model = dn.Myelinated(
        [8.0, 12.0],
        n_node=7,
        node_length=2.0,
        dtype=DTYPE,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(hh)
    model.initialize()
    raw = model.parametrizations.diam.original
    raw_before = raw.clone()
    fiber_before = model.diameters.clone()
    first = _mask(model, [0, 6], row=0)
    second = _mask(model, [1, 5], row=1)

    dn.passive_end_nodes_(model[0], diam=0.75, rhoa=None, cm=None)
    dn.passive_end_nodes_(model[1, 1:6], diam=1.25, rhoa=None, cm=None)
    dn.passive_end_nodes_(model[0, 0], diam=1.125, rhoa=None, cm=None)
    # Explicit None must not erase earlier overrides or freeze other nodes.
    dn.passive_end_nodes_(model[1, 3], diam=None, rhoa=None, cm=None)
    assert model.parametrizations.diam.original is raw
    torch.testing.assert_close(raw, raw_before, rtol=0, atol=0)
    torch.testing.assert_close(model.diameters, fiber_before, rtol=0, atol=0)

    # Change both the raw node input and its polynomial coefficient after
    # installing the override; unselected diameters must remain computed.
    with torch.no_grad():
        raw.add_(1.5)
        model.noded2.add_(0.1)
    expected = model.noded1 * raw.square() + model.noded2 * raw + model.noded3
    expected = expected.detach().clone()
    expected[first] = 0.75
    expected[second] = 1.25
    expected[0, 0] = 1.125
    for force_rebuild in (False, True):
        model.initialize(force_rebuild=force_rebuild)
        torch.testing.assert_close(_expanded_diameter(model), expected, rtol=0, atol=0)
    model.batch(2)
    model.initialize()
    torch.testing.assert_close(
        _expanded_diameter(model),
        expected.unsqueeze(0).expand(model.shape),
        rtol=0,
        atol=0,
    )


def test_diameter_updates_sparse_mechanism_area_and_initialized_axial_geometry():
    model = _model(kind="axon")
    selected = _mask(model, [0, 6])
    before_area = model.area.clone()
    dn.passive_end_nodes_(model, diam=100.0, rhoa=None, cm=None)
    model.initialize()

    diam = _expanded_diameter(model)
    length = torch.broadcast_to(model.dx, model.shape)
    area = torch.pi * diam * length * 1.0e-8
    torch.testing.assert_close(model.area, area, rtol=1e-14, atol=0)
    torch.testing.assert_close(
        model.area[~selected], before_area[~selected], rtol=0, atol=0
    )
    assert torch.all(model.area[selected] > before_area[selected])
    mechanism_diam = _mechanism_values(model, "pas", "diam")
    torch.testing.assert_close(mechanism_diam, diam, rtol=0, atol=0)

    # An edge consists of two half-cylinders in series. Verify the initialized
    # voltage solver uses the edited radii and resulting membrane capacitance.
    radius_cm = 0.5e-4 * diam
    rhoa = torch.broadcast_to(model.rhoa * model.rhoa_scale, model.shape)
    full_resistance = rhoa * (length * 1.0e-4) / (torch.pi * radius_cm.square())
    conductance = 2 / (full_resistance[..., :-1] + full_resistance[..., 1:])
    capacitance = 1.0e-6 * model.cm * model.cm_scale * area * model.area_scale
    model.integrator._initialize(
        model, torch.tensor(0.01, dtype=DTYPE), force=True, compile_scope="population"
    )
    torch.testing.assert_close(
        model.integrator.cm_inv, capacitance.reciprocal(), rtol=1e-14, atol=0
    )
    torch.testing.assert_close(
        model.integrator.g_edge_Cinv,
        conductance / capacitance[..., :-1],
        rtol=1e-14,
        atol=0,
    )
    torch.testing.assert_close(
        model.integrator.g_edge_Cinv_right,
        conductance / capacitance[..., 1:],
        rtol=1e-14,
        atol=0,
    )


@pytest.mark.parametrize("diam", [0.0, -1.0, float("nan"), torch.tensor([1.0, 2.0])])
def test_invalid_diameter_does_not_remove_mechanisms_or_edit_geometry(diam):
    model = _model()
    before = _snapshot(model)
    with pytest.raises((TypeError, ValueError), match="diam"):
        dn.passive_end_nodes_(model, diam=diam)
    _assert_tree(_snapshot(model), before)


def test_native_diameter_rejection_precedes_mechanism_deletion():
    model = _model(kind="native")
    before = _snapshot(model)
    with pytest.raises(ValueError, match="diam|geometry|immutable"):
        dn.passive_end_nodes_(model[:, 1:6], diam=0.75, rhoa=None, cm=None)
    _assert_tree(_snapshot(model), before)


def test_in_graph_diameter_rejection_precedes_mechanism_deletion():
    model = _model()
    model.register_parametrization_in_graph("diam", torch.nn.Identity())
    before = _snapshot(model)
    with pytest.raises(ValueError, match="in-graph diameter"):
        dn.passive_end_nodes_(model, diam=0.75)
    _assert_tree(_snapshot(model), before)
