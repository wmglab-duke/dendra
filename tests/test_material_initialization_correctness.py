"""Gradient and lifecycle contracts for generic Material initial values."""

from __future__ import annotations

import math

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._ions import Ion
from dendra.models.mechanisms._material_process import ClearanceProcess
from dendra.models.mechanisms._materials import Material
from dendra.models.parametric import PositiveParam

DTYPE = torch.float64
RATE = 0.4
TARGET = 1.25
DT = 0.3


class _ExactClearance(ClearanceProcess):
    ClearanceProcess.CLEAR("pool", field="c", rate=RATE, target=TARGET)


def _clearance_process(shape):
    return _ExactClearance(
        name="clearance",
        celsius=torch.tensor(37.0, dtype=DTYPE),
        diameters=torch.ones(shape, dtype=DTYPE),
        shape=shape,
        shape_f=shape,
    )


def _advance_clearance(material):
    process = _clearance_process(tuple(material.c.shape))
    process._bind_materials({"pool": material}.__getitem__)
    process.advance_materials(DT)
    return material.c.sum()


def _fixed_clearance_loss(initial):
    source = torch.nn.Parameter(torch.tensor(initial, dtype=DTYPE), requires_grad=False)
    material = Material("pool", (1, 3), fields={"c": source}).train()
    material.initialize()
    return float(_advance_clearance(material).item())


def test_trainable_material_initial_is_registered_leaf_and_source_of_truth():
    source = torch.nn.Parameter(torch.tensor(2.5, dtype=DTYPE))
    material = Material("pool", (2, 3), fields={"c": source}).train()

    assert material.initial_source("c") is source
    assert dict(material.named_parameters()) == {"_initial_sources.c": source}
    assert "_initial_sources.c" in material.state_dict()

    material.c = torch.full_like(material.c, -9.0)
    material.initialize()
    torch.testing.assert_close(material.c, source.expand_as(material.c))
    assert material.c.grad_fn is not None

    material.c.sum().backward()
    torch.testing.assert_close(source.grad, torch.tensor(6.0, dtype=DTYPE))


def test_parametric_material_initial_is_registered_and_resolved_each_initialize():
    source = PositiveParam(torch.tensor([1.5, 2.5], dtype=DTYPE), requires_grad=True)
    material = Material("pool", (3, 1, 2), fields={"c": source}).train()

    assert material.initial_source("c") is source
    assert dict(material.named_modules())["_initial_sources.c"] is source
    assert "_initial_sources.c.rho" in dict(material.named_parameters())

    material.initialize()
    expected = source().expand_as(material.c)
    torch.testing.assert_close(material.c, expected)
    material.c.sum().backward()
    assert source.rho.grad is not None
    assert torch.isfinite(source.rho.grad).all()


def test_material_initial_gradient_after_process_matches_analytic_and_central_fd():
    source = torch.nn.Parameter(torch.tensor(2.75, dtype=DTYPE))
    material = Material("pool", (1, 3), fields={"c": source}).train()
    material.initialize()

    loss = _advance_clearance(material)
    loss.backward()

    analytic = torch.tensor(3.0 * math.exp(-RATE * DT), dtype=DTYPE)
    eps = 1.0e-5
    finite_difference = torch.tensor(
        (_fixed_clearance_loss(2.75 + eps) - _fixed_clearance_loss(2.75 - eps))
        / (2.0 * eps),
        dtype=DTYPE,
    )
    torch.testing.assert_close(source.grad, analytic, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        source.grad, finite_difference, rtol=2.0e-10, atol=2.0e-10
    )


def test_repeated_material_initialize_builds_independent_backward_graphs():
    source = torch.nn.Parameter(torch.tensor(1.75, dtype=DTYPE))
    material = Material("pool", (2, 2), fields={"c": source}).train()

    material.initialize()
    first_loss = material.c.square().mean()
    material.initialize()
    second_loss = material.c.square().mean()
    (first_loss + second_loss).backward()
    torch.testing.assert_close(source.grad, 4.0 * source.detach())

    source.grad = None
    material.initialize()
    material.c.square().mean().backward()
    torch.testing.assert_close(source.grad, 2.0 * source.detach())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_material_initial_source_follows_dtype_and_device_moves(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    source = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    material = Material("pool", (2, 2), fields={"c": source}).to(
        device=device, dtype=DTYPE
    )
    material.train().initialize()

    moved_source = material.initial_source("c")
    assert moved_source.device.type == device
    assert moved_source.dtype == DTYPE
    assert material.c.device.type == device
    assert material.c.dtype == DTYPE
    material.c.sum().backward()
    assert moved_source.grad is not None
    assert moved_source.grad.device.type == device
    assert moved_source.grad.dtype == DTYPE


def test_batched_material_shape_broadcasts_one_registered_initial_source():
    source = torch.nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=DTYPE))
    material = Material("pool", (4, 3, 2), fields={"c": source}).train()

    material.initialize()
    torch.testing.assert_close(material.c, source.expand(4, 3, 2))
    material.c.sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 12.0))


def test_material_full_state_dict_restores_initial_source_and_runtime_field():
    source = torch.nn.Parameter(torch.tensor(2.0, dtype=DTYPE))
    material = Material("pool", (1, 2), fields={"c": source}).train()
    material.initialize()
    with torch.no_grad():
        material.c.add_(3.0)
    saved = {
        name: value.detach().clone() for name, value in material.state_dict().items()
    }

    target_source = torch.nn.Parameter(torch.tensor(-4.0, dtype=DTYPE))
    target = Material("pool", (1, 2), fields={"c": target_source}).train()
    target.load_state_dict(saved)
    torch.testing.assert_close(target.initial_source("c"), source)
    torch.testing.assert_close(target.c, material.c)

    target.initialize()
    target.c.sum().backward()
    torch.testing.assert_close(target_source.grad, torch.tensor(2.0, dtype=DTYPE))


def test_material_runtime_checkpoint_restores_field_but_not_model_initial_parameter():
    source = torch.nn.Parameter(torch.tensor(2.0, dtype=DTYPE))
    material = Material("pool", (1, 2), fields={"c": source}).train()
    material.initialize()
    handler = MechanismHandler(
        torch.tensor(37.0, dtype=DTYPE),
        torch.ones((1, 2), dtype=DTYPE),
        {},
        materials={"pool": material},
    )
    runtime = handler.mutable_state_dict()

    with torch.no_grad():
        source.fill_(7.0)
        material.c.fill_(11.0)
    handler.restore_mutable_state_dict(runtime)
    torch.testing.assert_close(material.c, torch.full_like(material.c, 2.0))
    torch.testing.assert_close(source, torch.tensor(7.0, dtype=DTYPE))

    material.initialize()
    torch.testing.assert_close(material.c, torch.full_like(material.c, 7.0))


def test_fixed_material_initial_remains_nondifferentiable_in_train_and_eval():
    material = Material("pool", (2, 2), fields={"c": torch.tensor([1.0, 2.0])})
    source = material.initial_source("c")
    assert isinstance(source, torch.nn.Parameter)
    assert not source.requires_grad

    material.train().initialize()
    assert not material.c.requires_grad
    torch.testing.assert_close(material.c, torch.tensor([[1.0, 2.0]]).expand(2, 2))
    material.c.fill_(9.0)
    material.eval().initialize()
    assert not material.c.requires_grad
    assert material.c.grad_fn is None
    torch.testing.assert_close(material.c, torch.tensor([[1.0, 2.0]]).expand(2, 2))


def test_ion_keeps_dedicated_initial_parameters_without_generic_duplicates():
    ion = Ion("na", (1, 2), einit=1, eadvance=1)

    state_names = set(ion.state_dict())
    assert {"e_init", "i_init", "o_init"} <= state_names
    assert not any(name.startswith("_initial_sources.") for name in state_names)
    assert not ion._material_has_initial_sources
    with pytest.raises(KeyError, match="no generic initial source"):
        ion.initial_source("nai")


def test_trainable_material_initial_created_by_context_is_exposed():
    with dn.ctx(REQUIRE_GRAD=1):
        material = Material("pool", (1, 1), fields={"c": 3.0})

    assert material.initial_source("c").requires_grad
    material.train().initialize()
    material.c.sum().backward()
    assert material.initial_source("c").grad is not None


def test_population_lazy_build_propagates_eval_and_train_modes_to_material():
    eval_source = torch.nn.Parameter(torch.tensor(2.0, dtype=DTYPE))
    eval_population = dn.Population(N=1, C=2, dtype=DTYPE)
    eval_population.material("pool", fields={"c": eval_source})

    assert not eval_population.training
    eval_population.initialize()
    eval_material = eval_population.mech.materials["pool"]
    assert not eval_population.integrator.training
    assert not eval_population.mech.training
    assert not eval_material.training
    assert not eval_material.c.requires_grad
    assert eval_material.c.grad_fn is None

    train_source = torch.nn.Parameter(torch.tensor(3.0, dtype=DTYPE))
    train_population = dn.Population(N=1, C=2, dtype=DTYPE)
    train_population.material("pool", fields={"c": train_source})
    train_population.train()
    train_population.initialize()
    train_material = train_population.mech.materials["pool"]
    assert train_population.integrator.training
    assert train_population.mech.training
    assert train_material.training
    assert train_material.c.requires_grad
    train_material.c.sum().backward()
    torch.testing.assert_close(train_source.grad, torch.tensor(2.0, dtype=DTYPE))
