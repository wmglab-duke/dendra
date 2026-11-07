"""Initialization diagnostics independently probe previous State and IC."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_initialization_previous_state import PROBE, X, _model

pytestmark = pytest.mark.cpu


class _BatchedPrevious(torch.nn.Module):
    def forward(self, voltage, previous):
        if torch._C._functorch.is_batchedtensor(previous):
            return (voltage + 1.0,)
        return (voltage,)


class _LeafPrevious(torch.nn.Module):
    def forward(self, voltage, previous):
        if previous.is_leaf:
            return (voltage,)
        return (voltage + 1.0,)


class _PurePrevious(torch.nn.Module):
    def forward(self, voltage, previous):
        return (voltage + previous.square(),)


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(
            JIT=0,
            REQUIRE_GRAD=1,
            IMEM=False,
            DEVICE="cpu",
            DTYPE=torch.float64,
            CHECK_PURITY=False,
        ),
    ):
        yield


def _lower(transform, *, check_purity):
    model = _model()
    model.register_pre_initialize_transform(
        "observe_context",
        transform,
        reads=("state.integrator.v", f"state.{X}"),
        writes=("state.integrator.v",),
    )
    with dn.ctx(CHECK_PURITY=check_purity):
        return dn.func.make_functional(model, dt=0.025)


@pytest.mark.parametrize(
    ("transform", "message"),
    (
        (_BatchedPrevious, "active torch.func vmap execution context"),
        (_LeafPrevious, "autograd metadata or grad-enabled execution context"),
    ),
    ids=("batched", "leaf"),
)
def test_previous_state_context_disables_initialization_but_retains_transition(
    transform, message
):
    functional, tensors = _lower(transform(), check_purity=True)
    with dn.ctx(CHECK_PURITY=False):
        with pytest.raises(dn.func.FunctionalizationError, match=message):
            functional.initialize(
                tensors.parameters, tensors.constants, tensors.initialization
            )
        state, _auxiliary = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
    assert torch.isfinite(state["integrator"]["v"]).all()


def test_pure_previous_state_remains_differentiable_under_diagnostics():
    functional, tensors = _lower(_PurePrevious(), check_purity=True)
    previous = {
        name: value.detach().clone().requires_grad_()
        for name, value in tensors.state["mechanisms"][PROBE].items()
    }
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state={"mechanisms": {PROBE: previous}},
    )
    torch.testing.assert_close(
        initialized.state["integrator"]["v"],
        torch.full_like(initialized.state["integrator"]["v"], -51.04),
    )
    gradient = torch.autograd.grad(
        initialized.state["integrator"]["v"].sum(), previous["x"]
    )[0]
    torch.testing.assert_close(gradient, torch.full_like(gradient, 3.8))


def test_disabled_diagnostics_trust_previous_state_context_with_captured_policy():
    functional, tensors = _lower(_LeafPrevious(), check_purity=False)
    previous = {
        name: value.detach().clone().requires_grad_()
        for name, value in tensors.state["mechanisms"][PROBE].items()
    }
    with dn.ctx(CHECK_PURITY=True):
        initialized = functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state={"mechanisms": {PROBE: previous}},
        )
    torch.testing.assert_close(
        initialized.state["integrator"]["v"],
        torch.full_like(initialized.state["integrator"]["v"], -52.0),
    )
