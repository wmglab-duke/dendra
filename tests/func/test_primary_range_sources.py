"""Primary spatial RANGE sources use the native broadcast materializer."""

from copy import deepcopy

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import pas

DT = 0.025
G = "integrator.mech.mechanisms.pas.g_param"
OTHER_G = "integrator.mech.mechanisms.other.g_param"
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        yield


def _model(source):
    model = dn.SingleCompartment(N=2, C=3, v_init=-65.0)
    model.insert(pas, g=source, e=-70.0)
    model.insert(pas.rename("other"), g=source, e=-72.0)
    model.train()
    model.initialize(DT)
    model.detach()
    return model


@pytest.mark.parametrize("shape", ((), (3,), (1, 3), (2, 1), (2, 3)))
def test_spatial_primary_range_fresh_snapshot_aliases_gradients_and_checkpoint(shape):
    source = torch.nn.Parameter(torch.full(shape, 0.001))
    model = _model(source)
    assert model.mech.pas.g_param is model.mech.other.g_param is source
    before = {key: value.clone() for key, value in model.state_dict().items()}
    native = deepcopy(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert G in tensors.parameters and OTHER_G not in tensors.parameters
    entry = next(
        entry
        for entry in dn.func.inspect_sources(model).entries
        if entry.path == f"parameters.{G}"
    )
    assert f"parameters.{OTHER_G}" in entry.aliases
    fresh = dn.func.initialize(functional, tensors)
    torch.testing.assert_close(fresh.state, tensors.state)
    state = tensors.state
    bound = functional.bind(tensors)
    for index in range(3):
        state, _ = bound(state)
        native.step(DT)
        torch.testing.assert_close(state, functional.extract(native).state)
        if index == 0:
            checkpoint = dn.func.export_continuation(
                functional,
                dn.func.Continuation(state, tensors.parameters, tensors.constants),
            )
            resumed = dn.func.restore_continuation(functional, checkpoint)
            assert resumed.parameters.keys() == tensors.parameters.keys()
            assert resumed.parameters[G] is not tensors.parameters[G]
            resumed_bound = functional.bind(resumed)
        else:
            resumed_state, _ = resumed_bound(resumed.state)
            resumed = dn.func.Continuation(
                resumed_state, resumed.parameters, resumed.constants
            )
            torch.testing.assert_close(state, resumed.state)
    actual_gradient = torch.autograd.grad(
        state["integrator"]["v"].square().sum(), tensors.parameters[G]
    )[0]
    native_gradient = torch.autograd.grad(
        native.v.square().sum(), native.mech.pas.g_param
    )[0]
    assert actual_gradient.abs().sum() > 0
    torch.testing.assert_close(actual_gradient, native_gradient)
    torch.testing.assert_close(model.state_dict(), before)


class _SpatialRate(State):
    State.STATE("x")
    State.RANGE(rate=0.5)
    State.DERIVATIVE("x' = rate")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}


class _StateLeak(Mechanism):
    Mechanism.STATE_BUNDLE(_SpatialRate)
    Mechanism.GLOBAL(g=0.001, e=-70.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * (1 + self.x) * (v - self.e)


def test_spatial_primary_state_range_preserves_fresh_values_and_gradients():
    rate = torch.nn.Parameter(torch.tensor([[0.2, 0.4, 0.6]]))
    model = dn.SingleCompartment(N=2, C=3, v_init=-65.0)
    model.insert(_StateLeak, rate=rate)
    model.train()
    model.initialize(DT)
    model.detach()
    native = deepcopy(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    fresh = dn.func.initialize(functional, tensors)
    torch.testing.assert_close(fresh.state, tensors.state)
    state = fresh.state
    bound = functional.bind(fresh)
    for _ in range(3):
        state, _ = bound(state)
        native.step(DT)
        torch.testing.assert_close(state, functional.extract(native).state)
    key = "integrator.mech.mechanisms._StateLeak.DE._SpatialRate.rate_param"
    actual = torch.autograd.grad(state["integrator"]["v"].sum(), fresh.parameters[key])[
        0
    ]
    expected = torch.autograd.grad(
        native.v.sum(), native.mech._StateLeak.DE["_SpatialRate"].rate_param
    )[0]
    assert actual.abs().sum() > 0
    torch.testing.assert_close(actual, expected)


class _GlobalLeak(Mechanism):
    Mechanism.GLOBAL(g=0.001, e=-70.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * (v - self.e)


@pytest.mark.parametrize("shape", ((4,), (3, 1), (1, 2, 3)))
def test_primary_range_rejects_shapes_that_do_not_broadcast_exactly_to_support(shape):
    model = _model(torch.nn.Parameter(torch.tensor(0.001)))
    model.mech.pas.g_param = torch.nn.Parameter(torch.full(shape, 0.001))
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"unsupported raw parameter shapes:.*pas.g_param.*broadcastable",
    ):
        dn.func.make_functional(model, dt=DT)


def test_primary_range_shared_source_must_broadcast_to_every_physical_support():
    model = dn.SingleCompartment(N=2, C=3, v_init=-65.0)
    model.insert(pas)
    model[:, :2].insert(pas.rename("partial"))
    model.initialize(DT)
    source = torch.nn.Parameter(torch.full((1, 3), 0.001))
    model.mech.pas.g_param = source
    model.mech.partial.g_param = source
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"unsupported raw parameter shapes:.*g_param.*broadcastable",
    ):
        dn.func.make_functional(model, dt=DT)


@pytest.mark.parametrize("shared_range", (False, True))
def test_primary_global_parameters_retain_scalar_contract_including_range_aliases(
    shared_range,
):
    model = dn.SingleCompartment(N=2, C=3, v_init=-65.0)
    model.insert(_GlobalLeak)
    if shared_range:
        model.insert(pas)
    model.initialize(DT)
    source = torch.nn.Parameter(torch.full((1, 3), 0.001))
    model.mech._GlobalLeak.g_param = source
    if shared_range:
        model.mech.pas.g_param = source
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"unsupported raw parameter shapes:.*g_param.*expected \(\)",
    ):
        dn.func.make_functional(model, dt=DT)
