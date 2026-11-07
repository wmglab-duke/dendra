"""Torch accessors and source bindings retain their numerical contract."""

from __future__ import annotations

import copy
import inspect
from types import MethodType

import pytest
import torch
from torch.nn.utils import parametrize

import dendra as dn
from dendra.func._torch_parametrizations import source_binding_signature
from dendra.models.mechanisms import Mechanism
from tests.func.test_torch_parametrizations import (
    DT,
    DTYPE,
    _Affine,
    _population,
    _refresh,
)

pytestmark = pytest.mark.cpu


class _Sampled(Mechanism):
    Mechanism.GLOBALRAND("draw", mu=0.1, sigma=0.01, seed=17)
    Mechanism.GLOBALNOISE("noise", mu=0.1, sigma=0.01, seed=23)


@pytest.fixture(autouse=True)
def _context():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu", CHECK_PURITY=False):
        yield


def _model():
    population = _population()
    parametrize.register_parametrization(population, "cm", _Affine())
    _refresh(population)
    return population


def _replace_property(population, accessor, monkeypatch):
    descriptor = inspect.getattr_static(type(population), "cm")
    if accessor == "getter":

        def getter(self):
            return 2 * descriptor.fget(self)

        replacement = property(getter, descriptor.fset, descriptor.fdel)
    else:

        def setter(self, value):
            return descriptor.fset(self, value / 2)

        replacement = property(descriptor.fget, setter, descriptor.fdel)
    monkeypatch.setattr(type(population), "cm", replacement)


@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("accessor", ["getter", "setter"])
def test_lowering_rejects_replaced_generated_accessor(checked, accessor, monkeypatch):
    population = _model()
    _replace_property(population, accessor, monkeypatch)
    with dn.ctx(CHECK_PURITY=checked):
        with pytest.raises(dn.func.FunctionalizationError, match="canonical.*property"):
            dn.func.make_functional(population, dt=DT)


def test_canonical_accessor_remains_valid_after_native_deepcopy():
    # PyTorch's getter closure retains its original owner as the optional
    # process-global cache key. It does not rebind that closure on deepcopy.
    population = copy.deepcopy(_model())
    functional, tensors = dn.func.make_functional(population, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    torch.testing.assert_close(prepared.values["population"]["cm"], population.cm)


@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("slot", ["original", "offset"])
def test_torch_original_and_config_cannot_capture_runtime_buffers(checked, slot):
    population = _model()
    transforms = population.parametrizations.cm
    if slot == "original":
        transforms.original = population.v
    else:
        transforms[0].offset = population.v
    with dn.ctx(CHECK_PURITY=checked):
        with pytest.raises(dn.func.FunctionalizationError, match="alias runtime"):
            dn.func.make_functional(population, dt=DT)


def test_lowering_rejects_mutated_native_list_adapter(monkeypatch):
    population = _model()
    native = parametrize.ParametrizationList.forward

    def alternate(self):
        return native(self) * 2

    monkeypatch.setattr(parametrize.ParametrizationList, "forward", alternate)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical list adapter"):
        dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("target", ["material", "draw", "noise", "workspace"])
def test_native_writable_runtime_targets_fail_with_clear_admission_error(
    checked, target
):
    if target == "workspace":
        population = _population()
        owner, name = population.integrator, "cmdt"
    else:
        population = dn.SingleCompartment(
            N=1, C=2, integrator=dn.bwd_euler_sc(imem=False)
        )
        if target == "material":
            population.material("pool", fields={"amount": 2.0})
        else:
            population.insert(_Sampled)
        population.initialize()
        if target == "material":
            owner, name = population.mech.materials["pool"], "amount"
        else:
            owner, name = next(iter(population.mech.mechanisms.values())), target
    # These are valid native registrations. Their normal simulation adapter
    # writes the effective target through its inverse, outside the fixed pure
    # materialization contract admitted by the functional parameter adapter.
    parametrize.register_parametrization(owner, name, torch.nn.Identity())
    assert torch.is_tensor(getattr(owner, name))
    with dn.ctx(CHECK_PURITY=checked):
        with pytest.raises(
            dn.func.FunctionalizationError, match="mutable state|workspace"
        ):
            dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize(
    "change", ["getter", "setter", "forward", "schema", "hook", "alias"]
)
def test_warmed_compiled_binding_guards_reject_changed_sources(change, monkeypatch):
    population = _model()
    transform = population.parametrizations.cm[0]
    if change == "alias":
        transform.register_buffer("mirror", transform.offset)
    expected = source_binding_signature(population)

    def evaluate(value):
        if source_binding_signature(population) != expected:
            raise dn.func.FunctionalizationError(
                "Torch parametrization bindings changed"
            )
        return value + 1

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(torch.zeros(2)), torch.ones(2))
    if change in {"getter", "setter"}:
        _replace_property(population, change, monkeypatch)
    elif change == "forward":
        transform.forward = MethodType(lambda self, value: value * 2, transform)
    elif change == "schema":
        population.parametrizations.cm.original = population.cm.new_ones((1, 3))
    elif change == "hook":
        transform.register_forward_hook(lambda self, arguments, result: result)
    else:
        transform.mirror = transform.offset.clone()
    with pytest.raises(
        (RuntimeError, dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)
    ):
        compiled(torch.zeros(2))


@pytest.mark.parametrize("accessor", ["getter", "setter"])
def test_warmed_compiled_population_guards_generated_accessor(accessor, monkeypatch):
    population = _model()
    functional, tensors = dn.func.make_functional(population, dt=DT)

    def evaluate(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)[0][
            "integrator"
        ]["v"]

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state)
    _replace_property(population, accessor, monkeypatch)
    with pytest.raises(
        (RuntimeError, dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported)
    ):
        compiled(tensors.parameters, tensors.constants, tensors.state)
