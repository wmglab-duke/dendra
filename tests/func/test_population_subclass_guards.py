"""Subclass source contracts remain visible to warmed compiled execution."""

from __future__ import annotations

import types

import pytest
import torch
from torch._dynamo.exc import Unsupported
from torch.nn.utils import parametrize

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from dendra.models.multi import MultiPopulation

DT = 0.01
DTYPE = torch.float64
INITIALIZATION_HELPERS = (
    "_refresh_compile_config_from_ctx",
    "_initialization_transform_argument",
    "_initialization_transform_numpy_rng_equal",
    "_preserved_stochastic_parameter_buffers",
    "make_contiguous",
    "clear_cache",
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _cell_type():
    class Cell(dn.SingleCompartment):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.insert(pas, g=4.0e-4, e=-71.0)

    return Cell


def _warm(atomic, tensors):
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(atomic, backend=backend, fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state)
    assert graphs
    return compiled


@pytest.mark.parametrize("base", (dn.Population, dn.SingleCompartment))
@pytest.mark.parametrize("helper", INITIALIZATION_HELPERS)
def test_subclasses_reject_transitive_initialization_helper_replacements(base, helper):
    class Cell(base):
        pass

    source = Cell(N=1, C=1).initialize()
    calls = []

    def unsafe(*args, **kwargs):
        del args, kwargs
        calls.append(helper)
        raise AssertionError("authored initialization helper ran during lowering")

    # Replace after native construction: rejection must not evaluate a custom
    # helper even if the model has no registered transform that uses it yet.
    setattr(Cell, helper, staticmethod(unsafe))
    with pytest.raises(dn.func.FunctionalizationError, match=helper):
        dn.func.make_functional(source, dt=DT)
    assert not calls


@pytest.mark.parametrize("binding", ("class", "instance"))
@pytest.mark.parametrize("helper", INITIALIZATION_HELPERS)
def test_warmed_subclass_rejects_changed_transitive_initialization_helper(
    helper, binding
):
    Cell = _cell_type()
    source = Cell(N=1, C=1).initialize()
    functional, tensors = dn.func.make_functional(source, dt=DT)

    def atomic(parameters, constants, state):
        del state
        return functional.initialize(parameters, constants, tensors.initialization)

    compiled = _warm(atomic, tensors)
    calls = []

    def unsafe(*args, **kwargs):
        del args, kwargs
        calls.append(helper)
        raise AssertionError("authored initialization helper ran after lowering")

    if binding == "class":
        setattr(Cell, helper, staticmethod(unsafe))
    else:
        object.__setattr__(source, helper, unsafe)
    for initialize in (atomic, compiled):
        with pytest.raises(
            (dn.func.FunctionalizationError, Unsupported), match="canonical execution"
        ):
            initialize(tensors.parameters, tensors.constants, tensors.state)
    assert not calls


@pytest.mark.parametrize("binding", ["custom", "foreign_receiver", "unbound"])
def test_warmed_subclass_rejects_instance_method_replacement(binding):
    Cell = _cell_type()
    source = Cell(N=1, C=1).initialize()
    other = Cell(N=1, C=1).initialize()
    functional, tensors = dn.func.make_functional(source, dt=DT)

    def atomic(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)

    compiled = _warm(atomic, tensors)
    calls = []

    def unsafe(self, *args, **kwargs):
        del self, args, kwargs
        calls.append("step")
        raise AssertionError("authored timestep replacement ran")

    method = {
        "custom": types.MethodType(unsafe, source),
        "foreign_receiver": other.step,
        "unbound": dn.SingleCompartment.step,
    }[binding]
    object.__setattr__(source, "step", method)
    for advance in (atomic, compiled):
        with pytest.raises(
            (dn.func.FunctionalizationError, Unsupported), match="canonical execution"
        ):
            advance(tensors.parameters, tensors.constants, tensors.state)
    assert not calls
    assert source.t.item() == other.t.item() == 0.0


@pytest.mark.parametrize("initialization", [False, True], ids=["snapshot", "fresh"])
def test_warmed_packed_subclass_rejects_changed_component_materialization(
    initialization,
):
    Cell = _cell_type()

    class Bundle(MultiPopulation):
        def __init__(self):
            super().__init__(left=Cell(N=1, C=1), right=Cell(N=1, C=1))

    source = Bundle().initialize()
    functional, tensors = dn.func.make_functional(source, dt=DT)

    def atomic(parameters, constants, state):
        if initialization:
            return functional.initialize(parameters, constants, tensors.initialization)
        return functional.prepare_and_step(parameters, constants, state)

    compiled = _warm(atomic, tensors)
    calls = []

    def unsafe(self, *args, **kwargs):
        del self, args, kwargs
        calls.append("populate")
        raise AssertionError("authored materialization replacement ran")

    component = source.populations["left"]
    object.__setattr__(
        component, "populate_parameter_buffers", types.MethodType(unsafe, component)
    )
    for advance in (atomic, compiled):
        with pytest.raises(
            (dn.func.FunctionalizationError, Unsupported), match="canonical execution"
        ):
            advance(tensors.parameters, tensors.constants, tensors.state)
    assert not calls
    assert source.t.item() == 0.0


@pytest.mark.parametrize("descriptor", [False, True], ids=["none", "property"])
def test_warmed_subclass_rejects_reserved_runtime_attribute_masking(descriptor):
    Cell = _cell_type()
    source = Cell(N=1, C=1).initialize()
    functional, tensors = dn.func.make_functional(source, dt=DT)

    def atomic(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)

    compiled = _warm(atomic, tensors)
    calls = []

    def unsafe(self):
        del self
        calls.append("clock")
        raise AssertionError("authored clock descriptor ran")

    Cell.t = property(unsafe) if descriptor else None
    for advance in (atomic, compiled):
        with pytest.raises(
            (dn.func.FunctionalizationError, Unsupported), match="canonical execution"
        ):
            advance(tensors.parameters, tensors.constants, tensors.state)
    assert not calls


def test_parametrized_subclass_inductor_voltage_and_gradients_match_shared_eager_path():
    class Scale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.1, dtype=DTYPE))

        def forward(self, value):
            return self.scale * value

    Cell = _cell_type()
    source = Cell(N=2, C=1, v_init=-64.0).initialize()
    parametrize.register_parametrization(source, "cm", Scale())
    source.populate_parameter_buffers()
    source.initialize().train()
    functional, tensors = dn.func.make_functional(source, dt=DT)

    def atomic(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)

    expected, _ = atomic(tensors.parameters, tensors.constants, tensors.state)
    compiled = torch.compile(atomic, backend="inductor", fullgraph=True)
    actual, _ = compiled(tensors.parameters, tensors.constants, tensors.state)
    torch.testing.assert_close(
        actual["integrator"]["v"], expected["integrator"]["v"], rtol=1e-11, atol=1e-11
    )
    parameters = tuple(tensors.parameters.values())
    actual_gradients = torch.autograd.grad(
        actual["integrator"]["v"].square().sum(), parameters, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(
        expected["integrator"]["v"].square().sum(), parameters, allow_unused=True
    )
    nonzero = 0
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert (actual_gradient is None) == (expected_gradient is None)
        if actual_gradient is not None:
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=1e-10, atol=1e-10
            )
            nonzero += bool(torch.count_nonzero(actual_gradient))
    assert nonzero >= 2
    assert parametrize.type_before_parametrizations(source) is Cell
    assert (
        parametrize.type_before_parametrizations(functional._transition.population)
        is Cell
    )
