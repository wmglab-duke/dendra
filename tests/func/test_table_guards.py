"""Warmed compiled functional entry points guard the live TABLE contract."""

from types import MethodType

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._tables import table_owners
from tests.func.test_table_execution import (
    _FixedTableLeak,
    _LearnableTableLeak,
    _network,
    _owner,
)
from tests.func.test_table_plan import _population, _TableCurrent


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("operation", ("step", "rollout"))
@pytest.mark.parametrize("mutation", ("flag", "metadata", "callable", "code"))
def test_compiled_table_contract_changes_invalidate_warmed_graph(
    operation, mutation, monkeypatch
):
    population = _population()
    functional, tensors = dn.func.make_functional(population, dt=0.125)

    def evaluate(parameters, constants, state):
        if operation == "step":
            return functional.prepare_and_step(parameters, constants, state)
        return functional.prepare_and_rollout(parameters, constants, state, steps=2)

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state)
    before = pytree.tree_map_only(torch.Tensor, lambda t: t.detach().clone(), tensors)
    owner = population.mech.table
    if mutation == "flag":
        owner.usetables(False)
    elif mutation == "metadata":
        owner.curve_table.exact_clamp = True
    elif mutation == "callable":

        def replacement(self, x):
            return self.gain * x * 2

        owner.curve_original = MethodType(replacement, owner)
    else:

        def replacement(self, x):
            return x

        monkeypatch.setattr(
            type(owner.curve_table).forward, "__code__", replacement.__code__
        )
    # Dynamo reports deliberate source validation failures reached on retrace
    # as Unsupported in fullgraph mode; the source contract remains explicit.
    with pytest.raises(
        (dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported), match="TABLE"
    ):
        compiled(tensors.parameters, tensors.constants, tensors.state)
    torch.testing.assert_close(tensors, before)


@pytest.mark.parametrize("operation", ("step", "rollout"))
@pytest.mark.parametrize("kind", ("buffer", "parameter"))
@pytest.mark.parametrize("change", ("merge", "split"))
def test_compiled_table_tensor_alias_changes_invalidate_warmed_graph(
    operation, kind, change
):
    population = _population()
    if kind == "parameter":
        population.insert(_TableCurrent.rename("second"))
        population.initialize()
        left, right = [
            table
            for _path, _owner, _name, table in table_owners(population)
            if table.learnable_y
        ]
        name = "y"
    else:
        left = right = population.mech.table.curve_table
        name = "_inv_dx"
    reference = left.y if kind == "parameter" else left._x0
    if change == "split":
        setattr(right, name, reference)
    functional, tensors = dn.func.make_functional(population, dt=0.125)

    def evaluate(parameters, constants, state):
        if operation == "step":
            return functional.prepare_and_step(parameters, constants, state)
        return functional.prepare_and_rollout(parameters, constants, state, steps=2)

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    compiled(tensors.parameters, tensors.constants, tensors.state)
    before = pytree.tree_map_only(torch.Tensor, lambda t: t.detach().clone(), tensors)
    replacement = (
        reference
        if change == "merge"
        else torch.nn.Parameter(reference.detach().clone())
        if kind == "parameter"
        else reference.detach().clone()
    )
    setattr(right, name, replacement)
    with pytest.raises(dn.func.FunctionalizationError, match="TABLE"):
        functional._table_plan.validate_source()
    with pytest.raises(
        (dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported), match="TABLE"
    ):
        compiled(tensors.parameters, tensors.constants, tensors.state)
    torch.testing.assert_close(tensors, before)


def test_table_tensor_replacements_preserve_unchanged_alias_topology():
    population = _population()
    functional, tensors = dn.func.make_functional(population, dt=0.125)
    plan = functional._table_plan
    before = plan.source_binding_signature()
    compiled = torch.compile(
        functional.prepare_and_step, backend="eager", fullgraph=True
    )
    expected = compiled(tensors.parameters, tensors.constants, tensors.state)
    fixed = population.mech.table.curve_table
    fixed._inv_dx = fixed._inv_dx.detach().clone() + 0.1
    learned = next(
        table
        for _path, _owner, _name, table in table_owners(population)
        if table.learnable_y
    )
    learned.y = torch.nn.Parameter(learned.y.detach().clone() + 0.1)
    plan.validate_source()
    assert plan.source_binding_signature() == before
    torch.testing.assert_close(
        compiled(tensors.parameters, tensors.constants, tensors.state), expected
    )


@pytest.mark.parametrize("kind", ("buffer", "parameter"))
@pytest.mark.parametrize("change", ("merge", "split"))
def test_compiled_network_guards_table_tensor_aliases_between_populations(kind, change):
    network = _network(
        _LearnableTableLeak if kind == "parameter" else _FixedTableLeak,
        connected=False,
        shared=kind == "parameter" and change == "split",
    )
    left = _owner(network.left).curve_table
    right = _owner(network.right).curve_table
    name = "y" if kind == "parameter" else "_inv_dx"
    reference = getattr(left, name)
    if change == "split":
        setattr(right, name, reference)
    functional, tensors = dn.func.make_functional_network(network)
    compiled = torch.compile(
        functional.prepare_and_step, backend="eager", fullgraph=True
    )
    compiled(*tensors)
    before = pytree.tree_map_only(torch.Tensor, lambda t: t.detach().clone(), tensors)
    replacement = (
        reference
        if change == "merge"
        else torch.nn.Parameter(reference.detach().clone())
        if kind == "parameter"
        else reference.detach().clone()
    )
    setattr(right, name, replacement)
    with pytest.raises(dn.func.FunctionalizationError, match="TABLE"):
        functional._validate_source()
    with pytest.raises(
        (dn.func.FunctionalizationError, torch._dynamo.exc.Unsupported), match="TABLE"
    ):
        compiled(*tensors)
    torch.testing.assert_close(tensors, before)
