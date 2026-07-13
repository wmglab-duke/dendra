"""Contracts for reusable parameter-selection helpers."""

import pytest
import torch

from dendra.models.modular import DNModule


class _Leaf(DNModule):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(2.0))


class _Tree(DNModule):
    def __init__(self):
        super().__init__()
        self.trunk = _Leaf()
        self.head = _Leaf()
        self.spare = torch.nn.Parameter(torch.tensor(3.0))


def _grad_flags(module):
    return {
        name: parameter.requires_grad for name, parameter in module.named_parameters()
    }


def test_collect_without_patterns_yields_every_parameter_in_module_order():
    module = _Tree()

    assert list(module.collect_parameters()) == list(module.parameters())
    assert list(module.collect_named_parameters()) == list(module.named_parameters())


def test_collect_patterns_select_matching_parameters_and_empty_matches():
    module = _Tree()

    selected = list(module.collect_named_parameters("trunk.a", "head.beta"))
    assert [name for name, _ in selected] == ["trunk.alpha", "head.beta"]
    assert list(module.collect_parameters("missing")) == []
    assert list(module.collect_named_parameters("missing")) == []


def test_freeze_and_unfreeze_every_parameter_without_patterns():
    module = _Tree()

    assert module.freeze() is module
    assert not any(_grad_flags(module).values())
    assert module.unfreeze() is module
    assert all(_grad_flags(module).values())


@pytest.mark.parametrize("operation", ["freeze", "unfreeze"])
def test_global_operation_honors_exclusion_patterns(operation):
    module = _Tree()
    initial = operation == "freeze"
    for parameter in module.parameters():
        parameter.requires_grad = initial

    result = getattr(module, operation)(exclude=("head",))

    assert result is module
    expected_selected = operation == "unfreeze"
    assert _grad_flags(module) == {
        "spare": expected_selected,
        "trunk.alpha": expected_selected,
        "trunk.beta": expected_selected,
        "head.alpha": initial,
        "head.beta": initial,
    }


@pytest.mark.parametrize("operation", ["freeze", "unfreeze"])
def test_named_operation_selects_patterns_and_honors_exclusions(operation):
    module = _Tree()
    initial = operation == "freeze"
    for parameter in module.parameters():
        parameter.requires_grad = initial

    result = getattr(module, operation)("trunk", exclude=("trunk.beta",))

    assert result is module
    expected_selected = operation == "unfreeze"
    assert _grad_flags(module) == {
        "spare": initial,
        "trunk.alpha": expected_selected,
        "trunk.beta": initial,
        "head.alpha": initial,
        "head.beta": initial,
    }


def test_inplace_aliases_delegate_and_keep_historical_none_return():
    module = _Tree()

    assert module.freeze_("trunk.alpha") is None
    assert not module.trunk.alpha.requires_grad
    assert module.unfreeze_("trunk.alpha") is None
    assert module.trunk.alpha.requires_grad
