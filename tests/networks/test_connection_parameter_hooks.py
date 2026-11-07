"""Native source materialization keeps authored module call semantics."""

from contextlib import nullcontext
from types import MethodType

import pytest
import torch

from dendra.models.networks.net import make_weight
from dendra.models.parametric import PositiveParam

pytestmark = pytest.mark.cpu


def _source():
    return PositiveParam(torch.tensor(1.0, dtype=torch.float64), requires_grad=True)


def test_connection_parameter_local_forward_hook_changes_materialized_value():
    source = _source()
    handle = source.register_forward_hook(lambda _module, _args, value: value * 2)
    try:
        weights = make_weight([source], [2]).init()
        torch.testing.assert_close(weights.w, source.rho.new_tensor([2.0, 2.0]))
    finally:
        handle.remove()


def test_connection_parameter_local_pre_hook_runs_before_materialization():
    source = _source()
    calls = []
    handle = source.register_forward_pre_hook(lambda _module, _args: calls.append(1))
    try:
        weights = make_weight([source], [2]).init()
        torch.testing.assert_close(weights.w, source.rho.new_tensor([1.0, 1.0]))
        assert calls == [1]
    finally:
        handle.remove()


@pytest.mark.parametrize("pre", [False, True])
def test_connection_parameter_local_backward_hooks_keep_source_gradient(pre):
    source = _source()
    expected = torch.autograd.grad(source().sum(), source.rho)[0]
    if pre:
        handle = source.register_full_backward_pre_hook(
            lambda _module, grad_output: tuple(value * 3 for value in grad_output)
        )
    else:
        handle = source.register_full_backward_hook(
            lambda _module, grad_input, grad_output: grad_input
        )
    calls = []
    observer = source.register_full_backward_hook(
        lambda _module, _grad_input, _grad_output: calls.append(1)
    )
    try:
        weights = make_weight([source], [2]).init()
        with pytest.warns(UserWarning, match="Full backward hook is firing"):
            gradient = torch.autograd.grad(weights.w.sum(), source.rho)[0]
        torch.testing.assert_close(gradient, expected * (6 if pre else 2))
        assert calls == [1]
    finally:
        observer.remove()
        handle.remove()


def test_connection_parameter_global_forward_hook_keeps_source_call_boundary():
    source = _source()

    def scale_parameter(module, _args, value):
        return value * 2 if module is source else None

    handle = torch.nn.modules.module.register_module_forward_hook(scale_parameter)
    try:
        weights = make_weight([source], [2]).init()
        torch.testing.assert_close(weights.w, source.rho.new_tensor([2.0, 2.0]))
    finally:
        handle.remove()


def test_connection_parameter_global_pre_hook_observes_source_call():
    source = _source()
    calls = []

    def observe_source(module, _args):
        if module is source:
            calls.append(1)

    handle = torch.nn.modules.module.register_module_forward_pre_hook(observe_source)
    try:
        make_weight([source], [2]).init()
        assert calls == [1]
    finally:
        handle.remove()


@pytest.mark.parametrize("pre", [False, True])
def test_connection_parameter_global_backward_hooks_keep_source_call(pre):
    source = _source()
    expected = torch.autograd.grad(source().sum(), source.rho)[0]
    calls = []
    if pre:

        def observe_source(module, grad_output):
            if module is source:
                calls.append(1)
                return tuple(value * 3 for value in grad_output)
            return None

        handle = torch.nn.modules.module.register_module_full_backward_pre_hook(
            observe_source
        )
    else:

        def observe_source(module, _grad_input, _grad_output):
            if module is source:
                calls.append(1)

        handle = torch.nn.modules.module.register_module_full_backward_hook(
            observe_source
        )
    try:
        weights = make_weight([source], [2]).init()
        warning = (
            nullcontext()
            if pre
            else pytest.warns(UserWarning, match="Full backward hook is firing")
        )
        with warning:
            gradient = torch.autograd.grad(weights.w.sum(), source.rho)[0]
        torch.testing.assert_close(gradient, expected * (6 if pre else 2))
        assert calls == [1]
    finally:
        handle.remove()


@pytest.mark.parametrize("method", ["forward", "repeat"])
def test_connection_parameter_instance_overrides_remain_authoritative(method):
    source = _source()
    if method == "forward":
        source.forward = MethodType(lambda self: self._compute() * 3, source)
    else:
        source.repeat = MethodType(
            lambda self, n: self._compute().repeat(n) * 3, source
        )
    weights = make_weight([source], [2]).init()
    torch.testing.assert_close(weights.w, source.rho.new_tensor([3.0, 3.0]))
