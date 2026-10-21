"""Runner selectors expose actual gradient leaves and preserve execution semantics."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_network import DT, DTYPE, _loss, _mixed_network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _population():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        model = dn.SingleCompartment(
            N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-65.0
        )
        model.insert(pas, g=0.001, e=-70.0)
        model.train()
        model.initialize()
    return model


def _compile(model, trainable, *, inference=False):
    return dn.compile(
        model,
        dt=DT,
        steps=3,
        inference=inference,
        trainable=trainable,
        backend="eager",
    )


def _detached(tree):
    if torch.is_tensor(tree):
        return tree.detach().clone()
    if isinstance(tree, Mapping):
        return {name: _detached(value) for name, value in tree.items()}
    if isinstance(tree, tuple):
        values = tuple(_detached(value) for value in tree)
        return type(tree)(*values) if hasattr(tree, "_fields") else values
    return tree


def test_population_runner_glob_regex_inspection_and_training_gradients():
    runner = _compile(_population(), ("*.pas.*_param", re.compile(r"\.pas\.g_param$")))
    selected = runner.trainable_parameters
    assert len(selected) == 2
    assert tuple(selected) == tuple(runner.find_parameters("*.pas.*_param"))
    assert tuple(runner.named_parameters) == tuple(runner.parameters)
    for name, value in selected.items():
        assert value is runner.named_parameters[name]
        assert isinstance(value, torch.nn.Parameter)
    g_name = runner.parameter_name("re:\\.pas\\.g_param$")
    assert runner.get_parameter(re.compile(r"\.pas\.g_param$")) is selected[g_name]
    with pytest.raises(TypeError):
        selected[g_name] = torch.tensor(1.0)
    with pytest.raises(KeyError, match="ambiguous"):
        runner.get_parameter("*.pas.*_param")
    final, _ = runner()
    gradients = torch.autograd.grad(
        final["integrator"]["v"].square().sum(), tuple(selected.values())
    )
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients
    )


@pytest.mark.parametrize("kind", ("population", "network"))
def test_runner_inspection_tracks_live_freezing_and_ignores_call_overrides(kind):
    model = _population() if kind == "population" else _mixed_network()
    selector = "*.pas.*_param" if kind == "population" else "synapses.*.weights"
    runner = _compile(model, selector)
    selected = tuple(runner.trainable_parameters)
    value = runner.trainable_parameters[selected[0]]
    value.requires_grad_(False)
    assert selected[0] not in runner.trainable_parameters
    value.requires_grad_(True)
    assert tuple(runner.trainable_parameters) == selected
    assert runner.find_parameters("re:^absent$") == {}
    with pytest.raises(TypeError):
        runner.named_parameters[selected[0]] = value.clone()
    # An explicit override is invocation-local and does not replace the runner's
    # inspectable independent defaults or their gradient flags.
    override = _detached(runner.parameters)
    runner(parameters=override)
    assert tuple(runner.trainable_parameters) == selected
    assert runner.trainable_parameters[selected[0]] is value


def test_network_alias_selection_reports_every_alias_and_accumulates_one_gradient():
    model = _mixed_network()
    connections = tuple(model.synapses.values())
    connections[1].weight.w = connections[0].weight.w
    names = tuple(f"synapses.{name}.weights" for name in model.synapses)
    runner = _compile(model, re.compile(re.escape(names[1]) + "$"))
    assert tuple(runner.trainable_parameters) == names
    first, second = runner.trainable_parameters.values()
    assert first is second
    assert runner.parameter_name("glob:" + names[1]) == names[1]
    assert tuple(runner.find_parameters("synapses.*.weights")) == names
    with pytest.raises(KeyError, match="ambiguous"):
        runner.parameter_name("synapses.*.weights")
    final, _ = runner()
    gradient = torch.autograd.grad(_loss(final), first)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    first.requires_grad_(False)
    assert not runner.trainable_parameters


@pytest.mark.parametrize("kind", ("population", "network"))
def test_inference_reports_selected_leaves_but_suppresses_execution_gradients(kind):
    model = _population() if kind == "population" else _mixed_network()
    selector = "*.pas.*_param" if kind == "population" else "synapses.*.weights"
    runner = _compile(model, selector, inference=True)
    assert len(runner.trainable_parameters) == 2
    final, _ = runner()
    voltage = (
        final["integrator"]["v"]
        if kind == "population"
        else final["populations"]["cell"]["integrator"]["v"]
    )
    assert not voltage.requires_grad


@pytest.mark.parametrize("kind", ("population", "network"))
def test_unmatched_trainable_patterns_fail_without_silently_training_nothing(kind):
    model = _population() if kind == "population" else _mixed_network()
    with pytest.raises(KeyError, match="no .*parameter"):
        _compile(model, "re:^absent$")
