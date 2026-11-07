"""Raw bindings preserve authored-boundary checks and PyTorch transforms."""

from __future__ import annotations

from functools import partial

import pytest
import torch

import dendra as dn
from tests.func.test_raw_bindings import DT, _case, _close
from tests.func.test_raw_bindings import _environment as _environment

pytestmark = pytest.mark.cpu


def test_raw_binding_duration_uses_live_prepared_dt_and_rejects_conflicting_clock():
    _, functional, parameters, constants, state, _ = _case()
    bound = functional.bind(parameters, constants=constants)
    with torch.no_grad():
        dn.func.run(bound, state, steps=0)
        previous = bound._inference_prepared
        constants["dt"].mul_(2)
        actual, _ = dn.func.run(bound, state, duration=2 * DT)
        expected, _ = functional.prepare_and_rollout(
            parameters, constants, state, steps=1
        )
        _close(actual, expected)
        assert bound._inference_prepared is not previous
        with pytest.raises(
            dn.func.FunctionalizationError, match="does not match runner dt"
        ):
            dn.func.run(bound, state, steps=0, dt=DT)


@pytest.mark.parametrize("compiled", (False, True))
def test_raw_binding_inside_torch_func_preserves_grad_and_vmap(compiled):
    _, functional, base_parameters, constants, state, name = _case()
    compile_calls = []

    def backend(graph, inputs):
        compile_calls.append(graph)
        return graph.forward

    owner = (
        functional.compile_rollout_chunk(2, backend=backend) if compiled else functional
    )
    width = 2 if compiled else 1

    def evaluate(conductance, *, bound):
        parameters = dict(base_parameters)
        parameters[name] = conductance
        if bound:
            result, _ = owner.bind(parameters, constants=constants)(state)
        else:
            result, _ = functional.prepare_and_rollout(
                parameters, constants, state, steps=width
            )
        return result["integrator"]["v"].square().mean()

    actual = partial(evaluate, bound=True)
    expected = partial(evaluate, bound=False)
    base = base_parameters[name]
    reference = torch.func.grad(expected)(base)
    assert reference.abs().sum() > 0
    _close(torch.func.grad(actual)(base), reference)
    lanes = torch.stack((0.9 * base, 1.1 * base))
    _close(torch.func.vmap(actual)(lanes), torch.func.vmap(expected)(lanes))
    # The compiled binding uses the existing eager transform path.
    assert not compile_calls


@pytest.mark.parametrize("compiled", (False, True))
def test_raw_binding_rechecks_source_after_authored_prepare_and_disables_its_cache(
    monkeypatch, compiled
):
    model, functional, parameters, constants, state, _ = _case()
    owner = (
        functional.compile_rollout_chunk(2, backend="eager") if compiled else functional
    )
    bound = owner.bind(parameters, constants=constants)
    original = functional.prepare
    calls = 0

    def authored_prepare(parameters, constants):
        nonlocal calls
        calls += 1
        prepared = original(parameters, constants)
        if calls == 2:
            model._flag_rebuild = True
        return prepared

    monkeypatch.setattr(functional, "prepare", authored_prepare)
    with torch.no_grad():
        bound(state)
        assert bound._inference_prepared is None
        with pytest.raises(
            dn.func.FunctionalizationError, match="pending structural changes"
        ):
            bound(state)
    assert calls == 2


@pytest.mark.parametrize("runner", ("run", "longrun", "longrun_checkpointed"))
def test_legacy_population_scheduling_adapts_raw_binding_once_per_forward(
    monkeypatch, runner
):
    _, functional, parameters, constants, state, name = _case()
    constants["diam"].requires_grad_()
    targets = (parameters[name], constants["diam"])
    bound = functional.bind(parameters, constants=constants)
    count = 0
    prepare = functional._prepare_source_validated

    def observe(*arguments):
        nonlocal count
        count += 1
        return prepare(*arguments)

    monkeypatch.setattr(functional, "_prepare_source_validated", observe)
    expected, _ = dn.func.run(bound, state, steps=5)
    loss = expected["integrator"]["v"].square().mean()
    reference = torch.autograd.grad(loss, targets)
    for iteration in range(2):
        if runner == "run":
            actual, _ = dn.func.run(functional, bound, state, tstop=5 * DT)
        else:
            actual, _ = getattr(dn.func, runner)(functional, bound, state, 5 * DT, 3)
        gradient = torch.autograd.grad(
            actual["integrator"]["v"].square().mean(), targets
        )
        _close(actual, expected)
        _close(gradient, reference)
        assert count == iteration + 2


def test_legacy_raw_binding_wrapper_retains_authored_calls_and_clock_validation():
    _, functional, parameters, constants, state, name = _case()
    bound = functional.bind(parameters, constants=constants)
    calls = []

    class ObservedPartial(partial):
        def __call__(self, *arguments, **keywords):
            calls.append((arguments, keywords))
            return super().__call__(*arguments, **keywords)

    wrapped = ObservedPartial(bound)
    expected, _ = dn.func.run(bound, state, steps=3)
    reference = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(), parameters[name]
    )[0]
    actual, _ = dn.func.run(functional, wrapped, state, tstop=3 * DT)
    _close(actual, expected)
    _close(
        torch.autograd.grad(
            actual["integrator"]["v"].square().mean(), parameters[name]
        )[0],
        reference,
    )
    assert len(calls) == 3
    with pytest.raises(
        dn.func.FunctionalizationError, match="does not match runner dt"
    ):
        dn.func.run(functional, wrapped, state, tstop=4 * DT, dt=2 * DT)
    assert len(calls) == 3
