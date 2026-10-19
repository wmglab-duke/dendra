"""Checkpoint saves every tensor operand rather than mutable nested containers."""

from collections import OrderedDict
from types import MappingProxyType

import pytest
import torch
from test_compiled_chunk import DT, GNABAR, _clone_mapping, _clone_state, _model

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._runners import _materialize_checkpoint_tree

pytestmark = pytest.mark.cpu


class _GainEnergy(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "nested": (torch.full_like(voltage, 1.25),),
            "energy": voltage.new_zeros(()),
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "nested": carry["nested"],
            "energy": carry["energy"] + (voltage * carry["nested"][0]).square().mean(),
        }, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry["energy"]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _binding():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = _clone_mapping(tensors.parameters)
    parameters[GNABAR].requires_grad_()
    state = _clone_state(tensors.state)
    state["integrator"]["v"].requires_grad_()
    prepared = functional.prepare(parameters, tensors.constants)
    return model, functional, functional.bind(parameters, prepared), state, parameters


@pytest.mark.parametrize("changed", ["state", "carry", "drive"])
def test_nested_checkpoint_operand_mutation_is_rejected_in_backward(changed):
    model, functional, bound, state, parameters = _binding()
    callbacks = functional.make_callbacks({"gain": _GainEnergy()})
    _, initial = dn.func.run(bound, state, steps=0, callbacks=callbacks)
    callback_state = initial["callbacks"].state
    drive = torch.full((4, *model.shape), 1e-10, dtype=torch.float64)
    final, auxiliary = dn.func.run(
        bound,
        state,
        steps=4,
        inputs=dn.func.RolloutInput(intra=drive),
        checkpoint_every=3,
        callbacks=callbacks,
        callback_state=callback_state,
    )
    loss = final["integrator"]["v"].square().mean() + auxiliary["callbacks"]["gain"]
    operand = {
        "state": state["integrator"]["v"],
        "carry": callback_state["gain"]["nested"][0],
        "drive": drive,
    }[changed]
    with torch.no_grad():
        operand.add_(0.125)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(loss, parameters[GNABAR])


@pytest.mark.parametrize("changed", ["parameter", "constant"])
def test_runtime_extra_nested_operands_are_version_checked(monkeypatch, changed):
    from test_stimulation import DT as EXTRA_DT
    from test_stimulation import _extra
    from test_stimulation import _model as extra_model

    model = extra_model()
    functional, tensors = dn.func.make_functional(model, dt=EXTRA_DT)
    bound = functional.bind(
        tensors.parameters, functional.prepare(tensors.parameters, tensors.constants)
    )
    extra = functional.make_extra(_extra(model.shape, requires_grad=True))
    captured = []
    original_extract = extra.extract

    def extract():
        bundle = original_extract()
        captured.append(bundle)
        return bundle

    monkeypatch.setattr(extra, "extract", extract)
    final, _ = dn.func.run(
        bound, tensors.state, steps=4, checkpoint_every=3, extra=extra
    )
    assert len(captured) == 1
    bundle = captured[0]
    amplitude = next(
        value for name, value in bundle.parameters.items() if name.endswith(".amp")
    )
    loss = final["integrator"]["v"].square().mean()
    operand = (
        amplitude
        if changed == "parameter"
        else next(
            value for name, value in bundle.constants.items() if name.endswith(".field")
        )
    )
    with torch.no_grad():
        operand.add_(0.125)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(loss, amplitude)


def test_checkpoint_receives_flat_materialized_leaves_and_preserves_callback_plan(
    monkeypatch,
):
    import dendra.func._runners as runners

    _, functional, bound, state, parameters = _binding()
    callbacks = functional.make_callbacks({"gain": _GainEnergy()})
    _, initial = dn.func.run(bound, state, steps=0, callbacks=callbacks)
    callback_state = initial["callbacks"].state
    original = runners.checkpoint
    widths = []

    def checkpoint(function, *args, **keywords):
        assert args
        assert all(torch.is_tensor(value) or value is None for value in args)
        assert not any(
            torch.is_tensor(value) and torch.is_inference(value) for value in args
        )
        widths.append(len(args))
        return original(function, *args, **keywords)

    monkeypatch.setattr(runners, "checkpoint", checkpoint)
    final, auxiliary = dn.func.run(
        bound,
        state,
        steps=4,
        checkpoint_every=3,
        callbacks=callbacks,
        callback_state=callback_state,
    )
    assert auxiliary["callbacks"].state.plan_token is callback_state.plan_token
    torch.autograd.grad(
        final["integrator"]["v"].square().mean() + auxiliary["callbacks"]["gain"],
        parameters[GNABAR],
    )
    assert len(widths) == 2


def test_checkpoint_materialization_preserves_pytree_context_and_ordinary_ad_leaves():
    _, functional, _, state, _ = _binding()
    callbacks = functional.make_callbacks({"gain": _GainEnergy()})
    callback_state, _ = callbacks._initialize(state)
    ordinary = torch.tensor(2.0, requires_grad=True)
    with torch.inference_mode():
        constant = torch.tensor(3.0)
    callback_state.carries["gain"]["nested"] = (constant,)
    operands = (
        callback_state,
        OrderedDict(value=ordinary),
        MappingProxyType({"field": constant}),
    )
    materialized = _materialize_checkpoint_tree(operands)
    assert isinstance(materialized[0], dn.func.FunctionalCallbackState)
    assert materialized[0].plan_token is callback_state.plan_token
    assert materialized[0].emission_schemas == callback_state.emission_schemas
    assert isinstance(materialized[1], OrderedDict)
    assert materialized[1]["value"] is ordinary
    assert isinstance(materialized[2], dict)
    assert not torch.is_inference(materialized[2]["field"])
    assert not torch.is_inference(materialized[0]["gain"]["nested"][0])
    gradient = torch.autograd.grad(
        materialized[1]["value"] * materialized[2]["field"], ordinary
    )[0]
    assert gradient.item() == 3.0
