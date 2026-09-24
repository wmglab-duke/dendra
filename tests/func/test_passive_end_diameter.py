"""Functional Myelinated geometry retains passive-end diameter overrides."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas

DT = 0.01
DTYPE = torch.float64


@pytest.fixture(autouse=True)
def _eager_models():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        yield


def _model(*, batched=False):
    model = dn.Myelinated(
        [8.0, 12.0],
        n_node=5,
        v_init=-70.0,
        dtype=DTYPE,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas, g=0.001, e=-60.0)
    dn.passive_end_nodes_(model, rhoa=None, cm=None, diam=1.5, e=-65.0)
    # The second override overlaps the first at column zero. Last write wins.
    dn.passive_end_nodes_(model[:, :3], rhoa=None, cm=None, diam=2.0, e=-65.0)
    if batched:
        model.batch(2)
    model.initialize()
    model.train()
    return model


def _override_key(tensors, index, field):
    suffix = f".diam.{index}.{field}"
    matches = [name for name in tensors.constants if name.endswith(suffix)]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("batched", [False, True])
def test_myelinated_end_diameter_preparation_and_step_match_imperative(batched):
    model = _model(batched=batched)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    for name in ("diam", "rhoa"):
        torch.testing.assert_close(
            prepared.values["population"][name], getattr(model, name), rtol=0, atol=0
        )
    diam = prepared.values["population"]["diam"]
    torch.testing.assert_close(diam[:, [0, 2]], torch.full((2, 2), 2.0, dtype=DTYPE))
    torch.testing.assert_close(diam[:, 4], torch.full((2,), 1.5, dtype=DTYPE))

    for index in (1, 2):
        for field in ("mask", "value"):
            key = _override_key(tensors, index, field)
            torch.testing.assert_close(
                tensors.constants[key],
                getattr(model.parametrizations.diam[index], field),
            )
    actual, _ = functional.step(tensors.parameters, prepared, tensors.state)
    model.run(tstop=DT, dt=DT)
    torch.testing.assert_close(
        actual["integrator"]["v"], model.v, rtol=2e-12, atol=2e-12
    )


def test_myelinated_end_diameter_constants_replace_values_and_masks_purely():
    model = _model()
    original_diam = model.diam.detach().clone()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    constants = dict(tensors.constants)
    value_key = _override_key(tensors, 2, "value")
    mask_key = _override_key(tensors, 2, "mask")
    constants[value_key] = torch.tensor(3.0, dtype=DTYPE)
    constants[mask_key] = torch.zeros_like(constants[mask_key])
    constants[mask_key][:, 1] = True
    prepared = functional.prepare(tensors.parameters, constants)
    expected = model.parametrizations.diam[0](tensors.constants["diam_original"])
    first_mask = tensors.constants[_override_key(tensors, 1, "mask")]
    expected = torch.where(
        first_mask, tensors.constants[_override_key(tensors, 1, "value")], expected
    )
    expected = torch.where(constants[mask_key], constants[value_key], expected)
    torch.testing.assert_close(prepared.values["population"]["diam"], expected)
    torch.testing.assert_close(model.diam, original_diam, rtol=0, atol=0)


def test_myelinated_diameter_source_gradient_is_zero_only_on_overridden_nodes():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    original = tensors.constants["diam_original"].detach().clone().requires_grad_()
    constants = {**tensors.constants, "diam_original": original}
    diam = functional.prepare(tensors.parameters, constants).values["population"][
        "diam"
    ]
    gradient = torch.autograd.grad(diam.sum(), original)[0]
    expected = 2 * model.noded1.detach() * original.detach() + model.noded2.detach()
    expected[:, [0, 2, 4]] = 0
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(gradient[:, [1, 3]]) == 4


def test_myelinated_end_diameter_compiles_and_composes_with_transforms():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    key = _override_key(tensors, 2, "value")
    ve = torch.linspace(-2.0, 2.0, model.v.numel(), dtype=DTYPE).reshape_as(model.v)

    def voltage(value):
        constants = {**tensors.constants, key: value}
        state, _ = functional.prepare_and_step(
            tensors.parameters, constants, tensors.state, dn.func.StepInput(ve=ve)
        )
        return state["integrator"]["v"]

    values = torch.tensor([1.8, 2.4], dtype=DTYPE)
    mapped = torch.func.vmap(voltage)(values)
    torch.testing.assert_close(
        mapped, torch.stack([voltage(value) for value in values])
    )
    compiled = torch.compile(voltage, backend="aot_eager", fullgraph=True)
    for candidate in values:
        value = candidate.detach().clone().requires_grad_()
        expected = voltage(value)
        expected_gradient = torch.autograd.grad(expected.square().sum(), value)[0]
        with torch_compiler_warning_context():
            actual = compiled(value)
            gradient = torch.autograd.grad(actual.square().sum(), value)[0]
        assert torch.isfinite(gradient) and gradient != 0
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(gradient, expected_gradient, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("mutation", ["custom", "forward", "hook"])
def test_myelinated_end_diameter_keeps_nonstandard_transforms_rejected(mutation):
    model = _model()
    override = model.parametrizations.diam[1]
    if mutation == "custom":
        model.register_parametrization("diam", torch.nn.Identity())
    elif mutation == "forward":
        override.forward = lambda diam: diam
    else:
        override.register_forward_hook(lambda _module, _inputs, output: output)
    with pytest.raises(dn.func.FunctionalizationError, match="standard registered"):
        dn.func.make_functional(model, dt=DT)


def test_myelinated_end_diameter_hook_mutation_invalidates_existing_plan():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    model.parametrizations.diam[1].register_forward_hook(
        lambda _module, _inputs, output: output
    )
    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)
