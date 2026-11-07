"""Float32 scalar-tree conservation and independent dense-reference checks."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas

DT = 0.025
CM = "cm_param.rho"
RHOA = "rhoa_scale_param.rho"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(not DENDRA_SOLVERS_AVAILABLE, reason="requires native CPU DHS"),
]


def _morphology():
    morphology = dn.Morphology(rhoa=1000.0, cm=1.0)
    soma = morphology.section("soma", L=13.5, diam=11.7, nseg=1)
    left = morphology.section("left", L=123.4, diam=0.73, nseg=1)
    right = morphology.section("right", L=59.9, diam=1.31, nseg=1)
    left.connect(soma.at(1.0), child_end=0)
    right.connect(soma.at(1.0), child_end=0)
    return morphology


def _model(*, passive=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE="cpu", DTYPE=torch.float32):
        model = dn.Tree.from_morphology(
            _morphology(),
            N=1,
            dtype=torch.float32,
            v_init=-62.0,
            integrator=dn.dhs(threads=32, imem=False),
        )
        # Functional admission requires a membrane declaration. A zero-G PAS
        # declaration supplies that structure while exerting exactly no current.
        model.internal_nodes.insert(pas, g=4e-4 if passive else 0.0, e=-71.0)
        model.initialize()
        model.train()
    assert model.v.shape[-1] == 4
    assert torch.count_nonzero(model.area == 0) == 1
    return model


@pytest.mark.parametrize("voltage", [-62.0, 17.125])
def test_float32_uniform_unforced_tree_is_exactly_stationary(voltage):
    model = _model()
    model.v = torch.full_like(model.v, voltage)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v",))})
    final, auxiliary = dn.func.run(
        functional.bind(tensors.parameters, prepared),
        tensors.state,
        steps=800,
        callbacks=callbacks,
    )
    recorded = auxiliary["callbacks"]["trace"]["v"]
    assert recorded.dtype == torch.float32
    assert torch.isfinite(recorded).all()
    # This is the exact constant null mode of the axial Laplacian, including
    # the zero-area algebraic junction; no numerical oracle is needed.
    torch.testing.assert_close(
        recorded, torch.full_like(recorded, voltage), rtol=0, atol=0
    )
    torch.testing.assert_close(final["integrator"]["v"], model.v, rtol=0, atol=0)


def test_float32_uniform_unforced_packed_groups_are_exactly_stationary():
    tree = _model()
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DEVICE="cpu", DTYPE=torch.float32):
        point = dn.SingleCompartment(N=2, C=1, v_init=17.125)
        point.insert(pas, g=0.0)
        model = dn.concat_models({"tree": tree, "point": point}, threads=32)
        model.initialize()
        model.train()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v",))})
    final, auxiliary = dn.func.run(
        functional.bind(tensors.parameters, prepared),
        tensors.state,
        steps=800,
        callbacks=callbacks,
    )
    recorded = auxiliary["callbacks"]["trace"]["v"]
    torch.testing.assert_close(recorded, model.v.expand_as(recorded), rtol=0, atol=0)
    torch.testing.assert_close(final["integrator"]["v"], model.v, rtol=0, atol=0)


def _case():
    model = _model(passive=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    voltage = torch.tensor([[-64.0, -51.0, -69.0, -58.0]], dtype=torch.float32)
    ve = torch.tensor([[0.5, -1.0, 2.0, -0.4]], dtype=torch.float32)
    intra = torch.tensor([[1.3e-7, -8e-8, 1.9e-7, 0.0]], dtype=torch.float32)
    # The algebraic junction carries no externally injected current.
    intra = torch.where(model.area > 0, intra, torch.zeros_like(intra))

    def response(v0, raw_cm, raw_rhoa):
        parameters = {**tensors.parameters, CM: raw_cm, RHOA: raw_rhoa}
        state = {
            **tensors.state,
            "integrator": {**tensors.state["integrator"], "v": v0},
        }
        next_state, _ = functional.prepare_and_step(
            parameters,
            tensors.constants,
            state,
            dn.func.StepInput(ve=ve, intra=intra),
        )
        return next_state["integrator"]["v"]

    arguments = (voltage, tensors.parameters[CM], tensors.parameters[RHOA])
    return model, functional, tensors, response, arguments, ve, intra


def _dense_oracle(model, ve, intra):
    """Solve the authored passive physical circuit directly in float64.

    This is an independent nodal matrix, in storage order, not a copy of DHS
    elimination or the proposed increment residual. Public parameter modules
    supply only their documented coordinate transformations.
    """
    area = model.area.detach().double()
    resistance = model.edge_resistance_ohm.detach().double()
    parent = model.compartment_graph.topology.parent_index
    children = [child for child, ancestor in enumerate(parent) if ancestor >= 0]
    incidence = torch.zeros(len(parent), len(children), dtype=torch.float64)
    for edge, child in enumerate(children):
        incidence[child, edge] = 1
        incidence[parent[child], edge] = -1
    physical = (area > 0).double()

    def response(v0, raw_cm, raw_rhoa):
        cm = torch.func.functional_call(model.cm_param, {"rho": raw_cm}, ())
        rhoa = torch.func.functional_call(model.rhoa_scale_param, {"rho": raw_rhoa}, ())
        cm = torch.broadcast_to(cm, area.shape)
        rhoa = torch.broadcast_to(rhoa, area.shape)
        edge = 1 / (resistance[:, children] * rhoa[:, children])
        laplacian = (incidence.unsqueeze(0) * edge.unsqueeze(1)) @ incidence.T
        capacitive = (1e-6 * cm * area) / (DT * 1e-3)
        conductance = 4e-4 * physical * area
        matrix = laplacian + torch.diag_embed(capacitive + conductance)
        rhs = capacitive * v0 + conductance * (-71.0) + intra.double()
        rhs = rhs - (laplacian @ ve.double().unsqueeze(-1)).squeeze(-1)
        return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)

    return response


def _assert_gradient(actual, expected):
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    assert torch.linalg.vector_norm(expected) > 0
    error = torch.linalg.vector_norm(actual.double() - expected)
    assert error <= 2e-3 * torch.linalg.vector_norm(expected)


def test_float32_nonuniform_driven_tree_matches_dense_values_and_gradients():
    model, _, _, response, arguments, ve, intra = _case()
    dense = _dense_oracle(model, ve, intra)
    actual_arguments = tuple(
        value.detach().clone().requires_grad_() for value in arguments
    )
    expected_arguments = tuple(
        value.detach().double().requires_grad_() for value in arguments
    )
    actual, expected = response(*actual_arguments), dense(*expected_arguments)
    torch.testing.assert_close(actual.double(), expected, rtol=5e-4, atol=3e-4)
    actual_gradients = torch.autograd.grad(actual.square().mean(), actual_arguments)
    expected_gradients = torch.autograd.grad(
        expected.square().mean(), expected_arguments
    )
    for value, reference in zip(actual_gradients, expected_gradients, strict=True):
        _assert_gradient(value, reference)


def test_float32_tree_vmap_and_jvp_match_dense_oracle():
    model, _, _, response, arguments, ve, intra = _case()
    dense = _dense_oracle(model, ve, intra)
    lanes = tuple(torch.stack((value, value + 0.03)) for value in arguments)
    actual = torch.vmap(response)(*lanes)
    expected = torch.stack(
        [dense(*(value[index].double() for value in lanes)) for index in range(2)]
    )
    torch.testing.assert_close(actual.double(), expected, rtol=5e-4, atol=3e-4)
    tangent = tuple(torch.full_like(value, 0.125) for value in arguments)
    _, actual_jvp = torch.func.jvp(response, arguments, tangent)
    _, expected_jvp = torch.func.jvp(
        dense,
        tuple(value.double() for value in arguments),
        tuple(value.double() for value in tangent),
    )
    _assert_gradient(actual_jvp, expected_jvp)


class _Float32Numerics(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.native_solves = []

    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        def check(values):
            floating = [
                value
                for value in pytree.tree_leaves(values)
                if torch.is_tensor(value) and value.is_floating_point()
            ]
            assert all(value.dtype == torch.float32 for value in floating), (
                operation,
                [value.dtype for value in floating],
            )
            assert all(torch.isfinite(value).all() for value in floating)
            return floating

        inputs = check((args, kwargs))
        result = operation(*args, **kwargs)
        outputs = check(result)
        if operation._schema.name.startswith("dendra_solvers::") and inputs:
            assert outputs
            self.native_solves.append(operation._schema.name)
        return result


def test_scalar_tree_numeric_workspace_step_and_backward_remain_float32(monkeypatch):
    model = _model(passive=True)
    integrator = model.integrator
    dt = torch.tensor(DT, dtype=torch.float32)
    builder = integrator._prepare_workspace
    physical = {}

    def capture(timestep, **inputs):
        physical.update(timestep=timestep, inputs=inputs)
        return builder(timestep, **inputs)

    with monkeypatch.context() as patch:
        patch.setattr(integrator, "_prepare_workspace", capture)
        # Host graph import reads authored binary64 geometry before converting
        # it to model dtype; guard the derived numerical workspace and raw
        # transition, not NumPy-to-Torch metadata import.
        integrator._initialize(model, dt, force=True)
    with _Float32Numerics():
        builder(physical["timestep"], **physical["inputs"])
    voltage = torch.tensor(
        [[-64.0, -51.0, -69.0, -58.0]], dtype=torch.float32, requires_grad=True
    )
    with _Float32Numerics() as mode:
        value, _ = integrator._step(
            voltage, dt, model.celsius, intra=torch.full_like(voltage, 1e-8)
        )
        gradient = torch.autograd.grad(value.square().mean(), voltage)[0]
    assert mode.native_solves
    assert gradient.dtype == torch.float32
