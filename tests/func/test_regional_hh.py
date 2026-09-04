"""Acceptance oracles for regional functional HH lowering."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms._support import SupportKind
from dendra.models.mod import hh, pas

DT = 0.01
STEPS = 3
GNABAR_NAME = "integrator.mech.mechanisms.hh.gnabar_param"

SUPPORT_CASES = (
    (
        "rectangular",
        (slice(None), slice(1, 4)),
        SupportKind.RECTANGULAR,
    ),
    (
        "shared_columns",
        (slice(None), [1, 3]),
        SupportKind.SHARED_COLUMNS,
    ),
    (
        "packed",
        ([0, 0, 1, 1], [0, 4, 1, 3]),
        SupportKind.PACKED_FLAT,
    ),
)

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _model(selector):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.75, 2.25],
            L=4.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -60.0, -56.0, -59.0, -63.0]),
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        # Keep every compartment well conditioned while exercising shipped HH
        # with its scalar defaults only on the selected regional support.
        model.insert(pas)
        model[selector].insert(hh)
        model.initialize()
        model.train()
    return model


def _functional(model):
    return dn.func.make_functional(model, dt=DT)


def _drives(model, steps=STEPS):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -1.5,
        1.5,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -8.0e-10,
        9.0e-10,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _initialize_imperative(model):
    dt = torch.as_tensor(DT, device=model.device(), dtype=model.dtype())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    return dt


def _imperative_step(model, dt, ve, intra):
    model.integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def _assert_every_leaf_close(actual, expected, *, rtol=0.0, atol=0.0):
    actual_with_paths, actual_spec = torch.utils._pytree.tree_flatten_with_path(actual)
    expected_with_paths, expected_spec = torch.utils._pytree.tree_flatten_with_path(
        expected
    )
    assert actual_spec == expected_spec
    for (actual_path, actual_leaf), (expected_path, expected_leaf) in zip(
        actual_with_paths,
        expected_with_paths,
        strict=True,
    ):
        assert actual_path == expected_path
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
            msg=lambda message: f"state leaf {actual_path}: {message}",
        )


@pytest.mark.parametrize(
    ("_case_name", "selector", "expected_kind"),
    SUPPORT_CASES,
    ids=[case[0] for case in SUPPORT_CASES],
)
def test_regional_hh_chained_steps_and_rollout_match_imperative(
    _case_name,
    selector,
    expected_kind,
):
    functional_model = _model(selector)
    imperative_model = _model(selector)
    assert functional_model.mech.hh.support_map.spec.kind is expected_kind

    functional, tensors = _functional(functional_model)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(functional_model)
    dt = _initialize_imperative(imperative_model)
    state = tensors.state

    for index in range(STEPS):
        state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(imperative_model, dt, ve[index], intra[index])
        _assert_every_leaf_close(
            state,
            functional.extract(imperative_model).state,
        )

    rolled, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_every_leaf_close(rolled, state)


def test_packed_regional_hh_gnabar_gradient_matches_imperative():
    selector = SUPPORT_CASES[-1][1]
    functional_model = _model(selector)
    imperative_model = _model(selector)
    functional, tensors = _functional(functional_model)
    ve, intra = _drives(functional_model)

    gnabar = tensors.parameters[GNABAR_NAME]

    def functional_loss(value):
        parameters = dict(tensors.parameters)
        parameters[GNABAR_NAME] = value
        state, _aux = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )
        return state["integrator"]["v"].square().mean()

    functional_gradient = torch.autograd.grad(functional_loss(gnabar), gnabar)[0]

    dt = _initialize_imperative(imperative_model)
    for index in range(STEPS):
        _imperative_step(imperative_model, dt, ve[index], intra[index])
    imperative_gnabar = dict(imperative_model.named_parameters())[GNABAR_NAME]
    imperative_loss = imperative_model.v.square().mean()
    imperative_gradient = torch.autograd.grad(
        imperative_loss,
        imperative_gnabar,
    )[0]

    assert bool(functional_gradient.abs() > 0.0)
    torch.testing.assert_close(
        functional_gradient,
        imperative_gradient,
        rtol=2.0e-9,
        atol=2.0e-10,
    )


@pytest.mark.parametrize(
    ("_case_name", "selector", "_expected_kind"),
    SUPPORT_CASES,
    ids=[case[0] for case in SUPPORT_CASES],
)
def test_regional_hh_parameter_only_vmap_matches_explicit_and_accepts_zero_lanes(
    _case_name,
    selector,
    _expected_kind,
):
    model = _model(selector)
    functional, tensors = _functional(model)
    ve, intra = _drives(model, steps=2)
    gnabar = tensors.parameters[GNABAR_NAME]

    def run_lane(value):
        parameters = dict(tensors.parameters)
        parameters[GNABAR_NAME] = value
        state, _aux = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )
        return state["integrator"]["v"]

    lanes = torch.stack((0.8 * gnabar, 1.2 * gnabar))
    actual = torch.vmap(run_lane)(lanes)
    expected = torch.stack(tuple(run_lane(value) for value in lanes))
    torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-13)

    empty = gnabar.new_empty((0, *gnabar.shape))
    zero_lanes = torch.vmap(run_lane)(empty)
    assert zero_lanes.shape == (0, *model.shape)


def test_packed_regional_hh_compile_of_jacrev_matches_eager():
    model = _model(SUPPORT_CASES[-1][1])
    functional, tensors = _functional(model)
    ve, intra = _drives(model, steps=2)
    gnabar = tensors.parameters[GNABAR_NAME]

    def final_voltage(value):
        parameters = dict(tensors.parameters)
        parameters[GNABAR_NAME] = value
        state, _aux = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )
        return state["integrator"]["v"]

    jacobian = torch.func.jacrev(final_voltage)
    expected = jacobian(gnabar)
    compiled = torch.compile(jacobian, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual = compiled(gnabar)
    torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-10)
