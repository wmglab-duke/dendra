"""Functional parameter materialization for regionally placed mechanisms."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._support import SupportKind
from dendra.models.mod import pas

DT = 0.01
DTYPE = torch.float64
G_BUFFER = "integrator.mech.mechanisms.pas.g"
G_OVERRIDE_SUFFIX = "integrator.mech.mechanisms.pas.g_param_0"

LAYOUTS = ("rectangular", "shared_columns", "packed", "duplicates")
EXPECTED_KINDS = {
    "rectangular": SupportKind.RECTANGULAR,
    "shared_columns": SupportKind.SHARED_COLUMNS,
    "packed": SupportKind.PACKED_FLAT,
    "duplicates": SupportKind.PACKED_FLAT,
}

SHAPED_CASES = (
    (
        "rectangular_rows",
        "rectangular",
        torch.tensor([[0.002], [0.004]], dtype=DTYPE),
        False,
    ),
    (
        "shared_columns",
        "shared_columns",
        torch.tensor([[0.002, 0.004]], dtype=DTYPE),
        True,
    ),
    (
        "packed_exact",
        "packed",
        torch.tensor([0.002, 0.004, 0.006], dtype=DTYPE),
        False,
    ),
)

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _RegionalRangeState(State):
    State.STATE("x")
    State.RANGE(rate=0.25)
    State.DERIVATIVE("x' = rate")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}


class _RegionalStateMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_RegionalRangeState)


class _PositiveRegionalRangeState(State):
    State.STATE("x")
    State.RANGEP(rate=0.25)
    State.DERIVATIVE("x' = rate")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}


class _PositiveRegionalStateMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_PositiveRegionalRangeState)


class _RegionalBatchState(State):
    State.STATE("x")
    State.BATCH(rate=0.25)
    State.DERIVATIVE("x' = rate")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}


class _RegionalBatchMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_RegionalBatchState)


def _insert_pas(model, layout: str, *, g) -> None:
    if layout == "rectangular":
        target = model[:, 1:4]
        preserve_multiplicity = False
    elif layout == "shared_columns":
        target = model[:, torch.tensor([1, 3])]
        preserve_multiplicity = False
    elif layout == "packed":
        target = model[
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 4, 2]),
        ]
        preserve_multiplicity = False
    elif layout == "duplicates":
        target = model[
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([1, 1, 3, 3]),
        ]
        preserve_multiplicity = True
    else:  # pragma: no cover - test-helper guard
        raise AssertionError(f"unknown regional layout {layout!r}")

    target.insert(
        pas,
        g=g,
        preserve_multiplicity=preserve_multiplicity,
    )


def _model(
    layout: str,
    *,
    g=0.002,
    preserve_mechanism_population_axis: bool = False,
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.5, 2.0],
            L=4.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -62.0, -60.0, -58.0, -56.0]),
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            preserve_mechanism_population_axis=preserve_mechanism_population_axis,
        )
        _insert_pas(model, layout, g=g)
        model.initialize()
        model.train()

    mechanism = model.mech.pas
    assert mechanism.support_spec.kind is EXPECTED_KINDS[layout]
    return model


def _case(
    layout: str,
    *,
    g=0.002,
    preserve_mechanism_population_axis: bool = False,
):
    model = _model(
        layout,
        g=g,
        preserve_mechanism_population_axis=preserve_mechanism_population_axis,
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)
    assert G_OVERRIDE_SUFFIX in tensors.parameters
    return model, functional, tensors


def _effective_g(functional, tensors, raw_g: torch.Tensor) -> torch.Tensor:
    parameters = dict(tensors.parameters)
    parameters[G_OVERRIDE_SUFFIX] = raw_g
    prepared = functional.prepare(parameters, tensors.constants)
    return prepared.values["mechanisms"][G_BUFFER]


def _expected_shaped_value(case_name: str, raw: torch.Tensor) -> torch.Tensor:
    if case_name == "rectangular_rows":
        return raw.expand(2, 3)
    if case_name == "shared_columns":
        return raw.expand(2, 2)
    if case_name == "packed_exact":
        return raw
    raise AssertionError(f"unknown shaped case {case_name!r}")


def _expected_shaped_jacobian(case_name: str, raw: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(raw.numel(), device=raw.device, dtype=raw.dtype)
    identity = identity.reshape(*raw.shape, *raw.shape)
    if case_name == "rectangular_rows":
        return identity.expand(2, 3, *raw.shape)
    if case_name == "shared_columns":
        return identity.expand(2, 2, *raw.shape)
    if case_name == "packed_exact":
        return identity
    raise AssertionError(f"unknown shaped case {case_name!r}")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_regional_pas_override_has_exact_prepared_value_and_jacobians(layout):
    model, functional, tensors = _case(layout)
    raw_g = tensors.parameters[G_OVERRIDE_SUFFIX]
    changed_g = 1.75 * raw_g

    actual = _effective_g(functional, tensors, changed_g)
    expected = changed_g.expand_as(model.mech.pas.g)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    reverse = torch.func.jacrev(lambda value: _effective_g(functional, tensors, value))(
        raw_g
    )
    forward = torch.func.jacfwd(lambda value: _effective_g(functional, tensors, value))(
        raw_g
    )
    expected_jacobian = torch.ones_like(model.mech.pas.g)

    torch.testing.assert_close(reverse, expected_jacobian, rtol=0.0, atol=0.0)
    torch.testing.assert_close(forward, expected_jacobian, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(reverse) == reverse.numel()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_regional_pas_override_vmap_matches_explicit_and_accepts_zero_lanes(layout):
    model, functional, tensors = _case(layout)
    raw_g = tensors.parameters[G_OVERRIDE_SUFFIX]
    lanes = torch.stack((0.5 * raw_g, raw_g, 1.5 * raw_g))

    def run_lane(value):
        return _effective_g(functional, tensors, value)

    actual = torch.vmap(run_lane)(lanes)
    expected = torch.stack(tuple(run_lane(value) for value in lanes))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    empty = torch.vmap(run_lane)(raw_g.new_empty((0, *raw_g.shape)))
    assert empty.shape == (0, *model.mech.pas.g.shape)


@pytest.mark.parametrize(
    ("case_name", "layout", "override", "preserve_population_axis"),
    SHAPED_CASES,
    ids=[case[0] for case in SHAPED_CASES],
)
def test_shaped_regional_pas_override_values_jacobians_and_vmap(
    case_name,
    layout,
    override,
    preserve_population_axis,
):
    model, functional, tensors = _case(
        layout,
        g=override,
        preserve_mechanism_population_axis=preserve_population_axis,
    )
    raw_g = tensors.parameters[G_OVERRIDE_SUFFIX]
    assert raw_g.shape == override.shape

    changed_g = 1.25 * raw_g
    actual = _effective_g(functional, tensors, changed_g)
    expected = _expected_shaped_value(case_name, changed_g)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, 1.25 * model.mech.pas.g, rtol=0.0, atol=0.0)

    def effective(value):
        return _effective_g(functional, tensors, value)

    reverse = torch.func.jacrev(effective)(raw_g)
    forward = torch.func.jacfwd(effective)(raw_g)
    expected_jacobian = _expected_shaped_jacobian(case_name, raw_g)
    torch.testing.assert_close(reverse, expected_jacobian, rtol=0.0, atol=0.0)
    torch.testing.assert_close(forward, expected_jacobian, rtol=0.0, atol=0.0)

    lanes = torch.stack((0.5 * raw_g, raw_g, 1.5 * raw_g))
    vmapped = torch.vmap(effective)(lanes)
    explicit = torch.stack(tuple(effective(value) for value in lanes))
    torch.testing.assert_close(vmapped, explicit, rtol=0.0, atol=0.0)

    empty = torch.vmap(effective)(raw_g.new_empty((0, *raw_g.shape)))
    assert empty.shape == (0, *model.mech.pas.g.shape)


def test_copied_regional_override_preserves_per_copy_layout_and_jacobian():
    override = torch.tensor([[0.001], [0.002], [0.003]], dtype=DTYPE)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.5, 2.0],
            L=4.0,
            dx=1.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model[:, torch.tensor([1, 3])].insert(pas, copies=3, g=override)
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    raw_g = tensors.parameters[G_OVERRIDE_SUFFIX]

    def effective(value):
        return _effective_g(functional, tensors, value)

    expected = raw_g.expand(3, 4).reshape(12)
    torch.testing.assert_close(effective(raw_g), expected, rtol=0.0, atol=0.0)
    identity = torch.eye(
        raw_g.numel(),
        device=raw_g.device,
        dtype=raw_g.dtype,
    ).reshape(*raw_g.shape, *raw_g.shape)
    expected_jacobian = identity.expand(3, 4, *raw_g.shape).reshape(
        12,
        *raw_g.shape,
    )
    torch.testing.assert_close(
        torch.func.jacrev(effective)(raw_g),
        expected_jacobian,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.func.jacfwd(effective)(raw_g),
        expected_jacobian,
        rtol=0.0,
        atol=0.0,
    )


def test_population_level_shaped_override_is_functionally_materialized():
    override = torch.tensor([[2.0], [3.0]], dtype=DTYPE)
    override_key = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8])
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.5, 2.0],
            L=4.0,
            dx=1.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas)
        model.parametrize("cm", override, key=override_key, alias="regional")
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    raw_cm = tensors.parameters["cm_regional"]

    def effective(value):
        parameters = dict(tensors.parameters)
        parameters["cm_regional"] = value
        return functional.prepare(parameters, tensors.constants).values["population"][
            "cm"
        ]

    expected = torch.cat((raw_cm.expand(2, 4), raw_cm.new_ones((2, 1))), dim=1)
    torch.testing.assert_close(effective(raw_cm), expected, rtol=0.0, atol=0.0)

    expected_jacobian = raw_cm.new_zeros((2, 5, *raw_cm.shape))
    expected_jacobian[:, :4] = (
        torch.eye(
            raw_cm.numel(),
            device=raw_cm.device,
            dtype=raw_cm.dtype,
        )
        .reshape(*raw_cm.shape, *raw_cm.shape)
        .expand(2, 4, *raw_cm.shape)
    )
    torch.testing.assert_close(
        torch.func.jacrev(effective)(raw_cm),
        expected_jacobian,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.func.jacfwd(effective)(raw_cm),
        expected_jacobian,
        rtol=0.0,
        atol=0.0,
    )


def test_compile_of_jacrev_tracks_duplicate_regional_pas_override():
    _model_value, functional, tensors = _case("duplicates")
    raw_g = tensors.parameters[G_OVERRIDE_SUFFIX]

    def next_voltage(value):
        parameters = dict(tensors.parameters)
        parameters[G_OVERRIDE_SUFFIX] = value
        state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return state["integrator"]["v"]

    transformed = torch.func.jacrev(next_voltage)
    expected = transformed(raw_g)
    assert torch.isfinite(expected).all()
    assert torch.count_nonzero(expected) > 0

    with torch_compiler_warning_context():
        compiled = torch.compile(
            transformed,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled(raw_g)

    torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-13)


def test_nested_state_shaped_override_is_exact_and_compile_jacrev_safe():
    rate = torch.tensor([[0.2], [0.35]], dtype=DTYPE)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.5, 2.0],
            L=4.0,
            dx=1.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model[:, 1:4].insert(_RegionalStateMechanism, rate=rate)
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    mechanism_name = "_RegionalStateMechanism"
    raw_name = (
        "integrator.mech.mechanisms._RegionalStateMechanism."
        "DE._RegionalRangeState.rate_param_0"
    )
    buffer_name = (
        "integrator.mech.mechanisms._RegionalStateMechanism.DE._RegionalRangeState.rate"
    )
    raw_rate = tensors.parameters[raw_name]

    def prepared_rate(value):
        parameters = dict(tensors.parameters)
        parameters[raw_name] = value
        return functional.prepare(parameters, tensors.constants).values["mechanisms"][
            buffer_name
        ]

    expected_rate = raw_rate.expand(2, 3)
    torch.testing.assert_close(
        prepared_rate(raw_rate),
        expected_rate,
        rtol=0.0,
        atol=0.0,
    )
    expected_parameter_jacobian = torch.eye(
        raw_rate.numel(),
        device=raw_rate.device,
        dtype=raw_rate.dtype,
    ).reshape(*raw_rate.shape, *raw_rate.shape)
    expected_parameter_jacobian = expected_parameter_jacobian.expand(
        2,
        3,
        *raw_rate.shape,
    )
    reverse = torch.func.jacrev(prepared_rate)(raw_rate)
    forward = torch.func.jacfwd(prepared_rate)(raw_rate)
    torch.testing.assert_close(
        reverse,
        expected_parameter_jacobian,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        forward,
        expected_parameter_jacobian,
        rtol=0.0,
        atol=0.0,
    )

    def next_x(value):
        parameters = dict(tensors.parameters)
        parameters[raw_name] = value
        state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return state["mechanisms"][mechanism_name]["x"]

    transformed = torch.func.jacrev(next_x)
    expected = DT * expected_parameter_jacobian
    torch.testing.assert_close(
        transformed(raw_rate),
        expected,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    with torch_compiler_warning_context():
        compiled = torch.compile(
            transformed,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled(raw_rate)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-13)

    assert tensors.state["mechanisms"][mechanism_name]["x"].shape == (2, 3)


def test_nested_positive_override_preserves_authored_transform_and_jacobians():
    rate = torch.tensor([[0.2], [0.35]], dtype=DTYPE)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.5, 2.0],
            L=4.0,
            dx=1.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model[:, 1:4].insert(_PositiveRegionalStateMechanism, rate=rate)
        model.initialize()
        model.eval()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    prefix = (
        "integrator.mech.mechanisms._PositiveRegionalStateMechanism."
        "DE._PositiveRegionalRangeState"
    )
    raw_name = f"{prefix}.rate_param_0.rho"
    buffer_name = f"{prefix}.rate"
    raw_rate = tensors.parameters[raw_name]
    source_transform = model.mech._PositiveRegionalStateMechanism.DE[
        "_PositiveRegionalRangeState"
    ].rate_param_0
    expected_transform = copy.deepcopy(source_transform).train()

    def prepared_rate(value):
        parameters = dict(tensors.parameters)
        parameters[raw_name] = value
        return functional.prepare(parameters, tensors.constants).values["mechanisms"][
            buffer_name
        ]

    def expected_rate(value):
        transformed = torch.func.functional_call(
            expected_transform,
            {"rho": value},
            (),
            tie_weights=False,
        )
        return transformed.expand(2, 3)

    changed_raw = raw_rate + 0.25
    torch.testing.assert_close(
        prepared_rate(changed_raw),
        expected_rate(changed_raw),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.func.jacrev(prepared_rate)(raw_rate),
        torch.func.jacrev(expected_rate)(raw_rate),
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    torch.testing.assert_close(
        torch.func.jacfwd(prepared_rate)(raw_rate),
        torch.func.jacfwd(expected_rate)(raw_rate),
        rtol=1.0e-12,
        atol=1.0e-13,
    )


def test_nested_batch_override_materializes_and_broadcasts_through_step():
    rate = torch.tensor([[0.2], [0.35]], dtype=DTYPE)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.5, 2.0],
            L=4.0,
            dx=1.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model[:, 1:4].insert(_RegionalBatchMechanism, rate=rate)
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    prefix = "integrator.mech.mechanisms._RegionalBatchMechanism.DE._RegionalBatchState"
    raw_name = f"{prefix}.rate_param_0"
    buffer_name = f"{prefix}.rate"
    raw_rate = tensors.parameters[raw_name]

    def prepared_rate(value):
        parameters = dict(tensors.parameters)
        parameters[raw_name] = value
        return functional.prepare(parameters, tensors.constants).values["mechanisms"][
            buffer_name
        ]

    torch.testing.assert_close(
        prepared_rate(raw_rate),
        raw_rate,
        rtol=0.0,
        atol=0.0,
    )
    identity = torch.eye(
        raw_rate.numel(),
        device=raw_rate.device,
        dtype=raw_rate.dtype,
    ).reshape(*raw_rate.shape, *raw_rate.shape)
    torch.testing.assert_close(
        torch.func.jacrev(prepared_rate)(raw_rate),
        identity,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.func.jacfwd(prepared_rate)(raw_rate),
        identity,
        rtol=0.0,
        atol=0.0,
    )

    parameters = dict(tensors.parameters)
    parameters[raw_name] = raw_rate
    next_state, _aux = functional.prepare_and_step(
        parameters,
        tensors.constants,
        tensors.state,
    )
    torch.testing.assert_close(
        next_state["mechanisms"]["_RegionalBatchMechanism"]["x"],
        DT * raw_rate.expand(2, 3),
        rtol=1.0e-12,
        atol=1.0e-13,
    )


def test_compatible_extraction_uses_target_regional_override_value():
    source = _model("shared_columns", g=0.002)
    target = _model("shared_columns", g=0.007)
    functional, _source_tensors = dn.func.make_functional(source, dt=DT)

    target_tensors = functional.extract(target)
    raw_g = target_tensors.parameters[G_OVERRIDE_SUFFIX]
    target_raw_g = dict(target.named_parameters())[G_OVERRIDE_SUFFIX]
    torch.testing.assert_close(
        raw_g,
        target_raw_g,
        rtol=0.0,
        atol=0.0,
    )
    prepared = functional.prepare(target_tensors.parameters, target_tensors.constants)
    torch.testing.assert_close(
        prepared.values["mechanisms"][G_BUFFER],
        target.mech.pas.g,
        rtol=0.0,
        atol=0.0,
    )


def test_existing_plan_accepts_same_value_override_key_rebinding():
    model, functional, tensors = _case("shared_columns")
    rebound_key = model.mech.pas.keys["g"].clone()
    model.mech.pas.keys["g"] = rebound_key

    first = _effective_g(
        functional,
        tensors,
        tensors.parameters[G_OVERRIDE_SUFFIX],
    )
    second = _effective_g(
        functional,
        tensors,
        tensors.parameters[G_OVERRIDE_SUFFIX],
    )

    assert model.mech.pas.keys["g"] is rebound_key
    torch.testing.assert_close(first, model.mech.pas.g, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second, model.mech.pas.g, rtol=0.0, atol=0.0)


def test_existing_plan_rejects_override_key_value_change():
    model, functional, tensors = _case("shared_columns")
    key = model.mech.pas.keys["g"]
    with torch.no_grad():
        key[0] = key[1]

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"regional-parameter placement changed|parameter.*placement",
    ):
        _effective_g(
            functional,
            tensors,
            tensors.parameters[G_OVERRIDE_SUFFIX],
        )


def test_custom_parameter_materializer_fails_closed():
    model = _model("rectangular")
    model.mech.pas._derive_parameter_buffers = model.mech.pas._derive_parameter_buffers

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"custom parameter-buffer materializers",
    ):
        dn.func.make_functional(model, dt=DT)


def _aliased_model(*, swapped: bool):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.5, 2.0],
            L=4.0,
            dx=1.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        hot_column, cold_column = (3, 1) if swapped else (1, 3)
        model[:, hot_column].insert(pas, alias="hot", g=0.003)
        model[:, cold_column].insert(pas, alias="cold", g=0.006)
        model.initialize()
        model.train()
    return model


def test_compatible_extraction_rejects_swapped_regional_override_placement():
    source = _aliased_model(swapped=False)
    target = _aliased_model(swapped=True)
    functional, _tensors = dn.func.make_functional(source, dt=DT)

    # The total mechanism support and raw parameter names/shapes are identical;
    # only the exact local slots owned by the two aliases differ.
    assert source.mech.pas.support_spec == target.mech.pas.support_spec
    assert set(dict(source.named_parameters())) == set(dict(target.named_parameters()))
    assert not torch.equal(source.mech.pas.keys["g"], target.mech.pas.keys["g"])

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"structure does not match|override.*layout|parameter.*placement",
    ):
        functional.extract(target)
