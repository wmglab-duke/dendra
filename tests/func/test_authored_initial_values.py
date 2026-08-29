"""Acceptance tests for pure authored initial-value overlays."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State

DT = 0.0125
DTYPE = torch.float64
MECHANISM_NAME = "_AuthoredInitializationMechanism"
STATE_NAME = "_AuthoredInitialValueState"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _AuthoredInitialValueState(State):
    State.STATE("inferred", "explicit", "layered")
    State.RANGE(state_gain=0.375)
    State.CARRY("state_carry")
    State.DERIVATIVE(
        "inferred' = 0.0 * inferred",
        "explicit' = 0.0 * explicit",
        "layered' = 0.0 * layered",
    )

    def state_defaults(self, v, values):
        del values
        return {
            "inferred": torch.sigmoid(0.02 * v),
            "layered": torch.tanh(-0.01 * v),
        }

    def initial_values(self, v, values):
        del v
        return {
            # State overlays run after the owning Mechanism overlay.
            "layered": values["layered"] + self.state_gain,
            "state_carry": (
                values["inferred"]
                - values["explicit"]
                + 0.5 * values["layered"]
                + 0.001 * values["celsius"]
                + 0.002 * values["diam"]
            ),
        }


class _SeededSodiumState(State):
    """Intentionally rely on the canonical writable-Ion initial seed."""

    State.STATE("nai")
    State.DERIVATIVE("nai' = 0.0 * nai")


class _SeededMaterialState(State):
    """Intentionally rely on the canonical writable-Material initial seed."""

    State.STATE("amount")
    State.DERIVATIVE("amount' = 0.0 * amount")


class _AuthoredInitializationMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(
        _AuthoredInitialValueState,
        _SeededSodiumState,
        _SeededMaterialState,
    )
    Mechanism.GLOBAL(gain=0.25, g=1.0e-8, e=-65.0)
    Mechanism.CARRY("mechanism_carry")
    Mechanism.USEION("na", write=["nai"])
    Mechanism.USEMATERIAL("pool", write=["amount"])
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def initial_values(self, v, values):
        inferred = values["inferred"] + self.gain * torch.tanh(0.01 * v)
        explicit = values["explicit"] + self.gain
        layered = values["layered"] + 2.0 * self.gain
        return {
            "inferred": inferred,
            "explicit": explicit,
            "layered": layered,
            "mechanism_carry": (
                inferred
                + explicit
                + layered
                + 0.001 * values["celsius"]
                + 0.002 * values["diam"]
            ),
        }

    def i(self, v):
        return self.g * self.inferred * (v - self.e)


def _model(mechanism=_AuthoredInitializationMechanism):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=3,
            v_init=torch.tensor([-68.0, -64.0, -60.0], dtype=DTYPE),
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.concentrations(nai0=torch.nn.Parameter(torch.tensor(9.25, dtype=DTYPE)))
        model.material(
            "pool",
            fields={"amount": torch.nn.Parameter(torch.tensor(3.5, dtype=DTYPE))},
            min_values={"amount": 0.5},
            conserved={"amount": False},
        )
        model.insert(mechanism, ic={"explicit": 0.75})
        model.initialize()
        model.train()
    return model


def _parameter_name(parameters, suffix):
    matches = [name for name in parameters if name.endswith(suffix)]
    assert len(matches) == 1, (suffix, tuple(parameters))
    return matches[0]


def _assert_tree_close(actual, expected, *, rtol=0.0, atol=0.0):
    actual_values, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_values, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_value, expected_value in zip(
        actual_values,
        expected_values,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=rtol,
            atol=atol,
        )


def _initialized(functional, tensors, *, v_init=None, replacements=None):
    parameters = dict(tensors.parameters)
    if replacements:
        parameters.update(replacements)
    inputs = dn.func.InitializationInput(
        v_init=tensors.initialization.v_init if v_init is None else v_init,
        states=tensors.initialization.states,
    )
    return functional.initialize(parameters, tensors.constants, inputs)


def test_authored_initial_values_match_imperative_and_obey_layered_precedence():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    v_init = torch.tensor([[-71.0, -63.0, -55.0]], dtype=DTYPE)

    actual = _initialized(functional, tensors, v_init=v_init)
    reference = copy.deepcopy(model)
    reference.set_v_init(v_init)
    reference.initialize()
    _assert_tree_close(actual.state, functional.extract(reference).state)

    mechanism = reference.mech.mechanisms[MECHANISM_NAME]
    gain = mechanism.gain
    state_gain = mechanism.DE[STATE_NAME].state_gain
    celsius = mechanism.celsius.reshape(1, 1)
    diam = mechanism.diam
    inferred_from_defaults = torch.sigmoid(0.02 * v_init)
    layered_from_defaults = torch.tanh(-0.01 * v_init)
    inferred_after_mechanism = inferred_from_defaults + gain * torch.tanh(0.01 * v_init)
    explicit_after_mechanism = torch.full_like(v_init, 0.75) + gain
    layered_after_mechanism = layered_from_defaults + 2.0 * gain

    states = actual.state["mechanisms"][MECHANISM_NAME]
    torch.testing.assert_close(states["inferred"], inferred_after_mechanism)
    torch.testing.assert_close(states["explicit"], explicit_after_mechanism)
    torch.testing.assert_close(
        states["layered"],
        layered_after_mechanism + state_gain,
    )

    expected_mechanism_carry = (
        inferred_after_mechanism
        + explicit_after_mechanism
        + layered_after_mechanism
        + 0.001 * celsius
        + 0.002 * diam
    )
    torch.testing.assert_close(
        actual.state["mechanism_buffers"][MECHANISM_NAME]["mechanism_carry"],
        expected_mechanism_carry,
    )
    expected_state_carry = (
        inferred_after_mechanism
        - explicit_after_mechanism
        + 0.5 * layered_after_mechanism
        + 0.001 * celsius
        + 0.002 * diam
    )
    torch.testing.assert_close(
        actual.state["state_buffers"][MECHANISM_NAME][STATE_NAME]["state_carry"],
        expected_state_carry,
    )

    # ``nai`` has no state_defaults or authored value: the canonical Ion initial
    # field is its complete declared-state seed and is committed coherently.
    expected_nai = mechanism.nai.new_full(mechanism.nai.shape, 9.25)
    torch.testing.assert_close(states["nai"], expected_nai)
    torch.testing.assert_close(actual.state["ions"]["na"]["nai"], expected_nai)
    expected_amount = mechanism.amount.new_full(mechanism.amount.shape, 3.5)
    torch.testing.assert_close(states["amount"], expected_amount)
    torch.testing.assert_close(
        actual.state["materials"]["pool"]["amount"],
        expected_amount,
    )

    # Pure returned outputs retain the same imperative autograd connectivity as
    # a imperative tensor assignment; only state_defaults keeps its historical detach.
    assert mechanism.mechanism_carry.requires_grad
    (gain_gradient,) = torch.autograd.grad(
        mechanism.mechanism_carry.sum(),
        mechanism.gain_param,
        retain_graph=True,
    )
    assert torch.count_nonzero(gain_gradient) > 0


def test_authored_carry_and_shared_seed_reset_fresh_on_every_initialize():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    expected = functional.extract(model).state
    mechanism = model.mech.mechanisms[MECHANISM_NAME]
    state = mechanism.DE[STATE_NAME]

    with torch.no_grad():
        mechanism.inferred.fill_(123.0)
        mechanism.explicit.fill_(-456.0)
        mechanism.layered.fill_(789.0)
        mechanism.nai.fill_(321.0)
        mechanism.amount.fill_(222.0)
        mechanism.mechanism_carry.fill_(654.0)
        state.state_carry.fill_(-987.0)
        model.mech.ions["na"].nai.fill_(111.0)
        model.mech.materials["pool"].amount.fill_(333.0)

    model.initialize()
    _assert_tree_close(functional.extract(model).state, expected)

    first = _initialized(functional, tensors)
    second = _initialized(functional, tensors)
    _assert_tree_close(first.state, second.state)


def test_authored_initial_values_compose_with_jacrev_jacfwd_and_empty_vmap():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    gain_name = _parameter_name(
        tensors.parameters,
        f"{MECHANISM_NAME}.gain_param",
    )
    state_gain_name = _parameter_name(
        tensors.parameters,
        f"{MECHANISM_NAME}.DE.{STATE_NAME}.state_gain_param",
    )
    nai_name = _parameter_name(tensors.parameters, "ions.na.i_init")
    base_v = tensors.initialization.v_init
    base_gain = tensors.parameters[gain_name]
    base_state_gain = tensors.parameters[state_gain_name]
    base_nai = tensors.parameters[nai_name]

    def outputs(v_init, gain, state_gain, nai):
        initialized = _initialized(
            functional,
            tensors,
            v_init=v_init,
            replacements={
                gain_name: gain,
                state_gain_name: state_gain,
                nai_name: nai,
            },
        )
        state = initialized.state
        return (
            state["mechanisms"][MECHANISM_NAME]["inferred"],
            state["mechanisms"][MECHANISM_NAME]["layered"],
            state["mechanisms"][MECHANISM_NAME]["nai"],
            state["mechanism_buffers"][MECHANISM_NAME]["mechanism_carry"],
            state["state_buffers"][MECHANISM_NAME][STATE_NAME]["state_carry"],
            state["ions"]["na"]["nai"],
        )

    arguments = (base_v, base_gain, base_state_gain, base_nai)
    reverse = torch.func.jacrev(outputs, argnums=(0, 1, 2, 3))(*arguments)
    forward = torch.func.jacfwd(outputs, argnums=(0, 1, 2, 3))(*arguments)
    _assert_tree_close(reverse, forward, rtol=2.0e-10, atol=2.0e-11)

    v_lanes = torch.stack((base_v - 1.0, base_v + 1.0))
    gain_lanes = torch.stack((0.8 * base_gain, 1.2 * base_gain))
    state_gain_lanes = torch.stack((0.9 * base_state_gain, 1.1 * base_state_gain))
    nai_lanes = torch.stack((0.75 * base_nai, 1.25 * base_nai))
    actual = torch.vmap(outputs)(
        v_lanes,
        gain_lanes,
        state_gain_lanes,
        nai_lanes,
    )
    expected = torch.utils._pytree.tree_map(
        lambda *values: torch.stack(values),
        *(
            outputs(
                v_lanes[index],
                gain_lanes[index],
                state_gain_lanes[index],
                nai_lanes[index],
            )
            for index in range(2)
        ),
    )
    _assert_tree_close(actual, expected, rtol=1.0e-14, atol=1.0e-15)

    empty = torch.vmap(outputs)(
        v_lanes[:0],
        gain_lanes[:0],
        state_gain_lanes[:0],
        nai_lanes[:0],
    )
    for value in empty:
        assert value.shape[0] == 0

    celsius_name = _parameter_name(tensors.parameters, "celsius_param")
    base_celsius = tensors.parameters[celsius_name]

    def celsius_output(celsius):
        initialized = _initialized(
            functional,
            tensors,
            replacements={celsius_name: celsius},
        )
        return initialized.state["mechanism_buffers"][MECHANISM_NAME]["mechanism_carry"]

    celsius_lanes = torch.stack((base_celsius - 2.0, base_celsius + 2.0))
    actual_celsius = torch.vmap(celsius_output)(celsius_lanes)
    expected_celsius = torch.stack(
        tuple(celsius_output(value) for value in celsius_lanes)
    )
    torch.testing.assert_close(actual_celsius, expected_celsius)
    empty_celsius = torch.vmap(celsius_output)(celsius_lanes[:0])
    assert empty_celsius.shape == (0, *model.shape)


def test_compile_of_jacrev_over_authored_initial_values_matches_eager():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    gain_name = _parameter_name(
        tensors.parameters,
        f"{MECHANISM_NAME}.gain_param",
    )
    base_gain = tensors.parameters[gain_name]

    def projected(gain):
        initialized = _initialized(
            functional,
            tensors,
            replacements={gain_name: gain},
        )
        return initialized.state["mechanism_buffers"][MECHANISM_NAME]["mechanism_carry"]

    transformed = torch.func.jacrev(projected)
    expected = transformed(base_gain)
    compiled = torch.compile(
        transformed,
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )
    actual = compiled(base_gain)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-13)

    hessian = torch.func.hessian(lambda gain: projected(gain).square().sum())(base_gain)
    assert torch.isfinite(hessian).all()
    assert torch.count_nonzero(hessian) > 0


class _MalformedInitialValue(Mechanism):
    Mechanism.CARRY("carry")
    MODE = "valid"

    def initial_values(self, v, values):
        del values
        if self.MODE == "nonmapping":
            return (v,)
        if self.MODE == "unknown":
            return {"undeclared": v}
        if self.MODE == "nontensor":
            return {"carry": 1.0}
        if self.MODE == "shape":
            return {"carry": torch.zeros((*v.shape, 2), dtype=v.dtype)}
        if self.MODE == "dtype":
            return {"carry": torch.zeros_like(v, dtype=torch.float32)}
        return {"carry": torch.zeros_like(v)}


class _NonMappingInitialValue(_MalformedInitialValue):
    MODE = "nonmapping"


class _UnknownInitialValue(_MalformedInitialValue):
    MODE = "unknown"


class _NonTensorInitialValue(_MalformedInitialValue):
    MODE = "nontensor"


class _WrongShapeInitialValue(_MalformedInitialValue):
    MODE = "shape"


class _WrongDtypeInitialValue(_MalformedInitialValue):
    MODE = "dtype"


class _MutatingInitialValue(Mechanism):
    Mechanism.CARRY("carry")

    def initial_values(self, v, values):
        del values
        self.carry.add_(1.0)
        return {"carry": v}


class _InputMutatingInitialValue(Mechanism):
    Mechanism.CARRY("carry")

    def initial_values(self, v, values):
        del values
        v.add_(1.0)
        return {"carry": v}


class _NestedInputMutatingInitialValue(Mechanism):
    Mechanism.CARRY("carry")

    def initial_values(self, v, values):
        values["celsius"].add_(1.0)
        return {"carry": v}


class _RandomInitialValue(Mechanism):
    Mechanism.CARRY("carry")

    def initial_values(self, v, values):
        del values
        return {"carry": torch.rand_like(v)}


@pytest.mark.parametrize(
    ("mechanism", "exception", "match"),
    [
        pytest.param(
            _NonMappingInitialValue,
            TypeError,
            r"initial_values\(\) must return a mapping",
            id="nonmapping",
        ),
        pytest.param(
            _UnknownInitialValue,
            KeyError,
            "returned undeclared outputs",
            id="unknown-name",
        ),
        pytest.param(
            _NonTensorInitialValue,
            TypeError,
            "must be a Tensor",
            id="non-tensor",
        ),
        pytest.param(
            _WrongShapeInitialValue,
            ValueError,
            "not broadcastable to declared shape",
            id="wrong-shape",
        ),
        pytest.param(
            _WrongDtypeInitialValue,
            ValueError,
            "must use",
            id="wrong-dtype",
        ),
    ],
)
def test_malformed_authored_initial_values_fail_at_imperative_initialization(
    mechanism,
    exception,
    match,
):
    with pytest.raises(exception, match=match):
        _model(mechanism)


@pytest.mark.parametrize(
    ("mechanism", "match"),
    [
        pytest.param(
            _MutatingInitialValue,
            "must not mutate registered parameters, buffers, or tensor arguments",
            id="registered-carry",
        ),
        pytest.param(
            _InputMutatingInitialValue,
            "must not mutate registered parameters, buffers, or tensor arguments",
            id="voltage-input",
        ),
        pytest.param(
            _NestedInputMutatingInitialValue,
            "must not mutate registered parameters, buffers, or tensor arguments",
            id="nested-values-input",
        ),
        pytest.param(
            _RandomInitialValue,
            "must not consume implicit RNG state",
            id="implicit-rng",
        ),
    ],
)
def test_effectful_authored_initial_values_fail_without_mutating_the_source(
    mechanism,
    match,
):
    with pytest.raises(RuntimeError, match=match):
        _model(mechanism)


class _IncompleteInitialValueState(State):
    State.STATE("value", "explicit")
    State.CARRY("carry")
    State.DERIVATIVE(
        "value' = 0.0 * value",
        "explicit' = 0.0 * explicit",
    )

    def initial_values(self, v, values):
        del values
        return {"carry": torch.zeros_like(v)}


class _IncompleteInitialValueMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_IncompleteInitialValueState)


def test_authored_initial_values_must_complete_every_functional_state():
    model = _model(_IncompleteInitialValueMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="did not produce declared states.*value",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


class _DirectScalarCelsiusInitialValue(Mechanism):
    Mechanism.CARRY("carry")

    def initial_values(self, v, values):
        del values
        return {"carry": torch.zeros_like(v) + self.celsius}


def test_authored_initial_values_use_support_visible_celsius_from_values_mapping():
    model = _model(_DirectScalarCelsiusInitialValue)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"reads receiver attributes.*celsius.*values mapping",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )
