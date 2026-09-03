"""Generic declaration-driven functional-initialization contracts."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State

DT = 0.0125
MECHANISM_NAME = "_DeclaredInitializationMechanism"
DECLARED_KEY = f"mechanisms.{MECHANISM_NAME}.declared_value"
INSERTED_KEY = f"mechanisms.{MECHANISM_NAME}.insertion_value"
LAYOUTS = ("dense", "rectangular", "shared_columns", "packed", "duplicates")

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _VoltageInferredState(State):
    """Use deliberately non-HH state names and a differentiable workspace."""

    State.STATE("activation", "reserve")
    State.RANGE(initial_gain=0.02)
    State.DERIVED_BUFFER("initial_bias")
    State.DERIVATIVE(
        "activation' = 0.0 * activation",
        "reserve' = 0.0 * reserve",
    )

    def derive_buffers(self):
        return {
            "initial_bias": self.initial_gain * (1.0 + 0.01 * self.celsius)
            + 0.02 * self.diam
        }

    def state_defaults(self, v, values):
        return {
            "activation": torch.sigmoid(0.05 * v + self.initial_bias),
            "reserve": torch.tanh(-0.025 * v + 0.5 * self.initial_bias),
        }


class _ExplicitlyInitializedState(State):
    """Expose insertion-time explicit initial-state precedence."""

    State.STATE("declared_value", "insertion_value")
    State.DERIVATIVE(
        "declared_value' = 0.0 * declared_value",
        "insertion_value' = 0.0 * insertion_value",
    )

    def state_defaults(self, v, values):
        del v
        # Initialization does not evaluate ``state_defaults`` when every state
        # in this module has an explicit insertion-time ``ic`` value. The pure path
        # must preserve that precedence instead of evaluating then overlaying.
        raise AssertionError("explicit initialization must bypass state_defaults")


class _DeclaredInitializationMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_VoltageInferredState, _ExplicitlyInitializedState)
    Mechanism.GLOBAL(g=1.0e-5, e=-61.0)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * self.activation * (v - self.e)


class _RandomDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": torch.rand_like(v)}


class _RandomDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_RandomDefaultsState)


class _MutatingDefaultsState(State):
    calls = 0

    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        type(self).calls += 1
        return {"value": torch.zeros_like(v)}


class _MutatingDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_MutatingDefaultsState)


class _PartialDefaultsState(State):
    State.STATE("first", "second")
    State.DERIVATIVE(
        "first' = 0.0 * first",
        "second' = 0.0 * second",
    )

    def state_defaults(self, v, values):
        return {"first": torch.zeros_like(v)}


class _PartialDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_PartialDefaultsState)


class _DtAdvanceState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        del v
        return {"value": values["value"] + dt}


class _DtAdvanceMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_DtAdvanceState)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return 0.0 * v


_EXTERNAL_DEFAULTS_CALLS = []
_IMMUTABLE_DEFAULTS_GAIN = 1.0


class _ExternalMutationDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        _EXTERNAL_DEFAULTS_CALLS.append("called")
        return {"value": torch.zeros_like(v)}


class _ExternalMutationDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_ExternalMutationDefaultsState)


class _ImmutableGlobalDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": _IMMUTABLE_DEFAULTS_GAIN * torch.ones_like(v)}


class _ImmutableGlobalDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_ImmutableGlobalDefaultsState)


class _GradMetadataDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": torch.zeros_like(v) + float(v.requires_grad)}


class _GradMetadataDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_GradMetadataDefaultsState)


class _ScalarEscapeDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": torch.zeros_like(v) + float(v[0, 0])}


class _ScalarEscapeDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_ScalarEscapeDefaultsState)


_DYNAMIC_HELPER_CALLS = []


class _DynamicHelperDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def helper(self, v):
        _DYNAMIC_HELPER_CALLS.append("called")
        return {"value": torch.zeros_like(v)}

    def state_defaults(self, v, values):
        return getattr(self, "helper")(v)


class _DynamicHelperDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_DynamicHelperDefaultsState)


class _TensorControlDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        if torch.all(v > 0):
            return {"value": torch.ones_like(v)}
        return {"value": torch.zeros_like(v)}


class _TensorControlDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_TensorControlDefaultsState)


class _ForwardADContextDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        tangent = torch.autograd.forward_ad.unpack_dual(v).tangent
        offset = 0.0 if tangent is None else 1.0
        return {"value": torch.zeros_like(v) + offset}


class _ForwardADContextDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_ForwardADContextDefaultsState)


_ALIASED_INFERENCE_MODE_QUERY = torch.is_inference_mode_enabled
_ALIASED_UNPACK_DUAL = torch.autograd.forward_ad.unpack_dual


class _AliasedContextDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": torch.zeros_like(v) + _ALIASED_INFERENCE_MODE_QUERY()}


class _AliasedContextDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_AliasedContextDefaultsState)


class _AliasedForwardADDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        tangent = _ALIASED_UNPACK_DUAL(v).tangent
        offset = 0.0 if tangent is None else 1.0
        return {"value": torch.zeros_like(v) + offset}


class _AliasedForwardADDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_AliasedForwardADDefaultsState)


class _ReceiverDependencyDefaultsState(State):
    hidden_gain = 1.0

    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(source, v, values):
        return {"value": source.hidden_gain * torch.ones_like(v)}


class _ReceiverDependencyDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_ReceiverDependencyDefaultsState)


class _ReceiverAliasDefaultsState(State):
    hidden_gain = 1.0

    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        source = self
        return {"value": source.hidden_gain * torch.ones_like(v)}


class _ReceiverAliasDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_ReceiverAliasDefaultsState)


_TYPE_ALIAS = type


class _IndirectClassDependencyDefaultsState(State):
    hidden_gain = 1.0

    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": _TYPE_ALIAS(self).hidden_gain * torch.ones_like(v)}


class _IndirectClassDependencyDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_IndirectClassDependencyDefaultsState)


class _NestedHelperState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": torch.zeros_like(v)}

    def helper(self, v):
        return torch.ones_like(v)


class _NestedHelperMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_NestedHelperState)
    Mechanism.ASSIGNED("probe")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def assigned_values(self, v, values):
        del values
        return {"probe": self.DE["_NestedHelperState"].helper(v)}

    def i(self, v):
        return 0.0 * (v + self.probe)


class _NestedFunctionDefaultsState(State):
    hidden_gain = 1.0

    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        def read_gain():
            return self.hidden_gain

        return {"value": read_gain() * torch.ones_like(v)}


class _NestedFunctionDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_NestedFunctionDefaultsState)


def _function_attribute_helper(v):
    _function_attribute_helper.calls += 1
    return _function_attribute_helper.gain * torch.ones_like(v)


_function_attribute_helper.gain = 1.0
_function_attribute_helper.calls = 0


class _FunctionAttributeDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": _function_attribute_helper(v)}


class _FunctionAttributeDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_FunctionAttributeDefaultsState)


_BOUND_ROUTINE_CALLS = []
_BOUND_ROUTINE = _BOUND_ROUTINE_CALLS.append


class _BoundRoutineDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        _BOUND_ROUTINE("called")
        return {"value": torch.zeros_like(v)}


class _BoundRoutineDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_BoundRoutineDefaultsState)


_PROCESS_EFFECT_ENABLED = False
_ALIASED_SET_NUM_THREADS = torch.set_num_threads


class _ProcessEffectDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        if _PROCESS_EFFECT_ENABLED:
            _ALIASED_SET_NUM_THREADS(torch.get_num_threads())
        return {"value": torch.zeros_like(v)}


class _ProcessEffectDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_ProcessEffectDefaultsState)


_PRINT_INITIALIZATION = False


class _PrintDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        if _PRINT_INITIALIZATION:
            print("initializer side effect")
        return {"value": torch.zeros_like(v)}


class _PrintDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_PrintDefaultsState)


_WRONG_SHAPE_INITIALIZATION_ENABLED = False


class _WrongShapeCurrentMechanism(_DtAdvanceMechanism):
    Mechanism.EXPLICIT("i")

    def i(self, v):
        if _WRONG_SHAPE_INITIALIZATION_ENABLED and self.dt == 0:
            self.value = self.value[..., :1]
        return 0.0 * v


class _ConditionalGeometryMutationState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        if torch.all(v > 0):
            self.diam.add_(1.0)
        return {"value": torch.zeros_like(v)}


class _ConditionalGeometryMutationMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_ConditionalGeometryMutationState)


class _NoOpInplaceDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        v.clamp_(-100.0, 100.0)
        return {"value": torch.zeros_like(v)}


class _NoOpInplaceDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_NoOpInplaceDefaultsState)


class _ReshapeDefaultsState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": v.reshape(v.shape)}


class _ReshapeDefaultsMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_ReshapeDefaultsState)


_ANALYTIC_FRAME_MUTATION_ENABLED = False
_ANALYTIC_FRAME_CALLS = []


class _AnalyticFrameState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": torch.zeros_like(v)}


class _AnalyticFrameMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_AnalyticFrameState)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return 0.0 * v

    def i_with_conductance(self, v):
        if _ANALYTIC_FRAME_MUTATION_ENABLED:
            _ANALYTIC_FRAME_CALLS.append("called")
        return 0.0 * v, 0.0 * v


_NUMERICAL_VALIDATOR_CALLS = []


class _NumericalFrameState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0.0 * value")

    def state_defaults(self, v, values):
        return {"value": torch.zeros_like(v)}


class _NumericalFrameMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_NumericalFrameState)
    Mechanism.GLOBAL(g=0.01)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return self.g * v


class _CanonicalUnmyelinatedSubclass(dn.Unmyelinated):
    pass


class _AuthoredLifecycleUnmyelinated(dn.Unmyelinated):
    def pre_initialize(self):
        return super().pre_initialize()


def _insert_mechanism(model, layout, mechanism=_DeclaredInitializationMechanism):
    kwargs = {
        "ic": {
            "declared_value": -0.25,
            "insertion_value": 0.75,
        }
    }
    if layout == "dense":
        model.insert(mechanism, **kwargs)
    elif layout == "rectangular":
        model[:, 1:4].insert(mechanism, **kwargs)
    elif layout == "shared_columns":
        model[:, torch.tensor([1, 3])].insert(mechanism, **kwargs)
    elif layout == "packed":
        model[
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 4, 2]),
        ].insert(mechanism, **kwargs)
    elif layout == "duplicates":
        model[
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([1, 1, 3, 3]),
        ].insert(mechanism, preserve_multiplicity=True, **kwargs)
    else:  # pragma: no cover - private test helper
        raise AssertionError(f"unknown support layout {layout!r}")


def _model(layout="dense", *, dtype=torch.float64, mechanism=None):
    mechanism = mechanism or _DeclaredInitializationMechanism
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        if layout == "dense":
            model = dn.SingleCompartment(
                N=2,
                C=3,
                v_init=torch.tensor([-66.0, -62.0, -58.0], dtype=dtype),
                dtype=dtype,
                integrator=dn.bwd_euler_sc(imem=False),
            )
        else:
            model = dn.Unmyelinated(
                [1.5, 2.25],
                L=4.0,
                dx=1.0,
                v_init=torch.tensor(
                    [-67.0, -64.0, -61.0, -58.0, -55.0],
                    dtype=dtype,
                ),
                dtype=dtype,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
        _insert_mechanism(model, layout, mechanism)
        model.initialize()
        model.train()
    return model


def _model_with_defaults(mechanism):
    partial_defaults = None
    if mechanism is _PartialDefaultsMechanism:
        # Build a valid source first, then restore the deliberately incomplete
        # callable so functional admission—not imperative construction—is the
        # operation under test.
        partial_defaults = _PartialDefaultsState.state_defaults
        _PartialDefaultsState.state_defaults = lambda self, v, values: {
            "first": torch.zeros_like(v),
            "second": torch.zeros_like(v),
        }
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        try:
            model = dn.SingleCompartment(
                N=1,
                C=2,
                dtype=torch.float64,
                integrator=dn.bwd_euler_sc(imem=False),
            )
            model.insert(mechanism)
            model.initialize()
            model.train()
        finally:
            if partial_defaults is not None:
                _PartialDefaultsState.state_defaults = partial_defaults
    return model


def _unmyelinated_subclass_model(population_type, *, v_init=-63.0):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = population_type(
            [1.5, 2.25],
            L=4.0,
            dx=1.0,
            v_init=v_init,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(
            _DeclaredInitializationMechanism,
            ic={"declared_value": -0.25, "insertion_value": 0.75},
        )
        model.initialize()
        model.train()
    return model


def _new_initial_voltage(model):
    return torch.linspace(
        -71.0,
        -53.0,
        model.v.numel(),
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(model.shape)


def _mechanism(model):
    return getattr(model.mech, MECHANISM_NAME)


def _parameter_name(parameters, suffix):
    matches = [name for name in parameters if name.endswith(suffix)]
    assert len(matches) == 1
    return matches[0]


def _local_field(field, mechanism):
    indices = mechanism.support_spec.materialized_flat_indices(
        mechanism.key,
        device=field.device,
    )
    return (
        field.reshape(-1).index_select(0, indices).reshape(mechanism.activation.shape)
    )


def _assert_tree_close(actual, expected, *, rtol=0.0, atol=0.0):
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        torch.testing.assert_close(actual_leaf, expected_leaf, rtol=rtol, atol=atol)


def _tensor_snapshot(value):
    return (
        id(value),
        value.untyped_storage().data_ptr(),
        value._version,
        value.detach().clone(),
    )


def _assert_tensor_snapshot(value, snapshot):
    identity, storage, version, expected = snapshot
    assert id(value) == identity
    assert value.untyped_storage().data_ptr() == storage
    assert value._version == version
    torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)


def _mapping_snapshot(values: Mapping[str, torch.Tensor]):
    return {name: _tensor_snapshot(value) for name, value in values.items()}


def _assert_mapping_snapshot(values, snapshot):
    assert tuple(values) == tuple(snapshot)
    for name, value in values.items():
        _assert_tensor_snapshot(value, snapshot[name])


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_declared_state_initialization_matches_imperative_for_every_support(
    layout,
    dtype,
):
    model = _model(layout, dtype=dtype)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    v_init = _new_initial_voltage(model)
    inputs = dn.func.InitializationInput(
        v_init=v_init,
        states=tensors.initialization.states,
    )

    actual = functional.initialize(tensors.parameters, tensors.constants, inputs)
    assert actual.initialization.v_init is v_init
    for name, value in inputs.states.items():
        assert actual.initialization.states[name] is value

    reference = _model(layout, dtype=dtype)
    reference.set_v_init(v_init)
    reference.initialize()
    expected = functional.extract(reference)
    _assert_tree_close(actual.state, expected.state)

    mechanism_state = actual.state["mechanisms"][MECHANISM_NAME]
    assert set(mechanism_state) == {
        "activation",
        "reserve",
        "declared_value",
        "insertion_value",
    }
    torch.testing.assert_close(
        mechanism_state["declared_value"],
        torch.full_like(mechanism_state["declared_value"], -0.25),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        mechanism_state["insertion_value"],
        torch.full_like(mechanism_state["insertion_value"], 0.75),
        rtol=0.0,
        atol=0.0,
    )


def test_canonical_unmyelinated_subclass_initializes_purely_and_differentiates():
    source = _unmyelinated_subclass_model(_CanonicalUnmyelinatedSubclass)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    v_init = _new_initial_voltage(source)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(v_init=v_init),
    )

    reference = _unmyelinated_subclass_model(
        _CanonicalUnmyelinatedSubclass,
        v_init=v_init,
    )
    _assert_tree_close(initialized.state, functional.extract(reference).state)

    gain_name = _parameter_name(tensors.parameters, "initial_gain_param")

    def activation(voltage, raw_gain):
        parameters = dict(tensors.parameters)
        parameters[gain_name] = raw_gain
        result = functional.initialize(
            parameters,
            tensors.constants,
            tensors.initialization._replace(v_init=voltage),
        )
        return result.state["mechanisms"][MECHANISM_NAME]["activation"]

    arguments = (v_init, tensors.parameters[gain_name])
    reverse = torch.func.jacrev(activation, argnums=(0, 1))(*arguments)
    forward = torch.func.jacfwd(activation, argnums=(0, 1))(*arguments)
    for reverse_jacobian, forward_jacobian in zip(reverse, forward, strict=True):
        assert torch.isfinite(reverse_jacobian).all()
        assert torch.count_nonzero(reverse_jacobian) > 0
        torch.testing.assert_close(
            reverse_jacobian,
            forward_jacobian,
            rtol=2.0e-10,
            atol=2.0e-11,
        )


def test_unmyelinated_subclass_with_authored_lifecycle_remains_fail_closed():
    model = _unmyelinated_subclass_model(_AuthoredLifecycleUnmyelinated)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"Population replaces canonical fresh-initialization hooks.*pre_initialize",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    next_state, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    assert set(next_state) == set(tensors.state)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_extraction_exposes_only_explicit_insertion_initial_state_leaves(layout):
    model = _model(layout)
    _functional, tensors = dn.func.make_functional(model, dt=DT)
    mechanism = _mechanism(model)

    assert set(tensors.initialization.states) == {DECLARED_KEY, INSERTED_KEY}
    expected = {
        DECLARED_KEY: torch.full_like(mechanism.declared_value, -0.25),
        INSERTED_KEY: torch.full_like(mechanism.insertion_value, 0.75),
    }
    _assert_tree_close(tensors.initialization.states, expected)
    for name, value in tensors.initialization.states.items():
        assert value.shape == getattr(mechanism, name.rsplit(".", 1)[-1]).shape
        assert value.dtype == model.dtype()
        assert value.device == model.device()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_inferred_states_follow_substituted_parameters_temperature_and_geometry(layout):
    model = _model(layout)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    mechanism = _mechanism(model)
    gain_name = _parameter_name(
        tensors.parameters,
        "DE._VoltageInferredState.initial_gain_param",
    )
    v_init = _new_initial_voltage(model)
    gain = tensors.parameters[gain_name] * 1.7
    celsius = tensors.parameters["celsius_param"] + 4.0
    diam = tensors.constants["diam"] * 1.25
    parameters = dict(tensors.parameters)
    parameters[gain_name] = gain
    parameters["celsius_param"] = celsius
    constants = dict(tensors.constants)
    constants["diam"] = diam

    initialized = functional.initialize(
        parameters,
        constants,
        dn.func.InitializationInput(
            v_init=v_init,
            states=tensors.initialization.states,
        ),
    )
    state = initialized.state["mechanisms"][MECHANISM_NAME]
    local_v = _local_field(v_init, mechanism)
    local_diam = _local_field(diam, mechanism)
    initial_bias = gain * (1.0 + 0.01 * celsius) + 0.02 * local_diam
    torch.testing.assert_close(
        state["activation"],
        torch.sigmoid(0.05 * local_v + initial_bias),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state["reserve"],
        torch.tanh(-0.025 * local_v + 0.5 * initial_bias),
        rtol=0.0,
        atol=0.0,
    )

    def loss(local_gain, local_celsius, full_diam):
        local_parameters = dict(parameters)
        local_parameters[gain_name] = local_gain
        local_parameters["celsius_param"] = local_celsius
        local_constants = dict(constants)
        local_constants["diam"] = full_diam
        local = functional.initialize(
            local_parameters,
            local_constants,
            dn.func.InitializationInput(
                v_init=v_init,
                states=tensors.initialization.states,
            ),
        )
        return local.state["mechanisms"][MECHANISM_NAME]["activation"].mean()

    gradients = torch.func.grad(loss, argnums=(0, 1, 2))(gain, celsius, diam)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("layout", ["dense", "packed"])
@pytest.mark.parametrize("key", [DECLARED_KEY, INSERTED_KEY])
def test_explicit_initial_state_inputs_are_replaceable_and_differentiable(layout, key):
    model = _model(layout)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    state_name = key.rsplit(".", 1)[-1]
    replacement = torch.linspace(
        -0.4,
        0.4,
        getattr(_mechanism(model), state_name).numel(),
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(getattr(_mechanism(model), state_name).shape)
    explicit = dict(tensors.initialization.states)
    explicit[key] = replacement
    inputs = dn.func.InitializationInput(
        v_init=tensors.initialization.v_init,
        states=explicit,
    )

    initialized = functional.initialize(tensors.parameters, tensors.constants, inputs)
    actual = initialized.state["mechanisms"][MECHANISM_NAME][state_name]
    torch.testing.assert_close(actual, replacement, rtol=0.0, atol=0.0)
    assert actual is not replacement
    assert (
        actual.untyped_storage().data_ptr() != replacement.untyped_storage().data_ptr()
    )

    jacobian = torch.func.jacrev(
        lambda value: functional.initialize(
            tensors.parameters,
            tensors.constants,
            dn.func.InitializationInput(
                v_init=tensors.initialization.v_init,
                states={**explicit, key: value},
            ),
        ).state["mechanisms"][MECHANISM_NAME][state_name]
    )(replacement)
    torch.testing.assert_close(
        jacobian.reshape(replacement.numel(), replacement.numel()),
        torch.eye(
            replacement.numel(),
            dtype=replacement.dtype,
            device=replacement.device,
        ),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("layout", ["dense", "packed"])
def test_generic_initialization_supports_func_transforms_and_derived_dependencies(
    layout,
):
    model = _model(layout)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    gain_name = _parameter_name(
        tensors.parameters,
        "DE._VoltageInferredState.initial_gain_param",
    )
    base_v = _new_initial_voltage(model)
    base_gain = tensors.parameters[gain_name]
    base_explicit = tensors.initialization.states[DECLARED_KEY]

    def initialized_values(v_init, gain, explicit_value):
        parameters = dict(tensors.parameters)
        parameters[gain_name] = gain
        explicit = dict(tensors.initialization.states)
        explicit[DECLARED_KEY] = explicit_value
        initialized = functional.initialize(
            parameters,
            tensors.constants,
            dn.func.InitializationInput(v_init=v_init, states=explicit),
        )
        state = initialized.state["mechanisms"][MECHANISM_NAME]
        return state["activation"] + 0.125 * state["declared_value"]

    reverse = torch.func.jacrev(initialized_values, argnums=(0, 1, 2))(
        base_v,
        base_gain,
        base_explicit,
    )
    forward = torch.func.jacfwd(initialized_values, argnums=(0, 1, 2))(
        base_v,
        base_gain,
        base_explicit,
    )
    _assert_tree_close(reverse, forward, rtol=2.0e-10, atol=2.0e-11)
    for derivative in reverse:
        assert torch.isfinite(derivative).all()
        assert torch.count_nonzero(derivative) > 0

    def loss(v_init):
        value = initialized_values(v_init, base_gain, base_explicit)
        return value.square().mean()

    gradient, value = torch.func.grad_and_value(loss)(base_v)
    hessian = torch.func.hessian(loss)(base_v)
    assert value.ndim == 0
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert torch.isfinite(hessian).all()
    assert torch.count_nonzero(hessian) > 0
    hessian_matrix = hessian.reshape(base_v.numel(), base_v.numel())
    torch.testing.assert_close(
        hessian_matrix,
        hessian_matrix.mT,
        rtol=2.0e-10,
        atol=2.0e-11,
    )


@pytest.mark.parametrize("layout", ["dense", "packed"])
def test_generic_initialization_vmap_matches_explicit_lanes(layout):
    model = _model(layout)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    gain_name = _parameter_name(
        tensors.parameters,
        "DE._VoltageInferredState.initial_gain_param",
    )
    base_v = _new_initial_voltage(model)
    base_gain = tensors.parameters[gain_name]
    base_explicit = tensors.initialization.states[DECLARED_KEY]

    def lane(v_init, gain, explicit_value):
        parameters = dict(tensors.parameters)
        parameters[gain_name] = gain
        states = dict(tensors.initialization.states)
        states[DECLARED_KEY] = explicit_value
        initialized = functional.initialize(
            parameters,
            tensors.constants,
            dn.func.InitializationInput(v_init=v_init, states=states),
        )
        return initialized.state["mechanisms"][MECHANISM_NAME]

    offsets = base_v.new_tensor([-1.0, 0.0, 1.0])
    scales = base_gain.new_tensor([0.8, 1.0, 1.2])
    voltage_lanes = base_v.unsqueeze(0) + offsets.reshape(3, *([1] * base_v.ndim))
    gain_lanes = base_gain * scales
    explicit_lanes = torch.stack(
        [base_explicit - 0.1, base_explicit, base_explicit + 0.1]
    )
    actual = torch.vmap(lane)(voltage_lanes, gain_lanes, explicit_lanes)
    expected = torch.utils._pytree.tree_map(
        lambda *values: torch.stack(values),
        *(lane(voltage_lanes[i], gain_lanes[i], explicit_lanes[i]) for i in range(3)),
    )
    _assert_tree_close(actual, expected, rtol=0.0, atol=0.0)

    empty = torch.vmap(lane)(
        voltage_lanes[:0],
        gain_lanes[:0],
        explicit_lanes[:0],
    )
    for value in empty.values():
        assert value.shape[0] == 0


def test_compile_of_generic_initialization_matches_eager():
    model = _model("packed")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    base_v = _new_initial_voltage(model)
    base_explicit = tensors.initialization.states[DECLARED_KEY]

    def initialized_values(v_init, explicit_value):
        states = dict(tensors.initialization.states)
        states[DECLARED_KEY] = explicit_value
        initialized = functional.initialize(
            tensors.parameters,
            tensors.constants,
            dn.func.InitializationInput(v_init=v_init, states=states),
        )
        mechanism_state = initialized.state["mechanisms"][MECHANISM_NAME]
        return mechanism_state["activation"] + mechanism_state["declared_value"]

    expected = initialized_values(base_v, base_explicit)
    compiled = torch.compile(
        initialized_values,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    actual = compiled(base_v, base_explicit)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_generic_initialization_is_pure_over_all_explicit_inputs():
    model = _model("packed")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    v_init = _new_initial_voltage(model).requires_grad_()
    states = {
        name: value.detach().clone().requires_grad_()
        for name, value in tensors.initialization.states.items()
    }
    inputs = dn.func.InitializationInput(v_init=v_init, states=states)
    source_parameters = _mapping_snapshot(
        dict(model.named_parameters(remove_duplicate=False))
    )
    source_buffers = _mapping_snapshot(
        dict(model.named_buffers(remove_duplicate=False))
    )
    parameters = _mapping_snapshot(tensors.parameters)
    constants = _mapping_snapshot(tensors.constants)
    explicit_states = _mapping_snapshot(states)
    voltage = _tensor_snapshot(v_init)
    rng = torch.random.get_rng_state().clone()

    first = functional.initialize(tensors.parameters, tensors.constants, inputs)
    second = functional.initialize(tensors.parameters, tensors.constants, inputs)
    _assert_tree_close(first.state, second.state, rtol=0.0, atol=0.0)

    _assert_mapping_snapshot(
        dict(model.named_parameters(remove_duplicate=False)), source_parameters
    )
    _assert_mapping_snapshot(
        dict(model.named_buffers(remove_duplicate=False)), source_buffers
    )
    _assert_mapping_snapshot(tensors.parameters, parameters)
    _assert_mapping_snapshot(tensors.constants, constants)
    _assert_mapping_snapshot(states, explicit_states)
    _assert_tensor_snapshot(v_init, voltage)
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape", "dtype"])
def test_explicit_initial_state_inputs_are_validated_exactly(mutation):
    model = _model("packed")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    states = dict(tensors.initialization.states)
    if mutation == "missing":
        states.pop(DECLARED_KEY)
    elif mutation == "unexpected":
        states[f"mechanisms.{MECHANISM_NAME}.unknown"] = states[DECLARED_KEY]
    elif mutation == "shape":
        states[DECLARED_KEY] = states[DECLARED_KEY].reshape(-1)[:1]
    else:
        states[DECLARED_KEY] = states[DECLARED_KEY].float()

    with pytest.raises((KeyError, ValueError), match="initial|state|shape|dtype"):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            dn.func.InitializationInput(
                v_init=tensors.initialization.v_init,
                states=states,
            ),
        )


@pytest.mark.parametrize(
    ("mechanism", "message"),
    [
        (_RandomDefaultsMechanism, r"RNG|random"),
        (_MutatingDefaultsMechanism, r"class state|mutation|dynamic-dispatch|type"),
        (_PartialDefaultsMechanism, r"did not return|could not be evaluated"),
    ],
)
def test_initializer_purity_audit_disables_only_unsafe_initialization(
    mechanism,
    message,
):
    model = _model_with_defaults(mechanism)
    rng = torch.random.get_rng_state().clone()
    calls = _MutatingDefaultsState.calls

    functional, tensors = dn.func.make_functional(model, dt=DT)

    assert torch.equal(torch.random.get_rng_state(), rng)
    assert _MutatingDefaultsState.calls == calls
    with pytest.raises(dn.func.FunctionalizationError, match=message):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )

    # Transition execution remains available because initializer admission is
    # deliberately a narrower capability layered over functional stepping.
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    next_state, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    assert set(next_state) == set(tensors.state)


def test_functional_initialization_does_not_run_accepted_step_advance():
    model = _model_with_defaults(_DtAdvanceMechanism)
    model.mech.set_dt(0.125)
    functional, tensors = dn.func.make_functional(model, dt=0.125)

    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
    )
    value = initialized.state["mechanisms"]["_DtAdvanceMechanism"]["value"]

    torch.testing.assert_close(value, torch.zeros_like(value), rtol=0.0, atol=0.0)

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    advanced, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        initialized.state,
    )
    torch.testing.assert_close(
        advanced["mechanisms"]["_DtAdvanceMechanism"]["value"],
        torch.full_like(value, 0.125),
        rtol=0.0,
        atol=0.0,
    )


def test_reinitializing_advanced_population_restores_fresh_hook_clock():
    model = _model_with_defaults(_DtAdvanceMechanism)
    mechanism = model.mech.mechanisms["_DtAdvanceMechanism"]
    model.t.fill_(0.25)
    model.mech.set_dt(0.125)

    model.initialize()

    torch.testing.assert_close(
        mechanism.value,
        torch.zeros_like(mechanism.value),
        rtol=0.0,
        atol=0.0,
    )
    assert model.t.item() == 0.0
    assert mechanism.dt.item() == 0.0


def test_mutable_module_global_disables_initializer_without_escaping_audit():
    model = _model_with_defaults(_ExternalMutationDefaultsMechanism)
    _EXTERNAL_DEFAULTS_CALLS.clear()

    functional, tensors = dn.func.make_functional(model, dt=DT)

    assert _EXTERNAL_DEFAULTS_CALLS == []
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"mutable external Python state|functional-initialization dependency",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )
    assert _EXTERNAL_DEFAULTS_CALLS == []

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    functional.step(tensors.parameters, prepared, tensors.state)


def test_immutable_global_rebinding_invalidates_only_initializer_freshness():
    global _IMMUTABLE_DEFAULTS_GAIN

    model = _model_with_defaults(_ImmutableGlobalDefaultsMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    original = _IMMUTABLE_DEFAULTS_GAIN
    try:
        _IMMUTABLE_DEFAULTS_GAIN = original + 1.0
        with pytest.raises(
            dn.func.FunctionalizationError,
            match=r"initialization structure changed|state_defaults",
        ):
            functional.initialize(
                tensors.parameters,
                tensors.constants,
                tensors.initialization,
            )

        prepared = functional.prepare(tensors.parameters, tensors.constants)
        functional.step(tensors.parameters, prepared, tensors.state)
    finally:
        _IMMUTABLE_DEFAULTS_GAIN = original


def test_initializer_audit_rejects_autograd_context_dependent_outputs():
    model = _model_with_defaults(_GradMetadataDefaultsMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"autograd metadata|grad-enabled|execution-context|requires_grad",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    functional.step(tensors.parameters, prepared, tensors.state)


@pytest.mark.parametrize(
    ("mechanism", "message"),
    [
        (_ScalarEscapeDefaultsMechanism, r"tensor-to-Python|float"),
        (
            _TensorControlDefaultsMechanism,
            r"vmap|data-dependent|could not be evaluated",
        ),
        (
            _ForwardADContextDefaultsMechanism,
            r"JVP|jvp|torch\.func|execution-context|unpack_dual",
        ),
        (
            _AliasedContextDefaultsMechanism,
            r"execution-context|is_inference_mode_enabled",
        ),
        (
            _AliasedForwardADDefaultsMechanism,
            r"execution-context|unpack_dual",
        ),
        (
            _IndirectClassDependencyDefaultsMechanism,
            r"dynamic-dispatch|tensor-to-Python|type",
        ),
        (_NoOpInplaceDefaultsMechanism, r"in-place|clamp_"),
        (_NestedFunctionDefaultsMechanism, r"nested function|comprehension"),
        (_ReceiverAliasDefaultsMechanism, r"receiver escape|bound receiver"),
        (_ProcessEffectDefaultsMechanism, r"set_num_threads|in-place"),
        (_PrintDefaultsMechanism, r"builtin|print"),
    ],
)
def test_initializer_rejects_transform_unsafe_state_defaultserence(mechanism, message):
    model = _model_with_defaults(mechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    with pytest.raises(dn.func.FunctionalizationError, match=message):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    functional.step(tensors.parameters, prepared, tensors.state)


def test_dynamic_getattr_cannot_hide_initializer_helper_side_effect():
    model = _model_with_defaults(_DynamicHelperDefaultsMechanism)
    _DYNAMIC_HELPER_CALLS.clear()

    functional, tensors = dn.func.make_functional(model, dt=DT)

    assert _DYNAMIC_HELPER_CALLS == []
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"dynamic-dispatch|getattr",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )
    assert _DYNAMIC_HELPER_CALLS == []


def test_immutable_receiver_dependency_rebinding_invalidates_initializer():
    model = _model_with_defaults(_ReceiverDependencyDefaultsMechanism)
    state = model.mech.mechanisms["_ReceiverDependencyDefaultsMechanism"].DE[
        "_ReceiverDependencyDefaultsState"
    ]
    functional, tensors = dn.func.make_functional(model, dt=DT)

    state.hidden_gain = 2.0

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"initialization structure changed|state_defaults",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    functional.step(tensors.parameters, prepared, tensors.state)


def test_immutable_class_receiver_rebinding_invalidates_initializer():
    original = _ReceiverDependencyDefaultsState.hidden_gain
    model = _model_with_defaults(_ReceiverDependencyDefaultsMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    try:
        _ReceiverDependencyDefaultsState.hidden_gain = original + 1.0
        with pytest.raises(
            dn.func.FunctionalizationError,
            match=r"initialization structure changed|state_defaults",
        ):
            functional.initialize(
                tensors.parameters,
                tensors.constants,
                tensors.initialization,
            )
    finally:
        _ReceiverDependencyDefaultsState.hidden_gain = original


def test_authored_function_attributes_fail_closed_without_audit_side_effects():
    model = _model_with_defaults(_FunctionAttributeDefaultsMechanism)
    _function_attribute_helper.calls = 0
    functional, tensors = dn.func.make_functional(model, dt=DT)

    assert _function_attribute_helper.calls == 0
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"function-object state|__dict__|attribute write",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )
    assert _function_attribute_helper.calls == 0


def test_bound_external_routine_is_rejected_without_running_audit_side_effects():
    model = _model_with_defaults(_BoundRoutineDefaultsMechanism)
    _BOUND_ROUTINE_CALLS.clear()
    functional, tensors = dn.func.make_functional(model, dt=DT)

    assert _BOUND_ROUTINE_CALLS == []
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"bound routine|external receiver",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )
    assert _BOUND_ROUTINE_CALLS == []


def test_nested_registered_module_initializer_dependencies_fail_closed():
    model = _model_with_defaults(_NestedHelperMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"registered child module|nested callable|\.DE",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


def test_exact_analytic_final_current_dependencies_are_checked_without_execution():
    model = _model_with_defaults(_AnalyticFrameMechanism)
    _ANALYTIC_FRAME_CALLS.clear()

    functional, tensors = dn.func.make_functional(model, dt=DT)

    assert _ANALYTIC_FRAME_CALLS == []
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"mutable external Python state|i_with_g",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )
    assert _ANALYTIC_FRAME_CALLS == []


def test_generated_numerical_validator_rebinding_invalidates_without_execution(
    monkeypatch,
):
    model = _model_with_defaults(_NumericalFrameMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    mechanism = model.mech.mechanisms["_NumericalFrameMechanism"]
    generated = mechanism.i_with_g.__func__

    def effectful_validator(*args, **kwargs):
        del args, kwargs
        _NUMERICAL_VALIDATOR_CALLS.append("called")

    _NUMERICAL_VALIDATOR_CALLS.clear()
    monkeypatch.setitem(
        generated.__globals__,
        "validate_declared_numerical_current",
        effectful_validator,
    )

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"generated numerical|initialization structure|lower it again|validator",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )
    assert _NUMERICAL_VALIDATOR_CALLS == []


def test_tensor_native_receiver_method_in_state_defaults_remains_admitted():
    model = _model_with_defaults(_ReshapeDefaultsMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    v_init = _new_initial_voltage(model)

    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        dn.func.InitializationInput(
            v_init=v_init,
            states=tensors.initialization.states,
        ),
    )

    value = initialized.state["mechanisms"]["_ReshapeDefaultsMechanism"]["value"]
    torch.testing.assert_close(value, v_init, rtol=0.0, atol=0.0)


def test_initializer_audit_rejects_wrong_output_schema():
    global _WRONG_SHAPE_INITIALIZATION_ENABLED

    model = _model_with_defaults(_WrongShapeCurrentMechanism)
    try:
        _WRONG_SHAPE_INITIALIZATION_ENABLED = True
        functional, tensors = dn.func.make_functional(model, dt=DT)
    finally:
        _WRONG_SHAPE_INITIALIZATION_ENABLED = False

    with pytest.raises(dn.func.FunctionalizationError, match=r"schema|shape"):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    functional.step(tensors.parameters, prepared, tensors.state)


def test_initializer_rejects_inplace_geometry_mutation_without_touching_input():
    model = _model_with_defaults(_ConditionalGeometryMutationMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    diam = tensors.constants["diam"]
    snapshot = _tensor_snapshot(diam)
    positive_v = torch.ones_like(tensors.initialization.v_init)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"in-place|add_",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            dn.func.InitializationInput(
                v_init=positive_v,
                states=tensors.initialization.states,
            ),
        )

    _assert_tensor_snapshot(diam, snapshot)


def test_functional_initialization_normalizes_noncontiguous_voltage_like_imperative():
    model = _model("dense")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    v_init = (
        torch.linspace(
            -70.0,
            -55.0,
            6,
            dtype=model.dtype(),
            device=model.device(),
        )
        .reshape(3, 2)
        .mT
    )
    assert v_init.shape == model.shape
    assert not v_init.is_contiguous()

    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        dn.func.InitializationInput(
            v_init=v_init,
            states=tensors.initialization.states,
        ),
    )
    reference = _model("dense")
    reference.set_v_init(v_init)
    reference.initialize()
    expected = functional.extract(reference)

    _assert_tree_close(initialized.state, expected.state, rtol=0.0, atol=0.0)
    assert initialized.state["integrator"]["v"].is_contiguous()


def test_imperative_initialization_preserves_partial_state_defaults_failure():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=2,
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(_PartialDefaultsMechanism)

    with pytest.raises(KeyError, match=r"did not return declared states|second"):
        model.initialize()


@pytest.mark.parametrize("mutation", ["state_defaults", "init_keys"])
def test_initializer_rechecks_its_own_source_structure_without_widening_transition(
    mutation,
):
    model = _model("dense")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    mechanism = _mechanism(model)
    if mutation == "state_defaults":
        mechanism.DE._VoltageInferredState.state_defaults = lambda v, values: {
            "activation": torch.zeros_like(v),
            "reserve": torch.zeros_like(v),
        }
    else:
        mechanism._init_params.pop("declared_value")

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"initialization structure changed|initial-state keys|ic keys|state_defaults",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    next_state, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    assert set(next_state) == set(tensors.state)
