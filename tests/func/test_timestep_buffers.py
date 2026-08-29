"""Functional contracts for pure timestep-derived mechanism workspaces."""

from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import pas

DT = 0.03125
ALT_DT = 0.078125
STEPS = 3
SCALE = 0.375
RATE = 1.75


class _TimestepAccumulator(Mechanism):
    """Expose an exactly soluble use of one timestep-derived workspace."""

    Mechanism.GLOBAL(scale=SCALE, rate=RATE)
    Mechanism.CARRY("accumulator")
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_timestep_buffers(self, dt):
        # Authors may return any shape broadcastable to shape_p. Dendra owns
        # expansion into the canonical registered-buffer layout, including for
        # scalar results inside empty outer vmap lanes.
        return {"coefficient": self.scale + self.rate * dt}

    def initial_values(self, v, values):
        del v
        return {"accumulator": torch.zeros_like(values["accumulator"])}

    def advance(self, v, dt, values):
        del v, dt
        return {"accumulator": values["accumulator"] + values["coefficient"]}


class _InheritedTimestepAccumulator(_TimestepAccumulator):
    """A child extends, rather than replaces, its parent's declaration."""

    Mechanism.TIMESTEP_BUFFER("child_coefficient")

    def derive_timestep_buffers(self, dt):
        coefficient = (self.scale + self.rate * dt) * torch.ones_like(self.diam)
        return {
            "coefficient": coefficient,
            "child_coefficient": 2.0 * coefficient,
        }


class _DroppingInheritedDeclaration(_TimestepAccumulator):
    Mechanism.TIMESTEP_BUFFER("child_coefficient")

    def derive_timestep_buffers(self, dt):
        return {"child_coefficient": dt * torch.ones_like(self.diam)}


class _MissingTimestepKey(Mechanism):
    Mechanism.TIMESTEP_BUFFER("first", "second")

    def derive_timestep_buffers(self, dt):
        return {"first": dt * torch.ones_like(self.diam)}


class _UnexpectedTimestepKey(Mechanism):
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_timestep_buffers(self, dt):
        value = dt * torch.ones_like(self.diam)
        return {"coefficient": value, "surprise": value}


class _NonTensorTimestepValue(Mechanism):
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_timestep_buffers(self, dt):
        del dt
        return {"coefficient": 1.0}


class _WrongShapeTimestepValue(Mechanism):
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_timestep_buffers(self, dt):
        return {"coefficient": dt.new_ones((2, 3, 4))}


class _WrongDtypeTimestepValue(Mechanism):
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_timestep_buffers(self, dt):
        return {"coefficient": torch.ones_like(self.diam, dtype=torch.float32) * dt}


class _MutatingTimestepBuilder(Mechanism):
    Mechanism.CARRY("scratch")
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_timestep_buffers(self, dt):
        self.scratch.add_(1.0)
        return {"coefficient": dt * torch.ones_like(self.diam)}


class _RebindingRegisteredTimestepBuilder(Mechanism):
    Mechanism.CARRY("scratch")
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_timestep_buffers(self, dt):
        self.scratch = self.scratch + 1.0
        return {"coefficient": dt * torch.ones_like(self.diam)}


class _MutatingTimestepArgument(_TimestepAccumulator):
    def derive_timestep_buffers(self, dt):
        dt.add_(1.0)
        return super().derive_timestep_buffers(dt)


class _PythonStateMutatingTimestepBuilder(_TimestepAccumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.builder_calls = 0

    def derive_timestep_buffers(self, dt):
        self.builder_calls += 1
        return super().derive_timestep_buffers(dt)


class _ClassStateMutatingTimestepBuilder(_TimestepAccumulator):
    builder_calls = 0

    def derive_timestep_buffers(self, dt):
        _ClassStateMutatingTimestepBuilder.builder_calls += 1
        return super().derive_timestep_buffers(dt)


class _ClassTensorResizingTimestepBuilder(_TimestepAccumulator):
    base = torch.arange(8.0, dtype=torch.float64)
    shared = base.as_strided((2,), (2,), 1)

    def derive_timestep_buffers(self, dt):
        _ClassTensorResizingTimestepBuilder.shared.resize_(12)
        return super().derive_timestep_buffers(dt)


class _ClassTensorMutatingAndRebindingTimestepBuilder(_TimestepAccumulator):
    base = torch.arange(8.0, dtype=torch.float64)
    shared = base.as_strided((2,), (2,), 1)

    def derive_timestep_buffers(self, dt):
        _ClassTensorMutatingAndRebindingTimestepBuilder.shared.add_(10.0)
        _ClassTensorMutatingAndRebindingTimestepBuilder.shared = torch.zeros(1)
        return super().derive_timestep_buffers(dt)


class _ClassTensorMetadataTimestepBuilder(_TimestepAccumulator):
    base = torch.arange(6.0, dtype=torch.float32)
    shared = base.as_strided((2,), (2,), 1)

    def derive_timestep_buffers(self, dt):
        shared = _ClassTensorMetadataTimestepBuilder.shared
        shared.data = shared.to(dtype=torch.float64)
        shared.requires_grad_(True)
        return super().derive_timestep_buffers(dt)


class _ExpandedClassTensorTimestepBuilder(_TimestepAccumulator):
    shared = torch.ones(1, dtype=torch.float64).expand(3)


class _ClassArrayResizingTimestepBuilder(_TimestepAccumulator):
    shared = np.asarray([1.0, 3.0], dtype=np.float64)

    def derive_timestep_buffers(self, dt):
        _ClassArrayResizingTimestepBuilder.shared.resize((4,), refcheck=False)
        return super().derive_timestep_buffers(dt)


class _ClassArrayMutatingAndRebindingTimestepBuilder(_TimestepAccumulator):
    shared = np.asarray([1.0, 3.0], dtype=np.float64)

    def derive_timestep_buffers(self, dt):
        _ClassArrayMutatingAndRebindingTimestepBuilder.shared += 10.0
        _ClassArrayMutatingAndRebindingTimestepBuilder.shared = np.zeros(1)
        return super().derive_timestep_buffers(dt)


class _GlobalRngConsumingTimestepBuilder(_TimestepAccumulator):
    def derive_timestep_buffers(self, dt):
        sample = torch.rand((), device=self.diam.device, dtype=self.diam.dtype)
        value = super().derive_timestep_buffers(dt)["coefficient"]
        return {"coefficient": value + 0.0 * sample}


class _RandomParameterWorkspace(Mechanism):
    Mechanism.GLOBALRAND(
        "global_sample",
        distribution="normal",
        mu=0.25,
        sigma=0.1,
        seed=101,
    )
    Mechanism.RANGERAND(
        "range_sample",
        distribution="normal",
        mu=0.5,
        sigma=0.1,
        seed=102,
    )
    Mechanism.BATCHRAND(
        "batch_sample",
        distribution="normal",
        mu=0.75,
        sigma=0.1,
        seed=103,
    )
    Mechanism.DERIVED_BUFFER("random_static")
    Mechanism.TIMESTEP_BUFFER("random_timestep")

    def derive_buffers(self):
        return {
            "random_static": self.global_sample + self.range_sample + self.batch_sample
        }

    def derive_timestep_buffers(self, dt):
        return {"random_timestep": self.random_static * (1.0 + dt)}


class _NestedClassAuditState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0.0 * x")

    builder_calls = 0

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}


class _NestedClassMutatingTimestepBuilder(Mechanism):
    Mechanism.STATE_BUNDLE(_NestedClassAuditState)
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_timestep_buffers(self, dt):
        _NestedClassAuditState.builder_calls += 1
        return {"coefficient": dt * torch.ones_like(self.diam)}


class _MutatingTimestepWorkspace(_TimestepAccumulator):
    """Deliberately mutate a prepared input during the authored transition."""

    def advance(self, v, dt, values):
        del v, dt
        if self.training:
            self.coefficient.add_(0.125)
        return {"accumulator": values["accumulator"] + self.coefficient}


class _RebindingTimestepWorkspace(_TimestepAccumulator):
    """Deliberately rebind a prepared input during the authored transition."""

    def advance(self, v, dt, values):
        del v, dt
        if self.training:
            self.coefficient = self.coefficient + 0.125
        return {"accumulator": values["accumulator"] + self.coefficient}


class _PythonStateMutatingTimestepWorkspace(_TimestepAccumulator):
    """Deliberately mutate unregistered transition-instance state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    def advance(self, v, dt, values):
        del v, dt
        self.counter += 1
        return {
            "accumulator": values["accumulator"] + values["coefficient"] + self.counter
        }


class _GlobalRngConsumingTimestepWorkspace(_TimestepAccumulator):
    """Deliberately consume implicit process-global random state."""

    def advance(self, v, dt, values):
        del v, dt
        coefficient = values["coefficient"]
        sample = torch.rand((), device=coefficient.device, dtype=coefficient.dtype)
        return {"accumulator": values["accumulator"] + coefficient + 0.0 * sample}


class _ClassTensorMetadataMutatingTimestepWorkspace(_TimestepAccumulator):
    base = torch.arange(6.0, dtype=torch.float32)
    shared = base.as_strided((2,), (2,), 1)

    def advance(self, v, dt, values):
        del v, dt
        shared = _ClassTensorMetadataMutatingTimestepWorkspace.shared
        shared.data = shared.to(dtype=torch.float64)
        shared.requires_grad_(True)
        return {"accumulator": values["accumulator"] + values["coefficient"]}


class _ScalarTimestepMechanism(Mechanism):
    Mechanism.GLOBAL(gain=1.25)
    Mechanism.TIMESTEP_BUFFER("scalar_coefficient")

    def derive_timestep_buffers(self, dt):
        return {"scalar_coefficient": self.gain.reshape(()) * dt}


class _TimestepState(State):
    State.STATE("x")
    State.GLOBAL(state_scale=SCALE, state_rate=RATE)
    State.TIMESTEP_BUFFER("state_coefficient")
    State.DERIVATIVE("x' = state_coefficient")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}

    def derive_timestep_buffers(self, dt):
        return {
            "state_coefficient": (self.state_scale + self.state_rate * dt)
            * torch.ones_like(self.diam)
        }


class _StateTimestepMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_TimestepState)


class _InheritedTimestepState(_TimestepState):
    State.TIMESTEP_BUFFER("child_state_coefficient")

    def derive_timestep_buffers(self, dt):
        coefficient = (self.state_scale + self.state_rate * dt) * torch.ones_like(
            self.diam
        )
        return {
            "state_coefficient": coefficient,
            "child_state_coefficient": 2.0 * coefficient,
        }


class _InheritedStateTimestepMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_InheritedTimestepState)


class _DroppingInheritedTimestepState(_TimestepState):
    State.TIMESTEP_BUFFER("child_state_coefficient")

    def derive_timestep_buffers(self, dt):
        return {
            "child_state_coefficient": dt * torch.ones_like(self.diam),
        }


class _DroppingInheritedStateTimestepMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_DroppingInheritedTimestepState)


class _TransactionalNestedState(State):
    State.STATE("x")
    State.TIMESTEP_BUFFER("state_coefficient")
    State.DERIVATIVE("x' = 0.0 * x")

    fail_timestep_builder = False

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}

    def derive_timestep_buffers(self, dt):
        if self.fail_timestep_builder:
            return {}
        return {
            "state_coefficient": (3.0 + dt) * torch.ones_like(self.diam),
        }


class _TransactionalTimestepMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_TransactionalNestedState)
    Mechanism.TIMESTEP_BUFFER("mechanism_coefficient")

    def derive_timestep_buffers(self, dt):
        return {
            "mechanism_coefficient": (2.0 + dt) * torch.ones_like(self.diam),
        }


class _RangeTimestepState(State):
    State.STATE("x")
    State.RANGE(state_gain=0.625)
    State.TIMESTEP_BUFFER("state_coefficient")
    State.DERIVATIVE("x' = state_coefficient")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}

    def derive_timestep_buffers(self, dt):
        # RANGE values retain a leading parameter axis around an explicit
        # Population batch; this is a valid local transition workspace.
        return {"state_coefficient": self.state_gain * (1.0 + dt)}


class _RangeTimestepMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_RangeTimestepState)
    Mechanism.RANGE(mechanism_gain=0.5)
    Mechanism.GLOBAL(e=-62.0)
    Mechanism.TIMESTEP_BUFFER("mechanism_coefficient")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def derive_timestep_buffers(self, dt):
        return {"mechanism_coefficient": self.mechanism_gain * (1.0 + dt)}

    def i(self, v):
        return self.mechanism_coefficient * (v - self.e)


class _StructuralTimestepMechanism(Mechanism):
    """Exercise scalar and population-independent timestep workspaces."""

    Mechanism.GLOBAL(structural_scale=0.75)
    Mechanism.CARRY("accumulator")
    Mechanism.TIMESTEP_BUFFER("scalar_coefficient", shape=())
    Mechanism.TIMESTEP_BUFFER("structural_table", shape=(1, 3, 1))

    def derive_timestep_buffers(self, dt):
        coefficient = self.structural_scale.reshape(()) * dt
        return {
            "scalar_coefficient": coefficient,
            # Deliberately omit the leading singleton to exercise canonical
            # broadcast materialization into the exact declared shape.
            "structural_table": coefficient.reshape(1, 1).expand(3, 1),
        }

    def initial_values(self, v, values):
        del v
        return {"accumulator": torch.zeros_like(values["accumulator"])}

    def advance(self, v, dt, values):
        del v, dt
        increment = (
            values["scalar_coefficient"] + values["structural_table"].sum()
        ).expand_as(values["accumulator"])
        return {"accumulator": values["accumulator"] + increment}


class _StructuralTimestepState(State):
    State.STATE("x")
    State.GLOBAL(structural_state_scale=0.25)
    State.TIMESTEP_BUFFER("scalar_rate", shape=())
    State.TIMESTEP_BUFFER("structural_rates", shape=(1, 2, 1))
    State.ASSIGNED("rate")
    State.DERIVATIVE("x' = rate")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}

    def derive_timestep_buffers(self, dt):
        rate = self.structural_state_scale.reshape(()) * dt
        return {
            "scalar_rate": rate,
            "structural_rates": rate.reshape(1, 1).expand(2, 1),
        }

    def assigned_values(self, v, values):
        del values
        rate = self.scalar_rate + self.structural_rates.sum()
        return {"rate": torch.zeros_like(v) + rate}


class _StructuralStateMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_StructuralTimestepState)


def _model(mechanism=_TimestepAccumulator, *, batch_calls=()):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0, 1.5],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas)
        if mechanism is not None:
            mechanisms = mechanism if isinstance(mechanism, tuple) else (mechanism,)
            for mechanism_type in mechanisms:
                model.insert(mechanism_type)
        for size in batch_calls:
            model.batch(size)
        # Initialization can evaluate assigned algebra. Keep the deliberately
        # mutating transition fixture quiescent until its initial state exists.
        model.eval()
        model.initialize()
        model.train()
    return model


def _parameter_name(parameters, suffix):
    matches = [name for name in parameters if name.endswith(suffix)]
    assert len(matches) == 1
    return matches[0]


def _workspace_key(mechanism, name="coefficient"):
    return f"integrator.mech.mechanisms.{mechanism.__name__}.{name}"


def _state_workspace_key(mechanism, state, name="state_coefficient"):
    return f"integrator.mech.mechanisms.{mechanism.__name__}.DE.{state.__name__}.{name}"


def _clone_tree(tree):
    return torch.utils._pytree.tree_map(lambda value: value.detach().clone(), tree)


def _mapping_snapshot(mapping):
    return {
        name: (id(value), value._version, value.detach().clone())
        for name, value in mapping.items()
    }


def _assert_mapping_unchanged(mapping, snapshot):
    assert set(mapping) == set(snapshot)
    for name, value in mapping.items():
        identity, version, expected = snapshot[name]
        assert id(value) == identity
        assert value._version == version
        torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)


def _assert_tree_close(actual, expected, *, rtol=1.0e-11, atol=1.0e-12):
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
        )


def _initialize_imperative(model, dt):
    dt_tensor = torch.as_tensor(dt, device=model.device(), dtype=model.dtype())
    model.integrator._initialize(
        model,
        dt_tensor,
        force=True,
        compile_scope="population",
    )
    return dt_tensor


def _imperative_step(model, dt):
    model.integrator.step(model, dt)
    model.t = model.t + dt


def _functional_fixture(*, batch_calls=(), dt=DT):
    model = _model(batch_calls=batch_calls)
    return (*dn.func.make_functional(model, dt=dt), model)


def test_timestep_workspace_is_prepared_not_carried_and_dt_is_explicit():
    functional, tensors, model = _functional_fixture()
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    mechanism_name = _TimestepAccumulator.__name__
    key = _workspace_key(_TimestepAccumulator)

    assert set(tensors.constants) == {"diam", "dx", "dt"}
    assert tensors.constants["dt"].shape == ()
    assert tensors.constants["dt"].dtype == model.dtype()
    assert tensors.constants["dt"].device == model.device()
    assert set(tensors.state["mechanism_buffers"][mechanism_name]) == {"accumulator"}
    assert "coefficient" not in tensors.state["mechanism_buffers"][mechanism_name]
    assert key in prepared.values["mechanisms"]

    expected = (SCALE + RATE * DT) * torch.ones_like(model.diam)
    torch.testing.assert_close(
        prepared.values["mechanisms"][key],
        expected,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        prepared.values["integrator"]["dt"],
        tensors.constants["dt"],
        rtol=0.0,
        atol=0.0,
    )


def test_one_step_chained_and_fused_execution_match_imperative():
    functional_model = _model()
    imperative_model = _model()
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    dt = _initialize_imperative(imperative_model, DT)
    key = _workspace_key(_TimestepAccumulator)

    torch.testing.assert_close(
        prepared.values["mechanisms"][key],
        imperative_model.mech._TimestepAccumulator.coefficient,
        rtol=0.0,
        atol=0.0,
    )

    chained = tensors.state
    for _ in range(STEPS):
        chained, _aux = functional.step(tensors.parameters, prepared, chained)
        _imperative_step(imperative_model, dt)

    fused, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=STEPS,
    )
    expected = functional.extract(imperative_model).state
    _assert_tree_close(chained, fused)
    _assert_tree_close(chained, expected)


def test_timestep_substitution_rebuilds_eager_and_atomic_workspaces():
    functional, tensors, model = _functional_fixture()
    key = _workspace_key(_TimestepAccumulator)
    changed_constants = dict(tensors.constants)
    changed_constants["dt"] = tensors.constants["dt"].new_tensor(ALT_DT)

    nominal = functional.prepare(tensors.parameters, tensors.constants)
    changed = functional.prepare(tensors.parameters, changed_constants)
    expected_nominal = (SCALE + RATE * DT) * torch.ones_like(model.diam)
    expected_changed = (SCALE + RATE * ALT_DT) * torch.ones_like(model.diam)
    torch.testing.assert_close(
        nominal.values["mechanisms"][key], expected_nominal, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        changed.values["mechanisms"][key], expected_changed, rtol=0.0, atol=0.0
    )

    actual, _aux = functional.prepare_and_step(
        tensors.parameters,
        changed_constants,
        tensors.state,
    )
    accumulator = actual["mechanism_buffers"][_TimestepAccumulator.__name__][
        "accumulator"
    ]
    torch.testing.assert_close(
        accumulator,
        torch.broadcast_to(expected_changed, accumulator.shape),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual["clock"]["t"],
        tensors.state["clock"]["t"] + changed_constants["dt"],
        rtol=0.0,
        atol=0.0,
    )


def test_eager_prepared_plan_invalidates_when_dt_source_changes_in_place():
    functional, tensors, _model_source = _functional_fixture()
    constants = _clone_tree(tensors.constants)
    prepared = functional.prepare(tensors.parameters, constants)
    constants["dt"].mul_(2.0)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="changed|fresh|version|prepare",
    ):
        functional.step(tensors.parameters, prepared, tensors.state)


def test_raw_parameters_and_dt_have_exact_reverse_and_forward_jacobians():
    functional, tensors, _model_source = _functional_fixture()
    mechanism_name = _TimestepAccumulator.__name__
    scale_name = _parameter_name(tensors.parameters, f"{mechanism_name}.scale_param")
    rate_name = _parameter_name(tensors.parameters, f"{mechanism_name}.rate_param")

    def next_accumulator(scale, rate, dt):
        parameters = dict(tensors.parameters)
        constants = dict(tensors.constants)
        parameters[scale_name] = scale
        parameters[rate_name] = rate
        constants["dt"] = dt
        state, _aux = functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )
        return state["mechanism_buffers"][mechanism_name]["accumulator"]

    scale = tensors.parameters[scale_name]
    rate = tensors.parameters[rate_name]
    dt = tensors.constants["dt"]
    expected = (
        torch.ones_like(next_accumulator(scale, rate, dt)),
        torch.full_like(next_accumulator(scale, rate, dt), DT),
        torch.full_like(next_accumulator(scale, rate, dt), RATE),
    )
    reverse = torch.func.jacrev(next_accumulator, argnums=(0, 1, 2))(scale, rate, dt)
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(next_accumulator, argnums=(0, 1, 2))(
            scale, rate, dt
        )
    for actual_reverse, actual_forward, expected_value in zip(
        reverse,
        forward,
        expected,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_reverse, expected_value, rtol=1.0e-12, atol=1.0e-13
        )
        torch.testing.assert_close(
            actual_forward, expected_value, rtol=1.0e-12, atol=1.0e-13
        )


def test_vmap_over_raw_parameters_and_dt_matches_exact_lane_oracle():
    functional, tensors, _model_source = _functional_fixture()
    mechanism_name = _TimestepAccumulator.__name__
    scale_name = _parameter_name(tensors.parameters, f"{mechanism_name}.scale_param")
    rate_name = _parameter_name(tensors.parameters, f"{mechanism_name}.rate_param")

    def lane(scale, rate, dt):
        parameters = dict(tensors.parameters)
        constants = dict(tensors.constants)
        parameters[scale_name] = scale
        parameters[rate_name] = rate
        constants["dt"] = dt
        state, _aux = functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )
        return state["mechanism_buffers"][mechanism_name]["accumulator"]

    scale_lanes = torch.tensor((0.1, SCALE, 0.9), dtype=torch.float64)
    rate_lanes = torch.tensor((0.5, RATE, 2.5), dtype=torch.float64)
    dt_lanes = torch.tensor((DT / 2.0, DT, ALT_DT), dtype=torch.float64)
    actual = torch.vmap(lane)(scale_lanes, rate_lanes, dt_lanes)
    expected_values = scale_lanes + rate_lanes * dt_lanes
    expected = expected_values.reshape(-1, 1, 1).expand_as(actual)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-13)

    empty = torch.vmap(lane)(scale_lanes[:0], rate_lanes[:0], dt_lanes[:0])
    assert empty.shape == (0, *tensors.state["integrator"]["v"].shape)


def test_fullgraph_compile_and_aot_backward_match_eager():
    functional, tensors, _model_source = _functional_fixture()
    mechanism_name = _TimestepAccumulator.__name__
    scale_name = _parameter_name(tensors.parameters, f"{mechanism_name}.scale_param")
    rate_name = _parameter_name(tensors.parameters, f"{mechanism_name}.rate_param")

    def next_accumulator(scale, rate, dt):
        parameters = dict(tensors.parameters)
        constants = dict(tensors.constants)
        parameters[scale_name] = scale
        parameters[rate_name] = rate
        constants["dt"] = dt
        state, _aux = functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )
        return state["mechanism_buffers"][mechanism_name]["accumulator"]

    scale = tensors.parameters[scale_name].detach().clone().requires_grad_()
    rate = tensors.parameters[rate_name].detach().clone().requires_grad_()
    dt = tensors.constants["dt"].detach().clone().requires_grad_()
    expected = next_accumulator(scale, rate, dt)
    compiled = torch.compile(next_accumulator, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual = compiled(scale, rate, dt)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def loss(scale, rate, dt):
        return next_accumulator(scale, rate, dt).square().mean()

    expected_loss = loss(scale, rate, dt)
    expected_gradients = torch.autograd.grad(expected_loss, (scale, rate, dt))
    compiled_loss = torch.compile(loss, backend="aot_eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual_loss = compiled_loss(scale, rate, dt)
    actual_gradients = torch.autograd.grad(actual_loss, (scale, rate, dt))
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0.0, atol=0.0)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1.0e-11,
            atol=1.0e-12,
        )


def test_explicit_population_batch_keeps_workspace_shared_and_carry_lane_local():
    functional, tensors, model = _functional_fixture(batch_calls=(3,))
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    mechanism_name = _TimestepAccumulator.__name__
    key = _workspace_key(_TimestepAccumulator)
    workspace = prepared.values["mechanisms"][key]
    accumulator = tensors.state["mechanism_buffers"][mechanism_name]["accumulator"]

    assert workspace.shape == model._calc_shape_p()
    assert accumulator.shape == model.shape
    state = _clone_tree(tensors.state)
    lane_offsets = torch.arange(3, dtype=model.dtype()).reshape(3, 1, 1)
    state["mechanism_buffers"][mechanism_name]["accumulator"] = (
        accumulator + lane_offsets
    )
    actual, _aux = functional.step(tensors.parameters, prepared, state)
    expected = state["mechanism_buffers"][mechanism_name]["accumulator"] + workspace
    torch.testing.assert_close(
        actual["mechanism_buffers"][mechanism_name]["accumulator"],
        expected,
        rtol=0.0,
        atol=0.0,
    )

    def next_accumulator(dt):
        constants = dict(tensors.constants)
        constants["dt"] = dt
        next_state, _aux = functional.prepare_and_step(
            tensors.parameters,
            constants,
            state,
        )
        return next_state["mechanism_buffers"][mechanism_name]["accumulator"]

    dt_jacobian = torch.func.jacrev(next_accumulator)(tensors.constants["dt"])
    torch.testing.assert_close(
        dt_jacobian,
        torch.full_like(expected, RATE),
        rtol=1.0e-12,
        atol=1.0e-13,
    )


def test_inherited_timestep_declarations_are_unioned_into_prepared_schema():
    model = _model(_InheritedTimestepAccumulator)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    keys = set(prepared.values["mechanisms"])

    assert _workspace_key(_InheritedTimestepAccumulator) in keys
    assert _workspace_key(_InheritedTimestepAccumulator, "child_coefficient") in keys
    assert set(
        tensors.state["mechanism_buffers"][_InheritedTimestepAccumulator.__name__]
    ) == {"accumulator"}


@pytest.mark.parametrize(
    ("mechanism", "message"),
    [
        (_DroppingInheritedDeclaration, "missing.*coefficient"),
        (_MissingTimestepKey, "missing.*second"),
        (_UnexpectedTimestepKey, "unexpected.*surprise"),
        (_NonTensorTimestepValue, "must be a Tensor"),
        (_WrongShapeTimestepValue, "shape|broadcast"),
        (_WrongDtypeTimestepValue, "dtype"),
    ],
)
def test_malformed_timestep_builder_schemas_fail_closed(mechanism, message):
    model = _model(mechanism)
    with pytest.raises(dn.func.FunctionalizationError, match=message):
        dn.func.make_functional(model, dt=DT)


def test_timestep_builder_cannot_mutate_registered_tensors():
    model = _model(_MutatingTimestepBuilder)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="mutat|pure|registered",
    ):
        dn.func.make_functional(model, dt=DT)


@pytest.mark.parametrize(
    "mechanism_type",
    [_MutatingTimestepBuilder, _RebindingRegisteredTimestepBuilder],
)
def test_failed_imperative_builder_write_leaves_source_tree_bitwise_unchanged(
    mechanism_type,
):
    model = _model(mechanism_type)
    mechanism = getattr(model.mech, mechanism_type.__name__)
    snapshots = {
        name: (
            getattr(mechanism, name),
            getattr(mechanism, name).detach().clone(),
            getattr(mechanism, name)._version,
        )
        for name in ("scratch", "dt", "coefficient")
    }

    with pytest.raises(RuntimeError, match="must not mutate registered"):
        mechanism._configure_timestep(ALT_DT)

    for name, (identity, value, version) in snapshots.items():
        current = getattr(mechanism, name)
        assert current is identity
        assert current._version == version
        torch.testing.assert_close(current, value, rtol=0.0, atol=0.0)


def test_imperative_builder_isolation_retains_parameter_and_dt_gradients():
    model = _model()
    mechanism = model.mech._TimestepAccumulator
    dt = torch.tensor(
        DT,
        device=mechanism.dt.device,
        dtype=mechanism.dt.dtype,
        requires_grad=True,
    )

    mechanism._configure_timestep(dt)
    scale_gradient, rate_gradient, dt_gradient = torch.autograd.grad(
        mechanism.coefficient.sum(),
        (mechanism.scale_param, mechanism.rate_param, dt),
    )
    workspace_size = mechanism.coefficient.numel()
    torch.testing.assert_close(
        scale_gradient,
        torch.full_like(scale_gradient, workspace_size),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        rate_gradient,
        torch.full_like(rate_gradient, workspace_size * DT),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        dt_gradient,
        torch.full_like(dt_gradient, workspace_size * RATE),
        rtol=0.0,
        atol=0.0,
    )


def test_random_parameter_builders_are_isolated_differentiable_and_consistent():
    model = _model(_RandomParameterWorkspace)
    mechanism = model.mech._RandomParameterWorkspace
    dt = _initialize_imperative(model, DT)

    expected_static = (
        mechanism.global_sample + mechanism.range_sample + mechanism.batch_sample
    )
    torch.testing.assert_close(
        mechanism.random_static,
        expected_static,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        mechanism.random_timestep,
        expected_static * (1.0 + dt),
        rtol=0.0,
        atol=0.0,
    )

    mu_parameters = tuple(
        getattr(mechanism, f"{name}_mu_param")
        for name in ("global_sample", "range_sample", "batch_sample")
    )
    gradients = torch.autograd.grad(
        (mechanism.random_static + mechanism.random_timestep).sum(),
        mu_parameters,
    )
    expected_gradient = mechanism.random_static.new_tensor(
        mechanism.random_static.numel() * (2.0 + DT)
    )
    for gradient in gradients:
        torch.testing.assert_close(
            gradient,
            expected_gradient.expand_as(gradient),
            rtol=1.0e-12,
            atol=1.0e-13,
        )


def test_mechanism_builder_audits_nested_state_class_mutation():
    model = _model(_NestedClassMutatingTimestepBuilder)
    calls = _NestedClassAuditState.builder_calls

    with pytest.raises(
        RuntimeError,
        match="shared class state.*_NestedClassAuditState.*builder_calls",
    ):
        model.mech._NestedClassMutatingTimestepBuilder._configure_timestep(ALT_DT)

    assert _NestedClassAuditState.builder_calls == calls


def test_failed_imperative_builder_cannot_mutate_caller_timestep():
    model = _model(_MutatingTimestepArgument)
    mechanism = model.mech._MutatingTimestepArgument
    dt = torch.tensor(
        ALT_DT,
        device=mechanism.dt.device,
        dtype=mechanism.dt.dtype,
        requires_grad=True,
    )
    dt_value = dt.detach().clone()
    dt_version = dt._version
    mechanism_dt = mechanism.dt
    mechanism_dt_value = mechanism_dt.clone()
    mechanism_dt_version = mechanism_dt._version

    with pytest.raises(RuntimeError, match="tensor arguments"):
        mechanism._configure_timestep(dt)

    assert dt._version == dt_version
    torch.testing.assert_close(dt, dt_value, rtol=0.0, atol=0.0)
    assert mechanism.dt is mechanism_dt
    assert mechanism.dt._version == mechanism_dt_version
    torch.testing.assert_close(mechanism.dt, mechanism_dt_value, rtol=0.0, atol=0.0)


def test_failed_imperative_builder_instance_mutation_is_confined_and_rejected():
    model = _model(_PythonStateMutatingTimestepBuilder)
    mechanism = model.mech._PythonStateMutatingTimestepBuilder
    calls = mechanism.builder_calls
    dt = mechanism.dt
    dt_value = dt.clone()

    with pytest.raises(RuntimeError, match="Python instance state.*builder_calls"):
        mechanism._configure_timestep(ALT_DT)

    assert mechanism.builder_calls == calls
    assert mechanism.dt is dt
    torch.testing.assert_close(mechanism.dt, dt_value, rtol=0.0, atol=0.0)


def test_failed_imperative_builder_restores_shared_class_state():
    model = _model(_ClassStateMutatingTimestepBuilder)
    calls = _ClassStateMutatingTimestepBuilder.builder_calls

    with pytest.raises(RuntimeError, match="shared class state.*builder_calls"):
        model.mech._ClassStateMutatingTimestepBuilder._configure_timestep(ALT_DT)

    assert _ClassStateMutatingTimestepBuilder.builder_calls == calls


def test_failed_imperative_builder_restores_class_tensor_layout_and_values():
    model = _model(_ClassTensorResizingTimestepBuilder)
    base = _ClassTensorResizingTimestepBuilder.base
    base_value = base.clone()
    shared = _ClassTensorResizingTimestepBuilder.shared
    value = shared.clone()
    layout = (shared.shape, shared.stride(), shared.storage_offset())

    with pytest.raises(RuntimeError, match="shared class state.*shared"):
        model.mech._ClassTensorResizingTimestepBuilder._configure_timestep(ALT_DT)

    restored = _ClassTensorResizingTimestepBuilder.shared
    assert restored is shared
    assert _ClassTensorResizingTimestepBuilder.base is base
    assert restored.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert (restored.shape, restored.stride(), restored.storage_offset()) == layout
    torch.testing.assert_close(base, base_value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(restored, value, rtol=0.0, atol=0.0)


def test_failed_imperative_builder_restores_mutated_then_rebound_class_tensor():
    model = _model(_ClassTensorMutatingAndRebindingTimestepBuilder)
    base = _ClassTensorMutatingAndRebindingTimestepBuilder.base
    base_value = base.clone()
    shared = _ClassTensorMutatingAndRebindingTimestepBuilder.shared
    value = shared.clone()
    layout = (shared.shape, shared.stride(), shared.storage_offset())

    with pytest.raises(RuntimeError, match="shared class state.*shared"):
        model.mech._ClassTensorMutatingAndRebindingTimestepBuilder._configure_timestep(
            ALT_DT
        )

    restored = _ClassTensorMutatingAndRebindingTimestepBuilder.shared
    assert restored is shared
    assert _ClassTensorMutatingAndRebindingTimestepBuilder.base is base
    assert restored.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert (restored.shape, restored.stride(), restored.storage_offset()) == layout
    torch.testing.assert_close(base, base_value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(restored, value, rtol=0.0, atol=0.0)


def test_failed_imperative_builder_restores_class_tensor_dtype_and_grad_metadata():
    model = _model(_ClassTensorMetadataTimestepBuilder)
    base = _ClassTensorMetadataTimestepBuilder.base
    shared = _ClassTensorMetadataTimestepBuilder.shared
    expected = shared.clone()
    metadata = (
        id(shared),
        shared.untyped_storage().data_ptr(),
        shared.dtype,
        shared.device,
        shared.layout,
        shared.shape,
        shared.stride(),
        shared.storage_offset(),
        shared.requires_grad,
    )

    with pytest.raises(RuntimeError, match="shared class state.*shared"):
        model.mech._ClassTensorMetadataTimestepBuilder._configure_timestep(ALT_DT)

    restored = _ClassTensorMetadataTimestepBuilder.shared
    assert restored.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert (
        id(restored),
        restored.untyped_storage().data_ptr(),
        restored.dtype,
        restored.device,
        restored.layout,
        restored.shape,
        restored.stride(),
        restored.storage_offset(),
        restored.requires_grad,
    ) == metadata
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)


def test_expanded_class_tensor_can_be_a_pure_builder_constant():
    model = _model(_ExpandedClassTensorTimestepBuilder)
    shared = _ExpandedClassTensorTimestepBuilder.shared
    version = shared._version

    model.mech._ExpandedClassTensorTimestepBuilder._configure_timestep(ALT_DT)

    assert _ExpandedClassTensorTimestepBuilder.shared is shared
    assert shared._version == version
    torch.testing.assert_close(shared, torch.ones_like(shared), rtol=0.0, atol=0.0)


def test_failed_imperative_builder_restores_class_array_shape_and_values():
    model = _model(_ClassArrayResizingTimestepBuilder)
    shared = _ClassArrayResizingTimestepBuilder.shared
    value = shared.copy()
    shape = shared.shape

    with pytest.raises(RuntimeError, match="shared class state.*shared"):
        model.mech._ClassArrayResizingTimestepBuilder._configure_timestep(ALT_DT)

    restored = _ClassArrayResizingTimestepBuilder.shared
    assert restored is shared
    assert restored.shape == shape
    np.testing.assert_array_equal(restored, value)


def test_failed_imperative_builder_restores_mutated_then_rebound_class_array():
    model = _model(_ClassArrayMutatingAndRebindingTimestepBuilder)
    shared = _ClassArrayMutatingAndRebindingTimestepBuilder.shared
    value = shared.copy()
    shape = shared.shape

    with pytest.raises(RuntimeError, match="shared class state.*shared"):
        model.mech._ClassArrayMutatingAndRebindingTimestepBuilder._configure_timestep(
            ALT_DT
        )

    restored = _ClassArrayMutatingAndRebindingTimestepBuilder.shared
    assert restored is shared
    assert restored.shape == shape
    np.testing.assert_array_equal(restored, value)


def test_failed_imperative_builder_restores_global_rng_state():
    model = _model(_GlobalRngConsumingTimestepBuilder)
    rng_state = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="implicit RNG state.*PyTorch"):
        model.mech._GlobalRngConsumingTimestepBuilder._configure_timestep(ALT_DT)

    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_builder_python_mutation_is_rejected_without_poisoning_source_or_preparation():
    model = _model(_PythonStateMutatingTimestepBuilder)
    source = model.mech._PythonStateMutatingTimestepBuilder
    source_calls = source.builder_calls

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="timestep-buffer materialization.*Python instance state.*builder_calls",
    ):
        dn.func.FunctionalPopulation(model, dt=DT)

    assert source.builder_calls == source_calls
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="timestep-buffer materialization.*Python instance state.*builder_calls",
    ):
        dn.func.make_functional(model, dt=DT)
    assert source.builder_calls == source_calls


def test_builder_class_mutation_is_rejected_and_restored():
    model = _model(_ClassStateMutatingTimestepBuilder)
    calls = _ClassStateMutatingTimestepBuilder.builder_calls

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="class state|class attributes",
    ):
        dn.func.make_functional(model, dt=DT)

    assert _ClassStateMutatingTimestepBuilder.builder_calls == calls


def test_functional_audit_restores_class_tensor_dtype_and_grad_metadata():
    model = _model(_ClassTensorMetadataMutatingTimestepWorkspace)
    base = _ClassTensorMetadataMutatingTimestepWorkspace.base
    shared = _ClassTensorMetadataMutatingTimestepWorkspace.shared
    expected = shared.clone()
    metadata = (
        id(shared),
        shared.untyped_storage().data_ptr(),
        shared.dtype,
        shared.device,
        shared.layout,
        shared.shape,
        shared.stride(),
        shared.storage_offset(),
        shared.requires_grad,
    )

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="class state|class attributes",
    ):
        dn.func.make_functional(model, dt=DT)

    restored = _ClassTensorMetadataMutatingTimestepWorkspace.shared
    assert restored.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert (
        id(restored),
        restored.untyped_storage().data_ptr(),
        restored.dtype,
        restored.device,
        restored.layout,
        restored.shape,
        restored.stride(),
        restored.storage_offset(),
        restored.requires_grad,
    ) == metadata
    torch.testing.assert_close(restored, expected, rtol=0.0, atol=0.0)


def test_builder_rng_consumption_is_rejected_and_rng_state_is_restored():
    model = _model(_GlobalRngConsumingTimestepBuilder)
    rng_state = torch.random.get_rng_state().clone()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="RNG state|deterministic",
    ):
        dn.func.make_functional(model, dt=DT)

    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_preparation_constructor_validates_builders_on_a_disposable_clone():
    from dendra.func._population import _PopulationPreparation

    model = _model(_TimestepAccumulator)
    _initialize_imperative(model, DT)
    source = model.mech._TimestepAccumulator
    source.builder_calls = 0

    def mutating_builder(self, dt):
        self.builder_calls += 1
        coefficient = self.scale.reshape(1) + self.rate.reshape(1) * dt.reshape(1)
        return {"coefficient": coefficient}

    source.derive_timestep_buffers = MethodType(mutating_builder, source)
    dt = torch.as_tensor(DT, device=model.device(), dtype=model.dtype())

    preparation = _PopulationPreparation(model, dt)

    assert preparation.population is model
    assert source.builder_calls == 0
    assert preparation.population.mech._TimestepAccumulator.builder_calls == 0


@pytest.mark.parametrize(
    "mutation_target",
    ["parameter", "celsius", "dt", "diam-rebind"],
)
def test_preparation_input_writes_are_contained_and_rejected(mutation_target):
    model = _model(_TimestepAccumulator)
    functional = dn.func.FunctionalPopulation(model, dt=DT)
    tensors = functional.extract(model)
    owner = functional._preparation.population.mech._TimestepAccumulator

    def invalid_builder(self, dt):
        if mutation_target == "parameter":
            self.scale_param.add_(0.5)
        elif mutation_target == "celsius":
            self.celsius.add_(0.5)
        elif mutation_target == "dt":
            dt.add_(0.5)
        else:
            self.diam = self.diam + 0.5
        coefficient = self.scale.reshape(1) + self.rate.reshape(1) * dt.reshape(1)
        return {"coefficient": coefficient}

    owner.derive_timestep_buffers = MethodType(invalid_builder, owner)
    parameter_snapshot = _mapping_snapshot(tensors.parameters)
    constant_snapshot = _mapping_snapshot(tensors.constants)

    # Even a deliberately invalid builder cannot write through the ordinary
    # preparation path into its caller-owned explicit tensor mappings.
    functional.prepare(tensors.parameters, tensors.constants)
    _assert_mapping_unchanged(tensors.parameters, parameter_snapshot)
    _assert_mapping_unchanged(tensors.constants, constant_snapshot)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="mutated or rebound registered preparation inputs",
    ):
        functional._audit_read_only_workspaces(tensors)
    _assert_mapping_unchanged(tensors.parameters, parameter_snapshot)
    _assert_mapping_unchanged(tensors.constants, constant_snapshot)


def test_preparation_audit_rejects_new_registered_tensor_slots_on_private_clone():
    model = _model(_TimestepAccumulator)
    functional = dn.func.FunctionalPopulation(model, dt=DT)
    tensors = functional.extract(model)
    owner = functional._preparation.population.mech._TimestepAccumulator

    def invalid_builder(self, dt):
        self.register_buffer("secret", torch.ones_like(self.accumulator))
        return _TimestepAccumulator.derive_timestep_buffers(self, dt)

    owner.derive_timestep_buffers = MethodType(invalid_builder, owner)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="registered preparation inputs|registered parameter/buffer state",
    ):
        functional._audit_read_only_workspaces(tensors)

    assert "secret" not in owner._buffers
    assert "secret" not in model.mech._TimestepAccumulator._buffers


def test_audit_digest_accepts_empty_stride_zero_expansions():
    from dendra.func._population import _audit_tensor_digest

    empty = torch.empty((1, 1), dtype=torch.float32).expand(0, 3)
    assert empty.numel() == 0
    assert 0 in empty.stride()
    assert _audit_tensor_digest(empty) == _audit_tensor_digest(empty.contiguous())

    resizable = torch.empty(2, dtype=torch.float32)
    _audit_tensor_digest(resizable)
    resizable.resize_(4)
    assert resizable.shape == (4,)


def test_class_state_rollback_restores_tensor_and_numpy_shape_in_place():
    from dendra.func._population import _class_state_snapshot, _restore_class_state

    class MutableClassState:
        tensor = torch.arange(6.0, dtype=torch.float64).reshape(2, 3)
        array = np.arange(6.0, dtype=np.float64).reshape(2, 3)

    tensor = MutableClassState.tensor
    array = MutableClassState.array
    expected_tensor = tensor.clone()
    expected_array = array.copy()
    tensor_identity = id(tensor)
    array_identity = id(array)
    snapshot = _class_state_snapshot((MutableClassState,))

    tensor.resize_(3, 2).fill_(-1.0)
    array.resize((3, 2), refcheck=False)
    array.fill(-1.0)
    changed = _restore_class_state(snapshot)

    assert changed
    assert id(MutableClassState.tensor) == tensor_identity
    assert id(MutableClassState.array) == array_identity
    assert MutableClassState.tensor.shape == (2, 3)
    assert MutableClassState.tensor.stride() == (3, 1)
    torch.testing.assert_close(MutableClassState.tensor, expected_tensor)
    assert MutableClassState.array.shape == (2, 3)
    np.testing.assert_array_equal(MutableClassState.array, expected_array)


def test_class_state_rollback_restores_rebound_expanded_and_offset_tensor_views():
    from dendra.func._population import _class_state_snapshot, _restore_class_state

    class MutableTensorViews:
        expanded = torch.arange(2.0, dtype=torch.float64).reshape(1, 2).expand(3, 2)
        offset = torch.arange(10.0, dtype=torch.float64)[2:8].reshape(2, 3)

    expanded = MutableTensorViews.expanded
    offset = MutableTensorViews.offset
    expected_expanded = expanded.clone()
    expected_offset = offset.clone()
    expanded_metadata = (
        id(expanded),
        expanded.storage_offset(),
        expanded.shape,
        expanded.stride(),
    )
    offset_metadata = (
        id(offset),
        offset.storage_offset(),
        offset.shape,
        offset.stride(),
    )
    snapshot = _class_state_snapshot((MutableTensorViews,))

    for value in (expanded, offset):
        storage = value.untyped_storage()
        raw = torch.empty(0, dtype=torch.uint8).set_(
            storage,
            0,
            (storage.nbytes(),),
            (1,),
        )
        raw.zero_()
    MutableTensorViews.expanded = torch.full((3, 2), -1.0, dtype=torch.float64)
    MutableTensorViews.offset = torch.full((2, 3), -1.0, dtype=torch.float64)

    changed = _restore_class_state(snapshot)

    assert changed
    assert (
        id(MutableTensorViews.expanded),
        MutableTensorViews.expanded.storage_offset(),
        MutableTensorViews.expanded.shape,
        MutableTensorViews.expanded.stride(),
    ) == expanded_metadata
    assert (
        id(MutableTensorViews.offset),
        MutableTensorViews.offset.storage_offset(),
        MutableTensorViews.offset.shape,
        MutableTensorViews.offset.stride(),
    ) == offset_metadata
    torch.testing.assert_close(MutableTensorViews.expanded, expected_expanded)
    torch.testing.assert_close(MutableTensorViews.offset, expected_offset)


@pytest.mark.parametrize(
    "mechanism_type",
    [_MutatingTimestepWorkspace, _RebindingTimestepWorkspace],
)
def test_authored_timestep_workspace_writes_fail_during_lowering(mechanism_type):
    model = _model(mechanism_type)
    source = getattr(model.mech, mechanism_type.__name__)
    original_source = source.coefficient.clone()
    source_version = source.coefficient._version

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="lowering.*mutation or rebinding.*read-only.*coefficient",
    ):
        dn.func.make_functional(model, dt=DT)

    assert source.coefficient._version == source_version
    torch.testing.assert_close(
        source.coefficient,
        original_source,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("registration", ["buffer", "parameter"])
def test_transition_audit_rejects_new_registered_tensor_slots(registration):
    model = _model(_TimestepAccumulator)
    functional = dn.func.FunctionalPopulation(model, dt=DT)
    tensors = functional.extract(model)
    owner = functional._transition.population.mech._TimestepAccumulator

    def invalid_advance(self, v, dt, values):
        del v, dt
        secret = torch.ones_like(values["accumulator"])
        if registration == "buffer":
            self.register_buffer("secret", secret)
        else:
            self.register_parameter("secret", torch.nn.Parameter(secret))
        return {
            "accumulator": values["accumulator"] + values["coefficient"] + self.secret
        }

    owner.advance = MethodType(invalid_advance, owner)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="registered parameter/buffer state",
    ):
        functional._audit_read_only_workspaces(tensors)

    assert "secret" not in owner._buffers
    assert "secret" not in owner._parameters
    source = model.mech._TimestepAccumulator
    assert "secret" not in source._buffers
    assert "secret" not in source._parameters


def test_transition_audit_rejects_hidden_unregistered_tensor_state():
    model = _model(_TimestepAccumulator)
    functional = dn.func.FunctionalPopulation(model, dt=DT)
    tensors = functional.extract(model)
    owner = functional._transition.population.mech._TimestepAccumulator

    def invalid_advance(self, v, dt, values):
        del v, dt
        self.secret_tensor = torch.ones_like(values["accumulator"])
        self.secret_tensor.add_(1.0)
        return {
            "accumulator": values["accumulator"]
            + values["coefficient"]
            + self.secret_tensor
        }

    owner.advance = MethodType(invalid_advance, owner)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="unregistered Python instance state|stateless",
    ):
        functional._audit_read_only_workspaces(tensors)

    assert not hasattr(owner, "secret_tensor")
    assert not hasattr(model.mech._TimestepAccumulator, "secret_tensor")


def test_transition_audit_rejects_write_to_undeclared_registered_buffer():
    model = _model(_TimestepAccumulator)
    functional = dn.func.FunctionalPopulation(model, dt=DT)
    tensors = functional.extract(model)
    owner = functional._transition.population.mech._TimestepAccumulator
    owner.register_buffer("hidden", torch.zeros_like(owner.accumulator))
    expected = owner.hidden.clone()
    version = owner.hidden._version

    def invalid_advance(self, v, dt, values):
        del v, dt
        self.hidden.add_(1.0)
        return {
            "accumulator": values["accumulator"] + values["coefficient"] + self.hidden
        }

    owner.advance = MethodType(invalid_advance, owner)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="registered parameter/buffer state",
    ):
        functional._audit_read_only_workspaces(tensors)

    assert owner.hidden._version == version
    torch.testing.assert_close(owner.hidden, expected, rtol=0.0, atol=0.0)
    assert "hidden" not in model.mech._TimestepAccumulator._buffers


@pytest.mark.parametrize("mutation", ["data", "detach"])
def test_transition_parameter_writes_are_bitwise_audited_and_contained(mutation):
    model = _model(_TimestepAccumulator)
    functional = dn.func.FunctionalPopulation(model, dt=DT)
    tensors = functional.extract(model)
    owner = functional._transition.population.mech._TimestepAccumulator
    parameter_snapshot = _mapping_snapshot(tensors.parameters)

    def invalid_advance(self, v, dt, values):
        del v, dt
        if mutation == "data":
            self.scale_param.data.add_(1.0)
        else:
            self.scale_param.detach().add_(1.0)
        return {"accumulator": values["accumulator"] + values["coefficient"]}

    owner.advance = MethodType(invalid_advance, owner)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="read-only transition inputs",
    ):
        functional._audit_read_only_workspaces(tensors)

    _assert_mapping_unchanged(tensors.parameters, parameter_snapshot)


def test_transition_voltage_argument_write_is_audited_and_caller_state_isolated():
    model = _model(_TimestepAccumulator)
    functional = dn.func.FunctionalPopulation(model, dt=DT)
    tensors = functional.extract(model)
    owner = functional._transition.population.mech._TimestepAccumulator
    state_snapshot = _clone_tree(tensors.state)

    def invalid_advance(self, v, dt, values):
        del dt
        v.zero_()
        return {"accumulator": values["accumulator"] + values["coefficient"]}

    owner.advance = MethodType(invalid_advance, owner)

    # The unaudited defensive path still cannot write through an explicit
    # caller carry while extensions construct and validate a private plan.
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    functional.step(tensors.parameters, prepared, tensors.state)
    _assert_tree_close(tensors.state, state_snapshot, rtol=0.0, atol=0.0)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="read-only transition inputs.*positional:v",
    ):
        functional._audit_read_only_workspaces(tensors)

    _assert_tree_close(tensors.state, state_snapshot, rtol=0.0, atol=0.0)


def test_repeated_step_from_same_explicit_state_is_exactly_deterministic():
    model = _model(_TimestepAccumulator)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state_snapshot = _clone_tree(tensors.state)

    first, _aux = functional.step(tensors.parameters, prepared, tensors.state)
    second, _aux = functional.step(tensors.parameters, prepared, tensors.state)

    _assert_tree_close(first, second, rtol=0.0, atol=0.0)
    _assert_tree_close(tensors.state, state_snapshot, rtol=0.0, atol=0.0)


def test_python_instance_mutation_is_rejected_without_contaminating_plan_or_source():
    model = _model(_PythonStateMutatingTimestepWorkspace)
    source = model.mech._PythonStateMutatingTimestepWorkspace

    # Inspect the audit directly to prove that its rejected probes run on a
    # second private clone rather than contaminating the retained transition.
    functional = dn.func.FunctionalPopulation(model, dt=DT)
    tensors = functional.extract(model)
    plan_mechanism = (
        functional._transition.population.mech._PythonStateMutatingTimestepWorkspace
    )
    assert source.counter == 0
    assert plan_mechanism.counter == 0

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="purity audit.*Python instance state.*counter|stateless",
    ):
        functional._audit_read_only_workspaces(tensors)

    assert source.counter == 0
    assert plan_mechanism.counter == 0
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="purity audit.*Python instance state.*counter|stateless",
    ):
        dn.func.make_functional(model, dt=DT)
    assert source.counter == 0
    assert plan_mechanism.counter == 0


def test_global_rng_consumption_is_rejected_and_rng_state_is_restored():
    model = _model(_GlobalRngConsumingTimestepWorkspace)
    rng_state = torch.random.get_rng_state().clone()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="purity audit.*RNG state|deterministic",
    ):
        dn.func.make_functional(model, dt=DT)

    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_rng_restoration_survives_class_rollback_failure(monkeypatch):
    import dendra.func._population as population_module

    model = _model(_TimestepAccumulator)
    rng_state = torch.random.get_rng_state().clone()

    def failing_restore(_snapshot):
        torch.rand(())
        raise RuntimeError("deliberate class rollback failure")

    monkeypatch.setattr(population_module, "_restore_class_state", failing_restore)
    with pytest.raises(RuntimeError, match="class rollback failure"):
        dn.func.make_functional(model, dt=DT)

    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_timestep_hook_identity_is_part_of_the_source_structure(monkeypatch):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def replacement(self, dt):
        return {"coefficient": 3.0 * dt * torch.ones_like(self.diam)}

    monkeypatch.setattr(
        _TimestepAccumulator,
        "derive_timestep_buffers",
        replacement,
    )
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure changed|lower it again",
    ):
        functional.prepare(tensors.parameters, tensors.constants)


def test_nested_network_state_dict_load_invalidates_population_timestep_workspaces():
    source = dn.Network({"cell": _model(_TimestepAccumulator)})
    target = dn.Network({"cell": _model(_TimestepAccumulator)})
    source.initialize(DT)
    target.initialize(ALT_DT)

    target.load_state_dict(source.state_dict())
    population = target.populations["cell"]
    assert not population.integrator.initialized
    with pytest.raises(RuntimeError, match="population integrator workspaces.*cell"):
        target.step()

    target.initialize(ALT_DT)
    expected = population.mech._TimestepAccumulator.scale + (
        population.mech._TimestepAccumulator.rate * ALT_DT
    )
    torch.testing.assert_close(
        population.mech._TimestepAccumulator.coefficient,
        expected.expand_as(population.mech._TimestepAccumulator.coefficient),
        rtol=0.0,
        atol=0.0,
    )


def test_mechanism_set_dt_is_rejected_in_favor_of_timestep_buffer_protocol():
    with pytest.raises(TypeError, match="Mechanism.set_dt.*TIMESTEP_BUFFER"):

        class _InvalidMechanismSetDt(Mechanism):
            def set_dt(self, dt):
                del dt


def test_mechanism_inherited_set_dt_is_rejected_in_favor_of_timestep_buffers():
    class _InheritedSetDtMixin:
        def set_dt(self, dt):
            del dt

    with pytest.raises(TypeError, match="Mechanism.set_dt.*TIMESTEP_BUFFER"):

        class _InvalidInheritedMechanismSetDt(_InheritedSetDtMixin, Mechanism):
            pass


def test_handler_does_not_dispatch_instance_level_set_dt_override():
    model = _model()
    mechanism = model.mech._TimestepAccumulator
    calls = []

    def instance_set_dt(self, dt):
        calls.append(float(dt))

    mechanism.set_dt = MethodType(instance_set_dt, mechanism)
    model.mech.set_dt(ALT_DT)
    assert calls == []
    torch.testing.assert_close(
        mechanism.dt,
        mechanism.dt.new_tensor(ALT_DT),
        rtol=0.0,
        atol=0.0,
    )
    functional, _tensors = dn.func.make_functional(model, dt=ALT_DT)
    assert functional is not None


def test_state_set_dt_is_rejected_in_favor_of_timestep_buffer_protocol():
    with pytest.raises(TypeError, match="State.set_dt.*TIMESTEP_BUFFER"):

        class _InvalidStateSetDt(State):
            State.STATE("x")
            State.DERIVATIVE("x' = 0.0 * x")

            def set_dt(self, dt):
                del dt


def test_state_inherited_set_dt_is_rejected_in_favor_of_timestep_buffers():
    class _InheritedSetDtMixin:
        def set_dt(self, dt):
            del dt

    with pytest.raises(TypeError, match="State.set_dt.*TIMESTEP_BUFFER"):

        class _InvalidInheritedStateSetDt(_InheritedSetDtMixin, State):
            State.STATE("x")
            State.DERIVATIVE("x' = 0.0 * x")


def test_models_without_timestep_buffers_use_explicit_dt_for_clock_and_integrator():
    model = _model(None)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def final_time(dt):
        constants = dict(tensors.constants)
        constants["dt"] = dt
        state, _aux = functional.prepare_and_rollout(
            tensors.parameters,
            constants,
            tensors.state,
            steps=STEPS,
        )
        return state["clock"]["t"]

    actual = final_time(tensors.constants["dt"])
    expected = tensors.state["clock"]["t"] + STEPS * tensors.constants["dt"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        torch.func.jacrev(final_time)(tensors.constants["dt"]),
        torch.full_like(actual, STEPS),
        rtol=0.0,
        atol=0.0,
    )
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(final_time)(tensors.constants["dt"])
    torch.testing.assert_close(
        forward,
        torch.full_like(actual, STEPS),
        rtol=0.0,
        atol=0.0,
    )


def test_state_timestep_workspace_matches_imperative_and_has_exact_jacobians():
    functional_model = _model(_StateTimestepMechanism)
    imperative_model = _model(_StateTimestepMechanism)
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    mechanism_name = _StateTimestepMechanism.__name__
    state_name = _TimestepState.__name__
    key = _state_workspace_key(_StateTimestepMechanism, _TimestepState)
    scale_name = _parameter_name(
        tensors.parameters,
        f"{mechanism_name}.DE.{state_name}.state_scale_param",
    )
    rate_name = _parameter_name(
        tensors.parameters,
        f"{mechanism_name}.DE.{state_name}.state_rate_param",
    )

    assert key in prepared.values["mechanisms"]
    state_buffers = tensors.state.get("state_buffers", {})
    assert "state_coefficient" not in state_buffers.get(mechanism_name, {}).get(
        state_name,
        {},
    )
    dt = _initialize_imperative(imperative_model, DT)
    imperative_state = imperative_model.mech._StateTimestepMechanism.DE._TimestepState
    torch.testing.assert_close(
        prepared.values["mechanisms"][key],
        imperative_state.state_coefficient,
        rtol=0.0,
        atol=0.0,
    )

    chained = tensors.state
    for _ in range(STEPS):
        chained, _aux = functional.step(tensors.parameters, prepared, chained)
        _imperative_step(imperative_model, dt)
    fused, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=STEPS,
    )
    expected_state = functional.extract(imperative_model).state
    _assert_tree_close(chained, fused)
    _assert_tree_close(chained, expected_state)

    def next_x(scale, rate, timestep):
        parameters = dict(tensors.parameters)
        constants = dict(tensors.constants)
        parameters[scale_name] = scale
        parameters[rate_name] = rate
        constants["dt"] = timestep
        next_state, _aux = functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )
        return next_state["mechanisms"][mechanism_name]["x"]

    scale = tensors.parameters[scale_name]
    rate = tensors.parameters[rate_name]
    timestep = tensors.constants["dt"]
    expected_jacobians = (
        torch.full_like(next_x(scale, rate, timestep), DT),
        torch.full_like(next_x(scale, rate, timestep), DT**2),
        torch.full_like(next_x(scale, rate, timestep), SCALE + 2.0 * RATE * DT),
    )
    reverse = torch.func.jacrev(next_x, argnums=(0, 1, 2))(
        scale,
        rate,
        timestep,
    )
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(next_x, argnums=(0, 1, 2))(
            scale,
            rate,
            timestep,
        )
    for actual_reverse, actual_forward, expected_jacobian in zip(
        reverse,
        forward,
        expected_jacobians,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_reverse,
            expected_jacobian,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        torch.testing.assert_close(
            actual_forward,
            expected_jacobian,
            rtol=1.0e-12,
            atol=1.0e-13,
        )


def test_state_timestep_declarations_inherit_and_schema_errors_fail_closed():
    model = _model(_InheritedStateTimestepMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    keys = set(prepared.values["mechanisms"])
    assert (
        _state_workspace_key(
            _InheritedStateTimestepMechanism,
            _InheritedTimestepState,
        )
        in keys
    )
    assert (
        _state_workspace_key(
            _InheritedStateTimestepMechanism,
            _InheritedTimestepState,
            "child_state_coefficient",
        )
        in keys
    )

    invalid = _model(_DroppingInheritedStateTimestepMechanism)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="missing.*state_coefficient",
    ):
        dn.func.make_functional(invalid, dt=DT)

    with pytest.raises(
        ValueError,
        match="both DERIVED_BUFFER and TIMESTEP_BUFFER",
    ):

        class _OverlappingStateWorkspace(State):
            State.STATE("x")
            State.DERIVED_BUFFER("workspace")
            State.TIMESTEP_BUFFER("workspace")
            State.DERIVATIVE("x' = 0.0 * x")


def test_timestep_declarations_reject_ambiguous_registered_slot_collisions():
    with pytest.raises(ValueError, match="do not also declare.*CARRY"):

        class _DuplicateMechanismBuffer(Mechanism):
            Mechanism.CARRY("workspace")
            Mechanism.TIMESTEP_BUFFER("workspace")

    with pytest.raises(ValueError, match="do not also declare.*CARRY"):

        class _DuplicateStateBuffer(State):
            State.STATE("x")
            State.CARRY("workspace")
            State.TIMESTEP_BUFFER("workspace")
            State.DERIVATIVE("x' = 0.0 * x")

    with pytest.raises(ValueError, match="do not also declare.*CARRY"):

        class _DuplicateDerivedMechanismBuffer(Mechanism):
            Mechanism.CARRY("workspace")
            Mechanism.DERIVED_BUFFER("workspace")

    with pytest.raises(ValueError, match="do not also declare.*CARRY"):

        class _DuplicateDerivedStateBuffer(State):
            State.STATE("x")
            State.CARRY("workspace")
            State.DERIVED_BUFFER("workspace")
            State.DERIVATIVE("x' = 0.0 * x")

    with pytest.raises(ValueError, match="State variables cannot also.*workspace"):

        class _DerivedStateVariableCollision(State):
            State.STATE("workspace")
            State.DERIVED_BUFFER("workspace")
            State.DERIVATIVE("workspace' = 0.0 * workspace")

    with pytest.raises(ValueError, match="conflict.*parameters"):

        class _MechanismParameterCollision(Mechanism):
            Mechanism.GLOBAL(workspace=1.0)
            Mechanism.TIMESTEP_BUFFER("workspace")

    with pytest.raises(ValueError, match="conflict.*parameters"):

        class _StateParameterCollision(State):
            State.STATE("x")
            State.GLOBAL(workspace=1.0)
            State.TIMESTEP_BUFFER("workspace")
            State.DERIVATIVE("x' = 0.0 * x")

    with pytest.raises(ValueError, match="reserved execution slots.*dt"):

        class _ReservedMechanismSlot(Mechanism):
            Mechanism.TIMESTEP_BUFFER("dt")

    with pytest.raises(ValueError, match="reserved execution slots.*dt"):

        class _ReservedStateSlot(State):
            State.STATE("x")
            State.TIMESTEP_BUFFER("dt")
            State.DERIVATIVE("x' = 0.0 * x")

    with pytest.raises(ValueError, match="reserved execution slots.*foo_"):

        class _SavedMechanismSlot(Mechanism):
            Mechanism.NONSPECIFIC_CURRENT("foo")
            Mechanism.SAVE_CURRENT("foo")
            Mechanism.TIMESTEP_BUFFER("foo_")

    with pytest.raises(ValueError, match="reserved execution slots.*foo_"):

        class _SavedDerivedMechanismSlot(Mechanism):
            Mechanism.NONSPECIFIC_CURRENT("foo")
            Mechanism.SAVE_CURRENT("foo")
            Mechanism.DERIVED_BUFFER("foo_")

    with pytest.raises(ValueError, match="ion/material runtime slots.*ena"):

        class _DerivedIonSlot(Mechanism):
            Mechanism.USEION("na", read=["ena"])
            Mechanism.DERIVED_BUFFER("ena")

    class _NestedDerivedIonState(State):
        State.STATE("x")
        State.DERIVED_BUFFER("ena")
        State.DERIVATIVE("x' = 0.0 * x")

    with pytest.raises(ValueError, match="ion/material runtime slots.*ena"):

        class _NestedDerivedIonSlot(Mechanism):
            Mechanism.STATE_BUNDLE(_NestedDerivedIonState)
            Mechanism.USEION("na", read=["ena"])

    with pytest.raises(ValueError, match="reserved execution slots.*training"):

        class _ModuleTrainingSlot(Mechanism):
            Mechanism.TIMESTEP_BUFFER("training")

    with pytest.raises(ValueError, match="reserved execution slots.*training"):

        class _StateTrainingSlot(State):
            State.STATE("x")
            State.TIMESTEP_BUFFER("training")
            State.DERIVATIVE("x' = 0.0 * x")


@pytest.mark.parametrize("workspace_kind", ["DERIVED_BUFFER", "TIMESTEP_BUFFER"])
@pytest.mark.parametrize("owner_kind", ["mechanism", "state"])
def test_workspace_declarations_reserve_physical_parameter_slots(
    workspace_kind,
    owner_kind,
):
    owner = Mechanism if owner_kind == "mechanism" else State
    declare = getattr(owner, workspace_kind)

    with pytest.raises(ValueError, match="conflict.*gain_param"):
        if owner_kind == "mechanism":

            class _PhysicalMechanismParameterSlot(Mechanism):
                Mechanism.GLOBAL(gain=1.0)
                declare("gain_param")

        else:

            class _PhysicalStateParameterSlot(State):
                State.STATE("x")
                State.GLOBAL(gain=1.0)
                declare("gain_param")
                State.DERIVATIVE("x' = 0.0 * x")


@pytest.mark.parametrize("workspace_kind", ["DERIVED_BUFFER", "TIMESTEP_BUFFER"])
def test_workspace_declarations_reserve_generated_random_rng_slots(workspace_kind):
    declare = getattr(Mechanism, workspace_kind)

    with pytest.raises(ValueError, match="reserved execution slots.*sample_rng"):

        class _RandomRngWorkspaceSlot(Mechanism):
            Mechanism.GLOBALRAND(
                "sample",
                distribution="normal",
                mu=0.0,
                sigma=1.0,
                seed=1,
            )
            declare("sample_rng")


@pytest.mark.parametrize("workspace_kind", ["DERIVED_BUFFER", "TIMESTEP_BUFFER"])
@pytest.mark.parametrize(
    "name",
    ["_solve", "method", "method_kwargs"],
)
def test_state_workspace_declarations_reserve_installed_runtime_names(
    workspace_kind,
    name,
):
    declare = getattr(State, workspace_kind)

    with pytest.raises(ValueError, match=rf"reserved execution slots.*{name}"):

        class _InstalledStateRuntimeSlot(State):
            State.STATE("x")
            declare(name)
            State.DERIVATIVE("x' = 0.0 * x")


@pytest.mark.parametrize("workspace_kind", ["DERIVED_BUFFER", "TIMESTEP_BUFFER"])
@pytest.mark.parametrize(
    "name",
    [
        "read_ion",
        "write_ion",
        "write_ion_c",
        "read_material",
        "write_material",
        "source_material",
        "factorable",
        "injected_waveforms",
        "support_map",
    ],
)
def test_mechanism_workspace_declarations_reserve_installed_runtime_names(
    workspace_kind,
    name,
):
    declare = getattr(Mechanism, workspace_kind)

    with pytest.raises(ValueError, match=rf"reserved execution slots.*{name}"):

        class _InstalledMechanismRuntimeSlot(Mechanism):
            declare(name)


@pytest.mark.parametrize("workspace_kind", ["DERIVED_BUFFER", "TIMESTEP_BUFFER"])
def test_mechanism_workspace_declarations_reserve_generated_current_methods(
    workspace_kind,
):
    declare = getattr(Mechanism, workspace_kind)

    with pytest.raises(ValueError, match="installed current.*i_with_g"):

        class _GeneratedCurrentRuntimeSlot(Mechanism):
            Mechanism.NONSPECIFIC_CURRENT("i")
            declare("i_with_g")

            def i(self, v):
                return 0.0 * v


def test_mechanism_timestep_configuration_is_transactional_when_nested_state_builder_fails():
    model = _model(_TransactionalTimestepMechanism)
    mechanism = model.mech._TransactionalTimestepMechanism
    state = mechanism.DE._TransactionalNestedState
    mechanism._configure_timestep(DT)

    snapshots = {
        "dt": (
            mechanism,
            "dt",
            mechanism.dt,
            mechanism.dt.clone(),
            mechanism.dt._version,
        ),
        "mechanism": (
            mechanism,
            "mechanism_coefficient",
            mechanism.mechanism_coefficient,
            mechanism.mechanism_coefficient.clone(),
            mechanism.mechanism_coefficient._version,
        ),
        "state": (
            state,
            "state_coefficient",
            state.state_coefficient,
            state.state_coefficient.clone(),
            state.state_coefficient._version,
        ),
    }
    state.fail_timestep_builder = True

    with pytest.raises(ValueError, match="missing.*state_coefficient"):
        mechanism._configure_timestep(ALT_DT)

    for owner, name, original, expected, version in snapshots.values():
        actual = getattr(owner, name)
        assert actual is original
        assert actual._version == version
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_handler_set_dt_stages_every_canonical_tree_before_any_commit():
    model = _model((_TimestepAccumulator, _TransactionalTimestepMechanism))
    accumulator = model.mech._TimestepAccumulator
    transactional = model.mech._TransactionalTimestepMechanism
    nested_state = transactional.DE._TransactionalNestedState
    model.mech.set_dt(DT)

    slots = (
        (accumulator, "dt"),
        (accumulator, "coefficient"),
        (transactional, "dt"),
        (transactional, "mechanism_coefficient"),
        (nested_state, "state_coefficient"),
    )
    snapshots = tuple(
        (owner, name, getattr(owner, name), getattr(owner, name).clone())
        for owner, name in slots
    )
    nested_state.fail_timestep_builder = True

    with pytest.raises(ValueError, match="missing.*state_coefficient"):
        model.mech.set_dt(ALT_DT)

    for owner, name, identity, value in snapshots:
        assert getattr(owner, name) is identity
        torch.testing.assert_close(getattr(owner, name), value, rtol=0.0, atol=0.0)


def test_explicit_batch_accepts_range_shaped_mechanism_and_state_workspaces():
    functional_model = _model(_RangeTimestepMechanism, batch_calls=(3,))
    imperative_model = _model(_RangeTimestepMechanism, batch_calls=(3,))
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    mechanism_name = _RangeTimestepMechanism.__name__
    state_name = _RangeTimestepState.__name__
    mechanism_key = _workspace_key(
        _RangeTimestepMechanism,
        "mechanism_coefficient",
    )
    state_key = _state_workspace_key(
        _RangeTimestepMechanism,
        _RangeTimestepState,
    )
    mechanism = functional_model.mech._RangeTimestepMechanism
    state_module = mechanism.DE._RangeTimestepState

    assert prepared.values["mechanisms"][mechanism_key].shape == (
        mechanism.mechanism_gain.shape
    )
    assert prepared.values["mechanisms"][state_key].shape == (
        state_module.state_gain.shape
    )
    assert prepared.values["mechanisms"][mechanism_key].shape == (
        1,
        *functional_model.core_shape(),
    )
    assert tensors.state["mechanisms"][mechanism_name]["x"].shape == (
        functional_model.shape
    )

    dt = _initialize_imperative(imperative_model, DT)
    actual, _aux = functional.step(tensors.parameters, prepared, tensors.state)
    _imperative_step(imperative_model, dt)
    expected = functional.extract(imperative_model).state
    _assert_tree_close(actual, expected)

    state_gain_name = _parameter_name(
        tensors.parameters,
        f"{mechanism_name}.DE.{state_name}.state_gain_param",
    )

    def next_x(state_gain, timestep):
        parameters = dict(tensors.parameters)
        constants = dict(tensors.constants)
        parameters[state_gain_name] = state_gain
        constants["dt"] = timestep
        next_state, _aux = functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )
        return next_state["mechanisms"][mechanism_name]["x"]

    state_gain = tensors.parameters[state_gain_name]
    timestep = tensors.constants["dt"]
    state_gain_jacobian, dt_jacobian = torch.func.jacrev(
        next_x,
        argnums=(0, 1),
    )(state_gain, timestep)
    torch.testing.assert_close(
        state_gain_jacobian,
        torch.full_like(state_gain_jacobian, DT * (1.0 + DT)),
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    torch.testing.assert_close(
        dt_jacobian,
        torch.full_like(dt_jacobian, 0.625 * (1.0 + 2.0 * DT)),
        rtol=1.0e-12,
        atol=1.0e-13,
    )


def test_timestep_workspace_shapes_remain_stable_across_dt_reconfiguration():
    model = _model(_RangeTimestepMechanism, batch_calls=(3,))
    mechanism = model.mech._RangeTimestepMechanism
    state = mechanism.DE._RangeTimestepState
    shapes = []
    for timestep in (DT, ALT_DT, DT / 2.0, DT):
        _initialize_imperative(model, timestep)
        shapes.append(
            (
                mechanism.mechanism_coefficient.shape,
                state.state_coefficient.shape,
            )
        )

    assert len(set(shapes)) == 1
    assert shapes[0] == (mechanism.mechanism_gain.shape, state.state_gain.shape)


@pytest.mark.parametrize(
    ("mechanism_type", "batch_calls"),
    [
        (_ScalarTimestepMechanism, ()),
        (_RangeTimestepMechanism, (3,)),
    ],
)
def test_timestep_workspace_state_dict_strictly_round_trips_into_fresh_model(
    mechanism_type,
    batch_calls,
):
    source = _model(mechanism_type, batch_calls=batch_calls)
    _initialize_imperative(source, DT)
    source_mechanism = getattr(source.mech, mechanism_type.__name__)

    # Every authored broadcast form is installed as shape_p, including a
    # scalar mechanism result and the RANGE-shaped nested-State result.
    for name in source_mechanism._timestep_buffers:
        assert getattr(source_mechanism, name).shape == source_mechanism.shape_p
    for source_state in source_mechanism.DE.values():
        for name in source_state._timestep_buffers:
            assert getattr(source_state, name).shape == source_state.shape_p

    saved = _clone_tree(source.state_dict())
    target = _model(mechanism_type, batch_calls=batch_calls)
    target_mechanism = getattr(target.mech, mechanism_type.__name__)

    # The registered schema is canonical even before the target's first
    # integrator initialization, which is the lifecycle boundary that used to
    # produce strict-load size mismatches for nested States.
    for name in target_mechanism._timestep_buffers:
        assert getattr(target_mechanism, name).shape == target_mechanism.shape_p
    for target_state in target_mechanism.DE.values():
        for name in target_state._timestep_buffers:
            assert getattr(target_state, name).shape == target_state.shape_p

    target.load_state_dict(saved)
    for name in source_mechanism._timestep_buffers:
        torch.testing.assert_close(
            getattr(target_mechanism, name),
            getattr(source_mechanism, name),
            rtol=0.0,
            atol=0.0,
        )
    for state_name, source_state in source_mechanism.DE.items():
        target_state = target_mechanism.DE[state_name]
        for name in source_state._timestep_buffers:
            torch.testing.assert_close(
                getattr(target_state, name),
                getattr(source_state, name),
                rtol=0.0,
                atol=0.0,
            )


def test_runtime_checkpoint_restore_does_not_resurrect_stale_timestep_workspace():
    model = _model(_TransactionalTimestepMechanism)
    mechanism = model.mech._TransactionalTimestepMechanism
    state = mechanism.DE._TransactionalNestedState
    dt = _initialize_imperative(model, DT)
    _imperative_step(model, dt)
    checkpoint = model.state_dict_for_checkpoint()
    old_mechanism_workspace = mechanism.mechanism_coefficient.clone()
    old_state_workspace = state.state_coefficient.clone()

    _initialize_imperative(model, ALT_DT)
    assert not torch.equal(mechanism.mechanism_coefficient, old_mechanism_workspace)
    assert not torch.equal(state.state_coefficient, old_state_workspace)
    model.restore_dict_from_checkpoint(checkpoint)

    active_dt = float(model.integrator.dt)
    assert float(mechanism.dt) == pytest.approx(active_dt)
    torch.testing.assert_close(
        mechanism.mechanism_coefficient,
        torch.full_like(mechanism.mechanism_coefficient, 2.0 + active_dt),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state.state_coefficient,
        torch.full_like(state.state_coefficient, 3.0 + active_dt),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("loader", ["load_state_dict", "load"])
def test_state_dict_load_invalidates_timestep_workspaces_before_reuse(loader):
    model = _model(_TransactionalTimestepMechanism)
    mechanism = model.mech._TransactionalTimestepMechanism
    state = mechanism.DE._TransactionalNestedState
    _initialize_imperative(model, DT)
    saved = _clone_tree(model.state_dict())

    _initialize_imperative(model, ALT_DT)
    getattr(model, loader)(saved)
    assert not model.integrator.initialized

    model.step(dt=ALT_DT)
    torch.testing.assert_close(
        mechanism.mechanism_coefficient,
        torch.full_like(mechanism.mechanism_coefficient, 2.0 + ALT_DT),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state.state_coefficient,
        torch.full_like(state.state_coefficient, 3.0 + ALT_DT),
        rtol=0.0,
        atol=0.0,
    )


def test_timestep_buffer_shape_declarations_validate_and_merge_idempotently():
    class _ScalarBase(Mechanism):
        Mechanism.TIMESTEP_BUFFER("workspace", shape=())

    class _ScalarChild(_ScalarBase):
        Mechanism.TIMESTEP_BUFFER("workspace", shape=())

    class _ScalarStateBase(State):
        State.STATE("x")
        State.TIMESTEP_BUFFER("workspace", shape=())
        State.DERIVATIVE("x' = 0.0 * x")

    class _ScalarStateChild(_ScalarStateBase):
        State.TIMESTEP_BUFFER("workspace", shape=())

    assert _ScalarChild._timestep_buffer_shapes == {"workspace": ()}
    assert _ScalarStateChild._timestep_buffer_shapes == {"workspace": ()}

    with pytest.raises(ValueError, match="conflicting shapes"):

        class _ConflictingMechanismShape(_ScalarBase):
            Mechanism.TIMESTEP_BUFFER("workspace", shape=(1,))

    with pytest.raises(ValueError, match="conflicting shapes"):

        class _ConflictingStateShape(_ScalarStateBase):
            State.TIMESTEP_BUFFER("workspace", shape=(1,))


@pytest.mark.parametrize(
    ("shape", "error"),
    [
        ([1], TypeError),
        ((True,), TypeError),
        ((np.int64(1),), TypeError),
        ((-1,), ValueError),
        ("structural", ValueError),
    ],
)
def test_timestep_buffer_shape_declarations_reject_ambiguous_dimensions(
    shape,
    error,
):
    with pytest.raises(error, match="TIMESTEP_BUFFER shape"):

        class _InvalidTimestepShape(Mechanism):
            Mechanism.TIMESTEP_BUFFER("workspace", shape=shape)

    # A failed class body must not contaminate the next declaration consumer.
    class _UnaffectedMechanism(Mechanism):
        pass

    assert not _UnaffectedMechanism._timestep_buffers
    assert not _UnaffectedMechanism._timestep_buffer_shapes


def test_structural_timestep_shapes_are_exact_stable_and_checkpointable():
    mechanism_types = (_StructuralTimestepMechanism, _StructuralStateMechanism)
    source = _model(mechanism_types, batch_calls=(2,))
    mechanism = source.mech._StructuralTimestepMechanism
    state = source.mech._StructuralStateMechanism.DE._StructuralTimestepState

    expected_shapes = {
        "scalar_coefficient": (),
        "structural_table": (1, 3, 1),
        "scalar_rate": (),
        "structural_rates": (1, 2, 1),
    }
    for timestep in (DT, ALT_DT, DT / 2.0):
        _initialize_imperative(source, timestep)
        assert (
            mechanism.scalar_coefficient.shape == expected_shapes["scalar_coefficient"]
        )
        assert mechanism.structural_table.shape == expected_shapes["structural_table"]
        assert state.scalar_rate.shape == expected_shapes["scalar_rate"]
        assert state.structural_rates.shape == expected_shapes["structural_rates"]

    # Structural workspaces do not acquire Population.batch() axes. Their
    # authored singleton axes remain available for intentional broadcasting.
    assert source.shape[0] == 2
    assert mechanism.structural_table.shape[0] == 1
    assert state.structural_rates.shape[0] == 1

    source.float()
    _initialize_imperative(source, DT)
    for value in (
        mechanism.scalar_coefficient,
        mechanism.structural_table,
        state.scalar_rate,
        state.structural_rates,
    ):
        assert value.dtype == torch.float32
        assert value.device == source.device()

    saved = _clone_tree(source.state_dict())
    target = _model(mechanism_types, batch_calls=(2,)).float()
    target_mechanism = target.mech._StructuralTimestepMechanism
    target_state = target.mech._StructuralStateMechanism.DE._StructuralTimestepState

    # Fresh placeholders already have the final schema, so strict loading does
    # not depend on whether timestep configuration has run at the checkpoint's
    # timestep.
    assert target_mechanism.scalar_coefficient.shape == ()
    assert target_mechanism.structural_table.shape == (1, 3, 1)
    assert target_state.scalar_rate.shape == ()
    assert target_state.structural_rates.shape == (1, 2, 1)
    target.load_state_dict(saved, strict=True)
    for owner, target_owner, names in (
        (
            mechanism,
            target_mechanism,
            ("scalar_coefficient", "structural_table"),
        ),
        (state, target_state, ("scalar_rate", "structural_rates")),
    ):
        for name in names:
            torch.testing.assert_close(
                getattr(target_owner, name),
                getattr(owner, name),
                rtol=0.0,
                atol=0.0,
            )

    mechanism_declarations, mechanism_realized = mechanism._timestep_buffer_schema
    state_declarations, state_realized = state._timestep_buffer_schema
    assert dict(mechanism_declarations) == {
        "scalar_coefficient": (),
        "structural_table": (1, 3, 1),
    }
    assert dict(mechanism_realized) == {
        "scalar_coefficient": (),
        "structural_table": (1, 3, 1),
    }
    assert dict(state_declarations) == {
        "scalar_rate": (),
        "structural_rates": (1, 2, 1),
    }
    assert dict(state_realized) == {
        "scalar_rate": (),
        "structural_rates": (1, 2, 1),
    }


def test_structural_timestep_workspaces_match_imperative_and_functional_batch():
    mechanism_types = (_StructuralTimestepMechanism, _StructuralStateMechanism)
    functional_model = _model(mechanism_types, batch_calls=(2,))
    imperative_model = _model(mechanism_types, batch_calls=(2,))
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    mechanism_key = _workspace_key(
        _StructuralTimestepMechanism,
        "structural_table",
    )
    state_key = _state_workspace_key(
        _StructuralStateMechanism,
        _StructuralTimestepState,
        "structural_rates",
    )
    assert prepared.values["mechanisms"][mechanism_key].shape == (1, 3, 1)
    assert prepared.values["mechanisms"][state_key].shape == (1, 2, 1)

    actual, _aux = functional.step(tensors.parameters, prepared, tensors.state)
    dt = _initialize_imperative(imperative_model, DT)
    _imperative_step(imperative_model, dt)
    _assert_tree_close(actual, functional.extract(imperative_model).state)


def test_structural_timestep_workspaces_support_ad_vmap_and_compile():
    model = _model(_StructuralTimestepMechanism)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    mechanism_name = _StructuralTimestepMechanism.__name__
    scale_name = _parameter_name(
        tensors.parameters,
        f"{mechanism_name}.structural_scale_param",
    )

    def next_accumulator(scale, timestep):
        parameters = dict(tensors.parameters)
        constants = dict(tensors.constants)
        parameters[scale_name] = scale
        constants["dt"] = timestep
        state, _aux = functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )
        return state["mechanism_buffers"][mechanism_name]["accumulator"]

    scale = tensors.parameters[scale_name]
    timestep = tensors.constants["dt"]
    expected_scale_jacobian = torch.full_like(
        next_accumulator(scale, timestep),
        4.0 * DT,
    )
    expected_dt_jacobian = torch.full_like(
        expected_scale_jacobian,
        4.0 * scale.detach().item(),
    )
    reverse = torch.func.jacrev(next_accumulator, argnums=(0, 1))(
        scale,
        timestep,
    )
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(next_accumulator, argnums=(0, 1))(
            scale,
            timestep,
        )
    for actual_reverse, actual_forward, expected in zip(
        reverse,
        forward,
        (expected_scale_jacobian, expected_dt_jacobian),
        strict=True,
    ):
        torch.testing.assert_close(
            actual_reverse,
            expected,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        torch.testing.assert_close(
            actual_forward,
            expected,
            rtol=1.0e-12,
            atol=1.0e-13,
        )

    scale_lanes = torch.tensor((0.25, 0.75, 1.25), dtype=scale.dtype)
    dt_lanes = torch.tensor((DT / 2.0, DT, ALT_DT), dtype=timestep.dtype)
    actual = torch.vmap(next_accumulator)(scale_lanes, dt_lanes)
    expected = (4.0 * scale_lanes * dt_lanes).reshape(-1, 1, 1).expand_as(actual)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-13)
    empty = torch.vmap(next_accumulator)(scale_lanes[:0], dt_lanes[:0])
    assert empty.shape == (0, *tensors.state["integrator"]["v"].shape)

    compiled = torch.compile(next_accumulator, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        compiled_actual = compiled(scale, timestep)
    torch.testing.assert_close(
        compiled_actual,
        next_accumulator(scale, timestep),
        rtol=0.0,
        atol=0.0,
    )


def test_timestep_shape_specs_participate_in_functional_structure_fingerprint():
    from dendra.func._population import _structure_signature

    model = _model(_StructuralTimestepMechanism)
    mechanism_type = type(model.mech._StructuralTimestepMechanism)
    before = _structure_signature(model)
    original = mechanism_type._timestep_buffer_shapes
    try:
        mechanism_type._timestep_buffer_shapes = {
            **original,
            "structural_table": (1, 1, 3, 1),
        }
        after = _structure_signature(model)
    finally:
        mechanism_type._timestep_buffer_shapes = original
    assert after != before
