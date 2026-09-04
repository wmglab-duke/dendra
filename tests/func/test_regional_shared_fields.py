"""Functional contracts for regionally shared Ion and Material fields."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mod import pas

DT = 0.03125
DTYPE = torch.float64
LAYOUTS = ("rectangular", "shared_columns", "packed", "duplicates")
INJECTIVE_LAYOUTS = LAYOUTS[:-1]

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _SharedReadState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = nai + amount")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}


class _RegionalSharedReader(Mechanism):
    Mechanism.STATE_BUNDLE(_SharedReadState)
    Mechanism.USEION("na", read=["nai"])
    Mechanism.USEMATERIAL("pool", read=["amount"])


class _CurrentReadState(State):
    State.STATE("charge")
    State.DERIVATIVE("charge' = ina")

    def state_defaults(self, v, values):
        del values
        return {"charge": torch.zeros_like(v)}


class _RegionalCurrentReader(Mechanism):
    Mechanism.STATE_BUNDLE(_CurrentReadState)
    Mechanism.USEION("na", read=["ina"])


class _RegionalSodiumCurrent(Mechanism):
    Mechanism.GLOBAL(g=1.0e-4, e=-50.0)
    Mechanism.USEION("na", write=["ina"])
    Mechanism.AFFINE("ina")

    def ina(self, v):
        return self.g * (v - self.e)

    def ina_with_conductance(self, v):
        return self.ina(v), self.g.expand_as(v)


class _RegionalIonWriter(Mechanism):
    Mechanism.GLOBAL(rate=0.25)
    Mechanism.USEION("na", write=["nai"])

    def initial_values(self, v, values):
        del values
        return {"nai": torch.full_like(v, 3.0)}

    def advance(self, v, dt, values):
        del v
        return {"nai": values["nai"] + dt * self.rate}


class _RegionalMaterialReaction(Mechanism):
    Mechanism.GLOBAL(retain=0.75, rate=0.25, source_scale=0.125)
    Mechanism.USEMATERIAL(
        "pool",
        read=["amount"],
        write=["amount"],
        source={"amount": "delta"},
    )

    def advance(self, v, dt, values):
        del v
        amount = values["amount"]
        return {
            "amount": self.retain * amount + dt * self.rate,
            "delta": dt * self.source_scale * amount,
        }


class _RegionalMaterialSource(Mechanism):
    Mechanism.GLOBAL(source_scale=0.125)
    Mechanism.USEMATERIAL(
        "pool",
        read=["amount"],
        source={"amount": "delta"},
    )

    def advance(self, v, dt, values):
        del v
        return {"delta": dt * self.source_scale * values["amount"]}


class _FirstMaterialReplacement(Mechanism):
    Mechanism.USEMATERIAL("pool", read=["amount"], write=["amount"])

    def advance(self, v, dt, values):
        del v, dt
        return {"amount": values["amount"] + 1.0}


class _LastMaterialReplacement(Mechanism):
    Mechanism.USEMATERIAL("pool", read=["amount"], write=["amount"])

    def advance(self, v, dt, values):
        del v, dt
        return {"amount": 2.0 * values["amount"]}


def _diameters():
    return torch.ones((2, 5), dtype=DTYPE)


def _target(model, layout):
    if layout == "rectangular":
        return model[:, 1:4], False
    if layout == "shared_columns":
        return model[:, torch.tensor([1, 3])], False
    if layout == "packed":
        return (
            model[torch.tensor([0, 0, 1]), torch.tensor([0, 4, 2])],
            False,
        )
    if layout == "duplicates":
        return (
            model[
                torch.tensor([0, 0, 1, 1]),
                torch.tensor([1, 1, 3, 3]),
            ],
            True,
        )
    raise AssertionError(f"unknown regional layout {layout!r}")


def _insert(model, layout, mechanism):
    target, preserve_multiplicity = _target(model, layout)
    target.insert(mechanism, preserve_multiplicity=preserve_multiplicity)


def _base_model(*, material=False):
    model = dn.Unmyelinated(
        _diameters(),
        L=4.0,
        dx=1.0,
        v_init=-65.0,
        dtype=DTYPE,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    if material:
        model.material(
            "pool",
            fields={"amount": 0.5},
            min_values={"amount": 0.0},
            domain={"amount": "intracellular"},
            conserved={"amount": False},
        )
    model.insert(
        pas,
        g=torch.tensor(1.0e-8, dtype=DTYPE),
        e=torch.tensor(-65.0, dtype=DTYPE),
    )
    return model


def _flat_indices(mechanism):
    return mechanism.support_spec.materialized_flat_indices(
        mechanism.key,
        device=mechanism.diam.device,
    )


def _gather(field, mechanism):
    flat = field.reshape(-1)
    return flat.index_select(0, _flat_indices(mechanism)).reshape(
        mechanism.support_spec.runtime_local_shape
    )


def _gather_jacobian(field, mechanism):
    selector = torch.eye(
        field.numel(),
        dtype=field.dtype,
        device=field.device,
    ).index_select(0, _flat_indices(mechanism))
    return selector.reshape(*mechanism.support_spec.runtime_local_shape, *field.shape)


def _scatter_jacobian(field, mechanism):
    key = _flat_indices(mechanism)
    jacobian = field.new_zeros(field.numel(), key.numel())
    jacobian[key, torch.arange(key.numel(), device=key.device)] = 1.0
    return jacobian.reshape(
        *field.shape,
        *mechanism.support_spec.runtime_local_shape,
    )


def _replace_shared(state, *, nai=None, amount=None):
    result = state
    if nai is not None:
        result = {
            **result,
            "ions": {
                **result["ions"],
                "na": {**result["ions"]["na"], "nai": nai},
            },
        }
    if amount is not None:
        result = {
            **result,
            "materials": {
                **result["materials"],
                "pool": {**result["materials"]["pool"], "amount": amount},
            },
        }
    return result


def _assert_tree_close(actual, expected, *, rtol=2.0e-10, atol=2.0e-11):
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


def _imperative_steps(model, steps):
    dt = torch.as_tensor(DT, dtype=model.dtype(), device=model.device())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    for _ in range(steps):
        model.integrator.step(model, dt)
        model.t = model.t + dt


def _shared_read_model(layout):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = _base_model(material=True)
        _insert(model, layout, _RegionalSharedReader)
        model.initialize()
        model.train()

    nai = torch.arange(1.0, 11.0, dtype=DTYPE).reshape(model.shape)
    amount = torch.arange(20.0, 30.0, dtype=DTYPE).reshape(model.shape)
    model.mech.ions["na"].nai = nai
    model.mech.materials["pool"].amount = amount
    model.mech.read_from_ions()
    model.mech.read_from_materials()
    return model


def _current_model(layout):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = _base_model()
        _insert(model, layout, _RegionalSodiumCurrent)
        _insert(model, layout, _RegionalCurrentReader)
        model.initialize()
        model.train()
    return model


def _ion_writer_model(layout):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = _base_model()
        _insert(model, layout, _RegionalIonWriter)
        model.initialize()
        model.train()
    return model


def _material_model(layout):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = _base_model(material=True)
        _insert(model, layout, _RegionalMaterialReaction)
        model.initialize()
        model.train()
    amount = torch.arange(1.0, 11.0, dtype=DTYPE).reshape(model.shape)
    model.mech.materials["pool"].amount = amount
    model.mech.read_from_materials()
    return model


def _material_source_model(layout):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = _base_model(material=True)
        _insert(model, layout, _RegionalMaterialSource)
        model.initialize()
        model.train()
    amount = torch.arange(1.0, 11.0, dtype=DTYPE).reshape(model.shape)
    model.mech.materials["pool"].amount = amount
    model.mech.read_from_materials()
    return model


@pytest.mark.parametrize("layout", LAYOUTS)
def test_regional_shared_reads_have_exact_values_and_jacobians(layout):
    model = _shared_read_model(layout)
    mechanism = model.mech._RegionalSharedReader
    functional, tensors = dn.func.make_functional(model, dt=DT)
    nai = tensors.state["ions"]["na"]["nai"]
    amount = tensors.state["materials"]["pool"]["amount"]

    def response(nai_value, amount_value):
        state = _replace_shared(
            tensors.state,
            nai=nai_value,
            amount=amount_value,
        )
        next_state, _aux = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
        )
        return next_state["mechanisms"]["_RegionalSharedReader"]["x"]

    expected = DT * (_gather(nai, mechanism) + _gather(amount, mechanism))
    torch.testing.assert_close(response(nai, amount), expected, rtol=0.0, atol=0.0)

    reverse = torch.func.jacrev(response, argnums=(0, 1))(nai, amount)
    forward = torch.func.jacfwd(response, argnums=(0, 1))(nai, amount)
    expected_jacobian = DT * _gather_jacobian(nai, mechanism)
    for actual in (*reverse, *forward):
        torch.testing.assert_close(
            actual,
            expected_jacobian,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("layout", ["shared_columns", "duplicates"])
def test_regional_shared_reads_vmap_zero_lanes_and_compile_jacrev(layout):
    model = _shared_read_model(layout)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    nai = tensors.state["ions"]["na"]["nai"]
    amount = tensors.state["materials"]["pool"]["amount"]

    def response(nai_value):
        state = _replace_shared(tensors.state, nai=nai_value, amount=amount)
        next_state, _aux = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
        )
        return next_state["mechanisms"]["_RegionalSharedReader"]["x"]

    lanes = torch.stack((0.5 * nai, nai + 2.0))
    actual = torch.vmap(response)(lanes)
    expected = torch.stack(tuple(response(value) for value in lanes))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    empty = torch.vmap(response)(nai.new_empty((0, *nai.shape)))
    assert empty.shape[0] == 0

    transformed = torch.func.jacrev(response)
    expected_jacobian = transformed(nai)
    with torch_compiler_warning_context():
        compiled = torch.compile(
            transformed,
            backend="aot_eager",
            fullgraph=True,
            dynamic=False,
        )
        actual_jacobian = compiled(nai)
    torch.testing.assert_close(
        actual_jacobian,
        expected_jacobian,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_regional_shared_reads_match_imperative_multistep_state(layout):
    source = _shared_read_model(layout)
    imperative = _shared_read_model(layout)
    functional, tensors = dn.func.make_functional(source, dt=DT)

    actual, _aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=2,
    )
    _imperative_steps(imperative, 2)
    expected = functional.extract(imperative).state
    _assert_tree_close(actual, expected)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_regional_ionic_current_frames_match_imperative_and_forward_ad(layout):
    source = _current_model(layout)
    imperative = _current_model(layout)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    parameter_name = next(
        name
        for name in tensors.parameters
        if name.endswith("_RegionalSodiumCurrent.g_param")
    )

    def response(g):
        parameters = dict(tensors.parameters)
        parameters[parameter_name] = g
        next_state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return next_state["mechanisms"]["_RegionalCurrentReader"]["charge"]

    g = tensors.parameters[parameter_name]
    reverse = torch.func.jacrev(response)(g)
    forward = torch.func.jacfwd(response)(g)
    assert torch.count_nonzero(reverse) > 0
    torch.testing.assert_close(reverse, forward, rtol=2.0e-10, atol=2.0e-11)

    actual, _aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=2,
    )
    _imperative_steps(imperative, 2)
    expected = functional.extract(imperative).state
    _assert_tree_close(actual, expected)


def test_duplicate_regional_ionic_current_support_vmap_and_compile_jacrev():
    model = _current_model("duplicates")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameter_name = next(
        name
        for name in tensors.parameters
        if name.endswith("_RegionalSodiumCurrent.g_param")
    )
    g = tensors.parameters[parameter_name]

    def response(value):
        parameters = dict(tensors.parameters)
        parameters[parameter_name] = value
        next_state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return next_state["mechanisms"]["_RegionalCurrentReader"]["charge"]

    lanes = torch.stack((0.5 * g, 1.5 * g))
    actual = torch.vmap(response)(lanes)
    expected = torch.stack(tuple(response(value) for value in lanes))
    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)
    empty = torch.vmap(response)(g.new_empty((0,)))
    assert empty.shape[0] == 0

    transformed = torch.func.jacrev(response)
    expected_jacobian = transformed(g)
    with torch_compiler_warning_context():
        compiled = torch.compile(
            transformed,
            backend="aot_eager",
            fullgraph=True,
            dynamic=False,
        )
        actual_jacobian = compiled(g)
    torch.testing.assert_close(
        actual_jacobian,
        expected_jacobian,
        rtol=2.0e-10,
        atol=2.0e-11,
    )


@pytest.mark.parametrize("layout", INJECTIVE_LAYOUTS)
def test_injective_regional_ion_writes_have_exact_carry_and_jacobians(layout):
    source = _ion_writer_model(layout)
    imperative = _ion_writer_model(layout)
    mechanism = source.mech._RegionalIonWriter
    functional, tensors = dn.func.make_functional(source, dt=DT)
    local = tensors.state["ion_write_buffers"]["_RegionalIonWriter"]["nai"]
    canonical = tensors.state["ions"]["na"]["nai"]

    def response(local_value):
        state = {
            **tensors.state,
            "ion_write_buffers": {
                "_RegionalIonWriter": {"nai": local_value},
            },
        }
        next_state, _aux = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
        )
        return next_state["ions"]["na"]["nai"]

    expected = canonical.clone()
    expected.reshape(-1)[_flat_indices(mechanism)] = local.reshape(-1) + DT * 0.25
    torch.testing.assert_close(response(local), expected, rtol=0.0, atol=0.0)

    reverse = torch.func.jacrev(response)(local)
    forward = torch.func.jacfwd(response)(local)
    expected_jacobian = _scatter_jacobian(canonical, mechanism)
    torch.testing.assert_close(reverse, expected_jacobian, rtol=0.0, atol=0.0)
    torch.testing.assert_close(forward, expected_jacobian, rtol=0.0, atol=0.0)

    actual, _aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=2,
    )
    _imperative_steps(imperative, 2)
    _assert_tree_close(actual, functional.extract(imperative).state)


@pytest.mark.parametrize("layout", ["shared_columns", "packed"])
def test_injective_regional_ion_writes_vmap_zero_lanes_and_compile(layout):
    model = _ion_writer_model(layout)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    local = tensors.state["ion_write_buffers"]["_RegionalIonWriter"]["nai"]

    def response(local_value):
        state = {
            **tensors.state,
            "ion_write_buffers": {
                "_RegionalIonWriter": {"nai": local_value},
            },
        }
        next_state, _aux = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
        )
        return next_state["ions"]["na"]["nai"]

    lanes = torch.stack((0.5 * local, 1.5 * local))
    torch.testing.assert_close(
        torch.vmap(response)(lanes),
        torch.stack(tuple(response(value) for value in lanes)),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.vmap(response)(local.new_empty((0, *local.shape))).shape[0] == 0

    transformed = torch.func.jacrev(response)
    expected = transformed(local)
    with torch_compiler_warning_context():
        actual = torch.compile(
            transformed,
            backend="aot_eager",
            fullgraph=True,
            dynamic=False,
        )(local)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("layout", INJECTIVE_LAYOUTS)
def test_injective_regional_material_reactions_have_exact_jacobians_and_parity(
    layout,
):
    source = _material_model(layout)
    imperative = _material_model(layout)
    mechanism = source.mech._RegionalMaterialReaction
    functional, tensors = dn.func.make_functional(source, dt=DT)
    amount = tensors.state["materials"]["pool"]["amount"]

    def response(value):
        state = _replace_shared(tensors.state, amount=value)
        next_state, _aux = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
        )
        return next_state["materials"]["pool"]["amount"]

    multiplier = 0.75 + DT * 0.125
    expected = amount.clone()
    expected.reshape(-1)[_flat_indices(mechanism)] = (
        multiplier * _gather(amount, mechanism).reshape(-1) + DT * 0.25
    )
    torch.testing.assert_close(response(amount), expected, rtol=0.0, atol=0.0)

    reverse = torch.func.jacrev(response)(amount)
    forward = torch.func.jacfwd(response)(amount)
    expected_jacobian = torch.eye(amount.numel(), dtype=DTYPE).reshape(
        *amount.shape,
        *amount.shape,
    )
    expected_jacobian = expected_jacobian.clone().reshape(amount.numel(), -1)
    key = _flat_indices(mechanism)
    expected_jacobian[key, key] = multiplier
    expected_jacobian = expected_jacobian.reshape(*amount.shape, *amount.shape)
    torch.testing.assert_close(reverse, expected_jacobian, rtol=0.0, atol=0.0)
    torch.testing.assert_close(forward, expected_jacobian, rtol=0.0, atol=0.0)

    actual, _aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=2,
    )
    _imperative_steps(imperative, 2)
    _assert_tree_close(actual, functional.extract(imperative).state)


@pytest.mark.parametrize("layout", ["shared_columns", "packed"])
def test_injective_regional_material_reactions_vmap_zero_lanes_and_compile(layout):
    model = _material_model(layout)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    amount = tensors.state["materials"]["pool"]["amount"]

    def response(value):
        state = _replace_shared(tensors.state, amount=value)
        next_state, _aux = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
        )
        return next_state["materials"]["pool"]["amount"]

    lanes = torch.stack((0.5 * amount, 1.5 * amount))
    torch.testing.assert_close(
        torch.vmap(response)(lanes),
        torch.stack(tuple(response(value) for value in lanes)),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.vmap(response)(amount.new_empty((0, *amount.shape))).shape[0] == 0

    transformed = torch.func.jacrev(response)
    expected = transformed(amount)
    with torch_compiler_warning_context():
        actual = torch.compile(
            transformed,
            backend="aot_eager",
            fullgraph=True,
            dynamic=False,
        )(amount)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_overlapping_regional_material_replacements_preserve_authored_order():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = _base_model(material=True)
        model[:, 1:4].insert(_FirstMaterialReplacement)
        model[:, 2:5].insert(_LastMaterialReplacement)
        model.initialize()
        model.train()
    amount = torch.arange(1.0, 11.0, dtype=DTYPE).reshape(model.shape)
    model.mech.materials["pool"].amount = amount
    model.mech.read_from_materials()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    next_state, _aux = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
    )
    actual = next_state["materials"]["pool"]["amount"]
    expected = amount.clone()
    expected[:, 1] = amount[:, 1] + 1.0
    expected[:, 2:5] = 2.0 * amount[:, 2:5]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_duplicate_regional_material_sources_accumulate_and_transform_correctly():
    source = _material_source_model("duplicates")
    imperative = _material_source_model("duplicates")
    mechanism = source.mech._RegionalMaterialSource
    functional, tensors = dn.func.make_functional(source, dt=DT)
    amount = tensors.state["materials"]["pool"]["amount"]

    def response(value):
        state = _replace_shared(tensors.state, amount=value)
        next_state, _aux = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            state,
        )
        return next_state["materials"]["pool"]["amount"]

    key = _flat_indices(mechanism)
    expected = amount.clone().reshape(-1)
    expected.scatter_add_(
        0,
        key,
        DT * 0.125 * amount.reshape(-1).index_select(0, key),
    )
    expected = expected.reshape(amount.shape)
    torch.testing.assert_close(response(amount), expected, rtol=0.0, atol=0.0)

    reverse = torch.func.jacrev(response)(amount)
    forward = torch.func.jacfwd(response)(amount)
    expected_jacobian = torch.eye(amount.numel(), dtype=DTYPE)
    multiplicity = torch.bincount(key, minlength=amount.numel()).to(DTYPE)
    diagonal = 1.0 + DT * 0.125 * multiplicity
    expected_jacobian.diagonal().copy_(diagonal)
    expected_jacobian = expected_jacobian.reshape(*amount.shape, *amount.shape)
    torch.testing.assert_close(reverse, expected_jacobian, rtol=0.0, atol=0.0)
    torch.testing.assert_close(forward, expected_jacobian, rtol=0.0, atol=0.0)

    source_scale_name = next(
        name
        for name in tensors.parameters
        if name.endswith("_RegionalMaterialSource.source_scale_param")
    )
    source_scale = tensors.parameters[source_scale_name]

    def scale_response(value):
        parameters = dict(tensors.parameters)
        parameters[source_scale_name] = value
        next_state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return next_state["materials"]["pool"]["amount"]

    scale_reverse = torch.func.jacrev(scale_response)(source_scale)
    scale_forward = torch.func.jacfwd(scale_response)(source_scale)
    expected_scale_gradient = torch.zeros_like(amount).reshape(-1)
    expected_scale_gradient.scatter_add_(
        0,
        key,
        DT * amount.reshape(-1).index_select(0, key),
    )
    expected_scale_gradient = expected_scale_gradient.reshape(amount.shape)
    torch.testing.assert_close(
        scale_reverse,
        expected_scale_gradient,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        scale_forward,
        expected_scale_gradient,
        rtol=0.0,
        atol=0.0,
    )
    scale_lanes = torch.stack((0.5 * source_scale, 1.5 * source_scale))
    torch.testing.assert_close(
        torch.vmap(scale_response)(scale_lanes),
        torch.stack(tuple(scale_response(value) for value in scale_lanes)),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.vmap(scale_response)(source_scale.new_empty((0,))).shape[0] == 0

    lanes = torch.stack((0.5 * amount, 1.5 * amount))
    torch.testing.assert_close(
        torch.vmap(response)(lanes),
        torch.stack(tuple(response(value) for value in lanes)),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.vmap(response)(amount.new_empty((0, *amount.shape))).shape[0] == 0

    transformed = torch.func.jacrev(response)
    with torch_compiler_warning_context():
        compiled = torch.compile(
            transformed,
            backend="aot_eager",
            fullgraph=True,
            dynamic=False,
        )
        compiled_jacobian = compiled(amount)
    torch.testing.assert_close(
        compiled_jacobian,
        expected_jacobian,
        rtol=0.0,
        atol=0.0,
    )

    actual, _aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=2,
    )
    _imperative_steps(imperative, 2)
    _assert_tree_close(actual, functional.extract(imperative).state)


@pytest.mark.parametrize("kind", ["ion", "material"])
def test_noninjective_regional_shared_field_writes_fail_closed(kind):
    model = (
        _ion_writer_model("duplicates")
        if kind == "ion"
        else _material_model("duplicates")
    )
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"non-injective regional shared-field replacement writes",
    ):
        dn.func.make_functional(model, dt=DT)
