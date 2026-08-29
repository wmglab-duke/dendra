"""Fresh functional initialization for canonical Ion and Material state."""

from __future__ import annotations

import copy
from types import MethodType

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.parametric import PositiveParam

DT = 0.01
DTYPE = torch.float64


class _SodiumConcentrationState(State):
    State.STATE("nai")
    State.DERIVATIVE("nai' = 0")

    def state_defaults(self, v, values):
        del values
        return {"nai": torch.full_like(v, 5.0)}


class _InitialSodiumWriter(Mechanism):
    Mechanism.STATE_BUNDLE(_SodiumConcentrationState)
    Mechanism.USEION("na", write=["nai"])


class _ClampedSodiumConcentrationState(State):
    State.STATE("nai")
    State.DERIVATIVE("nai' = 0")

    def state_defaults(self, v, values):
        del values
        return {"nai": torch.full_like(v, -2.0)}


class _ClampedSodiumWriter(Mechanism):
    Mechanism.STATE_BUNDLE(_ClampedSodiumConcentrationState)
    Mechanism.USEION("na", write=["nai"])


class _SodiumCurrent(Mechanism):
    Mechanism.GLOBAL(g=0.01)
    Mechanism.USEION("na", read=["ena"], write=["ina"])
    Mechanism.AFFINE("ina")

    def ina(self, v):
        return self.g * (v - self.ena)

    def ina_with_conductance(self, v):
        return self.ina(v), self.g.expand_as(v)


class _CurrentReaderState(State):
    State.STATE("seen")
    State.DERIVATIVE("seen' = 0")

    def initial_values(self, v, values):
        del v
        return {"seen": values["ina"].clone()}


class _InitialCurrentReader(Mechanism):
    Mechanism.STATE_BUNDLE(_CurrentReaderState)
    Mechanism.USEION("na", read=["ina"])


class _PlainSharedAliasState(State):
    State.STATE("seen")
    State.DERIVATIVE("seen' = 0")

    def initial_values(self, v, values):
        del v
        return {"seen": values["nai"] + values["delta"]}


class _PlainSharedAliasMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_PlainSharedAliasState)
    Mechanism.USEION("na", write=["nai"])
    Mechanism.USEMATERIAL("pool", source={"amount": "delta"})


class _HiddenSharedAliasState(State):
    State.STATE("seen")
    State.DERIVATIVE("seen' = 0")

    def initial_values(self, v, values):
        del v, values
        return {"seen": self.nai.clone()}


class _HiddenSharedAliasMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_HiddenSharedAliasState)
    Mechanism.USEION("na", write=["nai"])


class _SetupState(State):
    State.STATE("value")
    State.DERIVATIVE("value' = 0")

    def state_defaults(self, v, values):
        del values
        return {"value": torch.zeros_like(v)}


class _PopulateOverrideMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_SetupState)

    def populate(self, random_generation=None):
        return super().populate(random_generation=random_generation)


class _InitRngOverrideState(_SetupState):
    def init_rng(self):
        return super().init_rng()


class _StateInitRngOverrideMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_InitRngOverrideState)


class _AcceptedCurrentMean(torch.nn.Module):
    def forward(self, current):
        return (current.mean(),)


def _model(
    *,
    material_source=None,
    ion_style=None,
    sodium_writer=_InitialSodiumWriter,
):
    if material_source is None:
        material_source = torch.nn.Parameter(torch.tensor(1.25, dtype=DTYPE))
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=3,
            v_init=torch.tensor([-68.0, -64.0, -60.0], dtype=DTYPE),
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.material(
            "pool",
            fields={"amount": material_source},
            min_values={"amount": 0.5},
            conserved={"amount": False},
        )
        if ion_style is not None:
            model.ion_style("na", *ion_style)
        model.insert(sodium_writer)
        model.insert(_SodiumCurrent)
        model.insert(_InitialCurrentReader)
        model.initialize()
        model.train()
    return model


def _broadcast_source_model():
    ion_source = torch.nn.Parameter(torch.tensor([120.0, 140.0, 160.0], dtype=DTYPE))
    material_source = torch.nn.Parameter(torch.tensor([0.75, 1.25, 1.75], dtype=DTYPE))
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=2,
            C=3,
            v_init=torch.tensor([-68.0, -64.0, -60.0], dtype=DTYPE),
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.concentrations(nao0=ion_source)
        model.material(
            "pool",
            fields={"amount": material_source},
            min_values={"amount": 0.5},
            conserved={"amount": False},
        )
        model.insert(_InitialSodiumWriter)
        model.insert(_SodiumCurrent)
        model.insert(_InitialCurrentReader)
        model.initialize()
        model.train()
    return model


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


def _parameter_name(parameters, suffix):
    return next(name for name in parameters if name.endswith(suffix))


def test_fresh_shared_field_initialization_matches_every_imperative_leaf():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    v_init = torch.tensor([[-71.0, -63.0, -55.0]], dtype=DTYPE)

    actual = functional.initialize(
        tensors.parameters,
        tensors.constants,
        dn.func.InitializationInput(v_init=v_init),
    )
    reference = copy.deepcopy(model)
    reference.set_v_init(v_init)
    reference.initialize()
    expected = functional.extract(reference)

    _assert_tree_close(actual.state, expected.state)
    ion = actual.state["ions"]["na"]
    torch.testing.assert_close(ion["nai"], torch.full_like(ion["nai"], 5.0))
    torch.testing.assert_close(
        actual.state["materials"]["pool"]["amount"],
        torch.full_like(actual.state["materials"]["pool"]["amount"], 1.25),
    )

    reader_name = next(
        name
        for name, mechanism in model.mech.mechanisms.items()
        if isinstance(mechanism, _InitialCurrentReader)
    )
    provisional = actual.state["mechanisms"][reader_name]["seen"]
    assert not torch.equal(provisional, ion["ina"])


def test_advance_e_false_retains_the_precommit_initialized_reversal():
    model = _model(ion_style=(1, 0))
    functional, tensors = dn.func.make_functional(model, dt=DT)

    actual = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
    )
    reference = copy.deepcopy(model)
    reference.initialize()
    _assert_tree_close(actual.state, functional.extract(reference).state)

    source_ion = model.mech.ions["na"]
    ion = actual.state["ions"]["na"]
    expected_ena = (
        torch.log(source_ion.o_init / source_ion.i_init)
        * source_ion.rzf
        * (273.15 + model.celsius)
    ).expand_as(ion["ena"])
    torch.testing.assert_close(ion["ena"], expected_ena, rtol=0.0, atol=0.0)
    torch.testing.assert_close(ion["nai"], torch.full_like(ion["nai"], 5.0))

    reader_name = next(
        name
        for name, mechanism in model.mech.mechanisms.items()
        if isinstance(mechanism, _InitialCurrentReader)
    )
    provisional = actual.state["mechanisms"][reader_name]["seen"]
    torch.testing.assert_close(provisional, ion["ina"], rtol=0.0, atol=0.0)


def test_concentration_replacement_is_guarded_before_reversal_and_current_refresh():
    model = _model(
        ion_style=(1, 1),
        sodium_writer=_ClampedSodiumWriter,
    )
    functional, tensors = dn.func.make_functional(model, dt=DT)

    actual = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
    )
    reference = copy.deepcopy(model)
    reference.initialize()
    _assert_tree_close(actual.state, functional.extract(reference).state)

    source_ion = model.mech.ions["na"]
    ion = actual.state["ions"]["na"]
    minimum = source_ion._buffers["_min_nai"].expand_as(ion["nai"])
    torch.testing.assert_close(ion["nai"], minimum, rtol=0.0, atol=0.0)
    expected_ena = (
        torch.log(ion["nao"] / minimum) * source_ion.rzf * (273.15 + model.celsius)
    )
    torch.testing.assert_close(ion["ena"], expected_ena, rtol=0.0, atol=0.0)
    assert all(torch.isfinite(value).all() for value in ion.values())


def test_material_initial_source_is_guarded_in_the_fresh_functional_frame():
    source = torch.nn.Parameter(torch.tensor(-2.0, dtype=DTYPE))
    model = _model(material_source=source)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    actual = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
    )
    reference = copy.deepcopy(model)
    reference.initialize()
    _assert_tree_close(actual.state, functional.extract(reference).state)

    amount = actual.state["materials"]["pool"]["amount"]
    torch.testing.assert_close(amount, torch.full_like(amount, 0.5))


def test_shared_initial_sources_support_higher_order_autograd_vmap_and_compile():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    nao_name = _parameter_name(tensors.parameters, "ions.na.o_init")
    amount_name = _parameter_name(
        tensors.parameters,
        "materials.pool._initial_sources.amount",
    )
    base_nao = tensors.parameters[nao_name]
    base_amount = tensors.parameters[amount_name]

    def initialized_values(nao, amount):
        parameters = dict(tensors.parameters)
        parameters[nao_name] = nao
        parameters[amount_name] = amount
        initialized = functional.initialize(
            parameters,
            tensors.constants,
            tensors.initialization,
        )
        return (
            initialized.state["ions"]["na"]["ina"],
            initialized.state["materials"]["pool"]["amount"],
        )

    reverse = torch.func.jacrev(initialized_values, argnums=(0, 1))(
        base_nao,
        base_amount,
    )
    forward = torch.func.jacfwd(initialized_values, argnums=(0, 1))(
        base_nao,
        base_amount,
    )
    _assert_tree_close(reverse, forward, rtol=2.0e-10, atol=2.0e-11)
    assert torch.count_nonzero(reverse[0][0]) > 0
    assert torch.count_nonzero(reverse[1][1]) > 0

    def loss(nao):
        current, _amount = initialized_values(nao, base_amount)
        return current.square().mean()

    hessian = torch.func.hessian(loss)(base_nao)
    assert torch.isfinite(hessian).all()
    assert torch.count_nonzero(hessian) > 0

    nao_lanes = torch.stack((0.8 * base_nao, 1.2 * base_nao))
    amount_lanes = torch.stack((0.75 * base_amount, 1.25 * base_amount))
    actual_lanes = torch.vmap(initialized_values)(nao_lanes, amount_lanes)
    expected_lanes = tuple(
        torch.stack(values)
        for values in zip(
            *(
                initialized_values(nao, amount)
                for nao, amount in zip(
                    nao_lanes,
                    amount_lanes,
                    strict=True,
                )
            ),
            strict=True,
        )
    )
    _assert_tree_close(actual_lanes, expected_lanes)
    empty = torch.vmap(initialized_values)(nao_lanes[:0], amount_lanes[:0])
    assert empty[0].shape == (0, *model.shape)
    assert empty[1].shape == (0, *model.shape)

    transformed = torch.func.jacrev(lambda nao: initialized_values(nao, base_amount)[0])
    expected = transformed(base_nao)
    with torch_compiler_warning_context():
        actual = torch.compile(
            transformed,
            backend="aot_eager",
            fullgraph=True,
            dynamic=False,
        )(base_nao)
    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


def test_non_scalar_direct_initial_sources_broadcast_and_vmap_with_zero_lanes():
    model = _broadcast_source_model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    nao_name = _parameter_name(tensors.parameters, "ions.na.o_init")
    amount_name = _parameter_name(
        tensors.parameters,
        "materials.pool._initial_sources.amount",
    )
    base_nao = tensors.parameters[nao_name]
    base_amount = tensors.parameters[amount_name]
    assert base_nao.shape == (3,)
    assert base_amount.shape == (3,)

    def initialized_fields(nao, amount):
        parameters = dict(tensors.parameters)
        parameters[nao_name] = nao
        parameters[amount_name] = amount
        initialized = functional.initialize(
            parameters,
            tensors.constants,
            tensors.initialization,
        )
        return (
            initialized.state["ions"]["na"]["nao"],
            initialized.state["materials"]["pool"]["amount"],
        )

    actual = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
    )
    reference = copy.deepcopy(model)
    reference.initialize()
    _assert_tree_close(actual.state, functional.extract(reference).state)
    fields = initialized_fields(base_nao, base_amount)
    torch.testing.assert_close(fields[0], base_nao.expand(model.shape))
    torch.testing.assert_close(fields[1], base_amount.expand(model.shape))

    nao_lanes = torch.stack((0.9 * base_nao, 1.1 * base_nao))
    amount_lanes = torch.stack((0.8 * base_amount, 1.2 * base_amount))
    vmapped = torch.vmap(initialized_fields)(nao_lanes, amount_lanes)
    looped = tuple(
        torch.stack(values)
        for values in zip(
            *(
                initialized_fields(nao, amount)
                for nao, amount in zip(
                    nao_lanes,
                    amount_lanes,
                    strict=True,
                )
            ),
            strict=True,
        )
    )
    _assert_tree_close(vmapped, looped)

    empty = torch.vmap(initialized_fields)(nao_lanes[:0], amount_lanes[:0])
    assert empty[0].shape == (0, *model.shape)
    assert empty[1].shape == (0, *model.shape)


@pytest.mark.parametrize("owner_kind", ["ion", "material"])
def test_nonbroadcastable_direct_initial_source_fails_closed(owner_kind):
    model = _model()
    invalid = torch.nn.Parameter(torch.ones(2, dtype=DTYPE))
    if owner_kind == "ion":
        model.mech.ions["na"].o_init = invalid
    else:
        model.mech.materials["pool"]._initial_sources.amount = invalid

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"unsupported raw parameter shapes.*broadcastable",
    ):
        dn.func.make_functional(model, dt=DT)


def test_shared_initialization_vmaps_over_temperature_including_zero_lanes():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    celsius_name = "celsius_param"
    base_celsius = tensors.parameters[celsius_name]

    def initialized_reversal(celsius):
        parameters = dict(tensors.parameters)
        parameters[celsius_name] = celsius
        initialized = functional.initialize(
            parameters,
            tensors.constants,
            tensors.initialization,
        )
        return initialized.state["ions"]["na"]["ena"]

    lanes = torch.stack((base_celsius - 2.0, base_celsius + 3.0))
    actual = torch.vmap(initialized_reversal)(lanes)
    expected = torch.stack(tuple(initialized_reversal(value) for value in lanes))
    torch.testing.assert_close(actual, expected)

    empty = torch.vmap(initialized_reversal)(lanes[:0])
    assert empty.shape == (0, *model.shape)


def test_shared_column_initialization_vmaps_over_temperature_with_zero_lanes():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            torch.ones((2, 5), dtype=DTYPE),
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        target = model[:, torch.tensor([1, 3])]
        for mechanism in (
            _InitialSodiumWriter,
            _SodiumCurrent,
            _InitialCurrentReader,
        ):
            target.insert(mechanism)
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    base_celsius = tensors.parameters["celsius_param"]

    def initialized_reversal(celsius):
        parameters = dict(tensors.parameters)
        parameters["celsius_param"] = celsius
        initialized = functional.initialize(
            parameters,
            tensors.constants,
            tensors.initialization,
        )
        return initialized.state["ions"]["na"]["ena"]

    lanes = torch.stack((base_celsius - 2.0, base_celsius + 3.0))
    actual = torch.vmap(initialized_reversal)(lanes)
    expected = torch.stack(tuple(initialized_reversal(value) for value in lanes))
    torch.testing.assert_close(actual, expected)

    empty = torch.vmap(initialized_reversal)(lanes[:0])
    assert empty.shape == (0, *model.shape)


def test_post_transform_reads_the_final_accepted_ion_frame():
    model = _model()
    model.register_post_initialize_transform(
        "accepted_current",
        _AcceptedCurrentMean(),
        reads=("state.ions.na.ina",),
        writes=("parameters.celsius_param",),
    )
    model.initialize()
    model.train()
    functional, tensors = dn.func.make_functional(model, dt=DT)

    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
    )
    accepted = initialized.state["ions"]["na"]["ina"]
    torch.testing.assert_close(
        initialized.parameters["celsius_param"],
        accepted.mean(),
    )

    reference = copy.deepcopy(model)
    reference.initialize()
    expected = functional.extract(reference)
    _assert_tree_close(initialized.state, expected.state)
    torch.testing.assert_close(
        initialized.parameters["celsius_param"],
        expected.parameters["celsius_param"],
    )


def test_post_transform_reads_the_guarded_material_frame():
    source = torch.nn.Parameter(torch.tensor(-2.0, dtype=DTYPE))
    model = _model(material_source=source)
    model.register_post_initialize_transform(
        "accepted_material",
        _AcceptedCurrentMean(),
        reads=("state.materials.pool.amount",),
        writes=("parameters.celsius_param",),
    )
    model.initialize()
    model.train()
    functional, tensors = dn.func.make_functional(model, dt=DT)

    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
    )
    amount = initialized.state["materials"]["pool"]["amount"]
    torch.testing.assert_close(
        initialized.parameters["celsius_param"],
        amount.mean(),
    )

    reference = copy.deepcopy(model)
    reference.initialize()
    expected = functional.extract(reference)
    _assert_tree_close(initialized.state, expected.state)
    torch.testing.assert_close(
        initialized.parameters["celsius_param"],
        expected.parameters["celsius_param"],
    )


def test_shared_transform_references_are_post_read_only():
    model = _model()
    with pytest.raises(ValueError, match="only be read by post transforms"):
        model.register_pre_initialize_transform(
            "too_early",
            _AcceptedCurrentMean(),
            reads=("state.ions.na.ina",),
            writes=("parameters.celsius_param",),
        )
    with pytest.raises(ValueError, match="read-only"):
        model.register_post_initialize_transform(
            "write_shared",
            _AcceptedCurrentMean(),
            reads=("state.ions.na.ina",),
            writes=("state.ions.na.ina",),
        )


def test_state_initial_values_receive_explicit_shared_write_and_source_aliases():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=2,
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.material(
            "pool",
            fields={"amount": torch.tensor(1.0, dtype=DTYPE)},
            conserved={"amount": False},
        )
        model.insert(_PlainSharedAliasMechanism)
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    actual = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
    )
    expected = functional.extract(model)

    _assert_tree_close(actual.state, expected.state)
    seen = actual.state["mechanisms"]["_PlainSharedAliasMechanism"]["seen"]
    torch.testing.assert_close(
        seen,
        actual.state["ions"]["na"]["nai"],
        rtol=0.0,
        atol=0.0,
    )


def test_state_initial_values_reject_hidden_shared_alias_reads():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=2,
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(_HiddenSharedAliasMechanism)
        model.initialize()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"reads receiver attributes.*nai.*values mapping",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


@pytest.mark.parametrize(
    ("mechanism", "message"),
    [
        (_PopulateOverrideMechanism, r"overrides initialization hooks.*populate"),
        (
            _StateInitRngOverrideMechanism,
            r"State .*overrides initialization hooks.*init_rng",
        ),
    ],
)
def test_replaced_imperative_mechanism_setup_hooks_fail_closed(mechanism, message):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=2,
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(mechanism)
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(dn.func.FunctionalizationError, match=message):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


@pytest.mark.parametrize(
    ("owner_kind", "hook_name", "message"),
    [
        (
            "population",
            "pre_initialize",
            r"Population replaces canonical fresh-initialization hooks.*pre_initialize",
        ),
        (
            "integrator",
            "init_v",
            r"Integrator replaces canonical fresh-initialization hooks.*init_v",
        ),
    ],
)
def test_replaced_outer_initialization_hooks_fail_closed(
    owner_kind,
    hook_name,
    message,
):
    model = _model()
    owner = model if owner_kind == "population" else model.integrator

    def replacement(self, *args, **kwargs):
        del self, args, kwargs

    setattr(owner, hook_name, MethodType(replacement, owner))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(dn.func.FunctionalizationError, match=message):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


@pytest.mark.parametrize("owner_kind", ["ion", "material"])
def test_replaced_imperative_shared_initialization_hooks_fail_closed(owner_kind):
    model = _model()
    if owner_kind == "ion":
        owner = model.mech.ions["na"]

        def replacement(self, celsius):
            del self, celsius

    else:
        owner = model.mech.materials["pool"]

        def replacement(self, *args, **kwargs):
            del self, args, kwargs

    owner.initialize = MethodType(replacement, owner)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=rf"{owner_kind.capitalize()} .*replaces canonical initialization hooks",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


@pytest.mark.parametrize("owner_kind", ["ion", "material"])
def test_replaced_eval_detach_hooks_fail_closed(owner_kind):
    model = _model()
    model.eval()
    owner = (
        model.mech.ions["na"] if owner_kind == "ion" else model.mech.materials["pool"]
    )

    def replacement(self):
        del self

    owner.detach = MethodType(replacement, owner)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=rf"{owner_kind.capitalize()} .*replaces canonical initialization hooks",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


def test_class_level_shared_initialization_hook_replacement_fails_closed(monkeypatch):
    model = _model()
    ion_type = type(model.mech.ions["na"])

    def replacement(self, celsius):
        del self, celsius

    monkeypatch.setattr(ion_type, "advance", replacement)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"Ion .*replaces canonical initialization hooks.*advance",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


@pytest.mark.parametrize("hook_name", ["ion_init", "material_init", "write_to_ions"])
def test_replaced_handler_initialization_hooks_fail_closed(hook_name):
    model = _model()

    def replacement(self, *args, **kwargs):
        del self, args, kwargs

    setattr(model.mech, hook_name, MethodType(replacement, model.mech))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=rf"MechanismHandler replaces canonical initialization hooks.*{hook_name}",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


def test_callable_material_initial_source_remains_fail_closed():
    source = PositiveParam(torch.tensor(1.25, dtype=DTYPE), requires_grad=True)
    model = _model(material_source=source)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="initial sources.*callable Modules",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )
