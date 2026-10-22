"""TABLE contracts classify explicit leaves and guard authored dispatch."""

import copy
from types import MethodType

import pytest
import torch

import dendra as dn
from dendra.func._tables import (
    TablePlan,
    canonical_table_wrapper,
    table_buffer_slots,
    table_failures,
    table_owners,
    table_parameter_names,
    table_wrapper_original,
)
from dendra.models.mechanisms import Mechanism, State


class _TableState(State):
    State.STATE("m")
    State.GLOBAL(tau=1.0)
    State.TABLE("rates", -100.0, 100.0, 41, learnable=True)
    State.ASSIGNED("target")
    State.DERIVATIVE("m' = (target - m) / tau")
    State.METHOD("cnexp")

    def rates(self, x):
        return 0.5 + x * 0.001

    def assigned_values(self, v, values):
        del values
        return {"target": self.rates(v)}

    def state_defaults(self, v, values):
        del values
        return {"m": torch.zeros_like(v)}


class _TableCurrent(Mechanism):
    Mechanism.STATE_BUNDLE(_TableState)
    Mechanism.GLOBAL(gain=0.001)
    Mechanism.TABLE("curve", -100.0, 100.0, 41)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def curve(self, x):
        return self.gain * x

    def i(self, v):
        return self.curve(v)


def _population():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.SingleCompartment(C=2, dtype=torch.float64)
        population.insert(_TableCurrent.rename("table"))
        population.initialize()
    return population


def test_table_plan_classifies_state_and_mechanism_constants_and_parameters():
    population = _population()
    plan = TablePlan(population)
    owners = table_owners(population)
    assert [name for _path, _owner, name, _table in owners] == ["curve", "rates"]
    assert not table_failures(owners[0][1])
    assert not table_failures(owners[1][1])
    assert table_parameter_names(population) == frozenset(
        {"integrator.mech.mechanisms.table.DE._TableState.rates_table.y"}
    )
    slots = dict(table_buffer_slots(owners[0][1]))
    assert "curve_table._slopes" in slots
    assert "curve_table._y" in slots
    state_slots = dict(table_buffer_slots(owners[1][1]))
    assert "rates_table._slopes" not in state_slots
    assert "rates_table._inv_dx_uniform" in state_slots
    assert len(table_owners(population, (owners[1][1],))) == 1
    assert table_parameter_names(population, (owners[0][1],)) == frozenset()
    plan.validate_source()


def test_table_plan_accepts_owned_clones_and_compatible_numerical_values():
    population = _population()
    plan = TablePlan(population)
    clone = copy.deepcopy(population)
    for _path, _owner, _name, table in table_owners(clone):
        with torch.no_grad():
            if table.learnable_y:
                table.y.add_(0.1)
            else:
                table._y.add_(0.2)
    plan.validate_population(clone)
    plan.validate_population(_population())


@pytest.mark.parametrize("change", ("flag", "declaration", "metadata", "shape"))
def test_table_plan_rejects_changed_structure(change, monkeypatch):
    population = _population()
    plan = TablePlan(population)
    _path, owner, name, table = table_owners(population)[0]
    if change == "flag":
        owner.usetables(False)
    elif change == "declaration":
        monkeypatch.setattr(
            owner.__class__, "_table", copy.deepcopy(owner.__class__._table)
        )
        owner.__class__._table[name]["high"] = 200.0
    elif change == "metadata":
        table.exact_clamp = True
    else:
        table._slopes = table._slopes[:-1]
    with pytest.raises(dn.func.FunctionalizationError, match="TABLE"):
        plan.validate_source()


def test_table_plan_guards_original_helper_binding_and_generated_code():
    population = _population()
    plan = TablePlan(population)
    _path, owner, name, _table = table_owners(population)[0]
    assert canonical_table_wrapper(owner, name)
    assert table_wrapper_original(owner, owner.curve) == (name, owner.curve_original)
    assert table_wrapper_original(owner, owner.curve_original) is None

    def replacement(self, x):
        return self.gain * x * 2

    owner.curve_original = MethodType(replacement, owner)
    with pytest.raises(dn.func.FunctionalizationError, match="TABLE"):
        plan.validate_source()
    owner.curve = MethodType(replacement, owner)
    assert any("wrapper" in failure for failure in table_failures(owner))


@pytest.mark.parametrize(
    "change",
    ("forward", "module", "hooks", "schema", "_call_impl", "_compiled_call_impl"),
)
def test_table_plan_fails_closed_for_custom_interpolation(change):
    population = _population()
    _path, owner, name, table = table_owners(population)[0]
    if change == "forward":
        table.forward = lambda x: x
    elif change == "module":
        setattr(owner, f"{name}_table", torch.nn.Identity())
    elif change == "hooks":
        table.register_forward_hook(lambda _module, _args, result: result)
    elif change == "schema":
        table.register_buffer("hidden", torch.ones(()))
    else:
        setattr(table, change, lambda x: x)
    with pytest.raises(dn.func.FunctionalizationError, match="TABLE"):
        TablePlan(population)


@pytest.mark.parametrize(
    "field", ("outside", "fill_value", "batched", "N", "_use_uniform")
)
def test_table_plan_rejects_effectful_metadata_without_evaluating_it(field):
    class EffectfulMetadata:
        def __init__(self):
            self.calls = 0

        def __bool__(self):
            self.calls += 1
            return True

        def __eq__(self, _other):
            self.calls += 1
            return True

    population = _population()
    _path, _owner, _name, table = table_owners(population)[0]
    plan = TablePlan(population)
    before = plan.source_binding_signature()
    metadata = EffectfulMetadata()
    setattr(table, field, metadata)
    assert plan.source_binding_signature() != before
    with pytest.raises(dn.func.FunctionalizationError, match="literal"):
        TablePlan(population)
    assert metadata.calls == 0


def test_table_plan_checks_authored_helper_purity():
    population = _population()
    _path, owner, _name, _table = table_owners(population)[0]

    def mutating(self, x):
        return x.add_(self.gain)

    owner.curve_original = MethodType(mutating, owner)
    owner.usetables(False)
    with dn.ctx(CHECK_PURITY=True):
        with pytest.raises(dn.func.FunctionalizationError, match="in-place"):
            TablePlan(population)


def test_enabled_table_does_not_audit_construction_only_generator_side_effects():
    class CountingGenerator:
        def __init__(self):
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            return x * 0.001

    generator = CountingGenerator()

    class ConstructedTable(Mechanism):
        Mechanism.GLOBAL(scale=1.0)
        Mechanism.TABLE("curve", -100.0, 100.0, 41)
        Mechanism.NONSPECIFIC_CURRENT("i")
        Mechanism.NUMERICAL("i")

        def curve(self, x):
            return self.scale * generator(x)

        def i(self, v):
            return self.curve(v)

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.SingleCompartment(C=2, dtype=torch.float64)
        population.insert(ConstructedTable.rename("table"))
        population.initialize()
    assert generator.calls == 1
    plan = TablePlan(population)
    generator.calls += 1
    plan.validate_source()
    before = generator.calls
    functional, tensors = dn.func.make_functional(population, dt=0.125)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    assert generator.calls == before
    owner = population.mech.mechanisms["table"]
    owner.usetables(False)
    with dn.ctx(CHECK_PURITY=True):
        with pytest.raises(dn.func.FunctionalizationError, match="dependency|callable"):
            TablePlan(population)


@pytest.mark.parametrize("owner_kind", ("mechanism", "state"))
@pytest.mark.parametrize("learnable", (False, True))
def test_parameter_free_table_owners_preserve_initialized_dtype(owner_kind, learnable):
    if owner_kind == "mechanism":

        class ParameterFreeMechanism(Mechanism):
            Mechanism.TABLE("curve", -100.0, 100.0, 41, learnable=learnable)
            Mechanism.NONSPECIFIC_CURRENT("i")
            Mechanism.NUMERICAL("i")

            def curve(self, x):
                return x * 0.001

            def i(self, v):
                return self.curve(v)

        mechanism_type = ParameterFreeMechanism
    else:

        class ParameterFreeState(State):
            State.STATE("x")
            State.ASSIGNED("target")
            State.TABLE("curve", -100.0, 100.0, 41, learnable=learnable)
            State.DERIVATIVE("x' = target - x")
            State.METHOD("cnexp")

            def curve(self, v):
                return v * 0.001

            def assigned_values(self, v, values):
                del values
                return {"target": self.curve(v)}

            def state_defaults(self, v, values):
                del values
                return {"x": torch.zeros_like(v)}

        class StateMechanism(Mechanism):
            Mechanism.STATE_BUNDLE(ParameterFreeState)
            Mechanism.RANGE(g=0.001)
            Mechanism.NONSPECIFIC_CURRENT("i")
            Mechanism.AFFINE("i")

            def i_affine_terms(self, v):
                del v
                return self.g, self.g * -65.0

        mechanism_type = StateMechanism

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.SingleCompartment(C=2, dtype=torch.float64)
        population.insert(mechanism_type.rename("table"))
        population.initialize()
    TablePlan(population).validate_source()
    _path, owner, name, table = table_owners(population)[0]
    assert table._x_search.dtype == torch.float64
    assert table._x_search.device.type == "cpu"
    assert table.learnable_y is learnable
    query = torch.tensor([[-72.5, -12.5]], dtype=torch.float64)
    torch.testing.assert_close(getattr(owner, name)(query), query * 0.001)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("learnable", (False, True))
def test_table_construction_samples_beside_registered_cuda_parameters(learnable):
    class DeviceTable(Mechanism):
        Mechanism.GLOBAL(gain=0.001)
        Mechanism.TABLE("curve", -100.0, 100.0, 41, learnable=learnable)
        Mechanism.NONSPECIFIC_CURRENT("i")
        Mechanism.NUMERICAL("i")

        def curve(self, x):
            return self.gain * x

        def i(self, v):
            return self.curve(v)

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.SingleCompartment(C=2, dtype=torch.float64, device="cuda")
        population.insert(DeviceTable.rename("table"))
        population.initialize()
    TablePlan(population).validate_source()
    owner = population.mech.mechanisms["table"]
    table = owner.curve_table
    assert owner.gain.device.type == "cuda"
    assert table._x_search.device.type == "cuda"
    assert table._x_search.dtype == torch.float64
    query = torch.tensor([[-72.5, -12.5]], dtype=torch.float64, device="cuda")
    torch.testing.assert_close(owner.curve(query), owner.gain * query)
    if learnable:
        derivative = torch.autograd.grad(owner.curve(query).sum(), table.y)[0]
        assert derivative.count_nonzero() == 4


def test_table_diagnostics_are_opt_in_and_frozen_for_the_plan():
    population = _population()
    _path, owner, _name, _table = table_owners(population)[0]

    def mutating(self, x):
        return x.add_(self.gain)

    owner.curve_original = MethodType(mutating, owner)
    owner.usetables(False)
    with dn.ctx(CHECK_PURITY=False):
        plan = TablePlan(population)
    assert not plan.check_purity
    with dn.ctx(CHECK_PURITY=True):
        plan.validate_source()
        with pytest.raises(dn.func.FunctionalizationError, match="in-place"):
            TablePlan(population)
