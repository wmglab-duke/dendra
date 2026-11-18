"""Functional slot templates preserve declared aliases and copy on invocation."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import dendra as dn
from dendra.func._population import (
    _capture_slot_templates,
    _clone_slot_templates,
    _deepcopy_execution_module,
    _module_tensor_slots,
)

pytestmark = pytest.mark.cpu


def test_preparation_copies_preserve_exact_aliases_and_isolate_each_invocation():
    values = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    view = values.t()
    templates, aliases = _capture_slot_templates(
        (("weight", values), ("cache", values), ("view", view)),
        share_owned_storage=True,
    )

    assert templates["weight"] is templates["cache"]
    assert torch._C._is_alias_of(templates["weight"], values)
    assert not templates["weight"].requires_grad
    first = _clone_slot_templates(templates, aliases)
    second = _clone_slot_templates(templates, aliases)

    assert first["weight"] is first["cache"]
    assert not torch._C._is_alias_of(first["weight"], values)
    assert not torch._C._is_alias_of(first["weight"], first["view"])
    torch.testing.assert_close(first["view"], view)
    first["cache"].add_(10)
    torch.testing.assert_close(first["weight"], values + 10)
    torch.testing.assert_close(second["weight"], values)
    torch.testing.assert_close(templates["weight"], second["weight"])


def test_transition_slot_capture_keeps_writable_fallback_storage_independent():
    value = torch.arange(3.0)
    templates, aliases = _capture_slot_templates(
        (("first", value), ("writable", value), ("second", value)),
        mutable_slots={"writable"},
    )

    assert templates["first"] is templates["second"]
    assert not torch._C._is_alias_of(templates["first"], value)
    copies = _clone_slot_templates(templates, aliases)
    assert copies["first"] is copies["second"]
    copies["writable"].zero_()
    torch.testing.assert_close(copies["first"], value)


def test_fallback_sharing_requires_exact_supported_values():
    from dendra.func._population import _share_read_only_fallback_templates

    class AuthoredBuffer(torch.Tensor):
        pass

    def authored_detach():
        raise AssertionError("An authored Tensor method ran during storage sharing")

    values = {
        "nan": torch.full((2,), float("nan")),
        "signed_zero": torch.tensor([-0.0]),
        "strided": torch.arange(6.0).reshape(2, 3).t(),
        "custom": torch.arange(3.0).as_subclass(AuthoredBuffer),
        "dtype": torch.arange(3.0, dtype=torch.float32),
        "view": torch.arange(3.0),
        "shadowed": torch.arange(3.0),
    }
    preparation = {name: value.clone() for name, value in values.items()}
    preparation["signed_zero"] = torch.tensor([0.0])
    preparation["dtype"] = values["dtype"].double()
    preparation["view"] = torch.arange(6.0)[:3]
    preparation["shadowed"].detach = authored_detach
    aliases = {name: name for name in values}
    originals = dict(values)

    _share_read_only_fallback_templates(values, aliases, preparation, set())

    assert torch._C._is_alias_of(values["nan"], preparation["nan"])
    for name in ("signed_zero", "strided", "custom", "dtype", "view", "shadowed"):
        assert values[name] is originals[name]
        assert not torch._C._is_alias_of(values[name], preparation[name])
    assert torch.signbit(values["signed_zero"]).all()


def test_fallback_sharing_preserves_read_only_alias_groups():
    from dendra.func._population import _share_read_only_fallback_templates

    values = torch.arange(3.0)
    shared = values.clone()
    writable = values.clone()
    separate = values.clone()
    split = values.clone()
    preparation = {
        "first": shared,
        "first_alias": shared,
        "second": writable,
        "second_alias": writable,
        "write_target": writable,
        "storage_one": separate,
        "storage_two": separate.view_as(separate),
        "split_one": split,
        "split_two": split.clone(),
    }
    bases, base_aliases = _capture_slot_templates(
        (
            ("first", shared),
            ("first_alias", shared),
            ("second", writable),
            ("second_alias", writable),
            ("storage_one", separate),
            ("storage_two", preparation["storage_two"]),
            ("split_one", split),
            ("split_two", split),
        )
    )
    originals = dict(bases)
    original_aliases = dict(base_aliases)
    _share_read_only_fallback_templates(
        bases, base_aliases, preparation, {"write_target"}
    )

    assert base_aliases == original_aliases
    assert bases["first"] is bases["first_alias"]
    assert torch._C._is_alias_of(bases["first"], shared)
    assert bases["second"] is bases["second_alias"]
    assert bases["second"] is originals["second"]
    assert not torch._C._is_alias_of(bases["second"], writable)
    assert not torch._C._is_alias_of(bases["storage_one"], bases["storage_two"])
    assert bases["split_one"] is bases["split_two"] is originals["split_one"]


def test_private_topology_sharing_preserves_aliases_and_writable_slots():
    from dendra.func._population import _share_private_topology_storage

    def facade():
        module = torch.nn.Module()
        module.population = torch.nn.Module()
        module.population.integrator = torch.nn.Module()
        buffers = module.population.integrator
        for name in ("parents", "read_write", "cross_role", "split"):
            buffers.register_buffer(name, torch.arange(4, dtype=torch.int64))
            buffers.register_buffer(f"{name}_alias", buffers._buffers[name])
        buffers.register_buffer("different", torch.arange(4, dtype=torch.int64))
        buffers.register_buffer("cached", torch.arange(4, dtype=torch.int64))
        buffers.register_buffer("external", torch.arange(4, dtype=torch.int64))
        buffers.register_buffer("viewed", torch.arange(4, dtype=torch.int64))
        buffers.register_buffer("viewed_alias", buffers.viewed.view_as(buffers.viewed))
        buffers._cpu_plan_sources = (buffers.cached,)
        return module

    transition, initializer, native = facade(), facade(), facade()
    transition_slots = dict(_module_tensor_slots(transition))
    native.population.integrator.external = transition.population.integrator.external
    init_buffers = initializer.population.integrator._buffers
    init_buffers["split_alias"] = init_buffers["split"].clone()
    init_buffers["different"] = init_buffers["different"] + 1
    original = dict(init_buffers)
    prefix = "population.integrator."
    topology = {f"{prefix}{name}" for name in original if name != "cross_role_alias"}

    _share_private_topology_storage(
        transition, initializer, topology, {f"{prefix}read_write_alias"}, native=native
    )

    parents = transition_slots[f"{prefix}parents"]
    assert init_buffers["parents"] is init_buffers["parents_alias"]
    assert init_buffers["parents"] is not parents
    assert torch._C._is_alias_of(init_buffers["parents"], parents)
    assert all(
        init_buffers[name] is old
        for name, old in original.items()
        if name not in ("parents", "parents_alias")
    )
    assert not torch._C._is_alias_of(init_buffers["split"], init_buffers["split_alias"])
    assert init_buffers["viewed"] is not init_buffers["viewed_alias"]
    assert torch._C._is_alias_of(init_buffers["viewed"], init_buffers["viewed_alias"])
    assert dict(_module_tensor_slots(transition)).keys() == transition_slots.keys()
    for name, value in _module_tensor_slots(transition):
        assert value is transition_slots[name]


def test_execution_copy_supports_tensor_subclasses_with_colliding_hashes():
    class CollidingBuffer(torch.Tensor):
        def __hash__(self):
            return 42

    module = torch.nn.Module()
    module.register_buffer("left", torch.arange(2.0).as_subclass(CollidingBuffer))
    module.register_buffer("right", torch.arange(1.0, 3.0).as_subclass(CollidingBuffer))

    clone = _deepcopy_execution_module(module)

    torch.testing.assert_close(clone.left, module.left)
    torch.testing.assert_close(clone.right, module.right)
    assert not torch._C._is_alias_of(clone.left, module.left)
    assert not torch._C._is_alias_of(clone.right, module.right)


def test_transition_templates_only_keep_fallbacks_for_the_bound_call(monkeypatch):
    from dendra.func import _population as functional
    from dendra.models.mod import pas

    clone_devices = []
    clone_inputs = []
    clones = []
    clone_execution = functional._clone_execution_population

    def trace_clone(population):
        clone_devices.append(population.v.device.type)
        clone_inputs.append(population)
        clone = clone_execution(population)
        clones.append(clone)
        return clone

    monkeypatch.setattr(functional, "_clone_execution_population", trace_clone)

    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=True):
        model = dn.SingleCompartment(
            N=2, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas.rename("leak"), g=0.001, e=-45.0)
        model.initialize()
        model.register_buffer("fallback_diam_a", model.diam)
        model.register_buffer("fallback_diam_b", model.diam)
        runner = dn.compile(
            model,
            dt=0.005,
            steps=4,
            chunk_steps=4,
            execution="scan",
            inference=True,
            initialization="fresh",
            backend="eager",
        )
        plan = runner._functional
        assert clone_devices.count("meta") == 2
        transition_clone = clone_devices.index("meta")
        assert plan._preparation.population is clone_inputs[transition_clone]
        assert plan._transition.population is clones[transition_clone]
        assert torch._C._is_alias_of(
            plan._initialization_state_template["integrator"]["v"],
            plan._preparation.population.v,
        )
        rebound = {
            "population.v",
            "population.diam",
            "population.integrator.cmdt",
            "population.integrator.mech.mechanisms.leak.g_param",
        }
        registered_slots = dict(_module_tensor_slots(plan._transition))
        assert rebound <= registered_slots.keys()
        assert rebound.isdisjoint(plan._base_mapping)
        deferred_slot = "population.v"
        for module in (plan._transition, plan._initialization):
            slots = dict(_module_tensor_slots(module))
            voltage = slots[deferred_slot]
            assert voltage.device.type == "meta"
            assert voltage.shape == model.v.shape
            assert voltage.stride() == model.v.stride()
            assert voltage.dtype == model.v.dtype
            assert slots["population.x"].device == model.v.device
            # These ordinary buffers also serve as genuine fallback slots.
            assert slots["population.diam"] is slots["population.fallback_diam_a"]
            assert slots["population.diam"].device == model.v.device

        prepared = plan._prepare_values(runner.parameters, runner.constants)
        mapping, *_ = plan._mapping(
            runner.parameters, prepared, runner.initial_state, isolate_read_only=True
        )
        assert mapping.keys() == registered_slots.keys()
        fallback = plan._base_mapping["population.x"]
        preparation = plan._preparation_base_mapping["population.x"]
        assert torch._C._is_alias_of(fallback, preparation)
        assert not torch._C._is_alias_of(fallback, model.x)
        torch.testing.assert_close(mapping["population.x"], fallback)
        assert not torch._C._is_alias_of(mapping["population.x"], fallback)
        expected_x = preparation.clone()
        mapping["population.x"].add_(1)
        torch.testing.assert_close(preparation, expected_x, rtol=0, atol=0)
        a, b = "population.fallback_diam_a", "population.fallback_diam_b"
        assert plan._base_mapping[a] is plan._base_mapping[b]
        assert mapping[a] is mapping[b]
        assert not torch._C._is_alias_of(mapping[a], plan._base_mapping[a])

        initial = dn.func.initialize(runner)
        torch.testing.assert_close(
            initial.state["integrator"]["v"], model.v, rtol=0, atol=0
        )
        state, _ = dn.func.run(runner)
        model.longrun(dt=0.005, steps=4, chunklength=4, progressbar=False)
        torch.testing.assert_close(state["integrator"]["v"], model.v, rtol=0, atol=0)


def test_gradient_runner_keeps_preparation_and_fallback_templates_independent():
    from dendra.models.mod import pas

    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=True):
        model = dn.SingleCompartment(
            N=2, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas)
        model.initialize()
        runner = dn.compile(
            model,
            dt=0.005,
            steps=4,
            initialization="fresh",
            inference=False,
            backend="eager",
            disable=True,
        )

        plan = runner.functional
        slot = "population.x"
        fallback = plan._base_mapping[slot]
        preparation = plan._preparation_base_mapping[slot]
        torch.testing.assert_close(fallback, preparation, rtol=0, atol=0)
        assert not torch._C._is_alias_of(fallback, preparation)


def test_packed_while_shares_private_topology_without_rebinding_other_owner():
    from tests.func.test_membrane_current import DT, STEPS, _inputs, _model

    with (
        dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, CHECK_PURITY=True),
        torch.no_grad(),
    ):
        model = _model("multi_block", batch=None)
        runner = dn.compile(
            model,
            dt=DT,
            steps=STEPS,
            initialization="fresh",
            execution="while",
            inference=True,
            backend="eager",
        )
        plan = runner.functional
        preparation = plan._preparation.population.integrator
        transition = plan._transition.population.integrator
        initializer = plan._initialization.population.integrator
        preparation_order = preparation.solver_order
        transition_order = transition.solver_order
        initializer_order = initializer.solver_order
        orders = (preparation_order, transition_order, initializer_order)
        assert len({id(order) for order in orders}) == len(orders)
        for order in orders:
            assert order.device == model.integrator.solver_order.device
            assert torch._C._is_alias_of(order, preparation_order)
        assert not torch._C._is_alias_of(
            transition_order, model.integrator.solver_order
        )
        assert not torch._C._is_alias_of(transition_order, initializer.order)
        if any(
            source is initializer.parent_idx for source in initializer._cpu_plan_sources
        ):
            assert not torch._C._is_alias_of(
                transition.parent_idx, initializer.parent_idx
            )
        versions = tuple(order._version for order in orders)

        with pytest.MonkeyPatch.context() as patch:
            replacement = initializer_order.clone()
            patch.setitem(initializer._buffers, "solver_order", replacement)
            assert initializer.solver_order is replacement
            assert transition.solver_order is transition_order
        assert initializer.solver_order is initializer_order

        reference, tensors = dn.func.make_functional(model, dt=DT)
        initial = dn.func.initialize(reference, tensors)
        ve, intra = _inputs(model)
        inputs = dn.func.RolloutInput(ve=ve, intra=intra)
        expected, _ = reference.prepare_and_rollout(
            initial.parameters,
            initial.constants,
            initial.state,
            inputs,
            steps=STEPS,
        )
        actual, _ = dn.func.run(runner, inputs=inputs)
        assert runner.execution_report().strategy == "compiled_while_loop"
        assert transition.solver_order is transition_order
        assert initializer.solver_order is initializer_order
        assert preparation.solver_order is preparation_order
        assert tuple(order._version for order in orders) == versions
        actual_values, actual_tree = torch.utils._pytree.tree_flatten(actual)
        expected_values, expected_tree = torch.utils._pytree.tree_flatten(expected)
        assert actual_tree == expected_tree
        for value, oracle in zip(actual_values, expected_values, strict=True):
            torch.testing.assert_close(value, oracle, rtol=0, atol=0)


def _mirror_model(*, regional=False, with_noise_workspace=False):
    from dendra.models.mechanisms import Mechanism, State

    class MirrorState(State):
        State.STATE("x")
        State.DERIVATIVE("x' = nai + amount")

        def state_defaults(self, v, values):
            del values
            return {"x": torch.zeros_like(v)}

    class MirrorReader(Mechanism):
        Mechanism.STATE_BUNDLE(MirrorState)
        Mechanism.USEION("na", read=["nai"])
        Mechanism.USEMATERIAL("pool", read=["amount"])
        Mechanism.CARRY("initial_fields")

        def initial_values(self, v, values):
            del v
            return {"initial_fields": values["nai"] + values["amount"]}

    class ReaderWithNoiseWorkspace(MirrorReader):
        Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.1, seed=31)
        Mechanism.TIMESTEP_BUFFER("coefficient")

        def derive_timestep_buffers(self, dt):
            return {"coefficient": dt * (self.eta + self.nai).mean(0, keepdim=True)}

    model = dn.SingleCompartment(
        N=2, C=3, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    ).eval()
    model.material("pool", fields={"amount": 2.0})
    reader = ReaderWithNoiseWorkspace if with_noise_workspace else MirrorReader
    target = model[:, 1:] if regional else model
    target.insert(reader.rename("readout"))
    model.initialize()
    mechanism = model.mech.readout
    for name in ("nai", "amount"):
        value = getattr(mechanism, name).clone(memory_format=torch.contiguous_format)
        model.mech._bind_mechanism_field(mechanism, name, value)
    return model


@pytest.mark.parametrize("regional", [False, True], ids=["full-support", "regional"])
def test_compiled_transition_reads_canonical_ion_and_material_carry(
    monkeypatch, regional
):
    from dendra.func._population import (
        FunctionalPopulation,
        _local_synchronization_slots,
    )

    audited = []
    audit = FunctionalPopulation._audit_read_only_workspaces

    def check_audit(plan, tensors):
        base = plan._base_mapping
        audited.extend(
            slot for slot in mirrors if slot in base and not base[slot].is_meta
        )
        return audit(plan, tensors)

    monkeypatch.setattr(
        FunctionalPopulation, "_audit_read_only_workspaces", check_audit
    )

    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=True):
        model = _mirror_model(regional=regional)
        mirrors = _local_synchronization_slots(model)
        assert len(mirrors) == 4
        readout = model.mech.readout
        native_initial_fields = readout.initial_fields.clone()
        for name in ("nai", "amount"):
            stale = torch.full_like(getattr(readout, name), -100.0)
            model.mech._bind_mechanism_field(readout, name, stale)

        dt = 0.125
        runner = dn.compile(
            model,
            dt=dt,
            steps=4,
            initialization="fresh",
            execution="scan",
            inference=True,
            backend="eager",
        )
        assert set(audited) == set(mirrors)
        plan = runner.functional
        assert set(plan._deferred_local_mirror_seeds) == set(mirrors)
        if regional:
            assert not plan._defer_execution_storage_early
        prepared = plan._prepare_values(runner.parameters, runner.constants)
        synchronized, *_ = plan._mapping(
            runner.parameters,
            prepared,
            runner.initial_state,
            synchronize_local_mirrors=True,
        )
        isolated, *_ = plan._mapping(
            runner.parameters,
            prepared,
            runner.initial_state,
            isolate_read_only=True,
            synchronize_local_mirrors=True,
        )
        registered = dict(_module_tensor_slots(plan._transition))
        for slot in mirrors:
            seed = plan._deferred_local_mirror_seeds[slot]
            assert plan._base_mapping[slot].is_meta
            assert synchronized[slot].is_meta
            assert not isolated[slot].is_meta
            assert not torch._C._is_alias_of(isolated[slot], seed)
            torch.testing.assert_close(isolated[slot], seed, rtol=0, atol=0)
            if regional:
                assert not torch._C._is_alias_of(seed, registered[slot])
        for name in ("nai", "amount"):
            owners = [slot for slot in mirrors if slot.endswith(f".{name}")]
            first, second = owners
            assert torch._C._is_alias_of(
                plan._deferred_local_mirror_seeds[first],
                plan._deferred_local_mirror_seeds[second],
            )
            assert not torch._C._is_alias_of(isolated[first], isolated[second])
        initial = dn.func.initialize(runner)
        torch.testing.assert_close(
            initial.state["mechanism_buffers"]["readout"]["initial_fields"],
            native_initial_fields,
            rtol=0,
            atol=0,
        )

        state = runner.initial_state
        nai = torch.arange(1.0, 7.0, dtype=model.dtype()).reshape(model.shape)
        amount = torch.arange(11.0, 17.0, dtype=model.dtype()).reshape(model.shape)
        state["ions"]["na"]["nai"] = nai
        state["materials"]["pool"]["amount"] = amount
        result, _ = dn.func.run(runner, state=state)
        local_sum = nai + amount
        expected = 4 * dt * (local_sum[:, 1:] if regional else local_sum)
        torch.testing.assert_close(
            result["mechanisms"]["readout"]["x"], expected, rtol=0, atol=0
        )


def test_authored_shared_field_rebinder_keeps_concrete_mirror_slots(monkeypatch):
    from dendra.func._population import _synchronized_local_mirror_slots
    from dendra.models.mechanisms._handler import MechanismHandler

    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=True):
        model = _mirror_model()
        mirrors = _synchronized_local_mirror_slots(model)
        assert mirrors
        original = MechanismHandler._bind_mechanism_field

        def bind_using_previous_value(owner, field, value):
            original(owner, field, value + 0.0 * owner._buffers[field])

        monkeypatch.setattr(
            MechanismHandler,
            "_bind_mechanism_field",
            staticmethod(bind_using_previous_value),
        )
        runner = dn.compile(model, dt=0.125, steps=4, backend="eager")
        plan = runner.functional
        for slot in mirrors:
            assert slot not in plan._deferred_local_mirror_seeds
            assert not plan._base_mapping[slot].is_meta
        result, _ = dn.func.run(runner)
        assert torch.isfinite(result["mechanisms"]["readout"]["x"]).all()


def test_authored_state_attribute_access_keeps_concrete_mirrors():
    from dendra.func._population import _local_synchronization_slots
    from dendra.models.mechanisms import Mechanism, State

    class InspectingState(State):
        State.STATE("x")
        State.DERIVATIVE("x' = nai")

        def state_defaults(self, v, values):
            del values
            return {"x": torch.zeros_like(v)}

        def __getattribute__(self, name):
            if name == "_buffers":
                buffers = object.__getattribute__(self, name)
                previous = buffers.get("nai")
                if isinstance(previous, torch.Tensor) and previous.is_meta:
                    raise RuntimeError("authored State accessed the previous mirror")
                return buffers
            return super().__getattribute__(name)

    class Reader(Mechanism):
        Mechanism.USEION("na", read=["nai"])
        Mechanism.STATE_BUNDLE(InspectingState)

    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=True):
        model = dn.SingleCompartment(
            N=1, C=2, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(Reader.rename("readout"))
        model.initialize()
        reader = model.mech.readout
        model.mech._bind_mechanism_field(reader, "nai", reader.nai.clone())
        dt = 0.125
        runner = dn.compile(
            model, dt=dt, steps=4, initialization="fresh", backend="eager"
        )

        result, _ = dn.func.run(runner)
        expected = 4 * dt * model.mech.ions["na"].nai
        torch.testing.assert_close(
            result["mechanisms"]["readout"]["x"], expected, rtol=0, atol=0
        )
        plan = runner.functional
        mirrors = _local_synchronization_slots(model)
        assert len(mirrors) == 2
        assert not plan._deferred_local_mirror_seeds
        assert all(not plan._base_mapping[slot].is_meta for slot in mirrors)


def test_noise_workspace_configuration_can_read_real_mirror_values():
    from dendra.func._population import _local_synchronization_slots

    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=True):
        model = _mirror_model(with_noise_workspace=True)
        dt = 0.125
        runner = dn.compile(
            model,
            dt=dt,
            steps=4,
            initialization="fresh",
            backend="eager",
        )
        plan = runner.functional
        assert plan._noise_workspace_configuration.layout
        mirrors = _local_synchronization_slots(model)
        assert set(plan._deferred_local_mirror_seeds) == set(mirrors)
        assert all(plan._base_mapping[slot].is_meta for slot in mirrors)
        reference, _ = dn.func.make_functional(model, dt=dt)
        initial = dn.func.initialize(runner)
        constants = {**initial.constants, "dt": model.t.new_tensor(2 * dt)}

        result = plan._configure_noise_workspaces(
            initial.parameters, constants, initial.state
        )
        expected = reference._configure_noise_workspaces(
            initial.parameters, constants, initial.state
        )
        actual_leaves = torch.utils._pytree.tree_leaves(result["noise_workspaces"])
        expected_leaves = torch.utils._pytree.tree_leaves(expected["noise_workspaces"])
        assert actual_leaves
        for actual, oracle in zip(actual_leaves, expected_leaves, strict=True):
            assert not actual.is_meta
            torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


def test_fresh_inference_defers_editable_restart_state():
    from dendra.models.mod import pas

    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=True):
        model = dn.SingleCompartment(
            N=2, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas.rename("leak"), g=0.001, e=-45.0)
        model.initialize()
        model.t.neg_()
        model.v.add_(2)
        runner = dn.compile(
            model,
            dt=0.005,
            steps=4,
            initialization="fresh",
            backend="eager",
            callbacks={"trace": dn.func.Recorder("v")},
        )
        plan = runner.functional
        private = plan._initialization_state_template
        backing = runner._initial_state
        assert runner._restart_state_deferred
        assert backing["integrator"]["v"] is private["integrator"]["v"]
        assert backing["clock"]["t"] is not private["clock"]["t"]
        assert not torch._C._is_alias_of(backing["integrator"]["v"], model.v)

        model.t.fill_(9)
        model.v.fill_(21)
        initial = dn.func.initialize(runner)
        state, _ = dn.func.run(runner)
        assert runner._initial_state is backing
        assert runner._restart_state_deferred

        with torch.inference_mode():
            restart = runner.initial_state
        assert restart is runner.initial_state
        assert not runner._restart_state_deferred
        assert torch.signbit(restart["clock"]["t"])
        voltage = restart["integrator"]["v"]
        assert not torch.is_inference(voltage)
        assert not torch._C._is_alias_of(voltage, private["integrator"]["v"])
        torch.testing.assert_close(voltage, torch.full_like(voltage, -58))
        voltage.add_(5)
        _, aux = dn.func.run(runner, state=restart)
        torch.testing.assert_close(aux["callbacks"]["trace"]["v"][0], voltage)
        torch.testing.assert_close(
            dn.func.initialize(runner).state["integrator"]["v"],
            initial.state["integrator"]["v"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            dn.func.run(runner)[0]["integrator"]["v"],
            state["integrator"]["v"],
            rtol=0,
            atol=0,
        )


def test_private_conductance_scratch_shares_backing_and_rebinds_locally():
    from dendra.models.mod import pas
    from tests.func.test_regional_shared_fields import _RegionalSodiumCurrent

    with (
        dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, CHECK_PURITY=True),
        torch.no_grad(),
    ):
        model = dn.SingleCompartment(
            N=2, C=3, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        ).eval()
        model.insert(pas)
        model.insert(_RegionalSodiumCurrent)
        model.initialize()
        dt = 0.125
        runner = dn.compile(
            model, dt=dt, steps=4, initialization="fresh", backend="eager"
        )
        plan = runner.functional
        transition = plan._transition.population.mech
        preparation = plan._preparation.population.mech
        initializer = plan._initialization.population.mech
        saved = [value.clone() for value in transition._buf_g]
        assert len(saved) > 1
        for source, first, second, native in zip(
            transition._buf_g,
            preparation._buf_g,
            initializer._buf_g,
            model.mech._buf_g,
            strict=True,
        ):
            assert source is not first and source is not second
            assert torch._C._is_alias_of(source, first)
            assert torch._C._is_alias_of(source, second)
            assert not torch._C._is_alias_of(source, native)
        assert preparation._buf_g is not transition._buf_g

        reference, tensors = dn.func.make_functional(model, dt=dt)
        initial = dn.func.initialize(reference, tensors)
        expected, _ = reference.prepare_and_rollout(
            initial.parameters,
            initial.constants,
            initial.state,
            steps=4,
        )
        actual, _ = dn.func.run(runner)
        torch.testing.assert_close(
            actual["integrator"]["v"], expected["integrator"]["v"], rtol=0, atol=0
        )
        native_frame = model.mech.capture_ion_conductance_frame()
        assert len(native_frame) == 1 and torch.count_nonzero(native_frame[0])
        native_values = tuple(value.clone() for value in native_frame)
        model.mech.i(model.v + 1)
        preparation.i(model.v + 1)

        for previous, actual in zip(native_frame, native_values, strict=True):
            torch.testing.assert_close(previous, actual, rtol=0, atol=0)
        for seed, new, expected in zip(
            transition._buf_g, preparation._buf_g, saved, strict=True
        ):
            assert not torch._C._is_alias_of(seed, new)
            torch.testing.assert_close(seed, expected, rtol=0, atol=0)


def test_authored_conductance_scratch_reader_retains_independent_values():
    from dendra.models.mechanisms import Mechanism

    class ScratchReader(Mechanism):
        Mechanism.NONSPECIFIC_CURRENT("i")
        Mechanism.AFFINE("i")
        Mechanism.GLOBAL(g=0.002, e=-50.0)
        Mechanism.CARRY("seen")

        def i(self, v):
            return self.g * (v - self.e)

        def i_with_conductance(self, v):
            return self.i(v), self.g.expand_as(v)

        def initial_values(self, v, values):
            return {"seen": torch.zeros_like(v)}

        def advance(self, v, dt, values):
            return {"seen": self.scratch_owner._buf_g[0]}

    with (
        dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, CHECK_PURITY=True),
        torch.no_grad(),
    ):
        model = dn.SingleCompartment(
            N=1, C=3, v_init=-62.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(ScratchReader.rename("reader"))
        model.initialize()
        object.__setattr__(model.mech.reader, "scratch_owner", model.mech)
        runner = dn.compile(model, dt=0.125, steps=4, backend="eager")
        plan = runner.functional
        transition = plan._transition.population.mech._buf_g[0]
        preparation = plan._preparation.population.mech._buf_g[0]
        assert not torch._C._is_alias_of(transition, preparation)

        state, _ = dn.func.run(runner)
        observed = state["mechanism_buffers"]["reader"]["seen"]
        torch.testing.assert_close(observed, transition, rtol=0, atol=0)
        assert not torch._C._is_alias_of(observed, transition)


@pytest.fixture
def _ion_storage_cpu():
    from dendra.helpers import DEVICE

    # Reuse an active CPU default to avoid stacking redundant DeviceContexts.
    defaults_are_cpu = torch.get_default_device().type == "cpu" and DEVICE.value in (
        "",
        "cpu",
        torch.device("cpu"),
    )
    device = {} if defaults_are_cpu else {"DEVICE": "cpu"}
    with dn.ctx(DTYPE="float64", JIT=0, CHECK_PURITY=True, **device), torch.no_grad():
        yield


@pytest.mark.parametrize(
    "capture_alias", [False, True], ids=["ordinary", "authored-reader"]
)
def test_private_ion_mirrors_share_unexposed_carry_groups(capture_alias):
    _run_ownership_case_in_fresh_process("_run_private_ion_mirror_case", capture_alias)


@pytest.mark.parametrize("authored_setstate", [False, True], ids=["stock", "authored"])
def test_dynamo_generation_tagging_preserves_copy_hook_eligibility(authored_setstate):
    _run_ownership_case_in_fresh_process(
        "_run_dynamo_generation_tagging_case", authored_setstate
    )


def _run_dynamo_generation_tagging_case(authored_setstate):
    from torch._dynamo.mutation_guard import install_generation_tagging_init

    from dendra.func._population import _supports_early_storage_deferral

    module = torch.nn.Module()
    assert _supports_early_storage_deferral(module)
    if authored_setstate:
        original = torch.nn.Module.__setstate__

        def restore(self, state):
            original(self, state)

        torch.nn.Module.__setstate__ = restore

    install_generation_tagging_init()

    assert _supports_early_storage_deferral(module) is (not authored_setstate)

    if not authored_setstate:
        with pytest.MonkeyPatch.context() as monkeypatch:
            test_transition_templates_only_keep_fallbacks_for_the_bound_call(
                monkeypatch
            )


def _run_ownership_case_in_fresh_process(case, option):
    # Dynamo installs a process-wide Module copy hook on its first compile.
    # A fresh process exercises the ordinary copy path consistently.
    script = (
        "import ast, runpy, sys; "
        "runpy.run_path(sys.argv[1])[sys.argv[2]](ast.literal_eval(sys.argv[3]))"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, __file__, case, repr(option)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=True)
@torch.no_grad()
def _run_private_ion_mirror_case(capture_alias):
    from dendra.func._population import _local_synchronization_slots
    from dendra.models.mechanisms import Mechanism, State
    from tests.func.test_regional_shared_fields import _RegionalSodiumCurrent

    class Accumulator(State):
        State.STATE("charge")
        State.DERIVATIVE("charge' = nai + ina")

        def state_defaults(self, v, values):
            return {"charge": torch.zeros_like(v)}

    class Reader(Mechanism):
        Mechanism.STATE_BUNDLE(Accumulator)
        Mechanism.USEION("na", read=["nai", "ina"])
        Mechanism.CARRY("initial_sodium")

        def initial_values(self, v, values):
            return {"initial_sodium": values["nai"].clone()}

    class CapturedReader(Reader):
        Mechanism.CARRY("seen")

        def initial_values(self, v, values):
            return {
                "initial_sodium": values["nai"].clone(),
                "seen": torch.zeros_like(v),
            }

        def advance(self, v, dt, values):
            return {"seen": self.setup_seed.clone()}

    def build():
        model = dn.SingleCompartment(
            N=2, C=3, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        ).eval()
        model.insert(_RegionalSodiumCurrent)
        reader = CapturedReader if capture_alias else Reader
        model.insert(reader.rename("readout"))
        model.initialize()
        if capture_alias:
            object.__setattr__(
                model.mech.readout, "setup_seed", model.mech.ions["na"].nai
            )
        return model

    model = build()
    reference = build()
    dt = 0.125
    runner = dn.compile(
        model,
        dt=dt,
        steps=4,
        initialization="fresh",
        execution="scan",
        inference=True,
        backend="eager",
    )
    plan = runner.functional
    assert plan._defer_execution_storage_early
    owners = (
        model.mech,
        plan._preparation.population.mech,
        plan._transition.population.mech,
        plan._initialization.population.mech,
    )

    def assert_group(field):
        seeds = []
        for handler in owners:
            seed = getattr(handler.ions["na"], field)
            reader = handler.readout
            state = next(iter(reader.DE.values()))
            assert seed.device == model.v.device
            assert getattr(reader, field) is getattr(state, field) is seed
            if field == "ina":
                assert any(value is seed for value in handler._buf_i)
            if capture_alias and field == "nai":
                assert reader.setup_seed is seed
            seeds.append(seed)
        assert len({id(seed) for seed in seeds}) == len(seeds)
        native, preparation, transition, initialization = seeds
        for private in (preparation, transition, initialization):
            assert not torch._C._is_alias_of(native, private)
        for first, second in (
            (preparation, transition),
            (preparation, initialization),
            (transition, initialization),
        ):
            share = field == "nai" and not capture_alias
            assert torch._C._is_alias_of(first, second) == share
        return seeds

    native_nai, *private_nai = assert_group("nai")
    assert_group("ina")
    versions = tuple(seed._version for seed in private_nai)
    mirror_slots = {
        slot for slot in _local_synchronization_slots(model) if slot.endswith(".nai")
    }
    assert len(mirror_slots) == 2
    for slot in mirror_slots:
        saved_seed = plan._deferred_local_mirror_seeds[slot]
        assert torch._C._is_alias_of(saved_seed, private_nai[0]) == (not capture_alias)
    captured_nai = native_nai.clone()
    native_nai.add_(3)

    fresh = dn.func.initialize(runner).state
    torch.testing.assert_close(fresh["ions"]["na"]["nai"], captured_nai, rtol=0, atol=0)
    torch.testing.assert_close(
        fresh["mechanism_buffers"]["readout"]["initial_sodium"],
        captured_nai,
        rtol=0,
        atol=0,
    )
    actual, _ = dn.func.run(runner)
    reference.longrun(dt=dt, steps=4, chunklength=4, progressbar=False)
    assert runner.execution_report().strategy == "compiled_scan"
    torch.testing.assert_close(actual["integrator"]["v"], reference.v, rtol=0, atol=0)
    charge = actual["mechanisms"]["readout"]["charge"]
    torch.testing.assert_close(charge, reference.mech.readout.charge, rtol=0, atol=0)

    changed_nai = fresh["ions"]["na"]["nai"]
    for seed in private_nai:
        assert not torch._C._is_alias_of(changed_nai, seed)
    changed_nai.add_(2)
    changed, _ = dn.func.run(runner, state=fresh)
    torch.testing.assert_close(
        changed["mechanisms"]["readout"]["charge"], charge + 4 * dt * 2
    )
    if capture_alias:
        torch.testing.assert_close(
            changed["mechanism_buffers"]["readout"]["seen"],
            captured_nai,
            rtol=0,
            atol=0,
        )
    assert tuple(seed._version for seed in private_nai) == versions
    for seed, saved in zip(assert_group("nai")[1:], private_nai, strict=True):
        assert seed is saved
        torch.testing.assert_close(seed, captured_nai, rtol=0, atol=0)


@pytest.mark.parametrize("reader", [None, "root-transform", "noise-descriptor"])
def test_preparation_rebuilt_storage_preserves_early_reads(reader):
    _run_ownership_case_in_fresh_process(
        "_run_preparation_geometry_storage_case", reader
    )


@dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=True)
@torch.no_grad()
def _run_preparation_geometry_storage_case(reader):
    from dendra.models.mod import pas

    class NoiseReader(pas):
        @property
        def runtime_noises(self):
            diameter = self._buffers.get("diam")
            if diameter is not None and diameter.is_meta:
                raise RuntimeError("noise reset read metadata-only diameter")
            return self.__dict__.get("_stored_runtime_noises", {})

        @runtime_noises.setter
        def runtime_noises(self, value):
            self.__dict__["_stored_runtime_noises"] = value

    class EarlyReader(torch.nn.Module):
        def __init__(self, owner):
            super().__init__()
            object.__setattr__(self, "owner", owner)

        def forward(self, cm):
            setup_values = self.owner.integrator.diag_base + self.owner.mech.leak.diam
            return torch.where(torch.isfinite(setup_values), cm, torch.nan)

    model = dn.Unmyelinated(
        [2.0, 3.0],
        L=40.0,
        dx=10.0,
        v_init=-63.0,
        integrator=dn.bwd_euler_ub(method="thomas", imem=False),
    ).eval()
    mechanism = NoiseReader if reader == "noise-descriptor" else pas
    model.insert(mechanism.rename("leak"))
    model.initialize()
    if reader == "root-transform":
        model.register_parametrization_in_graph("cm", EarlyReader(model))

    dt = 0.125
    runner = dn.compile(
        model,
        dt=dt,
        steps=4,
        initialization="fresh",
        execution="scan",
        inference=True,
        backend="eager",
    )
    plan = runner.functional
    if reader == "root-transform":
        population = plan._preparation.population
        transform, _ = population.in_graph_parametrizations["cm"][0]
        assert transform.owner is population

    preparation_slots = dict(_module_tensor_slots(plan._preparation))
    native_slots = {
        f"population.{name}": value for name, value in _module_tensor_slots(model)
    }
    rebuilt = (
        "population.diam",
        "population.dx",
        "population.integrator.mech.mechanisms.leak.diam",
        "population.integrator.diag_base",
        "population.integrator.cm_inv",
        "population.integrator.scale",
    )
    for name in rebuilt:
        slot, template = preparation_slots[name], plan._preparation_base_mapping[name]
        original = native_slots[name]
        assert slot.is_meta == template.is_meta == (reader is None)
        assert slot.shape == template.shape == original.shape
        assert slot.dtype == template.dtype == original.dtype
        assert slot.stride() == template.stride() == original.stride()
        assert not original.is_meta
        if reader is not None:
            torch.testing.assert_close(slot, template, rtol=0, atol=0)
        else:
            assert slot is template
    assert not preparation_slots["population.integrator.g_ax"].is_meta

    geometry = {name: runner.constants[name] + 1 for name in ("diam", "dx")}
    constants = {**runner.constants, **geometry}
    root_values, _ = plan._prepare_initialization_root(runner.parameters, constants)
    torch.testing.assert_close(root_values["cm"], model.cm, rtol=0, atol=0)
    prepared_geometry = plan.prepare(runner.parameters, constants).values["geometry"]
    for name, value in geometry.items():
        torch.testing.assert_close(prepared_geometry[name], value, rtol=0, atol=0)
    initial = dn.func.initialize(runner)
    torch.testing.assert_close(
        initial.state["integrator"]["v"], model.v, rtol=0, atol=0
    )

    first = plan.prepare(initial.parameters, initial.constants)
    workspace = first.values["integrator"]["diag_base"]
    previous = workspace.clone()
    workspace.add_(5.0)
    second = plan.prepare(initial.parameters, initial.constants)
    torch.testing.assert_close(
        second.values["integrator"]["diag_base"], previous, rtol=0, atol=0
    )
    eager, _ = plan.rollout(initial.parameters, second, initial.state, steps=4)
    compiled, _ = dn.func.run(runner)
    model.longrun(dt=dt, steps=4, chunklength=4, progressbar=False)
    assert runner.execution_report().strategy == "compiled_scan"
    for result in (eager, compiled):
        torch.testing.assert_close(result["integrator"]["v"], model.v, rtol=0, atol=0)
    restored = dict(_module_tensor_slots(plan._preparation))
    for name in rebuilt:
        assert restored[name] is preparation_slots[name]


def test_private_ion_initial_storage_shares_only_concrete_owned_seeds(_ion_storage_cpu):
    from tests.func.test_regional_shared_fields import _RegionalSodiumCurrent

    def build():
        model = dn.SingleCompartment(
            N=2, C=3, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        ).eval()
        model.insert(_RegionalSodiumCurrent)
        model.initialize()
        return model

    model = build()
    reference = build()
    native = model.mech.ions["na"]
    initial_nai = native._initial_nai
    captured_nai = initial_nai + torch.arange(
        initial_nai.numel(), dtype=initial_nai.dtype, device=initial_nai.device
    ).reshape(initial_nai.shape)
    initial_nai.copy_(captured_nai)

    dt = 0.125
    runner = dn.compile(
        model,
        dt=dt,
        steps=4,
        initialization="fresh",
        execution="scan",
        inference=True,
        backend="eager",
    )
    plan = runner.functional
    owners = tuple(
        facade.population.mech.ions["na"]
        for facade in (plan._preparation, plan._transition, plan._initialization)
    )
    field_seeds = {}
    for field in native.fields:
        name = f"_initial_{field}"
        original = native._buffers[name]
        seeds = tuple(owner._buffers[name] for owner in owners)
        assert len({id(seed) for seed in seeds}) == len(seeds)
        for seed in seeds:
            assert seed.device == original.device and seed.stride() == original.stride()
            assert torch._C._is_alias_of(seed, seeds[0])
            assert not torch._C._is_alias_of(seed, original)
            torch.testing.assert_close(seed, original, rtol=0, atol=0)
        field_seeds[name] = seeds

    initial_nai.add_(1)
    fresh = dn.func.initialize(runner).state
    torch.testing.assert_close(
        fresh["ions"]["na"]["nai"], reference.mech.ions["na"].nai
    )
    actual, _ = dn.func.run(runner)
    reference.longrun(dt=dt, steps=4, chunklength=4, progressbar=False)
    assert runner.execution_report().strategy == "compiled_scan"
    torch.testing.assert_close(actual["integrator"]["v"], reference.v, rtol=0, atol=0)
    for owner, seed in zip(owners, field_seeds["_initial_nai"], strict=True):
        assert owner._initial_nai is seed
        torch.testing.assert_close(seed, captured_nai, rtol=0, atol=0)


def test_authored_reader_keeps_its_private_ion_seed_alias(_ion_storage_cpu):
    from dendra.models.mechanisms import Mechanism
    from tests.func.test_regional_shared_fields import _RegionalSodiumCurrent

    class SeedReader(Mechanism):
        Mechanism.CARRY("seen")

        def initial_values(self, v, values):
            return {"seen": torch.zeros_like(v)}

        def advance(self, v, dt, values):
            return {"seen": self.setup_seed.clone()}

    model = dn.SingleCompartment(
        N=2, C=3, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    ).eval()
    model.insert(_RegionalSodiumCurrent)
    model.insert(SeedReader.rename("reader"))
    model.initialize()
    native = model.mech.ions["na"]._initial_nai
    native.add_(2)
    captured = native.clone()
    object.__setattr__(model.mech.reader, "setup_seed", native)
    runner = dn.compile(
        model,
        dt=0.125,
        steps=4,
        initialization="fresh",
        execution="scan",
        inference=True,
        backend="eager",
    )
    plan = runner.functional
    handlers = tuple(
        facade.population.mech
        for facade in (plan._preparation, plan._transition, plan._initialization)
    )
    sources = [native]
    for handler in handlers:
        seed = handler.ions["na"]._initial_nai
        assert handler.reader.setup_seed is seed
        assert all(not torch._C._is_alias_of(seed, previous) for previous in sources)
        sources.append(seed)

    native.add_(3)
    actual, _ = dn.func.run(runner)
    assert runner.execution_report().strategy == "compiled_scan"
    observed = actual["mechanism_buffers"]["reader"]["seen"]
    torch.testing.assert_close(observed, captured, rtol=0, atol=0)


def test_authored_ion_seed_output_cannot_alias_private_physical_bindings(
    _ion_storage_cpu,
):
    from dendra.models.mechanisms import Mechanism

    class SeedReader(Mechanism):
        Mechanism.CARRY("seen")
        Mechanism.USEION("na", read=["nai"])

        def initial_values(self, v, values):
            return {"seen": torch.zeros_like(v)}

        def advance(self, v, dt, values):
            return {"seen": self.ion_handler.ions.na._initial_nai}

    model = dn.SingleCompartment(
        N=1, C=3, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    ).eval()
    model.insert(SeedReader.rename("reader"))
    model.initialize()
    object.__setattr__(model.mech.reader, "ion_handler", model.mech)
    runner = dn.compile(
        model,
        dt=0.125,
        steps=4,
        initialization="fresh",
        execution="scan",
        inference=True,
        backend="eager",
    )
    plan = runner.functional
    prepared = plan.prepare(runner.parameters, runner.constants)
    result, _ = plan.rollout(
        runner.parameters,
        prepared,
        runner.initial_state,
        steps=4,
    )
    output = result["mechanism_buffers"]["reader"]["seen"]
    owned = tuple(
        facade.population.mech.ions.na._initial_nai
        for facade in (plan._transition, plan._initialization)
    )
    seed = model.mech.ions.na._initial_nai.clone()
    torch.testing.assert_close(output, seed, atol=0, rtol=0)
    for value in owned:
        assert not torch._C._is_alias_of(output, value)
    output.add_(1)
    for value in owned:
        torch.testing.assert_close(value, seed, atol=0, rtol=0)


def test_first_restart_snapshot_access_in_full_graph_tracks_public_edits():
    from dendra.models.mod import pas

    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False):
        model = dn.SingleCompartment(
            N=2, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas)
        model.initialize()
        runner = dn.compile(
            model, dt=0.005, steps=4, initialization="fresh", backend="eager"
        )

        @torch.compile(fullgraph=True, backend="eager")
        def shifted_voltage(offset):
            return runner.initial_state["integrator"]["v"] + offset

        offset = torch.tensor(1.0)
        assert runner._restart_state_deferred
        torch.testing.assert_close(shifted_voltage(offset), model.v + offset)
        assert not runner._restart_state_deferred
        integrator = runner.initial_state["integrator"]
        integrator["v"].add_(7)
        torch.testing.assert_close(shifted_voltage(offset), model.v + 7 + offset)
        integrator["v"] = torch.full_like(model.v, 25)
        torch.testing.assert_close(shifted_voltage(offset), integrator["v"] + offset)


@pytest.mark.parametrize("hook_location", ["registered_module", "python_cache"])
def test_compiled_inference_copy_hooks_receive_materialized_values(hook_location):
    from dendra.models.mod import pas

    class LeakWithRestoreHook(pas):
        def __setstate__(self, state):
            pas.__setstate__(self, state)
            self.e.reshape(-1)[0].item()

    class CacheWithCopyHook:
        def __init__(self, owner):
            self.owner = owner

        def __deepcopy__(self, memo):
            self.owner.e.reshape(-1)[0].item()
            result = object.__new__(type(self))
            memo[id(self)] = result
            result.owner = copy.deepcopy(self.owner, memo)
            return result

    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=True):
        model = dn.SingleCompartment(
            N=2, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        leak = LeakWithRestoreHook if hook_location == "registered_module" else pas
        model.insert(leak.rename("leak"), g=0.001, e=-45.0)
        model.initialize()
        if hook_location == "python_cache":
            model.mech.leak.cache_with_copy_hook = CacheWithCopyHook(model.mech.leak)
        runner = dn.compile(
            model,
            dt=0.005,
            steps=4,
            chunk_steps=4,
            execution="scan",
            inference=True,
            initialization="fresh",
            backend="eager",
        )
        plan = runner._functional
        assert plan._transition.population.v.device.type == "meta"
        assert plan._initialization.population.v.device.type == "meta"
        state, _ = dn.func.run(runner)
        model.longrun(dt=0.005, steps=4, chunklength=4, progressbar=False)
        torch.testing.assert_close(state["integrator"]["v"], model.v, rtol=0, atol=0)


def test_constructor_preparation_runs_on_its_own_parametrization():
    from dendra.models.mod import pas

    class CountingIdentity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return value

    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, IMEM=False, CHECK_PURITY=False):
        model = dn.SingleCompartment(
            N=1, C=2, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas)
        model.initialize()
        transform = CountingIdentity()
        model.register_parametrization_in_graph("cm", transform)
        model.populate_parameter_buffers()
        model.integrator.initialized = False
        model.initialize()
        before = transform.calls

        runner = dn.compile(
            model,
            dt=0.005,
            steps=4,
            chunk_steps=4,
            execution="scan",
            inference=True,
            initialization="fresh",
            backend="eager",
        )
        plan = runner._functional

        def calls(population):
            return population.in_graph_parametrizations["cm"][0][0].calls

        assert transform.calls == before
        assert calls(plan._transition.population) == before
        assert calls(plan._initialization.population) == before
        assert calls(plan._preparation.population) == before + 1
