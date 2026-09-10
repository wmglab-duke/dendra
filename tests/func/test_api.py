from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State
from dendra.models.mechanisms._material_process import DiffusionProcess
from dendra.models.mod import hh, pas

DT = 0.01


class _DiameterState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = diam")

    def state_defaults(self, v, values):
        del values
        return {"x": torch.zeros_like(v)}


class _DiameterMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_DiameterState)


class _TableMechanism(Mechanism):
    Mechanism.GLOBAL(a=1.0)
    Mechanism.TABLE("curve", -100.0, 100.0, 201)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def curve(self, x):
        return self.a * x

    def i(self, v):
        return self.curve(v)


class _FlagMechanism(Mechanism):
    Mechanism.FLAG(flip=False)
    Mechanism.GLOBAL(g=1.0e-4)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        sign = -1.0 if self.flip else 1.0
        return sign * self.g * v.square()


class _InplaceBufferMechanism(Mechanism):
    Mechanism.CARRY("gain")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def initial_values(self, v, values):
        del values
        return {"gain": torch.ones_like(v)}

    def advance(self, v, dt, values):
        del v, dt
        return {"gain": values["gain"] + 1.0}

    def i(self, v):
        return self.gain * v

    def i_with_conductance(self, v):
        return self.i(v), self.gain.expand_as(v)


class _SavedCurrentMechanism(Mechanism):
    Mechanism.GLOBAL(g=1.0e-4)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.SAVE_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * v


class _WriteOnlySodium(Mechanism):
    Mechanism.USEION("na", write=["nai"])

    def initial_values(self, v, values):
        del values
        return {"nai": torch.full_like(v, 3.0)}

    def advance(self, v, dt, values):
        del v, dt
        return {"nai": values["nai"] + 1.0}


class _WriteOnlyMaterialLocals(Mechanism):
    Mechanism.USEMATERIAL(
        "pool",
        write=["amount"],
        source={"amount": "delta"},
    )

    def initial_values(self, v, values):
        del values
        return {
            "amount": torch.full_like(v, 3.0),
            "delta": torch.full_like(v, 0.25),
        }

    def advance(self, v, dt, values):
        del v, dt
        return {
            "amount": values["amount"] + 1.0,
            "delta": values["delta"] + 0.5,
        }


def _model(*, dtype=torch.float64, method="pcr", jit=0):
    with dn.ctx(JIT=jit, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -60.0, -56.0, -59.0, -63.0]),
            dtype=dtype,
            integrator=dn.bwd_euler_ub(method=method, imem=False),
        )
        model.insert(hh)
        model.initialize()
        model.train()
    return model


def test_func_namespace_exports_the_functional_population_api():
    assert "func" in dn.__all__
    assert set(dn.func.__all__) == {
        "APCount",
        "BoundPopulation",
        "CompiledPopulationChunk",
        "AnomalyDetector",
        "ExecutionCapabilities",
        "ExecutionReport",
        "FunctionalCallback",
        "FunctionalCallbackResults",
        "FunctionalCallbackState",
        "FunctionalCallbacks",
        "FunctionalExtra",
        "FunctionalIntra",
        "FunctionalPopulation",
        "FunctionalizationError",
        "InitializationInput",
        "PopulationTensors",
        "Raster",
        "Recorder",
        "RolloutInput",
        "StepInput",
        "StimulusTensors",
        "longrun",
        "longrun_checkpointed",
        "make_functional",
        "run",
    }


def test_make_functional_extracts_the_minimal_hh_carry_without_aliasing():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    assert functional.shape == model.shape
    assert "local_geometry" not in prepared.values
    assert set(tensors.state) == {"clock", "control", "integrator", "mechanisms"}
    assert set(tensors.state["integrator"]) == {"v"}
    assert set(tensors.state["mechanisms"]) == {"hh"}
    assert set(tensors.state["mechanisms"]["hh"]) == {"m", "h", "n"}
    assert set(tensors.state["clock"]) == {"t"}
    assert set(tensors.state["control"]) == {"duration_remainder"}
    assert set(tensors.constants) == {"diam", "dx", "dt"}
    torch.testing.assert_close(
        tensors.constants["dt"],
        torch.as_tensor(DT, dtype=model.dtype(), device=model.device()),
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(tensors.parameters) == tuple(dict(model.named_parameters()))

    source = {
        "v": model.v,
        "m": model.mech.hh.m,
        "h": model.mech.hh.h,
        "n": model.mech.hh.n,
        "t": model.t,
        "duration_remainder": model._duration_remainder,
        "diam": model.diam,
        "dx": model.dx,
    }
    extracted = {
        "v": tensors.state["integrator"]["v"],
        **tensors.state["mechanisms"]["hh"],
        "t": tensors.state["clock"]["t"],
        "duration_remainder": tensors.state["control"]["duration_remainder"],
        "diam": tensors.constants["diam"],
        "dx": tensors.constants["dx"],
    }
    for name, value in extracted.items():
        assert (
            value.untyped_storage().data_ptr()
            != source[name].untyped_storage().data_ptr()
        )


def test_make_functional_accepts_exact_single_compartment_and_batch_layouts():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=2,
            C=1,
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(pas)
        model.batch(3)
        model.initialize()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    assert functional.shape == (3, 2, 1)
    assert prepared.values["integrator"]["cmdt"].shape == (1, 2, 1)
    assert prepared.values["integrator"]["area"].shape == (2, 1)


@pytest.mark.parametrize(
    ("model_factory", "message"),
    [
        (
            lambda: dn.SingleCompartment(
                N=1,
                C=1,
                dtype=torch.float64,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            ),
            "SingleCompartment functionalization requires exactly bwd_euler_sc",
        ),
        (
            lambda: dn.Unmyelinated(
                [2.0],
                L=4.0,
                dx=1.0,
                dtype=torch.float64,
                integrator=dn.bwd_euler_sc(imem=False),
            ),
            "Unmyelinated functionalization requires bwd_euler_ub",
        ),
    ],
)
def test_make_functional_rejects_mismatched_topology_integrator_pairs(
    model_factory,
    message,
):
    with dn.ctx(JIT=0):
        model = model_factory()
        model.insert(pas)
        model.initialize()

    with pytest.raises(dn.func.FunctionalizationError, match=message):
        dn.func.make_functional(model, dt=DT)


def test_make_functional_rejects_single_compartment_imem_recording():
    with dn.ctx(JIT=0):
        model = dn.SingleCompartment(N=1, C=1, dtype=torch.float64)
        model.imem = True
        model.insert(pas)
        model.initialize()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="i_membrane recording is not supported yet",
    ):
        dn.func.make_functional(model, dt=DT)


def test_make_functional_rejects_single_compartment_subclasses():
    class CustomSingleCompartment(dn.SingleCompartment):
        pass

    with dn.ctx(JIT=0):
        model = CustomSingleCompartment(N=1, C=1, dtype=torch.float64)
        model.insert(pas)
        model.initialize()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="exact SingleCompartment",
    ):
        dn.func.make_functional(model, dt=DT)


def test_make_functional_rejects_uninitialized_population():
    model = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        integrator=dn.bwd_euler_ub(method="pcr"),
    )
    model.insert(hh)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="must be initialized",
    ):
        dn.func.make_functional(model, dt=DT)


def test_make_functional_rejects_unlowered_material_process_workspaces():
    class Diff(DiffusionProcess):
        DiffusionProcess.GLOBAL(D=0.125)
        DiffusionProcess.METHOD("explicit", solver="dense")
        DiffusionProcess.DIFFUSE("solute", field="c", D="D")

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.material(
            "solute",
            fields={"c": 0.5},
            min_values={"c": 0.0},
            domain="intracellular",
        )
        model.insert(pas)
        model.insert(Diff)
        model.initialize()
        model.train()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="MaterialProcesses.*workspaces",
    ):
        dn.func.make_functional(model, dt=DT)


def test_make_functional_rejects_unlowered_table_workspaces():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(_TableMechanism)
        model.initialize()
        model.train()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="TABLE lookup workspaces",
    ):
        dn.func.make_functional(model, dt=DT)


def test_mechanism_timestep_configuration_is_not_a_public_author_hook():
    assert not hasattr(Mechanism, "set_dt")


def test_runtime_morphology_reads_follow_extracted_geometry_and_are_differentiable():
    def make_model(diameter):
        with dn.ctx(JIT=0, REQUIRE_GRAD=1):
            model = dn.Unmyelinated(
                [diameter],
                L=4.0,
                dx=1.0,
                dtype=torch.float64,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
            model.insert(pas)
            model.insert(_DiameterMechanism)
            model.initialize()
            model.train()
        return model

    source = make_model(1.0)
    target = make_model(2.0)
    imperative = make_model(2.0)
    functional, _source_tensors = dn.func.make_functional(source, dt=DT)
    tensors = functional.extract(target)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    state, _aux = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    dt = torch.as_tensor(DT, dtype=imperative.dtype())
    imperative.integrator._initialize(
        imperative,
        dt,
        force=True,
        compile_scope="population",
    )
    imperative.integrator.step(imperative, dt)
    imperative.t = imperative.t + dt

    torch.testing.assert_close(
        state["mechanisms"]["_DiameterMechanism"]["x"],
        imperative.mech._DiameterMechanism.x,
        rtol=0.0,
        atol=0.0,
    )

    def next_x(diameter):
        constants = dict(tensors.constants)
        constants["diam"] = diameter
        local_prepared = functional.prepare(tensors.parameters, constants)
        next_state, _aux = functional.step(
            tensors.parameters,
            local_prepared,
            tensors.state,
        )
        return next_state["mechanisms"]["_DiameterMechanism"]["x"]

    jacobian = torch.func.jacrev(next_x)(tensors.constants["diam"])
    expected = DT * torch.eye(
        tensors.constants["diam"].numel(),
        dtype=tensors.constants["diam"].dtype,
        device=tensors.constants["diam"].device,
    ).reshape(
        *tensors.constants["diam"].shape,
        *tensors.constants["diam"].shape,
    )
    torch.testing.assert_close(jacobian, expected, rtol=0.0, atol=0.0)


def test_authored_carry_updates_are_isolated_and_differentiable():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(_InplaceBufferMechanism)
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    gain = tensors.state["mechanism_buffers"]["_InplaceBufferMechanism"]["gain"]
    original = gain.clone()
    original_version = gain._version

    next_state, _aux = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    assert gain._version == original_version
    torch.testing.assert_close(gain, original, rtol=0.0, atol=0.0)
    assert not torch.equal(
        next_state["mechanism_buffers"]["_InplaceBufferMechanism"]["gain"],
        gain,
    )

    def next_voltage(local_gain):
        state = {
            **tensors.state,
            "mechanism_buffers": {
                **tensors.state["mechanism_buffers"],
                "_InplaceBufferMechanism": {
                    "gain": local_gain,
                },
            },
        }
        return functional.step(tensors.parameters, prepared, state)[0]["integrator"][
            "v"
        ]

    jacobian = torch.func.jacrev(next_voltage)(gain)
    assert torch.isfinite(jacobian).all()
    assert torch.count_nonzero(jacobian) > 0


def test_saved_current_mirror_is_framework_owned_functional_checkpoint_state():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(_SavedCurrentMechanism)
        model.build()
        mechanism = model.mech._SavedCurrentMechanism
        torch.testing.assert_close(mechanism.i_, torch.zeros_like(mechanism.i_))
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    mirror = tensors.state["mechanism_buffers"]["_SavedCurrentMechanism"]["i_"]
    torch.testing.assert_close(mirror, mechanism.i_)
    checkpoint = model.mech.mutable_state_dict()
    torch.testing.assert_close(checkpoint["_SavedCurrentMechanism.i_"], mirror)

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    next_state, _aux = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    assert "i_" in next_state["mechanism_buffers"]["_SavedCurrentMechanism"]


@pytest.mark.parametrize("kind", ["ion", "material"])
def test_write_only_shared_field_locals_are_complete_differentiable_carry(kind):
    def make_model():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1):
            model = dn.Unmyelinated(
                [1.0],
                L=4.0,
                dx=1.0,
                dtype=torch.float64,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
            model.insert(pas)
            if kind == "ion":
                model.insert(_WriteOnlySodium)
            else:
                model.material(
                    "pool",
                    fields={"amount": 0.0},
                    min_values={"amount": 0.0},
                    domain="intracellular",
                )
                model.insert(_WriteOnlyMaterialLocals)
            model.initialize()
            model.train()
        return model

    source = make_model()
    imperative = make_model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    if kind == "ion":
        section = "ion_write_buffers"
        mechanism_name = "_WriteOnlySodium"
        local_name = "nai"
        canonical_path = ("ions", "na", "nai")
    else:
        section = "material_source_buffers"
        mechanism_name = "_WriteOnlyMaterialLocals"
        local_name = "delta"
        canonical_path = ("materials", "pool", "amount")
        assert set(tensors.state["material_write_buffers"][mechanism_name]) == {
            "amount"
        }

    assert set(tensors.state[section][mechanism_name]) == {local_name}

    chained = tensors.state
    for _ in range(2):
        chained, _aux = functional.step(
            tensors.parameters,
            prepared,
            chained,
        )
    fused, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        steps=2,
    )

    def atomic(state):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            steps=5,
        )[0]

    assert functional.prewarm_structured_rollout()
    structured_expected = atomic(tensors.state)
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    compiled = torch.compile(atomic, backend=backend, fullgraph=True)
    with torch.no_grad():
        structured_actual = compiled(tensors.state)
    for actual_leaf, expected_leaf in zip(
        torch.utils._pytree.tree_leaves(structured_actual),
        torch.utils._pytree.tree_leaves(structured_expected),
        strict=True,
    ):
        torch.testing.assert_close(actual_leaf, expected_leaf, rtol=0.0, atol=0.0)
    assert len(graphs) == 1
    assert any(
        node.op == "call_function" and node.target is torch.ops.higher_order.while_loop
        for node in graphs[0].graph.nodes
    )

    dt = torch.as_tensor(DT, dtype=imperative.dtype())
    imperative.integrator._initialize(
        imperative,
        dt,
        force=True,
        compile_scope="population",
    )
    for _ in range(2):
        imperative.integrator.step(imperative, dt)
        imperative.t = imperative.t + dt
    expected = functional.extract(imperative).state

    for actual_leaf, expected_leaf in zip(
        torch.utils._pytree.tree_leaves(chained),
        torch.utils._pytree.tree_leaves(expected),
        strict=True,
    ):
        torch.testing.assert_close(actual_leaf, expected_leaf, rtol=0.0, atol=0.0)
    for actual_leaf, expected_leaf in zip(
        torch.utils._pytree.tree_leaves(fused),
        torch.utils._pytree.tree_leaves(expected),
        strict=True,
    ):
        torch.testing.assert_close(actual_leaf, expected_leaf, rtol=0.0, atol=0.0)

    def canonical_after_step(local_value):
        state = {
            **tensors.state,
            section: {
                **tensors.state[section],
                mechanism_name: {
                    **tensors.state[section][mechanism_name],
                    local_name: local_value,
                },
            },
        }
        next_state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
        )
        value = next_state
        for path_part in canonical_path:
            value = value[path_part]
        return value

    local_value = tensors.state[section][mechanism_name][local_name]
    jacobian = torch.func.jacrev(canonical_after_step)(local_value)
    assert torch.isfinite(jacobian).all()
    assert torch.count_nonzero(jacobian) > 0


def test_state_commit_is_validated_before_any_target_mutation():
    source = _model()
    target = _model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    checkpoint = target.state_dict_for_checkpoint()
    bad_state = {
        **tensors.state,
        "integrator": {"v": torch.zeros(1, dtype=source.dtype())},
    }

    with pytest.raises(ValueError, match=r"state leaf 'integrator\.v'.*shape"):
        functional.commit_state_(target, bad_state)

    after = target.state_dict_for_checkpoint()
    assert torch.equal(after["integrator"]["v"], checkpoint["integrator"]["v"])
    for name in ("hh.m", "hh.h", "hh.n"):
        assert torch.equal(after["mech"][name], checkpoint["mech"][name])


def test_structure_changes_require_lowering_again():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    model.build(force_rebuild=True)

    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)


def test_parameter_transform_and_integrator_config_are_plan_structure():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    model.rhoa_scale_param.beta = 2.0

    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)

    target = _model()
    target.integrator.imem = True
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure does not match",
    ):
        functional.extract(target)


def test_flags_and_training_mode_are_plan_structure():
    def make_flag_model(*, flip=False):
        with dn.ctx(JIT=0, REQUIRE_GRAD=1):
            model = dn.Unmyelinated(
                [1.0],
                L=4.0,
                dx=1.0,
                dtype=torch.float64,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
            model.insert(_FlagMechanism, flip=flip)
            model.initialize()
            model.train()
        return model

    source = make_flag_model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    different_flag = make_flag_model(flip=True)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure does not match",
    ):
        functional.extract(different_flag)

    source.mech._FlagMechanism.flip = True
    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)

    training_source = _model()
    training_functional, training_tensors = dn.func.make_functional(
        training_source,
        dt=DT,
    )
    eval_target = _model().eval()
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure does not match",
    ):
        training_functional.extract(eval_target)

    training_source.eval()
    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        training_functional.prepare(
            training_tensors.parameters,
            training_tensors.constants,
        )


def test_make_functional_rejects_subclasses_with_unlowered_geometry_semantics():
    class AlteredAreaUnmyelinated(dn.Unmyelinated):
        @property
        def area(self):
            return 100.0 * super().area

    model = AlteredAreaUnmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        dtype=torch.float64,
        integrator=dn.bwd_euler_ub(method="pcr"),
    )
    model.insert(hh)
    model.initialize()

    with pytest.raises(dn.func.FunctionalizationError, match="exactly Unmyelinated"):
        dn.func.make_functional(model, dt=DT)


def test_prepared_values_are_plan_bound_and_reject_stale_dependencies():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    parameter_name = "integrator.mech.mechanisms.hh.gnabar_param"
    changed_parameters = dict(tensors.parameters)
    changed_parameters[parameter_name] = 1.1 * changed_parameters[parameter_name]

    with pytest.raises(dn.func.FunctionalizationError, match="parameters changed"):
        functional.step(changed_parameters, prepared, tensors.state)

    other_model = _model()
    other_functional, other_tensors = dn.func.make_functional(other_model, dt=2 * DT)
    other_prepared = other_functional.prepare(
        other_tensors.parameters,
        other_tensors.constants,
    )
    with pytest.raises(dn.func.FunctionalizationError, match="different.*plan"):
        functional.step(tensors.parameters, other_prepared, tensors.state)

    tensors.constants["diam"].add_(0.1)
    with pytest.raises(dn.func.FunctionalizationError, match="constants changed"):
        functional.step(tensors.parameters, prepared, tensors.state)


def test_compilation_requires_atomic_preparation_and_consumption():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    compiled_prepare = torch.compile(
        functional.prepare,
        backend="eager",
        fullgraph=True,
    )
    with pytest.raises(Exception, match="prepare_and_(step|rollout)"):
        compiled_prepare(tensors.parameters, tensors.constants)

    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def consume_opaque(parameters, opaque, state):
        return functional.step(parameters, opaque, state)[0]["integrator"]["v"]

    compiled_consumer = torch.compile(
        consume_opaque,
        backend="eager",
        fullgraph=True,
    )
    with pytest.raises(Exception, match="prepare_and_(step|rollout)"):
        compiled_consumer(tensors.parameters, prepared, tensors.state)

    def atomic(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)[0][
            "integrator"
        ]["v"]

    compiled_atomic = torch.compile(atomic, backend="eager", fullgraph=True)
    actual = compiled_atomic(
        tensors.parameters,
        tensors.constants,
        tensors.state,
    )
    expected = functional.step(tensors.parameters, prepared, tensors.state)[0][
        "integrator"
    ]["v"]
    torch.testing.assert_close(actual, expected)


def test_functional_api_reports_malformed_nested_inputs_at_the_boundary():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    malformed_state = {**tensors.state, "integrator": tensors.state["integrator"]["v"]}
    with pytest.raises(TypeError, match=r"state\['integrator'\].*mapping"):
        functional.step(tensors.parameters, prepared, malformed_state)
    with pytest.raises(TypeError, match="ve must be a Tensor"):
        functional.step(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.StepInput(ve=[0.0]),
        )
    with pytest.raises(TypeError, match="opaque value returned by prepare"):
        functional.step(tensors.parameters, None, tensors.state)
    with pytest.raises(TypeError, match="constants must be a mapping"):
        functional.prepare(tensors.parameters, None)


def test_make_functional_rejects_boolean_timestep_and_commit_rejects_bad_control():
    model = _model()
    with pytest.raises(TypeError, match="finite positive scalar"):
        dn.func.make_functional(model, dt=True)

    functional, tensors = dn.func.make_functional(model, dt=DT)
    bad_state = {
        **tensors.state,
        "control": {
            "duration_remainder": torch.tensor(-0.1, dtype=torch.float64),
        },
    }
    checkpoint = model.state_dict_for_checkpoint()
    with pytest.raises(ValueError, match="finite and non-negative"):
        functional.commit_state_(model, bad_state)
    assert torch.equal(model.v, checkpoint["integrator"]["v"])
