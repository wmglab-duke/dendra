"""Functional correctness oracles for dense custom Material semantics."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import pas

DT = 0.03125
STEPS = 3

FIRST_RATE = 0.25
FIRST_SOURCE_SCALE = 0.125
LAST_RETAIN = 0.875
LAST_RATE = 0.0625
LAST_SOURCE_SCALE = 0.03125


class _FirstMaterialWriter(Mechanism):
    """Write first, then contribute an additive source from that local value."""

    Mechanism.GLOBAL(rate=FIRST_RATE, source_scale=FIRST_SOURCE_SCALE)
    Mechanism.USEMATERIAL(
        "pool",
        read=["amount"],
        write=["amount"],
        source={"amount": "delta"},
    )

    def _advance(self, v, dt):
        self.amount = self.amount + dt * self.rate
        self.delta = dt * self.source_scale * self.amount


class _LastMaterialWriter(Mechanism):
    """Provide the accepted replacement and a material-dependent current."""

    Mechanism.GLOBAL(
        retain=LAST_RETAIN,
        rate=LAST_RATE,
        source_scale=LAST_SOURCE_SCALE,
        g=2.0**-12,
        e=-50.0,
    )
    Mechanism.USEMATERIAL(
        "pool",
        read=["amount"],
        write=["amount"],
        source={"amount": "delta"},
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def _advance(self, v, dt):
        amount = self.amount
        self.amount = self.retain * amount + dt * self.rate
        self.delta = dt * self.source_scale * amount

    def i(self, v):
        conductance = self.g * self.amount
        return conductance * (v - self.e)

    def i_with_conductance(self, v):
        conductance = self.g * self.amount
        return conductance * (v - self.e), conductance


class _FirstMaterialReader(Mechanism):
    Mechanism.USEMATERIAL("pool", read=["amount"])


class _SecondMaterialReader(Mechanism):
    Mechanism.USEMATERIAL("pool", read=["amount"])


def _model(*, training=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        # A scalar source keeps the raw initialization parameter compatible
        # with the current scalar-parameter functional capability slice. The
        # nonuniform accepted state below is deliberately independent of it.
        model.material(
            "pool",
            fields={"amount": 0.5},
            min_values={"amount": 0.0},
            domain={"amount": "intracellular"},
            units={"amount": "a.u."},
            conserved={"amount": False},
        )
        model.insert(pas, g=2.0**-16, e=-65.0)

        def scalar(value):
            return torch.as_tensor(value, dtype=torch.float64)

        model.insert(
            _FirstMaterialWriter,
            rate=scalar(FIRST_RATE),
            source_scale=scalar(FIRST_SOURCE_SCALE),
        )
        model.insert(
            _LastMaterialWriter,
            retain=scalar(LAST_RETAIN),
            rate=scalar(LAST_RATE),
            source_scale=scalar(LAST_SOURCE_SCALE),
            g=scalar(2.0**-12),
            e=scalar(-50.0),
        )
        model.insert(_FirstMaterialReader)
        model.insert(_SecondMaterialReader)
        model.train(training)
        model.initialize()

        amount = torch.linspace(
            0.4,
            0.8,
            model.v.numel(),
            device=model.device(),
            dtype=model.dtype(),
        ).reshape(model.shape)
        model.mech.materials["pool"].amount = amount
        model.mech.read_from_materials()
    return model


def _drives(model, steps):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -0.4,
        0.3,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -2.0e-10,
        3.0e-10,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _imperative_step(model, ve=None, intra=None):
    dt = torch.as_tensor(DT, device=model.device(), dtype=model.dtype())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    model.integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def _replace_amount(state, amount):
    return {
        **state,
        "materials": {
            **state["materials"],
            "pool": {
                **state["materials"]["pool"],
                "amount": amount,
            },
        },
    }


def _material_update(amount):
    """Independent one-step oracle for ordered replacements plus sources."""
    first_replacement = amount + DT * FIRST_RATE
    first_source = DT * FIRST_SOURCE_SCALE * first_replacement
    last_replacement = LAST_RETAIN * amount + DT * LAST_RATE
    last_source = DT * LAST_SOURCE_SCALE * amount
    return last_replacement + first_source + last_source


def _assert_every_leaf_close(actual, expected, *, rtol=2.0e-10, atol=2.0e-11):
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


def _assert_read_locals_are_synchronized(model):
    canonical = model.mech.materials["pool"].amount
    for name in (
        "_FirstMaterialWriter",
        "_LastMaterialWriter",
        "_FirstMaterialReader",
        "_SecondMaterialReader",
    ):
        torch.testing.assert_close(
            model.mech.mechanisms[name].amount,
            canonical,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("training", [False, True])
def test_dense_material_ordering_and_every_leaf_multistep_parity(training):
    functional_model = _model(training=training)
    imperative_model = _model(training=training)
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    assert not functional_model.mech.material_processes
    assert set(tensors.state) == {
        "clock",
        "control",
        "integrator",
        "materials",
        "material_source_buffers",
    }
    assert set(tensors.state["materials"]) == {"pool"}
    assert set(tensors.state["materials"]["pool"]) == {"amount"}
    assert {
        name: set(values)
        for name, values in tensors.state["material_source_buffers"].items()
    } == {
        "_FirstMaterialWriter": {"delta"},
        "_LastMaterialWriter": {"delta"},
    }

    handler = functional_model.mech
    assert [entry[2].name for entry in handler._material_write_support_plan] == [
        "_FirstMaterialWriter",
        "_LastMaterialWriter",
    ]
    assert [entry[3].name for entry in handler._material_source_support_plan] == [
        "_FirstMaterialWriter",
        "_LastMaterialWriter",
    ]

    ve, intra = _drives(functional_model, STEPS)
    initial_state = tensors.state
    chained = initial_state
    expected_amount = initial_state["materials"]["pool"]["amount"]

    for index in range(STEPS):
        chained, _aux = functional.step(
            tensors.parameters,
            prepared,
            chained,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(imperative_model, ve[index], intra[index])
        imperative_state = functional.extract(imperative_model).state
        _assert_every_leaf_close(chained, imperative_state)

        previous_amount = expected_amount
        expected_amount = _material_update(previous_amount)
        torch.testing.assert_close(
            chained["materials"]["pool"]["amount"],
            expected_amount,
            rtol=0.0,
            atol=2.0e-16,
        )
        torch.testing.assert_close(
            chained["material_source_buffers"]["_FirstMaterialWriter"]["delta"],
            DT * FIRST_SOURCE_SCALE * (previous_amount + DT * FIRST_RATE),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            chained["material_source_buffers"]["_LastMaterialWriter"]["delta"],
            DT * LAST_SOURCE_SCALE * previous_amount,
            rtol=0.0,
            atol=0.0,
        )
        _assert_read_locals_are_synchronized(imperative_model)

    fused, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        initial_state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_every_leaf_close(fused, chained)
    _assert_every_leaf_close(fused, functional.extract(imperative_model).state)


def test_dense_material_state_and_parameter_gradients_match_analytic_oracles():
    model = _model(training=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    initial_amount = tensors.state["materials"]["pool"]["amount"]

    def final_amount(amount):
        state, _aux = functional.rollout(
            tensors.parameters,
            prepared,
            _replace_amount(tensors.state, amount),
            steps=STEPS,
        )
        return state["materials"]["pool"]["amount"]

    jacobian = torch.func.jacrev(final_amount)(initial_amount)
    multiplier = LAST_RETAIN + DT * (FIRST_SOURCE_SCALE + LAST_SOURCE_SCALE)
    expected_jacobian = (multiplier**STEPS) * torch.eye(
        initial_amount.numel(),
        device=initial_amount.device,
        dtype=initial_amount.dtype,
    ).reshape(*initial_amount.shape, *initial_amount.shape)
    torch.testing.assert_close(
        jacobian,
        expected_jacobian,
        rtol=2.0e-12,
        atol=2.0e-13,
    )

    parameter_name = "integrator.mech.mechanisms._FirstMaterialWriter.rate_param"

    def loss(first_rate):
        parameters = dict(tensors.parameters)
        parameters[parameter_name] = first_rate
        state, _aux = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=STEPS,
        )
        return state["materials"]["pool"]["amount"].sum()

    actual_gradient = torch.func.grad(loss)(tensors.parameters[parameter_name])
    geometric_sum = sum(multiplier**power for power in range(STEPS))
    expected_gradient = initial_amount.new_tensor(
        initial_amount.numel() * DT**2 * FIRST_SOURCE_SCALE * geometric_sum
    )
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        rtol=2.0e-12,
        atol=2.0e-13,
    )


def test_dense_material_vmap_matches_explicit_lanes_and_accepts_zero_lanes():
    model = _model(training=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    initial_amount = tensors.state["materials"]["pool"]["amount"]

    def run_lane(amount):
        return functional.rollout(
            tensors.parameters,
            prepared,
            _replace_amount(tensors.state, amount),
            steps=STEPS,
        )[0]

    lane_amounts = torch.stack((0.75 * initial_amount, 1.25 * initial_amount))
    vmapped = torch.vmap(run_lane)(lane_amounts)
    explicit = torch.utils._pytree.tree_map(
        lambda *values: torch.stack(values),
        *(run_lane(amount) for amount in lane_amounts),
    )
    _assert_every_leaf_close(vmapped, explicit)

    empty = initial_amount.new_empty((0, *initial_amount.shape))
    zero_lanes = torch.vmap(run_lane)(empty)
    for leaf in torch.utils._pytree.tree_leaves(zero_lanes):
        assert leaf.shape[0] == 0


def test_dense_material_atomic_rollout_is_fullgraph_compilable():
    model = _model(training=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, STEPS)

    def atomic(state, ve_values, intra_values):
        next_state, _aux = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
        )
        return (
            next_state["integrator"]["v"],
            next_state["materials"]["pool"]["amount"],
            next_state["material_source_buffers"]["_FirstMaterialWriter"]["delta"],
            next_state["material_source_buffers"]["_LastMaterialWriter"]["delta"],
        )

    expected = atomic(tensors.state, ve, intra)
    compiled = torch.compile(atomic, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        actual = compiled(tensors.state, ve, intra)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-10,
            atol=2.0e-11,
        )


@pytest.mark.parametrize("training", [False, True])
def test_dense_material_commit_and_imperative_resume(training):
    source = _model(training=training)
    target = _model(training=training)
    reference = _model(training=training)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, 5)

    checkpoint, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve[:STEPS], intra=intra[:STEPS]),
    )
    functional.commit_state_(target, checkpoint)
    _assert_read_locals_are_synchronized(target)

    for index in range(STEPS, 5):
        _imperative_step(target, ve[index], intra[index])
    for index in range(5):
        _imperative_step(reference, ve[index], intra[index])

    _assert_every_leaf_close(
        functional.extract(target).state,
        functional.extract(reference).state,
    )
    _assert_read_locals_are_synchronized(target)
    _assert_read_locals_are_synchronized(reference)
