from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import hh

DT = 0.01
STEPS = 2
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _ClockMutator(Mechanism):
    """Invalid mechanism used to verify audit-clone transaction isolation."""

    Mechanism.ASSIGNED("probe")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def assigned_values(self, v, values):
        del values
        self.t.add_(1.0)
        return {"probe": torch.zeros_like(v)}

    def i(self, v):
        return torch.zeros_like(v)

    def i_with_conductance(self, v):
        zero = torch.zeros_like(v)
        return zero, zero


def _model(method: str):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0],
            L=2.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -58.0, -62.0]),
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method=method, imem=False),
        )
        model.insert(hh)
        model.initialize()
        model.train()
    return model


def _case(method: str = "pcr"):
    if method == "thomas":
        try:
            from dendra_solvers import thomas_solve_t
        except ImportError:
            pytest.skip("transform-compatible dendra-solvers facade is unavailable")

    model = _model(method)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    if method == "thomas":
        assert functional._transition.solver is thomas_solve_t
    ve = torch.linspace(
        -1.0,
        1.0,
        STEPS * model.v.numel(),
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(STEPS, *model.shape)
    return model, functional, tensors, ve


def _atomic_final_voltage(functional, tensors):
    """Expose every differentiated source before atomic preparation."""

    def final_voltage(gnabar, diameter, initial_voltage, ve):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        constants = dict(tensors.constants)
        constants["diam"] = diameter
        state = dict(tensors.state)
        state["integrator"] = dict(tensors.state["integrator"])
        state["integrator"]["v"] = initial_voltage
        final, _aux = functional.prepare_and_rollout(
            parameters,
            constants,
            state,
            dn.func.RolloutInput(ve=ve),
            steps=STEPS,
        )
        return final["integrator"]["v"]

    return final_voltage


def _explicit_sources(tensors, ve):
    return (
        tensors.parameters[GNABAR],
        tensors.constants["diam"],
        tensors.state["integrator"]["v"],
        ve,
    )


def _compiled(transformed):
    return torch.compile(
        transformed,
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )


def _assert_derivative_tree_close(actual, expected, names):
    assert len(actual) == len(expected) == len(names)
    for name, actual_value, expected_value in zip(
        names,
        actual,
        expected,
        strict=True,
    ):
        assert torch.isfinite(actual_value).all(), f"non-finite {name} derivative"
        assert torch.count_nonzero(actual_value) > 0, f"disconnected {name} derivative"
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-9,
            atol=2.0e-10,
            msg=f"compiled {name} derivative differs from eager torch.func",
        )


@pytest.mark.parametrize("method", ["pcr", "thomas"])
def test_compile_jacrev_of_atomic_rollout_tracks_every_explicit_source(method):
    _model_value, functional, tensors, ve = _case(method)
    final_voltage = _atomic_final_voltage(functional, tensors)
    sources = _explicit_sources(tensors, ve)
    transformed = torch.func.jacrev(final_voltage, argnums=(0, 1, 2, 3))
    expected = transformed(*sources)

    with torch_compiler_warning_context():
        actual = _compiled(transformed)(*sources)

    _assert_derivative_tree_close(
        actual,
        expected,
        ("raw parameter", "geometry", "initial state", "drive"),
    )


def test_compile_jacfwd_of_atomic_rollout_tracks_geometry_through_preparation():
    _model_value, functional, tensors, ve = _case()
    final_voltage = _atomic_final_voltage(functional, tensors)
    gnabar, diameter, initial_voltage, _ve = _explicit_sources(tensors, ve)

    def voltage_from_diameter(local_diameter):
        return final_voltage(gnabar, local_diameter, initial_voltage, ve)

    transformed = torch.func.jacfwd(voltage_from_diameter)
    expected = transformed(diameter)
    with torch_compiler_warning_context():
        actual = _compiled(transformed)(diameter)

    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) > 0, "geometry derivative was disconnected"
    torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-10)


def test_compile_vmap_of_atomic_rollout_matches_eager_torch_func_lanes():
    model, functional, tensors, ve = _case()
    final_voltage = _atomic_final_voltage(functional, tensors)
    gnabar, diameter, initial_voltage, _ve = _explicit_sources(tensors, ve)
    scales = model.v.new_tensor([0.9, 1.0, 1.1])
    offsets = model.v.new_tensor([-0.25, 0.0, 0.25])
    drive_offsets = model.v.new_tensor([-0.1, 0.0, 0.1])
    gnabar_lanes = gnabar * scales
    diameter_lanes = diameter.unsqueeze(0) * scales.reshape(3, 1, 1)
    voltage_lanes = initial_voltage.unsqueeze(0) + offsets.reshape(3, 1, 1)
    ve_lanes = ve.unsqueeze(0) + drive_offsets.reshape(3, 1, 1, 1)

    transformed = torch.func.vmap(final_voltage)
    expected = transformed(
        gnabar_lanes,
        diameter_lanes,
        voltage_lanes,
        ve_lanes,
    )
    with torch_compiler_warning_context():
        actual = _compiled(transformed)(
            gnabar_lanes,
            diameter_lanes,
            voltage_lanes,
            ve_lanes,
        )

    assert actual.shape == (3, *model.shape)
    assert not torch.equal(actual[0], actual[2])
    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


def test_compile_hessian_of_atomic_rollout_matches_eager_torch_func():
    _model_value, functional, tensors, ve = _case()
    final_voltage = _atomic_final_voltage(functional, tensors)
    gnabar, diameter, initial_voltage, _ve = _explicit_sources(tensors, ve)

    def loss(local_gnabar):
        final_voltage_value = final_voltage(
            local_gnabar,
            diameter,
            initial_voltage,
            ve,
        )
        return final_voltage_value.square().mean()

    transformed = torch.func.hessian(loss)
    expected = transformed(gnabar)
    with torch_compiler_warning_context():
        actual = _compiled(transformed)(gnabar)

    assert torch.isfinite(actual)
    assert actual.abs() > 0, "raw-parameter Hessian was disconnected"
    torch.testing.assert_close(actual, expected, rtol=2.0e-9, atol=2.0e-10)


def test_audit_clone_clock_reference_is_isolated_and_mutation_fails_closed(
    monkeypatch,
):
    import dendra.func._population as population_lowering

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(1, dtype=torch.float64)
        model.insert(_ClockMutator)
        model.initialize()
        model.train()

    source_clock = model.t
    source_clock_value = model.t.detach().clone()
    accepted = {}
    audit = population_lowering.FunctionalPopulation._audit_read_only_workspaces

    def capture_accepted_clock(functional, tensors):
        accepted["before"] = functional._transition.population.t
        accepted["value"] = accepted["before"].detach().clone()
        try:
            return audit(functional, tensors)
        finally:
            accepted["after"] = functional._transition.population.t

    monkeypatch.setattr(
        population_lowering.FunctionalPopulation,
        "_audit_read_only_workspaces",
        capture_accepted_clock,
    )

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"read-only transition inputs.*positional:t",
    ):
        dn.func.make_functional(model, dt=DT)

    assert model.t is source_clock
    torch.testing.assert_close(model.t, source_clock_value, rtol=0.0, atol=0.0)
    assert accepted["after"] is accepted["before"]
    torch.testing.assert_close(
        accepted["after"],
        accepted["value"],
        rtol=0.0,
        atol=0.0,
    )
