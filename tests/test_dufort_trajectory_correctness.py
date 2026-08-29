"""Trajectory-level correctness oracles for Dufort--Frankel integration."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mod import exp2syn

DTYPE = torch.float64
DT = 0.01
STEPS = 50
G = 1.0e-4
E = -62.0
INITIAL_V = torch.tensor([-72.0, -66.0, -50.0, -61.0, -75.0], dtype=DTYPE)


class _NumericalAffineLeak(Mechanism):
    Mechanism.RANGE(g=G, e=E)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * (v - self.e)


class _ExplicitAffineLeak(Mechanism):
    Mechanism.RANGE(g=G, e=E)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")
    Mechanism.EXPLICIT("i")

    def i(self, v):
        return self.g * (v - self.e)


class _NumericalNonlinearCurrent(Mechanism):
    Mechanism.RANGE(scale=2.0e-6)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")

    def i(self, v):
        return self.scale * v**2


class _AnalyticNonlinearCurrent(Mechanism):
    Mechanism.RANGE(scale=2.0e-6)
    Mechanism.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.scale * v**2

    def i_with_conductance(self, v):
        return self.i(v), 2.0 * self.scale * v


class _SavedNumericalNonlinearCurrent(_NumericalNonlinearCurrent):
    Mechanism.SAVE_CURRENT("i")


class _ScalarExplicitCurrent(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.EXPLICIT("i")

    def i(self, v):
        return 1.25


class _AffineOverrideWithoutAssertion(_NumericalAffineLeak):
    def i(self, v):
        return self.g * (v - self.e) ** 2


class _AffineOverrideWithAssertion(_NumericalAffineLeak):
    Mechanism.AFFINE("i")

    def i(self, v):
        return 2.0 * self.g * (v - self.e)


class _HigherPriorityNonAffineMethod(Mechanism):
    def i(self, v):
        return self.g * (v - self.e) ** 2


class _HigherPriorityAliasedMethod(Mechanism):
    # Even the same function object is a new method-ownership boundary unless
    # this class explicitly adopts the affine assertion.
    i = _NumericalAffineLeak.i


class _MultipleInheritanceNonAffineWins(
    _HigherPriorityNonAffineMethod, _NumericalAffineLeak
):
    pass


class _MultipleInheritanceAffineWins(
    _NumericalAffineLeak, _HigherPriorityNonAffineMethod
):
    pass


class _MultipleInheritanceAliasedMethodWins(
    _HigherPriorityAliasedMethod, _NumericalAffineLeak
):
    pass


_RenamedAffineLeak = _NumericalAffineLeak.rename("renamed_numerical_affine_leak")
_RenamedMultipleInheritanceAffineWins = _MultipleInheritanceAffineWins.rename(
    "renamed_multiple_inheritance_affine_wins"
)
_RenamedExp2Syn = exp2syn.rename("renamed_df_exp2syn")
_HooklessRenamedExp2Syn = exp2syn.rename("hookless_renamed_df_exp2syn")
delattr(_HooklessRenamedExp2Syn, "i_with_conductance")


def _mechanism(mechanism_cls, shape=(1, 5)):
    return mechanism_cls(
        mechanism_cls.__name__,
        torch.full(shape, 37.0, dtype=DTYPE),
        torch.ones(shape, dtype=DTYPE),
        shape,
        shape,
        dtype=DTYPE,
    )


def _new_cable(mechanism_cls, integrator_kind, *, initialize=True):
    integrator = dn.dfh() if integrator_kind == "homogeneous" else dn.df()
    model = dn.Population(
        N=1,
        C=INITIAL_V.numel(),
        integrator=integrator,
        v_init=INITIAL_V,
        cm=1.0,
        rhoa=100.0,
        dtype=DTYPE,
    )
    model.diam.fill_(1.0)
    model.dx.fill_(200.0)
    model.insert(mechanism_cls, g=G, e=E)
    if initialize:
        model.initialize()
    return model


def _exact_semidiscrete_cable_trajectory(model):
    """Solve the physical compartment ODE with a dense matrix exponential.

    This oracle is assembled from geometry, capacitance, axial resistance, and
    the authored membrane current. It deliberately does not use any Dufort
    coefficient or production stencil helper.
    """
    diam_cm = model.diam[0] / 10_000.0
    dx_cm = model.dx[0] / 10_000.0
    area = torch.pi * diam_cm * dx_cm
    capacitance = model.cm[0] / 1000.0 * area
    resistance = model.rhoa[0] * dx_cm / (torch.pi * (diam_cm / 2.0) ** 2)
    axial_rate = 1.0 / (capacitance * resistance)
    leak_rate = G * area / capacitance

    compartments = model.nc
    generator = torch.zeros((compartments, compartments), dtype=DTYPE)
    for index in range(compartments):
        left = 1 if index == 0 else index - 1
        right = compartments - 2 if index == compartments - 1 else index + 1
        generator[index, index] -= axial_rate[index] * 2.0 + leak_rate[index]
        generator[index, left] += axial_rate[index]
        generator[index, right] += axial_rate[index]

    displacement = INITIAL_V - E
    return torch.stack(
        [
            E + torch.matrix_exp(generator * (step * DT)) @ displacement
            for step in range(STEPS + 1)
        ]
    ).unsqueeze(1)


def _run_trajectory(model, *, steps=STEPS, dt=DT):
    trace = [model.v.detach().clone()]
    for _ in range(steps):
        dn.step(model, dt=dt)
        trace.append(model.v.detach().clone())
    return torch.stack(trace)


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value


@pytest.mark.parametrize("integrator_kind", ["homogeneous", "heterogeneous"])
@pytest.mark.parametrize(
    "mechanism_cls,mode,factorable",
    [
        (_NumericalAffineLeak, "numerical-declared", True),
        (_ExplicitAffineLeak, "explicit", False),
    ],
)
def test_dufort_complete_cable_trajectory_matches_dense_continuous_oracle(
    integrator_kind, mechanism_cls, mode, factorable
):
    model = _new_cable(mechanism_cls, integrator_kind)
    mechanism = next(iter(model.mech.mechanisms.values()))
    expected = _exact_semidiscrete_cable_trajectory(model)

    actual = _run_trajectory(model)

    assert mechanism._current_conductance_mode == {"i": mode}
    assert mechanism._current_factorable == {"i": factorable}
    # The one-step Euler starter has O(dt^2) local error; the subsequent
    # two-level trajectory remains within that independently bounded error.
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-3)


def test_affine_assertion_is_not_inherited_across_a_current_override_but_rename_is():
    assert _NumericalAffineLeak._affine == {"i"}
    assert _RenamedAffineLeak._affine == {"i"}
    assert _AffineOverrideWithoutAssertion._affine == set()
    assert _AffineOverrideWithAssertion._affine == {"i"}

    assert _mechanism(_NumericalAffineLeak)._current_factorable == {"i": True}
    assert _mechanism(_RenamedAffineLeak)._current_factorable == {"i": True}
    assert _mechanism(_AffineOverrideWithoutAssertion)._current_factorable == {
        "i": False
    }
    assert _mechanism(_AffineOverrideWithAssertion)._current_factorable == {"i": True}


@pytest.mark.parametrize(
    "mechanism_cls,expected_owner,expected_assertion,expected_factorable",
    [
        (
            _MultipleInheritanceNonAffineWins,
            _HigherPriorityNonAffineMethod,
            False,
            False,
        ),
        (_MultipleInheritanceAffineWins, _NumericalAffineLeak, True, True),
        (
            _MultipleInheritanceAliasedMethodWins,
            _HigherPriorityAliasedMethod,
            False,
            True,
        ),
        (
            _RenamedMultipleInheritanceAffineWins,
            _NumericalAffineLeak,
            True,
            True,
        ),
    ],
)
def test_affine_assertion_follows_the_current_method_selected_by_the_mro(
    mechanism_cls, expected_owner, expected_assertion, expected_factorable
):
    assert (
        next(base for base in mechanism_cls.__mro__ if "i" in base.__dict__)
        is expected_owner
    )
    assert ("i" in mechanism_cls._affine) is expected_assertion
    assert _mechanism(mechanism_cls)._current_factorable == {"i": expected_factorable}


def test_numerical_derivative_does_not_make_a_nonlinear_current_df_factorable():
    mechanism = _mechanism(_NumericalNonlinearCurrent)

    assert mechanism._current_conductance_mode == {"i": "numerical-declared"}
    assert mechanism._current_factorable == {"i": False}


def test_analytic_derivative_does_not_make_a_nonlinear_current_df_factorable():
    shape = (1, 5)
    mechanism = _mechanism(_AnalyticNonlinearCurrent, shape)
    handler = MechanismHandler(
        torch.full(shape, 37.0, dtype=DTYPE),
        torch.ones(shape, dtype=DTYPE),
        {"nonlinear": mechanism},
        currents={"nonspecific": {"nonlinear": ["i"]}},
    ).to(DTYPE)
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros(shape, dtype=DTYPE))
    voltage = torch.tensor([[-4.0, -2.0, 0.0, 1.0, 3.0]], dtype=DTYPE)

    current, conductance = handler.idf(voltage, torch.full_like(voltage, 40.0))

    assert mechanism._current_conductance_mode == {"i": "analytic"}
    assert mechanism._current_factorable == {"i": False}
    torch.testing.assert_close(current, mechanism.i(voltage))
    torch.testing.assert_close(conductance, torch.zeros_like(voltage))


@pytest.mark.parametrize(
    "mechanism_cls,mode",
    [
        (exp2syn, "analytic"),
        (_RenamedExp2Syn, "analytic"),
        (_HooklessRenamedExp2Syn, "symbolic"),
    ],
)
def test_exp2syn_variants_are_affine_for_dufort_classification(mechanism_cls, mode):
    mechanism = _mechanism(mechanism_cls)

    assert mechanism_cls._affine == {"i"}
    assert mechanism._current_conductance_mode == {"i": mode}
    assert mechanism._current_factorable == {"i": True}


def test_nonfactorable_idf_uses_current_voltage_zero_g_and_preserves_save_mirror():
    shape = (1, 5)
    mechanism = _mechanism(_SavedNumericalNonlinearCurrent, shape)
    handler = MechanismHandler(
        torch.full(shape, 37.0, dtype=DTYPE),
        torch.ones(shape, dtype=DTYPE),
        {"nonlinear": mechanism},
        currents={"nonspecific": {"nonlinear": ["i"]}},
    ).to(DTYPE)
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros(shape, dtype=DTYPE))
    voltage = torch.tensor([[-4.0, -2.0, 0.0, 1.0, 3.0]], dtype=DTYPE)
    previous = torch.full_like(voltage, 40.0)

    current, conductance = handler.idf(voltage, previous)
    expected = mechanism.i(voltage)

    torch.testing.assert_close(current, expected)
    torch.testing.assert_close(conductance, torch.zeros_like(voltage))
    torch.testing.assert_close(mechanism.i_, expected)


def test_nonfactorable_idf_preserves_scalar_explicit_current_compatibility():
    shape = (1, 5)
    mechanism = _mechanism(_ScalarExplicitCurrent, shape)
    handler = MechanismHandler(
        torch.full(shape, 37.0, dtype=DTYPE),
        torch.ones(shape, dtype=DTYPE),
        {"scalar": mechanism},
        currents={"nonspecific": {"scalar": ["i"]}},
    ).to(DTYPE)
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros(shape, dtype=DTYPE))

    current, conductance = handler.idf(
        torch.zeros(shape, dtype=DTYPE),
        torch.zeros(shape, dtype=DTYPE),
    )

    torch.testing.assert_close(current, torch.full(shape, 1.25, dtype=DTYPE))
    torch.testing.assert_close(conductance, torch.zeros(shape, dtype=DTYPE))


def test_dufort_checkpoint_restore_replays_a_started_two_level_suffix():
    source = _new_cable(_NumericalAffineLeak, "homogeneous")
    _run_trajectory(source, steps=3)
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())

    assert bool(checkpoint["integrator"]["_df_history_valid"])
    assert checkpoint["integrator"]["_df_history_dt"] == DT
    expected = _run_trajectory(source, steps=6)

    resumed = _new_cable(_NumericalAffineLeak, "homogeneous")
    resumed.restore_dict_from_checkpoint(checkpoint)
    assert bool(resumed._df_history_valid)
    actual = _run_trajectory(resumed, steps=6)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_dufort_history_state_batches_and_initializes_independently_of_voltage_shape():
    model = _new_cable(_NumericalAffineLeak, "homogeneous")
    model.batch(3)
    model.initialize()

    assert model._df_history_valid.shape == (3,)
    assert model._df_history_dt.shape == (3,)
    assert not bool(torch.any(model._df_history_valid))
    dn.step(model, dt=DT)
    assert bool(torch.all(model._df_history_valid))
    assert model.v.shape == (3, 1, INITIAL_V.numel())


@pytest.mark.parametrize("integrator_kind", ["homogeneous", "heterogeneous"])
def test_dufort_same_dt_training_rebuilds_preserve_two_level_history(integrator_kind):
    reference = _new_cable(_ExplicitAffineLeak, integrator_kind)
    training = _new_cable(_ExplicitAffineLeak, integrator_kind)
    training.train()

    expected = _run_trajectory(reference, steps=8)
    actual = _run_trajectory(training, steps=8)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert bool(training._df_history_valid)
    assert training._df_history_dt == DT


@pytest.mark.parametrize("integrator_kind", ["homogeneous", "heterogeneous"])
def test_dufort_euler_starter_honors_configured_spatial_smoothing(integrator_kind):
    plain = _new_cable(_ExplicitAffineLeak, integrator_kind, initialize=False)
    smoothed = _new_cable(_ExplicitAffineLeak, integrator_kind, initialize=False)
    plain._integrator_class = (
        dn.dfh(beta=1.0) if integrator_kind == "homogeneous" else dn.df(beta=1.0)
    )
    smoothed._integrator_class = (
        dn.dfh(beta=0.5) if integrator_kind == "homogeneous" else dn.df(beta=0.5)
    )
    plain.initialize(force_rebuild=True)
    smoothed.initialize(force_rebuild=True)

    dn.step(plain, dt=DT)
    dn.step(smoothed, dt=DT)

    values = plain.v
    reflected = torch.cat(
        [
            values[..., 1:3].flip(-1),
            values,
            values[..., -3:-1].flip(-1),
        ],
        dim=-1,
    )
    filtered = 0.25 * (
        reflected[..., :-4]
        + reflected[..., 1:-3]
        + reflected[..., 3:-1]
        + reflected[..., 4:]
    )
    expected = 0.5 * values + 0.5 * filtered
    torch.testing.assert_close(smoothed.v, expected, rtol=0.0, atol=1.0e-12)


def test_dufort_timestep_change_restarts_with_exactly_one_new_dt():
    model = dn.Population(
        N=1,
        C=1,
        integrator=dn.dfh(),
        v_init=-72.0,
        cm=1.0,
        rhoa=100.0,
        dtype=DTYPE,
    )
    model.insert(_NumericalAffineLeak, g=G, e=E)
    model.initialize()
    dn.step(model, dt=DT)

    before = model.v.detach().clone()
    new_dt = 0.02
    expected = before - new_dt * (1000.0 * G) * (before - E)
    dn.step(model, dt=new_dt)

    torch.testing.assert_close(model.v, expected, rtol=0.0, atol=1.0e-9)
    torch.testing.assert_close(model.v_prev, before, rtol=0.0, atol=0.0)
    assert bool(model._df_history_valid)
