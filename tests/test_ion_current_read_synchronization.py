"""Regression tests for ionic-current reads during mechanism state advance."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State

DTYPE = torch.float64
DT = 0.025


class _ThreeIonCurrent(Mechanism):
    Mechanism.ASSIGNED("stage_marker")
    Mechanism.RANGE(
        gna=0.01,
        gk=0.02,
        gca=0.03,
        na_curvature=1.0e-4,
        na_curve_origin=-65.0,
        ena=55.0,
        ek=-90.0,
        eca=120.0,
    )
    Mechanism.USEION("na", write=["ina"])
    Mechanism.USEION("k", write=["ik"])
    Mechanism.USEION("ca", write=["ica"])

    def assigned_values(self, v, values):
        del values
        return {"stage_marker": torch.zeros_like(v)}

    def ina(self, v):
        return (
            self.gna * (v - self.ena)
            + self.na_curvature * (v - self.na_curve_origin) ** 2
        )

    def ik(self, v):
        return self.gk * (v - self.ek)

    def ica(self, v):
        return self.gca * (v - self.eca)

    def ina_with_conductance(self, v):
        conductance = self.gna + 2 * self.na_curvature * (v - self.na_curve_origin)
        return self.ina(v), conductance

    def ik_with_conductance(self, v):
        return self.ik(v), self.gk

    def ica_with_conductance(self, v):
        return self.ica(v), self.gca


class _CurrentSnapshot(State):
    State.STATE("seen")
    State.DERIVATIVE("seen' = 0 * seen")

    def state_defaults(self, v, values):
        del values
        return {"seen": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        del v, dt
        return {"seen": values["ina"] + values["ik"] + values["ica"]}


class _ThreeIonCurrentReader(Mechanism):
    Mechanism.STATE_BUNDLE(_CurrentSnapshot)
    Mechanism.USEION("na", read=["ina"])
    Mechanism.USEION("k", read=["ik"])
    Mechanism.USEION("ca", read=["ica"])


class _CurrentReaderWriterCycle(Mechanism):
    Mechanism.RANGE(gk=0.01)
    Mechanism.USEION("na", read=["ina"])
    Mechanism.USEION("k", write=["ik"])

    def ik(self, v):
        return self.gk * v

    def ik_with_conductance(self, v):
        return self.ik(v), self.gk


class _AcceptedStepProbe(Mechanism):
    Mechanism.CARRY("steps")

    def initial_values(self, v, values):
        del values
        return {"steps": torch.zeros_like(v)}

    def advance(self, v, dt, values):
        del v, dt
        return {"steps": values["steps"] + 1}


def _model(*, with_breakpoint_probe=False, gna=None, integrator=None):
    with dn.ctx(DTYPE=DTYPE):
        model = dn.Population(
            N=1,
            C=4,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=integrator,
        )
        current_parameters = {} if gna is None else {"gna": gna}
        model.insert(_ThreeIonCurrent, **current_parameters)
        # A non-contiguous insertion exercises mechanism-local indexed views.
        model[:, 1::2].insert(_ThreeIonCurrentReader)
        if with_breakpoint_probe:
            model.insert(_AcceptedStepProbe)
        if gna is not None:
            model.train(True)
        model.initialize()
    return model


def _mechanism(model, cls):
    return next(
        mech for mech in model.mech.mechanisms.values() if isinstance(mech, cls)
    )


def _expected_currents(model, v):
    channel = _mechanism(model, _ThreeIonCurrent)
    reader = _mechanism(model, _ThreeIonCurrentReader)
    currents = {
        "ina": reader.get(channel.ina(v)),
        "ik": reader.get(channel.ik(v)),
        "ica": reader.get(channel.ica(v)),
    }
    return currents, sum(currents.values())


def _full_currents(model, v):
    channel = _mechanism(model, _ThreeIonCurrent)
    return {
        "ina": channel.ina(v),
        "ik": channel.ik(v),
        "ica": channel.ica(v),
    }


def _assert_committed_currents(model, expected, *, assert_seen=True):
    reader = _mechanism(model, _ThreeIonCurrentReader)
    state = next(iter(reader.DE.values()))
    local_expected = {}
    for ion, current_name in (("na", "ina"), ("k", "ik"), ("ca", "ica")):
        expected_current = expected[current_name]
        local_expected[current_name] = reader.get(expected_current)
        torch.testing.assert_close(
            model.mech.ions[ion]._buffers[current_name], expected_current
        )
        torch.testing.assert_close(
            getattr(reader, current_name), local_expected[current_name]
        )
        torch.testing.assert_close(
            getattr(state, current_name), local_expected[current_name]
        )
    if assert_seen:
        torch.testing.assert_close(reader.seen, sum(local_expected.values()))


def _explicit_oracle(model, integrator_name):
    v0 = model.v.detach().clone()
    dt = torch.as_tensor(DT, dtype=v0.dtype, device=v0.device)
    # The explicit workspace is lazily initialized by the first public step.
    # In an isolated uniform cable, membrane area cancels and the authored
    # density-current conversion is exactly 1000 / cm.
    voltage_scale = 1000.0 / model.cm.expand_as(v0)

    def stage(v):
        currents = _full_currents(model, v)
        slope = -voltage_scale * sum(currents.values())
        return slope, currents

    k1, i1 = stage(v0)
    if integrator_name == "euler":
        return v0 + dt * k1, i1

    k2, i2 = stage(v0 + 0.5 * dt * k1)
    if integrator_name == "rk2":
        return v0 + dt * k2, i2

    k3, i3 = stage(v0 + 0.5 * dt * k2)
    k4, i4 = stage(v0 + dt * k3)
    voltage = v0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    accepted = {
        name: (i1[name] + 2 * i2[name] + 2 * i3[name] + i4[name]) / 6 for name in i1
    }
    return voltage, accepted


def _reset_evaluation_count(model):
    channel = _mechanism(model, _ThreeIonCurrent)
    channel._test_evaluation_calls = 0
    evaluate_assigned = channel._evaluate_assigned

    def counted_evaluation(*args, **kwargs):
        channel._test_evaluation_calls += 1
        return evaluate_assigned(*args, **kwargs)

    channel._evaluate_assigned = counted_evaluation
    return channel


def _committed_current_snapshot(model):
    reader = _mechanism(model, _ThreeIonCurrentReader)
    state = next(iter(reader.DE.values()))
    snapshot = {"seen": reader.seen.detach().clone()}
    for ion, current_name in (("na", "ina"), ("k", "ik"), ("ca", "ica")):
        snapshot[f"ion.{current_name}"] = (
            model.mech.ions[ion]._buffers[current_name].detach().clone()
        )
        snapshot[f"reader.{current_name}"] = (
            getattr(reader, current_name).detach().clone()
        )
        snapshot[f"state.{current_name}"] = (
            getattr(state, current_name).detach().clone()
        )
    return snapshot


def test_initialize_commits_coherent_full_and_indexed_ion_currents():
    model = _model()

    expected = _full_currents(model, model.v)
    _assert_committed_currents(model, expected, assert_seen=False)


@pytest.mark.parametrize(
    ("integrator_name", "expected_evaluations"),
    [
        ("euler", 1),
        ("rk2", 2),
        ("rk4", 4),
        ("bwd_euler_sc", 1),
    ],
)
def test_solver_step_evaluates_each_current_source_once_per_required_stage(
    integrator_name,
    expected_evaluations,
):
    model = _model(integrator=getattr(dn, integrator_name)())
    channel = _reset_evaluation_count(model)

    model.step(dt=DT)

    assert channel._test_evaluation_calls == expected_evaluations


@pytest.mark.parametrize(
    ("integrator_name", "expected_evaluations"),
    [("euler", 1), ("rk2", 2), ("rk4", 4)],
)
def test_explicit_solver_commits_the_runge_kutta_weighted_ionic_flux(
    integrator_name,
    expected_evaluations,
):
    model = _model(integrator=getattr(dn, integrator_name)())
    v0 = model.v.detach().clone()
    expected_v, accepted = _explicit_oracle(model, integrator_name)
    channel = _reset_evaluation_count(model)

    model.step(dt=DT)

    torch.testing.assert_close(model.v, expected_v, rtol=2.0e-12, atol=2.0e-12)
    _assert_committed_currents(model, accepted)
    assert channel._test_evaluation_calls == expected_evaluations

    # With uniform voltage there is no axial current, so the accepted per-ion
    # quadrature must be exactly the flux that changed membrane voltage.
    accepted_total = sum(accepted.values())
    voltage_flux = (
        -(model.v - v0)
        * model.integrator.cm_c
        / (torch.as_tensor(DT, dtype=DTYPE) * model.integrator.area_c)
    )
    torch.testing.assert_close(
        accepted_total,
        voltage_flux,
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def test_implicit_solver_commits_the_linearized_endpoint_ionic_flux():
    model = _model(integrator=dn.bwd_euler_sc())
    v0 = model.v.detach().clone()
    reference = _full_currents(model, v0)
    channel = _mechanism(model, _ThreeIonCurrent)
    conductances = {
        "ina": channel.gna + 2 * channel.na_curvature * (v0 - channel.na_curve_origin),
        "ik": channel.gk,
        "ica": channel.gca,
    }
    total_conductance = sum(conductances.values())
    cmdt = (
        1.0e-3
        * model.cm.expand_as(v0)
        * model.cm_scale.expand_as(v0)
        / torch.as_tensor(DT, dtype=DTYPE)
    )
    expected_v = v0 - sum(reference.values()) / (cmdt + total_conductance)
    accepted = {
        name: current + conductances[name] * (expected_v - v0)
        for name, current in reference.items()
    }
    channel = _reset_evaluation_count(model)

    model.step(dt=DT)

    torch.testing.assert_close(model.v, expected_v, rtol=2.0e-12, atol=2.0e-12)
    _assert_committed_currents(model, accepted)
    # The nonlinear sodium endpoint differs from the affine current predicted
    # by the one evaluated current/conductance pair. This distinguishes correct
    # endpoint linearization from a hidden second current evaluation at v_new.
    assert not torch.allclose(_full_currents(model, expected_v)["ina"], accepted["ina"])
    assert channel._test_evaluation_calls == 1
    torch.testing.assert_close(
        sum(accepted.values()),
        -cmdt * (model.v - v0),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


@pytest.mark.parametrize("integrator_name", ["dfh", "df"])
def test_dufort_regular_step_reuses_its_centered_ionic_frame(
    integrator_name,
    monkeypatch,
):
    with dn.ctx(IMEM=1):
        model = _model(integrator=getattr(dn, integrator_name)())
    # The first call constructs the missing history level with Euler.
    model.step(dt=DT)

    v = model.v.detach().clone()
    v_prev = model.v_prev.detach().clone()
    channel = _reset_evaluation_count(model)

    def forbidden_itot(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Dufort must not perform a second current evaluation")

    monkeypatch.setattr(model.mech, "itot", forbidden_itot)
    model.step(dt=DT)

    accepted = {}
    for current_name in ("ina", "ik", "ica"):
        if channel._current_factorable[current_name]:
            current, conductance = getattr(channel, f"{current_name}_with_conductance")(
                0.5 * v_prev
            )
            accepted[current_name] = current + 0.5 * conductance * model.v
        else:
            accepted[current_name] = getattr(channel, current_name)(v)

    _assert_committed_currents(model, accepted)
    assert channel._test_evaluation_calls == 1
    expected_imem = (model.v - v_prev) / model.integrator.s1 + sum(
        accepted.values()
    ) * model.integrator.area
    torch.testing.assert_close(
        model.i_membrane,
        expected_imem,
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def test_missing_current_sources_publish_zero_to_full_and_indexed_readers():
    with dn.ctx(DTYPE=DTYPE):
        model = dn.Population(N=1, C=4, v_init=-65.0, dtype=DTYPE)
        model[:, 1::2].insert(_ThreeIonCurrentReader)
        model.initialize()

    reader = _mechanism(model, _ThreeIonCurrentReader)
    state = next(iter(reader.DE.values()))
    for ion, current_name in (("na", "ina"), ("k", "ik"), ("ca", "ica")):
        shared = model.mech.ions[ion]._buffers[current_name]
        model.mech.ions[ion]._buffers[current_name] = torch.full_like(shared, 7.0)

    model.mech.advance(model.v, torch.as_tensor(DT), model.celsius)

    torch.testing.assert_close(reader.seen, torch.zeros_like(reader.seen))
    for ion, current_name in (("na", "ina"), ("k", "ik"), ("ca", "ica")):
        shared = model.mech.ions[ion]._buffers[current_name]
        torch.testing.assert_close(shared, torch.zeros_like(shared))
        torch.testing.assert_close(
            getattr(reader, current_name),
            torch.zeros_like(getattr(reader, current_name)),
        )
        torch.testing.assert_close(
            getattr(state, current_name), torch.zeros_like(getattr(state, current_name))
        )


def test_current_reader_writer_dependency_cycle_is_rejected():
    with dn.ctx(DTYPE=DTYPE):
        model = dn.Population(N=1, C=1, v_init=-65.0, dtype=DTYPE)
        model.insert(_CurrentReaderWriterCycle)
        with pytest.raises(ValueError, match="READ iion"):
            model.initialize()


def test_current_reads_use_step_voltage_not_last_current_evaluation():
    model = _model(with_breakpoint_probe=True)
    handler = model.mech
    reader = _mechanism(model, _ThreeIonCurrentReader)
    step_probe = _mechanism(model, _AcceptedStepProbe)
    state = next(iter(reader.DE.values()))

    accepted_v = torch.tensor([[-80.0, -70.0, -60.0, -50.0]], dtype=DTYPE)
    distractor_v = torch.tensor([[20.0, 25.0, 30.0, 35.0]], dtype=DTYPE)
    handler.iexp(distractor_v)
    accepted_steps = step_probe.steps.clone()
    distractor = sum(
        reader.get(handler.ions[ion]._buffers[f"i{ion}"]) for ion in ("na", "k", "ca")
    )
    expected_currents, expected_total = _expected_currents(model, accepted_v)
    assert not torch.allclose(distractor, expected_total)

    handler.advance(accepted_v, torch.as_tensor(DT, dtype=DTYPE), model.celsius)
    torch.testing.assert_close(step_probe.steps, accepted_steps + 1)

    torch.testing.assert_close(reader.seen, expected_total)
    for current_name, expected in expected_currents.items():
        torch.testing.assert_close(getattr(reader, current_name), expected)
        torch.testing.assert_close(getattr(state, current_name), expected)


def test_diagnostics_do_not_overwrite_committed_frame_or_coupled_trajectory():
    baseline = _model()
    with_diagnostics = _model()

    for step_index in range(4):
        distractor_v = torch.full_like(with_diagnostics.v, 30.0 + 10.0 * step_index)
        committed_before = _committed_current_snapshot(with_diagnostics)
        with_diagnostics.mech.i(distractor_v)
        with_diagnostics.mech.iexp(distractor_v + 5.0)
        with_diagnostics.mech.idf(distractor_v + 10.0, distractor_v - 10.0)
        with_diagnostics.mech.itot(distractor_v + 15.0)
        committed_after = _committed_current_snapshot(with_diagnostics)
        for name, expected in committed_before.items():
            torch.testing.assert_close(
                committed_after[name], expected, rtol=0.0, atol=0.0
            )

        baseline.step(dt=DT)
        with_diagnostics.step(dt=DT)

        torch.testing.assert_close(with_diagnostics.v, baseline.v)
        torch.testing.assert_close(
            _mechanism(with_diagnostics, _ThreeIonCurrentReader).seen,
            _mechanism(baseline, _ThreeIonCurrentReader).seen,
        )
        for ion, current_name in (("na", "ina"), ("k", "ik"), ("ca", "ica")):
            torch.testing.assert_close(
                with_diagnostics.mech.ions[ion]._buffers[current_name],
                baseline.mech.ions[ion]._buffers[current_name],
            )


def test_current_read_refresh_is_dynamo_compatible():
    eager = _model()
    eager.run(tstop=2 * DT, dt=DT)

    # The eager backend exercises Dynamo graph capture without requiring a
    # platform compiler, keeping this regression portable across CPU CI hosts.
    with dn.ctx(JIT=1, BACKEND="eager", FULLGRAPH=0):
        compiled = _model()
        compiled.run(tstop=2 * DT, dt=DT)

    assert compiled.integrator._compiled_kernels
    torch.testing.assert_close(compiled.v, eager.v)
    torch.testing.assert_close(
        _mechanism(compiled, _ThreeIonCurrentReader).seen,
        _mechanism(eager, _ThreeIonCurrentReader).seen,
    )


def test_current_read_refresh_preserves_multistep_autograd():
    gna = torch.nn.Parameter(torch.tensor(0.01, dtype=DTYPE))
    model = _model(gna=gna)
    model.step(dt=DT)
    model.step(dt=DT)

    reader = _mechanism(model, _ThreeIonCurrentReader)
    loss = model.v.sum() + reader.seen.sum()
    loss.backward()

    assert gna.grad is not None
    assert torch.isfinite(gna.grad).all()
    assert gna.grad.abs().item() > 0.0
