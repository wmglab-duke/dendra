"""Holistic correctness checks for an ion used through the Material interface.

This file intentionally tests only coupling that Dendra exposes today.  An
``Ion`` is a ``Material``, so local mechanisms and full-field material processes
may share its concentration fields.  Dendra does *not* automatically turn an
ionic current into a concentration derivative; ``_PotassiumFlux`` therefore
declares the Faraday conversion explicitly, as a real concentration mechanism
would.
"""

from __future__ import annotations

import math

import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._material_process import (
    ClearanceProcess,
    DiffusionProcess,
)

DTYPE = torch.float64
N_COMPARTMENTS = 4
TEMPERATURE_C = 34.0

# Physical constants copied independently from the implementation under test.
FARADAY_C_PER_MOL = 96485.33212331001
GAS_CONSTANT_MV_C_PER_MOL_K = 1.0e3 * 8.31446261815324

CONDUCTANCE = 2.0e-3  # S/cm² (equivalently 2 mS/cm²)
CLEARANCE_RATE = 5.0e-2  # 1 / ms
CLEARANCE_TARGET = 52.0  # mM
DIFFUSIVITY = 0.4  # um^2 / ms

DIAMETER_UM = torch.tensor([2.0, 3.0, 1.5, 2.5], dtype=DTYPE)
DX_UM = torch.tensor([1.0, 1.2, 0.8, 1.4], dtype=DTYPE)
CM = torch.tensor([0.9, 1.1, 1.0, 1.2], dtype=DTYPE)
INITIAL_V = torch.tensor([-52.0, -47.0, -60.0, -55.0], dtype=DTYPE)
INITIAL_KI = torch.tensor([61.0, 44.0, 72.0, 49.0], dtype=DTYPE)
INITIAL_KO = torch.tensor([3.0, 4.5, 2.5, 5.0], dtype=DTYPE)


def _geometry():
    cross_section_um2 = torch.pi * (0.5 * DIAMETER_UM) ** 2
    volume_um3 = cross_section_um2 * DX_UM
    membrane_area_cm2 = torch.pi * DIAMETER_UM * DX_UM * 1.0e-8
    edge_area_um2 = 0.5 * (cross_section_um2[:-1] + cross_section_um2[1:])
    edge_length_um = 0.5 * (DX_UM[:-1] + DX_UM[1:])
    diffusion_geometry_um = edge_area_um2 / edge_length_um
    return volume_um3, membrane_area_cm2, diffusion_geometry_um


VOLUME_UM3, MEMBRANE_AREA_CM2, DIFFUSION_GEOMETRY_UM = _geometry()

# For an outward-positive current density i (mA/cm^2), Faraday's law gives
#
#   d c_i / dt = -i A 10^12 / (z F V)
#
# in mM/ms when A is in cm^2 and V is in um^3.  Potassium has z=+1.
CURRENT_TO_CONCENTRATION = MEMBRANE_AREA_CM2 * 1.0e12 / (FARADAY_C_PER_MOL * VOLUME_UM3)


class _PotassiumFlux(Mechanism):
    """Ohmic K current plus an explicit current-to-concentration conversion."""

    Mechanism.RANGE(g=CONDUCTANCE, current_to_concentration=1.0)
    Mechanism.USEION("k", write=["ik"])
    # USEMATERIAL permits a concentration field to be both read and written.
    # Because registered ions are Material subclasses, this is the same shared
    # k.ki/k.ko/k.ek state used by USEION above.
    Mechanism.USEMATERIAL("k", read=["ki", "ko", "ek"], write=["ki"])

    def advance(self, v, dt, values):
        outward_current = self.g * (v - values["ek"])
        return {
            "ki": values["ki"] - dt * self.current_to_concentration * outward_current
        }

    def ik(self, v):
        return self.g * (v - self.ek)

    def ik_with_conductance(self, v):
        return self.ik(v), self.g


class _PotassiumClearance(ClearanceProcess):
    ClearanceProcess.GLOBAL(rate=CLEARANCE_RATE, target=CLEARANCE_TARGET)
    ClearanceProcess.CLEAR("k", field="ki", rate="rate", target="target")


class _PotassiumDiffusion(DiffusionProcess):
    DiffusionProcess.GLOBAL(D=DIFFUSIVITY)
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE("k", field="ki", D="D", domain="intracellular")


def _model(*, conductance=CONDUCTANCE, training=False):
    # Mechanism parameters are constructed lazily during initialize(), so keep
    # Dendra's construction context aligned with the population's requested
    # dtype as well as passing dtype= below.
    with dn.ctx(DTYPE=DTYPE):
        model = dn.Population(
            N=1,
            C=N_COMPARTMENTS,
            integrator=dn.bwd_euler_sc(),
            v_init=INITIAL_V,
            cm=CM,
            celsius=TEMPERATURE_C,
            dtype=DTYPE,
        )
        model.diam.copy_(DIAMETER_UM.reshape_as(model.diam))
        model.dx.copy_(DX_UM.reshape_as(model.dx))
        model.concentrations(
            ki0=INITIAL_KI.reshape(1, -1),
            ko0=INITIAL_KO.reshape(1, -1),
        )
        model.insert(
            _PotassiumFlux,
            g=torch.as_tensor(conductance, dtype=DTYPE),
            current_to_concentration=CURRENT_TO_CONCENTRATION.reshape(1, -1),
        )
        # Insertion order and declared phases define the intentional Lie split:
        # local current flux -> clearance -> diffusion -> Nernst refresh.
        model.insert(
            _PotassiumClearance,
            rate=torch.as_tensor(CLEARANCE_RATE, dtype=DTYPE),
            target=torch.as_tensor(CLEARANCE_TARGET, dtype=DTYPE),
        )
        model.insert(
            _PotassiumDiffusion,
            D=torch.as_tensor(DIFFUSIVITY, dtype=DTYPE),
        )
        model.train(training)
        model.initialize()
    return model


def _nernst(ki, ko):
    thermal_factor = (
        GAS_CONSTANT_MV_C_PER_MOL_K * (273.15 + TEMPERATURE_C) / FARADAY_C_PER_MOL
    )
    return torch.log(ko / ki) * thermal_factor


def _diffuse_implicit(ki, dt):
    """Independent sealed finite-volume backward-Euler diffusion solve."""
    flat = ki.reshape(-1, N_COMPARTMENTS)
    volume = VOLUME_UM3.to(ki).reshape(1, -1)
    matrix = torch.diag_embed(volume.expand(flat.shape[0], -1)).clone()
    for edge, geometric_factor in enumerate(DIFFUSION_GEOMETRY_UM):
        coupling = torch.as_tensor(DIFFUSIVITY, dtype=ki.dtype, device=ki.device)
        coupling = coupling * geometric_factor.to(ki)
        left, right = edge, edge + 1
        matrix[:, left, left] += dt * coupling
        matrix[:, right, right] += dt * coupling
        matrix[:, left, right] -= dt * coupling
        matrix[:, right, left] -= dt * coupling
    rhs = volume * flat
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1).reshape_as(ki)


def _oracle_step(state, dt, *, conductance=CONDUCTANCE):
    v, ki, ko, ek = state
    g = torch.as_tensor(conductance, dtype=v.dtype, device=v.device)
    dt_t = torch.as_tensor(dt, dtype=v.dtype, device=v.device)

    # Handler phase order: local concentration write, exact clearance,
    # transport, Ion.advance (Nernst), then the implicit voltage solve.
    flux_current = g * (v - ek)
    ki_after_flux = ki - dt_t * CURRENT_TO_CONCENTRATION.to(v) * flux_current
    ki_after_clearance = CLEARANCE_TARGET + (
        ki_after_flux - CLEARANCE_TARGET
    ) * torch.exp(-CLEARANCE_RATE * dt_t)
    ki_new = _diffuse_implicit(ki_after_clearance, dt)
    ek_new = _nernst(ki_new, ko)

    cmdt = 1.0e-3 * CM.to(v).reshape_as(v) / dt_t
    v_new = (cmdt * v + g * ek_new) / (cmdt + g)
    # The implicit solver evaluates once at ``v`` and commits the affine
    # endpoint current that actually closes its linearized voltage balance.
    voltage_current = g * (v_new - ek_new)
    return (v_new, ki_new, ko, ek_new), {
        "flux_current": flux_current,
        "voltage_current": voltage_current,
        "ki_after_flux": ki_after_flux,
        "ki_after_clearance": ki_after_clearance,
    }


def _model_state(model):
    ion = model.mech.ions["k"]
    channel = next(
        mechanism
        for mechanism in model.mech.mechanisms.values()
        if isinstance(mechanism, _PotassiumFlux)
    )
    return {
        "v": model.v,
        "ki": ion.ki,
        "ko": ion.ko,
        "ek": ion.ek,
        "ik": ion.ik,
        "local_ki": channel.ki,
        "local_ek": channel.ek,
        "t": model.t,
    }


def _assert_runtime_close(actual, expected, *, atol=3.0e-11, rtol=3.0e-11):
    for field in ("v", "ki", "ko", "ek"):
        torch.testing.assert_close(actual[field], expected[field], atol=atol, rtol=rtol)
    torch.testing.assert_close(actual["local_ki"], expected["ki"], atol=atol, rtol=rtol)
    torch.testing.assert_close(actual["local_ek"], expected["ek"], atol=atol, rtol=rtol)


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


def test_ion_material_phase_order_matches_oracle_and_faraday_mass_balance():
    model = _model()
    ion = model.mech.ions["k"]
    assert model.mech._get_material("k") is ion
    assert [
        type(process)._material_process_phase
        for process in model.mech.material_processes.values()
    ] == ["post_local", "transport"]
    assert model.get_ion_style("k") == (1, 1)

    initial = _model_state(model)
    oracle_state = (initial["v"], initial["ki"], initial["ko"], initial["ek"])
    dt = 0.05
    oracle_state, intermediates = _oracle_step(oracle_state, dt)
    model.step(dt=dt)

    expected = dict(zip(("v", "ki", "ko", "ek"), oracle_state))
    actual = _model_state(model)
    _assert_runtime_close(actual, expected)
    torch.testing.assert_close(
        actual["ik"], intermediates["voltage_current"], atol=2.0e-12, rtol=2.0e-12
    )

    # The local ionic-current phase obeys Faraday's law before any clearance.
    initial_mass = (VOLUME_UM3.reshape_as(initial["ki"]) * initial["ki"]).sum()
    post_flux_mass = (
        VOLUME_UM3.reshape_as(initial["ki"]) * intermediates["ki_after_flux"]
    ).sum()
    expected_charge_mass = (
        -dt
        * (
            MEMBRANE_AREA_CM2.reshape_as(initial["ki"]) * intermediates["flux_current"]
        ).sum()
        * 1.0e12
        / FARADAY_C_PER_MOL
    )
    torch.testing.assert_close(
        post_flux_mass - initial_mass,
        expected_charge_mass,
        atol=2.0e-12,
        rtol=2.0e-12,
    )

    # Sealed diffusion preserves the weighted mass produced by the preceding
    # clearance phase, so the final field has an independent analytic total.
    post_clearance_mass = (
        VOLUME_UM3.reshape_as(initial["ki"]) * intermediates["ki_after_clearance"]
    ).sum()
    final_mass = (VOLUME_UM3.reshape_as(actual["ki"]) * actual["ki"]).sum()
    torch.testing.assert_close(
        final_mass, post_clearance_mass, atol=3.0e-11, rtol=3.0e-11
    )

    # The post-process reversal potential must be the Nernst potential of the
    # final concentration, and that refreshed value must affect voltage now.
    torch.testing.assert_close(actual["ek"], _nernst(actual["ki"], actual["ko"]))
    assert not torch.equal(actual["ek"], initial["ek"])
    assert torch.all(actual["v"] < initial["v"])


def test_fresh_checkpoint_replays_coupled_ion_material_suffix_exactly():
    source = _model()
    dt = 0.05
    for _ in range(3):
        source.step(dt=dt)
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())

    for _ in range(5):
        source.step(dt=dt)
    expected = {
        key: value.detach().clone() for key, value in _model_state(source).items()
    }

    resumed = _model()
    resumed.restore_dict_from_checkpoint(checkpoint)
    for _ in range(5):
        resumed.step(dt=dt)
    actual = _model_state(resumed)

    for field in expected:
        torch.testing.assert_close(actual[field], expected[field], atol=0.0, rtol=0.0)


def _run_to_time(dt, *, conductance=CONDUCTANCE):
    model = _model(conductance=conductance)
    n_steps = round(0.8 / dt)
    for _ in range(n_steps):
        model.step(dt=dt)
    state = _model_state(model)
    return torch.cat(
        (state["v"].reshape(-1), state["ki"].reshape(-1), state["ek"].reshape(-1))
    )


def test_coupled_ion_material_solution_improves_under_dt_refinement():
    reference = _run_to_time(0.00625)
    coarse = _run_to_time(0.1)
    fine = _run_to_time(0.05)

    coarse_error = torch.linalg.vector_norm(coarse - reference)
    fine_error = torch.linalg.vector_norm(fine - reference)
    assert fine_error < 0.65 * coarse_error


def _oracle_loss(conductance):
    v = INITIAL_V.reshape(1, -1).clone()
    ki = INITIAL_KI.reshape(1, -1).clone()
    ko = INITIAL_KO.reshape(1, -1).clone()
    state = (v, ki, ko, _nernst(ki, ko))
    for _ in range(5):
        state, _ = _oracle_step(state, 0.05, conductance=conductance)
    v, ki, _, ek = state
    return 0.01 * v.square().sum() + 0.2 * ki.square().sum() + 0.05 * ek.sum()


def test_ion_current_concentration_reversal_voltage_gradient_matches_fd():
    model = _model(training=True)
    channel = next(
        mechanism
        for mechanism in model.mech.mechanisms.values()
        if isinstance(mechanism, _PotassiumFlux)
    )
    conductance = torch.full_like(channel.g, CONDUCTANCE, requires_grad=True)
    channel._buffers["g"] = conductance

    for _ in range(5):
        model.step(dt=0.05)
    state = _model_state(model)
    loss = (
        0.01 * state["v"].square().sum()
        + 0.2 * state["ki"].square().sum()
        + 0.05 * state["ek"].sum()
    )
    gradient = torch.autograd.grad(loss, conductance)[0].sum()

    eps = 1.0e-6
    expected = (_oracle_loss(CONDUCTANCE + eps) - _oracle_loss(CONDUCTANCE - eps)) / (
        2.0 * eps
    )
    assert math.isfinite(float(gradient))
    torch.testing.assert_close(gradient, expected, atol=2.0e-4, rtol=2.0e-6)
