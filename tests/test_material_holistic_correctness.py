"""Holistic correctness oracles for material-coupled simulations.

These tests deliberately assemble the finite-volume material and membrane
systems independently.  They exercise the public Population/Tree lifecycle so
phase ordering, local/full-field synchronization, transport, voltage coupling,
chunking, checkpointing, batching, and autograd are checked together.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._material_process import (
    ClearanceProcess,
    DiffusionProcess,
    ExchangeProcess,
)

DTYPE = torch.float64
C = 4
DT = 0.0625
STEPS = 6

G_CURRENT = 2.0**-10
E_CURRENT = -42.0
PRODUCTION = 0.125
DIFFUSIVITY = 0.125
CLEARANCE_RATE = 0.0625
EXCHANGE_RATE = 0.125

INITIAL_V = torch.tensor([-68.0, -64.0, -61.0, -66.0], dtype=DTYPE)
INITIAL_SOLUTE = torch.tensor([1.4, 0.2, 0.7, 0.1], dtype=DTYPE)
INITIAL_RESERVE = torch.tensor([0.1, 0.9, 0.3, 1.1], dtype=DTYPE)
CHAIN_DIAM = torch.tensor([1.8, 2.2, 1.6, 2.0], dtype=DTYPE)
CHAIN_DX = torch.tensor([1.0, 1.4, 0.8, 1.2], dtype=DTYPE)
CHAIN_CM = torch.tensor([0.9, 1.1, 0.8, 1.2], dtype=DTYPE)


class _MaterialWritingCurrent(Mechanism):
    """Local reaction followed by a material-dependent membrane current."""

    Mechanism.GLOBAL(g=G_CURRENT, e=E_CURRENT, production=PRODUCTION)
    Mechanism.USEMATERIAL("solute", read=["c"], write=["c"])
    Mechanism.NONSPECIFIC_CURRENT("i")

    def _advance(self, v, dt):
        # This is a local reaction/write. The handler must publish it before
        # clearance/exchange/transport and then resynchronize the final field.
        self.c = self.c + dt * self.production

    def i(self, v):
        conductance = self.g * self.c
        return conductance * (v - self.e)

    def i_with_conductance(self, v):
        conductance = self.g * self.c
        return conductance * (v - self.e), conductance


class _ExactClearance(ClearanceProcess):
    ClearanceProcess.GLOBAL(rate=CLEARANCE_RATE)
    ClearanceProcess.CLEAR("solute", field="c", rate="rate", target=0.0)


class _ExactExchange(ExchangeProcess):
    ExchangeProcess.GLOBAL(rate=EXCHANGE_RATE)
    ExchangeProcess.EXCHANGE("solute.c", "reserve.c", rate="rate")


class _ImplicitDiffusion(DiffusionProcess):
    DiffusionProcess.GLOBAL(D=DIFFUSIVITY)
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE("solute", field="c", D="D")


class _ExplicitDiffusion(DiffusionProcess):
    DiffusionProcess.GLOBAL(D=DIFFUSIVITY)
    DiffusionProcess.METHOD("explicit", solver="dense")
    DiffusionProcess.DIFFUSE("solute", field="c", D="D")


@dataclass(frozen=True)
class _OracleGeometry:
    volume: torch.Tensor
    diffusion_edges: tuple[tuple[int, int, float], ...]
    cm: torch.Tensor
    area: torch.Tensor | None = None
    axial_edges: tuple[tuple[int, int, float], ...] = ()

    @property
    def is_tree(self):
        return self.area is not None


def _chain_geometry() -> _OracleGeometry:
    cross_section = torch.pi * (0.5 * CHAIN_DIAM) ** 2
    volume = cross_section * CHAIN_DX
    edge_area = 0.5 * (cross_section[:-1] + cross_section[1:])
    edge_length = 0.5 * (CHAIN_DX[:-1] + CHAIN_DX[1:])
    diffusion_edges = tuple(
        (index, index + 1, float(edge_area[index] / edge_length[index]))
        for index in range(C - 1)
    )
    return _OracleGeometry(volume, diffusion_edges, CHAIN_CM)


TREE_VOLUME = torch.tensor([1.2, 0.8, 1.5, 1.0], dtype=DTYPE)
TREE_AREA_UM2 = torch.tensor([11.0, 8.0, 14.0, 10.0], dtype=DTYPE)
TREE_AREA_CM2 = TREE_AREA_UM2 * 1.0e-8
TREE_CM = torch.tensor([1.0, 0.9, 1.2, 0.8], dtype=DTYPE)
TREE_EDGES = ((0, 1), (0, 2), (2, 3))
TREE_DIFF_GEOM = (0.65, 0.45, 0.80)
TREE_RESISTANCE = (8.0e7, 1.1e8, 9.0e7)


def _tree_graph():
    graph = nx.DiGraph()
    for node in range(C):
        graph.add_node(
            node,
            name=("Cell.soma[0](0.5)" if node == 0 else f"Cell.dend[{node}](0.5)"),
            L=1.0 + 0.2 * node,
            diam=1.0 + 0.1 * node,
            Ra=90.0 + 5.0 * node,
            cm=float(TREE_CM[node]),
            area=float(TREE_AREA_UM2[node]),
            volume=float(TREE_VOLUME[node]),
            volume_i=float(TREE_VOLUME[node]),
            volume_o=0.0,
            x=float(node),
            y=0.0,
            z=0.0,
        )
    for (parent, child), geom, resistance in zip(
        TREE_EDGES, TREE_DIFF_GEOM, TREE_RESISTANCE
    ):
        graph.add_edge(
            parent,
            child,
            diff_geom_um=float(geom),
            R_ohm=float(resistance),
            L=1.0,
        )
    return graph


def _tree_geometry() -> _OracleGeometry:
    diffusion_edges = tuple(
        (parent, child, geom)
        for (parent, child), geom in zip(TREE_EDGES, TREE_DIFF_GEOM)
    )
    axial_edges = tuple(
        (parent, child, 1.0 / resistance)
        for (parent, child), resistance in zip(TREE_EDGES, TREE_RESISTANCE)
    )
    return _OracleGeometry(
        TREE_VOLUME,
        diffusion_edges,
        TREE_CM,
        area=TREE_AREA_CM2,
        axial_edges=axial_edges,
    )


def _expanded_profile(profile, shape, *, scales=None):
    value = profile.reshape((1,) * (len(shape) - 1) + (C,)).expand(shape).clone()
    if scales is not None:
        scale = torch.as_tensor(scales, dtype=profile.dtype).reshape(
            len(scales), *([1] * (len(shape) - 1))
        )
        value = value * scale
    return value


def _insert_material_system(
    model,
    diffusion_cls,
    *,
    initial_scale=1.0,
    production=PRODUCTION,
    diffusivity=DIFFUSIVITY,
    clearance_rate=CLEARANCE_RATE,
    exchange_rate=EXCHANGE_RATE,
    batch_scales=None,
):
    def scalar(value):
        return torch.as_tensor(value, dtype=DTYPE)

    solute = _expanded_profile(INITIAL_SOLUTE * initial_scale, model.shape)
    reserve = _expanded_profile(INITIAL_RESERVE, model.shape)
    if batch_scales is not None:
        solute = solute * torch.as_tensor(batch_scales, dtype=DTYPE).reshape(
            len(batch_scales), *([1] * (len(model.shape) - 1))
        )
        reserve = reserve * torch.as_tensor(
            list(reversed(batch_scales)), dtype=DTYPE
        ).reshape(len(batch_scales), *([1] * (len(model.shape) - 1)))

    model.material(
        "solute",
        fields={"c": solute},
        min_values={"c": 0.0},
        domain="intracellular",
    )
    model.material(
        "reserve",
        fields={"c": reserve},
        min_values={"c": 0.0},
        domain="intracellular",
    )
    model.insert(
        _MaterialWritingCurrent,
        g=scalar(G_CURRENT),
        e=scalar(E_CURRENT),
        production=scalar(production),
    )
    # These two share post_local; insertion order is therefore semantically
    # observable and is part of the hand-stepped oracle below.
    model.insert(_ExactClearance, rate=scalar(clearance_rate))
    model.insert(_ExactExchange, rate=scalar(exchange_rate))
    model.insert(diffusion_cls, D=scalar(diffusivity))
    return model


def _population(
    diffusion_method="implicit",
    *,
    batch=None,
    training=False,
    initial_scale=1.0,
    production=PRODUCTION,
    diffusivity=DIFFUSIVITY,
    clearance_rate=CLEARANCE_RATE,
    exchange_rate=EXCHANGE_RATE,
):
    diffusion_cls = (
        _ImplicitDiffusion if diffusion_method == "implicit" else _ExplicitDiffusion
    )
    model = dn.Population(
        N=1,
        C=C,
        integrator=dn.bwd_euler_sc(),
        v_init=INITIAL_V,
        cm=CHAIN_CM,
        dtype=DTYPE,
    )
    model.diam.copy_(CHAIN_DIAM.reshape_as(model.diam))
    model.dx.copy_(CHAIN_DX.reshape_as(model.dx))
    batch_scales = None
    if batch is not None:
        model.batch(batch)
        batch_scales = torch.linspace(0.65, 1.15, batch, dtype=DTYPE)
    _insert_material_system(
        model,
        diffusion_cls,
        initial_scale=initial_scale,
        production=production,
        diffusivity=diffusivity,
        clearance_rate=clearance_rate,
        exchange_rate=exchange_rate,
        batch_scales=batch_scales,
    )
    model.train(training)
    model.initialize()
    return model


def _tree(
    *,
    training=False,
    initial_scale=1.0,
    production=PRODUCTION,
    diffusivity=DIFFUSIVITY,
    clearance_rate=CLEARANCE_RATE,
    exchange_rate=EXCHANGE_RATE,
):
    model = dn.Tree.from_graph(
        _tree_graph(),
        N=1,
        integrator=dn.dhs(threads=2),
        v_init=INITIAL_V,
        dtype=DTYPE,
    )
    _insert_material_system(
        model,
        _ImplicitDiffusion,
        initial_scale=initial_scale,
        production=production,
        diffusivity=diffusivity,
        clearance_rate=clearance_rate,
        exchange_rate=exchange_rate,
    )
    model.train(training)
    model.initialize()
    return model


def _material_state(model):
    current = next(
        mechanism
        for mechanism in model.mech.mechanisms.values()
        if isinstance(mechanism, _MaterialWritingCurrent)
    )
    return {
        "v": model.v.detach().clone(),
        "solute": model.mech.materials["solute"].c.detach().clone(),
        "reserve": model.mech.materials["reserve"].c.detach().clone(),
        "local_solute": current.c.detach().clone(),
        "t": model.t.detach().clone(),
    }


def _diffuse_oracle(c, geometry, dt, diffusivity, method):
    shape = c.shape
    flat = c.reshape(-1, C)
    volume = geometry.volume.to(c).reshape(1, C)
    net = torch.zeros_like(flat)
    matrix = torch.diag_embed(volume.expand(flat.shape[0], -1)).clone()
    for parent, child, geometric_factor in geometry.diffusion_edges:
        coupling = torch.as_tensor(diffusivity, dtype=c.dtype, device=c.device)
        coupling = coupling * geometric_factor
        flux = coupling * (flat[:, child] - flat[:, parent])
        net[:, parent] = net[:, parent] + flux
        net[:, child] = net[:, child] - flux
        matrix[:, parent, parent] += dt * coupling
        matrix[:, child, child] += dt * coupling
        matrix[:, parent, child] -= dt * coupling
        matrix[:, child, parent] -= dt * coupling

    if method == "explicit":
        return (flat + dt * net / volume).reshape(shape)
    rhs = volume * flat
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1).reshape(shape)


def _voltage_oracle(v, solute, geometry, dt):
    flat_v = v.reshape(-1, C)
    flat_c = solute.reshape(-1, C)
    cm = geometry.cm.to(v).reshape(1, C)
    conductance_density = G_CURRENT * flat_c

    if not geometry.is_tree:
        cmdt = 1e-6 * cm / (1e-3 * dt)
        return (
            (cmdt * flat_v + conductance_density * E_CURRENT)
            / (cmdt + conductance_density)
        ).reshape_as(v)

    area = geometry.area.to(v).reshape(1, C)
    cmdt = 1e-6 * cm * area / (1e-3 * dt)
    membrane_g = conductance_density * area
    matrix = torch.diag_embed(cmdt + membrane_g)
    for parent, child, axial_conductance in geometry.axial_edges:
        matrix[:, parent, parent] += axial_conductance
        matrix[:, child, child] += axial_conductance
        matrix[:, parent, child] -= axial_conductance
        matrix[:, child, parent] -= axial_conductance
    rhs = cmdt * flat_v + membrane_g * E_CURRENT
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1).reshape_as(v)


def _oracle_step(
    state,
    geometry,
    dt,
    diffusion_method,
    *,
    production=PRODUCTION,
    diffusivity=DIFFUSIVITY,
    clearance_rate=CLEARANCE_RATE,
    exchange_rate=EXCHANGE_RATE,
):
    v, solute, reserve = state

    # Handler order: local mechanism write -> exact clearance -> exact exchange
    # -> transport -> material guard/readback -> material-dependent voltage solve.
    solute = solute + dt * production
    solute = solute * torch.exp(
        torch.as_tensor(-clearance_rate * dt, dtype=solute.dtype)
    )

    mean = 0.5 * (solute + reserve)
    difference = (solute - reserve) * torch.exp(
        torch.as_tensor(-2.0 * exchange_rate * dt, dtype=solute.dtype)
    )
    solute = mean + 0.5 * difference
    reserve = mean - 0.5 * difference

    solute = _diffuse_oracle(solute, geometry, dt, diffusivity, diffusion_method)
    v = _voltage_oracle(v, solute, geometry, dt)
    return v, solute, reserve


def _initial_oracle_state(model):
    return (
        model.v.detach().clone(),
        model.mech.materials["solute"].c.detach().clone(),
        model.mech.materials["reserve"].c.detach().clone(),
    )


def _assert_state_close(actual, expected, *, atol=2e-11, rtol=2e-11):
    for name in ("v", "solute", "reserve"):
        torch.testing.assert_close(actual[name], expected[name], atol=atol, rtol=rtol)
    torch.testing.assert_close(
        actual["local_solute"], expected["solute"], atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("diffusion_method", ["implicit", "explicit"])
def test_population_material_voltage_trajectory_matches_hand_stepped_oracle(
    diffusion_method,
):
    model = _population(diffusion_method)
    geometry = _chain_geometry()
    oracle = _initial_oracle_state(model)

    process_phases = [
        type(process)._material_process_phase
        for process in model.mech.material_processes.values()
    ]
    assert process_phases == ["post_local", "post_local", "transport"]

    for step in range(1, STEPS + 1):
        oracle = _oracle_step(oracle, geometry, DT, diffusion_method)
        model.run(tstop=DT, dt=DT)
        expected = {
            "v": oracle[0],
            "solute": oracle[1],
            "reserve": oracle[2],
        }
        actual = _material_state(model)
        _assert_state_close(actual, expected)
        assert actual["t"].item() == pytest.approx(step * DT)
        assert torch.all(actual["solute"] >= 0)
        assert torch.all(actual["reserve"] >= 0)


def test_tree_material_voltage_trajectory_matches_independent_dense_system():
    model = _tree()
    geometry = _tree_geometry()
    oracle = _initial_oracle_state(model)

    for _ in range(STEPS):
        oracle = _oracle_step(oracle, geometry, DT, "implicit")
        model.run(tstop=DT, dt=DT)
        _assert_state_close(
            _material_state(model),
            {"v": oracle[0], "solute": oracle[1], "reserve": oracle[2]},
            # The production DHS kernel and this deliberately independent
            # dense solve accumulate their floating-point sums in different
            # orders.  Material fields still agree near machine precision;
            # allow the resulting few-ULP voltage difference.
            atol=5e-8,
            rtol=1e-9,
        )

    # The current is evaluated from the post-transport field, so spatially
    # varying concentration must measurably alter the complete voltage solve.
    assert model.v.max() - model.v.min() > 1e-3


@pytest.mark.parametrize("diffusion_method", ["implicit", "explicit"])
def test_exchange_and_diffusion_conserve_weighted_mass_and_positivity(
    diffusion_method,
):
    model = _population(
        diffusion_method,
        production=0.0,
        clearance_rate=0.0,
    )
    volume = _chain_geometry().volume.reshape(1, C)
    initial_mass = (
        volume * (model.mech.materials["solute"].c + model.mech.materials["reserve"].c)
    ).sum()

    model.run(tstop=STEPS * DT, dt=DT)

    final_solute = model.mech.materials["solute"].c
    final_reserve = model.mech.materials["reserve"].c
    final_mass = (volume * (final_solute + final_reserve)).sum()
    torch.testing.assert_close(final_mass, initial_mass, atol=2e-12, rtol=2e-12)
    assert torch.all(final_solute >= 0)
    assert torch.all(final_reserve >= 0)


def test_run_longrun_chunking_and_batching_have_identical_material_state():
    run_model = _population("implicit", batch=2)
    long_model = _population("implicit", batch=2)
    geometry = _chain_geometry()
    oracle = _initial_oracle_state(run_model)
    for _ in range(STEPS):
        oracle = _oracle_step(oracle, geometry, DT, "implicit")

    run_model.run(tstop=STEPS * DT, dt=DT)
    long_model.longrun(tstop=STEPS * DT, chunklength=2, dt=DT)

    run_state = _material_state(run_model)
    long_state = _material_state(long_model)
    _assert_state_close(run_state, long_state)
    _assert_state_close(
        run_state,
        {"v": oracle[0], "solute": oracle[1], "reserve": oracle[2]},
    )
    assert run_state["solute"].shape == (2, 1, C)
    assert not torch.equal(run_state["solute"][0], run_state["solute"][1])


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {name: _clone_nested(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value


def test_fresh_model_checkpoint_replay_restores_full_and_local_material_state():
    source = _population("implicit")
    source.run(tstop=3 * DT, dt=DT)
    checkpoint = _clone_nested(source.state_dict_for_checkpoint())

    source.run(tstop=3 * DT, dt=DT)
    expected = _material_state(source)

    resumed = _population("implicit")
    resumed.restore_dict_from_checkpoint(checkpoint)
    restored = _material_state(resumed)
    assert restored["t"].item() == pytest.approx(3 * DT)
    torch.testing.assert_close(restored["solute"], restored["local_solute"])
    resumed.run(tstop=3 * DT, dt=DT)

    _assert_state_close(_material_state(resumed), expected)


def test_dt_reconfiguration_rebuilds_transport_and_matches_mixed_step_oracle():
    model = _population("implicit")
    geometry = _chain_geometry()
    oracle = _initial_oracle_state(model)

    dt_first, dt_second = 0.04, 0.09
    for dt in (dt_first, dt_first, dt_second, dt_second, dt_second):
        oracle = _oracle_step(oracle, geometry, dt, "implicit")
        model.step(dt=dt)

    process = next(
        process
        for process in model.mech.material_processes.values()
        if isinstance(process, DiffusionProcess)
    )
    operator = next(iter(process._spatial_operators.values()))
    torch.testing.assert_close(operator.dt, torch.tensor(dt_second, dtype=DTYPE))
    assert process.dt.item() == pytest.approx(dt_second)
    _assert_state_close(
        _material_state(model),
        {"v": oracle[0], "solute": oracle[1], "reserve": oracle[2]},
    )


def _holistic_loss(v, solute, reserve):
    return 0.02 * v.square().sum() + solute.square().sum() + 0.3 * reserve.sum()


def _oracle_population_loss(diffusivity, clearance_rate, initial_scale):
    geometry = _chain_geometry()
    v = INITIAL_V.reshape(1, C).clone()
    solute = (INITIAL_SOLUTE * initial_scale).reshape(1, C)
    reserve = INITIAL_RESERVE.reshape(1, C).clone()
    state = (v, solute, reserve)
    for _ in range(4):
        state = _oracle_step(
            state,
            geometry,
            DT,
            "implicit",
            diffusivity=diffusivity,
            clearance_rate=clearance_rate,
        )
    return _holistic_loss(*state)


def test_material_diffusivity_clearance_and_initial_value_autograd_match_fd():
    model = _population("implicit", training=True)
    diffusion = next(
        process
        for process in model.mech.material_processes.values()
        if isinstance(process, DiffusionProcess)
    )
    clearance = next(
        process
        for process in model.mech.material_processes.values()
        if isinstance(process, ClearanceProcess)
    )

    diffusivity = torch.tensor(DIFFUSIVITY, dtype=DTYPE, requires_grad=True)
    clearance_rate = torch.tensor(CLEARANCE_RATE, dtype=DTYPE, requires_grad=True)

    solute = model.mech.materials["solute"]
    initial_source = solute.initial_source("c")
    initial_source.requires_grad_(True)
    model.initialize()
    # Reinitialization resolves registered process parameters into runtime
    # buffers; inject these two independent leaves only after that lifecycle.
    diffusion._buffers["D"] = diffusivity
    clearance._buffers["rate"] = clearance_rate

    model.run(tstop=4 * DT, dt=DT)
    loss = _holistic_loss(
        model.v,
        model.mech.materials["solute"].c,
        model.mech.materials["reserve"].c,
    )
    diffusivity_grad, clearance_grad, initial_grad = torch.autograd.grad(
        loss, (diffusivity, clearance_rate, initial_source)
    )
    initial_direction = INITIAL_SOLUTE.reshape(1, C).expand_as(initial_source)
    initial_scale_grad = (initial_grad * initial_direction).sum()
    actual = (diffusivity_grad, clearance_grad, initial_scale_grad)

    values = [DIFFUSIVITY, CLEARANCE_RATE, 1.0]
    expected = []
    eps = 1.0e-5
    for index in range(3):
        plus = list(values)
        minus = list(values)
        plus[index] += eps
        minus[index] -= eps
        derivative = (
            _oracle_population_loss(*plus) - _oracle_population_loss(*minus)
        ) / (2.0 * eps)
        expected.append(derivative)

    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
