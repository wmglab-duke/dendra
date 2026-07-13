"""Holistic solver contracts for Slice-scoped intracellular stimulation."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.callbacks import RecorderLambda
from dendra.models.mod import pas
from dendra.models.stim.waveform import constant

DTYPE = torch.float64
DT = 0.025
N_STEPS = 8
BATCH_SIZE = 3
AMP = 0.1
V_REST = -65.0

SETUP_ORDERS = (
    "inject_then_batch",
    "batch_then_inject",
    "batch_then_configure_and_inject",
    "initialize_then_inject",
)


def _target(population, compartment=1):
    """Select one compartment while retaining every model and batch axis."""
    batch_axes = (slice(None),) * (len(population.shape) - 2)
    return population[batch_axes + (slice(None), compartment)]


def _duplicate_target(population, compartment=1):
    """Select the same physical compartment twice for every model replica."""
    batch_axes = (slice(None),) * (len(population.shape) - 2)
    return population[batch_axes + (slice(None), [compartment, compartment])]


def _initialize(container_kind, container):
    if container_kind == "population":
        container.initialize()
    else:
        container.initialize(dt=DT)


def _voltage(container_kind, container):
    population = container if container_kind == "population" else container.cell
    return population.v


def _inject(population, style: str, *, amplitude=AMP):
    waveform = constant(value=amplitude)
    if style == "single":
        _target(population).inject(waveform)
    elif style == "duplicate_index":
        _duplicate_target(population).inject(waveform)
    elif style == "double_amplitude":
        _target(population).inject(constant(value=2.0 * amplitude))
    elif style == "two_injections":
        _target(population).inject(waveform)
        _target(population).inject(constant(value=amplitude))
    else:  # pragma: no cover - test helper guard
        raise ValueError(style)


def _container(
    container_kind: str,
    *,
    setup_order: str,
    injection_style: str = "single",
    initialize: bool = True,
):
    population = dn.Population(
        N=2,
        C=3,
        v_init=V_REST,
        dt=DT,
        dtype=DTYPE,
    )

    container = population if container_kind == "population" else None

    def ensure_container():
        nonlocal container
        if container is None:
            if container_kind != "network":  # pragma: no cover - helper guard
                raise ValueError(container_kind)
            # Network deliberately accepts only unbatched populations. Network
            # batching must go through Network.batch so connection coordinate
            # systems and all component populations remain synchronized.
            container = dn.Network({"cell": population})
        return container

    def batch_container():
        current = ensure_container()
        if container_kind == "population":
            population.batch(BATCH_SIZE)
        else:
            current.batch(BATCH_SIZE)

    if setup_order == "inject_then_batch":
        population.insert(pas, g=0.001, e=V_REST)
        _inject(population, injection_style)
        batch_container()
    elif setup_order == "batch_then_inject":
        population.insert(pas, g=0.001, e=V_REST)
        batch_container()
        _inject(population, injection_style)
    elif setup_order == "batch_then_configure_and_inject":
        batch_container()
        population.insert(pas, g=0.001, e=V_REST)
        _inject(population, injection_style)
    elif setup_order == "initialize_then_inject":
        population.insert(pas, g=0.001, e=V_REST)
        batch_container()
    else:  # pragma: no cover - test helper guard
        raise ValueError(setup_order)

    container = ensure_container()

    if initialize:
        _initialize(container_kind, container)
        if setup_order == "initialize_then_inject":
            _inject(population, injection_style)
    return container


def _trajectory(container_kind: str, container, execution: str) -> torch.Tensor:
    if execution == "step":
        values = [_voltage(container_kind, container).detach().clone()]
        for _ in range(N_STEPS):
            if container_kind == "population":
                container.step(dt=DT)
            else:
                container.step()
            values.append(_voltage(container_kind, container).detach().clone())
        return torch.stack(values)

    if execution == "run":
        recorder = RecorderLambda(
            {"v": lambda model: _voltage(container_kind, model).detach().clone()}
        )
        if container_kind == "population":
            container.run(tstop=N_STEPS * DT, dt=DT, callbacks=[recorder])
        else:
            container.run(tstop=N_STEPS * DT, callbacks=[recorder])
        return recorder.stack("v")

    raise ValueError(execution)  # pragma: no cover - test helper guard


def _assert_scoped_response(trajectory: torch.Tensor) -> None:
    assert trajectory.shape == (N_STEPS + 1, BATCH_SIZE, 2, 3)
    assert torch.isfinite(trajectory).all()

    # With passive reversal equal to the initial voltage, compartments that do
    # not receive current remain at the exact equilibrium trajectory.
    unselected = trajectory[..., [0, 2]]
    torch.testing.assert_close(
        unselected,
        torch.full_like(unselected, V_REST),
        rtol=0.0,
        atol=1e-12,
    )

    selected = trajectory[..., 1]
    assert torch.all(selected[1:] > V_REST)
    assert not torch.equal(selected[-1], selected[0])

    # Scalar Slice stimulation is shared over the explicit batch grid, and the
    # two otherwise-identical cells follow the same trajectory.
    torch.testing.assert_close(
        selected,
        selected[:, :1, :1].expand_as(selected),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("container_kind", ["population", "network"])
@pytest.mark.parametrize("execution", ["step", "run"])
def test_slice_injection_trajectory_is_independent_of_batch_and_init_order(
    container_kind, execution
):
    trajectories = {}
    for setup_order in SETUP_ORDERS:
        container = _container(container_kind, setup_order=setup_order)
        trajectory = _trajectory(container_kind, container, execution)
        _assert_scoped_response(trajectory)
        trajectories[setup_order] = trajectory

    expected = trajectories["inject_then_batch"]
    for setup_order, trajectory in trajectories.items():
        torch.testing.assert_close(
            trajectory,
            expected,
            rtol=1e-12,
            atol=1e-12,
            msg=f"trajectory differs for setup order {setup_order!r}",
        )


@pytest.mark.parametrize("container_kind", ["population", "network"])
@pytest.mark.parametrize("execution", ["step", "run"])
def test_repeated_slice_indices_accumulate_current_exactly_once_per_occurrence(
    container_kind, execution
):
    trajectories = {}
    for injection_style in (
        "duplicate_index",
        "double_amplitude",
        "two_injections",
    ):
        container = _container(
            container_kind,
            setup_order="batch_then_inject",
            injection_style=injection_style,
        )
        trajectories[injection_style] = _trajectory(
            container_kind, container, execution
        )
        _assert_scoped_response(trajectories[injection_style])

    expected = trajectories["double_amplitude"]
    torch.testing.assert_close(
        trajectories["duplicate_index"], expected, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        trajectories["two_injections"], expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("execution", ["step", "run"])
def test_population_and_network_slice_injection_trajectories_agree(execution):
    population = _container("population", setup_order="batch_then_inject")
    network = _container("network", setup_order="batch_then_inject")

    population_trajectory = _trajectory("population", population, execution)
    network_trajectory = _trajectory("network", network, execution)

    torch.testing.assert_close(
        network_trajectory,
        population_trajectory,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("container_kind", ["population", "network"])
@pytest.mark.parametrize("execution", ["step", "run"])
def test_failed_preinitialization_execution_does_not_consume_slice_injection(
    container_kind, execution
):
    container = _container(
        container_kind,
        setup_order="inject_then_batch",
        initialize=False,
    )
    before = _voltage(container_kind, container).clone()

    with pytest.raises((ValueError, RuntimeError)):
        if execution == "step":
            if container_kind == "population":
                container.step(dt=DT)
            else:
                container.step()
        elif container_kind == "population":
            container.run(tstop=N_STEPS * DT, dt=DT)
        else:
            container.run(tstop=N_STEPS * DT)

    assert torch.equal(_voltage(container_kind, container), before)
    _initialize(container_kind, container)
    recovered = _trajectory(container_kind, container, execution)

    reference = _container(container_kind, setup_order="inject_then_batch")
    expected = _trajectory(container_kind, reference, execution)
    torch.testing.assert_close(recovered, expected, rtol=1e-12, atol=1e-12)
