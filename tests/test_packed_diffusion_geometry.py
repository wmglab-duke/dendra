"""Packing analytic cables preserves their declared transport coefficients."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import DiffusionProcess
from dendra.models.multi import MultiPopulation

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("threads", (1, 4))
@pytest.mark.parametrize("per_batch_diffusivity", (False, True))
def test_packed_edge_diffusivity_keeps_explicit_batch_axes(
    threads, per_batch_diffusivity
):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64):

            class Transport(DiffusionProcess):
                DiffusionProcess.BATCH(D=0.5)
                DiffusionProcess.METHOD("implicit", solver="dense")
                DiffusionProcess.DIFFUSE(
                    "pool", field="c", D="D", D_location="edge", geometry="chemical"
                )

            first = dn.Population(
                N=2, C=3, integrator=dn.bwd_euler_sc(), dtype=torch.float64
            )
            second = dn.Population(
                N=1, C=2, integrator=dn.bwd_euler_sc(), dtype=torch.float64
            )
            for population in (first, second):
                population.material("pool", fields={"c": 0.0}, domain="intracellular")
            first.register_material_geometry(
                "chemical",
                domain="intracellular",
                volume=torch.ones((2, 3), dtype=torch.float64),
                edge_factor=torch.ones((2, 2), dtype=torch.float64),
            )
            first.insert(Transport)
            packed = MultiPopulation(first=first, second=second)
            packed.batch(2)
            packed.train().initialize()
            process = next(iter(packed.mech.material_processes.values()))
            if per_batch_diffusivity:
                # A prepared/functional adapter can supply lane-specific
                # numerical buffers without changing the authored edge frame.
                diffusivity = torch.tensor(
                    [0.3, 0.7], dtype=torch.float64, requires_grad=True
                )
                process._buffers["D"] = diffusivity.reshape(2, 1, 1)
            else:
                diffusivity = process.D_param
                assert process.D.shape == (1, 1, 1)
            initial = torch.linspace(
                0.2, 2.5, 16, dtype=torch.float64, requires_grad=True
            ).reshape(2, 1, 8)
            packed.mech.materials["pool"]._buffers["c"] = initial
            dt = initial.new_tensor(0.1, requires_grad=True)
            process.set_dt(dt)
            actual_trace = [initial]
            for _step in range(4):
                process.advance_materials(dt)
                actual_trace.append(packed.mech.materials["pool"].c)
            actual = torch.stack(actual_trace)

            # Independent dense blocks for two three-node cells, followed by
            # two unchanged compartments in the component without transport.
            incidence = initial.new_zeros((4, 8))
            for edge, (parent, child) in enumerate(((0, 1), (1, 2), (3, 4), (4, 5))):
                incidence[edge, parent] = -1.0
                incidence[edge, child] = 1.0
            laplacian = incidence.T @ incidence
            matrix = torch.eye(8, dtype=initial.dtype) + (
                dt * diffusivity.reshape(-1, 1, 1) * laplacian
            )
            value = initial
            expected_trace = [value]
            for _step in range(4):
                value = (
                    torch.linalg.solve(matrix, value.squeeze(-2).unsqueeze(-1))
                    .squeeze(-1)
                    .unsqueeze(-2)
                )
                expected_trace.append(value)
            expected = torch.stack(expected_trace)
            torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
            weights = torch.linspace(0.2, 1.3, actual.numel()).reshape_as(actual)
            targets = (initial, diffusivity, dt)
            actual_gradients = torch.autograd.grad(
                (actual * weights).sum(), targets, retain_graph=True
            )
            expected_gradients = torch.autograd.grad(
                (expected * weights).sum(), targets
            )
            for actual_gradient, expected_gradient in zip(
                actual_gradients, expected_gradients
            ):
                torch.testing.assert_close(
                    actual_gradient, expected_gradient, rtol=3e-12, atol=3e-12
                )
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("threads", (1, 4))
@pytest.mark.parametrize("scheme", ("arithmetic", "series"))
@pytest.mark.parametrize("population_refs", (False, True))
@pytest.mark.parametrize("batch", (None, 2))
def test_packed_diffusion_preserves_analytic_fractions_and_relaxation(
    scheme, threads, population_refs, batch
):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64):
            # The node/edge shapes are intentionally component-local rather
            # than already flattened into the parent population's frame.
            diffusivity = torch.nn.Parameter(
                torch.tensor([[0.3, 0.5, 0.8], [0.6, 0.9, 0.4]])
            )

            class Transport(DiffusionProcess):
                DiffusionProcess.METHOD(
                    "implicit",
                    solver="dense",
                    interface_scheme=scheme,
                    volume_fraction=torch.tensor([[0.8, 0.9, 1.1], [0.7, 1.0, 0.6]]),
                    area_fraction=torch.tensor([[0.7, 0.8], [0.9, 0.6]]),
                )
                DiffusionProcess.DIFFUSE("pool", field="c", D=diffusivity)
                DiffusionProcess.RELAX(
                    "pool",
                    field="c",
                    rate="bath_rate"
                    if population_refs
                    else torch.tensor([[0.2, 0.3, 0.1], [0.1, 0.2, 0.4]]),
                    target="bath_target"
                    if population_refs
                    else torch.tensor([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]),
                    where="bath_mask" if population_refs else None,
                )

            def model(shift):
                population = dn.Population(
                    N=2, C=3, integrator=dn.bwd_euler_sc(), dtype=torch.float64
                )
                population.diam.copy_(torch.tensor([[2.0, 2.5, 1.5], [1.8, 2.2, 2.6]]))
                population.dx.copy_(torch.tensor([[0.7, 1.0, 1.2], [0.9, 1.3, 0.8]]))
                initial = torch.nn.Parameter(
                    torch.tensor([[0.5, 1.0, 2.0], [1.3, 0.7, 1.8]]) + shift
                )
                population.material(
                    "pool", fields={"c": initial}, domain="intracellular"
                )
                if population_refs:
                    population.register_parameter(
                        "bath_rate",
                        torch.nn.Parameter(
                            torch.tensor([[0.2, 0.3, 0.1], [0.1, 0.2, 0.4]])
                        ),
                    )
                    population.register_buffer(
                        "bath_target", torch.tensor([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
                    )
                    population.register_buffer(
                        "bath_mask",
                        torch.tensor([[True, False, True], [False, True, True]]),
                    )
                population.insert(Transport)
                return population, initial

            independent = [model(shift) for shift in (0.0, 2.0)]
            children = [model(shift) for shift in (0.0, 2.0)]
            packed = MultiPopulation(a=children[0][0], b=children[1][0])
            if batch is not None:
                packed.batch(batch)
                for population, _initial in independent:
                    population.batch(batch)

            def trace(population):
                population.initialize()
                initial = population.mech.materials["pool"]._buffers["c"].detach()
                initial.requires_grad_(True)
                population.mech.materials["pool"]._buffers["c"] = initial
                processes = tuple(population.mech.material_processes.values())
                for process in processes:
                    process.set_dt(0.1)
                field = population.mech.materials["pool"]._buffers["c"]
                records = [field]
                for _step in range(4):
                    for process in processes:
                        process.advance_materials(0.1)
                    records.append(population.mech.materials["pool"]._buffers["c"])
                return torch.stack(records), initial

            independent_traces = [trace(population) for population, _ in independent]
            expected = torch.cat(
                [recording.flatten(-2) for recording, _ in independent_traces],
                dim=-1,
            )
            actual, actual_initial = trace(packed)
            actual = actual.flatten(-2)
            torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
            weights = torch.linspace(0.3, 1.7, actual.numel()).reshape_as(actual)
            expected_gradients = torch.autograd.grad(
                (expected * weights).sum(),
                tuple(initial for _, initial in independent_traces)
                + (diffusivity,)
                + tuple(
                    population.bath_rate
                    for population, _ in independent
                    if population_refs
                ),
                retain_graph=True,
            )
            actual_gradients = torch.autograd.grad(
                (actual * weights).sum(),
                (actual_initial, diffusivity)
                + tuple(
                    population.bath_rate
                    for population, _ in children
                    if population_refs
                ),
            )
            expected_gradients = (
                torch.cat(
                    [value.flatten(-2) for value in expected_gradients[:2]], dim=-1
                ).reshape_as(actual_initial),
            ) + expected_gradients[2:]
            for actual_gradient, expected_gradient in zip(
                actual_gradients, expected_gradients
            ):
                torch.testing.assert_close(
                    actual_gradient, expected_gradient, rtol=3e-12, atol=3e-12
                )
    finally:
        torch.set_num_threads(previous)
