"""Stock material operators retain independent detached noise samples for AD."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from dendra.models.mechanisms import ClearanceProcess, DiffusionProcess
from dendra.models.mod import pas

pytestmark = pytest.mark.cpu
DT = 0.125


class _RegionalStochasticDiffusion(DiffusionProcess):
    DiffusionProcess.GLOBAL(rate=0.2)
    DiffusionProcess.RANGERAND("D", distribution="uniform", low=0.3, high=0.7, seed=91)
    DiffusionProcess.RANGENOISE("target", mu=0.7, sigma=0.1, seed=92)
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE("pool", field="amount", D="D", geometry="chemical")
    DiffusionProcess.RELAX("pool", field="amount", rate="rate", target="target")


def _regional_population(region):
    population = dn.SingleCompartment(
        N=1, C=3, v_init=-65.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.diam.fill_(2.0)
    population.dx.fill_(1.0)
    population.material(
        "pool", fields={"amount": population.v.new_tensor([[1.0, 2.0, 4.0]])}
    )
    population.register_material_geometry(
        "chemical",
        domain="intracellular",
        volume=torch.nn.Parameter(population.v.new_tensor([[1.0, 1.2, 0.8]])),
        edge_factor=torch.nn.Parameter(population.v.new_tensor([[0.7, 0.9]])),
    )
    population.insert(pas)
    population[:, list(region)].insert(
        _RegionalStochasticDiffusion.rename("regional"),
        rate=torch.nn.Parameter(population.v.new_tensor(0.2)),
        preserve_duplicate_indices=True,
    )
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def _native_step(population):
    population._step(
        population.integrator, population, population.t.new_tensor(DT), None, None
    )
    population.t = population.t + DT


def _assert_state_close(actual, expected):
    left, left_spec = _pytree.tree_flatten(actual)
    right, right_spec = _pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for result, reference in zip(left, right, strict=True):
        torch.testing.assert_close(result, reference, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("region", ((2, 1), (2, 0, 1)), ids=("subset", "permuted"))
def test_regional_diffusion_rand_and_noise_use_physical_compartment_frame(region):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        source, native = _regional_population(region), _regional_population(region)
        process = source.mech.material_processes["regional"]
        like = source.mech.materials["pool"].amount
        index = torch.tensor(region, device=like.device).reshape(1, -1)
        for name in ("D", "target"):
            local = torch.broadcast_to(getattr(process, name), index.shape)
            expected = torch.zeros_like(like).scatter(-1, index, local)
            torch.testing.assert_close(
                process._resolve_spatial_quantity(name, like), expected
            )
        functional, tensors = dn.func.make_functional(source, dt=DT)
        tape = functional.sample_noise_draw_tape(tensors.state, 3)
        actual, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(noise=tape),
        )
        for _ in range(3):
            _native_step(native)
        _assert_state_close(actual, functional.extract(native).state)


def test_diffusion_held_noise_preserves_geometry_relaxation_and_rand_gradients():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        source, native = (
            _regional_population((2, 0, 1)),
            _regional_population((2, 0, 1)),
        )
        # The latest resident noise differs from the realization used by the
        # held solver. Its in-place refresh must not overwrite saved AD inputs.
        for population in (source, native):
            _native_step(population)
            process = population.mech.material_processes["regional"]
            assert not torch.equal(process.target, process._material_held_noise_target)
            material = population.mech.materials["pool"]
            material._buffers["amount"] = material.amount.detach()
        functional, tensors = dn.func.make_functional(source, dt=DT)
        parameters = {
            name: value.detach().clone().requires_grad_(value.requires_grad)
            for name, value in tensors.parameters.items()
        }
        tape = functional.sample_noise_draw_tape(tensors.state, 3)
        actual, _ = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(noise=tape),
        )
        for _ in range(3):
            _native_step(native)
        _assert_state_close(actual, functional.extract(native).state)
        volume = source.material_geometry("chemical")["volume"]
        names = tuple(
            next(name for name in parameters if name.endswith(suffix))
            for suffix in (
                "regional.rate_param",
                volume,
                "regional.D_low_param",
                "regional.target_mu_param",
            )
        )
        actual_gradients = torch.autograd.grad(
            actual["materials"]["pool"]["amount"].square().sum(),
            tuple(parameters[name] for name in names),
            allow_unused=True,
        )
        native_gradients = torch.autograd.grad(
            native.mech.materials["pool"].amount.square().sum(),
            tuple(native.get_parameter(name) for name in names),
            allow_unused=True,
        )
        for result, expected in zip(
            actual_gradients[:-1], native_gradients[:-1], strict=True
        ):
            assert (
                result is not None
                and result.isfinite().all()
                and result.abs().sum() > 0
            )
            torch.testing.assert_close(result, expected, atol=1e-11, rtol=1e-10)
        assert actual_gradients[-1] is None
        assert native_gradients[-1] is None


def test_packed_custom_material_stochastic_scopes_keep_component_support():
    from tests.func.test_material_stochastic_sampling import (
        _population as stochastic_population,
    )

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):

        def build():
            population = dn.concat_models(
                {
                    name: stochastic_population(regional="permuted")
                    for name in ("left", "right")
                },
                threads=1,
            )
            population.batch(2)
            population.train()
            population.initialize()
            population.integrator._initialize(
                population, population.t.new_tensor(DT), force=True
            )
            return population

        source, native = build(), build()
        functional, tensors = dn.func.make_functional(source, dt=DT)
        packet = functional.sample_initialization_draw(
            tensors.parameters, tensors.state
        )
        initialized = functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization._replace(random_draw=packet),
            state=tensors.state,
        )
        native.initialize()
        native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
        _assert_state_close(initialized.state, functional.extract(native).state)
        tape = functional.sample_noise_draw_tape(initialized.state, 3)
        actual, _ = functional.prepare_and_rollout(
            initialized.parameters,
            initialized.constants,
            initialized.state,
            dn.func.RolloutInput(noise=tape),
        )
        for _ in range(3):
            _native_step(native)
        _assert_state_close(actual, functional.extract(native).state)


@pytest.mark.parametrize("method", ("implicit", "exact"))
def test_native_clearance_noise_keeps_multistep_parameter_gradients(method):
    class _NoisyBath(ClearanceProcess):
        ClearanceProcess.GLOBAL(rate=0.4)
        ClearanceProcess.GLOBALNOISE("target", mu=1.0, sigma=0.1, seed=81)
        ClearanceProcess.METHOD(method)
        ClearanceProcess.CLEAR("pool", field="amount", rate="rate", target="target")

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64):
        population = dn.SingleCompartment(N=1, C=3)
        population.material("pool", fields={"amount": 2.0})
        population.insert(pas)
        population.insert(
            _NoisyBath, rate=torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        )
        population.train()
        population.initialize()
        dt = population.t.new_tensor(0.125)
        population.integrator._initialize(population, dt, force=True)
        process = next(iter(population.mech.material_processes.values()))
        initial = population.mech.materials["pool"].amount.detach().clone()
        samples = []
        for _ in range(3):
            population._step(population.integrator, population, dt, None, None)
            population.t = population.t + dt
            samples.append(process.target.detach().clone())
        rate = process.rate_param
        noise_parameters = tuple(
            value
            for name, value in process.named_parameters()
            if name.startswith(("target_mu_param", "target_sigma_param"))
        )
        assert len(noise_parameters) == 2
        actual, *noise_gradients = torch.autograd.grad(
            population.mech.materials["pool"].amount.sum(),
            (rate, *noise_parameters),
            allow_unused=True,
        )

    reference_rate = rate.detach().clone().requires_grad_()
    reference = initial
    for sample in samples:
        if method == "implicit":
            reference = (reference + dt * reference_rate * sample) / (
                1.0 + dt * reference_rate
            )
        else:
            reference = sample + (reference - sample) * torch.exp(-dt * reference_rate)
    expected = torch.autograd.grad(reference.sum(), reference_rate)[0]
    torch.testing.assert_close(actual, expected)
    assert actual.abs().item() > 0
    assert all(value is None for value in noise_gradients)
