"""Inference lifecycle boundaries retain stochastic material preparation."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree

import dendra as dn
from tests.func.test_material_diffusion import DT
from tests.func.test_material_stochastic_diffusion_checkpoint import _population, _step
from tests.func.test_material_stochastic_sampling import DT as NOISY_DT
from tests.func.test_material_stochastic_sampling import (
    _diffusion_population,
    _native_step,
    _NoisyThomasDiffusion,
)

pytestmark = pytest.mark.cpu


def _clone(tree):
    return _pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    left, left_spec = _pytree.tree_flatten(actual)
    right, right_spec = _pytree.tree_flatten(expected)
    assert left_spec == right_spec
    for result, reference in zip(left, right, strict=True):
        if torch.is_tensor(result):
            torch.testing.assert_close(result, reference, atol=1e-12, rtol=1e-12)
        else:
            assert result == reference


def _inference_population(kind, monkeypatch, *, configured):
    population = _population(kind, monkeypatch, configure=configured)
    if configured:
        # Expose a checkpoint that cannot reconstruct the held operator from
        # the latest resident NOISE alone.
        _step(population)
        process = next(iter(population.mech.material_processes.values()))
        assert not torch.equal(process.D, process._material_held_noise_D)
    population.eval()
    return population


@pytest.mark.parametrize("kind", ("path", "tree"))
@pytest.mark.parametrize("configured", (False, True), ids=("cold", "configured"))
def test_eval_stochastic_diffusion_lowering_preserves_public_run_lifecycle(
    kind, configured, monkeypatch
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        source, native = (
            _inference_population(kind, monkeypatch, configured=configured),
            _inference_population(kind, monkeypatch, configured=configured),
        )
        saved = _clone(source.state_dict_for_checkpoint())
        functional, tensors = dn.func.make_functional(source, dt=DT)
        _close(source.state_dict_for_checkpoint(), saved)
        tape = functional.sample_noise_draw_tape(tensors.state, 2)
        actual, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(noise=tape),
        )
        # Public execution owns cold solver initialization and retains an
        # already configured evaluation solver without a forced rebuild.
        native.run(tstop=2 * DT, dt=DT)
        _close(actual, functional.extract(native).state)
        _close(source.state_dict_for_checkpoint(), saved)


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_stochastic_diffusion_cold_commit_continues_through_public_eval_run(
    kind, monkeypatch
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        source = _inference_population(kind, monkeypatch, configured=True)
        target = _inference_population(kind, monkeypatch, configured=False)
        functional, tensors = dn.func.make_functional(source, dt=DT)
        actual, _ = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(noise=functional.sample_noise_draw(tensors.state)),
        )
        expected, _ = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            actual,
            dn.func.StepInput(noise=functional.sample_noise_draw(actual)),
        )
        functional.commit_state_(target, actual)
        process = next(iter(target.mech.material_processes.values()))
        held = process._material_held_noise_D.detach().clone()
        assert not torch.equal(held, process.D)
        target.run(tstop=DT, dt=DT)
        torch.testing.assert_close(process._material_held_noise_D, held, atol=0, rtol=0)
        _close(functional.extract(target).state, expected)


def test_eval_stochastic_diffusion_new_timestep_uses_current_noise(monkeypatch):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        source, native = (
            _inference_population("path", monkeypatch, configured=True),
            _inference_population("path", monkeypatch, configured=True),
        )
        saved = _clone(source.state_dict_for_checkpoint())
        timestep = 2 * DT
        functional, tensors = dn.func.make_functional(source, dt=timestep)
        actual, _ = functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(noise=functional.sample_noise_draw(tensors.state)),
        )
        native.run(tstop=timestep, dt=timestep)
        _close(actual, functional.extract(native).state)
        _close(source.state_dict_for_checkpoint(), saved)


def test_compiled_stochastic_diffusion_eval_while_records_and_continues_snapshot():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        source, native = (
            _diffusion_population(_NoisyThomasDiffusion),
            _diffusion_population(_NoisyThomasDiffusion),
        )
        for population in (source, native):
            _native_step(population)
            population.eval()
        runner = dn.compile(
            source,
            dt=NOISY_DT,
            steps=4,
            initialization="snapshot",
            execution="while",
            inference=True,
            backend="aot_eager",
            trainable=(),
            callbacks={"trace": dn.func.Recorder(("v", "pool_material.amount"))},
        )
        state, auxiliary = runner()
        voltage = [native.v.detach().clone()]
        amount = [native.mech.materials["pool"].amount.detach().clone()]
        for _ in range(4):
            native.run(tstop=NOISY_DT, dt=NOISY_DT)
            voltage.append(native.v.detach().clone())
            amount.append(native.mech.materials["pool"].amount.detach().clone())
        _close(state, runner.functional.extract(native).state)
        trace = auxiliary["callbacks"]["trace"]
        torch.testing.assert_close(trace["v"], torch.stack(voltage))
        torch.testing.assert_close(trace["pool_material.amount"], torch.stack(amount))
        continued, _ = runner(state)
        native.run(tstop=4 * NOISY_DT, dt=NOISY_DT)
        _close(continued, runner.functional.extract(native).state)
        assert runner.execution_report().strategy == "compiled_while_loop"
