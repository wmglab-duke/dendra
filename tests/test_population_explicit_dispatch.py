"""Preserve public explicit integrators' inherited-wrapper fusion eligibility."""

from collections import Counter
from types import MethodType

import pytest
import torch

import dendra as dn
import dendra.models.core as population_core
from dendra.models.mod import pas
from tests.test_population_fused_intra import _counting_compiler, _NoopPreStep

pytestmark = pytest.mark.cpu


def _explicit_case(name):
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float64):
        model = dn.Population(
            N=2,
            C=4,
            dx=100.0,
            diam=1.0,
            rhoa=100.0,
            cm=1.0,
            v_init=-60.0,
            dtype=torch.float64,
            integrator=getattr(dn, name)(imem=False),
        )
        model.insert(pas)
        model[:, 0].inject(dn.constant(value=0.02 * dn.units.nA))
        model.requires_grad_(False)
        model.train(False)
        model.initialize()
    return model


def _execute(model, entrypoint, *, split=False, jit=1):
    recorder = dn.Recorder(["v"])
    with dn.ctx(
        JIT=jit,
        BACKEND="eager",
        FULLGRAPH=1,
        REQUIRE_GRAD=0,
        DEVICE="cpu",
        DTYPE=torch.float64,
    ):
        kwargs = dict(
            tstop=0.008,
            dt=0.001,
            callbacks=[recorder, *([_NoopPreStep()] if split else [])],
        )
        if entrypoint == "longrun":
            kwargs["chunklength"] = 3
        getattr(model, entrypoint)(**kwargs)
    return recorder.stack("v")


@pytest.mark.parametrize("name", ("euler", "rk1", "eulerv1", "rk2", "rk4"))
@pytest.mark.parametrize("entrypoint", ("run", "longrun"))
def test_public_explicit_integrators_preserve_fused_dispatch(
    monkeypatch, name, entrypoint
):
    candidate = _explicit_case(name)
    reference = _explicit_case(name)
    assert candidate.integrator._supports_fused_intra_step()
    calls, _active = _counting_compiler(monkeypatch)
    actual = _execute(candidate, entrypoint)
    assert Counter(calls) == {"_step_with_intra": 8}
    calls.clear()
    expected = _execute(reference, entrypoint, split=True)
    assert Counter(calls) == {"make_intra": 8, "_step": 8}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate.t, reference.t, rtol=0, atol=0)


@pytest.mark.parametrize("name", ("eulerv1", "rk2", "rk4"))
def test_explicit_tensor_kernel_override_still_rejects_fusion(name):
    model = _explicit_case(name)
    original = model.integrator._step

    def custom_step(_integrator, *args, **kwargs):
        return original(*args, **kwargs)

    model.integrator._step = MethodType(custom_step, model.integrator)
    assert not model.integrator._supports_fused_intra_step()


@pytest.mark.parametrize("entrypoint", ("run", "longrun"))
def test_eager_execution_does_not_inspect_compiled_segment_stability(
    monkeypatch, entrypoint
):
    model = _explicit_case("euler")

    def unexpected_proof(*args, **kwargs):
        pytest.fail("Eager execution must bypass compiled-segment eligibility walks")

    monkeypatch.setattr(population_core, "_runtime_execution_stable", unexpected_proof)
    actual = _execute(model, entrypoint, jit=0)
    assert actual.shape == (9, *model.shape)
    assert torch.isfinite(actual).all()
