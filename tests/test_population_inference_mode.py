"""Execution-only inference retains a prewarmed plan, then resets for AD.

This does not assert that constructing or lazily preparing a Population inside
inference mode is equivalent: versionless topology deliberately uses the raw
validated solver. Both forward controls prepare their solver outside inference.
"""

from contextlib import nullcontext

import pytest
import torch

import dendra as dn
import dendra.models.integrators.tree as tree_module
from tests.test_population_fused_intra import DT, _case

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not callable(getattr(tree_module.dendra_solvers, "DHSPlan", None)),
        reason="requires the owning-plan native solver candidate",
    ),
]
STATES = ("v", "hh.m", "hh.h", "hh.n", "t")


def _carry(model):
    mechanism = model.mech.hh
    return (
        model.v,
        mechanism.m,
        mechanism.h,
        mechanism.n,
        model.t,
        model._duration_remainder,
    )


def _topology_signature(model):
    topology = tuple(
        getattr(model.integrator, name) for name in ("parent_idx", "order", "layer_ptr")
    )
    assert all(not torch.is_inference(value) for value in topology)
    return tuple((id(value), value._version) for value in topology)


def _prewarm(model):
    with torch.inference_mode(False), torch.no_grad():
        before = tuple(value.clone() for value in _carry(model))
        model.run(tstop=0.0, dt=DT)
        for expected, actual in zip(before, _carry(model), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    plan = model.integrator._cpu_topology_plan
    assert isinstance(plan, tree_module.dendra_solvers.DHSPlan)
    return plan, _topology_signature(model)


def _trace(model, *, inference=False):
    recorder = dn.Recorder(list(STATES))
    context = torch.inference_mode() if inference else nullcontext()
    with context:
        model.run(tstop=8 * DT, dt=DT, callbacks=[recorder])
        return {name: recorder.stack(name) for name in STATES}


def _assert_traces(actual, expected):
    assert actual.keys() == expected.keys()
    for name in STATES:
        assert actual[name].shape[0] == 9  # Initial sample plus all eight steps.
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-11, atol=1e-10)


def test_prewarmed_inference_full_trace_and_gradient_roundtrip():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1471)
        candidate, amplitude = _case(batch=2, dtype=torch.float64, tree=True)
        torch.manual_seed(1471)
        reference, _ = _case(batch=2, dtype=torch.float64, tree=True)

    with dn.ctx(
        JIT=1,
        BACKEND="inductor",
        FULLGRAPH=1,
        DEVICE="cpu",
        DTYPE=torch.float64,
        REQUIRE_GRAD=0,
    ):
        plan, topology = _prewarm(candidate)
        reference_plan, reference_topology = _prewarm(reference)
        with torch.no_grad():
            expected = _trace(reference)
        actual = _trace(candidate, inference=True)
        _assert_traces(actual, expected)
        assert candidate.integrator._cpu_topology_plan is plan
        assert reference.integrator._cpu_topology_plan is reference_plan
        assert _topology_signature(candidate) == topology
        assert _topology_signature(reference) == reference_topology

        # Never feed inference-created carry into a gradient-bearing run.
        with torch.inference_mode(False), torch.no_grad():
            candidate.initialize()
        assert all(not torch.is_inference(value) for value in _carry(candidate))
        assert candidate.t.item() == 0
        assert candidate._duration_remainder.item() == 0
        assert not candidate.integrator.initialized

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1471)
        fresh, fresh_amplitude = _case(
            batch=2,
            dtype=torch.float64,
            training=True,
            tree=True,
        )
    assert not torch.is_inference(amplitude)
    amplitude.requires_grad_(True)
    candidate.train()
    with dn.ctx(
        JIT=1,
        BACKEND="inductor",
        FULLGRAPH=1,
        DEVICE="cpu",
        DTYPE=torch.float64,
        REQUIRE_GRAD=1,
    ):
        # Re-materialize parameter-derived initialization after enabling AD,
        # just as for the independently constructed training reference.
        candidate.initialize()
        actual = _trace(candidate)
        expected = _trace(fresh)
        _assert_traces(actual, expected)
        actual_loss = sum(actual[name].square().mean() for name in STATES[:-1])
        expected_loss = sum(expected[name].square().mean() for name in STATES[:-1])
        actual_gradient = torch.autograd.grad(actual_loss, amplitude)[0]
        expected_gradient = torch.autograd.grad(expected_loss, fresh_amplitude)[0]
    assert torch.isfinite(actual_gradient).all()
    assert torch.all(actual_gradient != 0)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=1e-10, atol=1e-10
    )
    assert isinstance(
        candidate.integrator._cpu_topology_plan, tree_module.dendra_solvers.DHSPlan
    )
    _topology_signature(candidate)
