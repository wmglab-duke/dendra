"""General branched HH execution preserves gradients and live source updates."""

import pytest
import torch

import dendra as dn
from tests.test_population_eligibility_cache import (
    CHANNELS,
    _assert_traces_equal,
    _execute_full,
)
from tests.test_population_fused_intra import _case, _counting_compiler
from tests.test_population_run_dispatch import _assert_no_payload

pytestmark = pytest.mark.cpu


def _gradient_case():
    pytest.importorskip("dendra_solvers")
    model, amplitude = _case(tree=True, batch=32, training=True)
    channels = tuple(getattr(model.mech.hh, name + "_param") for name in CHANNELS)
    assert all(value.ndim == 0 for value in channels)
    for parameter in channels:
        parameter.requires_grad_(True)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=model.dtype()):
        model.initialize()
    voltage = model.v.detach().clone().requires_grad_(True)
    model.v = voltage
    for gate in ("m", "h", "n"):
        setattr(model.mech.hh, gate, getattr(model.mech.hh, gate).detach())
    return model, (amplitude, voltage, *channels)


def test_large_branched_hh_longrun_matches_split_native_trace_and_eight_gradients(
    monkeypatch,
):
    calls, _ = _counting_compiler(monkeypatch)
    candidate, candidate_targets = _gradient_case()
    reference, reference_targets = _gradient_case()
    actual = _execute_full(candidate, "longrun")
    expected = _execute_full(reference, "longrun", jit=0)
    assert calls == ["_step_with_intra"] * 8
    _assert_traces_equal(actual, expected)
    actual_gradients = torch.autograd.grad(
        actual["v"].square().mean(), candidate_targets
    )
    expected_gradients = torch.autograd.grad(
        expected["v"].square().mean(), reference_targets
    )
    assert len(actual_gradients) == 8
    for observed, wanted in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)
        assert torch.isfinite(observed).all() and torch.count_nonzero(observed)
    for model in (candidate, reference):
        assert model.v.grad_fn is not None
        assert all(
            getattr(model.mech.hh, gate).grad_fn is not None for gate in ("m", "h", "n")
        )
        assert (
            not {"m_a", "m_b", "h_a", "h_b", "n_a", "n_b"}
            & model.mech.hh._buffers.keys()
        )
        _assert_no_payload(model)


def test_branched_hh_aliases_and_live_parameter_state_changes_match_eager_native_segments():
    pytest.importorskip("dendra_solvers")
    candidate, _ = _case(tree=True, batch=2)
    reference, _ = _case(tree=True, batch=2)
    for model in (candidate, reference):
        model.mech.hh.h = model.mech.hh.m
        model.mech.hh.gkbar_param = model.mech.hh.gnabar_param
        assert model.mech.hh.h is model.mech.hh.m
        assert model.mech.hh.gkbar_param is model.mech.hh.gnabar_param

    def run_pair():
        actual = _execute_full(candidate, "run")
        expected = _execute_full(reference, "run", jit=0)
        _assert_traces_equal(actual, expected)
        for model in (candidate, reference):
            assert model.mech.hh.h is not model.mech.hh.m
            _assert_no_payload(model)
        return actual

    first = run_pair()
    retained = {name: value.clone() for name, value in first.items()}
    for model in (candidate, reference):
        owner = model.mech.hh
        owner.gkbar_param = torch.nn.Parameter(
            torch.full_like(owner.gkbar_param, 0.036), requires_grad=False
        )
        with torch.no_grad():
            owner.gnabar_param.mul_(0.85)
        model.v = model.v + 0.25
        for gate in ("m", "h", "n"):
            setattr(owner, gate, getattr(owner, gate) * 0.99)
        assert owner.gkbar_param is not owner.gnabar_param
    run_pair()
    for model in (candidate, reference):
        owner = model.mech.hh
        shared = torch.nn.Parameter(
            torch.full_like(owner.gnabar_param, 0.09), requires_grad=False
        )
        owner.gnabar_param = shared
        owner.gkbar_param = shared
        assert owner.gkbar_param is owner.gnabar_param
    run_pair()
    _assert_traces_equal(first, retained)
