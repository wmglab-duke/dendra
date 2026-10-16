"""Opt-in tensor-only HH/DHS steps preserve the imperative host contract.

The counting compiler executes the captured FX graph without building Inductor
kernels. The CPU benchmark separately checks the real compiled forward/backward
paths, including full recording, at one and four CPU threads.
"""

from __future__ import annotations

import copy
import functools
import pickle
import sys
from contextlib import contextmanager

import pytest
import torch

import dendra as dn
from dendra.models.integrators.core import Integrator
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import hh
from tests.test_population_eligibility_cache import (
    CHANNELS,
    _assert_traces_equal,
    _execute_full,
)
from tests.test_population_fused_intra import (
    _case,
    _NoopPreStep,
    _RuntimeNoise,
)

pytestmark = pytest.mark.cpu


class _SavedHH(hh):
    Mechanism.SAVE_CURRENT("i")


class _AssignedHH(hh):
    Mechanism.ASSIGNED("phase_voltage")

    def assigned_values(self, voltage, values):
        del values
        return {"phase_voltage": voltage.clone()}


@pytest.fixture(autouse=True)
def native_dhs_available():
    solvers = pytest.importorskip("dendra_solvers")
    if not callable(getattr(solvers, "DHSPlan", None)):
        pytest.skip("tensor-only HH/DHS prototype requires native topology plans")


def _model(*, enabled=True, training=False, dtype=torch.float64, batch=2):
    model, amplitude = _case(tree=True, batch=batch, dtype=dtype, training=training)
    model.integrator._tensor_step_enabled = enabled
    return model, amplitude


def _compiler(monkeypatch, *, watched=()):
    """Observe flat graph operands without changing mechanism provenance."""
    calls, active, tensor_calls = [], [], []

    def compile_(function, **kwargs):
        graph = getattr(function, "__self__", None)
        is_tensor_step = isinstance(graph, torch.fx.GraphModule)

        @functools.wraps(function)
        def compiled(*args, **kw):
            calls.append(function.__name__)
            active.append(function.__name__)
            try:
                if is_tensor_step:
                    assert not kw
                    assert args and all(
                        isinstance(value, torch.Tensor) for value in args
                    )
                    tensor_calls.append((graph, args))
                    before = tuple(
                        (
                            model.v,
                            *(getattr(model.mech.hh, gate) for gate in ("m", "h", "n")),
                        )
                        for model in watched
                    )
                result = function(*args, **kw)
                if is_tensor_step:
                    for model, old in zip(watched, before, strict=True):
                        after = (
                            model.v,
                            *(getattr(model.mech.hh, gate) for gate in ("m", "h", "n")),
                        )
                        assert all(a is b for a, b in zip(after, old, strict=True)), (
                            "The tensor graph must return state, not publish it"
                        )
                return result
            finally:
                active.pop()

        compiled._torchdynamo_orig_callable = function
        return compiled

    monkeypatch.setattr(torch, "compile", compile_)
    return calls, active, tensor_calls


@contextmanager
def _steps(model):
    from dendra.models.integrators._tensor_step import _TensorStepBinding

    code = _TensorStepBinding.try_step.__code__
    previous, observed = sys.getprofile(), []

    def profile(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is code
            and frame.f_locals.get("model") is model
        ):
            observed.append(arg)
        if callable(previous):
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield observed
    finally:
        sys.setprofile(previous)


def _assert_idle(model):
    assert model.integrator._tensor_step_binding is None
    assert model.integrator._direct_membrane_structure_prevalidated is False


def test_tensor_graph_returns_state_without_mutating_live_population(monkeypatch):
    model, _ = _model()
    _, active, tensor_calls = _compiler(monkeypatch, watched=(model,))
    initial_voltage = model.v
    with _steps(model) as selected:
        trace = _execute_full(model, "run")
    assert selected == [True] * 8
    assert len(tensor_calls) == 8
    assert not active
    assert model.v is not initial_voltage
    torch.testing.assert_close(model.v, trace["v"][-1], rtol=0, atol=0)
    _assert_idle(model)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensor_step_full_traces_match_default_imperative_path(
    monkeypatch, entrypoint, dtype
):
    calls, _, tensor_calls = _compiler(monkeypatch)
    candidate, _ = _model(dtype=dtype)
    reference, _ = _model(enabled=False, dtype=dtype)
    with _steps(candidate) as selected:
        actual = _execute_full(candidate, entrypoint)
    assert selected == [True] * 8
    assert calls == ["forward"] * 8
    assert len(tensor_calls) == 8
    for graph, operands in tensor_calls:
        placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
        assert len(placeholders) == len(operands)
    expected = _execute_full(reference, entrypoint)
    _assert_traces_equal(actual, expected)
    for gate in ("m", "h", "n"):
        torch.testing.assert_close(
            getattr(candidate.mech.hh, gate),
            getattr(reference.mech.hh, gate),
            rtol=0,
            atol=0,
        )
    # State-bundle ASSIGNED rates are ephemeral solver locals, not new public
    # buffers. Tensor lowering must not turn them into persistent/public state.
    assert not set(("m_a", "m_b", "h_a", "h_b", "n_a", "n_b")) & set(
        candidate.mech.hh._buffers
    )
    _assert_idle(candidate)
    _assert_idle(reference)


def _gradient_case(*, enabled, batch=2):
    model, amplitude = _model(enabled=enabled, training=True, batch=batch)
    channels = tuple(getattr(model.mech.hh, name + "_param") for name in CHANNELS)
    for parameter in channels:
        parameter.requires_grad_(True)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=model.dtype()):
        model.initialize()
    voltage = model.v.detach().clone().requires_grad_(True)
    model.v = voltage
    for gate in ("m", "h", "n"):
        setattr(model.mech.hh, gate, getattr(model.mech.hh, gate).detach())
    return model, (amplitude, voltage, *channels)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
@pytest.mark.parametrize("batch", [2, 32])
def test_tensor_step_retains_all_eight_gradients(monkeypatch, entrypoint, batch):
    _compiler(monkeypatch)
    candidate, candidate_targets = _gradient_case(enabled=True, batch=batch)
    reference, reference_targets = _gradient_case(enabled=False, batch=batch)
    with _steps(candidate) as selected:
        actual = _execute_full(candidate, entrypoint)
    expected = _execute_full(reference, entrypoint)
    assert selected == [True] * 8
    _assert_traces_equal(actual, expected)
    assert candidate.v.grad_fn is not None
    assert all(
        getattr(candidate.mech.hh, gate).grad_fn is not None for gate in ("m", "h", "n")
    )
    actual_gradients = torch.autograd.grad(
        actual["v"].square().mean(), candidate_targets
    )
    expected_gradients = torch.autograd.grad(
        expected["v"].square().mean(), reference_targets
    )
    assert len(actual_gradients) == 8
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        assert torch.isfinite(actual_gradient).all()
        assert torch.count_nonzero(actual_gradient)
    _assert_idle(candidate)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_aliased_initial_gates_keep_general_execution_until_next_segment(
    monkeypatch, entrypoint
):
    _, _, tensor_calls = _compiler(monkeypatch)
    candidate, _ = _model()
    reference, _ = _model(enabled=False)
    for model in (candidate, reference):
        model.mech.hh.h = model.mech.hh.m
        assert model.mech.hh.h is model.mech.hh.m
    with _steps(candidate) as selected:
        actual = _execute_full(candidate, entrypoint, chunklength=8)
    expected = _execute_full(reference, entrypoint, chunklength=8)
    # make_fx may specialize same-object inputs as one placeholder. Although
    # the first generic step splits m/h, that must not authorize a different
    # recurrent graph halfway through this segment.
    assert selected == [False] * 8
    assert not tensor_calls
    _assert_traces_equal(actual, expected)
    assert candidate.mech.hh.h is not candidate.mech.hh.m
    _assert_idle(candidate)

    with _steps(candidate) as resumed:
        actual = _execute_full(candidate, entrypoint, chunklength=8)
    expected = _execute_full(reference, entrypoint, chunklength=8)
    assert resumed == [True] * 8
    assert len(tensor_calls) == 8
    _assert_traces_equal(actual, expected)
    _assert_idle(candidate)


def test_readonly_parameter_alias_pattern_is_normalized_across_host_segments(
    monkeypatch,
):
    _, _, tensor_calls = _compiler(monkeypatch)
    candidate, _ = _model()
    reference, _ = _model(enabled=False)

    def run_pair():
        before = len(tensor_calls)
        with _steps(candidate) as selected:
            actual = _execute_full(candidate, "run")
        expected = _execute_full(reference, "run")
        assert selected == [True] * 8
        _assert_traces_equal(actual, expected)
        _assert_idle(candidate)
        return tensor_calls[before][0]

    for model in (candidate, reference):
        owner = model.mech.hh
        owner.gkbar_param = owner.gnabar_param
        assert owner.gkbar_param is owner.gnabar_param
    aliased_graph = run_pair()

    for model in (candidate, reference):
        owner = model.mech.hh
        owner.gkbar_param = torch.nn.Parameter(
            torch.full_like(owner.gnabar_param, 0.036), requires_grad=False
        )
        assert owner.gkbar_param is not owner.gnabar_param
    distinct_graph = run_pair()
    assert distinct_graph is not aliased_graph

    for model in (candidate, reference):
        owner = model.mech.hh
        shared = torch.nn.Parameter(
            torch.full_like(owner.gnabar_param, 0.09), requires_grad=False
        )
        owner.gnabar_param = shared
        owner.gkbar_param = shared
    # Equal alias structure must reuse the original graph even though all
    # parameter object identities and numerical values have changed.
    assert run_pair() is aliased_graph


@pytest.mark.parametrize("mutation", ["inplace", "replacement"])
def test_new_host_call_reads_live_channel_parameters_and_rebound_state(
    monkeypatch, mutation
):
    _compiler(monkeypatch)
    candidate, _ = _model()
    reference, _ = _model(enabled=False)
    first = _execute_full(candidate, "run")
    retained = {key: value.clone() for key, value in first.items()}
    _execute_full(reference, "run")
    for model in (candidate, reference):
        owner = model.mech.hh
        old = owner.gnabar_param
        if mutation == "inplace":
            with torch.no_grad():
                old.mul_(0.85)
        else:
            owner.gnabar_param = torch.nn.Parameter(
                old.detach().clone() * 0.85, requires_grad=False
            )
        model.v = model.v + 0.25
        for gate in ("m", "h", "n"):
            setattr(owner, gate, getattr(owner, gate) * 0.99)
    with _steps(candidate) as selected:
        actual = _execute_full(candidate, "run")
    expected = _execute_full(reference, "run")
    assert selected == [True] * 8
    _assert_traces_equal(actual, expected)
    _assert_traces_equal(first, retained)
    _assert_idle(candidate)


@pytest.mark.parametrize("policy", ["off", "validate"])
def test_nonauto_membrane_policy_uses_general_step(monkeypatch, policy):
    calls, _, tensor_calls = _compiler(monkeypatch)
    candidate, _ = _model()
    reference, _ = _model(enabled=False)
    with dn.ctx(MEMBRANE_ASSEMBLY=policy):
        with _steps(candidate) as selected:
            actual = _execute_full(candidate, "run")
        expected = _execute_full(reference, "run")
    assert not any(selected)
    assert not tensor_calls
    assert "forward" not in calls
    _assert_traces_equal(actual, expected)
    _assert_idle(candidate)


@pytest.mark.parametrize("reason", ["callback", "noise", "getter", "imem"])
def test_unknown_or_observed_execution_keeps_general_step(monkeypatch, reason):
    _, _, tensor_calls = _compiler(monkeypatch)
    candidate, _ = _model()
    reference, _ = _model(enabled=False)
    callbacks = [_NoopPreStep()] if reason == "callback" else []
    for model in (candidate, reference):
        if reason == "noise":
            with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=model.dtype()):
                model.insert(_RuntimeNoise)
                model.initialize()
        elif reason == "getter":
            model.setreference("unused_authored_probe", lambda model=model: model.v)
        elif reason == "imem":
            model.integrator.imem = True
    with _steps(candidate) as selected:
        actual = _execute_full(candidate, "run", callbacks=callbacks)
    expected = _execute_full(reference, "run", callbacks=callbacks)
    assert not any(selected)
    assert not tensor_calls
    _assert_traces_equal(actual, expected)
    if reason == "imem":
        torch.testing.assert_close(
            candidate.i_membrane, reference.i_membrane, rtol=0, atol=0
        )
    _assert_idle(candidate)


@pytest.mark.parametrize("mechanism", [_SavedHH, _AssignedHH])
def test_hh_subclasses_preserve_saved_currents_and_assigned_phase(
    monkeypatch, mechanism
):
    _, _, tensor_calls = _compiler(monkeypatch)
    candidate, _ = _model()
    reference, _ = _model(enabled=False)
    for model in (candidate, reference):
        with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=model.dtype()):
            model.delete(hh)
            model.internal_nodes.insert(mechanism)
            model.initialize()
    fields = [
        "v",
        "t",
        *(f"{mechanism.__name__}.{gate}" for gate in ("m", "h", "n")),
    ]
    observed = "i_" if mechanism is _SavedHH else "phase_voltage"
    fields.append(f"{mechanism.__name__}.{observed}")

    def run(model):
        recorder = dn.Recorder(fields)
        with dn.ctx(
            JIT=1,
            BACKEND="eager",
            REQUIRE_GRAD=0,
            DEVICE="cpu",
            DTYPE=model.dtype(),
        ):
            model.run(tstop=0.08, dt=0.01, callbacks=[recorder])
        return {name: recorder.stack(name) for name in fields}

    with _steps(candidate) as selected:
        actual = run(candidate)
    expected = run(reference)
    assert not any(selected)
    assert not tensor_calls
    for name in fields:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert torch.isfinite(actual[fields[-1]]).all()
    assert torch.count_nonzero(actual[fields[-1]])
    _assert_idle(candidate)


@pytest.mark.parametrize(
    "owner_kind, attribute",
    [
        ("mechanism", "_advance_states"),
        ("state", "_solve"),
        ("integrator", "_advance_pre_current"),
    ],
)
def test_instance_callable_override_keeps_authored_dispatch_live(
    monkeypatch, owner_kind, attribute
):
    _, _, tensor_calls = _compiler(monkeypatch)
    candidate, _ = _model()
    reference, _ = _model(enabled=False)

    def install(model):
        owner = {
            "mechanism": model.mech.hh,
            "state": next(iter(model.mech.hh.DE.values())),
            "integrator": model.integrator,
        }[owner_kind]
        original = getattr(owner, attribute)
        calls = []

        def wrapped(*args, **kwargs):
            calls.append(None)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, attribute, wrapped)
        return calls

    candidate_calls = install(candidate)
    reference_calls = install(reference)
    with _steps(candidate) as selected:
        actual = _execute_full(candidate, "run")
    expected = _execute_full(reference, "run")
    assert not any(selected)
    assert not tensor_calls
    assert len(candidate_calls) == len(reference_calls) == 8
    _assert_traces_equal(actual, expected)
    _assert_idle(candidate)


def test_default_disabled_and_explicit_cache_clear_preserve_lazy_compilation(
    monkeypatch,
):
    calls, _, tensor_calls = _compiler(monkeypatch)
    default, _ = _case(tree=True, batch=1)
    assert default.integrator._tensor_step_enabled is False
    with _steps(default) as selected:
        _execute_full(default, "run")
    assert not selected and not tensor_calls
    assert calls == ["_step_with_intra"] * 8

    candidate, _ = _model(batch=1)
    first = _execute_full(candidate, "run")
    assert tensor_calls
    graphs = candidate.integrator._tensor_step_graphs.copy()
    assert graphs
    candidate.clear_jit_cache()
    assert not candidate.integrator._compiled_kernels
    assert candidate.integrator._tensor_step_graphs == graphs
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=candidate.dtype()):
        candidate.initialize()
    with _steps(candidate) as repeated:
        second = _execute_full(candidate, "run")
    assert repeated == [True] * 8
    _assert_traces_equal(second, first)
    _assert_idle(candidate)


def test_serialization_discards_active_binding_and_captured_graphs():
    # Avoid serializing generated HH proxy methods, which is a separate model
    # persistence concern. This directly exercises the Integrator contract.
    original = Integrator.__new__(Integrator)
    torch.nn.Module.__init__(original)
    original._compiled_kernels = {"compiled": lambda: None}
    original._tensor_step_binding = object()
    original._tensor_step_graphs = {"captured": object()}
    original._direct_membrane_structure_prevalidated = True
    state = original.__getstate__()
    assert state["_tensor_step_binding"] is None
    assert state["_tensor_step_graphs"] == {}
    assert state["_compiled_kernels"] == {}
    for restored in (copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
        assert restored._tensor_step_binding is None
        assert restored._tensor_step_graphs == {}
        assert restored._compiled_kernels == {}
        assert restored._direct_membrane_structure_prevalidated is False
    assert original._tensor_step_binding is not None
    assert original._tensor_step_graphs


def test_nested_and_exceptional_scope_restores_binding(monkeypatch):
    _compiler(monkeypatch)
    model, _ = _model()
    with dn.ctx(JIT=1, BACKEND="eager", DEVICE="cpu", DTYPE=model.dtype()):
        # Public runner setup refreshes both the Population JIT policy and its
        # integrator workspace, even when no timesteps are requested.
        model.run(tstop=0, dt=0.01)
        with pytest.raises(RuntimeError, match="scope failure"):
            with model.integrator._population_step_scope(stable=True):
                outer = model.integrator._tensor_step_binding
                assert outer is not None
                with model.integrator._population_step_scope(stable=False):
                    assert model.integrator._tensor_step_binding is None
                assert model.integrator._tensor_step_binding is outer
                raise RuntimeError("scope failure")
    _assert_idle(model)
