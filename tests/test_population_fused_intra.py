"""CPU run() fuses pure stimulus scatter without moving user hooks/state commit."""

import copy
import functools
import pickle
import sys
from collections import Counter
from contextlib import contextmanager
from types import MethodType

import pytest
import torch

import dendra as dn
import dendra.models.core as population_core
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.callbacks import Callback, CallbackList
from dendra.models.core import (
    _intra_fusion_callbacks_safe,
    _intra_fusion_callbacks_stable,
)
from dendra.models.integrators.core import Integrator, _DeferredIntra
from dendra.models.integrators.implicit import _bwd_euler_sc
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mod import hh
from dendra.models.stim.intra import Intra

pytestmark = pytest.mark.cpu
DT = 0.01


class _NoopPreStep(Callback):
    def pre_step_hook(self, model):
        pass


class _RuntimeNoise(Mechanism):
    Mechanism.RANGENOISE("eta", distribution="normal", mu=0.0, sigma=1.0, seed=715)


class _CustomIntra(Intra):
    pass


class _KernelCacheOwner(Integrator):
    """Exercise inherited pickle handling without unrelated dynamic HH proxies."""

    def __init__(self, kernels):
        torch.nn.Module.__init__(self)
        self._compiled_kernels = kernels


def _case(*, batch=3, dtype=torch.float64, training=False, noise=False, tree=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=dtype):
        if tree:
            morphology = dn.Morphology(rhoa=100.0, cm=1.0)
            soma = morphology.section("soma", L=20.0, diam=20.0, nseg=1)
            left = morphology.section("left", L=30.0, diam=1.5, nseg=3)
            right = morphology.section("right", L=25.0, diam=1.0, nseg=2)
            left.connect(soma.at(1.0), child_end=0)
            right.connect(soma.at(1.0), child_end=0)
            model = dn.Tree.from_morphology(
                morphology,
                N=batch,
                celsius=6.3,
                v_init=-62.0,
                dtype=dtype,
                integrator=dn.dhs(threads=32, imem=False),
            )
            model.internal_nodes.insert(hh)
        else:
            model = dn.Population(
                N=batch,
                C=4,
                celsius=6.3,
                v_init=-62.0,
                dtype=dtype,
                integrator=_bwd_euler_sc,
            )
            model.insert(hh)
        if noise:
            model.insert(_RuntimeNoise)
        waveform = (
            dn.mono_rect(
                amp=torch.linspace(0.1, 0.3, batch, dtype=dtype),
                delay=0.01,
                pw=0.06,
            )
            * dn.units.nA
        )
        model[:, 0].inject(waveform)
        # Overlapping sites ensure the fused scatter retains accumulate=True.
        model[:, 0].inject(dn.constant(value=0.02 * dn.units.nA))
        model.requires_grad_(False)
        amplitude = dict(waveform.named_parameters())["waveforms.0.amp"]
        amplitude.requires_grad_(training)
        model.train(training)
        model.initialize()
    return model, amplitude


def _counting_compiler(monkeypatch):
    calls = []
    active = []

    def compile_(function, **kwargs):
        @functools.wraps(function)
        def compiled(*args, **kw):
            calls.append(function.__name__)
            active.append(function.__name__)
            try:
                return function(*args, **kw)
            finally:
                active.pop()

        compiled._torchdynamo_orig_callable = function
        return compiled

    monkeypatch.setattr(torch, "compile", compile_)
    return calls, active


def _run(model, *, callbacks=None, backend="eager", steps=8):
    recorder = dn.Recorder(["v"])
    with dn.ctx(
        JIT=1,
        BACKEND=backend,
        FULLGRAPH=1,
        REQUIRE_GRAD=int(model.training),
        DEVICE="cpu",
        DTYPE=model.dtype(),
    ):
        with torch_compiler_warning_context():
            model.run(tstop=steps * DT, dt=DT, callbacks=[recorder, *(callbacks or [])])
    return recorder.stack("v")


@contextmanager
def _count_handler_noise_calls():
    """Observe canonical sampling without replacing a guarded method."""
    calls = []
    code = MechanismHandler.sample_runtime_noises_.__code__
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is code:
            calls.append(None)
        if callable(previous):
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def test_fused_run_dispatches_once_per_step_and_commits_eagerly(monkeypatch):
    model, _ = _case()
    calls, active = _counting_compiler(monkeypatch)
    observed = []
    original_setattr = type(model).__setattr__

    def setattr_(owner, name, value):
        if owner is model and name in ("v", "t", "i_membrane"):
            assert not active, "Population state commit escaped into compile"
        return original_setattr(owner, name, value)

    monkeypatch.setattr(type(model), "__setattr__", setattr_)

    class Observe(Callback):
        def post_step_hook(self, population):
            assert not active, "Population state commit/callback escaped into compile"
            observed.append(population.v.clone())

    trace = _run(model, callbacks=[Observe()])
    assert Counter(calls) == {"_step_with_intra": 8}
    assert trace.shape == (9, 3, 4)
    torch.testing.assert_close(trace[1:], torch.stack(observed), rtol=0, atol=0)
    assert all(
        key[0] == "_step_with_intra" for key in model.integrator._compiled_kernels
    )
    assert not any(
        isinstance(value, _DeferredIntra) for value in model.__dict__.values()
    )
    assert not any(
        isinstance(value, _DeferredIntra)
        for value in model.integrator.__dict__.values()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_full_traces_and_batched_stimulus_gradients_match_split_path(dtype, backend):
    baseline, baseline_amplitude = _case(dtype=dtype, training=True)
    candidate, candidate_amplitude = _case(dtype=dtype, training=True)
    reference = _run(baseline, callbacks=[_NoopPreStep()], backend=backend)
    actual = _run(candidate, backend=backend)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    reference_gradient = torch.autograd.grad(
        reference.square().mean(), baseline_amplitude
    )[0]
    actual_gradient = torch.autograd.grad(actual.square().mean(), candidate_amplitude)[
        0
    ]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)
    assert torch.isfinite(actual_gradient).all()
    assert torch.all(actual_gradient != 0)
    assert any(
        key[0] == "_step_with_intra" for key in candidate.integrator._compiled_kernels
    )
    assert not any(
        key[0] == "_step_with_intra" for key in baseline.integrator._compiled_kernels
    )
    # Reinitializing must not retain a consumed autograd graph or prior payload.
    with dn.ctx(JIT=1, BACKEND=backend, FULLGRAPH=1, DEVICE="cpu", DTYPE=dtype):
        candidate.initialize()
    repeated = _run(candidate, backend=backend)
    repeated_gradient = torch.autograd.grad(
        repeated.square().mean(), candidate_amplitude
    )[0]
    torch.testing.assert_close(repeated, reference, rtol=0, atol=0)
    torch.testing.assert_close(repeated_gradient, reference_gradient, rtol=0, atol=0)


def test_custom_pre_step_hook_observes_original_scatter_before_hook_order(monkeypatch):
    candidate, _ = _case()
    reference, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)

    class MutateIndices(Callback):
        def pre_step_hook(self, model):
            assert calls[-1] == "make_intra"
            # The current step must use the old index; this affects the next step.
            model.intra.indices[0][-1].fill_(1)

    actual = _run(candidate, callbacks=[MutateIndices()])
    assert Counter(calls) == {"make_intra": 8, "_step": 8}
    calls.clear()
    expected = _run(reference, callbacks=[MutateIndices(), _NoopPreStep()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_branched_dhs_fusion_with_plan_preserves_full_trace_and_batched_gradient():
    solvers = pytest.importorskip("dendra_solvers")
    baseline, baseline_amplitude = _case(dtype=torch.float32, training=True, tree=True)
    candidate, candidate_amplitude = _case(
        dtype=torch.float32, training=True, tree=True
    )
    reference = _run(baseline, callbacks=[_NoopPreStep()], backend="inductor")
    actual = _run(candidate, backend="inductor")
    reference_gradient = torch.autograd.grad(
        reference.square().mean(), baseline_amplitude
    )[0]
    actual_gradient = torch.autograd.grad(actual.square().mean(), candidate_amplitude)[
        0
    ]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)
    assert actual.shape[1] == 3
    assert torch.isfinite(actual_gradient).all()
    assert torch.all(actual_gradient != 0)
    if callable(getattr(solvers, "DHSPlan", None)):
        assert candidate.integrator._cpu_topology_plan is not None
        assert baseline.integrator._cpu_topology_plan is not None
    assert any(
        key[0] == "_step_with_intra" for key in candidate.integrator._compiled_kernels
    )
    assert not any(
        key[0] == "_step_with_intra" for key in baseline.integrator._compiled_kernels
    )


def test_callback_added_pre_hook_disables_fusion_on_following_step(monkeypatch):
    model, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)
    observed = []

    class ChangeHook(Callback):
        def post_step_hook(self, population):
            if not hasattr(self, "changed"):
                self.changed = True

                def before(callback, population):
                    assert calls[-1] == "make_intra"
                    observed.append(population.t.item())

                self.pre_step_hook = MethodType(before, self)

    _run(model, callbacks=[ChangeHook()])
    assert Counter(calls) == {"_step_with_intra": 1, "make_intra": 7, "_step": 7}
    assert len(observed) == 7


def test_callback_list_and_instance_overrides_are_not_treated_as_noops():
    plain = Callback()
    assert _intra_fusion_callbacks_safe(
        CallbackList([plain, CallbackList([Callback()])])
    )
    plain.pre_step_hook = MethodType(lambda self, model: None, plain)
    assert not _intra_fusion_callbacks_safe(CallbackList([plain]))

    class CustomList(CallbackList):
        def pre_step_hook(self, model):
            pass

    assert not _intra_fusion_callbacks_safe(CustomList())
    assert not _intra_fusion_callbacks_safe(CallbackList([CustomList()]))


def test_callback_eligibility_scans_the_actual_forwarded_children():
    class HiddenIterator(CallbackList):
        def __iter__(self):
            return iter(())

    class CustomPostStep(Callback):
        def post_step_hook(self, model):
            pass

    hidden_pre = HiddenIterator([_NoopPreStep()])
    assert not _intra_fusion_callbacks_safe(hidden_pre)
    assert not _intra_fusion_callbacks_stable(hidden_pre)
    assert not _intra_fusion_callbacks_safe(CallbackList([hidden_pre]))
    assert not _intra_fusion_callbacks_stable(CallbackList([hidden_pre]))
    hidden_post = HiddenIterator([CustomPostStep()])
    assert _intra_fusion_callbacks_safe(hidden_post)
    assert not _intra_fusion_callbacks_stable(hidden_post)
    assert not _intra_fusion_callbacks_stable(CallbackList([hidden_post]))


def test_callback_subclasses_and_custom_containers_keep_dynamic_classification():
    class CustomCallback(Callback):
        pass

    class CustomCallbackList(CallbackList):
        pass

    class CustomModuleList(torch.nn.ModuleList):
        pass

    custom_children = CallbackList()
    custom_children.callbacks = CustomModuleList([Callback()])
    for callbacks in (
        CallbackList([CustomCallback()]),
        CustomCallbackList([Callback()]),
        custom_children,
    ):
        assert _intra_fusion_callbacks_safe(callbacks)
        assert not _intra_fusion_callbacks_stable(callbacks)


@pytest.mark.parametrize("target", ["integrator_step", "make_intra", "noise_sampler"])
def test_post_step_runtime_replacement_refreshes_eligibility(monkeypatch, target):
    model, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)
    observed = []

    class ReplaceRuntime(Callback):
        def post_step_hook(self, population):
            if hasattr(self, "changed"):
                return
            self.changed = True
            if target == "integrator_step":
                old = population.integrator.step

                def step_(owner, dt, ve=None, intra=None):
                    assert torch.is_tensor(intra)
                    observed.append("step")
                    return old(owner, dt, ve, intra)

                population.integrator.step = step_
            elif target == "make_intra":
                old = population.make_intra

                def scatter(*args):
                    observed.append("scatter")
                    return old(*args)

                population.make_intra = scatter
            else:
                old = population.integrator.mech.sample_runtime_noises_

                def sample(**kwargs):
                    assert calls[-1] == "make_intra"
                    observed.append("noise")
                    return old(**kwargs)

                population.integrator.mech.sample_runtime_noises_ = sample

    _run(model, callbacks=[ReplaceRuntime()])
    assert Counter(calls) == {"_step_with_intra": 1, "make_intra": 7, "_step": 7}
    assert len(observed) == 7


def test_plain_recorder_avoids_per_step_structural_scan(monkeypatch):
    model, _ = _case()
    _counting_compiler(monkeypatch)
    checks = []
    callback_checks = []
    original_callbacks = population_core._intra_fusion_callbacks_safe
    original_code = model._can_fuse_intra_step.__func__.__code__
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is original_code
            and frame.f_locals.get("self") is model
        ):
            checks.append(None)
        if callable(previous):
            previous(frame, event, arg)

    def check_callbacks(callbacks):
        callback_checks.append(callbacks)
        return original_callbacks(callbacks)

    monkeypatch.setattr(
        population_core, "_intra_fusion_callbacks_safe", check_callbacks
    )
    sys.setprofile(profile)
    try:
        _run(model)
    finally:
        sys.setprofile(previous)
    assert len(checks) == 1
    # Stable callbacks are now inspected statically without a dynamic hook read.
    assert not callback_checks


def test_custom_post_hook_rechecks_callback_safety_each_step(monkeypatch):
    model, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)
    checks = []
    original = population_core._intra_fusion_callbacks_safe

    def check(callbacks):
        checks.append(callbacks)
        return original(callbacks)

    class CustomPostStep(Callback):
        def post_step_hook(self, model):
            pass

    monkeypatch.setattr(population_core, "_intra_fusion_callbacks_safe", check)
    _run(model, callbacks=[CustomPostStep()])
    assert len(checks) == 8  # Static classification, then a dynamic check each step.
    assert Counter(calls) == {"_step_with_intra": 8}


def test_dynamic_callback_hook_lookup_keeps_per_step_scatter_order(monkeypatch):
    model, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)
    observed = []

    def dynamic_pre_hook(callback, population):
        assert calls[-1] == "make_intra"
        observed.append(population.t.item())

    class DynamicHook(Callback):
        def __getattribute__(self, name):
            if name == "pre_step_hook" and model.t.item() > 0:
                return MethodType(dynamic_pre_hook, self)
            return super().__getattribute__(name)

    callback = DynamicHook()
    assert _intra_fusion_callbacks_safe(CallbackList([callback]))
    assert not _intra_fusion_callbacks_stable(CallbackList([callback]), model)
    _run(model, callbacks=[callback])
    assert Counter(calls) == {"_step_with_intra": 1, "make_intra": 7, "_step": 7}
    assert len(observed) == 7


@pytest.mark.parametrize("training", [False, True])
def test_fused_run_skips_only_proven_empty_noise_dispatch(monkeypatch, training):
    candidate, amplitude = _case(training=training)
    reference, reference_amplitude = _case(training=training)
    compiled_calls, _ = _counting_compiler(monkeypatch)
    with _count_handler_noise_calls() as fused_calls:
        actual = _run(candidate)
    assert not fused_calls
    assert Counter(compiled_calls) == {"_step_with_intra": 8}
    compiled_calls.clear()
    with _count_handler_noise_calls() as split_calls:
        expected = _run(reference, callbacks=[_NoopPreStep()])
    assert len(split_calls) == 8
    assert Counter(compiled_calls) == {"make_intra": 8, "_step": 8}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if training:
        actual_gradient = torch.autograd.grad(actual.square().mean(), amplitude)[0]
        expected_gradient = torch.autograd.grad(
            expected.square().mean(), reference_amplitude
        )[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("level", ["handler", "mechanism", "state"])
def test_custom_noise_sampler_keeps_one_call_per_step(monkeypatch, level):
    model, _ = _case()
    compiled_calls, _ = _counting_compiler(monkeypatch)
    mechanism = model.mech.hh
    owner = {
        "handler": model.mech,
        "mechanism": mechanism,
        "state": next(iter(mechanism.DE.values())),
    }[level]
    original = owner.sample_runtime_noises_
    sampled = []

    def sample(*args, **kwargs):
        assert compiled_calls[-1] == "make_intra"
        sampled.append(None)
        return original(*args, **kwargs)

    owner.sample_runtime_noises_ = sample
    _run(model)
    assert len(sampled) == 8
    assert Counter(compiled_calls) == {"make_intra": 8, "_step": 8}


def test_recorded_lazy_getter_refreshes_runtime_eligibility_and_preserves_trace(
    monkeypatch,
):
    calls, _ = _counting_compiler(monkeypatch)

    def run_case(*, split):
        model, _ = _case()
        reads = []
        steps = []
        original_step = model.integrator.step

        def getter():
            reads.append(model.t.item())
            if len(reads) == 2:  # Initial pre-loop read, then the first post-step read.

                def replacement(owner, dt, ve=None, intra=None):
                    assert torch.is_tensor(intra), (
                        "Deferred payload reached a custom step"
                    )
                    steps.append(owner.t.item())
                    return original_step(owner, dt, ve, intra)

                model.integrator.step = replacement
            return model.v

        model.setreference("probe", getter)
        recorder = dn.Recorder(["probe"])
        trace = _run(model, callbacks=[recorder, *([_NoopPreStep()] if split else [])])
        return trace, recorder.stack("probe"), reads, steps

    actual, observed, reads, steps = run_case(split=False)
    assert Counter(calls) == {"_step_with_intra": 1, "make_intra": 7, "_step": 7}
    assert len(reads) == 9
    assert len(steps) == 7
    calls.clear()
    expected, expected_observed, expected_reads, expected_steps = run_case(split=True)
    assert Counter(calls) == {"make_intra": 8, "_step": 8}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(observed, expected_observed, rtol=0, atol=0)
    assert reads == expected_reads
    assert steps == expected_steps


def test_recorder_stability_uses_generated_targets_not_mutable_state_metadata():
    model, _ = _case()
    calls = []
    model.setreference("probe", lambda: calls.append("getter") or model.v)
    recorder = dn.Recorder(["probe"])
    recorder.states = ["v"]
    assert not _intra_fusion_callbacks_stable(CallbackList([recorder]), model)
    assert not calls, "Static classification must not evaluate the getter"
    assert _intra_fusion_callbacks_stable(CallbackList([dn.Recorder(["v"])]), model)
    assert _intra_fusion_callbacks_stable(CallbackList([dn.Recorder(["hh.m"])]), model)


def test_pre_hook_replacing_itself_cannot_leave_stale_runtime_eligibility(monkeypatch):
    model, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)
    observed = []

    class ReplaceRuntimeAndHook(Callback):
        def pre_step_hook(self, population):
            old = population.integrator.step

            def step_(owner, dt, ve=None, intra=None):
                assert torch.is_tensor(intra)
                observed.append("step")
                return old(owner, dt, ve, intra)

            population.integrator.step = step_
            self.pre_step_hook = MethodType(Callback.pre_step_hook, self)

    _run(model, callbacks=[ReplaceRuntimeAndHook()])
    assert Counter(calls) == {"make_intra": 8, "_step": 8}
    assert len(observed) == 8


@pytest.mark.parametrize(
    "override", ["intra", "make_intra", "step", "kernel", "noise_sampler"]
)
def test_custom_runtime_paths_keep_split_dispatch(monkeypatch, override):
    model, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)
    with dn.ctx(JIT=1, BACKEND="eager", FULLGRAPH=1, DEVICE="cpu", DTYPE=model.dtype()):
        model._refresh_compile_config_from_ctx()
        if override == "intra":
            model.intra = _CustomIntra(model, model.injections)
        elif override == "make_intra":
            old = model.make_intra
            model.make_intra = lambda *args: old(*args)
        elif override == "step":
            old = model._step
            model._step = lambda *args: old(*args)
        elif override == "kernel":
            old = model.integrator._call_kernel
            model.integrator._call_kernel = lambda *args, **kwargs: old(*args, **kwargs)
        else:
            old = model.integrator.mech.sample_runtime_noises_
            model.integrator.mech.sample_runtime_noises_ = lambda **kwargs: old(
                **kwargs
            )
        assert not model._can_fuse_intra_step(model.intra)
        _run(model)
    assert "_step_with_intra" not in calls
    assert calls.count("make_intra") == 8


def test_runtime_noise_falls_back_and_keeps_seeded_trace_and_noise(monkeypatch):
    candidate, _ = _case(noise=True)
    baseline, _ = _case(noise=True)
    calls, _ = _counting_compiler(monkeypatch)
    with _count_handler_noise_calls() as sampled:
        actual = _run(candidate)
    assert len(sampled) == 8
    assert "_step_with_intra" not in calls
    expected = _run(baseline, callbacks=[_NoopPreStep()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate.mech._RuntimeNoise.eta,
        baseline.mech._RuntimeNoise.eta,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("restore", ["deepcopy", "clear"])
def test_fused_compiled_cache_is_disposable_and_not_serialized(monkeypatch, restore):
    model, _ = _case(training=False)
    calls, _ = _counting_compiler(monkeypatch)
    expected = _run(model)
    assert model.integrator._compiled_kernels
    if restore == "deepcopy":
        restored = copy.deepcopy(model)
    else:
        restored = model.clear_jit_cache()
    assert not restored.integrator._compiled_kernels
    assert restored._make_intra_config is None
    with dn.ctx(JIT=1, BACKEND="eager", DEVICE="cpu", DTYPE=model.dtype()):
        restored.initialize()
    calls.clear()
    actual = _run(restored)
    assert calls.count("_step_with_intra") == 8
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_integrator_pickle_drops_fused_compiled_closures(monkeypatch):
    model, _ = _case()
    _counting_compiler(monkeypatch)
    _run(model)
    assert model.integrator._compiled_kernels
    owner = _KernelCacheOwner(model.integrator._compiled_kernels)
    restored = pickle.loads(pickle.dumps(owner))
    assert not restored._compiled_kernels
    assert owner._compiled_kernels
    assert model.integrator._compiled_kernels
    assert not model.integrator.__getstate__()["_compiled_kernels"]


def test_eager_execution_does_not_use_fused_payload(monkeypatch):
    model, _ = _case()
    calls, _ = _counting_compiler(monkeypatch)
    with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=model.dtype()):
        model.run(tstop=0.02, dt=DT)
        assert not model._can_fuse_intra_step(model.intra)
    assert not calls
    assert not model.integrator._compiled_kernels
