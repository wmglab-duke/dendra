"""Segment-local execution proofs preserve imperative run/longrun semantics.

The compiler below only counts Python dispatch: these tests do not build new
Inductor kernels. Numerical native/C++ parity is covered by the CPU campaign.
"""

import sys
from collections import Counter
from contextlib import contextmanager

import pytest
import torch

import dendra as dn
import dendra.models.callbacks as callbacks_module
import dendra.models.core as population_core
from dendra.models.callbacks import Callback, CallbackList, Recorder
from tests.test_population_fused_intra import (
    DT,
    _case,
    _count_handler_noise_calls,
    _counting_compiler,
    _NoopPreStep,
    _RuntimeNoise,
)
from tests.test_population_run_dispatch import _assert_no_payload

pytestmark = pytest.mark.cpu
FIELDS = ("v", "hh.m", "hh.h", "hh.n", "t")
CHANNELS = ("gnabar", "gkbar", "gl", "ena", "ek", "el")


def test_population_tensor_subclass_rejects_segment_cache():
    model, _ = _case()

    class AuthoredTensor(torch.Tensor):
        pass

    model.v = model.v.as_subclass(AuthoredTensor)
    assert not population_core._runtime_execution_stable(model.mech, model)


def test_unrecorded_population_authored_getter_rejects_segment_cache():
    model, _ = _case()
    reads = []
    model.setreference("unrecorded_probe", lambda: reads.append(True) or model.v)
    assert not population_core._runtime_execution_stable(model.mech, model)
    assert not reads


def test_intra_metadata_proof_does_not_iterate_authored_containers():
    model, _ = _case()
    intra = model.build_intra()
    calls = []

    class AuthoredShapes(list):
        def __iter__(self):
            calls.append("iteration")
            return super().__iter__()

    assert population_core._runtime_plain_intra_metadata(intra)
    intra.selected_batch_shapes = AuthoredShapes(intra.selected_batch_shapes)
    assert not population_core._runtime_plain_intra_metadata(intra)
    assert not calls


def _execute_full(model, entrypoint, *, callbacks=(), steps=8, jit=1, chunklength=3):
    recorder = dn.Recorder(list(FIELDS))
    with dn.ctx(
        JIT=jit,
        BACKEND="eager",
        FULLGRAPH=1,
        REQUIRE_GRAD=int(model.training),
        DEVICE="cpu",
        DTYPE=model.dtype(),
    ):
        kwargs = dict(tstop=steps * DT, dt=DT, callbacks=[recorder, *callbacks])
        if entrypoint == "longrun":
            kwargs["chunklength"] = chunklength
        getattr(model, entrypoint)(**kwargs)
    return {name: recorder.stack(name) for name in FIELDS}


def _assert_traces_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in FIELDS:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        assert torch.isfinite(actual[name]).all()


@contextmanager
def _eligibility_calls(model):
    """Count canonical proof methods without installing executable overrides."""
    methods = {
        population_core.Population._can_fuse_intra_step.__code__: "fusion",
        population_core.Population._can_skip_runtime_noise_sampling.__code__: "noise",
    }
    calls = Counter()
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_locals.get("self") is model:
            name = methods.get(frame.f_code)
            if name is not None:
                calls[name] += 1
        if callable(previous):
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("entrypoint, segments", [("run", 1), ("longrun", 3)])
@pytest.mark.parametrize("with_intra", [False, True])
def test_full_hh_recorder_reuses_fusion_and_empty_noise_proof(
    monkeypatch, entrypoint, segments, with_intra
):
    model, _ = _case()
    if not with_intra:
        model.delete_injections()
    compiled, _ = _counting_compiler(monkeypatch)
    with _eligibility_calls(model) as checks, _count_handler_noise_calls() as sampled:
        actual = _execute_full(model, entrypoint)
    expected_checks = {"noise": segments}
    if with_intra:
        expected_checks["fusion"] = segments
    assert checks == expected_checks
    assert not sampled
    assert Counter(compiled) == {"_step_with_intra" if with_intra else "_step": 8}
    assert actual["v"].shape == actual["hh.m"].shape == (9, 3, 4)
    assert actual["t"].shape == (9,)
    _assert_no_payload(model)


def test_registered_hh_chain_proof_does_not_evaluate_lazy_getters():
    model, _ = _case()
    reads = []
    mechanism = model.integrator.mech.hh
    model.setreference("probe", lambda: reads.append("model") or model.v)
    mechanism.setreference("probe", lambda: reads.append("mechanism") or mechanism.m)
    assert population_core._intra_fusion_callbacks_stable(
        CallbackList([dn.Recorder(list(FIELDS))]), model
    )
    for selected in ("probe", "hh.probe"):
        recorder = dn.Recorder([selected])
        # Metadata is not the source of truth for the already-generated code.
        recorder.states = ["v"]
        assert not population_core._intra_fusion_callbacks_stable(
            CallbackList([recorder]), model
        )
    assert not reads, "A static absence proof must never call a lazy getter"


def test_replacing_a_recorded_gate_with_a_getter_rejects_static_proof():
    model, _ = _case()
    mechanism = model.integrator.mech.hh
    gate, reads = mechanism.m, []
    mechanism.setreference("m", lambda: reads.append("m") or gate)
    assert not population_core._intra_fusion_callbacks_stable(
        CallbackList([dn.Recorder(list(FIELDS))]), model
    )
    assert not reads


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_full_traces_and_eight_gradient_targets_match_dynamic_split_path(
    monkeypatch, entrypoint
):
    _counting_compiler(monkeypatch)

    def case():
        model, amplitude = _case(batch=2, training=True)
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

    candidate, candidate_targets = case()
    reference, reference_targets = case()
    actual = _execute_full(candidate, entrypoint)
    expected = _execute_full(reference, entrypoint, callbacks=[_NoopPreStep()])
    _assert_traces_equal(actual, expected)
    actual_gradients = torch.autograd.grad(
        actual["v"].square().mean(), candidate_targets
    )
    expected_gradients = torch.autograd.grad(
        expected["v"].square().mean(), reference_targets
    )
    assert len(actual_gradients) == 8
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        assert torch.isfinite(actual_gradient).all()
        assert torch.count_nonzero(actual_gradient) > 0
    for model in (candidate, reference):
        _assert_no_payload(model)


@pytest.mark.parametrize("reset", ["initialize", "clear_jit_cache"])
def test_new_run_rebuilds_proof_after_reset(monkeypatch, reset):
    model, _ = _case()
    _counting_compiler(monkeypatch)
    expected = _execute_full(model, "run")
    if reset == "clear_jit_cache":
        model.clear_jit_cache()
    with dn.ctx(JIT=1, BACKEND="eager", DEVICE="cpu", DTYPE=model.dtype()):
        model.initialize()
    with _eligibility_calls(model) as checks:
        actual = _execute_full(model, "run")
    assert checks == {"fusion": 1, "noise": 1}
    _assert_traces_equal(actual, expected)
    _assert_no_payload(model)


def test_jit_context_change_cannot_reuse_previous_positive_proof(monkeypatch):
    model, _ = _case()
    compiled, _ = _counting_compiler(monkeypatch)
    expected = _execute_full(model, "run")
    with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=model.dtype()):
        model.initialize()
    compiled.clear()
    with _count_handler_noise_calls() as sampled:
        actual = _execute_full(model, "run", jit=0)
    assert not compiled
    assert len(sampled) == 8
    _assert_traces_equal(actual, expected)
    with dn.ctx(JIT=1, BACKEND="eager", DEVICE="cpu", DTYPE=model.dtype()):
        model.initialize()
    with _eligibility_calls(model) as checks:
        repeated = _execute_full(model, "run")
    assert checks == {"fusion": 1, "noise": 1}
    _assert_traces_equal(repeated, expected)


def test_dtype_move_rebuilds_full_recording_proof(monkeypatch):
    model, _ = _case(dtype=torch.float64)
    reference, _ = _case(dtype=torch.float32)
    _counting_compiler(monkeypatch)
    _execute_full(model, "run")
    model.to(device="cpu", dtype=torch.float32)
    with dn.ctx(JIT=1, BACKEND="eager", DEVICE="cpu", DTYPE=torch.float32):
        model.initialize()
    with _eligibility_calls(model) as checks:
        actual = _execute_full(model, "run")
    expected = _execute_full(reference, "run")
    assert checks == {"fusion": 1, "noise": 1}
    assert all(value.dtype == torch.float32 for value in actual.values())
    # Floating-point waveform construction may differ after a dtype conversion;
    # the proof must follow new storage, not alter its mathematical result.
    for name in FIELDS:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_runtime_noise_inserted_between_calls_is_sampled(monkeypatch, entrypoint):
    model, _ = _case()
    _counting_compiler(monkeypatch)
    _execute_full(model, entrypoint)
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=model.dtype()):
        model.insert(_RuntimeNoise)
        model.initialize()
    reference, _ = _case(noise=True)
    with _count_handler_noise_calls() as sampled:
        actual = _execute_full(model, entrypoint)
    assert len(sampled) == 8
    expected = _execute_full(reference, entrypoint, callbacks=[_NoopPreStep()])
    _assert_traces_equal(actual, expected)
    torch.testing.assert_close(
        model.mech._RuntimeNoise.eta, reference.mech._RuntimeNoise.eta, rtol=0, atol=0
    )


@pytest.mark.parametrize("scope", ["instance", "class"])
@pytest.mark.parametrize("replacement", ["sampler", "step", "make_intra"])
def test_runtime_replacement_between_calls_cannot_reuse_positive_proof(
    monkeypatch, scope, replacement
):
    model, _ = _case()
    compiled, _ = _counting_compiler(monkeypatch)
    expected = _execute_full(model, "run")
    with dn.ctx(JIT=1, BACKEND="eager", DEVICE="cpu", DTYPE=model.dtype()):
        model.initialize()
    observed = []
    owner, attribute = {
        "sampler": (model.mech, "sample_runtime_noises_"),
        "step": (model.integrator, "step"),
        "make_intra": (model, "_call_make_intra"),
    }[replacement]
    if scope == "class":
        original = getattr(type(owner), attribute)

        def replaced(self, *args, **kwargs):
            if self is owner:
                observed.append(None)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(type(owner), attribute, replaced)
    else:
        original = getattr(owner, attribute)

        def replaced(*args, **kwargs):
            observed.append(None)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, attribute, replaced)
    compiled.clear()
    actual = _execute_full(model, "run")
    assert len(observed) == 8
    assert Counter(compiled) == {"make_intra": 8, "_step": 8}
    _assert_traces_equal(actual, expected)
    _assert_no_payload(model)


@pytest.mark.parametrize("entrypoint", ["run", "longrun"])
def test_custom_post_step_keeps_per_step_sampler_invalidation(monkeypatch, entrypoint):
    compiled, _ = _counting_compiler(monkeypatch)

    class ReplaceSampler(Callback):
        def __init__(self):
            super().__init__()
            self.sample_times = []
            self.replaced = False

        def post_step_hook(self, model):
            if self.replaced:
                return
            self.replaced = True
            original = model.mech.sample_runtime_noises_

            def sample(*args, **kwargs):
                assert compiled[-1] == "make_intra"
                self.sample_times.append(model.t.detach().clone())
                return original(*args, **kwargs)

            model.mech.sample_runtime_noises_ = sample

    candidate, _ = _case()
    mutation = ReplaceSampler()
    actual = _execute_full(candidate, entrypoint, callbacks=[mutation])
    assert Counter(compiled) == {"_step_with_intra": 1, "make_intra": 7, "_step": 7}
    reference, _ = _case()
    control = ReplaceSampler()
    expected = _execute_full(reference, entrypoint, callbacks=[control, _NoopPreStep()])
    _assert_traces_equal(actual, expected)
    torch.testing.assert_close(
        torch.stack(mutation.sample_times),
        torch.stack(control.sample_times),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("phase", ["pre_loop", "pre_chunk", "post_chunk"])
def test_longrun_rechecks_after_external_chunk_hook_mutation(monkeypatch, phase):
    model, _ = _case()
    compiled, _ = _counting_compiler(monkeypatch)
    sampled = []

    class Replace(Callback):
        def __init__(self):
            super().__init__()
            self.chunks = 0
            self.replaced = False

        def replace(self, model):
            if self.replaced:
                return
            self.replaced = True
            original = model.mech.sample_runtime_noises_

            def sample(*args, **kwargs):
                assert compiled[-1] == "make_intra"
                sampled.append(model.t.detach().clone())
                return original(*args, **kwargs)

            model.mech.sample_runtime_noises_ = sample

        def pre_loop_hook(self, model):
            if phase == "pre_loop":
                self.replace(model)

        def pre_chunk_hook(self, model, timepoints=None):
            self.chunks += 1
            if phase == "pre_chunk" and self.chunks == 2:
                self.replace(model)

        def post_chunk_hook(self, model, timepoints=None):
            if phase == "post_chunk":
                self.replace(model)

    actual = _execute_full(model, "longrun", callbacks=[Replace()])
    fused = 0 if phase == "pre_loop" else 3
    expected_calls = {"make_intra": 8 - fused, "_step": 8 - fused}
    if fused:
        expected_calls["_step_with_intra"] = fused
    assert Counter(compiled) == expected_calls
    assert len(sampled) == 8 - fused
    assert actual["t"].shape == (9,)
    _assert_no_payload(model)


def test_class_overridden_recorder_cannot_hide_runtime_mutation(monkeypatch):
    model, _ = _case()
    compiled, _ = _counting_compiler(monkeypatch)
    original_hook = Recorder.post_step_hook
    original_sampler = model.mech.sample_runtime_noises_
    sampled = []
    changed = []

    def sample(*args, **kwargs):
        assert compiled[-1] == "make_intra"
        sampled.append(None)
        return original_sampler(*args, **kwargs)

    def hook(recorder, population):
        original_hook(recorder, population)
        if population is model and not changed:
            changed.append(True)
            model.mech.sample_runtime_noises_ = sample

    monkeypatch.setattr(Recorder, "post_step_hook", hook)
    actual = _execute_full(model, "run")
    assert Counter(compiled) == {"_step_with_intra": 1, "make_intra": 7, "_step": 7}
    assert len(sampled) == 7
    reference, _ = _case()
    expected = _execute_full(reference, "run", callbacks=[_NoopPreStep()])
    _assert_traces_equal(actual, expected)


def test_generated_recorder_source_change_rejects_static_proof(monkeypatch):
    model, _ = _case()
    recorder = dn.Recorder(list(FIELDS))
    generated = recorder._post_step_hook.__func__
    monkeypatch.setattr(generated, "__source__", generated.__source__ + "\n# edited\n")
    assert not population_core._intra_fusion_callbacks_stable(
        CallbackList([recorder]), model
    )


@pytest.mark.parametrize("phase", ["forward", "pre_forward"])
def test_global_module_hooks_reject_runtime_stability_without_calling_them(phase):
    model, _ = _case()
    observed = []
    register = (
        torch.nn.modules.module.register_module_forward_hook
        if phase == "forward"
        else torch.nn.modules.module.register_module_forward_pre_hook
    )
    handle = register(lambda *args: observed.append(None))
    try:
        assert not population_core._runtime_execution_stable(model.mech, model)
        assert not observed
    finally:
        handle.remove()


@pytest.mark.parametrize("kind", ["buffer", "parameter"])
def test_unrecorded_tensor_subclasses_reject_runtime_stability(kind):
    model, _ = _case()

    class BufferSubclass(torch.Tensor):
        pass

    class ParameterSubclass(torch.nn.Parameter):
        pass

    if kind == "buffer":
        value = torch.Tensor._make_subclass(BufferSubclass, torch.zeros(1), False)
        model.mech.hh.register_buffer("unused_proof_probe", value)
    else:
        value = ParameterSubclass(torch.zeros(1), requires_grad=False)
        model.mech.hh.register_parameter("unused_proof_probe", value)
    assert not population_core._runtime_execution_stable(model.mech, model)


def test_recorded_gate_class_descriptor_is_not_evaluated_by_static_proof(monkeypatch):
    model, _ = _case()
    mechanism = model.mech.hh
    gate, observed = mechanism.m, []

    def getter(owner):
        observed.append(owner)
        return gate

    monkeypatch.setattr(type(mechanism), "m", property(getter), raising=False)
    assert not population_core._intra_fusion_callbacks_stable(
        CallbackList([dn.Recorder(list(FIELDS))]), model
    )
    assert not observed


def test_generated_append_helper_replacement_rejects_static_proof(monkeypatch):
    model, _ = _case()
    observed = []
    original = callbacks_module._append_tensor

    def append(*args, **kwargs):
        observed.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(callbacks_module, "_append_tensor", append)
    assert not population_core._intra_fusion_callbacks_stable(
        CallbackList([dn.Recorder(list(FIELDS))]), model
    )
    assert not observed


@pytest.mark.parametrize("hook", ["pre_step_hook", "post_step_hook"])
def test_callback_list_class_hook_replacement_rejects_static_proof(monkeypatch, hook):
    model, _ = _case()
    observed = []
    original = getattr(CallbackList, hook)

    def replaced(callbacks, population):
        observed.append(None)
        return original(callbacks, population)

    monkeypatch.setattr(CallbackList, hook, replaced)
    assert not population_core._intra_fusion_callbacks_stable(
        CallbackList([dn.Recorder(list(FIELDS))]), model
    )
    assert not observed
