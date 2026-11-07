"""MPS stimulus fusion preserves authored hooks and native sampling order."""

from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.callbacks import Callback
from dendra.models.core import Population
from dendra.models.integrators.core import Integrator
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mod import hh

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS is unavailable"
    ),
]
DT = 0.01
STEPS = 4


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(DEVICE="mps", DTYPE=torch.float32, REQUIRE_GRAD=1, JIT=0),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


class _NoopPreStep(Callback):
    def pre_step_hook(self, model):
        pass


class _MPSRuntimeNoise(Mechanism):
    Mechanism.RANGENOISE("eta", distribution="normal", mu=0.0, sigma=1.0, seed=715)


def _case(*, noise=False, training=True):
    model = dn.Population(
        N=3,
        C=4,
        device="mps",
        dtype=torch.float32,
        celsius=6.3,
        v_init=-62.0,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.insert(hh)
    if noise:
        model.insert(_MPSRuntimeNoise)
    waveform = (
        dn.mono_rect(
            amp=torch.tensor([0.1, 0.2, 0.3], device="mps", dtype=torch.float32),
            delay=0.0,
            pw=1.0,
        )
        * dn.units.nA
    )
    model[:, 0].inject(waveform)
    model[:, 2].inject(dn.constant(value=0.05 * dn.units.nA))
    model[:, 0].inject(dn.constant(value=0.02 * dn.units.nA))
    model.requires_grad_(False)
    amplitude = dict(waveform.named_parameters())["waveforms.0.amp"]
    amplitude.requires_grad_(training)
    model.train(training)
    model.initialize()
    return model, amplitude


def _run(model, *, callbacks=(), backend="aot_eager"):
    recorder = dn.Recorder(("v",))
    with dn.ctx(
        JIT=1, BACKEND=backend, FULLGRAPH=True, REQUIRE_GRAD=int(model.training)
    ):
        model.run(tstop=STEPS * DT, dt=DT, callbacks=[recorder, *callbacks])
    return recorder.stack("v")


@contextmanager
def _observe_runtime(model):
    """Observe canonical eager boundaries without wrapping guarded methods."""
    events = []
    codes = {
        Population._call_make_intra.__code__: (model, "scatter"),
        MechanismHandler.sample_runtime_noises_.__code__: (
            model.integrator.mech,
            "noise",
        ),
        Integrator._kernel.__code__: (model.integrator, "kernel"),
    }
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            owner, name = codes[frame.f_code]
            if frame.f_locals.get("self") is owner:
                events.append(
                    name if name != "kernel" else "kernel:" + frame.f_locals["name"]
                )
        if callable(previous):
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield events
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("backend", ("aot_eager", "inductor"))
def test_mps_fused_stimulus_trace_and_waveform_gradient_match_forced_split(backend):
    fused, fused_amplitude = _case()
    split, split_amplitude = _case()
    with _observe_runtime(fused) as fused_events:
        actual = _run(fused, backend=backend)
    with _observe_runtime(split) as split_events:
        expected = _run(split, callbacks=(_NoopPreStep(),), backend=backend)
    assert fused_events == ["kernel:_step_with_intra"] * STEPS
    assert split_events == ["scatter", "noise", "kernel:_step"] * STEPS
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    actual_gradient = torch.autograd.grad(actual.square().mean(), fused_amplitude)[0]
    expected_gradient = torch.autograd.grad(expected.square().mean(), split_amplitude)[
        0
    ]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-4, atol=1e-7)
    assert torch.isfinite(actual_gradient).all()
    assert torch.all(expected_gradient != 0)
    assert any(
        key[0] == "_step_with_intra" for key in fused.integrator._compiled_kernels
    )
    assert not any(
        key[0] == "_step_with_intra" for key in split.integrator._compiled_kernels
    )


@pytest.mark.parametrize("hook", ("pre", "post"))
def test_mps_fusion_keeps_index_mutation_on_the_next_accepted_step(hook):
    candidate, _ = _case(training=False)
    reference, _ = _case(training=False)

    class MutateBefore(Callback):
        def pre_step_hook(self, model):
            events.append("pre")
            model.intra.indices[0][-1].fill_(1)

    class MutateAfter(Callback):
        def post_step_hook(self, model):
            events.append("post")
            model.intra.indices[0][-1].fill_(1)

    callback = MutateBefore() if hook == "pre" else MutateAfter()
    with torch.no_grad():
        with _observe_runtime(candidate) as events:
            actual = _run(candidate, callbacks=(callback,))
        expected = [reference.v.clone()]
        for _ in range(STEPS):
            reference.step(DT)
            expected.append(reference.v.clone())
            reference.intra.indices[0][-1].fill_(1)
    expected_events = (
        ["scatter", "pre", "noise", "kernel:_step"]
        if hook == "pre"
        else ["kernel:_step_with_intra", "post"]
    )
    assert events == expected_events * STEPS
    torch.testing.assert_close(actual, torch.stack(expected), rtol=3e-5, atol=2e-5)


def test_mps_runtime_noise_forbids_fusion_and_keeps_scatter_hook_sampling_order():
    candidate, _ = _case(noise=True, training=False)
    reference, _ = _case(noise=True, training=False)

    class Observe(Callback):
        def post_step_hook(self, model):
            events.append("post")

    with torch.no_grad():
        with _observe_runtime(candidate) as events:
            actual = _run(candidate, callbacks=(Observe(),))
        expected = _run(reference, callbacks=(_NoopPreStep(),))
    # A post-step observer permits fusion. Runtime NOISE itself must force the
    # scatter/sampling split, before the ordinary compiled numerical kernel.
    assert events == ["scatter", "noise", "kernel:_step", "post"] * STEPS
    assert not any(
        key[0] == "_step_with_intra" for key in candidate.integrator._compiled_kernels
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        candidate.mech._MPSRuntimeNoise.eta,
        reference.mech._MPSRuntimeNoise.eta,
        rtol=0,
        atol=0,
    )
