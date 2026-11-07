"""Native MPS indexed stimulus inputs retain aliases and continuation."""

from __future__ import annotations

import io

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.callbacks import Callback
from dendra.models.stim._assembly import _IndexedCurrentSamples
from tests.test_indexed_intra_inputs import (
    STEPS,
    _model,
    _observe_stimulus_payloads,
    _parameters,
    _run,
    _SplitIntra,
)

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS is unavailable"
    ),
]


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(DEVICE="mps", DTYPE=torch.float32, JIT=0, REQUIRE_GRAD=1),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "phase,mutation,runner",
    (
        ("post", "values", "run"),
        ("post", "storage", "run"),
        ("pre_loop", "storage", "run"),
        ("pre_chunk", "storage", "longrun"),
    ),
)
def test_mps_inference_keeps_live_parameter_tape_aliases(phase, mutation, runner):
    candidate, waveforms = _model("mps", live=True, shared=True)
    reference, reference_waveforms = _model("mps", live=True, shared=True)
    for waveform in (*waveforms, *reference_waveforms):
        waveform.samples.requires_grad_(True)
    original_parameters = tuple(waveform.samples for waveform in waveforms)

    class MutateFuture(Callback):
        def __init__(self, sources):
            super().__init__()
            self.sources = sources
            self.mutated = False

        def mutate(self):
            if self.mutated:
                return
            self.mutated = True
            if mutation == "values":
                self.sources[0].samples[1:].mul_(3.0)
                self.sources[1].samples[1:].neg_()
            else:
                for source in self.sources[:2]:
                    tape = source.outputs["tape"]
                    tape.set_(tape.new_full(tape.shape, 0.9 * dn.units.nA))

        def pre_loop_hook(self, model):
            if phase == "pre_loop":
                self.mutate()

        def pre_chunk_hook(self, model, times):
            if phase == "pre_chunk":
                self.mutate()

        def post_step_hook(self, model):
            if phase == "post":
                self.mutate()

    with _observe_stimulus_payloads(candidate) as payloads:
        actual = _run(
            candidate,
            runner=runner,
            callbacks=(MutateFuture(waveforms),),
            backend="inductor",
        )
    expected = _run(
        reference,
        runner=runner,
        callbacks=(MutateFuture(reference_waveforms), _SplitIntra()),
        backend="inductor",
        # Legacy compiled slices also hit PyTorch FakeTensor's no-grad view
        # conversion error after a base storage rebind. Native eager execution
        # provides the original input-alias semantics for these cases.
        jit=mutation != "storage",
    )
    assert len(payloads) == STEPS
    assert all(isinstance(payload, _IndexedCurrentSamples) for payload in payloads)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    assert all(
        waveform.samples is original
        for waveform, original in zip(waveforms, original_parameters, strict=True)
    )
    assert waveforms[0].samples is waveforms[2].samples
    assert waveforms[0].samples is not waveforms[1].samples


@pytest.mark.parametrize("steps,time_stride", ((1, 1), (STEPS, 4)))
def test_mps_raw_parameter_tapes_keep_storage_before_first_compiled_step(
    steps, time_stride
):
    candidate, waveforms = _model("mps", live=True, shared=True)
    reference, reference_waveforms = _model("mps", live=True, shared=True)
    for sources in (waveforms, reference_waveforms):
        for parameter in dict.fromkeys(source.samples for source in sources):
            with torch.no_grad():
                values = (
                    parameter[:steps]
                    .unsqueeze(-1)
                    .expand(steps, time_stride)
                    .clone()[:, 0]
                )
                parameter.set_(values)
            parameter.requires_grad_(True)

    class RebindStorage(Callback):
        def __init__(self, sources):
            super().__init__()
            self.sources = sources

        def pre_loop_hook(self, model):
            for source in self.sources[:2]:
                tape = source.outputs["tape"]
                assert tape is source.samples
                tape.set_(tape.new_full(tape.shape, 0.9 * dn.units.nA))

    with _observe_stimulus_payloads(candidate) as payloads:
        actual = _run(
            candidate,
            steps=steps,
            callbacks=(RebindStorage(waveforms),),
            backend="inductor",
        )
    expected = _run(
        reference,
        steps=steps,
        callbacks=(RebindStorage(reference_waveforms), _SplitIntra()),
        # Compiled legacy views hit FakeTensor's no-grad base-rebind error;
        # the eager oracle preserves the original storage-alias semantics.
        jit=False,
    )
    assert len(payloads) == steps
    assert all(isinstance(payload, _IndexedCurrentSamples) for payload in payloads)
    assert all(source.samples.requires_grad for source in waveforms)
    assert all(
        not hasattr(source.samples, "_dynamo_dynamic_indices") for source in waveforms
    )
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)


def test_mps_inference_mode_uses_indexed_inputs_with_live_parameter_tapes():
    candidate, waveforms = _model("mps", live=True, shared=True)
    reference, reference_waveforms = _model("mps", live=True, shared=True)
    for source in (*waveforms, *reference_waveforms):
        source.samples.requires_grad_(True)
    with torch.inference_mode(), _observe_stimulus_payloads(candidate) as payloads:
        actual = _run(candidate, backend="inductor")
    with torch.no_grad():
        expected = _run(reference, callbacks=(_SplitIntra(),), backend="inductor")
    assert len(payloads) == STEPS
    assert all(isinstance(payload, _IndexedCurrentSamples) for payload in payloads)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)


@pytest.mark.parametrize("steps", (0, 1, STEPS))
def test_mps_indexed_run_and_chunk_tails_match_split_execution(steps):
    candidate, _ = _model("mps")
    chunked, _ = _model("mps")
    reference, _ = _model("mps")
    actual = _run(candidate, steps=steps, backend="inductor")
    chunks = _run(chunked, steps=steps, runner="longrun", backend="inductor")
    expected = _run(
        reference, steps=steps, callbacks=(_SplitIntra(),), backend="inductor"
    )
    assert actual.shape[0] == steps + 1
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    torch.testing.assert_close(chunks, expected, rtol=3e-5, atol=2e-5)
    torch.testing.assert_close(candidate.t, reference.t, rtol=0, atol=0)
    torch.testing.assert_close(chunked.t, reference.t, rtol=0, atol=0)


def test_mps_indexed_inputs_survive_serialized_checkpoint_continuation():
    prefix, _ = _model("mps")
    _run(prefix, steps=3, backend="inductor")
    checkpoint = prefix.state_dict_for_checkpoint()
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    restored, _ = _model("mps")
    restored.restore_dict_from_checkpoint(torch.load(stream, weights_only=False))
    expected = _run(prefix, steps=4, backend="inductor")
    actual = _run(restored, steps=4, backend="inductor")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(restored.v, prefix.v, rtol=0, atol=0)
    torch.testing.assert_close(restored.t, prefix.t, rtol=0, atol=0)


def test_mps_training_keeps_waveform_and_checkpoint_gradients():
    direct, direct_waveforms = _model("mps", training=True, shared=True)
    checkpointed, checkpoint_waveforms = _model("mps", training=True, shared=True)
    with _observe_stimulus_payloads(direct) as payloads:
        _run(direct, backend="inductor")
    assert len(payloads) == STEPS
    assert all(type(payload) is list for payload in payloads)
    _run(checkpointed, runner="checkpointed", backend="inductor")
    torch.testing.assert_close(checkpointed.v, direct.v, rtol=3e-5, atol=2e-5)
    actual_gradients = torch.autograd.grad(
        checkpointed.v.square().sum(), _parameters(checkpoint_waveforms)
    )
    expected_gradients = torch.autograd.grad(
        direct.v.square().sum(), _parameters(direct_waveforms)
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-7)
        assert expected.abs().max().item() > 0


def test_mps_custom_public_preparation_keeps_the_slice_route():
    class PreparedPopulation(dn.Population):
        def prep_intra(self, intra, n, dt):
            self.preparation_calls = getattr(self, "preparation_calls", 0) + 1
            return super().prep_intra(intra, n, dt)

    candidate, _ = _model("mps", population_cls=PreparedPopulation)
    reference, _ = _model("mps")
    with _observe_stimulus_payloads(candidate) as payloads:
        actual = _run(candidate, backend="inductor")
    expected = _run(reference, callbacks=(_SplitIntra(),), backend="inductor")
    assert candidate.preparation_calls == 1
    assert len(payloads) == STEPS
    assert all(type(payload) is list for payload in payloads)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)


def test_mps_indexed_inputs_refresh_mutated_and_rebound_target_coordinates():
    candidate, _ = _model("mps")
    reference, _ = _model("mps")

    class MoveTargets(Callback):
        def __init__(self):
            super().__init__()
            self.accepted = 0

        def post_step_hook(self, model):
            self.accepted += 1
            if self.accepted == 1:
                model.intra.indices[0][-1].fill_(1)
            elif self.accepted == 2:
                replacement = tuple(item.clone() for item in model.intra.indices[1])
                replacement[-1].fill_(3)
                model.intra.indices[1] = replacement

    with _observe_stimulus_payloads(candidate) as payloads:
        actual = _run(candidate, callbacks=(MoveTargets(),), backend="inductor")
    expected = _run(
        reference, callbacks=(MoveTargets(), _SplitIntra()), backend="inductor"
    )
    assert len(payloads) == STEPS
    assert all(isinstance(payload, _IndexedCurrentSamples) for payload in payloads)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)


def _poisson_model():
    generator = torch.Generator().manual_seed(715)
    model = dn.Population(
        N=2,
        C=4,
        device="mps",
        dtype=torch.float32,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    from dendra.models.mod import pas

    model.insert(pas, g=0.001, e=-65.0)
    for contact, compartment in enumerate((0, 2, 0)):
        waveform = dn.mono_rect(
            amp=(contact + 1) * 0.075 * dn.units.nA, pw=0.03
        ).poisson(
            interval=0.035,
            n=5,
            start=-0.015,
            noise=0.7,
            randomize_every_call=True,
            generator=generator,
        )
        model[:, compartment].inject(waveform)
    model.requires_grad_(False)
    model.eval()
    model.initialize()
    return model, generator


@pytest.mark.parametrize(
    "runner,chunklength", (("run", 3), ("longrun", 3), ("longrun", 2))
)
def test_mps_indexed_poisson_tapes_keep_shared_generator_draw_order(
    runner, chunklength
):
    candidate, generator = _poisson_model()
    reference, reference_generator = _poisson_model()
    with _observe_stimulus_payloads(candidate) as payloads:
        actual = _run(
            candidate,
            runner=runner,
            chunklength=chunklength,
            backend="inductor",
        )
    expected = _run(
        reference,
        runner=runner,
        chunklength=chunklength,
        callbacks=(_SplitIntra(),),
        backend="inductor",
    )
    assert len(payloads) == STEPS
    if runner == "run" or chunklength == 3:
        assert all(isinstance(payload, _IndexedCurrentSamples) for payload in payloads)
    else:
        assert all(
            isinstance(payload, _IndexedCurrentSamples) for payload in payloads[:-1]
        )
        assert type(payloads[-1]) is list
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
    torch.testing.assert_close(
        generator.get_state(), reference_generator.get_state(), rtol=0, atol=0
    )
