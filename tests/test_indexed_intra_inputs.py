"""Indexed stimulus inputs preserve native source ownership and training."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import MethodType

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.callbacks import Callback
from dendra.models.core import _unbind_intra_tapes
from dendra.models.integrators.core import Integrator
from dendra.models.mod import pas
from dendra.models.stim._assembly import (
    _aligned_current_indices,
    _indexed_current_tapes,
    _IndexedCurrentSamples,
)
from dendra.models.stim.intra import Intra
from dendra.models.stim.waveform import Waveform
from tests.test_injection_assembly import _ordered_reference

pytestmark = pytest.mark.cpu
DT = 0.025
STEPS = 7


class _SplitIntra(Callback):
    def pre_step_hook(self, model):
        pass


class _AliasedTape(Waveform):
    """Native waveform returning a live view, not a materialized time tape."""

    def __init__(self, values):
        super().__init__()
        self.register_parameter("samples", torch.nn.Parameter(values))
        self.outputs = {}

    def fn(self, t):
        value = (
            self.samples
            if t.numel() == self.samples.numel()
            else self.samples.narrow(-1, 0, t.numel())
        )
        self.outputs["tape"] = value
        return value


def _model(
    device="cpu",
    *,
    live=False,
    training=False,
    shared=False,
    population_cls=dn.Population,
):
    dtype = torch.float32 if device == "mps" else torch.float64
    with dn.ctx(JIT=0, DEVICE=device, DTYPE=dtype, REQUIRE_GRAD=1):
        model = population_cls(
            N=2,
            C=4,
            device=device,
            dtype=dtype,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(pas, g=0.001, e=-65.0)
        waveforms = []
        for index, compartment in enumerate((0, 2, 0)):
            if live:
                waveform = _AliasedTape(
                    torch.linspace(
                        0.025 * (index + 1),
                        0.075 * (index + 1),
                        STEPS,
                        device=device,
                        dtype=dtype,
                    )
                    * dn.units.nA
                )
            else:
                waveform = dn.mono_rect(
                    amp=0.075 * (index + 1) * dn.units.nA,
                    delay=0.0,
                    pw=1.0,
                )
            waveforms.append(waveform)
            model[:, compartment].inject(waveform)
        if shared:
            if live:
                waveforms[2].samples = waveforms[0].samples
            else:
                waveforms[2].amp = waveforms[0].amp
        model.requires_grad_(False)
        for waveform in waveforms:
            waveform.requires_grad_(False)
            amplitude = waveform.samples if live else waveform.amp
            amplitude.requires_grad_(training)
        model.train(training)
        model.initialize()
    return model, waveforms


def _parameters(waveforms):
    return tuple(
        dict.fromkeys(
            parameter
            for waveform in waveforms
            for parameter in waveform.parameters()
            if parameter.requires_grad
        )
    )


def _run(
    model,
    *,
    steps=STEPS,
    runner="run",
    callbacks=(),
    backend="aot_eager",
    jit=True,
    chunklength=3,
):
    recorder = dn.Recorder(("v",))
    with (
        torch_compiler_warning_context(),
        dn.ctx(
            JIT=int(jit),
            BACKEND=backend,
            FULLGRAPH=True,
            DEVICE=model.device(),
            DTYPE=model.dtype(),
            REQUIRE_GRAD=int(model.training),
        ),
    ):
        if runner == "run":
            model.run(tstop=steps * DT, dt=DT, callbacks=[recorder, *callbacks])
        elif runner == "longrun":
            model.longrun(
                tstop=steps * DT,
                dt=DT,
                chunklength=chunklength,
                callbacks=[recorder, *callbacks],
            )
            if steps == 0:
                # longrun(0) emits no Recorder samples; compare final state.
                return model.v.unsqueeze(0)
        else:
            model.longrun_checkpointed(
                tstop=steps * DT,
                dt=DT,
                chunklength=chunklength,
                safe_checkpoint=True,
            )
            return model.v.unsqueeze(0)
    return recorder.stack("v")


def _backward_nodes(value):
    pending = [value.grad_fn]
    seen = set()
    names = []
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.append(type(node).__name__)
        pending.extend(child for child, _ in node.next_functions)
    return names


@contextmanager
def _observe_stimulus_payloads(model):
    """Inspect the actual eager dispatch boundary without replacing methods."""
    payloads = []
    code = Integrator._kernel.__code__
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is code
            and frame.f_locals.get("self") is model.integrator
            and frame.f_locals.get("name") == "_step_with_intra"
        ):
            payloads.append(frame.f_locals["args"][3])
        if callable(previous):
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield payloads
    finally:
        sys.setprofile(previous)


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with dn.ctx(JIT=0, DEVICE="cpu", DTYPE=torch.float64, REQUIRE_GRAD=1):
            yield
    finally:
        torch.set_num_threads(previous)


def test_training_keeps_one_unbind_per_waveform_and_shared_parameter_gradients():
    candidate, waveforms = _model(live=True, training=True, shared=True)
    reference, reference_waveforms = _model(live=True, training=True, shared=True)
    actual = _run(candidate)
    expected = _run(reference, callbacks=(_SplitIntra(),))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    # Unbinding each evaluated tape once lets backward collect the timestep
    # gradients together. Per-step gathers from whole tapes would lose this
    # structure and produce one tape-sized gradient for every accepted step.
    names = _backward_nodes(actual)
    assert sum(name.startswith("UnbindBackward") for name in names) == len(waveforms)
    parameters = _parameters(waveforms)
    assert len(parameters) == 2
    assert waveforms[0].samples is waveforms[2].samples
    weights = torch.arange(1, actual.numel() + 1, dtype=actual.dtype).reshape_as(actual)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), parameters)
    expected_gradients = torch.autograd.grad(
        (expected * weights).sum(), _parameters(reference_waveforms)
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12
        )
        assert torch.all(actual_gradient != 0)


def test_training_checkpointed_chunks_preserve_independent_waveform_gradients():
    direct, direct_waveforms = _model(training=True, shared=True)
    checkpointed, checkpoint_waveforms = _model(training=True, shared=True)
    _run(direct)
    _run(checkpointed, runner="checkpointed")
    torch.testing.assert_close(checkpointed.v, direct.v, rtol=1e-12, atol=1e-12)
    actual_gradients = torch.autograd.grad(
        checkpointed.v.square().sum(), _parameters(checkpoint_waveforms)
    )
    expected_gradients = torch.autograd.grad(
        direct.v.square().sum(), _parameters(direct_waveforms)
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


def test_indexed_tape_views_keep_live_values_but_freeze_storage_metadata():
    original = torch.arange(8, dtype=torch.float64)
    with torch.no_grad():
        captured = _indexed_current_tapes(
            [original], 8, device=original.device, dtype=original.dtype
        )
        assert captured is not None
        view = captured[0]
        assert view is not original
        assert (
            view.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        )
        assert view.shape == original.shape and view.stride() == original.stride()
        assert not hasattr(original, "_dynamo_dynamic_indices")
        assert view._dynamo_dynamic_indices == {0}
        original.add_(3)
        torch.testing.assert_close(view, original, rtol=0, atol=0)
        expected = view.clone()
        original.set_(torch.full((12,), 99.0, dtype=original.dtype))
        assert view.shape == (8,)
        torch.testing.assert_close(view, expected, rtol=0, atol=0)
        assert (
            view.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
        )


@pytest.mark.parametrize("steps", (0, 1, 11))
def test_framework_owned_indices_have_aligned_sample_starts(steps):
    indices = _aligned_current_indices(steps, device=torch.device("cpu"))
    assert len(indices) == steps
    assert [index.item() for index in indices] == list(range(steps))
    assert all(index.shape == (1,) and index.dtype == torch.int64 for index in indices)
    assert all(index.data_ptr() % 16 == 0 for index in indices)


@pytest.mark.parametrize(
    "case", ("zero", "scalar", "short", "unaligned", "dtype", "subclass")
)
def test_unsupported_tape_contracts_keep_the_slice_fallback(case):
    class AuthoredTensor(torch.Tensor):
        pass

    tape = torch.arange(8, dtype=torch.float64)
    steps = 3
    if case == "zero":
        steps = 0
    elif case == "scalar":
        tape = tape[0]
    elif case == "short":
        tape = tape[:2]
    elif case == "unaligned":
        tape = tape[1:]
        assert tape.data_ptr() % 16 != 0
    elif case == "dtype":
        tape = tape.to(torch.float32)
    else:
        tape = tape.as_subclass(AuthoredTensor)
    with torch.no_grad():
        assert (
            _indexed_current_tapes(
                [tape], steps, device=torch.device("cpu"), dtype=torch.float64
            )
            is None
        )


def test_indexed_tapes_are_disabled_with_gradients_enabled():
    tape = torch.arange(8, dtype=torch.float64, requires_grad=True)
    assert (
        _indexed_current_tapes([tape], 8, device=tape.device, dtype=tape.dtype) is None
    )


def test_indexed_assembly_reuses_a_fullgraph_for_dynamic_tape_extents():
    model, _ = _model()
    intra = Intra(model, model.injections)
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    def assemble(tapes, index):
        return intra(_IndexedCurrentSamples(tapes, index), prepared)

    compiled = torch.compile(assemble, backend=backend, fullgraph=True)
    with torch_compiler_warning_context(), torch.no_grad():
        for length, step in ((7, 3), (11, 9)):
            originals = [
                (torch.arange(2 * length, dtype=torch.float64).reshape(2, length) + 1)
                * (contact + 0.25)
                for contact in range(3)
            ]
            captured = _indexed_current_tapes(
                originals, length, device=intra.device, dtype=intra.dtype
            )
            assert captured is not None
            samples = [tape[..., step] for tape in originals]
            prepared = intra._prepare_assembly(samples, intra.indices)
            index = _aligned_current_indices(length, device=intra.device)[step]
            actual = compiled(captured, index)
            torch.testing.assert_close(
                actual, _ordered_reference(intra, samples), rtol=0, atol=0
            )
    assert len(graphs) == 1


def test_authored_unbind_is_called_once_and_keeps_its_output():
    model, _ = _model()
    intra = Intra(model, model.injections)
    tape = torch.arange(4, dtype=intra.dtype)
    calls = []

    def unbind(self, dim):
        calls.append(dim)
        return tuple(
            torch.tensor(100.0 + index, dtype=self.dtype) for index in range(4)
        )

    tape.unbind = MethodType(unbind, tape)
    with torch.no_grad():
        stims, captured = _unbind_intra_tapes([tape], intra, 4, indexed=True)
    assert captured is None
    assert calls == [-1]
    assert [sample.item() for sample in stims[0]] == [100.0, 101.0, 102.0, 103.0]


def test_parameter_tape_capture_keeps_a_private_metadata_view():
    parameter = torch.nn.Parameter(torch.arange(7, dtype=torch.float64))
    with torch.no_grad():
        captured = _indexed_current_tapes(
            [parameter], 7, device=parameter.device, dtype=parameter.dtype
        )
        assert captured is not None
        assert type(captured[0]) is torch.Tensor
        assert (
            captured[0].untyped_storage().data_ptr()
            == parameter.untyped_storage().data_ptr()
        )
        assert not hasattr(parameter, "_dynamo_dynamic_indices")
        parameter[1:].mul_(2)
        expected = parameter.clone()
        parameter.set_(parameter.new_full(parameter.shape, 99.0))
        torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ("single", "aligned", "broadcast"))
@pytest.mark.parametrize("requires_grad", (False, True))
def test_indexed_capture_skips_unneeded_gathers_but_keeps_parameter_aliases(
    layout, requires_grad
):
    if layout == "single":
        values = torch.arange(1, dtype=torch.float64)
    elif layout == "aligned":
        values = torch.arange(14, dtype=torch.float64).reshape(7, 2)[:, 0]
        assert values.stride(-1) * values.element_size() == 16
    else:
        values = torch.ones(1, dtype=torch.float64).expand(7)
        assert values.stride(-1) == 0
    tape = torch.nn.Parameter(values, requires_grad=requires_grad)
    with torch.no_grad():
        captured = _indexed_current_tapes(
            [tape], tape.numel(), device=tape.device, dtype=tape.dtype
        )
        if not requires_grad:
            assert captured is None
        else:
            # A raw Parameter still needs an independent metadata view when
            # hooks may rebind its storage before the first compiled step.
            assert captured is not None
            assert not captured[0].requires_grad
            assert captured[0].stride() == tape.stride()
            assert (
                captured[0].untyped_storage().data_ptr()
                == tape.untyped_storage().data_ptr()
            )
            expected = tape.clone()
            tape.set_(tape.new_full(tape.shape, 99.0))
            torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    assert tape.requires_grad == requires_grad
    assert not hasattr(tape, "_dynamo_dynamic_indices")


@pytest.mark.parametrize("aligned_first", (False, True))
def test_mixed_time_strides_keep_indexed_capture_and_source_order(aligned_first):
    aligned = torch.arange(14, dtype=torch.float64).reshape(7, 2)[:, 0]
    ordinary = torch.arange(7, dtype=torch.float64) + 100
    originals = (aligned, ordinary) if aligned_first else (ordinary, aligned)
    with torch.no_grad():
        captured = _indexed_current_tapes(
            originals, 7, device=ordinary.device, dtype=ordinary.dtype
        )
    assert captured is not None
    for actual, original in zip(captured, originals, strict=True):
        assert actual.stride() == original.stride()
        assert (
            actual.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("method", ("as_strided", "detach"))
def test_authored_metadata_method_is_not_invoked_by_indexed_preparation(method):
    tape = torch.arange(7, dtype=torch.float64)
    calls = []

    def authored(*args, **kwargs):
        calls.append(None)
        raise AssertionError("authored as_strided must not execute during preparation")

    setattr(tape, method, authored)
    with torch.no_grad():
        assert (
            _indexed_current_tapes([tape], 7, device=tape.device, dtype=tape.dtype)
            is None
        )
    assert not calls


@pytest.mark.parametrize("inference_mode", (False, True))
def test_forward_dual_tapes_keep_the_original_slice_route(inference_mode):
    primal = torch.arange(7, dtype=torch.float64)
    with torch.autograd.forward_ad.dual_level():
        tape = torch.autograd.forward_ad.make_dual(primal, torch.ones_like(primal))
        with torch.inference_mode(inference_mode), torch.no_grad():
            # Inference mode hides existing tangents from an ordinary
            # unpack_dual query; preparation must still detect the dual and
            # avoid discarding it when creating its private metadata view.
            assert (
                _indexed_current_tapes([tape], 7, device=tape.device, dtype=tape.dtype)
                is None
            )
