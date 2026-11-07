"""Native Population deterministic packets preserve drive ownership and timing."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.callbacks import Callback
from dendra.models.mod import pas
from dendra.units import nA
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_mechanism_injection_execution import _InjectedLeak
from tests.func.test_native_handoff import _capture, _unchanged
from tests.func.test_stimulation_rng_sharing import (
    _population as _stochastic_population,
)

DT, STEPS = 0.025, 5
pytestmark = pytest.mark.cpu
_METHODS = ("step", "run", "longrun", "longrun_checkpointed")


@pytest.fixture(autouse=True)
def _environment():
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)


def _model(*, batch=False, injection=None):
    model = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-55.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(pas.rename("leak"), g=0.001, e=-70.0)
    if injection == "mechanism":
        model.insert(_InjectedLeak.rename("owned"))
    if batch:
        model.batch(2)
    waveform = None
    if injection is not None:
        waveform = dn.mono_rect(
            amp=0.2 if injection == "mechanism" else 0.2 * nA, pw=0.2
        )
        model[..., 1].inject(waveform)
    model.train()
    model.initialize(DT)
    return model, waveform


def _tapes(model, steps, *, broadcast=False):
    shape = (1, model.shape[-1]) if broadcast else model.shape
    ve = torch.linspace(
        -0.2,
        0.3,
        steps * torch.Size(shape).numel(),
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *shape)
    current_shape = (1, 1) if broadcast else model.shape
    intra = torch.linspace(
        0.8e-10,
        1.2e-10,
        steps * torch.Size(current_shape).numel(),
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *current_shape)
    return dn.func.Inputs(ve.requires_grad_(), intra.requires_grad_())


def _call(model, method, packet, *, steps=STEPS, **options):
    if method == "step":
        return model.step(DT, inputs=packet, **options)
    options.update(tstop=steps * DT, dt=DT, inputs=packet)
    if method != "run":
        options["chunklength"] = 3
    if method == "longrun_checkpointed":
        options["safe_checkpoint"] = True
    return getattr(model, method)(**options)


def _reference(model, packet, steps):
    functional, tensors = dn.func.make_functional(model, dt=DT)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    expected, _ = dn.func.run(functional, initial, inputs=packet, steps=steps)
    return functional, expected


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("broadcast", (False, True), ids=("dense", "broadcast"))
def test_population_inputs_match_functional_values_and_all_drive_gradients(
    method, broadcast
):
    model, _ = _model(batch=broadcast)
    steps = 1 if method == "step" else STEPS
    tapes = _tapes(model, steps, broadcast=broadcast)
    frozen = _snapshot(tapes)
    functional, expected = _reference(model, tapes, steps)
    source = model.mech.leak.g_param
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(), (source, tapes.ve, tapes.intra)
    )
    packet = dn.func.Inputs(tapes.ve[0], tapes.intra[0]) if method == "step" else tapes

    result = _call(model, method, packet, steps=steps)
    if method == "step":
        assert result is model
    torch.testing.assert_close(model.v, expected["integrator"]["v"])
    _close(functional.extract(model).state, expected)
    actual = torch.autograd.grad(
        model.v.square().mean(), (source, tapes.ve, tapes.intra)
    )
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    _assert_packet_equal(tapes, frozen)


@pytest.mark.parametrize("method", ("longrun", "longrun_checkpointed"))
def test_training_chunks_retain_input_gradients_inside_an_outer_no_grad(method):
    model, _ = _model()
    packet = _tapes(model, STEPS)
    _, expected = _reference(model, packet, STEPS)
    wanted = torch.autograd.grad(expected["integrator"]["v"].square().sum(), packet)
    with torch.no_grad():
        _call(model, method, packet)
    actual = torch.autograd.grad(model.v.square().sum(), packet)
    _close(actual, wanted)


def test_population_run_infers_a_canonical_horizon_and_keeps_legacy_ve_authoritative():
    model, _ = _model()
    legacy, _ = _model()
    source = _tapes(model, 3).ve.to(torch.float32)
    legacy.run(ve=source, tstop=99 * DT, dt=DT)
    inputs = dn.func.Inputs(ve=source.to(torch.float64))
    model.run(inputs=inputs, dt=DT)
    torch.testing.assert_close(model.v, legacy.v)
    torch.testing.assert_close(model.t, model.t.new_tensor(3 * DT))
    with pytest.raises(ValueError):
        model.run(inputs=inputs, tstop=99 * DT, dt=DT)


def test_legacy_ve_and_canonical_current_keep_the_voltage_horizon_and_all_vjps():
    model, _ = _model()
    packet = _tapes(model, 3)
    functional, expected = _reference(model, packet, 3)
    targets = model.mech.leak.g_param, packet.ve, packet.intra
    wanted = torch.autograd.grad(expected["integrator"]["v"].square().mean(), targets)
    frozen = _snapshot(packet)

    model.run(
        ve=packet.ve,
        inputs=dn.func.Inputs(intra=packet.intra),
        tstop=99 * DT,
        dt=DT,
    )
    _close(functional.extract(model).state, expected)
    torch.testing.assert_close(model.t, model.t.new_tensor(3 * DT))
    actual = torch.autograd.grad(model.v.square().mean(), targets)
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    _assert_packet_equal(packet, frozen)


@pytest.mark.parametrize("method", ("run", "longrun", "longrun_checkpointed"))
def test_canonical_tapes_must_match_the_explicit_accepted_step_budget(method):
    model, _ = _model()
    packet = _tapes(model, STEPS - 1)
    saved = _capture(model)
    with pytest.raises(ValueError):
        _call(model, method, packet, steps=STEPS)
    _unchanged(model, saved)


@pytest.mark.parametrize("invalid", ("dtype", "device", "shape", "nontensor", "packet"))
def test_invalid_inputs_reject_without_advancing_state_or_rng(invalid):
    model, _ = _model()
    ve = _tapes(model, 2).ve
    if invalid == "dtype":
        packet = dn.func.Inputs(ve.to(torch.float32))
    elif invalid == "device":
        packet = dn.func.Inputs(torch.empty(ve.shape, dtype=ve.dtype, device="meta"))
    elif invalid == "shape":
        packet = dn.func.Inputs(torch.zeros(2, 3, 5, 7, dtype=model.dtype()))
    elif invalid == "nontensor":
        packet = dn.func.Inputs([0.0, 1.0])
    else:
        packet = dn.func.RolloutInput(ve=ve)
    saved = _capture(model)
    with pytest.raises((TypeError, ValueError)):
        model.run(tstop=2 * DT, dt=DT, inputs=packet)
    _unchanged(model, saved)


@pytest.mark.parametrize("family", ("ve", "extra"))
def test_canonical_extracellular_inputs_reject_competing_legacy_sources(
    family, monkeypatch
):
    model, _ = _model()
    packet = dn.func.Inputs(ve=_tapes(model, 2).ve)
    waveform = dn.constant(value=0.2)

    def forbidden(*args, **kwargs):
        raise AssertionError("conflicting ve must reject before waveform evaluation")

    monkeypatch.setattr(waveform, "forward", forbidden)
    arguments = (
        {"ve": packet.ve}
        if family == "ve"
        else {"extra": (torch.ones_like(model.v), waveform)}
    )
    saved = _capture(model)
    with pytest.raises(ValueError, match="ve|extra|extracellular"):
        model.run(tstop=2 * DT, dt=DT, inputs=packet, **arguments)
    _unchanged(model, saved)


def test_slice_injection_remains_active_when_inputs_only_supplies_ve():
    model, waveform = _model(injection="solver")
    inputs = dn.func.Inputs(ve=_tapes(model, STEPS).ve)
    _, expected = _reference(model, inputs, STEPS)
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().sum(), (inputs.ve, waveform.amp)
    )
    model.run(inputs=inputs, dt=DT)
    torch.testing.assert_close(model.v, expected["integrator"]["v"])
    _close(
        torch.autograd.grad(model.v.square().sum(), (inputs.ve, waveform.amp)), wanted
    )
    assert wanted[1].abs().sum() > 0


def test_mechanism_owned_injection_coexists_with_explicit_intracellular_inputs():
    model, waveform = _model(injection="mechanism")
    packet = _tapes(model, 1)
    functional, expected = _reference(model, packet, 1)
    assert not functional._intra.enabled and model.mechanism_injections
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].sum(), (packet.intra, waveform.amp)
    )
    model.step(DT, inputs=dn.func.Inputs(packet.ve[0], packet.intra[0]))
    torch.testing.assert_close(model.v, expected["integrator"]["v"])
    _close(torch.autograd.grad(model.v.sum(), (packet.intra, waveform.amp)), wanted)


@pytest.mark.parametrize("method", _METHODS)
def test_registered_solver_intra_conflict_precedes_randomized_waveform_evaluation(
    method, monkeypatch
):
    model, extra, pulse = _stochastic_population(role="intra")
    assert extra is None
    steps = 1 if method == "step" else STEPS
    current = torch.zeros((steps, *model.shape), dtype=model.dtype())
    packet = dn.func.Inputs(intra=current[0] if method == "step" else current)
    saved = _capture(model)
    generator_state = pulse.generator.get_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "input rejection must precede sampling or waveform evaluation"
        )

    monkeypatch.setattr(pulse, "forward", forbidden)
    for name in ("rand", "randn", "rand_like", "randn_like"):
        monkeypatch.setattr(torch, name, forbidden)
    with pytest.raises(ValueError, match="intra|intracellular"):
        _call(model, method, packet, steps=steps)
    _unchanged(model, saved)
    torch.testing.assert_close(
        pulse.generator.get_state(), generator_state, rtol=0, atol=0
    )


def test_extra_and_explicit_intra_preserve_checkpointed_field_waveform_and_tape_vjps():
    model, _ = _model()
    current = _tapes(model, STEPS).intra
    field = (
        torch.linspace(-0.2, 0.3, model.v.numel()).reshape(model.shape).requires_grad_()
    )
    waveform = dn.sin(amp=0.2, freq=0.8, phase=0.3)
    voltage = torch.stack(
        [field * waveform(model.t.new_tensor(index * DT)) for index in range(STEPS)]
    )
    _, expected = _reference(model, dn.func.Inputs(ve=voltage, intra=current), STEPS)
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(), (field, waveform.amp, current)
    )
    _call(
        model,
        "longrun_checkpointed",
        dn.func.Inputs(intra=current),
        extra=[(field, waveform)],
    )
    torch.testing.assert_close(model.v, expected["integrator"]["v"])
    actual = torch.autograd.grad(
        model.v.square().mean(), (field, waveform.amp, current)
    )
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)


def test_zero_step_and_split_duration_continuation_preserve_all_tape_gradients():
    model, _ = _model()
    packet = _tapes(model, STEPS)
    functional, expected = _reference(model, packet, STEPS)
    wanted = torch.autograd.grad(expected["integrator"]["v"].square().mean(), packet)
    empty = dn.func.Inputs(packet.ve[:0], packet.intra[:0])
    frozen = _snapshot(model.state_dict_for_checkpoint())
    model.run(inputs=empty, dt=DT)
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    model.longrun(
        2.4 * DT,
        chunklength=3,
        dt=DT,
        inputs=dn.func.Inputs(packet.ve[:2], packet.intra[:2]),
    )
    model.run(tstop=0.3 * DT, dt=DT, inputs=empty)
    model.longrun_checkpointed(
        2.3 * DT,
        chunklength=2,
        dt=DT,
        inputs=dn.func.Inputs(packet.ve[2:], packet.intra[2:]),
        safe_checkpoint=True,
    )
    _close(functional.extract(model).state, expected)
    _close(torch.autograd.grad(model.v.square().mean(), packet), wanted)


class _HookLog(Callback):
    def __init__(self):
        super().__init__()
        self.events = []

    def pre_loop_hook(self, model):
        self.events.append(("pre_loop", model.t.item()))

    def pre_chunk_hook(self, model, timepoints=None):
        self.events.append(("pre_chunk", model.t.item()))

    def pre_step_hook(self, model):
        self.events.append(("pre_step", model.t.item()))

    def post_step_hook(self, model):
        self.events.append(("post_step", model.t.item()))

    def post_chunk_hook(self, model, timepoints=None):
        self.events.append(("post_chunk", model.t.item()))

    def post_loop_hook(self, model):
        self.events.append(("post_loop", model.t.item()))


def test_canonical_inputs_preserve_native_hook_order_across_chunk_tails():
    model, _ = _model()
    reference, _ = _model()
    field = torch.linspace(-0.2, 0.3, model.v.numel()).reshape(model.shape)
    ve = (0.2 * field).expand(STEPS, *model.shape)
    actual, expected = _HookLog(), _HookLog()
    model.longrun(
        STEPS * DT,
        chunklength=3,
        dt=DT,
        inputs=dn.func.Inputs(ve=ve),
        callbacks=[actual],
    )
    reference.longrun(
        STEPS * DT,
        chunklength=3,
        dt=DT,
        extra=(field, dn.constant(value=0.2)),
        callbacks=[expected],
    )
    assert actual.events == expected.events
    torch.testing.assert_close(model.v, reference.v)


def test_zero_explicit_current_preserves_shared_native_noise_and_waveform_rng_order():
    model, _, pulse = _stochastic_population(role="mechanism")
    reference, _, reference_pulse = _stochastic_population(role="mechanism")
    reference.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    current = torch.zeros((STEPS, *model.shape), dtype=model.dtype())
    model.run(tstop=STEPS * DT, dt=DT, inputs=dn.func.Inputs(intra=current))
    reference.run(tstop=STEPS * DT, dt=DT)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    torch.testing.assert_close(
        pulse.generator.get_state(),
        reference_pulse.generator.get_state(),
        rtol=0,
        atol=0,
    )


def test_checkpoint_backward_detects_mutation_of_a_full_shaped_input_leaf():
    model, _ = _model()
    current = _tapes(model, STEPS).intra
    _call(model, "longrun_checkpointed", dn.func.Inputs(intra=current))
    loss = model.v.square().sum()
    with torch.no_grad():
        current.add_(1e-11)
    with pytest.raises(RuntimeError, match="modified|version|inplace|in-place"):
        loss.backward()


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_native_inputs_match_float32_functional_values_and_drive_gradients():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        model, _ = _model()
        packet = _tapes(model, STEPS)
        _, expected = _reference(model, packet, STEPS)
        wanted = torch.autograd.grad(
            expected["integrator"]["v"].square().mean(), packet
        )
        _call(model, "longrun_checkpointed", packet)
        torch.testing.assert_close(
            model.v, expected["integrator"]["v"], rtol=3e-5, atol=3e-5
        )
        observed = torch.autograd.grad(model.v.square().mean(), packet)
        for actual, reference in zip(observed, wanted, strict=True):
            torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-6)
            assert actual.device.type == "mps" and actual.isfinite().all()
            assert actual.abs().sum() > 0
