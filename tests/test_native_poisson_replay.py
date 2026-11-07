"""Native Population banks consume one supplied draw per Poisson invocation."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas
from dendra.models.stim.waveform._poisson import (
    native_poisson_schedule_from_draws,
    poisson_generator_state,
)
from dendra.units import nA
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_poisson_invocation_execution import _case as _invocation_case
from tests.func.test_poisson_invocation_execution import _native_state
from tests.func.test_poisson_invocation_guards import _CustomOuter
from tests.func.test_poisson_stimulation import DT, _population, _waveform
from tests.func.test_stimulation_rng_sharing import _one_stream
from tests.func.test_stimulation_rng_sharing import _population as _borrowed_population
from tests.func.test_stimulation_rng_sharing import _RandOnlyLeak
from tests.test_native_runtime_draws import (
    _base_draws,
)
from tests.test_native_runtime_draws import _environment as _environment
from tests.test_native_runtime_draws import (
    _forbid_owned_sampling,
    _parameters,
    _same_parameters,
)

pytestmark = pytest.mark.cpu


def _case(kind="repeated", role="intra"):
    if kind != "custom":
        return _invocation_case(kind, role)
    generator = torch.Generator().manual_seed(73)
    scale = nA if role == "intra" else 1.0
    first = _waveform(generator=generator, amp=0.2 * scale)
    second = _waveform(generator=generator, amp=0.3 * scale)
    expression = _CustomOuter(first, second)
    model, _, extra = _population(role, waveform=expression)
    return model, extra, first, generator


def _root(model, extra):
    return extra[1] if extra is not None else model.injections[0][0]


def _changed_uniforms(packet):
    draws = {
        name: torch.linspace(
            0.1 + 0.05 * index,
            0.75 + 0.05 * index,
            value.numel(),
            dtype=value.dtype,
            device=value.device,
        )
        .reshape(value.shape)
        .requires_grad_()
        for index, (name, value) in enumerate(packet.draws["stimulation"].items())
    }
    return packet._replace(draws={**packet.draws, "stimulation": draws})


def _waveform_bindings(root):
    return tuple(
        (name, module, type(module))
        for name, module in root.named_modules(remove_duplicate=False)
    )


def _same_waveforms(root, saved):
    current = dict(root.named_modules(remove_duplicate=False))
    assert tuple(current) == tuple(name for name, _, _ in saved)
    assert all(
        current[name] is module and type(module) is kind for name, module, kind in saved
    )


def _amplitudes(root):
    return tuple(
        value for name, value in root.named_parameters() if name.endswith("amp")
    )


@pytest.mark.parametrize("role", ("intra", "extra"))
@pytest.mark.parametrize("kind", ("repeated", "custom"))
def test_bank_replay_preserves_invocation_order_aliases_schedules_and_source_gradients(
    kind, role, monkeypatch
):
    source, extra, _, _ = _case(kind, role)
    native, native_extra, pulse, generator = _case(kind, role)
    automatic, automatic_extra, automatic_pulse, _ = _case(kind, role)
    if extra is not None:
        extra[0].requires_grad_()
        native_extra[0].requires_grad_()
    functional, tensors = dn.func.make_functional(source, dt=DT, extra=extra)
    packet = _changed_uniforms(functional.sample_runtime_draws(tensors.state))
    saved_packet, source_state = (
        _snapshot(packet),
        _snapshot(source.state_dict_for_checkpoint()),
    )
    root, source_root = _root(native, native_extra), _root(source, extra)
    bindings, parameters = _waveform_bindings(root), _parameters(native)
    global_rng = torch.get_rng_state().clone()
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, native, generator)
        assert (
            dn.step_population(native, DT, extra=native_extra, draws=packet) is native
        )
    automatic.step(DT, extra=automatic_extra)
    _close(_native_state(native, native_extra), expected)
    assert not torch.equal(pulse._spike_times, automatic_pulse._spike_times)
    assert not pulse._spike_times.requires_grad and pulse._spike_times.grad_fn is None
    _close(poisson_generator_state(pulse), packet.rng["stimulation"]["stream_0"])
    targets, references = _amplitudes(root), _amplitudes(source_root)
    if extra is not None:
        targets += (native_extra[0],)
        references += (extra[0],)
    actual = torch.autograd.grad(
        native.v.square().mean(), (*targets, *_base_draws(packet)), allow_unused=True
    )
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(), references
    )
    _close(actual[: len(targets)], wanted)
    assert all(
        value.isfinite().all() and value.abs().sum() > 0
        for value in actual[: len(targets)]
    )
    assert all(
        value is None or value.count_nonzero() == 0 for value in actual[len(targets) :]
    )
    _same_waveforms(root, bindings)
    _same_parameters(native, parameters)
    _assert_packet_equal(packet, saved_packet)
    _assert_packet_equal(source.state_dict_for_checkpoint(), source_state)
    _close(torch.get_rng_state(), global_rng)


def test_fixed_nested_poisson_replay_publishes_the_last_physical_child_schedule():
    source, extra, source_pulse, _ = _case("nested")
    native, native_extra, pulse, _ = _case("nested")
    functional, tensors = dn.func.make_functional(source, dt=DT, extra=extra)
    packet = _changed_uniforms(functional.sample_runtime_draws(tensors.state))
    bindings = _waveform_bindings(_root(native, native_extra))
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    entry = next(
        entry
        for entry in functional._poisson_plan.entries
        if entry.source is source_pulse
    )
    rows = packet.draws["stimulation"][entry.key]
    assert rows.shape[0] == _root(native, native_extra).n == 2
    first_schedule = native_poisson_schedule_from_draws(pulse, rows[0])
    last_schedule = native_poisson_schedule_from_draws(pulse, rows[1])
    assert not torch.equal(first_schedule, last_schedule)
    native.step(DT, draws=packet)
    _close(pulse._spike_times, last_schedule)
    _close(_native_state(native), expected)
    _same_waveforms(_root(native, native_extra), bindings)
    _close(poisson_generator_state(pulse), packet.rng["stimulation"]["stream_0"])


@pytest.mark.parametrize("shape", ("trim", "zero_count", "start_at_off"))
def test_nonnested_finite_off_and_empty_schedules_preserve_native_shape_and_sentinel(
    shape,
):
    def build():
        pulse = dn.mono_rect(amp=0.2 * nA, pw=0.045).poisson(
            interval=0.055,
            n=0 if shape == "zero_count" else 5,
            start=0.1 if shape == "start_at_off" else -0.035,
            off=0.1
            if shape == "start_at_off"
            else 0.04
            if shape == "trim"
            else torch.inf,
            noise=0.7,
            randomize_every_call=True,
            generator=torch.Generator().manual_seed(73),
        )
        return _population("intra", waveform=pulse)[0], pulse

    source, _ = build()
    native, pulse = build()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = _changed_uniforms(functional.sample_runtime_draws(tensors.state))
    draw = next(iter(packet.draws["stimulation"].values()))
    trailing_dims = pulse._spike_time_trailing_dims
    schedule = native_poisson_schedule_from_draws(pulse, draw)
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    native.step(DT, draws=packet)
    _close(_native_state(native), expected)
    _close(pulse._spike_times, schedule)
    assert pulse._spike_times.ndim == 1 + trailing_dims
    assert pulse._spike_times.shape[1:] == (1,) * trailing_dims
    if shape == "trim":
        assert 0 < torch.isfinite(pulse._spike_times).sum().item() < pulse.n
    else:
        assert pulse._spike_times.numel() == 1 and torch.isinf(pulse._spike_times).all()


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "mps",
            marks=(
                pytest.mark.mps,
                pytest.mark.skipif(
                    not torch.backends.mps.is_available(), reason="MPS is unavailable"
                ),
            ),
        ),
    ),
)
def test_cold_cpu_float32_extra_waveform_replays_at_the_receiving_model_placement(
    device,
):
    if device == "mps":
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"

    def build():
        with dn.ctx(DEVICE="cpu", DTYPE=torch.float32):
            pulse = _waveform(amp=0.2)
            expression = pulse + pulse
        model, _, extra = _population("extra", waveform=expression)
        dtype = torch.float64 if device == "cpu" else torch.float32
        if device == "mps":
            model.to(device=device, dtype=dtype)
            model.initialize()
        field = (
            extra[0].to(device=device, dtype=dtype).detach().clone().requires_grad_()
        )
        assert pulse._spike_times.device.type == "cpu"
        assert pulse._spike_times.dtype == torch.float32
        return model, (field, expression), pulse

    source, extra, source_pulse = build()
    native, native_extra, pulse = build()
    functional, tensors = dn.func.make_functional(source, dt=DT, extra=extra)
    packet = _changed_uniforms(functional.sample_runtime_draws(tensors.state))
    saved = _snapshot(packet)
    generator = pulse.generator
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    native.step(DT, extra=native_extra, draws=packet)
    torch.testing.assert_close(
        native.v, expected["integrator"]["v"], rtol=2e-5, atol=2e-6
    )
    assert pulse._spike_times.device == native.v.device
    assert pulse._spike_times.dtype == native.v.dtype
    actual = torch.autograd.grad(
        native.v.square().mean(), (pulse.waveform.amp, native_extra[0])
    )
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(),
        (source_pulse.waveform.amp, extra[0]),
    )
    for observed, reference in zip(actual, wanted, strict=True):
        torch.testing.assert_close(
            observed, reference.to(observed), rtol=2e-5, atol=2e-6
        )
        assert observed.isfinite().all() and observed.abs().sum() > 0
    assert pulse.generator is generator
    _close(poisson_generator_state(pulse), packet.rng["stimulation"]["stream_0"])
    _assert_packet_equal(packet, saved)


def test_poisson_replay_coexists_with_explicit_voltage_and_callback_sampling_phase():
    source, _, _, _ = _case()
    native, _, _, _ = _case()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    callbacks = dn.func.make_callbacks(
        native, {"trace": dn.Recorder(("v", "t"), dt=2 * DT)}, dt=DT
    )
    voltage = torch.zeros_like(native.v, requires_grad=True)
    inputs = dn.func.Inputs(ve=voltage)
    state, carry = tensors.state, None
    results = []
    for _ in range(2):
        packet = _changed_uniforms(functional.sample_runtime_draws(state))
        state, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state, inputs=inputs, draws=packet
        )
        result = dn.step(
            native,
            DT,
            inputs=inputs,
            draws=packet,
            callbacks=callbacks,
            callback_state=carry,
        )
        results.append(result)
        carry = result.state
    _close(native.v, state["integrator"]["v"])
    _close(
        torch.cat([part["trace"]["t"] for part in results]),
        torch.tensor([0.0, 2 * DT], dtype=native.t.dtype),
    )
    assert carry.plan_token is results[0].state.plan_token
    _close(results[-1]["trace"]["v"][-1], native.v)
    assert inputs.ve is voltage
    actual = torch.autograd.grad(native.v.square().mean(), voltage)[0]
    wanted = torch.autograd.grad(state["integrator"]["v"].square().mean(), voltage)[0]
    _close(actual, wanted)
    assert actual.isfinite().all() and actual.abs().sum() > 0


def test_explicit_global_poisson_draws_supersede_incoming_deferred_frame_before_continuation():
    source, _, _ = _population(
        "intra", waveform=_waveform(generator=torch.default_generator, amp=0.2 * nA)
    )
    native, pulse, _ = _population(
        "intra", waveform=_waveform(generator=torch.default_generator, amp=0.2 * nA)
    )
    receiver, receiving_pulse, _ = _population(
        "intra", waveform=_waveform(generator=torch.default_generator, amp=0.2 * nA)
    )
    functional, tensors = dn.func.make_functional(source, dt=DT)
    boundary = dn.func.capture_continuation(functional)
    dn.func.commit_continuation_(functional, native, boundary)
    pending = vars(pulse).get("_functional_poisson_rng")
    assert pending is not None and pending.state is not None
    packet = _changed_uniforms(functional.sample_runtime_draws(boundary.state))
    expected, _ = functional.prepare_and_step(
        boundary.parameters, boundary.constants, boundary.state, draws=packet
    )
    torch.manual_seed(9931)
    ambient = torch.get_rng_state().clone()
    native.step(DT, draws=packet)
    _close(torch.get_rng_state(), packet.rng["stimulation"]["stream_0"])
    assert not torch.equal(torch.get_rng_state(), ambient)
    pending = vars(pulse).get("_functional_poisson_rng")
    assert pending is None or pending.state is None
    _close(poisson_generator_state(pulse), packet.rng["stimulation"]["stream_0"])
    checkpoint = _snapshot(native.state_dict_for_checkpoint())
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(expected, boundary.parameters, boundary.constants),
        steps=1,
    )
    native.step(DT)
    _close(native.v, expected["integrator"]["v"])
    receiver.restore_dict_from_checkpoint(checkpoint)
    _close(
        poisson_generator_state(receiving_pulse), packet.rng["stimulation"]["stream_0"]
    )
    receiver.step(DT)
    _close(receiver.v, native.v)
    _close(poisson_generator_state(receiving_pulse), poisson_generator_state(pulse))


def test_bank_poisson_and_model_noise_sde_keep_one_borrowed_rng_authority():
    source, _, _ = _borrowed_population("intra")
    native, _, pulse = _borrowed_population("intra")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = _changed_uniforms(functional.sample_runtime_draws(tensors.state))
    assert packet.rng["model"] and not packet.rng["stimulation"]
    stream = native.mech.probe.stream
    generator = pulse.generator
    gate = native.mech.probe.DE._SharedStochasticGate
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    native.step(DT, draws=packet)
    _close(functional.extract(native).state, expected)
    assert stream is gate.stream is gate.x_dW_rng
    assert stream.rng is pulse.generator is generator
    _close(generator.get_state(), _one_stream(expected))
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(expected, tensors.parameters, tensors.constants),
        steps=1,
    )
    native.step(DT)
    _close(functional.extract(native).state, expected)
    _close(generator.get_state(), _one_stream(expected))


def test_rand_only_borrowed_poisson_stream_replays_without_a_runtime_noise_family(
    monkeypatch,
):
    def build():
        model = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-45.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(_RandOnlyLeak.rename("probe"))
        model.train()
        model.initialize()
        pulse = _waveform(generator=model.mech.probe.offset_rng.rng, amp=0.2 * nA)
        model[:, 1].inject(pulse)
        model.initialize()
        model.detach()
        return model, pulse

    source, _ = build()
    native, pulse = build()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = _changed_uniforms(functional.sample_runtime_draws(tensors.state))
    assert packet.draws["noise"] is None
    assert packet.rng["model"] and not packet.rng["stimulation"]
    saved_packet, parameters = _snapshot(packet), _parameters(native)
    stream, generator = native.mech.probe.offset_rng, pulse.generator
    offset = native.mech.probe.offset.detach().clone()
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, native, generator)
        assert dn.step_population(native, DT, draws=packet) is native
    _close(functional.extract(native).state, expected)
    _close(native.mech.probe.offset, offset)
    _close(generator.get_state(), _one_stream(expected))
    assert native.mech.probe.offset_rng is stream
    assert stream.rng is pulse.generator is generator
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(expected, tensors.parameters, tensors.constants),
        steps=1,
    )
    native.step(DT)
    _close(functional.extract(native).state, expected)
    _close(native.mech.probe.offset, offset)
    _close(generator.get_state(), _one_stream(expected))
    _same_parameters(native, parameters)
    _assert_packet_equal(packet, saved_packet)


def test_live_bank_exception_cleans_replay_slots_and_allows_retry_without_rng_rewind(
    monkeypatch,
):
    source, _, _, _ = _case("custom")
    native, _, _, generator = _case("custom")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = _changed_uniforms(functional.sample_runtime_draws(tensors.state))
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    root = _root(native, None)
    original = type(root).fn
    bindings, parameters = _waveform_bindings(root), _parameters(native)
    saved_packet = _snapshot(packet)
    voltage, time, rng = (
        native.v.detach().clone(),
        native.t.clone(),
        generator.get_state().clone(),
    )

    def fail_live(self, times):
        if self is root:
            self.second.fn(times)
            raise RuntimeError("native bank evaluation failed after one invocation")
        return original(self, times)

    with monkeypatch.context() as guard:
        guard.setattr(type(root), "fn", fail_live)
        _forbid_owned_sampling(guard, native, generator)
        with pytest.raises(RuntimeError, match="native bank evaluation failed"):
            native.step(DT, draws=packet)
    assert all(
        "_native_poisson_replay" not in vars(module) for module in root.modules()
    )
    _close(native.v, voltage)
    _close(native.t, time)
    _close(generator.get_state(), rng)
    _same_waveforms(root, bindings)
    _same_parameters(native, parameters)
    _assert_packet_equal(packet, saved_packet)
    # A failed native call may retain a schedule consumed before the error.
    # Retrying installs fresh cursors and must consume the complete packet.
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, native, generator)
        native.step(DT, draws=packet)
    _close(_native_state(native), expected)
    _close(generator.get_state(), packet.rng["stimulation"]["stream_0"])
    assert all(
        "_native_poisson_replay" not in vars(module) for module in root.modules()
    )
    _same_waveforms(root, bindings)
    _assert_packet_equal(packet, saved_packet)


@pytest.mark.parametrize(
    "location", ("mechanism_unique", "mechanism_alias", "population_orphan")
)
def test_unrepresented_registered_poisson_nodes_reject_before_any_native_mutation(
    location, monkeypatch
):
    if location == "population_orphan":
        # This model has no registered bank: its returned extra recipe is
        # deliberately omitted when preparing the valid empty draw packet.
        model, pulse, _ = _population("extra")
    else:
        model, _, pulse, _ = _case()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    packet = functional.sample_runtime_draws(tensors.state)
    node = _waveform(seed=907) if location == "mechanism_unique" else pulse
    owner = model if location == "population_orphan" else model.mech.pas
    owner.unrepresented_poisson = node
    checkpoint = _snapshot(model.state_dict_for_checkpoint())
    saved_packet, time = _snapshot(packet), model.t.clone()
    global_rng = torch.get_rng_state().clone()
    streams = {
        id(waveform.generator): (
            waveform.generator,
            waveform.generator.get_state().clone(),
        )
        for waveform in (pulse, node)
    }
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(
            guard, model, *(generator for generator, _state in streams.values())
        )
        with pytest.raises(
            (ValueError, dn.func.FunctionalizationError), match="Poisson|poisson"
        ):
            model.step(DT, draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), checkpoint)
    _close(model.t, time)
    _close(torch.get_rng_state(), global_rng)
    for generator, state in streams.values():
        _close(generator.get_state(), state)
    _assert_packet_equal(packet, saved_packet)


@pytest.mark.parametrize(
    "invalid", ("names", "shape", "dtype", "uniform", "rng", "tape")
)
def test_bad_bank_draw_packets_reject_before_clock_schedule_or_rng_mutation(
    invalid, monkeypatch
):
    model, _, pulse, generator = _case()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    packet = functional.sample_runtime_draws(tensors.state)
    draws = dict(packet.draws["stimulation"])
    key = next(iter(draws))
    if invalid == "names":
        draws.pop(key)
    elif invalid == "shape":
        draws[key] = draws[key][..., :-1]
    elif invalid == "dtype":
        draws[key] = draws[key].to(
            torch.float64 if draws[key].dtype == torch.float32 else torch.float32
        )
    elif invalid == "uniform":
        draws[key] = torch.full_like(draws[key], 1.0)
    if invalid == "rng":
        packet = packet._replace(rng={**packet.rng, "stimulation": {}})
    elif invalid == "tape":
        packet = functional.sample_runtime_draws(tensors.state, steps=1)
    else:
        packet = packet._replace(draws={**packet.draws, "stimulation": draws})
    checkpoint, schedule, rng = (
        _snapshot(model.state_dict_for_checkpoint()),
        pulse._spike_times.clone(),
        generator.get_state().clone(),
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model, generator)
        with pytest.raises((ValueError, TypeError, dn.func.FunctionalizationError)):
            model.step(DT, draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), checkpoint)
    _close(pulse._spike_times, schedule)
    _close(generator.get_state(), rng)


def test_draw_dependent_nested_invocation_counts_reject_without_advancing_native_state():
    source, _, _, _ = _case("nested")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = functional.sample_runtime_draws(tensors.state)
    model, _, pulse, generator = _case("nested")
    _root(model, None).off = 0.01
    checkpoint, rng = (
        _snapshot(model.state_dict_for_checkpoint()),
        generator.get_state().clone(),
    )
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="invocation|count|nested"
    ):
        model.step(DT, draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), checkpoint)
    _close(generator.get_state(), rng)
    assert pulse.randomize_every_call


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_bank_replay_keeps_float32_waveforms_schedules_and_amplitude_gradients():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):

        def build():
            model = dn.Unmyelinated(
                [2.0],
                L=4.0,
                dx=1.0,
                dtype=torch.float32,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
            model.insert(pas)
            pulse = _waveform(amp=0.2 * nA)
            expression = pulse + pulse
            model[:, 1].inject(expression)
            model.train()
            model.initialize()
            return model, pulse

        source, source_pulse = build()
        native, pulse = build()
        functional, tensors = dn.func.make_functional(source, dt=DT)
        packet = _changed_uniforms(functional.sample_runtime_draws(tensors.state))
        saved = _snapshot(packet)
        expected, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=packet
        )
        dn.step_population(native, DT, draws=packet)
        torch.testing.assert_close(
            native.v, expected["integrator"]["v"], rtol=2e-5, atol=2e-6
        )
        actual = torch.autograd.grad(native.v.square().mean(), pulse.waveform.amp)[0]
        wanted = torch.autograd.grad(
            expected["integrator"]["v"].square().mean(), source_pulse.waveform.amp
        )[0]
        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-6)
        assert actual.isfinite().all() and actual.abs().sum() > 0
        assert (
            pulse._spike_times.device.type == "mps"
            and pulse._spike_times.dtype == torch.float32
        )
        assert not pulse._spike_times.requires_grad
        _close(poisson_generator_state(pulse), packet.rng["stimulation"]["stream_0"])
        _assert_packet_equal(packet, saved)
