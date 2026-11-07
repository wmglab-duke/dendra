"""Native draw replay spans waveform banks and every mechanism invocation."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.stim.waveform.core import _randomized_poisson
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_mechanism_injection_execution import (
    _InjectedLeak,
    _InjectedVoltage,
)
from tests.func.test_mechanism_injection_execution import _population as _point
from tests.func.test_native_handoff import _capture, _unchanged
from tests.func.test_poisson_invocation_execution import _case as _invocation_case
from tests.func.test_poisson_invocation_execution import (
    _native_state,
    _network_case,
    _ordered_network_case,
    _TwiceInjectedLeak,
)
from tests.func.test_poisson_invocation_guards import (
    _AdvanceOnly,
)
from tests.func.test_poisson_invocation_guards import _case as _phase_case
from tests.func.test_poisson_invocation_guards import (
    _CustomOuter,
    _InitializationOnly,
    _UnusedPoissonChild,
)
from tests.func.test_poisson_stimulation import DT, _population, _waveform
from tests.test_native_poisson_replay import (
    _same_waveforms,
    _waveform_bindings,
)
from tests.test_native_runtime_draw_tapes import (
    METHODS,
    _at,
    _functional_sources,
    _loss,
    _run,
    _state_loss,
)
from tests.test_native_runtime_draws import (
    _base_draws,
    _changed_draws,
)
from tests.test_native_runtime_draws import _environment as _environment
from tests.test_native_runtime_draws import (
    _forbid_owned_sampling,
    _parameters,
    _same_parameters,
)

pytestmark = pytest.mark.cpu
STEPS = 5


def _case(kind="custom", *, extra=False):
    if kind in ("advance", "initialization"):
        model, pulse = _phase_case(
            _AdvanceOnly if kind == "advance" else _InitializationOnly
        )
        model.train()
        model.detach()
        return model, None, pulse.generator
    if kind in ("repeated", "nested"):
        model, _, pulse, generator = _invocation_case(
            kind, "mechanism", mechanism=_TwiceInjectedLeak
        )
    else:
        generator = torch.Generator().manual_seed(421)
        pulse = _waveform(generator=generator)
        if kind == "unused":
            expression = _UnusedPoissonChild(pulse)
        else:
            other = _waveform(generator=generator, amp=0.13)
            other.waveform.amp = pulse.waveform.amp
            expression = _CustomOuter(pulse, other) + pulse
        model, _, _ = _population(
            "mechanism", waveform=expression, mechanism=_TwiceInjectedLeak
        )
    field = (
        torch.linspace(-0.2, 0.3, model.v.numel(), dtype=model.dtype())
        .reshape(model.shape)
        .requires_grad_()
    )
    return model, (field, pulse + pulse) if extra else None, generator


def _amplitudes(model):
    return tuple(
        value for name, value in model.named_parameters() if name.endswith("amp")
    )


def _roots(model):
    return tuple(
        module for module in model.modules() if isinstance(module, _randomized_poisson)
    )


@pytest.mark.parametrize(
    "kind", ("repeated", "nested", "custom", "unused", "advance", "initialization")
)
@pytest.mark.parametrize("diagnostic", (False, True))
def test_step_replay_uses_full_mechanism_profile_without_sampling_or_mutating_source(
    kind, diagnostic, monkeypatch
):
    source, _, source_generator = _case(kind)
    model, _, generator = _case(kind)
    source_snapshot = _snapshot(source.state_dict_for_checkpoint())
    before_source_rng = source_generator.get_state().clone()
    ambient = torch.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=diagnostic):
        functional, tensors = dn.func.make_functional(source, dt=DT)
        packet = _changed_draws(functional.sample_runtime_draws(tensors.state))
        saved_packet = _snapshot(packet)
        bindings, parameters = _waveform_bindings(model), _parameters(model)
        expected, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=packet
        )
        with monkeypatch.context() as guard:
            _forbid_owned_sampling(guard, model, generator)
            assert dn.step_population(model, DT, draws=packet) is model
        _close(_native_state(model), expected)
    _same_waveforms(model, bindings)
    _same_parameters(model, parameters)
    for pulse in _roots(model):
        assert not pulse._spike_times.requires_grad
        assert pulse._spike_times.grad_fn is None
    _assert_packet_equal(packet, saved_packet)
    _assert_packet_equal(source.state_dict_for_checkpoint(), source_snapshot)
    _close(source_generator.get_state(), before_source_rng)
    _close(torch.get_rng_state(), ambient)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", METHODS)
def test_mechanism_draw_tapes_keep_shared_bank_aliases_gradients_and_checkpoint_suffix(
    network, method, monkeypatch
):
    if network:
        source, source_extra, _ = _ordered_network_case()
        model, extra, generator = _ordered_network_case()
    else:
        source, source_extra, _ = _case(extra=True)
        model, extra, generator = _case(extra=True)
    functional, tensors = dn.func.make_functional(source, dt=DT, extra=source_extra)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=STEPS))
    references, targets = _amplitudes(source), _amplitudes(model)
    if not network:
        references, targets = (*references, source_extra[0]), (*targets, extra[0])
    saved_packet = _snapshot(packet)
    saved_source = _snapshot(source.state_dict_for_checkpoint())
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=STEPS,
        draws=packet,
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model, generator)
        _run(model, method, DT, draws=packet, extra=extra)
    _close(_native_state(model, extra), expected)
    final = _snapshot(model.state_dict_for_checkpoint())
    schedules = tuple((node, node._spike_times) for node in _roots(model))
    extra_roots = extra.values() if network else (extra,)
    schedules += tuple(
        (node, node._spike_times)
        for _, waveform in extra_roots
        for node in waveform.modules()
        if isinstance(node, _randomized_poisson)
    )
    before_rng = generator.get_state().clone()
    wanted = torch.autograd.grad(
        _state_loss(expected, network),
        _functional_sources(functional, tensors, references),
    )
    actual = torch.autograd.grad(
        _loss(model), (*targets, *_base_draws(packet)), allow_unused=True
    )
    _close(actual[: len(wanted)], wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in wanted)
    assert all(
        value is None or value.count_nonzero() == 0 for value in actual[len(wanted) :]
    )
    _assert_packet_equal(model.state_dict_for_checkpoint(), final)
    assert all(node._spike_times is saved for node, saved in schedules)
    _close(generator.get_state(), before_rng)
    _assert_packet_equal(packet, saved_packet)
    _assert_packet_equal(source.state_dict_for_checkpoint(), saved_source)

    if network:
        receiver, receiving_extra, _ = _ordered_network_case()
    else:
        receiver, receiving_extra, _ = _case(extra=True)
    receiver.restore_dict_from_checkpoint(final)
    following, _ = dn.func.run(
        functional,
        dn.func.Continuation(expected, tensors.parameters, tensors.constants),
        steps=1,
    )
    receiver.step(**({} if network else {"dt": DT}), extra=receiving_extra)
    _close(_native_state(receiver, receiving_extra), following)


@pytest.mark.parametrize("method", ("step", *METHODS))
def test_network_shared_physical_mechanism_waveform_keeps_population_order(
    method, monkeypatch
):
    source, _ = _network_case("waveform")
    model, generator = _network_case("waveform")
    functional, tensors = dn.func.make_functional_network(source)
    steps = 1 if method == "step" else STEPS
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=steps))
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=steps,
        draws=packet,
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model, generator)
        if method == "step":
            assert dn.step_network(model, draws=_at(packet, 0)) is model
        else:
            _run(model, method, DT, steps=steps, draws=packet)
    _close(_native_state(model), expected)
    assert len(_roots(model)) == 1
    _close(generator.get_state(), next(iter(expected["stimulation_rng"].values())))


@pytest.mark.parametrize("method", METHODS)
def test_late_invalid_mechanism_draw_rejects_before_any_live_invocation(method):
    model, extra, generator = _case(extra=True)
    functional, tensors = dn.func.make_functional(model, dt=DT, extra=extra)
    packet = _snapshot(functional.sample_runtime_draws(tensors.state, steps=STEPS))
    key = next(
        key for key in packet.draws["stimulation"] if "injected_waveforms" in key
    )
    packet.draws["stimulation"][key][-1].fill_(torch.nan)
    saved, rng = _capture(model), generator.get_state().clone()
    with pytest.raises((ValueError, RuntimeError, dn.func.FunctionalizationError)):
        _run(model, method, DT, draws=packet, extra=extra)
    _unchanged(model, saved)
    _close(generator.get_state(), rng)


@pytest.mark.parametrize("kind", ("unused", "initialization", "advance"))
def test_checkpointed_zero_step_mechanism_tape_keeps_schedules_and_stream(kind):
    model, extra, generator = _case(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT, extra=extra)
    packet = functional.sample_runtime_draws(tensors.state, steps=0)
    saved, rng = _capture(model), generator.get_state().clone()
    _run(model, "longrun_checkpointed", DT, steps=0, draws=packet, extra=extra)
    _unchanged(model, saved)
    _close(generator.get_state(), rng)


@pytest.mark.parametrize("method", ("rk2", "rk4", "scnv"))
def test_stage_and_voltage_process_profiles_match_native_checkpointed_execution(method):
    def make():
        waveform = _waveform(generator=torch.Generator().manual_seed(467))
        if method == "scnv":
            return _point(_InjectedVoltage, waveform=waveform), waveform
        model = dn.SingleCompartment(
            N=1, C=2, v_init=-47.0, integrator=getattr(dn, method)(imem=False)
        )
        model.insert(_InjectedLeak.rename("owner"))
        model[:, 0].inject(waveform)
        model.train()
        model.initialize()
        model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
        model.detach()
        return model, waveform

    source, _ = make()
    model, pulse = make()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=3))
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=3,
        draws=packet,
    )
    _run(model, "longrun_checkpointed", DT, steps=3, chunklength=2, draws=packet)
    _close(functional.extract(model).state, expected)
    final = _snapshot(model.state_dict_for_checkpoint())
    expected_amp = _functional_sources(functional, tensors, _amplitudes(source))
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(), expected_amp
    )
    actual = torch.autograd.grad(model.v.square().mean(), pulse.waveform.amp)
    _close(actual, wanted)
    assert wanted[0].abs().sum() > 0
    _assert_packet_equal(model.state_dict_for_checkpoint(), final)


def test_compiled_native_mechanism_replay_reuses_kernels_and_retains_gradients():
    source, _, _ = _case("repeated")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    tape = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=2))
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=2,
        draws=tape,
    )
    with dn.ctx(JIT=1, BACKEND="aot_eager", FULLGRAPH=True):
        model, _, _ = _case("repeated")
        model.step(DT, draws=_at(tape, 0))
        kernels = dict(model.integrator._compiled_kernels)
        assert kernels
        model.step(DT, draws=_at(tape, 1))
        assert model.integrator._compiled_kernels.keys() == kernels.keys()
        assert all(
            model.integrator._compiled_kernels[name] is kernel
            for name, kernel in kernels.items()
        )
        assert all("_native_poisson_replay" not in vars(node) for node in _roots(model))
    _close(functional.extract(model).state, expected)
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(),
        _functional_sources(functional, tensors, _amplitudes(source)),
    )
    actual = torch.autograd.grad(model.v.square().mean(), _amplitudes(model))
    _close(actual, wanted)
    assert wanted[0].abs().sum() > 0


@pytest.mark.parametrize("layout", ("batch", "packed"))
def test_mechanism_replay_keeps_batched_and_concatenated_ownership_and_gradients(
    layout,
):
    def make():
        generator = torch.Generator().manual_seed(617)

        def component():
            pulse = _waveform(generator=generator)
            return _point(
                _TwiceInjectedLeak,
                layout="batch" if layout == "batch" else "dense",
                waveform=pulse + pulse,
            )

        model = (
            component()
            if layout == "batch"
            else dn.concat_models(
                {"left": component(), "right": component()}, threads=1
            )
        )
        model.train()
        model.initialize()
        model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
        model.detach()
        return model, generator

    def amplitudes(model):
        # Concatenated children are not the runtime owners; select the parent
        # handler's authoritative waveform registrations.
        return tuple(
            {
                id(value): value
                for owner in model.mech.mechanisms.values()
                for name, value in owner.named_parameters()
                if name.endswith("amp")
            }.values()
        )

    source, _ = make()
    model, generator = make()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    tape = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=3))
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=3,
        draws=tape,
    )
    _run(model, "longrun_checkpointed", DT, steps=3, chunklength=2, draws=tape)
    _close(functional.extract(model).state, expected)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    rng = generator.get_state().clone()
    wanted = torch.autograd.grad(
        expected["integrator"]["v"].square().mean(),
        _functional_sources(functional, tensors, amplitudes(source)),
    )
    actual = torch.autograd.grad(model.v.square().mean(), amplitudes(model))
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in wanted)
    _close(generator.get_state(), rng)
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
