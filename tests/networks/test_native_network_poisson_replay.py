"""Native Network banks consume one shared, invocation-ordered draw packet."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import pas
from dendra.models.rng import RNGModule
from dendra.models.stim.waveform.core import _randomized_poisson
from dendra.units import nA
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_network_rng_sharing import _network as _noise_network
from tests.func.test_poisson_invocation_execution import _native_state
from tests.func.test_poisson_invocation_guards import _CustomOuter
from tests.func.test_poisson_stimulation import DT, _population, _waveform
from tests.func.test_stimulation_rng_sharing import _one_stream, _RandOnlyLeak
from tests.test_native_poisson_replay import _same_waveforms, _waveform_bindings
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


class _NoneReturningNoiseLeak(Mechanism):
    Mechanism.RANGE(g=0.001)
    Mechanism.RANGENOISE("e", distribution="uniform", low=-68.0, high=-62.0, seed=719)
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00002, seed=727)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def sample_runtime_noises_(self, *args, **kwargs):
        Mechanism.sample_runtime_noises_(self, *args, **kwargs)
        return None

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e.clone() - self.eta.clone()


def _unique(values):
    return tuple({id(value): value for value in values}.values())


def _case(*, sharing="waveform", cross_banks=False, global_stream=False):
    generator = (
        torch.default_generator if global_stream else torch.Generator().manual_seed(197)
    )
    populations, intra, extra, intra_order, extra_order = {}, {}, {}, {}, {}
    shared = _waveform(generator=generator, amp=0.2 * nA)
    pulses = {}
    for name in ("left", "right"):
        pulse = (
            shared
            if sharing == "waveform" and not cross_banks
            else _waveform(generator=generator, amp=0.2 * nA)
        )
        if cross_banks:
            expression = pulse + pulse
            order = (pulse, pulse)
        else:
            second = _waveform(generator=generator, amp=0.13 * nA)
            second.waveform.amp = pulse.waveform.amp
            expression = _CustomOuter(pulse, second) + pulse
            order = (second, pulse, pulse)
        populations[name] = _population("intra", waveform=expression)[0]
        intra[name], intra_order[name], pulses[name] = expression, order, pulse
    for name in ("right", "left"):
        pulse = (
            pulses["left" if name == "right" else "right"]
            if cross_banks
            else _waveform(generator=generator)
        )
        field = (
            torch.linspace(
                -1.0e6 if cross_banks else -1.25,
                1.5e6 if cross_banks else 1.75,
                populations[name].v.numel(),
                dtype=populations[name].dtype(),
            )
            .reshape(populations[name].shape)
            .requires_grad_()
        )
        extra[name] = (field, pulse + pulse)
        extra_order[name] = (pulse, pulse)
    model = dn.Network(populations)
    model.train()
    model.initialize(DT)
    for population in model.populations.values():
        population.detach()
    order = tuple(
        node
        for groups in (extra_order, intra_order)
        for nodes in groups.values()
        for node in nodes
    )
    return SimpleNamespace(
        model=model,
        extra=extra,
        roots=(*intra.values(), *(root for _field, root in extra.values())),
        nodes=_unique(order),
        order=order,
        generator=generator,
    )


def _changed_packet(packet):
    packet = _changed_draws(packet)
    index, populations = 0, {}
    for name, local in packet.draws["populations"].items():
        draws = local["stimulation"]
        if draws is not None:
            changed = {}
            for key, value in draws.items():
                changed[key] = (
                    torch.linspace(
                        0.1 + 0.01 * index,
                        0.7 + 0.01 * index,
                        value.numel(),
                        device=value.device,
                        dtype=value.dtype,
                    )
                    .reshape(value.shape)
                    .requires_grad_()
                )
                index += 1
            draws = changed
        populations[name] = {**local, "stimulation": draws}
    return packet._replace(draws={**packet.draws, "populations": populations})


def _loss(model):
    return sum(
        population.v.square().mean() for population in model.populations.values()
    )


def _state_loss(state):
    return sum(
        local["integrator"]["v"].square().mean()
        for local in state["populations"].values()
    )


def _gradient_sources(case):
    amplitudes = _unique(
        parameter
        for root in case.roots
        for name, parameter in root.named_parameters()
        if name.endswith("amp")
    )
    return (*amplitudes, *(field for field, _root in case.extra.values()))


def _functional_sources(functional, tensors, originals):
    """Find each authored source's independently owned Network tensor leaf."""
    requested = {id(value) for value in originals}
    selected = {}
    for name, plan in functional.populations.items():
        explicit = tensors.parameters["populations"][name]
        authored = dict(plan._population.named_parameters(remove_duplicate=False))
        authored.update(plan._intra._tensor_sources())
        authored.update(plan._extra._tensor_sources())
        for key, source in authored.items():
            if id(source) in requested and key in explicit:
                previous = selected.setdefault(id(source), explicit[key])
                assert previous is explicit[key]
    return tuple(selected[id(value)] for value in originals)


def _noise_bindings(model):
    identities = {
        id(owner._buffers[name])
        for owner in model.modules()
        for name in getattr(owner, "runtime_noises", {})
    }
    bindings = {
        path: value
        for path, value in model.named_buffers(remove_duplicate=False)
        if id(value) in identities
    }
    assert bindings
    return bindings


def _assert_noise_bindings(model, saved):
    current = dict(model.named_buffers(remove_duplicate=False))
    assert all(current[path] is value for path, value in saved.items())


def test_float32_network_python_timestep_matches_standalone_tensor_timestep():
    with dn.ctx(DEVICE="cpu", DTYPE=torch.float32):

        def build():
            population = dn.Unmyelinated(
                [2.0],
                L=4.0,
                dx=1.0,
                dtype=torch.float32,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
            population.insert(pas)
            population[:, 1].inject(dn.mono_rect(amp=0.2 * nA, pw=0.2))
            population.train()
            population.initialize()
            return population

        standalone = build()
        network = dn.Network({"cell": build()})
        network.train()
        network.initialize(DT)
        standalone.step(DT)
        network.step()
        torch.testing.assert_close(network.cell.v, standalone.v, rtol=1e-6, atol=2e-6)
        assert network.cell.v.dtype == standalone.v.dtype == torch.float32


@pytest.mark.parametrize("sharing", ("generator", "waveform"))
def test_network_bank_replay_preserves_global_invocation_order_aliases_and_gradients(
    sharing, monkeypatch
):
    source, native = _case(sharing=sharing), _case(sharing=sharing)
    functional, tensors = dn.func.make_functional_network(
        source.model, extra=source.extra
    )
    packet = _changed_packet(functional.sample_runtime_draws(tensors.state))
    saved_packet = _snapshot(packet)
    saved_source = _snapshot(source.model.state_dict_for_checkpoint())
    parameters = _parameters(native.model)
    bindings = tuple(_waveform_bindings(root) for root in native.roots)
    global_rng = torch.get_rng_state().clone()
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    callbacks = dn.func.make_callbacks(
        native.model,
        {"trace": dn.Recorder(("left.v", "right.v", "t"), dt=2 * DT)},
        dt=DT,
    )
    original, observed = _randomized_poisson.fn, []
    physical = {id(node) for node in native.nodes}

    def consume(self, times):
        if id(self) in physical:
            observed.append(self)
        return original(self, times)

    with monkeypatch.context() as guard:
        guard.setattr(_randomized_poisson, "fn", consume)
        _forbid_owned_sampling(guard, native.model, native.generator)
        results = dn.step_network(
            native.model, extra=native.extra, draws=packet, callbacks=callbacks
        )
    assert observed == list(native.order)
    _close(_native_state(native.model, native.extra), expected)
    assert results["trace"]["t"].shape == (1,)
    _close(
        results["trace"]["left.v"][0],
        tensors.state["populations"]["left"]["integrator"]["v"],
    )
    actual = torch.autograd.grad(
        _loss(native.model),
        (*_gradient_sources(native), *_base_draws(packet)),
        allow_unused=True,
    )
    wanted = torch.autograd.grad(
        _state_loss(expected),
        _functional_sources(functional, tensors, _gradient_sources(source)),
    )
    _close(actual[: len(wanted)], wanted)
    assert all(
        value.isfinite().all() and value.abs().sum() > 0
        for value in actual[: len(wanted)]
    )
    assert all(
        value is None or value.count_nonzero() == 0 for value in actual[len(wanted) :]
    )
    assert all(not node._spike_times.requires_grad for node in native.nodes)
    _close(native.generator.get_state(), packet.rng["stimulation"]["stream_0"])
    for root, saved in zip(native.roots, bindings, strict=True):
        _same_waveforms(root, saved)
    _same_parameters(native.model, parameters)
    _assert_packet_equal(packet, saved_packet)
    _assert_packet_equal(source.model.state_dict_for_checkpoint(), saved_source)
    _close(torch.get_rng_state(), global_rng)


def test_shared_physical_nodes_cross_population_intra_extra_banks_keep_the_last_schedule():
    source, native = _case(cross_banks=True), _case(cross_banks=True)
    functional, tensors = dn.func.make_functional_network(
        source.model, extra=source.extra
    )
    packet = _changed_packet(functional.sample_runtime_draws(tensors.state))
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    native.model.step(extra=native.extra, draws=packet)
    _close(_native_state(native.model, native.extra), expected)
    assert native.extra["right"][1].waveforms[0] is native.roots[0].waveforms[0]
    assert native.extra["left"][1].waveforms[0] is native.roots[1].waveforms[0]
    actual = torch.autograd.grad(_loss(native.model), _gradient_sources(native))
    wanted = torch.autograd.grad(
        _state_loss(expected),
        _functional_sources(functional, tensors, _gradient_sources(source)),
    )
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)


def test_network_replay_continues_automatic_sampling_after_native_checkpoint_restore():
    source, native, restored = _case(), _case(), _case()
    functional, tensors = dn.func.make_functional_network(
        source.model, extra=source.extra
    )
    packet = _changed_packet(functional.sample_runtime_draws(tensors.state))
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    native.model.step(extra=native.extra, draws=packet)
    saved = _snapshot(native.model.state_dict_for_checkpoint())
    external = {
        name: _snapshot(root.state_dict())
        for name, (_field, root) in native.extra.items()
    }
    restored.model.restore_dict_from_checkpoint(saved)
    for name, (_field, root) in restored.extra.items():
        root.load_state_dict(external[name])
    following = functional.sample_runtime_draws(expected)
    final, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, expected, draws=following
    )
    native.model.step(extra=native.extra)
    dn.step(restored.model, extra=restored.extra)
    _close(_native_state(native.model, native.extra), final)
    _close(_native_state(restored.model, restored.extra), final)
    _close(restored.generator.get_state(), native.generator.get_state())


def _mixed_case():
    model = _noise_network()
    generator = model.left.mech.probe.primary.rng
    pulse = _waveform(generator=generator, amp=0.2 * nA)
    model.left[:, 1].inject(pulse + pulse)
    temporal = _waveform(generator=generator)
    extra = {
        "right": (
            torch.linspace(-1.25, 1.75, model.right.v.numel()).reshape(
                model.right.shape
            ),
            temporal + temporal,
        )
    }
    model.attach_netstim(
        dn.NetStim(
            N=1, interval=0.075, noise=0.4, start=0.0, max_spikes=8, tau=DT, seed=71
        )
    )
    model.netstim._seeder.manual_seed(709)
    model.initialize(DT)
    with torch.no_grad():
        model.netstim.next_stoch_time.zero_()
    return model, extra, pulse, generator


def test_borrowed_noise_sde_stream_and_netstim_keep_shared_and_population_boundaries(
    monkeypatch,
):
    source, extra, _, _ = _mixed_case()
    native, native_extra, pulse, generator = _mixed_case()
    functional, tensors = dn.func.make_functional_network(source, extra=extra)
    packet = _changed_packet(functional.sample_runtime_draws(tensors.state))
    assert not packet.rng["stimulation"]
    assert packet.draws["netstim"] is not None
    assert any(set(frame["cpu"]) == {"step"} for frame in packet.rng["model"].values())
    assert any(
        set(frame["cpu"]) == {"left", "right"} for frame in packet.rng["model"].values()
    )
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, native, generator, native.netstim._rng)
        assert dn.step(native, extra=native_extra, draws=packet) is native
    _close(_native_state(native, native_extra), expected)
    assert native.left.mech.probe.primary is native.right.mech.probe.primary
    assert native.left.mech.probe.primary.rng is pulse.generator is generator
    _close(generator.get_state(), packet.rng["model"]["stream_0"]["cpu"]["step"])
    _close(native.netstim._rng.get_state(), packet.rng["netstim"])


def _foreign_rand_case():
    donor = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-45.0,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    donor.insert(_RandOnlyLeak.rename("probe"))
    donor.initialize()
    generator = donor.mech.probe.offset_rng.rng
    pulse = _waveform(generator=generator, amp=0.2 * nA)
    borrower = _population("intra", waveform=pulse + pulse)[0]
    model = dn.Network({"borrower": borrower, "donor": donor})
    model.train()
    model.initialize(DT)
    for population in model.populations.values():
        population.detach()
    return model, pulse, generator


def test_foreign_rand_only_owner_supplies_one_authoritative_rng_for_borrower(
    monkeypatch,
):
    source, _, _ = _foreign_rand_case()
    native, pulse, generator = _foreign_rand_case()
    functional, tensors = dn.func.make_functional_network(source)
    packet = _changed_packet(functional.sample_runtime_draws(tensors.state))
    assert all(local["noise"] is None for local in packet.draws["populations"].values())
    assert packet.rng["model"] and not packet.rng["stimulation"]
    offset = native.donor.mech.probe.offset.detach().clone()
    stream = native.donor.mech.probe.offset_rng
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, native, generator)
        native.step(draws=packet)
    _close(functional.extract(native).state, expected)
    _close(native.donor.mech.probe.offset, offset)
    assert native.donor.mech.probe.offset_rng is stream
    assert stream.rng is pulse.generator is generator
    _close(generator.get_state(), _one_stream(expected))
    following = functional.sample_runtime_draws(expected)
    final, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, expected, draws=following
    )
    native.step()
    _close(functional.extract(native).state, final)
    _close(generator.get_state(), _one_stream(final))


@pytest.mark.parametrize(
    "invalid", ("names", "uniform", "rng", "model_boundary", "netstim", "tape")
)
def test_bad_network_poisson_packets_reject_before_bank_clock_or_rng_mutation(
    invalid, monkeypatch
):
    model, extra, pulse, generator = _mixed_case()
    functional, tensors = dn.func.make_functional_network(model, extra=extra)
    packet = functional.sample_runtime_draws(tensors.state)
    if invalid == "tape":
        packet = functional.sample_runtime_draws(tensors.state, steps=1)
    else:
        draws, rng = _snapshot(packet.draws), _snapshot(packet.rng)
        if invalid == "names":
            draws["populations"]["right"]["stimulation"] = {}
        elif invalid == "uniform":
            key, value = next(
                iter(draws["populations"]["right"]["stimulation"].items())
            )
            draws["populations"]["right"]["stimulation"][key] = torch.ones_like(value)
        elif invalid == "rng":
            frame = next(
                frame
                for frame in rng["model"].values()
                if set(frame["cpu"]) == {"step"}
            )
            frame["cpu"]["step"] = torch.zeros_like(frame["cpu"]["step"])
        elif invalid == "model_boundary":
            frame = next(
                frame
                for frame in rng["model"].values()
                if set(frame["cpu"]) == {"step"}
            )
            frame["cpu"] = {"left": frame["cpu"]["step"]}
        else:
            draws["netstim"] = None
        packet = packet._replace(draws=draws, rng=rng)
    saved = _snapshot(model.state_dict_for_checkpoint())
    external = _snapshot(extra["right"][1].state_dict())
    saved_packet = _snapshot(packet)
    global_rng = torch.get_rng_state().clone()
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model, generator, model.netstim._rng)
        with pytest.raises(
            (ValueError, TypeError, RuntimeError, dn.func.FunctionalizationError)
        ):
            model.step(extra=extra, draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _assert_packet_equal(_snapshot(extra["right"][1].state_dict()), external)
    _assert_packet_equal(packet, saved_packet)
    _close(torch.get_rng_state(), global_rng)
    assert pulse.generator is generator


def test_bank_failure_restores_all_shared_bindings_and_retries_without_publishing_rng(
    monkeypatch,
):
    source, native = _case(), _case()
    functional, tensors = dn.func.make_functional_network(
        source.model, extra=source.extra
    )
    packet = _changed_packet(functional.sample_runtime_draws(tensors.state))
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    voltage = {
        name: population.v.detach().clone()
        for name, population in native.model.populations.items()
    }
    time, rng = native.model.t.clone(), native.generator.get_state().clone()
    original = _randomized_poisson.fn
    last_extra = native.extra["left"][1].waveforms[0]

    def fail_live(self, times):
        result = original(self, times)
        if self is last_extra:
            raise RuntimeError(
                "network bank failed after consuming a supplied invocation"
            )
        return result

    with monkeypatch.context() as guard:
        guard.setattr(_randomized_poisson, "fn", fail_live)
        _forbid_owned_sampling(guard, native.model, native.generator)
        with pytest.raises(RuntimeError, match="network bank failed"):
            native.model.step(extra=native.extra, draws=packet)
    for name, population in native.model.populations.items():
        _close(population.v, voltage[name])
    _close(native.model.t, time)
    _close(native.generator.get_state(), rng)
    assert all("_native_poisson_replay" not in vars(node) for node in native.nodes)
    native.model.step(extra=native.extra, draws=packet)
    _close(_native_state(native.model, native.extra), expected)
    assert all("_native_poisson_replay" not in vars(node) for node in native.nodes)


@pytest.mark.parametrize("attribute", ("_rng", "_seeder"))
def test_network_poisson_cannot_borrow_netstim_private_rng_before_native_mutation(
    attribute,
    monkeypatch,
):
    model, extra, pulse, generator = _mixed_case()
    functional, tensors = dn.func.make_functional_network(model, extra=extra)
    packet = functional.sample_runtime_draws(tensors.state)
    pulse.generator = getattr(model.netstim, attribute)
    saved = _snapshot(model.state_dict_for_checkpoint())
    external = _snapshot(extra["right"][1].state_dict())
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(
            guard, model, generator, model.netstim._rng, model.netstim._seeder
        )
        with pytest.raises(
            (ValueError, dn.func.FunctionalizationError), match="Poisson|generator|RNG"
        ):
            dn.func.make_functional_network(model, extra=extra)
        with pytest.raises(
            (ValueError, dn.func.FunctionalizationError), match="Poisson|generator|RNG"
        ):
            model.step(extra=extra, draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _assert_packet_equal(_snapshot(extra["right"][1].state_dict()), external)


def test_explicit_global_network_poisson_replay_clears_deferred_handoff_frames():
    source, native = _case(global_stream=True), _case(global_stream=True)
    # Handoff transfers model-owned state. The caller-owned extra bindings
    # participate in the subsequent replay rather than the handoff plan.
    handoff, _ = dn.func.make_functional_network(source.model)
    dn.func.commit_continuation_(
        handoff, native.model, dn.func.capture_continuation(handoff)
    )
    functional, _ = dn.func.make_functional_network(source.model, extra=source.extra)
    boundary = dn.func.capture_continuation(functional)
    assert any(
        vars(node).get("_functional_poisson_rng") is not None for node in native.nodes
    )
    packet = _changed_packet(functional.sample_runtime_draws(boundary.state))
    expected, _ = functional.prepare_and_step(
        boundary.parameters, boundary.constants, boundary.state, draws=packet
    )
    torch.manual_seed(9931)
    native.model.step(extra=native.extra, draws=packet)
    _close(torch.get_rng_state(), packet.rng["stimulation"]["stream_0"])
    assert all(
        (pending := vars(node).get("_functional_poisson_rng")) is None
        or pending.state is None
        for node in native.nodes
    )
    following = functional.sample_runtime_draws(expected)
    final, _ = functional.prepare_and_step(
        boundary.parameters, boundary.constants, expected, draws=following
    )
    native.model.step(extra=native.extra)
    _close(_native_state(native.model, native.extra), final)


def test_network_poisson_replay_reuses_actual_fullgraph_native_numerical_kernels(
    monkeypatch,
):
    source, extra, _, _ = _mixed_case()
    native, native_extra, pulse, generator = _mixed_case()
    functional, tensors = dn.func.make_functional_network(source, extra=extra)
    first = _changed_packet(functional.sample_runtime_draws(tensors.state))
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=first
    )
    second = _changed_packet(functional.sample_runtime_draws(expected))
    final, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, expected, draws=second
    )
    noise_bindings = _noise_bindings(native)
    with dn.ctx(JIT=2, BACKEND="aot_eager", FULLGRAPH=True):
        with monkeypatch.context() as guard:
            _forbid_owned_sampling(guard, native, generator, native.netstim._rng)
            native.step(extra=native_extra, draws=first)
            _assert_noise_bindings(native, noise_bindings)
            kernels = {
                name: dict(population.integrator._compiled_kernels)
                for name, population in native.populations.items()
            }
            assert all(kernels.values())
            native.step(extra=native_extra, draws=second)
            _assert_noise_bindings(native, noise_bindings)
        for name, population in native.populations.items():
            assert population.integrator.backend == "aot_eager"
            assert (
                population.integrator._compiled_kernels.keys() == kernels[name].keys()
            )
            assert all(
                population.integrator._compiled_kernels[key] is value
                for key, value in kernels[name].items()
            )
    actual = torch.autograd.grad(
        _loss(native), (native.left.mech.probe.g_mu_param, pulse.waveform.amp)
    )
    source_pulse = source.left.injections[0][0].waveforms[0]
    wanted = torch.autograd.grad(
        _state_loss(final),
        _functional_sources(
            functional,
            tensors,
            (source.left.mech.probe.g_mu_param, source_pulse.waveform.amp),
        ),
    )
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    _close(_native_state(native, native_extra), final)
    assert all(
        "_native_poisson_replay" not in vars(node)
        for root in (native.left.injections[0][0], native_extra["right"][1])
        for node in root.modules()
    )


def test_compiled_noise_ownership_restores_registered_aliases_after_kernel_failure(
    monkeypatch,
):
    native = _noise_network()
    functional, tensors = dn.func.make_functional_network(native)
    packet = _changed_packet(functional.sample_runtime_draws(tensors.state))
    original = native.left.mech.probe._buffers["e"]
    native.left.integrator.register_buffer("noise_test_alias", original)
    noise_bindings = _noise_bindings(native)
    saved_packet = _snapshot(packet)
    clock = native.t.clone()
    voltages = {
        name: population.v.detach().clone()
        for name, population in native.populations.items()
    }
    streams = tuple(
        module for module in native.modules() if isinstance(module, RNGModule)
    )
    rng = tuple(_snapshot(stream.get_extra_state()) for stream in streams)
    entered = []

    def fail(*args, **kwargs):
        del args, kwargs
        entered.append(True)
        borrowed = native.left.mech.probe._buffers["e"]
        assert borrowed is not original
        assert native.left.integrator._buffers["noise_test_alias"] is borrowed
        assert not borrowed.requires_grad
        raise RuntimeError("compiled numerical kernel failed after NOISE sampling")

    with dn.ctx(JIT=2, BACKEND="aot_eager", FULLGRAPH=True):
        with monkeypatch.context() as guard:
            guard.setattr(native.left.integrator, "_kernel", fail)
            _forbid_owned_sampling(guard, native)
            with pytest.raises(
                RuntimeError, match="numerical kernel failed after NOISE"
            ):
                native.step(draws=packet)
    assert entered == [True]
    _assert_noise_bindings(native, noise_bindings)
    _close(native.t, clock)
    for name, population in native.populations.items():
        _close(population.v, voltages[name])
    for stream, saved in zip(streams, rng, strict=True):
        _assert_packet_equal(stream.get_extra_state(), saved)
    _assert_packet_equal(packet, saved_packet)


def test_custom_none_returning_noise_sampler_retains_compiled_training_frames():
    def build():
        population = dn.SingleCompartment(
            N=1, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(_NoneReturningNoiseLeak.rename("probe"))
        model = dn.Network({"cell": population})
        model.train()
        model.initialize(DT)
        return model

    reference, compiled = build(), build()
    bindings = _noise_bindings(compiled)
    for _index in range(2):
        reference.step()
    with dn.ctx(JIT=2, BACKEND="aot_eager", FULLGRAPH=True):
        compiled.step()
        _assert_noise_bindings(compiled, bindings)
        kernels = dict(compiled.cell.integrator._compiled_kernels)
        assert kernels
        compiled.step()
        _assert_noise_bindings(compiled, bindings)
        assert compiled.cell.integrator._compiled_kernels.keys() == kernels.keys()
        assert all(
            compiled.cell.integrator._compiled_kernels[key] is value
            for key, value in kernels.items()
        )
    _close(compiled.cell.v, reference.cell.v)
    _close(compiled.cell.mech.probe.e, reference.cell.mech.probe.e)
    _close(compiled.cell.mech.probe.eta, reference.cell.mech.probe.eta)
    actual = torch.autograd.grad(_loss(compiled), compiled.cell.mech.probe.g_param)[0]
    wanted = torch.autograd.grad(_loss(reference), reference.cell.mech.probe.g_param)[0]
    _close(actual, wanted)
    assert actual.isfinite().all() and actual.abs().sum() > 0
    for name, module in compiled.named_modules():
        if isinstance(module, RNGModule):
            _assert_packet_equal(
                _snapshot(module.get_extra_state()),
                _snapshot(reference.get_submodule(name).get_extra_state()),
            )


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_network_poisson_banks_preserve_shared_nodes_float32_and_gradients():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):

        def build():
            generator = torch.Generator().manual_seed(197)
            pulse = _waveform(generator=generator, amp=0.2 * nA)
            expression = pulse + pulse
            populations = {}
            for name in ("left", "right"):
                population = dn.Unmyelinated(
                    [2.0],
                    L=4.0,
                    dx=1.0,
                    dtype=torch.float32,
                    integrator=dn.bwd_euler_ub(method="pcr", imem=False),
                )
                population.insert(pas)
                population[:, 1].inject(expression)
                populations[name] = population
            model = dn.Network(populations)
            model.train()
            model.initialize(DT)
            return model, pulse, generator

        source, source_pulse, _ = build()
        native, pulse, generator = build()
        functional, tensors = dn.func.make_functional_network(source)
        packet = _changed_packet(functional.sample_runtime_draws(tensors.state))
        expected, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=packet
        )
        dn.step_network(native, draws=packet)
        for name, population in native.populations.items():
            torch.testing.assert_close(
                population.v,
                expected["populations"][name]["integrator"]["v"],
                rtol=2e-5,
                atol=2e-6,
            )
        actual = torch.autograd.grad(_loss(native), pulse.waveform.amp)[0]
        wanted = torch.autograd.grad(
            _state_loss(expected),
            _functional_sources(functional, tensors, (source_pulse.waveform.amp,)),
        )[0]
        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-6)
        assert actual.isfinite().all() and actual.abs().sum() > 0
        assert (
            pulse._spike_times.device.type == "mps"
            and pulse._spike_times.dtype == torch.float32
        )
        _close(generator.get_state(), packet.rng["stimulation"]["stream_0"])
