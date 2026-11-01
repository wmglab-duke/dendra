"""One runtime sampler leaves native sources untouched and replays in both engines."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra.models.mod import pas
from dendra.models.stim.waveform._poisson import _poisson_waveforms
from dendra.units import nA
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_native_handoff import _capture, _unchanged
from tests.func.test_network_runtime_noise import DT as NETWORK_DT
from tests.func.test_network_runtime_noise import _network as _noise_network
from tests.func.test_poisson_invocation_execution import _native_state
from tests.func.test_poisson_invocation_guards import _RepeatedInitializer
from tests.func.test_poisson_stimulation import DT as POISSON_DT
from tests.func.test_poisson_stimulation import _population as _bank_population
from tests.func.test_poisson_stimulation import _waveform
from tests.func.test_state_sde_execution import DT as STATE_DT
from tests.func.test_state_sde_execution import _HeunLeak
from tests.func.test_state_sde_execution import _population as _state_population
from tests.func.test_state_sde_execution import _SharedLeak
from tests.test_native_mechanism_poisson_profiles import _model as _profile_model
from tests.test_native_mechanism_poisson_streams import _shared_case
from tests.test_native_runtime_draw_tapes import (
    _functional_sources,
    _loss,
    _run,
    _state_loss,
)
from tests.test_native_runtime_draws import _environment as _environment
from tests.test_native_runtime_draws import _forbid_owned_sampling

pytestmark = pytest.mark.cpu
STEPS = 3


def _case(kind):
    if kind == "noise":
        return _state_population(_SharedLeak), None, STATE_DT
    if kind == "netstim":
        return _noise_network(stimulus=True), None, NETWORK_DT
    model, extra, _generator = _shared_case(
        "population" if kind == "poisson" else "mixed"
    )
    return model, extra, POISSON_DT


def _options(model, dt, extra):
    return {"dt": dt, "extra": extra}


def _runtime_packet(functional, packet, *, steps):
    """Select the runtime invocations from the unchanged public legacy sampler."""

    def local(plan, values):
        poisson = plan._poisson_plan
        if poisson is None:
            return values
        draws = {}
        for entry in poisson.entries:
            value = values["stimulation"][entry.key]
            width = max(poisson._runtime_counts[entry.key], 1)
            if poisson._capacities[entry.key] != width:
                value = value[:width] if steps is None else value[:, :width]
                if width == 1:
                    value = value.squeeze(0 if steps is None else 1)
            draws[entry.key] = value
        return {**values, "stimulation": draws}

    if isinstance(functional, dn.func.FunctionalNetwork):
        draws = {
            **packet.draws,
            "populations": {
                name: local(plan, packet.draws["populations"][name])
                for name, plan in functional.populations.items()
            },
        }
    else:
        draws = local(functional, packet.draws)
    return packet._replace(draws=draws)


def _generators(model, extra=None):
    values = list(_poisson_waveforms(model).values())
    bindings = (
        () if extra is None else extra.values() if isinstance(extra, dict) else (extra,)
    )
    for _field, waveform in bindings:
        values.extend(_poisson_waveforms(waveform).values())
    return tuple({id(value.generator): value.generator for value in values}.values())


def _forbid_lowering(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime sampling lowered or captured a numerical program")

    for name in ("make_functional", "make_functional_network", "capture_continuation"):
        monkeypatch.setattr(dn.func, name, forbidden)
    monkeypatch.setattr(dn, "compile", forbidden)
    monkeypatch.setattr(dn.func.FunctionalPopulation, "extract", forbidden)
    monkeypatch.setattr(dn.func.FunctionalNetwork, "extract", forbidden)


@pytest.mark.parametrize("kind", ("noise", "netstim", "poisson", "mixed"))
@pytest.mark.parametrize("steps", (None, 0, STEPS), ids=("frame", "empty", "tape"))
def test_native_sampling_matches_shared_requests_without_lowering_or_source_mutation(
    kind, steps, monkeypatch
):
    model, extra, dt = _case(kind)
    functional, tensors = dn.func.make_functional(model, dt=dt, extra=extra)
    legacy = functional.sample_runtime_draws(tensors.state, steps=steps)
    expected = _runtime_packet(functional, legacy, steps=steps)
    saved = _capture(model)
    saved_extra = _snapshot(extra)
    generators = _generators(model, extra)
    generator_states = tuple(generator.get_state().clone() for generator in generators)
    external = (
        () if extra is None else extra.values() if isinstance(extra, dict) else (extra,)
    )
    external_tensors = tuple(
        (owner, registry, name, value, value.detach().clone(), value._version)
        for _field, waveform in external
        for owner in waveform.modules()
        for registry in ("_parameters", "_buffers")
        for name, value in getattr(owner, registry).items()
        if value is not None
    )
    nodes = list(_poisson_waveforms(model).values())
    for _field, waveform in (
        () if extra is None else extra.values() if isinstance(extra, dict) else (extra,)
    ):
        nodes.extend(_poisson_waveforms(waveform).values())
    pending = tuple((node, vars(node).get("_functional_poisson_rng")) for node in nodes)
    with monkeypatch.context() as guard:
        _forbid_lowering(guard)
        netstim = model.netstim if isinstance(model, dn.Network) else None
        _forbid_owned_sampling(
            guard, model, *generators, *(() if netstim is None else (netstim._rng,))
        )
        packet = dn.func.sample_runtime_draws(
            model, steps=steps, **_options(model, dt, extra)
        )
        repeated = dn.func.sample_runtime_draws(
            model, steps=steps, **_options(model, dt, extra)
        )
    _assert_packet_equal(packet, expected)
    _assert_packet_equal(repeated, packet)
    _unchanged(model, saved)
    _assert_packet_equal(extra, saved_extra)
    for owner, registry, name, value, frozen, version in external_tensors:
        assert getattr(owner, registry)[name] is value and value._version == version
        torch.testing.assert_close(value, frozen, rtol=0, atol=0)
    assert all(
        vars(node).get("_functional_poisson_rng") is holder for node, holder in pending
    )
    for generator, state in zip(generators, generator_states, strict=True):
        _close(generator.get_state(), state)
    for value in pytree.tree_leaves(packet):
        if torch.is_tensor(value):
            assert not value.requires_grad and value.grad_fn is None


def _selected(owner):
    return tuple(
        value
        for name, value in owner.named_parameters()
        if name.endswith(("waveform.amp", "offset_mu_param"))
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", ("run", "longrun_checkpointed"))
def test_sampled_packets_replay_values_gradients_and_checkpoint_continuation(
    network, method
):
    kind = "mixed" if network else "poisson"
    source, source_extra, dt = _case(kind)
    model, extra, _ = _case(kind)
    functional, tensors = dn.func.make_functional(source, dt=dt, extra=source_extra)
    packet = dn.func.sample_runtime_draws(
        model, steps=STEPS, **_options(model, dt, extra)
    )
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=STEPS,
        draws=packet,
    )
    _run(
        model,
        method,
        dt,
        steps=STEPS,
        extra=extra,
        draws=packet,
        **({"chunklength": 2} if method != "run" else {}),
    )
    _close(_native_state(model, extra), expected)
    saved = _snapshot(model.state_dict_for_checkpoint())
    wanted = torch.autograd.grad(
        _state_loss(expected, network),
        _functional_sources(functional, tensors, _selected(source)),
    )
    actual = torch.autograd.grad(_loss(model), _selected(model))
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in wanted)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    following = dn.func.sample_runtime_draws(
        model, steps=2, **_options(model, dt, extra)
    )
    receiver, receiving_extra, _ = _case(kind)
    receiver.restore_dict_from_checkpoint(saved)
    restored = dn.func.sample_runtime_draws(
        receiver, steps=2, **_options(receiver, dt, receiving_extra)
    )
    continued = dn.func.sample_runtime_draws(
        functional,
        dn.func.Continuation(expected, tensors.parameters, tensors.constants),
        steps=2,
    )
    _assert_packet_equal(following, restored)
    _assert_packet_equal(following, continued)


@pytest.mark.parametrize(
    ("network", "kind"),
    [
        (network, kind)
        for network in (False, True)
        for kind in ("plan", "raw", "prepared", "kernel", "runner")
        if not (network and kind == "prepared")
    ],
)
def test_explicit_state_sampling_is_shared_by_plans_bindings_kernels_and_runners(
    network, kind
):
    model, extra, dt = _case("netstim" if network else "noise")
    functional, tensors = dn.func.make_functional(model, dt=dt, extra=extra)
    if kind == "plan":
        target = functional
    elif kind == "raw":
        target = functional.bind(tensors)
    elif kind == "prepared":
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        target = functional.bind(tensors.parameters, prepared)
    elif kind == "kernel":
        target = functional.compile_rollout_chunk(2, backend="eager", disable=True)
    else:
        target = dn.compile(model, dt=dt, steps=2, backend="eager", disable=True)
        functional = target.functional
        tensors = functional.extract()
    boundary = dn.func.Continuation(
        tensors.state, tensors.parameters, tensors.constants
    )
    saved = _snapshot(boundary)
    legacy = functional.sample_runtime_draws(tensors.state, steps=2)
    expected = _runtime_packet(functional, legacy, steps=2)
    packet = dn.func.sample_runtime_draws(target, boundary, steps=2)
    _assert_packet_equal(packet, expected)
    _assert_packet_equal(boundary, saved)
    with pytest.raises((TypeError, ValueError)):
        dn.func.sample_runtime_draws(target, steps=2)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("cached", (False, True), ids=("fresh", "cached"))
def test_canonical_poisson_layout_is_runtime_only_and_sampling_never_initializes(
    network, cached, monkeypatch
):
    model, _pulses = _profile_model(network, cached=cached)
    functional, tensors = dn.func.make_functional(model, dt=POISSON_DT)
    legacy = functional.sample_runtime_draws(tensors.state, steps=STEPS)
    before = _snapshot(legacy)
    expected = _runtime_packet(functional, legacy, steps=STEPS)

    def forbidden(self, v, values):
        raise AssertionError("Runtime sampling entered fresh initialization")

    with monkeypatch.context() as guard:
        guard.setattr(_RepeatedInitializer, "initial_values", forbidden)
        packet = dn.func.sample_runtime_draws(model, dt=POISSON_DT, steps=STEPS)
    _assert_packet_equal(packet, expected)
    _assert_packet_equal(legacy, before)
    local = packet.draws["populations"].values() if network else (packet.draws,)
    assert all(
        value.shape == (STEPS, 5)
        for draws in local
        for value in draws["stimulation"].values()
    )
    _assert_packet_equal(
        dn.func.sample_runtime_draws(functional, tensors.state, steps=STEPS), packet
    )
    if not cached:
        locals_legacy = (
            legacy.draws["populations"].values() if network else (legacy.draws,)
        )
        assert all(
            value.shape == (STEPS, 3, 5)
            for draws in locals_legacy
            for value in draws["stimulation"].values()
        )
    actual, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=STEPS,
        draws=packet,
    )
    reference, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=STEPS,
        draws=legacy,
    )
    _close(actual, reference)


@pytest.mark.parametrize("defect", ("reserved_nan", "arbitrary_rows"))
def test_functional_runtime_ingress_validates_legacy_rows_before_adapting(defect):
    model, _ = _profile_model(False, cached=False)
    functional, tensors = dn.func.make_functional(model, dt=POISSON_DT)
    packet = _snapshot(functional.sample_runtime_draws(tensors.state))
    (key,) = packet.draws["stimulation"]
    if defect == "reserved_nan":
        packet.draws["stimulation"][key][-1].fill_(torch.nan)
    else:
        packet.draws["stimulation"][key] = packet.draws["stimulation"][key][:2]
    with pytest.raises(ValueError, match="Poisson|poisson"):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state, draws=packet
        )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_canonical_runtime_packets_replay_through_both_compiled_engines(
    network, execution
):
    model, _ = _profile_model(network, cached=False)
    runner = dn.compile(
        model,
        dt=POISSON_DT,
        steps=4,
        backend="aot_eager",
        execution=execution,
        inference=execution == "while",
        trainable=("*waveform.amp",),
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    initial = dn.func.Continuation(
        runner.initial_state, runner.parameters, runner.constants
    )
    legacy = runner.functional.sample_runtime_draws(initial.state, steps=4)
    canonical = dn.func.sample_runtime_draws(model, dt=POISSON_DT, steps=4)
    expected, _ = runner(initial, draws=legacy)
    actual, _ = runner(initial, draws=canonical)
    _close(actual, expected)
    if execution == "scan":
        leaves = tuple(runner.trainable_parameters.values())
        wanted = torch.autograd.grad(_state_loss(expected, network), leaves)
        observed = torch.autograd.grad(_state_loss(actual, network), leaves)
        _close(observed, wanted)
        assert all(value.isfinite().all() and value.abs().sum() > 0 for value in wanted)


def test_native_sampling_reads_logical_deferred_rng_without_resetting_ambient_rng():
    model, pulse, _ = _bank_population(
        "intra", waveform=_waveform(generator=torch.default_generator, amp=0.2 * nA)
    )
    functional, _ = dn.func.make_functional(model, dt=POISSON_DT)
    boundary = dn.func.capture_continuation(functional)
    dn.func.commit_continuation_(functional, model, boundary)
    holder = vars(pulse)["_functional_poisson_rng"]
    pending = holder.state
    expected = functional.sample_runtime_draws(boundary.state, steps=2)
    torch.manual_seed(9173)
    ambient = torch.get_rng_state().clone()
    saved = _capture(model)
    packet = dn.func.sample_runtime_draws(model, dt=POISSON_DT, steps=2)
    _assert_packet_equal(packet, expected)
    _unchanged(model, saved)
    assert vars(pulse)["_functional_poisson_rng"] is holder and holder.state is pending
    _close(torch.get_rng_state(), ambient)


def test_inference_context_sampling_returns_versioned_detached_checkpoint_operands():
    model = _state_population(_HeunLeak)
    saved = _capture(model)
    with torch.inference_mode():
        packet = dn.func.sample_runtime_draws(model, dt=STATE_DT, steps=STEPS)
    _unchanged(model, saved)
    for value in pytree.tree_leaves(packet):
        if torch.is_tensor(value):
            assert not value.is_inference() and not value.requires_grad
    model.longrun_checkpointed(
        steps=STEPS, chunklength=2, dt=STATE_DT, draws=packet, safe_checkpoint=True
    )
    gate = next(iter(model.mech.sde.DE.values()))
    gradient = torch.autograd.grad(model.v.square().mean(), gate.sigma_param)[0]
    assert gradient.isfinite().all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("steps", (-1, True, 1.5, "3"))
def test_invalid_sampling_horizons_reject_before_touching_native_sources(steps):
    model, extra, dt = _case("poisson")
    saved = _capture(model)
    with pytest.raises((TypeError, ValueError)):
        dn.func.sample_runtime_draws(model, steps=steps, **_options(model, dt, extra))
    _unchanged(model, saved)


def test_sampling_rejects_implicit_initialization_native_state_and_bound_overrides():
    uninitialized = dn.SingleCompartment(
        N=1, C=1, integrator=dn.bwd_euler_sc(imem=False)
    )
    uninitialized.insert(pas)
    assert not uninitialized.is_built
    with pytest.raises((ValueError, dn.func.FunctionalizationError)):
        dn.func.sample_runtime_draws(uninitialized, dt=STATE_DT)
    assert not uninitialized.is_built
    model, extra, dt = _case("noise")
    functional, tensors = dn.func.make_functional(model, dt=dt, extra=extra)
    for target, state, options in (
        (model, {}, {"dt": dt}),
        (functional, tensors.state, {"dt": 2 * dt}),
        (
            functional.bind(tensors),
            tensors.state,
            {"extra": (torch.ones_like(model.v), dn.constant(value=1.0))},
        ),
    ):
        with pytest.raises((TypeError, ValueError)):
            dn.func.sample_runtime_draws(target, state, **options)
    with pytest.raises(TypeError):
        dn.func.sample_runtime_draws(object())


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_runtime_sampler_preserves_device_streams_and_checkpointed_gradients():
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        model = _state_population(_HeunLeak)
        functional, tensors = dn.func.make_functional(model, dt=STATE_DT)
        saved, ambient = _capture(model), torch.mps.get_rng_state().clone()
        packet = dn.func.sample_runtime_draws(model, dt=STATE_DT, steps=STEPS)
        expected = functional.sample_runtime_draws(tensors.state, steps=STEPS)
        _assert_packet_equal(packet, expected)
        _unchanged(model, saved)
        _close(torch.mps.get_rng_state(), ambient)
        model.longrun_checkpointed(
            steps=STEPS, chunklength=2, dt=STATE_DT, draws=packet, safe_checkpoint=True
        )
        gate = next(iter(model.mech.sde.DE.values()))
        gradient = torch.autograd.grad(model.v.square().mean(), gate.sigma_param)[0]
        assert (
            gradient.device.type == "mps"
            and gradient.isfinite().all()
            and gradient.abs().sum() > 0
        )


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_native_sampling_rejects_multidevice_networks_before_touching_sources():
    with dn.ctx(DTYPE=torch.float32, DEVICE="cpu"):
        cpu = _state_population(_SharedLeak)
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        accelerator = _state_population(_SharedLeak)
    model = dn.Network({"cpu": cpu, "accelerator": accelerator})
    model.train()
    model.initialize(STATE_DT)
    saved = _capture(model)
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="device"):
        dn.func.sample_runtime_draws(model, steps=2)
    _unchanged(model, saved)
