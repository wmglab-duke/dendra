"""Detached continuation handoff aligns native sources and runtime together."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _assign, _close, _plain, _snapshot
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_network_connection_parameters import DT as CONNECTION_DT
from tests.func.test_network_connection_parameters import (
    _native_targets,
)
from tests.func.test_network_connection_parameters import _network as _parameter_network
from tests.func.test_network_connection_parameters import (
    _packet_targets,
)
from tests.func.test_network_connection_sampling import (
    _check_native as _check_sampled_native,
)
from tests.func.test_network_connection_sampling import _network as _sampled_network
from tests.func.test_network_connection_sampling import _targets as _sampled_targets
from tests.networks.test_native_shared_schedule_bindings import (
    NETWORK_DT,
    _scheduled_network,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous_rng = torch.get_rng_state().clone()
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield
    torch.set_rng_state(previous_rng)


def _make(network, runner=False, *, callbacks=None, evaluation=False):
    model = _network() if network else _population()
    if evaluation:
        model.eval()
        model.initialize(DT)
    if runner:
        subject = dn.compile(
            model,
            dt=DT,
            steps=2,
            trainable="*g_param",
            callbacks=callbacks,
            backend="eager",
            disable=True,
        )
        functional = subject.functional
        initial = dn.func.Continuation(
            subject.initial_state, subject.parameters, subject.constants
        )
    else:
        functional, tensors = dn.func.make_functional(model, dt=DT)
        subject = functional
        initial = dn.func.Continuation(
            tensors.state,
            tensors.independent_parameters(trainable="*g_param"),
            tensors.constants,
        )
    parameters = _plain(initial.parameters)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    replacement = 1.4 * upstream
    _assign(parameters, network, replacement)
    initial = dn.func.Continuation(initial.state, parameters, initial.constants)
    _, first = dn.func.run(subject, initial, steps=2)
    return subject, functional, first["continuation"], upstream


def _native_parameter(model, network):
    return model.left.mech.leak.g_param if network else model.mech.leak.g_param


def _native_voltage(model, network):
    return (
        torch.cat([value.v.flatten() for value in model.populations.values()])
        if network
        else model.v
    )


def _capture(model):
    bindings = {
        **{
            "parameter." + name: value
            for name, value in model.named_parameters(remove_duplicate=False)
        },
        **{
            "buffer." + name: value
            for name, value in model.named_buffers(remove_duplicate=False)
        },
    }
    return (
        {
            name: (
                value,
                value.detach().clone(),
                None if torch.is_inference(value) else value._version,
            )
            for name, value in bindings.items()
        },
        _snapshot(model.state_dict_for_checkpoint()),
        torch.get_rng_state().clone(),
    )


def _unchanged(model, saved):
    previous, state, rng = saved
    bindings = {
        **{
            "parameter." + name: value
            for name, value in model.named_parameters(remove_duplicate=False)
        },
        **{
            "buffer." + name: value
            for name, value in model.named_buffers(remove_duplicate=False)
        },
    }
    assert bindings.keys() == previous.keys()
    for name, (original, value, version) in previous.items():
        assert bindings[name] is original
        assert (None if torch.is_inference(original) else original._version) == version
        torch.testing.assert_close(original, value, rtol=0, atol=0)
    _assert_packet_equal(model.state_dict_for_checkpoint(), state)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("runner", (False, True), ids=("plan", "runner"))
def test_handoff_preserves_native_parameters_and_owns_the_detached_suffix(
    network, runner
):
    subject, functional, packet, upstream = _make(network, runner)
    target = _network() if network else _population()
    native = _native_parameter(target, network)
    native.grad = torch.full_like(native, 0.2)
    previous_grad = native.grad
    parameters = dict(target.named_parameters(remove_duplicate=False))
    flags = {name: value.requires_grad for name, value in parameters.items()}
    frozen = _snapshot((packet.state, packet.parameters, packet.constants))
    source = _target(packet.parameters, network)
    source_graph = source.grad_fn
    expected, _ = dn.func.run(subject, packet, steps=3)

    assert dn.func.commit_continuation_(subject, target, packet) is target
    assert _native_parameter(target, network) is native
    assert native.grad is previous_grad
    torch.testing.assert_close(native, source)
    for name, value in target.named_parameters(remove_duplicate=False):
        assert value is parameters[name] and value.requires_grad == flags[name]
    if network:
        assert target.right.mech.leak.g_param is native
    _close(functional.extract(target).state, packet.state)
    leaves = pytree.tree_leaves(functional.extract(target).state)
    assert all(
        not leaf.requires_grad and leaf.grad_fn is None
        for leaf in leaves
        if isinstance(leaf, torch.Tensor)
    )

    if network:
        target.run(3 * DT)
    else:
        target.run(tstop=3 * DT, dt=DT)
    torch.testing.assert_close(
        _native_voltage(target, network), _voltage(expected, network)
    )
    gradient, previous_graph = torch.autograd.grad(
        _native_voltage(target, network).square().sum(),
        (native, upstream),
        allow_unused=True,
    )
    assert gradient.isfinite().all() and gradient.abs().sum() > 0
    assert previous_graph is None
    assert source.grad_fn is source_graph and source_graph is not None
    _close((packet.state, packet.parameters, packet.constants), frozen)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("invalid", ("state", "parameter", "constant"))
def test_handoff_rejects_invalid_packets_before_mutating_the_target(network, invalid):
    subject, _, packet, _ = _make(network)
    state, parameters, constants = _plain(
        (packet.state, packet.parameters, packet.constants)
    )
    if invalid == "state":
        local = state["populations"]["left"] if network else state
        local["integrator"]["v"] = local["integrator"]["v"][..., :0]
    elif invalid == "parameter":
        raw = _target(parameters, network)
        _assign(parameters, network, raw.new_zeros(raw.numel() + 1))
    else:
        local = constants["populations"]["left"] if network else constants
        local["dt"] = local["dt"] * 2
    malformed = dn.func.Continuation(state, parameters, constants)
    target = _network() if network else _population()
    saved = _capture(target)
    with pytest.raises((dn.func.FunctionalizationError, TypeError, ValueError)):
        dn.func.commit_continuation_(subject, target, malformed)
    _unchanged(target, saved)


@pytest.mark.parametrize("changed", ("packet", "target"))
def test_handoff_rejects_broken_source_aliases_before_publication(changed):
    subject, _, packet, _ = _make(True)
    target = _network()
    if changed == "packet":
        parameters = _plain(packet.parameters)
        parameters["populations"]["right"][G] = parameters["populations"]["left"][
            G
        ].clone()
        packet = dn.func.Continuation(packet.state, parameters, packet.constants)
    else:
        value = target.right.mech.leak.g_param
        target.right.mech.leak.g_param = torch.nn.Parameter(
            value.detach().clone(), requires_grad=value.requires_grad
        )
    saved = _capture(target)
    with pytest.raises(
        (dn.func.FunctionalizationError, ValueError), match="alias|shar|structure"
    ):
        dn.func.commit_continuation_(subject, target, packet)
    _unchanged(target, saved)


def test_parametrized_sources_shared_with_connections_refresh_without_reinitializing():
    source, _ = _parameter_network(parametrized=True)
    functional, tensors = dn.func.make_functional(source, dt=CONNECTION_DT)
    parameters = tensors.independent_parameters(trainable="*")
    original, factor = _packet_targets(parameters, True)
    with torch.no_grad():
        original.add_(0.15)
        factor.mul_(1.1)
    initial = dn.func.Continuation(tensors.state, parameters, tensors.constants)
    _, first = dn.func.run(functional, initial, steps=2)
    packet = first["continuation"]
    expected, _ = dn.func.run(functional, packet, steps=3)
    target, shared = _parameter_network(parametrized=True)
    native = _native_targets(shared, True)
    target_buffer = shared.parametrizations.rho[0].offset
    before_rng = torch.get_rng_state().clone()

    dn.func.commit_continuation_(functional, target, packet)
    assert all(
        observed is previous
        for observed, previous in zip(
            _native_targets(shared, True), native, strict=True
        )
    )
    assert shared.parametrizations.rho[0].offset is target_buffer
    for wanted, observed in zip(_packet_targets(parameters, True), native, strict=True):
        torch.testing.assert_close(observed, wanted)
    for collection in (target.synapses, target.continuous_synapses):
        for connection in collection.values():
            assert connection.weight.weights[0].param is shared
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    target.run(3 * CONNECTION_DT)
    _close(functional.extract(target).state, expected)


def test_indexed_schedule_handoff_keeps_sources_shared_and_netstim_rng_exact():
    source, _, _ = _scheduled_network()
    functional, tensors = dn.func.make_functional(source, dt=NETWORK_DT)
    parameters = tensors.independent_parameters(trainable="*")
    schedule = parameters["netstim"]["schedule"]
    with torch.no_grad():
        schedule.time_source[1].add_(0.05 * NETWORK_DT)
        schedule.amplitude_source.mul_(1.2)
    initial = dn.func.Continuation(tensors.state, parameters, tensors.constants)
    _, first = dn.func.run(functional, initial, steps=2)
    packet = first["continuation"]
    expected, _ = dn.func.run(functional, packet, steps=4)
    target, times, amplitudes = _scheduled_network()
    generator = target.netstim._rng
    before_rng = torch.get_rng_state().clone()

    dn.func.commit_continuation_(functional, target, packet)
    assert target.netstim._rng is generator
    assert target.netstim._sched_t_source is times
    assert target.netstim._sched_w_source is amplitudes
    for connection in target.synapses.values():
        assert connection._sched_t_source is times
        assert connection._sched_w_source is amplitudes
    torch.testing.assert_close(times, schedule.time_source)
    torch.testing.assert_close(amplitudes, schedule.amplitude_source)
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    target.run(4 * NETWORK_DT)
    _close(functional.extract(target).state, expected)


def test_sampled_connection_handoff_reuses_retained_draws_without_advancing_rng():
    source, *_ = _sampled_network(parametrized=True)
    functional, tensors = dn.func.make_functional(source, dt=CONNECTION_DT)
    parameters = tensors.independent_parameters(trainable="*")
    initial = dn.func.Continuation(tensors.state, parameters, tensors.constants)
    with torch.no_grad():
        _sampled_targets(initial, True)[0].add_(0.0001)
    _, first = dn.func.run(functional, initial, steps=2)
    packet = first["continuation"]
    expected, _ = dn.func.run(functional, packet, steps=3)
    target, weights, delays, _ = _sampled_network(parametrized=True)
    native_sources = (
        weights.parametrizations.mean.original,
        weights.parametrizations.mean[0].factor,
        delays.mean,
    )
    before_rng = torch.get_rng_state().clone()

    dn.func.commit_continuation_(functional, target, packet)
    assert native_sources[0] is weights.parametrizations.mean.original
    assert native_sources[1] is weights.parametrizations.mean[0].factor
    assert native_sources[2] is delays.mean
    _check_sampled_native(functional, packet, target)
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    target.run(3 * CONNECTION_DT)
    _close(functional.extract(target).state, expected)


def test_handoff_preserves_callback_carry_for_its_separate_portable_bridge():
    configuration = {"trace": dn.Recorder(("v", "t"), dt=2 * DT)}
    subject, _, packet, _ = _make(False, True, callbacks=configuration)
    callback_state = packet.callback_state
    frozen = _snapshot(dict(callback_state))
    exported = subject.callbacks.export_state(callback_state)
    target = _population()

    dn.func.commit_continuation_(subject, target, packet)
    assert packet.callback_state is callback_state
    _close(dict(callback_state), frozen)
    receiving = dn.func.make_callbacks(
        target, {"trace": dn.Recorder(("v", "t"), dt=2 * DT)}, dt=DT
    )
    restored = receiving.restore_state(exported)
    assert restored.plan_token is not callback_state.plan_token
    expected, auxiliary = dn.func.run(subject, packet, steps=3)
    actual = target.run(
        tstop=3 * DT, dt=DT, callbacks=receiving, callback_state=restored
    )
    torch.testing.assert_close(target.v, expected["integrator"]["v"])
    _close(actual["trace"], auxiliary["callbacks"]["trace"])


def test_handoff_requires_an_exact_continuation_and_a_cold_host_boundary(monkeypatch):
    subject, _, packet, _ = _make(False)
    target = _population()
    with pytest.raises(TypeError, match="Continuation|continuation"):
        dn.func.commit_continuation_(subject, target, packet.state)

    class DerivedContinuation(dn.func.Continuation):
        pass

    with pytest.raises(TypeError, match="exact|Continuation|continuation"):
        dn.func.commit_continuation_(
            subject,
            target,
            DerivedContinuation(packet.state, packet.parameters, packet.constants),
        )
    with monkeypatch.context() as patch:
        patch.setattr(torch.compiler, "is_compiling", lambda: True)
        with pytest.raises(dn.func.FunctionalizationError, match="outside|host"):
            dn.func.commit_continuation_(object(), object(), object())
    with torch.autograd.forward_ad.dual_level():
        with pytest.raises(dn.func.FunctionalizationError, match="outside|host"):
            dn.func.commit_continuation_(object(), object(), object())
    with pytest.raises(dn.func.FunctionalizationError, match="outside|host"):
        torch.func.grad(
            lambda value: dn.func.commit_continuation_(object(), object(), object())
        )(torch.ones(()))


@pytest.mark.parametrize("network", (False, True))
def test_late_publication_failure_restores_bindings_versions_rng_and_old_graph(
    network, monkeypatch
):
    from dendra.func import _native_handoff as handoff

    subject, _, packet, _ = _make(network)
    target = _network() if network else _population()
    population = target.left if network else target
    native = _native_parameter(target, network)
    previous_voltage = population.v.detach().clone().requires_grad_()
    population.v = previous_voltage + 0.2 * native
    voltage, grad_fn = population.v, population.v.grad_fn
    # Squaring the raw Parameter saves its current version. A copy-and-copy-back
    # rollback would invalidate this prefix even if every value were recovered.
    prefix_loss = population.v.square().sum() + native.square().sum()
    wanted = torch.autograd.grad(
        prefix_loss, (native, previous_voltage), retain_graph=True
    )
    saved = _capture(target)
    publish = handoff._publish_runtime
    calls = 0

    def publish_then_fail(*args, **kwargs):
        nonlocal calls
        assert args[0] is target
        calls += 1
        publish(*args, **kwargs)
        raise RuntimeError("injected late publication failure")

    monkeypatch.setattr(handoff, "_publish_runtime", publish_then_fail)
    with pytest.raises(RuntimeError, match="injected late publication failure"):
        dn.func.commit_continuation_(subject, target, packet)

    assert calls == 1
    _unchanged(target, saved)
    assert population.v is voltage and population.v.grad_fn is grad_fn
    observed = torch.autograd.grad(prefix_loss, (native, previous_voltage))
    _close(observed, wanted)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("context", ("no_grad", "inference", "eval"))
def test_immediate_native_inference_suffix_uses_transferred_sources(network, context):
    subject, functional, packet, _ = _make(network, evaluation=context == "eval")
    target = _network() if network else _population()
    if context == "eval":
        target.eval()
        target.initialize(DT)
    guard = torch.inference_mode if context == "inference" else torch.no_grad
    with guard():
        expected, _ = dn.func.run(subject, packet, steps=3)
        dn.func.commit_continuation_(subject, target, packet)
        if network:
            target.run(3 * DT)
        else:
            target.run(tstop=3 * DT, dt=DT)
        torch.testing.assert_close(
            _native_voltage(target, network), _voltage(expected, network)
        )
        _close(functional.extract(target).state, expected)


@pytest.mark.parametrize("network", (False, True))
def test_inference_tensor_packet_is_owned_as_ordinary_native_runtime(network):
    subject, functional, packet, _ = _make(network)
    with torch.inference_mode():
        _, auxiliary = dn.func.run(subject, packet, steps=1)
        produced = auxiliary["continuation"]
        memo = {}

        def inference_copy(value):
            if not isinstance(value, torch.Tensor):
                return value
            if id(value) not in memo:
                memo[id(value)] = value.detach().clone()
            return memo[id(value)]

        state, parameters, constants = pytree.tree_map(
            inference_copy, (produced.state, produced.parameters, produced.constants)
        )
        packet = dn.func.Continuation(state, parameters, constants)
    assert all(
        torch.is_inference(value)
        for value in pytree.tree_leaves(packet.state)
        if isinstance(value, torch.Tensor)
    )
    target = _network() if network else _population()
    with torch.no_grad():
        ordinary = dn.func.Continuation(
            *_snapshot((packet.state, packet.parameters, packet.constants))
        )
        expected, _ = dn.func.run(subject, ordinary, steps=2)
        dn.func.commit_continuation_(subject, target, packet)
        committed = functional.extract(target).state
        assert all(
            not torch.is_inference(value) and not value.requires_grad
            for value in pytree.tree_leaves(committed)
            if isinstance(value, torch.Tensor)
        )
        if network:
            target.run(2 * DT)
        else:
            target.run(tstop=2 * DT, dt=DT)
        _close(functional.extract(target).state, expected)


@pytest.mark.parametrize("layout", ("expanded", "overlapping_views", "inference"))
def test_unsupported_native_source_storage_rejects_before_publication(layout):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    packet = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    target = _population()
    # Point Population geometry is a writable raw constant with the authored
    # [N, C] shape, unlike scalar GLOBAL mechanism parameters.
    if layout == "expanded":
        target._buffers["diam"] = target.diam[:1, :].expand(2, 1)
    elif layout == "overlapping_views":
        storage = torch.tensor([500.0, 500.0, 100.0], dtype=torch.float64)
        target._buffers["diam"] = storage[:2].reshape(2, 1)
        target._buffers["dx"] = storage[1:].reshape(2, 1)
    else:
        with torch.inference_mode():
            geometry = target.diam.detach().clone()
        target._buffers["diam"] = geometry
    saved = _capture(target)
    with pytest.raises(
        (dn.func.FunctionalizationError, RuntimeError),
        match="overlap|storage|[Ii]nference",
    ):
        dn.func.commit_continuation_(functional, target, packet)
    _unchanged(target, saved)


@pytest.mark.parametrize("network", (False, True))
def test_unchanged_source_values_keep_parameter_versions_and_existing_graphs(network):
    source = _network() if network else _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    _, auxiliary = dn.func.run(functional, initial, steps=2)
    target = _network() if network else _population()
    parameters = tuple(target.parameters())
    versions = tuple(value._version for value in parameters)
    native = _native_parameter(target, network)
    old_loss = native.square().sum()

    dn.func.commit_continuation_(functional, target, auxiliary["continuation"])
    assert tuple(value._version for value in parameters) == versions
    torch.testing.assert_close(torch.autograd.grad(old_loss, native)[0], 2 * native)


@pytest.mark.parametrize("kind", ("cable", "multi"))
def test_cable_and_mixed_population_handoffs_continue_public_native_runs(kind):
    if kind == "cable":
        from tests.func.test_native_cable import DT as dt
        from tests.func.test_native_cable import _model as make_model

        name = "integrator.mech.mechanisms.pas.g_param"
    else:
        from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
        from tests.func.test_multi_population import DT as dt
        from tests.func.test_multi_population import HH_GNABAR as name
        from tests.func.test_multi_population import _mixed_multi as make_model

        if not DENDRA_SOLVERS_AVAILABLE:
            pytest.skip("mixed MultiPopulation requires dendra-solvers")
    source, target = make_model(), make_model()
    functional, tensors = dn.func.make_functional(source, dt=dt)
    parameters = tensors.independent_parameters(trainable="*")
    with torch.no_grad():
        parameters[name].mul_(1.1)
    initial = dn.func.Continuation(tensors.state, parameters, tensors.constants)
    _, auxiliary = dn.func.run(functional, initial, steps=2)
    packet = auxiliary["continuation"]
    expected, _ = dn.func.run(functional, packet, steps=3)
    source_identities = dict(target.named_parameters(remove_duplicate=False))

    dn.func.commit_continuation_(functional, target, packet)
    _close(functional.extract(target).state, packet.state)
    target.run(tstop=3 * dt, dt=dt)
    _close(functional.extract(target).state, expected)
    for path, value in target.named_parameters(remove_duplicate=False):
        assert value is source_identities[path]


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_handoff_matches_float32_values_and_native_suffix_gradients():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        functional, tensors = dn.func.make_functional(_population(), dt=DT)
        parameters = tensors.independent_parameters(trainable="*g_param")
        with torch.no_grad():
            parameters[G].mul_(1.4)
        initial = dn.func.Continuation(tensors.state, parameters, tensors.constants)
        _, auxiliary = dn.func.run(functional, initial, steps=2)
        packet = auxiliary["continuation"]
        detached = dn.func.detach_continuation(functional, packet)
        expected, _ = dn.func.run(functional, detached, steps=3)
        wanted = torch.autograd.grad(
            expected["integrator"]["v"].square().sum(), parameters[G]
        )[0]
        target = _population()
        native = target.mech.leak.g_param

        dn.func.commit_continuation_(functional, target, packet)
        assert target.mech.leak.g_param is native
        target.run(tstop=3 * DT, dt=DT)
        torch.testing.assert_close(
            target.v, expected["integrator"]["v"], rtol=3e-5, atol=3e-5
        )
        actual = torch.autograd.grad(target.v.square().sum(), native)[0]
        torch.testing.assert_close(actual, wanted, rtol=3e-4, atol=3e-6)
        assert actual.device.type == "mps" and actual.abs().sum() > 0


def test_late_eval_netstim_publication_failure_restores_donated_clock_and_rng(
    monkeypatch,
):
    from dendra.func import _native_handoff as handoff

    source, _, _ = _scheduled_network()
    source.eval()
    source.initialize(NETWORK_DT)
    functional, tensors = dn.func.make_functional(source, dt=NETWORK_DT)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    with torch.no_grad():
        _, auxiliary = dn.func.run(functional, initial, steps=2)
    target, _, _ = _scheduled_network()
    target.eval()
    target.initialize(NETWORK_DT)
    previous = target.netstim.next_stoch_time
    generator = target.netstim._rng
    generator_state = generator.get_state().clone()
    saved = _capture(target)
    publish = handoff._publish_runtime
    calls = 0

    def publish_then_fail(*args, **kwargs):
        nonlocal calls
        assert args[0] is target
        calls += 1
        publish(*args, **kwargs)
        raise RuntimeError("injected NetStim publication failure")

    monkeypatch.setattr(handoff, "_publish_runtime", publish_then_fail)
    with pytest.raises(RuntimeError, match="injected NetStim publication failure"):
        dn.func.commit_continuation_(functional, target, auxiliary["continuation"])
    assert calls == 1
    assert target.netstim.next_stoch_time is previous
    assert target.netstim._rng is generator
    torch.testing.assert_close(generator.get_state(), generator_state, rtol=0, atol=0)
    _unchanged(target, saved)


class _ComplexRealRelease(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(complex(1.0, float("nan")), dtype=torch.complex128)
        )

    def forward(self, voltage):
        # The nonfinite imaginary component is deliberately unused. Changing
        # the finite real component still changes a finite native current.
        return torch.sigmoid(self.scale.real * voltage / 64.0)


def test_handoff_copies_changed_complex_component_despite_other_component_nan():
    from tests.test_network_detach import DT as NATIVE_DT
    from tests.test_network_detach import _continuous

    def case():
        model, connection = _continuous()
        # The native connection constructor casts a supplied transform to the
        # simulation dtype. Install the valid complex raw source after that
        # setup, before either model is lowered or advanced.
        connection.transform = _ComplexRealRelease()
        return model, connection

    source, source_connection = case()
    target, target_connection = case()
    functional, tensors = dn.func.make_functional(source, dt=NATIVE_DT)
    name = next(iter(source.continuous_synapses))
    scale = tensors.parameters["continuous_synapses"][name]["transform_parameters"][
        "scale"
    ]
    native_scale = target_connection.transform.scale
    assert source_connection.transform.scale.real.item() == 1.0
    assert native_scale.real.item() == 1.0
    assert torch.isnan(native_scale.imag)

    with torch.no_grad():
        scale.real.fill_(2.0)
        initial = dn.func.Continuation(
            tensors.state, tensors.parameters, tensors.constants
        )
        _, prefix = dn.func.run(functional, initial, steps=2)
        packet = prefix["continuation"]
        expected, _ = dn.func.run(functional, packet, steps=3)

        assert dn.func.commit_continuation_(functional, target, packet) is target
        assert target_connection.transform.scale is native_scale
        assert native_scale.real.item() == 2.0
        assert torch.isnan(native_scale.imag)
        # Check both component bits, including the preserved NaN payload;
        # equal_nan=True on an entire complex element can hide this regression.
        assert torch.equal(
            torch.view_as_real(native_scale).view(torch.int64),
            torch.view_as_real(scale).view(torch.int64),
        )
        target.run(3 * NATIVE_DT)
        _close(functional.extract(target).state, expected)

    assert source_connection.transform.scale.real.item() == 1.0


def test_unrepresentable_effective_connection_override_rejects_before_publication():
    source, _ = _parameter_network()
    functional, tensors = dn.func.make_functional(source, dt=CONNECTION_DT)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    _, auxiliary = dn.func.run(functional, initial, steps=2)
    packet = auxiliary["continuation"]
    frozen = _snapshot((packet.state, packet.parameters, packet.constants))
    constants = _plain(packet.constants)
    name = next(iter(constants["synapses"]))
    connection = constants["synapses"][name]
    assert bool(connection["use_offset.weights"])
    connection["stored.weights"] = connection["stored.weights"] + 0.125
    invalid = dn.func.Continuation(packet.state, packet.parameters, constants)
    target, _ = _parameter_network()
    saved = _capture(target)

    with pytest.raises(
        dn.func.FunctionalizationError, match="retained|override|represent"
    ):
        dn.func.commit_continuation_(functional, target, invalid)

    _unchanged(target, saved)
    _close((packet.state, packet.parameters, packet.constants), frozen)


class _SignbitRelease(torch.nn.Module):
    def __init__(self, nonfinite):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(float("nan") if nonfinite else 0.0)
        )

    def forward(self, voltage):
        gain = torch.where(torch.signbit(self.scale), 1.0, 0.5)
        return gain * torch.sigmoid(voltage / 64.0)


@pytest.mark.parametrize("nonfinite", (False, True), ids=("zero", "nan"))
def test_handoff_preserves_sign_bits_used_by_a_custom_transform(nonfinite):
    from tests.test_network_detach import DT as NATIVE_DT
    from tests.test_network_detach import _continuous

    def case():
        model, connection = _continuous()
        connection.transform = _SignbitRelease(nonfinite)
        return model, connection

    source, _ = case()
    target, connection = case()
    functional, tensors = dn.func.make_functional(source, dt=NATIVE_DT)
    name = next(iter(source.continuous_synapses))
    scale = tensors.parameters["continuous_synapses"][name]["transform_parameters"][
        "scale"
    ]
    native = connection.transform.scale
    assert not torch.signbit(native)
    with torch.no_grad():
        scale.fill_(-float("nan") if nonfinite else -0.0)
        initial = dn.func.Continuation(
            tensors.state, tensors.parameters, tensors.constants
        )
        _, prefix = dn.func.run(functional, initial, steps=2)
        packet = prefix["continuation"]
        expected, _ = dn.func.run(functional, packet, steps=3)
        dn.func.commit_continuation_(functional, target, packet)
        assert connection.transform.scale is native and torch.signbit(native)
        target.run(3 * NATIVE_DT)
        _close(functional.extract(target).state, expected)
