"""Handoff staging owns model/extra bindings without splitting live aliases."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.func import _native_handoff as handoff_module
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mod import pas
from dendra.models.stim.waveform._poisson import (
    borrow_pending_poisson_rng,
    defer_poisson_generator_restore,
    poisson_generator_state,
    reconcile_extra_poisson_rng,
    restore_pending_poisson_rng,
)
from dendra.models.stim.waveform.core import Waveform
from dendra.units import nA
from tests.func.test_continuation import _close, _plain, _snapshot
from tests.func.test_native_handoff import _capture, _unchanged
from tests.func.test_native_handoff_stochastic import _environment as _environment
from tests.func.test_poisson_stimulation import DT, _waveform
from tests.networks.test_native_network_poisson_replay import _case as _network_case

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _deterministic_stream(request):
    # The enclosing fixture restores ambient randomness after each test.
    request.getfixturevalue("_environment")
    torch.manual_seed(17)


class _SilentExtra(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self, idle):
        super().__init__()
        self.idle = idle

    def fn(self, times):
        return torch.zeros_like(times)


def _case(network=False, *, shared_parameter=False):
    generator = torch.default_generator
    intra = _waveform(generator=generator, amp=0.2 * nA)
    external = _waveform(generator=generator)
    if shared_parameter:
        external.waveform.amp = intra.waveform.amp
    populations = {}
    for name in ("left", "right") if network else ("cell",):
        population = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(pas, g=0.001, e=-65.0)
        population[:, 1].inject(intra + intra)
        population.train()
        population.initialize()
        populations[name] = population
    if network:
        model = dn.Network(populations)
        model.train()
        model.initialize(DT)
    else:
        model = populations["cell"]
        model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
    for population in populations.values():
        population.detach()
    fields = {
        name: torch.linspace(
            -1.25e6 if shared_parameter else -1.25,
            1.75e6 if shared_parameter else 1.75,
            population.v.numel(),
            dtype=population.dtype(),
            device=population.device(),
        )
        .reshape(population.shape)
        .requires_grad_()
        for name, population in populations.items()
    }
    extras = {
        name: (fields[name], external + external)
        for name in reversed(tuple(populations))
    }
    return SimpleNamespace(
        model=model,
        extra=extras if network else extras["cell"],
        roots=tuple(root for _field, root in extras.values()),
        fields=fields,
        intra=intra,
        external=external,
        generator=generator,
    )


def _plans(functional):
    return (
        tuple(functional.populations.values())
        if isinstance(functional, dn.func.FunctionalNetwork)
        else (functional,)
    )


def _bindings(functional):
    """Retain live plan readers whose private staging counterparts may differ."""
    return tuple(
        (
            plan,
            plan._population,
            plan._intra,
            plan._extra,
            plan._poisson_plan,
            plan._extra._waveforms,
            plan._extra._waveforms._sources,
            plan._poisson_plan._extra_sources,
            plan._poisson_plan.streams,
            tuple(
                (entry.source, type(entry.source), entry.source.generator)
                for entry in plan._poisson_plan.entries
            ),
            tuple(plan._extra._tensor_sources().items()),
        )
        for plan in _plans(functional)
    )


def _same_bindings(functional, saved):
    current = _bindings(functional)
    assert len(current) == len(saved)
    for actual, previous in zip(current, saved, strict=True):
        assert all(
            value is before
            for value, before in zip(actual[:-2], previous[:-2], strict=True)
        )
        assert len(actual[-2]) == len(previous[-2])
        assert all(
            all(value is before for value, before in zip(pair, old_pair, strict=True))
            for pair, old_pair in zip(actual[-2], previous[-2], strict=True)
        )
        assert tuple(key for key, _value in actual[-1]) == tuple(
            key for key, _value in previous[-1]
        )
        assert all(
            value is before
            for (_key, value), (_other, before) in zip(
                actual[-1], previous[-1], strict=True
            )
        )


def _external_capture(case):
    values = {"field." + name: value for name, value in case.fields.items()}
    for index, root in enumerate(case.roots):
        values.update(
            (f"{index}.parameter.{name}", value)
            for name, value in root.named_parameters(remove_duplicate=False)
        )
        values.update(
            (f"{index}.buffer.{name}", value)
            for name, value in root.named_buffers(remove_duplicate=False)
        )
    return {
        name: (value, value.detach().clone(), value._version)
        for name, value in values.items()
    }, case.generator.get_state().clone()


def _same_external(case, saved, *, writable=()):
    current, rng = _external_capture(case)
    previous, expected_rng = saved
    assert current.keys() == previous.keys()
    writable = {id(value) for value in writable}
    for name, (value, _copy, version) in current.items():
        original, expected, old_version = previous[name]
        assert value is original
        if id(value) not in writable:
            assert version == old_version
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    torch.testing.assert_close(rng, expected_rng, rtol=0, atol=0)


def _voltage(model):
    populations = (
        model.populations.values() if isinstance(model, dn.Network) else (model,)
    )
    return torch.cat([population.v.reshape(-1) for population in populations])


def _state_voltage(state, network):
    values = state["populations"].values() if network else (state,)
    return torch.cat([value["integrator"]["v"].reshape(-1) for value in values])


def _stimulation_rng(state, network):
    return state["stimulation_rng"] if network else state["stimulation"]["rng"]


def _advance(case, *, draws=None):
    options = {} if draws is None else {"draws": draws}
    return (
        case.model.step(extra=case.extra, **options)
        if isinstance(case.model, dn.Network)
        else case.model.step(DT, extra=case.extra, **options)
    )


def _source_keys(functional, parameters, original):
    result = []
    for name, plan in (
        functional.populations.items()
        if isinstance(functional, dn.func.FunctionalNetwork)
        else ((None, functional),)
    ):
        local = parameters if name is None else parameters["populations"][name]
        authored = dict(plan._population.named_parameters(remove_duplicate=False))
        authored.update(plan._intra._tensor_sources())
        authored.update(plan._extra._tensor_sources())
        result.extend(
            (local, key)
            for key, value in authored.items()
            if value is original and key in local
        )
    assert result
    first = result[0][0][result[0][1]]
    assert all(local[key] is first for local, key in result)
    return result


def _prefix(functional, tensors, *, replacement=None, original=None):
    parameters = tensors.independent_parameters(trainable="*amp")
    if replacement is not None:
        for local, key in _source_keys(functional, parameters, original):
            local[key] = replacement
    initial = dn.func.Continuation(tensors.state, parameters, tensors.constants)
    _, auxiliary = dn.func.run(functional, initial, steps=2)
    return dn.func.detach_continuation(functional, auxiliary["continuation"])


@contextmanager
def _no_initialization_or_sampling(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Continuation staging initialized or sampled the model")

    codes = {
        dn.Population.initialize.__code__,
        dn.Network.initialize.__code__,
        MechanismHandler.initialize.__code__,
    }
    previous = sys.getprofile()

    def profile(frame, event, argument):
        if event == "call" and frame.f_code in codes:
            forbidden()
        if previous is not None:
            previous(frame, event, argument)

    with monkeypatch.context() as guard:
        for name in ("rand", "randn", "rand_like", "randn_like"):
            guard.setattr(torch, name, forbidden)
        sys.setprofile(profile)
        try:
            yield
        finally:
            sys.setprofile(previous)


def _observe_live_bindings(monkeypatch, functional, saved):
    refresh = handoff_module._refresh_native

    def checked(*args, **kwargs):
        _same_bindings(functional, saved)
        return refresh(*args, **kwargs)

    monkeypatch.setattr(handoff_module, "_refresh_native", checked)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("target_kind", ("source", "compatible"))
def test_full_extra_plan_handoff_stages_shared_rng_aliases_and_resumes_in_native_order(
    network, target_kind, monkeypatch
):
    source = _case(network)
    functional, tensors = dn.func.make_functional(
        source.model, dt=DT, extra=source.extra
    )
    packet = _prefix(functional, tensors)
    target = source if target_kind == "source" else _case(network)
    # The functional plan's extra remains the caller's invocation configuration.
    target.extra, target.roots, target.fields, target.external = (
        source.extra,
        source.roots,
        source.fields,
        source.external,
    )
    saved_external, bindings = _external_capture(source), _bindings(functional)
    packet_saved = _snapshot((packet.state, packet.parameters, packet.constants))
    parameters = tuple(
        (name, value, value.requires_grad, value.grad)
        for name, value in target.model.named_parameters(remove_duplicate=False)
    )
    _observe_live_bindings(monkeypatch, functional, bindings)
    with _no_initialization_or_sampling(monkeypatch):
        assert (
            dn.func.commit_continuation_(functional, target.model, packet)
            is target.model
        )
    _same_bindings(functional, bindings)
    _same_external(source, saved_external)
    captured = dn.func.capture_continuation(functional, target.model)
    _close(
        _stimulation_rng(captured.state, network),
        _stimulation_rng(packet.state, network),
    )
    current = dict(target.model.named_parameters(remove_duplicate=False))
    assert all(
        current[name] is value
        and value.requires_grad is trainable
        and value.grad is gradient
        for name, value, trainable, gradient in parameters
    )
    _close(_voltage(target.model), _state_voltage(packet.state, network))
    # Extra is evaluated first. It must consume the deferred stream endpoint,
    # rather than the deliberately perturbed ambient default-generator state.
    torch.manual_seed(9931)
    expected, auxiliary = dn.func.run(functional, packet, steps=1)
    _advance(target)
    _close(_voltage(target.model), _state_voltage(expected, network))
    _close(
        poisson_generator_state(target.intra), poisson_generator_state(target.external)
    )
    saved = _snapshot(target.model.state_dict_for_checkpoint())
    extra_saved = tuple(_snapshot(root.state_dict()) for root in target.roots)
    continued = auxiliary["continuation"]
    draws = functional.sample_runtime_draws(continued.state)
    expected, _ = functional.prepare_and_step(
        continued.parameters, continued.constants, continued.state, draws=draws
    )
    _advance(target, draws=draws)
    wanted = _voltage(target.model).detach().clone()
    target.model.restore_dict_from_checkpoint(saved)
    for root, values in zip(target.roots, extra_saved, strict=True):
        root.load_state_dict(values)
    _advance(target)
    _close(wanted, _state_voltage(expected, network))
    _close(_voltage(target.model), wanted)
    torch.testing.assert_close(
        _plain((packet.state, packet.parameters, packet.constants)),
        packet_saved,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_unused_registered_extra_child_does_not_restore_pending_rng_before_consumption(
    network, monkeypatch
):
    case = _case(network)
    silent = _SilentExtra(case.external)
    if network:
        case.extra = {
            name: (field, silent) for name, (field, _root) in case.extra.items()
        }
        case.roots = tuple(silent for _name in case.extra)
    else:
        case.extra = (case.extra[0], silent)
        case.roots = (silent,)
    functional, tensors = dn.func.make_functional(case.model, dt=DT, extra=case.extra)
    packet = _prefix(functional, tensors)
    dn.func.commit_continuation_(functional, case.model, packet)
    expected, _ = dn.func.run(functional, packet, steps=1)
    torch.manual_seed(9931)
    ambient = torch.get_rng_state().clone()
    observed = []
    original = _SilentExtra.fn

    def observe(self, times):
        if self is silent:
            observed.append(torch.get_rng_state().clone())
        return original(self, times)

    with monkeypatch.context() as guard:
        guard.setattr(_SilentExtra, "fn", observe)
        _advance(case)
    assert len(observed) == (2 if network else 1)
    assert all(torch.equal(frame, ambient) for frame in observed)
    _close(_voltage(case.model), _state_voltage(expected, network))


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_equal_independent_pending_holders_are_consumed_once_across_unused_extra(
    network,
):
    case = _case(network)
    silent = _SilentExtra(case.external)
    if network:
        case.extra = {
            name: (field, silent) for name, (field, _root) in case.extra.items()
        }
    else:
        case.extra = (case.extra[0], silent)
    functional, tensors = dn.func.make_functional(case.model, dt=DT, extra=case.extra)
    packet = _prefix(functional, tensors)
    dn.func.commit_continuation_(functional, case.model, packet)
    frame = _stimulation_rng(packet.state, network)["stream_0"]
    defer_poisson_generator_restore((case.intra,), frame)
    defer_poisson_generator_restore((case.external,), frame)
    owned = case.intra._functional_poisson_rng
    caller = case.external._functional_poisson_rng
    assert owned is not caller
    _close(owned.state, caller.state)

    # The extra bank visits its registered child without consuming it. Its
    # independent holder must still observe consumption by the later intra bank.
    torch.manual_seed(9931)
    for _index in range(2):
        expected, auxiliary = dn.func.run(functional, packet, steps=1)
        _advance(case)
        assert case.intra._functional_poisson_rng is owned
        assert case.external._functional_poisson_rng is caller
        assert owned.state is None and caller.state is None
        _close(_voltage(case.model), _state_voltage(expected, network))
        expected_rng = _stimulation_rng(expected, network)["stream_0"]
        _close(poisson_generator_state(case.intra), expected_rng)
        _close(poisson_generator_state(case.external), expected_rng)
        captured = dn.func.capture_continuation(functional, case.model)
        _close(
            _stimulation_rng(captured.state, network),
            _stimulation_rng(expected, network),
        )
        packet = auxiliary["continuation"]


def test_pending_holder_union_is_transitive_across_overlapping_borrow_scopes():
    generator = torch.Generator().manual_seed(811)
    first, middle, last = tuple(_waveform(generator=generator) for _index in range(3))
    initial = generator.get_state().clone()
    for waveform in (first, middle, last):
        defer_poisson_generator_restore((waveform,), initial)
    holders = tuple(
        waveform._functional_poisson_rng for waveform in (first, middle, last)
    )
    assert len({id(holder) for holder in holders}) == 3
    torch.rand(2, generator=generator)
    ambient = generator.get_state().clone()

    with borrow_pending_poisson_rng(first, middle):
        pass
    # Reparenting the middle holder must retain the first scope's aliases.
    with borrow_pending_poisson_rng(last, middle):
        pass
    assert all(
        waveform._functional_poisson_rng is holder
        for waveform, holder in zip((first, middle, last), holders, strict=True)
    )
    _close(generator.get_state(), ambient)
    assert all(holder.state is not None for holder in holders)

    oracle = torch.Generator()
    oracle.set_state(initial)
    restore_pending_poisson_rng(last)
    _close(torch.rand(3, generator=generator), torch.rand(3, generator=oracle))
    assert all(holder.state is None for holder in holders)
    restore_pending_poisson_rng(first)
    restore_pending_poisson_rng(middle)
    _close(torch.rand(3, generator=generator), torch.rand(3, generator=oracle))


def test_multicontact_reconciliation_shares_every_holder_before_first_consumption():
    case = _case()
    second = _waveform(generator=case.generator)
    initial = case.generator.get_state().clone()
    waveforms = (case.intra, case.external, second)
    for waveform in waveforms:
        defer_poisson_generator_restore((waveform,), initial)
    holders = tuple(waveform._functional_poisson_rng for waveform in waveforms)
    assert len({id(holder) for holder in holders}) == 3
    extra = [case.extra, (case.extra[0], second + second)]
    torch.manual_seed(9931)
    ambient = case.generator.get_state().clone()
    reconcile_extra_poisson_rng(case.model, extra)
    _close(case.generator.get_state(), ambient)
    assert all(
        waveform._functional_poisson_rng is holder
        for waveform, holder in zip(waveforms, holders, strict=True)
    )
    assert all(holder.state is not None for holder in holders)

    oracle = torch.Generator()
    oracle.set_state(initial)
    restore_pending_poisson_rng(case.intra)
    _close(torch.rand(3, generator=case.generator), torch.rand(3, generator=oracle))
    assert all(holder.state is None for holder in holders)
    for waveform in waveforms[1:]:
        restore_pending_poisson_rng(waveform)
    _close(torch.rand(3, generator=case.generator), torch.rand(3, generator=oracle))


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_zero_step_run_reconciles_pending_holders_without_changing_bank_order(
    network,
):
    case = _case(network)
    second = _waveform(generator=case.generator)
    second_root = second + second
    if network:
        first, last = tuple(case.extra)
        extra = {
            first: case.extra[first],
            last: (case.extra[last][0], second_root),
        }
    else:
        extra = [case.extra, (case.extra[0], second_root)]
    waveforms = (case.intra, case.external, second)
    boundary = _snapshot(case.model.state_dict_for_checkpoint())
    external_boundary = tuple(
        (root, _snapshot(root.state_dict())) for root in (*case.roots, second_root)
    )
    initial = case.generator.get_state().clone()

    def execute():
        options = {"steps": 0, "extra": extra}
        if not network:
            options["dt"] = DT
        case.model.run(**options)

    # Empty native banks still invoke their waveforms. Retain their existing
    # cadence while consuming an equivalent deferred stream only once.
    defer_poisson_generator_restore(waveforms, initial)
    torch.manual_seed(1061)
    execute()
    endpoint = case.generator.get_state().clone()
    schedules = tuple(waveform._spike_times.clone() for waveform in waveforms)

    case.model.restore_dict_from_checkpoint(boundary)
    for root, values in external_boundary:
        root.load_state_dict(values)
    for waveform in waveforms:
        defer_poisson_generator_restore((waveform,), initial)
    holders = tuple(waveform._functional_poisson_rng for waveform in waveforms)
    assert len({id(holder) for holder in holders}) == len(waveforms)
    torch.manual_seed(9931)
    execute()
    _close(case.generator.get_state(), endpoint)
    for waveform, holder, schedule in zip(waveforms, holders, schedules, strict=True):
        assert waveform._functional_poisson_rng is holder
        assert holder.state is None
        _close(waveform._spike_times, schedule)
        _close(poisson_generator_state(waveform), endpoint)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("mode", ("run", "longrun", "longrun_checkpointed"))
def test_native_run_banks_reconcile_equal_pending_holders_before_intra_sampling(
    network, mode
):
    case = _case(network)
    functional, tensors = dn.func.make_functional(case.model, dt=DT, extra=case.extra)
    packet = _prefix(functional, tensors)
    dn.func.commit_continuation_(functional, case.model, packet)
    boundary = _snapshot(case.model.state_dict_for_checkpoint())
    external_boundary = tuple(_snapshot(root.state_dict()) for root in case.roots)
    frame = _stimulation_rng(packet.state, network)["stream_0"]

    def execute(steps):
        options = {"steps": steps, "extra": case.extra}
        if not network:
            options["dt"] = DT
        if mode != "run":
            options["chunklength"] = 2
        getattr(case.model, mode)(**options)

    # Native run methods prepare whole banks in their own order. A shared holder
    # supplies an oracle with exactly those invocation counts and chunk bounds.
    defer_poisson_generator_restore((case.intra, case.external), frame)
    torch.manual_seed(1061)
    expected = []
    for steps in (3, 1):
        execute(steps)
        expected.append(
            (_voltage(case.model).detach().clone(), case.generator.get_state().clone())
        )

    case.model.restore_dict_from_checkpoint(boundary)
    for root, values in zip(case.roots, external_boundary, strict=True):
        root.load_state_dict(values)
    defer_poisson_generator_restore((case.intra,), frame)
    defer_poisson_generator_restore((case.external,), frame)
    owned = case.intra._functional_poisson_rng
    caller = case.external._functional_poisson_rng
    assert owned is not caller
    torch.manual_seed(9931)
    for steps, (voltage, endpoint) in zip((3, 1), expected, strict=True):
        execute(steps)
        assert case.external._functional_poisson_rng is caller
        assert caller.state is None
        if mode != "longrun_checkpointed":
            assert case.intra._functional_poisson_rng is owned
            assert owned.state is None
        _close(_voltage(case.model), voltage)
        _close(case.generator.get_state(), endpoint)
        _close(poisson_generator_state(case.intra), endpoint)
        _close(poisson_generator_state(case.external), endpoint)
        captured = dn.func.capture_continuation(functional, case.model)
        _close(_stimulation_rng(captured.state, network)["stream_0"], endpoint)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_handoff_preserves_parameter_aliases_across_native_and_caller_owned_sources(
    network, monkeypatch
):
    case = _case(network, shared_parameter=True)
    functional, tensors = dn.func.make_functional(case.model, dt=DT, extra=case.extra)
    keys = _source_keys(functional, tensors.parameters, case.intra.waveform.amp)
    upstream = keys[0][0][keys[0][1]].detach().clone().requires_grad_()
    replacement = 1.3 * upstream
    packet = _prefix(
        functional, tensors, replacement=replacement, original=case.intra.waveform.amp
    )
    raw = case.intra.waveform.amp
    raw.grad = torch.full_like(raw, 0.2)
    old_grad, flag = raw.grad, raw.requires_grad
    saved_external, bindings = _external_capture(case), _bindings(functional)
    _observe_live_bindings(monkeypatch, functional, bindings)
    with _no_initialization_or_sampling(monkeypatch):
        dn.func.commit_continuation_(functional, case.model, packet)
    assert case.external.waveform.amp is case.intra.waveform.amp is raw
    assert raw.grad is old_grad and raw.requires_grad is flag
    _close(raw, replacement)
    _same_bindings(functional, bindings)
    _same_external(case, saved_external, writable=(raw,))
    expected, _ = dn.func.run(functional, packet, steps=1)
    _advance(case)
    _close(_voltage(case.model), _state_voltage(expected, network))
    actual, old_graph = torch.autograd.grad(
        _voltage(case.model).square().sum(), (raw, upstream), allow_unused=True
    )
    wanted = torch.autograd.grad(
        _state_voltage(expected, network).square().sum(), replacement
    )[0]
    _close(actual, wanted)
    assert actual.isfinite().all() and actual.abs().sum() > 0
    assert old_graph is None


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize(
    "invalid", ("external_override", "parameter_alias", "late_failure")
)
def test_staging_rejection_preserves_live_sources_external_bindings_and_old_graphs(
    network, invalid, monkeypatch
):
    case = _case(network, shared_parameter=invalid != "external_override")
    _advance(case)
    previous_loss = _voltage(case.model).square().sum()
    functional, tensors = dn.func.make_functional(case.model, dt=DT, extra=case.extra)
    packet = dn.func.Continuation(
        tensors.state,
        tensors.independent_parameters(trainable="*amp"),
        tensors.constants,
    )
    parameters = _plain(packet.parameters)
    rejected = []
    if invalid == "external_override":
        keys = _source_keys(functional, parameters, case.external.waveform.amp)
        replacement = keys[0][0][keys[0][1]] * 1.2
        for local, key in keys:
            local[key] = replacement
    elif invalid == "parameter_alias":
        keys = _source_keys(functional, parameters, case.intra.waveform.amp)
        local, key = next((local, key) for local, key in keys if "extra" in key)
        original = local[key]
        local[key] = original * 1.2
        assert any(
            mapping[other] is original
            for mapping, other in keys
            if mapping is not local or other != key
        )
    else:

        def reject(*_args, **_kwargs):
            rejected.append(True)
            raise dn.func.FunctionalizationError("staged preparation rejected")

        monkeypatch.setattr(handoff_module, "_compare_preparation", reject)
    packet = dn.func.Continuation(packet.state, parameters, packet.constants)
    saved, external, bindings = (
        _capture(case.model),
        _external_capture(case),
        _bindings(functional),
    )
    raw = case.intra.waveform.amp
    old_grad = torch.full_like(raw, 0.4)
    raw.grad = old_grad
    _observe_live_bindings(monkeypatch, functional, bindings)
    with _no_initialization_or_sampling(monkeypatch):
        with pytest.raises((dn.func.FunctionalizationError, ValueError)):
            dn.func.commit_continuation_(functional, case.model, packet)
    if invalid == "late_failure":
        assert rejected == [True]
    _unchanged(case.model, saved)
    _same_external(case, external)
    _same_bindings(functional, bindings)
    assert raw.grad is old_grad
    gradient = torch.autograd.grad(previous_loss, raw)[0]
    assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_incompatible_target_generator_aliases_reject_before_joint_staging(monkeypatch):
    source, target = _case(True), _case(True)
    functional, tensors = dn.func.make_functional_network(
        source.model, extra=source.extra
    )
    target.extra, target.roots, target.fields, target.external = (
        source.extra,
        source.roots,
        source.fields,
        source.external,
    )
    target.intra.generator = torch.Generator().manual_seed(809)
    _advance(target)
    previous_loss = _voltage(target.model).square().sum()
    packet = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    saved, external, bindings = (
        _capture(target.model),
        _external_capture(source),
        _bindings(functional),
    )
    with _no_initialization_or_sampling(monkeypatch):
        with pytest.raises(
            (dn.func.FunctionalizationError, ValueError),
            match="alias|sharing|generator",
        ):
            dn.func.commit_continuation_(functional, target.model, packet)
    _unchanged(target.model, saved)
    _same_external(source, external)
    _same_bindings(functional, bindings)
    gradient = torch.autograd.grad(previous_loss, target.intra.waveform.amp)[0]
    assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_network_handoff_preserves_physical_waveform_aliases_across_intra_and_extra():
    case = _network_case(cross_banks=True, global_stream=True)
    functional, tensors = dn.func.make_functional_network(case.model, extra=case.extra)
    packet = _prefix(functional, tensors)
    left = case.model.left.injections[0][0].waveforms[0]
    right = case.model.right.injections[0][0].waveforms[0]
    assert case.extra["right"][1].waveforms[0] is left
    assert case.extra["left"][1].waveforms[0] is right
    dn.func.commit_continuation_(functional, case.model, packet)
    assert case.model.left.injections[0][0].waveforms[0] is left
    assert case.model.right.injections[0][0].waveforms[0] is right
    assert case.extra["right"][1].waveforms[0] is left
    assert case.extra["left"][1].waveforms[0] is right
    expected, _ = dn.func.run(functional, packet, steps=1)
    torch.manual_seed(9931)
    case.model.step(extra=case.extra)
    _close(_voltage(case.model), _state_voltage(expected, True))


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_handoff_staging_preserves_extra_aliases_and_float32_suffix_gradients():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        case = _case(True, shared_parameter=True)
        functional, tensors = dn.func.make_functional_network(
            case.model, extra=case.extra
        )
        packet = _prefix(functional, tensors)
        raw = case.intra.waveform.amp
        dn.func.commit_continuation_(functional, case.model, packet)
        expected, _ = dn.func.run(functional, packet, steps=1)
        _advance(case)
        torch.testing.assert_close(
            _voltage(case.model), _state_voltage(expected, True), rtol=2e-5, atol=2e-6
        )
        assert case.external.waveform.amp is raw
        actual = torch.autograd.grad(_voltage(case.model).square().sum(), raw)[0]
        keys = _source_keys(functional, packet.parameters, raw)
        wanted = torch.autograd.grad(
            _state_voltage(expected, True).square().sum(), keys[0][0][keys[0][1]]
        )[0]
        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-6)
        assert actual.isfinite().all() and actual.abs().sum() > 0
