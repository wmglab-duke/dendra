"""Runtime replay uses the same packet capacity for cached snapshot models."""

import pytest
import torch

import dendra as dn
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_poisson_invocation_guards import _case, _RepeatedInitializer
from tests.func.test_poisson_stimulation import DT, _waveform
from tests.test_native_runtime_draw_tapes import (
    _at,
    _functional_sources,
    _loss,
    _run,
    _state_loss,
)
from tests.test_native_runtime_draws import _changed_draws
from tests.test_native_runtime_draws import _environment as _environment

pytestmark = pytest.mark.cpu


def _model(network, *, cached):
    cells, pulses = {}, []
    for name in ("left", "right") if network else ("cell",):
        cell, pulse = _case(_RepeatedInitializer)
        cell.train()
        cells[name] = cell
        pulses.append(pulse)
    model = dn.Network(cells) if network else cells["cell"]
    if network:
        for pulse in pulses:
            pulse.randomize_every_call = False
        model.train()
        model.initialize(DT)
        for pulse in pulses:
            pulse.randomize_every_call = True
    for cell in cells.values():
        if cached:
            cell.cache("_steady_state")
        cell.detach()
    return model, tuple(pulses)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("method", ("step", "longrun_checkpointed"))
@pytest.mark.parametrize("cached", (False, True), ids=("reserved", "runtime"))
def test_mechanism_packets_match_runtime_or_reserved_initialization_capacity(
    network, method, cached
):
    source, source_pulses = _model(network, cached=cached)
    model, pulses = _model(network, cached=cached)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    plans = functional.populations.values() if network else (functional,)
    for plan in plans:
        if cached:
            assert plan._initialization is None
            assert not any(plan._poisson_plan._initialization_counts.values())
        else:
            assert plan._initialization is not None
            assert set(plan._poisson_plan._initialization_counts.values()) == {3}
        assert set(plan._poisson_plan._capacities.values()) == {1 if cached else 3}
    steps = 1 if method == "step" else 3
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=steps))
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=steps,
        draws=packet,
    )
    if method == "step":
        model.step(**({} if network else {"dt": DT}), draws=_at(packet, 0))
    else:
        _run(model, method, DT, steps=steps, chunklength=2, draws=packet)
    _close(functional.extract(model).state, expected)
    saved = _snapshot(model.state_dict_for_checkpoint())
    targets = tuple(pulse.waveform.amp for pulse in pulses)
    references = _functional_sources(
        functional, tensors, tuple(pulse.waveform.amp for pulse in source_pulses)
    )
    wanted = torch.autograd.grad(_state_loss(expected, network), references)
    actual = torch.autograd.grad(_loss(model), targets)
    _close(actual, wanted)
    assert all(value.abs().sum() > 0 for value in wanted)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)


def test_mechanism_packet_rejects_arbitrary_reserved_row_counts_before_live_mutation():
    model, (pulse,) = _model(False, cached=False)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    packet = _snapshot(functional.sample_runtime_draws(tensors.state))
    (key,) = packet.draws["stimulation"]
    value = packet.draws["stimulation"][key]
    assert value.shape == (3, pulse.n)
    packet.draws["stimulation"][key] = torch.cat((value, value[:1]))
    saved = _snapshot(model.state_dict_for_checkpoint())
    rng = pulse.generator.get_state().clone()
    with pytest.raises(
        ValueError, match="Poisson stimulation draw changed Tensor schema"
    ):
        model.step(DT, draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _close(pulse.generator.get_state(), rng)


@pytest.mark.parametrize("invalid", (False, True), ids=("valid", "uniform"))
def test_cached_runtime_packet_does_not_probe_initial_values(invalid, monkeypatch):
    model, (pulse,) = _model(False, cached=True)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    packet = _snapshot(functional.sample_runtime_draws(tensors.state))
    expected, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state, draws=packet
    )
    saved = _snapshot(model.state_dict_for_checkpoint())
    rng = pulse.generator.get_state().clone()
    if invalid:
        (key,) = packet.draws["stimulation"]
        packet.draws["stimulation"][key].fill_(torch.nan)

    def forbidden(self, v, values):
        raise AssertionError("Runtime replay invoked a fresh-initialization hook")

    with monkeypatch.context() as guard:
        guard.setattr(_RepeatedInitializer, "initial_values", forbidden)
        if invalid:
            with pytest.raises(ValueError, match="uniforms must be finite"):
                model.step(DT, draws=packet)
        else:
            model.step(DT, draws=packet)
    if invalid:
        _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
        _close(pulse.generator.get_state(), rng)
    else:
        _close(functional.extract(model).state, expected)


def test_reserved_packet_checks_later_uniforms_before_probing_initial_values(
    monkeypatch,
):
    first, second = _waveform(seed=23), _waveform(seed=29)
    model, _ = _case(_RepeatedInitializer, waveform=first + second)
    model.train()
    model.detach()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    packet = _snapshot(functional.sample_runtime_draws(tensors.state))
    keys = tuple(packet.draws["stimulation"])
    assert len(keys) == 2
    assert packet.draws["stimulation"][keys[0]].shape == (3, first.n)
    packet.draws["stimulation"][keys[1]].fill_(torch.nan)
    saved = _snapshot(model.state_dict_for_checkpoint())
    rng = (first.generator.get_state().clone(), second.generator.get_state().clone())

    def forbidden(self, v, values):
        raise AssertionError("Malformed runtime draws invoked an initialization hook")

    with monkeypatch.context() as guard:
        guard.setattr(_RepeatedInitializer, "initial_values", forbidden)
        with pytest.raises(ValueError, match="uniforms must be finite"):
            model.step(DT, draws=packet)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _close(first.generator.get_state(), rng[0])
    _close(second.generator.get_state(), rng[1])
