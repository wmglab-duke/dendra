"""MaterialProcess-owned waveform recipes span native material phases."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import MaterialProcess
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_poisson_stimulation import DT, _waveform
from tests.test_native_runtime_draw_tapes import _at, _functional_sources, _run
from tests.test_native_runtime_draws import (
    _base_draws,
    _changed_draws,
)
from tests.test_native_runtime_draws import _environment as _environment
from tests.test_native_runtime_draws import (
    _forbid_owned_sampling,
)

pytestmark = pytest.mark.cpu


class _InjectedTransfer(MaterialProcess):
    MaterialProcess.GLOBAL(rate=0.3)
    material_fields = (("pool", "amount"), ("pool", "reserve"))

    def prepare_materials(self, fields, dt):
        return (dt * self._resolve_quantity("rate", fields[("pool", "amount")]),)

    def material_updates(self, fields, prepared):
        (coefficient,) = prepared
        amount, reserve = fields[("pool", "amount")], fields[("pool", "reserve")]
        drive = self.injection_values(amount, t=self.t)
        drive = drive + 0.25 * self.injection_values(amount, t=self.t)
        transfer = coefficient * (amount - reserve) * drive
        return self._mask_material_updates(
            fields,
            {
                ("pool", "amount"): amount - transfer,
                ("pool", "reserve"): reserve + transfer,
            },
        )


class _TransportInjection(_InjectedTransfer):
    MaterialProcess.PHASE("transport")


class _FinalInjection(_InjectedTransfer):
    MaterialProcess.PHASE("post_transport")


def _case(network, process):
    generator = torch.Generator().manual_seed(557)
    cells, pulses = {}, []
    for name in ("left", "right") if network else ("cell",):
        cell = dn.SingleCompartment(
            N=1, C=2, v_init=-47.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.material("pool", fields={"amount": 1.7, "reserve": 0.4})
        cell.insert(pas)
        cell.insert(process.rename("transfer"))
        cell.train()
        cell.initialize()
        owner = cell.mech.material_processes["transfer"]
        pulse = _waveform(generator=generator)
        # MaterialProcess inherits the public waveform registration helper.
        # Slice.inject offers waveforms only to ordinary mechanism inject hooks.
        assert owner.register_waveform_injection(
            pulse + pulse,
            index=(slice(None), 0),
            model_shape=cell.shape,
            model=cell,
        )
        cells[name] = cell
        pulses.append(pulse)
    model = dn.Network(cells) if network else cells["cell"]
    model.train()
    if network:
        model.initialize(DT)
    else:
        model.integrator._initialize(model, model.t.new_tensor(DT), force=True)
    for cell in cells.values():
        cell.detach()
    return model, tuple(pulses), generator


def _native_loss(model, network):
    cells = model.populations.values() if network else (model,)
    return sum(cell.mech.materials["pool"].amount.square().mean() for cell in cells)


def _state_loss(state, network):
    states = state["populations"].values() if network else (state,)
    return sum(local["materials"]["pool"]["amount"].square().mean() for local in states)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize(
    "process", (_InjectedTransfer, _TransportInjection, _FinalInjection)
)
@pytest.mark.parametrize("method", ("step", "longrun_checkpointed"))
def test_material_owned_poisson_replay_matches_runtime_phases_and_checkpoint_gradients(
    network, process, method, monkeypatch
):
    source, source_pulses, _ = _case(network, process)
    model, pulses, generator = _case(network, process)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    steps = 1 if method == "step" else 3
    packet = _changed_draws(functional.sample_runtime_draws(tensors.state, steps=steps))
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=steps,
        draws=packet,
    )
    before_source = _snapshot(source.state_dict_for_checkpoint())
    before_packet = _snapshot(packet)
    with monkeypatch.context() as guard:
        _forbid_owned_sampling(guard, model, generator)
        if method == "step":
            model.step(**({} if network else {"dt": DT}), draws=_at(packet, 0))
        else:
            _run(model, method, DT, steps=steps, chunklength=2, draws=packet)
    _close(functional.extract(model).state, expected)
    references = tuple(pulse.waveform.amp for pulse in source_pulses)
    targets = tuple(pulse.waveform.amp for pulse in pulses)
    saved = _snapshot(model.state_dict_for_checkpoint())
    schedules = tuple(pulse._spike_times for pulse in pulses)
    before_rng = generator.get_state().clone()
    wanted = torch.autograd.grad(
        _state_loss(expected, network),
        _functional_sources(functional, tensors, references),
    )
    actual = torch.autograd.grad(
        _native_loss(model, network),
        (*targets, *_base_draws(packet)),
        allow_unused=True,
    )
    _close(actual[: len(wanted)], wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in wanted)
    assert all(
        value is None or value.count_nonzero() == 0 for value in actual[len(wanted) :]
    )
    assert all(
        pulse._spike_times is schedule
        for pulse, schedule in zip(pulses, schedules, strict=True)
    )
    _close(generator.get_state(), before_rng)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
    _assert_packet_equal(source.state_dict_for_checkpoint(), before_source)
    _assert_packet_equal(packet, before_packet)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_material_owned_poisson_replay_supports_optional_purity_diagnostics(network):
    source, _source_pulses, source_generator = _case(network, _InjectedTransfer)
    model, _pulses, generator = _case(network, _InjectedTransfer)
    before_source = _snapshot(source.state_dict_for_checkpoint())
    source_rng = source_generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    with dn.ctx(CHECK_PURITY=True):
        functional, tensors = dn.func.make_functional(source, dt=DT)
        packet = functional.sample_runtime_draws(tensors.state, steps=1)
        expected, _ = dn.func.run(
            functional,
            dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
            steps=1,
            draws=packet,
        )
        model.step(**({} if network else {"dt": DT}), draws=_at(packet, 0))
        _close(functional.extract(model).state, expected)
    _assert_packet_equal(source.state_dict_for_checkpoint(), before_source)
    _close(source_generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)
    _close(generator.get_state(), next(iter(packet.rng["stimulation"].values()))[0])
