"""Explicit native steps share pure observations without resetting their phase."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_imperative_checkpoint_callbacks import DT, _gnabar, _model
from tests.func.test_native_network_checkpoint_callbacks import _NetworkEnergy
from tests.func.test_native_network_checkpoint_callbacks import _plan as _network_plan
from tests.func.test_native_network_functional_callbacks import _driven
from tests.func.test_native_population_functional_callbacks import (
    _plan as _population_plan,
)
from tests.func.test_native_population_functional_callbacks import (
    _PortableEnergyReducer,
)
from tests.func.test_network_population_callbacks import _network
from tests.func.test_stimulation_rng_sharing import DT as STOCHASTIC_DT
from tests.func.test_stimulation_rng_sharing import _network as _stochastic_network
from tests.func.test_stimulation_rng_sharing import (
    _population as _stochastic_population,
)
from tests.test_recorder_h5_integration import local_hdf5_writer as local_hdf5_writer

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=True),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


def _case(network):
    return _driven() if network else (_model(require_grad=True, stimulated=True), None)


def _population(model, network):
    return model.left if network else model


def _plan(model, network, *, field=None):
    return _network_plan(model, field=field) if network else _population_plan(model)


def _step(
    model,
    network,
    *,
    callbacks=None,
    state=None,
    native=None,
    loop_hooks=False,
    **inputs,
):
    options = dict(callbacks=native, loop_hooks=loop_hooks, **inputs)
    if callbacks is not None:
        options["functional_callbacks"] = callbacks
    if state is not None:
        options["functional_callback_state"] = state
    return model.step(**options) if network else model.step(DT, **options)


def _run(model, network, method, callbacks, *, state=None, steps=6):
    options = dict(tstop=steps * DT, functional_callbacks=callbacks)
    if not network:
        options["dt"] = DT
    if state is not None:
        options["functional_callback_state"] = state
    if method != "run":
        options["chunklength"] = 2
    if method == "longrun_checkpointed":
        options.update(safe_checkpoint=True, restore_state_after_backward=True)
    result = getattr(model, method)(**options)
    if method == "longrun_checkpointed":
        assert result[0] is None
        return result[1]
    return result


def _parameter(model, network):
    return model.left.mech.pas.g_param if network else _gnabar(model)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_native_step_default_return_remains_chainable(network):
    model, _ = _case(network)
    first = _step(model, network)
    assert first is model
    assert _step(first, network) is model
    assert model.t.item() == pytest.approx(2 * DT)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("loop_hooks", (False, True))
def test_step_pure_initialization_and_updates_keep_native_hook_order_and_owned_views(
    network, loop_hooks
):
    class Hooks(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.observed = self.time = None

        def pre_loop_hook(self, model):
            self.calls.append("pre_loop")
            population = _population(model, network)
            assert population.integrator.dt == DT
            population.v.add_(0.5)

        def pre_step_hook(self, model):
            self.calls.append("pre_step")
            _population(model, network).v.add_(1)

        def post_step_hook(self, model):
            self.calls.append("post_step")
            population = _population(model, network)
            self.observed, self.time = population.v.clone(), population.t.clone()
            population.v.add_(5)

        def post_loop_hook(self, model):
            self.calls.append("post_loop")
            _population(model, network).v.add_(7)

    model, _ = _case(network)
    model.eval()
    initial = _population(model, network).v.detach().clone()
    callbacks, native = _plan(model, network), Hooks()
    result = _step(
        model, network, callbacks=callbacks, native=[native], loop_hooks=loop_hooks
    )
    assert type(result) is dn.func.FunctionalCallbackResults
    assert native.calls == (
        ["pre_loop", "pre_step", "post_step", "post_loop"]
        if loop_hooks
        else ["pre_step", "post_step"]
    )
    _close(result["views"]["initial"], initial + (0.5 if loop_hooks else 0))
    _close(result["views"]["latest"], native.observed[..., ::2])
    _close(result["views"]["trace"]["v"][1], native.observed)
    _close(result["views"]["trace"]["t"][1], native.time)
    assert result["energy"]["steps"].item() == 1
    frozen = _snapshot(result.outputs)
    with torch.no_grad():
        _population(model, network).v.fill_(-999)
        model.t.fill_(999)
    _close(result.outputs, frozen)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_repeated_steps_share_cadence_reducers_scoped_lfp_and_strict_native_gradients(
    network,
):
    model, amplitude = _case(network)
    field = torch.ones_like(model.left.v, requires_grad=True) if network else None
    callbacks = _plan(model, network, field=field)
    parts, carry = [], None
    for _ in range(6):
        result = _step(model, network, callbacks=callbacks, state=carry)
        parts.append(result)
        carry = result.state
    reference, reference_amplitude = _case(network)
    reference_field = (
        torch.ones_like(reference.left.v, requires_grad=True) if network else None
    )
    expected = _run(
        reference, network, "run", _plan(reference, network, field=reference_field)
    )
    stream = "qualified" if network else "trace"
    for name in expected[stream]:
        _close(
            torch.cat(tuple(part[stream][name] for part in parts)),
            expected[stream][name],
        )
    _close(
        torch.cat(tuple(part["views"]["trace"]["v"] for part in parts)),
        expected["views"]["trace"]["v"],
    )
    _close(parts[-1]["energy"], expected["energy"])
    assert parts[-1]["energy"]["steps"].item() == 6
    objective = parts[-1]["energy"]["energy"].sum()
    wanted_objective = expected["energy"]["energy"].sum()
    targets, reference_targets = (
        (_parameter(model, network),),
        (_parameter(reference, network),),
    )
    if network:
        lfp = torch.cat(tuple(part["lfp"] for part in parts))
        _close(lfp, expected["lfp"])
        objective = objective + (lfp * 1e6).square().mean()
        wanted_objective = wanted_objective + (expected["lfp"] * 1e6).square().mean()
        targets += (field, amplitude)
        reference_targets += (reference_field, reference_amplitude)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    observed = torch.autograd.grad(objective, targets)
    wanted = torch.autograd.grad(wanted_objective, reference_targets)
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_step_run_checkpoint_longrun_and_portable_step_resume_keep_one_phase(
    network, tmp_path
):
    model, _ = _case(network)
    callbacks = _plan(model, network)
    first = _step(model, network, callbacks=callbacks)
    second = _run(model, network, "run", callbacks, state=first.state, steps=2)
    third = _run(
        model, network, "longrun_checkpointed", callbacks, state=second.state, steps=2
    )
    saved_model = _snapshot(model.state_dict_for_checkpoint())
    saved_carry = _disk(callbacks.export_state(third.state), tmp_path / "callbacks.pt")
    last = _run(model, network, "longrun", callbacks, state=third.state, steps=1)
    reference, _ = _case(network)
    expected = _run(reference, network, "run", _plan(reference, network))
    stream = "qualified" if network else "trace"
    for field in expected[stream]:
        _close(
            torch.cat(
                tuple(part[stream][field] for part in (first, second, third, last))
            ),
            expected[stream][field],
        )
    _close(last["energy"], expected["energy"])
    receiver, _ = _case(network)
    receiver.restore_dict_from_checkpoint(saved_model)
    receiving = _plan(receiver, network)
    restored = receiving.restore_state(saved_carry)
    suffix = _step(
        receiver, network, callbacks=receiving, state=restored, loop_hooks=True
    )
    _close(suffix.outputs, last.outputs)
    early_reference, _ = _case(network)
    early = _run(
        early_reference, network, "run", _plan(early_reference, network), steps=1
    )
    frozen = _snapshot(model.state_dict_for_checkpoint())
    observed = torch.autograd.grad(
        first["energy"]["energy"].sum(), _parameter(model, network)
    )[0]
    wanted = torch.autograd.grad(
        early["energy"]["energy"].sum(), _parameter(early_reference, network)
    )[0]
    _close(observed, wanted)
    assert torch.isfinite(observed).all() and observed.abs().sum() > 0
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_explicit_step_preserves_pending_fractional_duration(network):
    model, _ = _case(network)
    callbacks = _plan(model, network)
    _run(model, network, "run", callbacks, steps=0.4)
    pending = model._duration_remainder.detach().clone()
    first = _step(model, network, callbacks=callbacks)
    _close(model._duration_remainder, pending)
    assert model.t.item() == pytest.approx(DT)
    last = _run(model, network, "run", callbacks, state=first.state, steps=0.6)
    assert model.t.item() == pytest.approx(2 * DT)
    assert model._duration_remainder.item() == 0
    assert last["energy"]["steps"].item() == 2
    assert last["views"]["trace"]["v"].shape[0] == 1


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_step_observation_preserves_extra_evaluation_relative_to_native_pre_step(
    network, monkeypatch
):
    model, _ = _case(network)
    model.eval()
    population = _population(model, network)
    waveform = dn.mono_rect(amp=0.01, delay=0, pw=0.1)
    field = torch.linspace(
        -0.1, 0.1, population.v.numel(), dtype=population.v.dtype
    ).reshape(population.shape)
    seen, forward = [], waveform.forward

    def observe(times):
        seen.append((float(waveform.amp), times.detach().clone()))
        return forward(times)

    monkeypatch.setattr(waveform, "forward", observe)

    class ChangeAmplitude(dn.callbacks.Callback):
        def pre_step_hook(self, model):
            waveform.amp.mul_(2)

    extra = {"left": (field, waveform)} if network else (field, waveform)
    result = _step(
        model,
        network,
        callbacks=_plan(model, network),
        native=[ChangeAmplitude()],
        extra=extra,
    )
    assert len(seen) == 1
    assert seen[0][0] == pytest.approx(0.01 if network else 0.02)
    _close(seen[0][1], torch.zeros_like(seen[0][1]))
    _close(result["views"]["trace"]["v"][-1], population.v)


def test_population_step_precomputed_voltage_matches_one_frame_native_run():
    model, _ = _case(False)
    reference, _ = _case(False)
    voltage = torch.linspace(-0.2, 0.3, model.v.numel(), dtype=model.v.dtype).reshape(
        1, *model.shape
    )
    actual = _step(model, False, callbacks=_plan(model, False), ve=voltage)
    expected = reference.run(
        ve=voltage, dt=DT, functional_callbacks=_plan(reference, False)
    )
    _close(actual.outputs, expected.outputs)


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_step_pure_observations_do_not_consume_shared_stochastic_draws(network):
    if network:
        model, pulses = _stochastic_network()
        reference, reference_pulses = _stochastic_network()
    else:
        model, _, pulse = _stochastic_population()
        reference, _, reference_pulse = _stochastic_population()
        pulses, reference_pulses = [pulse], [reference_pulse]
    reference.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    functional, _ = dn.func.make_functional(model, dt=STOCHASTIC_DT)
    callbacks = functional.make_callbacks(
        {
            "trace": dn.Recorder(("left.v", "right.v", "t") if network else ("v", "t")),
            "energy": _NetworkEnergy() if network else _PortableEnergyReducer(),
        }
    )
    global_rng = torch.get_rng_state().clone()
    if network:
        assert reference.step() is reference
        actual = model.step(functional_callbacks=callbacks)
    else:
        assert reference.step(STOCHASTIC_DT) is reference
        actual = model.step(STOCHASTIC_DT, functional_callbacks=callbacks)
    assert actual["energy"]["steps"].item() == 1
    assert actual["trace"]["left.v" if network else "v"].shape[0] == 2
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    for pulse, expected in zip(pulses, reference_pulses, strict=True):
        _close(pulse.generator.get_state(), expected.generator.get_state())
    _close(torch.get_rng_state(), global_rng)


def test_network_step_scoped_recorder_selects_a_named_packed_population_region():
    model, reference = _network(packed=True), _network(packed=True)
    functional, _ = dn.func.make_functional_network(model)
    callbacks = functional.make_callbacks(
        {
            "selected": dn.callbacks.PopulationCallback(
                "packed",
                dn.Recorder(("v", "i_membrane", "t"), region=model.packed.left.soma),
            )
        }
    )
    first = model.step(functional_callbacks=callbacks)
    last = model.step(
        functional_callbacks=callbacks, functional_callback_state=first.state
    )
    recorder = dn.Recorder(("v", "i_membrane", "t"), region=reference.packed.left.soma)
    reference.run(
        2 * DT, callbacks=[dn.callbacks.PopulationCallback("packed", recorder)]
    )
    for name in last["selected"]:
        _close(
            torch.cat((first["selected"][name], last["selected"][name])),
            recorder.stack(name),
        )


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
def test_step_observer_guards_preserve_source_state(
    network,
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
):
    model, _ = _case(network)
    callbacks = _plan(model, network)
    _run(model, network, "run", callbacks, steps=0)
    before = _snapshot(model.state_dict_for_checkpoint())
    with pytest.raises(ValueError, match="requires functional_callbacks"):
        _step(model, network, state=object())
    with pytest.raises(TypeError, match="FunctionalCallbacks"):
        _step(model, network, callbacks={})
    with pytest.raises(dn.func.FunctionalizationError, match="exact source"):
        _step(_case(network)[0], network, callbacks=callbacks)
    with pytest.raises(TypeError, match="FunctionalCallbackState"):
        _step(model, network, callbacks=callbacks, state=object())
    if not network:
        with pytest.raises(dn.func.FunctionalizationError, match="dt|timestep"):
            model.step(2 * DT, functional_callbacks=callbacks)
    disk = dn.Recorder(("left.v", "t") if network else ("v", "t")).set_hdf5(
        str(tmp_path / "unsupported.h5"), cache_every=2
    )
    functional, _ = dn.func.make_functional(model, dt=DT)
    try:
        with pytest.raises(dn.func.FunctionalizationError, match="HDF5"):
            _step(model, network, callbacks=functional.make_callbacks({"disk": disk}))
    finally:
        disk.close()
    _assert_packet_equal(model.state_dict_for_checkpoint(), before)


@pytest.mark.mps
def test_mps_public_step_and_native_step_share_callback_carry_cadence_and_gradients():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"

    def build():
        model = dn.SingleCompartment(
            N=1, C=2, v_init=-63, integrator=dn.bwd_euler_sc(imem=False)
        )
        model.insert(pas, g=0.001, e=-60)
        model.train()
        model.initialize()
        return model

    with dn.ctx(DEVICE="mps", DTYPE=torch.float32, REQUIRE_GRAD=1):
        model, reference = build(), build()
        callbacks = _population_plan(model, fields=("v", "t"))
        first = dn.step(model, dt=DT, functional_callbacks=callbacks)
        last = model.step(
            DT, functional_callbacks=callbacks, functional_callback_state=first.state
        )
        expected = reference.run(
            tstop=2 * DT,
            dt=DT,
            functional_callbacks=_population_plan(reference, fields=("v", "t")),
        )
        for field in last["trace"]:
            _close(
                torch.cat((first["trace"][field], last["trace"][field])),
                expected["trace"][field],
            )
        observed = torch.autograd.grad(
            last["energy"]["energy"].sum(), model.mech.pas.g_param
        )[0]
        wanted = torch.autograd.grad(
            expected["energy"]["energy"].sum(), reference.mech.pas.g_param
        )[0]
        _close(observed, wanted)
        assert observed.device.type == "mps" and observed.abs().sum() > 0
