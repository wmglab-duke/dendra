"""Native Networks share qualified and Population-scoped pure observations."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn, graded_syn, pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_native_population_functional_callbacks import _ViewHistory
from tests.func.test_network import DT as EVENT_DT
from tests.func.test_network_pending_calendar import _network as _pending_network
from tests.func.test_network_population_callbacks import CONDUCTANCE, DT, _network
from tests.func.test_network_source_history import _stim_history
from tests.func.test_stimulation_rng_sharing import DT as STOCHASTIC_DT
from tests.func.test_stimulation_rng_sharing import _network as _stochastic_network

pytestmark = pytest.mark.cpu
_METHODS = ("run", "longrun")


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


def _frame_tensors(state):
    """Keep every Tensor and alias, omitting only static None dictionary entries."""
    if isinstance(state, dict):
        return {
            name: _frame_tensors(value)
            for name, value in state.items()
            if value is not None
        }
    assert torch.is_tensor(state)
    return state


def _restore_frame(state, template):
    """Restore fixed None entries only for the model-state schema oracle."""
    if isinstance(template, dict):
        return {
            name: None if value is None else _restore_frame(state[name], value)
            for name, value in template.items()
        }
    return state


class _NetworkFrames(dn.func.FunctionalCallback):
    """Retain complete native frames without copying their Tensor views."""

    def initialize(self, state, auxiliary):
        assert auxiliary is None
        tensors = _frame_tensors(state)
        return {"initial": tensors, "latest": tensors}, None

    def update(self, carry, state, auxiliary):
        assert "populations" in auxiliary
        assert set(auxiliary["populations"]) == set(state["populations"])
        for name, values in auxiliary["populations"].items():
            assert set(values) == {"v"}
            assert values["v"] is state["populations"][name]["integrator"]["v"]
        return {"initial": carry["initial"], "latest": _frame_tensors(state)}, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry

    def checkpoint_configuration(self):
        return {"kind": "tests.native-network.frames", "version": 1, "fields": {}}


def _driven():
    network = _network()
    waveform = dn.mono_rect(amp=2e-6, delay=0, pw=10 * DT)
    network.left[:, 0].inject(waveform)
    network.right[:, 0].inject(dn.constant(value=-1e-6))
    network.initialize(DT)
    return network, waveform.amp


def _specifications(network, field, scope=dn.callbacks.PopulationCallback):
    return {
        "qualified": dn.Recorder(("left.v", "right.v", "t"), dt=2 * DT),
        "left": scope("left", dn.Recorder(("v", "i_membrane", "t"), dt=2 * DT)),
        "lfp": scope("left", dn.callbacks.LFP([field, field])),
        "count": scope(
            "right", dn.callbacks.APCount(threshold=-100, node_check=[0, -1])
        ),
        "views": scope("left", _ViewHistory()),
        "network": _NetworkFrames(),
    }


def _plan(network, *, scope=dn.callbacks.PopulationCallback, field=None):
    if field is None:
        field = torch.ones_like(network.left.v)
    functional, tensors = dn.func.make_functional_network(network)
    specifications = _specifications(network, field, scope)
    return (
        functional,
        tensors,
        functional.make_callbacks(specifications),
        specifications,
    )


def _execute(
    network, method, *, callbacks=None, state=None, steps=6, native=None, **kwargs
):
    options = dict(tstop=steps * network.dt, callbacks=native, **kwargs)
    if method == "longrun":
        options["chunklength"] = 4
    if callbacks is not None:
        options["functional_callbacks"] = callbacks
    if state is not None:
        options["functional_callback_state"] = state
    return getattr(network, method)(**options)


@pytest.mark.parametrize("method", _METHODS)
def test_native_network_callbacks_match_native_and_functional_oracles_with_shared_gradients(
    method,
):
    model, amplitude = _driven()
    field = torch.ones_like(model.left.v, requires_grad=True)
    functional, tensors, callbacks, configured = _plan(model, field=field)
    assert (
        tensors.parameters["populations"]["left"][CONDUCTANCE]
        is tensors.parameters["populations"]["right"][CONDUCTANCE]
    )
    assert (
        tensors.parameters["populations"]["left"][CONDUCTANCE]
        is not model.left.mech.pas.g_param
    )
    native_left = dn.Recorder(("v", "i_membrane", "t"), dt=2 * DT)
    native_right = dn.Recorder(("v", "t"), dt=2 * DT)
    native_lfp = dn.callbacks.LFP([field, field])
    native_count = dn.callbacks.APCount(threshold=-100, node_check=[0, -1])
    native = [
        dn.callbacks.PopulationCallback("left", native_left),
        dn.callbacks.PopulationCallback("right", native_right),
        dn.callbacks.PopulationCallback("left", native_lfp),
        dn.callbacks.PopulationCallback("right", native_count),
    ]
    actual = _execute(model, method, callbacks=callbacks, native=native)
    assert type(actual) is dn.func.FunctionalCallbackResults
    for name in ("v", "i_membrane", "t"):
        _close(actual["left"][name], native_left.stack(name))
    _close(actual["qualified"]["left.v"], native_left.stack("v"))
    _close(actual["qualified"]["right.v"], native_right.stack("v"))
    _close(actual["qualified"]["t"], native_right.stack("t"))
    _close(actual["lfp"], native_lfp.lfp)
    _close(actual["count"], native_count.record)
    assert actual["count"].dtype == torch.int32 and actual["count"].all()
    _close(actual["network"]["initial"], _frame_tensors(tensors.state))
    _close(actual["network"]["latest"], _frame_tensors(functional.extract(model).state))
    assert (
        actual.state["views"]["initial"]
        is actual.state["network"]["initial"]["populations"]["left"]["integrator"]["v"]
    )
    _, reference = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=6,
        callbacks=callbacks,
    )
    _close(actual.outputs, reference["callbacks"].outputs)

    def loss(results):
        return (results["lfp"] * 1e6).square().mean() + results["views"]["trace"][
            "v"
        ].square().mean()

    targets = (field, amplitude, model.left.mech.pas.g_param)
    observed = torch.autograd.grad(loss(actual), targets, retain_graph=True)
    reference_amplitudes = tuple(
        value
        for name, value in tensors.parameters["populations"]["left"].items()
        if name.startswith("stimulation.intra.") and name.endswith(".amp")
    )
    assert len(reference_amplitudes) == 1
    wanted = torch.autograd.grad(
        loss(reference["callbacks"]),
        (
            field,
            reference_amplitudes[0],
            tensors.parameters["populations"]["left"][CONDUCTANCE],
        ),
    )
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert configured["qualified"].rec["left.v"] == []
    assert configured["left"].callback.rec["v"] == []
    assert configured["lfp"].callback._lfp == []
    assert configured["count"].callback.record is None


def test_native_network_carry_keeps_phase_across_run_longrun_and_portable_legacy_scopes(
    tmp_path,
):
    model, _ = _driven()
    _, _, callbacks, _ = _plan(model)
    first = _execute(model, "run", callbacks=callbacks, steps=3)
    saved_model = _snapshot(model.state_dict_for_checkpoint())
    saved = _disk(callbacks.export_state(first.state), tmp_path / "callbacks.pt")
    empty = _execute(model, "longrun", callbacks=callbacks, state=first.state, steps=0)
    assert empty["left"]["v"].shape[0] == empty["lfp"].shape[0] == 0
    _close(empty.state.carries, first.state.carries)
    last = _execute(model, "longrun", callbacks=callbacks, state=empty.state, steps=3)
    reference, _ = _driven()
    _, _, reference_callbacks, _ = _plan(reference)
    whole = _execute(reference, "run", callbacks=reference_callbacks)
    for stream in ("qualified", "left"):
        for field in whole[stream]:
            _close(
                torch.cat(
                    (first[stream][field], empty[stream][field], last[stream][field])
                ),
                whole[stream][field],
            )
    _close(torch.cat((first["lfp"], last["lfp"])), whole["lfp"])
    _close(last["network"], whole["network"])
    _close(last["count"], whole["count"])
    receiver, _ = _driven()
    receiver.restore_dict_from_checkpoint(saved_model)
    _, _, receiving, _ = _plan(receiver, scope=dn.func.PopulationCallback)
    restored = receiving.restore_state(saved)
    tail = _execute(receiver, "longrun", callbacks=receiving, state=restored, steps=3)
    _close(tail.outputs, last.outputs)
    with pytest.raises(dn.func.FunctionalizationError, match="different callback plan"):
        _execute(model, "run", callbacks=_plan(model)[2], state=first.state, steps=0)


@pytest.mark.parametrize("method", _METHODS)
def test_native_network_zero_steps_preserve_hooks_initial_observation_and_resume(
    method,
):
    class Hooks(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.calls = []

        def pre_loop_hook(self, network):
            self.calls.append("pre")
            network.left.v.add_(0.5)

        def post_loop_hook(self, network):
            self.calls.append("post")
            network.left.v.add_(0.25)

    model = _network()
    model.eval()
    initial = model.left.v.detach().clone()
    callbacks, native = _plan(model)[2], Hooks()
    first = _execute(model, method, callbacks=callbacks, native=[native], steps=0)
    assert native.calls == ["pre", "post"]
    _close(first["views"]["initial"], initial + 0.5)
    assert first["left"]["v"].shape == (1, *model.left.shape)
    native.calls.clear()
    last = _execute(
        model, method, callbacks=callbacks, state=first.state, native=[native], steps=0
    )
    assert native.calls == ["pre", "post"]
    assert last["left"]["v"].shape[0] == last["lfp"].shape[0] == 0
    _close(last.state.carries, first.state.carries)
    assert model.t.item() == 0


@pytest.mark.parametrize("method", _METHODS)
def test_native_network_pure_views_precede_post_step_mutation_and_own_initial_frames(
    method,
):
    class Overwrite(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.voltages, self.times = [], []

        def post_step_hook(self, network):
            self.voltages.append(network.left.v.detach().clone())
            self.times.append(network.t.detach().clone())
            network.left.v.add_(5)

    model = _network()
    model.eval()
    initial = model.left.v.detach().clone()
    callbacks, native = _plan(model)[2], Overwrite()
    result = _execute(model, method, callbacks=callbacks, native=[native], steps=3)
    _close(result["views"]["trace"]["v"], torch.stack((initial, *native.voltages)))
    _close(
        result["views"]["trace"]["t"][1:], torch.stack(native.times).to(model.left.t)
    )
    _close(
        result["network"]["latest"]["populations"]["left"]["integrator"]["v"],
        native.voltages[-1],
    )
    _close(result["network"]["latest"]["clock"]["t"], native.times[-1])
    frozen = _snapshot(result.outputs)
    with torch.no_grad():
        model.left.v.fill_(-999)
        model.right.v.fill_(-998)
        model.t.fill_(999)
    _close(result.outputs, frozen)
    _close(result.state["views"]["initial"], initial)


def test_native_packed_network_observations_keep_named_population_regions():
    model = _network(packed=True)
    functional, _ = dn.func.make_functional_network(model)
    region = model.packed.left.soma
    configured = dn.Recorder(("v", "i_membrane", "t"), region=region)
    callbacks = functional.make_callbacks(
        {"selected": dn.callbacks.PopulationCallback("packed", configured)}
    )
    native = dn.Recorder(("v", "i_membrane", "t"), region=region)
    result = _execute(
        model,
        "longrun",
        callbacks=callbacks,
        native=[dn.callbacks.PopulationCallback("packed", native)],
        steps=3,
    )
    for field in result["selected"]:
        _close(result["selected"][field], native.stack(field))
    assert result["selected"]["v"].shape == (4, *model.packed.v[region.index].shape)


def test_native_network_callback_frames_include_netstim_event_and_continuous_state():
    cell = dn.SingleCompartment(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
    cell.insert(pas, g=0.001, e=-65)
    cell.insert(expsyn.rename("syn"), tau=1.2, e=0)
    cell.insert(graded_syn.rename("analog"), e=-10)
    cell.initialize()
    stimulus = dn.NetStim(
        N=1, interval=0.5, start=0, noise=0.6, max_spikes=20, seed=812
    )
    model = dn.Network(
        {"cell": cell},
        netstim=stimulus,
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    model.connect_one_to_one(
        stimulus[:], cell[:], cell.mech.syn, weight=0.003, delay=EVENT_DT
    )
    model.connect_continuous_one_to_one(
        model.cell[:],
        model.cell[:],
        model.cell.mech.analog,
        weight=0.002,
        delay=EVENT_DT,
        transform=torch.sigmoid,
        input="g_pre",
        allow_autapses=True,
    )
    model.train()
    model.initialize(EVENT_DT)
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=True,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    functional, tensors = dn.func.make_functional_network(model)
    callbacks = functional.make_callbacks(
        {"network": _NetworkFrames(), "trace": dn.Recorder(("cell.v", "t"))}
    )
    draws = functional.sample_runtime_draws(tensors.state, steps=3)
    expected, auxiliary = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        steps=3,
        draws=draws,
        callbacks=callbacks,
    )
    actual = model.run(3 * EVENT_DT, functional_callbacks=callbacks)
    _close(actual.outputs, auxiliary["callbacks"].outputs)
    _close(actual["network"]["latest"], _frame_tensors(expected))
    assert actual["network"]["latest"]["netstim"] is not None
    assert actual["network"]["latest"]["synapses"]
    assert actual["network"]["latest"]["continuous_synapses"]


@pytest.mark.parametrize(
    "backend", ("source_history", "bitpacked_history", "sparse_calendar")
)
def test_native_network_observation_frames_normalize_each_event_backend(backend):
    def build():
        return (
            _stim_history(taps=None)
            if backend == "source_history"
            else _pending_network(backend)
        )

    model, baseline = build(), build()
    baseline.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    functional, tensors = dn.func.make_functional_network(model, pending_capacity=64)
    callbacks = functional.make_callbacks({"network": _NetworkFrames()})
    result = _execute(model, "longrun", callbacks=callbacks, steps=6)
    assert _execute(baseline, "longrun", steps=6) is None
    functional._validate_state(
        _restore_frame(result["network"]["initial"], tensors.state)
    )
    functional._validate_state(
        _restore_frame(result["network"]["latest"], tensors.state)
    )
    _close(result["network"]["initial"], _frame_tensors(tensors.state))
    _close(result["network"]["latest"], _frame_tensors(functional.extract(model).state))
    _close(
        result["network"]["latest"], _frame_tensors(functional.extract(baseline).state)
    )
    assert next(iter(model.synapses.values()))._checkpoint_runtime_kind() == backend


@pytest.mark.parametrize("method", _METHODS)
def test_native_network_observations_preserve_shared_rng_draw_order_and_stimulation(
    method,
):
    observed, pulses = _stochastic_network()
    baseline, baseline_pulses = _stochastic_network()
    baseline.restore_dict_from_checkpoint(
        _snapshot(observed.state_dict_for_checkpoint())
    )
    functional, _ = dn.func.make_functional_network(observed)
    initial = _snapshot(functional.extract().state)
    callbacks = functional.make_callbacks(
        {
            "qualified": dn.Recorder(("left.v", "right.v", "t")),
            "network": _NetworkFrames(),
        }
    )
    options = dict(tstop=3 * STOCHASTIC_DT)
    if method == "longrun":
        options["chunklength"] = 2
    global_rng = torch.get_rng_state().clone()
    assert getattr(baseline, method)(**options) is None
    result = getattr(observed, method)(**options, functional_callbacks=callbacks)
    assert result["qualified"]["left.v"].shape[0] == 4
    _close(result["network"]["initial"], _frame_tensors(initial))
    _close(
        result["network"]["latest"], _frame_tensors(functional.extract(observed).state)
    )
    _assert_packet_equal(
        observed.state_dict_for_checkpoint(), baseline.state_dict_for_checkpoint()
    )
    for pulse, reference in zip(pulses, baseline_pulses, strict=True):
        _close(pulse.generator.get_state(), reference.generator.get_state())
    _close(torch.get_rng_state(), global_rng)


def test_older_native_network_observation_gradients_preserve_later_model_state():
    model, amplitude = _driven()
    callbacks = _plan(model)[2]
    first = _execute(model, "run", callbacks=callbacks, steps=2)
    _execute(model, "longrun", callbacks=callbacks, state=first.state, steps=3)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    first["views"]["trace"]["v"].square().mean().backward()
    for gradient in (amplitude.grad, model.left.mech.pas.g_param.grad):
        assert (
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
        )
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


@pytest.mark.parametrize("method", _METHODS)
def test_native_network_observer_defaults_and_exact_owner_state_validation(method):
    model = _network()
    assert _execute(model, method, steps=0) is None
    callbacks = _plan(model)[2]
    frozen = _snapshot(model.state_dict_for_checkpoint())
    with pytest.raises(ValueError, match="requires functional_callbacks"):
        _execute(model, method, state=object(), steps=0)
    with pytest.raises(TypeError, match="FunctionalCallbacks"):
        _execute(model, method, callbacks={}, steps=0)
    with pytest.raises(dn.func.FunctionalizationError, match="exact source Network"):
        _execute(_network(), method, callbacks=callbacks, steps=0)
    with pytest.raises(TypeError, match="FunctionalCallbackState"):
        _execute(model, method, callbacks=callbacks, state=object(), steps=0)
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


@pytest.mark.mps
def test_mps_native_network_observations_keep_mixed_clock_devices_and_continuation():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"

    def build():
        populations = {}
        for name, size, reversal in (("left", 2, -60), ("right", 1, -70)):
            cell = dn.SingleCompartment(
                N=size, C=2, integrator=dn.bwd_euler_sc(imem=True)
            )
            cell.insert(pas, g=0.001, e=reversal)
            populations[name] = cell
        network = dn.Network(populations)
        network.eval()
        network.initialize(DT)
        return network

    with dn.ctx(DEVICE="mps", DTYPE=torch.float32, REQUIRE_GRAD=0):
        actual, reference = build(), build()
        callbacks = _plan(actual)[2]
        first = _execute(actual, "run", callbacks=callbacks, steps=2)
        last = _execute(
            actual, "longrun", callbacks=callbacks, state=first.state, steps=3
        )
        whole = _execute(reference, "run", callbacks=_plan(reference)[2], steps=5)
        _close(
            torch.cat((first["qualified"]["left.v"], last["qualified"]["left.v"])),
            whole["qualified"]["left.v"],
        )
        _close(last["network"], whole["network"])
        assert last["left"]["v"].device.type == "mps"
        assert last["qualified"]["t"].device.type == "cpu"
