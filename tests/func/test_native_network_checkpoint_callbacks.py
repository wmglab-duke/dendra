"""Pure Network observations survive native activation-checkpoint replay."""

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
from tests.func.test_native_network_functional_callbacks import (
    DT,
    _driven,
    _NetworkFrames,
    _specifications,
)
from tests.func.test_network import _mixed_network, _stim_network
from tests.func.test_stimulation_rng_sharing import (
    _gradient_leaves,
)
from tests.func.test_stimulation_rng_sharing import _network as _stochastic_network
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


class _NetworkEnergy(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        assert auxiliary is None
        voltage = next(iter(state["populations"].values()))["integrator"]["v"]
        return {
            "energy": voltage.new_zeros(()),
            "steps": voltage.new_zeros((), dtype=torch.int64),
        }, None

    def update(self, carry, state, auxiliary):
        del state
        energy = sum(
            values["v"].square().mean() for values in auxiliary["populations"].values()
        )
        return {"energy": carry["energy"] + energy, "steps": carry["steps"] + 1}, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry

    def checkpoint_configuration(self):
        return {"kind": "tests.native-network.energy", "version": 1, "fields": {}}


class _NativeEnergyLoss(dn.callbacks.Callback):
    def pre_step_hook(self, network):
        raise AssertionError("Network checkpointing has no native pre-step hook phase")

    def post_step_hook(self, network):
        return sum(
            population.v.square().mean() for population in network.populations.values()
        )


def _plan(network, *, field=None):
    functional, _ = dn.func.make_functional_network(network)
    if set(network.populations) == {"left", "right"} and network.left.integrator.imem:
        field = torch.ones_like(network.left.v) if field is None else field
        specifications = _specifications(network, field)
    else:
        specifications = {
            "trace": dn.Recorder((*[f"{name}.v" for name in network.populations], "t")),
            "network": _NetworkFrames(),
        }
    specifications["energy"] = _NetworkEnergy()
    return functional.make_callbacks(specifications)


def _checkpoint(
    network,
    callbacks=None,
    *,
    steps=5,
    state=None,
    final=False,
    restore=True,
    native=None,
):
    options = dict(
        tstop=steps * network.dt,
        chunklength=2,
        callbacks=native,
        safe_checkpoint=True,
        restore_state_after_backward=restore,
        return_final_state=final,
    )
    if callbacks is not None:
        options["functional_callbacks"] = callbacks
    if state is not None:
        options["functional_callback_state"] = state
    return network.longrun_checkpointed(**options)


def _loss(results):
    return results["energy"]["energy"] + (results["lfp"] * 1e6).square().mean()


@pytest.mark.parametrize("differentiation", ("backward", "grad", "unisolated_grad"))
def test_checkpointed_scoped_results_match_ordinary_execution_and_raw_native_gradients(
    differentiation,
):
    reference, reference_amplitude = _driven()
    reference_field = torch.ones_like(reference.left.v, requires_grad=True)
    expected = reference.run(
        5 * DT, functional_callbacks=_plan(reference, field=reference_field)
    )
    wanted = torch.autograd.grad(
        _loss(expected),
        (reference_field, reference_amplitude, reference.left.mech.pas.g_param),
    )
    model, amplitude = _driven()
    field = torch.ones_like(model.left.v, requires_grad=True)
    callbacks = _plan(model, field=field)
    isolated = differentiation != "unisolated_grad"
    native_loss, result = _checkpoint(model, callbacks, restore=isolated)
    assert native_loss is None
    _close(result.outputs, expected.outputs)
    assert result["energy"]["steps"].item() == 5
    assert result["views"]["trace"]["v"].shape[0] == 6
    assert result["qualified"]["t"].shape[0] == 3
    frozen = _snapshot(model.state_dict_for_checkpoint())
    targets = field, amplitude, model.left.mech.pas.g_param
    if differentiation == "backward":
        _loss(result).backward()
        observed = tuple(value.grad for value in targets)
    else:
        observed = torch.autograd.grad(_loss(result), targets)
    _close(observed, wanted)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    if isolated:
        _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


def test_partial_repeated_and_higher_order_callback_replay_preserves_latest_network():
    model, _ = _driven()
    _, result = _checkpoint(model, _plan(model))
    frozen = _snapshot(model.state_dict_for_checkpoint())
    parameter = model.left.mech.pas.g_param
    partial = result["views"]["trace"]["v"][1].square().mean()
    first = torch.autograd.grad(
        partial, parameter, create_graph=True, retain_graph=True
    )[0]
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    second = torch.autograd.grad(first, parameter, retain_graph=True)[0]
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    repeated = torch.autograd.grad(partial, parameter, retain_graph=True)[0]
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    reference, _ = _driven()
    expected = reference.run(5 * DT, functional_callbacks=_plan(reference))
    target = reference.left.mech.pas.g_param
    reference_first = torch.autograd.grad(
        expected["views"]["trace"]["v"][1].square().mean(), target, create_graph=True
    )[0]
    reference_second = torch.autograd.grad(reference_first, target)[0]
    _close(first, reference_first)
    _close(second, reference_second)
    _close(repeated, reference_first)
    assert torch.isfinite(second).all() and second.abs().sum() > 0


def test_older_callback_graph_preserves_later_execution_and_fractional_zero_step_state():
    model, _ = _driven()
    callbacks = _plan(model)
    _, first = _checkpoint(model, callbacks, steps=2)
    later = model.longrun(
        3 * DT,
        chunklength=2,
        functional_callbacks=callbacks,
        functional_callback_state=first.state,
    )
    loss, empty = _checkpoint(model, callbacks, steps=0.5, state=later.state)
    assert loss is None and empty["views"]["trace"]["v"].shape[0] == 0
    assert empty["energy"]["steps"].item() == 5
    frozen = _snapshot(model.state_dict_for_checkpoint())
    assert frozen["duration_remainder"].item() == pytest.approx(0.5 * DT)
    first["energy"]["energy"].backward()
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    assert model.left.mech.pas.g_param.grad.abs().sum() > 0


def test_run_checkpoint_longrun_resume_and_portable_callback_state_keep_one_recording_phase(
    tmp_path,
):
    model, _ = _driven()
    callbacks = _plan(model)
    first = model.run(2 * DT, functional_callbacks=callbacks)
    _, middle = _checkpoint(model, callbacks, steps=3, state=first.state)
    saved_model = _snapshot(model.state_dict_for_checkpoint())
    saved_callbacks = _disk(
        callbacks.export_state(middle.state), tmp_path / "callbacks.pt"
    )
    last = model.longrun(
        DT,
        chunklength=2,
        functional_callbacks=callbacks,
        functional_callback_state=middle.state,
    )
    complete, _ = _driven()
    whole = complete.run(6 * DT, functional_callbacks=_plan(complete))
    for name in ("qualified", "left"):
        for field in whole[name]:
            _close(
                torch.cat(tuple(part[name][field] for part in (first, middle, last))),
                whole[name][field],
            )
    _close(
        torch.cat(tuple(part["lfp"] for part in (first, middle, last))), whole["lfp"]
    )
    _close(last["energy"], whole["energy"])
    receiver, _ = _driven()
    receiver.restore_dict_from_checkpoint(saved_model)
    receiving = _plan(receiver)
    restored = receiving.restore_state(saved_callbacks)
    suffix = receiver.longrun(
        DT,
        chunklength=2,
        functional_callbacks=receiving,
        functional_callback_state=restored,
    )
    _close(suffix.outputs, last.outputs)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    actual_gradient = torch.autograd.grad(
        last["energy"]["energy"], model.left.mech.pas.g_param
    )[0]
    expected_gradient = torch.autograd.grad(
        whole["energy"]["energy"], complete.left.mech.pas.g_param
    )[0]
    _close(actual_gradient, expected_gradient)
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


@pytest.mark.parametrize("final", (False, True))
def test_native_loss_and_observation_return_forms_preserve_default_checkpoint_api(
    final,
):
    baseline, _ = _driven()
    default = _checkpoint(baseline, steps=3, final=final, native=[_NativeEnergyLoss()])
    model, _ = _driven()
    actual = _checkpoint(
        model, _plan(model), steps=3, final=final, native=[_NativeEnergyLoss()]
    )
    assert isinstance(actual, tuple) and len(actual) == (3 if final else 2)
    result = actual[-1]
    expected_loss = default[0] if final else default
    _close(actual[0], expected_loss)
    _close(actual[0], result["energy"]["energy"])
    if final:
        assert isinstance(default, tuple) and len(default) == 2
        _assert_packet_equal(actual[1], model.state_dict_for_checkpoint())
    else:
        assert torch.is_tensor(default)
    (actual[0] + result["energy"]["energy"]).backward()
    assert torch.isfinite(model.left.mech.pas.g_param.grad).all()
    without_loss, _ = _driven()
    untouched = _checkpoint(without_loss, steps=0, final=final)
    if final:
        assert isinstance(untouched, tuple) and len(untouched) == 2
        assert untouched[0] is None
    else:
        assert untouched is None


def test_checkpoint_zero_step_callbacks_keep_native_loop_hooks_and_explicit_phase():
    class Hooks(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.calls = []

        def pre_loop_hook(self, network):
            self.calls.append("pre")

        def post_loop_hook(self, network):
            self.calls.append("post")

        def pre_step_hook(self, network):
            raise AssertionError("no step is accepted")

    model, _ = _driven()
    callbacks, native = _plan(model), Hooks()
    loss, first = _checkpoint(model, callbacks, steps=0, native=[native])
    assert loss is None and native.calls == ["pre", "post"]
    assert first["views"]["trace"]["v"].shape[0] == 1
    native.calls.clear()
    loss, last = _checkpoint(
        model, callbacks, steps=0, state=first.state, native=[native]
    )
    assert loss is None and native.calls == ["pre", "post"]
    assert last["views"]["trace"]["v"].shape[0] == last["lfp"].shape[0] == 0
    _close(last.state.carries, first.state.carries)


def test_stochastic_callback_replay_preserves_shared_draw_order_rng_and_collection():
    model, pulses = _stochastic_network()
    baseline, baseline_pulses = _stochastic_network()
    baseline.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    # The fixture detaches its initialized buffers. Fresh initialization reconnects
    # RAND realizations to their raw means on both synchronized models.
    model.initialize(model.dt)
    baseline.initialize(baseline.dt)
    functional, _ = dn.func.make_functional_network(model)
    callbacks = functional.make_callbacks(
        {
            "trace": dn.Recorder(("left.v", "right.v", "t")),
            "energy": _NetworkEnergy(),
            "network": _NetworkFrames(),
        }
    )
    expected_loss = _checkpoint(baseline, steps=4, native=[_NativeEnergyLoss()])
    loss, result = _checkpoint(model, callbacks, steps=4)
    assert loss is None and result["trace"]["left.v"].shape[0] == 5
    assert result["energy"]["steps"].item() == 4
    _close(result["energy"]["energy"], expected_loss)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), baseline.state_dict_for_checkpoint()
    )
    frozen, outputs = (
        _snapshot(model.state_dict_for_checkpoint()),
        _snapshot(result.outputs),
    )
    targets, reference_targets = _gradient_leaves(model), _gradient_leaves(baseline)
    observed = torch.autograd.grad(result["energy"]["energy"], targets)
    expected = torch.autograd.grad(expected_loss, reference_targets)
    _close(observed, expected)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    _close(result.outputs, outputs)
    for pulse, reference in zip(pulses, baseline_pulses, strict=True):
        _close(pulse.generator.get_state(), reference.generator.get_state())


@pytest.mark.parametrize("kind", ("continuous", "netstim"))
def test_connected_checkpoint_observations_restore_complete_event_continuous_and_rng_carry(
    kind,
):
    build = _mixed_network if kind == "continuous" else _stim_network
    model, baseline = build(), build()
    baseline.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    callbacks = _plan(model)
    assert _checkpoint(baseline, steps=3) is None
    loss, result = _checkpoint(model, callbacks, steps=3)
    assert loss is None
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), baseline.state_dict_for_checkpoint()
    )
    frozen = _snapshot(model.state_dict_for_checkpoint())
    gradient = torch.autograd.grad(
        result["energy"]["energy"], model.cell.mech.pas.g_param
    )[0]
    assert torch.isfinite(gradient).all()
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    assert result["network"]["latest"]["synapses"]
    if kind == "continuous":
        assert result["network"]["latest"]["continuous_synapses"]
    else:
        assert result["network"]["latest"]["netstim"]


def test_no_observer_native_loss_gradients_restore_later_state_and_zero_step_budget():
    model, _ = _driven()
    loss = _checkpoint(model, steps=3, native=[_NativeEnergyLoss()])
    model.run(2 * DT)
    assert _checkpoint(model, steps=0.5) is None
    frozen = _snapshot(model.state_dict_for_checkpoint())
    target = model.left.mech.pas.g_param
    first = torch.autograd.grad(loss, target, create_graph=True)[0]
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    second = torch.autograd.grad(first, target)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


def test_checkpoint_observer_guards_precede_execution(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
):
    model, _ = _driven()
    callbacks = _plan(model)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    with pytest.raises(ValueError, match="requires functional_callbacks"):
        _checkpoint(model, steps=0, state=object())
    with pytest.raises(TypeError, match="FunctionalCallbacks"):
        _checkpoint(model, {}, steps=0)
    with pytest.raises(dn.func.FunctionalizationError, match="exact source Network"):
        _checkpoint(_driven()[0], callbacks, steps=0)
    with pytest.raises(TypeError, match="FunctionalCallbackState"):
        _checkpoint(model, callbacks, steps=0, state=object())
    disk = dn.Recorder(("left.v", "t")).set_hdf5(
        str(tmp_path / "unsupported.h5"), cache_every=2
    )
    functional, _ = dn.func.make_functional_network(model)
    try:
        with pytest.raises(dn.func.FunctionalizationError, match="HDF5"):
            _checkpoint(model, functional.make_callbacks({"disk": disk}), steps=0)
    finally:
        disk.close()
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


@pytest.mark.mps
def test_mps_native_network_checkpoint_callbacks_preserve_gradients_and_model_state():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"

    def build():
        populations = {}
        for name, initial, reversal in (("left", -63, -60), ("right", -67, -70)):
            cell = dn.SingleCompartment(
                N=1, C=2, v_init=initial, integrator=dn.bwd_euler_sc(imem=False)
            )
            cell.insert(pas, g=0.001, e=reversal)
            populations[name] = cell
        network = dn.Network(populations)
        network.train()
        network.initialize(DT)
        return network

    with dn.ctx(DEVICE="mps", DTYPE=torch.float32, REQUIRE_GRAD=1):
        model, reference = build(), build()
        expected = reference.run(3 * DT, functional_callbacks=_plan(reference))
        loss, result = _checkpoint(model, _plan(model), steps=3)
        assert loss is None and result["energy"]["energy"].device.type == "mps"
        _close(result.outputs, expected.outputs)
        frozen = _snapshot(model.state_dict_for_checkpoint())
        observed = torch.autograd.grad(
            result["energy"]["energy"], model.left.mech.pas.g_param
        )[0]
        wanted = torch.autograd.grad(
            expected["energy"]["energy"], reference.left.mech.pas.g_param
        )[0]
        _close(observed, wanted)
        assert torch.isfinite(observed).all() and observed.abs().sum() > 0
        _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
