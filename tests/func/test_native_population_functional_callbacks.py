"""Native Population loops share pure observations and explicit callback carry."""

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
from tests.func.test_imperative_checkpoint_callbacks import (
    DT,
    EnergyReducer,
    LegacyVoltageLoss,
    _gnabar,
    _model,
)
from tests.func.test_stimulation_rng_sharing import DT as SHARED_DT
from tests.func.test_stimulation_rng_sharing import _lower as _lower_stochastic
from tests.func.test_stimulation_rng_sharing import (
    _population as _stochastic_population,
)
from tests.test_recorder_h5_integration import local_hdf5_writer as local_hdf5_writer

pytestmark = pytest.mark.cpu
_METHODS = ("run", "longrun", "longrun_checkpointed")


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


class _ViewHistory(dn.func.FunctionalCallback):
    """Borrow frame views so the native observation boundary must own storage."""

    def initialize(self, state, auxiliary):
        assert auxiliary is None
        voltage = state["integrator"]["v"]
        return {"initial": voltage, "latest": voltage[..., ::2]}, {
            "v": voltage,
            "t": state["clock"]["t"],
        }

    def update(self, carry, state, auxiliary):
        voltage = state["integrator"]["v"]
        return {"initial": carry["initial"], "latest": voltage[..., ::2]}, {
            "v": auxiliary["v"],
            "t": state["clock"]["t"],
        }

    def finalize(self, carry, emissions):
        return {**carry, "trace": emissions}

    def checkpoint_configuration(self):
        return {"kind": "tests.native-population.views", "version": 1, "fields": {}}


class _PortableEnergyReducer(EnergyReducer):
    def checkpoint_configuration(self):
        return {"kind": "tests.native-population.energy", "version": 1, "fields": {}}


def _plan(model, *, fields=("v", "hh.m", "t")):
    functional, _ = dn.func.make_functional(model, dt=DT)
    return functional.make_callbacks(
        {
            "trace": dn.Recorder(fields, dt=2 * DT),
            "views": _ViewHistory(),
            "energy": _PortableEnergyReducer(),
            "count": dn.callbacks.APCount(threshold=-100.0, node_check=[0, -1]),
        }
    )


def _execute(
    model, method, *, plan=None, state=None, steps=5, callbacks=None, **inputs
):
    options = dict(tstop=steps * DT, dt=DT, callbacks=callbacks, **inputs)
    if method != "run":
        options["chunklength"] = 2
    if plan is not None:
        options["functional_callbacks"] = plan
    if state is not None:
        options["functional_callback_state"] = state
    if method == "longrun_checkpointed":
        options.update(safe_checkpoint=True, restore_state_after_backward=True)
    result = getattr(model, method)(**options)
    if method == "longrun_checkpointed" and plan is not None:
        assert isinstance(result, tuple) and len(result) == 2
        return result[1]
    return result


@pytest.mark.parametrize("method", _METHODS)
def test_native_loops_share_custom_emissions_reducers_mechanism_recording_and_gradients(
    method,
):
    reference = _model(require_grad=True, stimulated=True)
    recorder = dn.Recorder(("v", "hh.m", "t"))
    reference.run(tstop=5 * DT, dt=DT, callbacks=[recorder])
    voltage = recorder.stack("v")
    expected_energy = voltage[1:].square().mean(-1).sum(0)
    expected_loss = expected_energy.sum() + voltage.square().mean()
    expected_gradient = torch.autograd.grad(expected_loss, _gnabar(reference))[0]

    model = _model(require_grad=True, stimulated=True)
    callbacks = _plan(model)
    native = dn.Recorder(("v", "t"))
    actual = _execute(
        model,
        method,
        plan=callbacks,
        callbacks=None if method == "longrun_checkpointed" else [native],
    )
    assert type(actual) is dn.func.FunctionalCallbackResults
    for field in ("v", "hh.m", "t"):
        _close(actual["trace"][field], recorder.stack(field)[::2])
    _close(actual["views"]["trace"]["v"], voltage)
    _close(actual["views"]["trace"]["t"], recorder.stack("t"))
    _close(actual["views"]["initial"], voltage[0])
    _close(actual["views"]["latest"], voltage[-1, ..., ::2])
    _close(actual["energy"]["energy"], expected_energy)
    assert actual["energy"]["steps"].item() == 5
    assert torch.equal(actual["count"], torch.ones((1, 2), dtype=torch.int32))
    if method != "longrun_checkpointed":
        _close(native.stack("v"), voltage)
        _close(native.stack("t"), recorder.stack("t"))
    objective = (
        actual["energy"]["energy"].sum() + actual["views"]["trace"]["v"].square().mean()
    )
    gradient = torch.autograd.grad(objective, _gnabar(model))[0]
    _close(gradient, expected_gradient)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_native_callback_carry_continues_across_methods_and_portable_restore(tmp_path):
    model = _model(require_grad=True)
    callbacks = _plan(model)
    first = _execute(model, "run", plan=callbacks, steps=3)
    saved_model = _snapshot(model.state_dict_for_checkpoint())
    saved_callbacks = _disk(
        callbacks.export_state(first.state), tmp_path / "callbacks.pt"
    )
    empty = _execute(model, "longrun", plan=callbacks, state=first.state, steps=0)
    assert empty["trace"]["v"].shape[0] == 0
    _close(empty.state.carries, first.state.carries)
    middle = _execute(
        model, "longrun_checkpointed", plan=callbacks, state=empty.state, steps=2
    )
    last = _execute(model, "longrun", plan=callbacks, state=middle.state, steps=2)
    complete = _model(require_grad=True)
    whole = _execute(complete, "run", plan=_plan(complete), steps=7)
    for field in ("v", "hh.m", "t"):
        _close(
            torch.cat([part["trace"][field] for part in (first, empty, middle, last)]),
            whole["trace"][field],
        )
    _close(
        torch.cat(
            [part["views"]["trace"]["v"] for part in (first, empty, middle, last)]
        ),
        whole["views"]["trace"]["v"],
    )
    _close(last["energy"], whole["energy"])
    _close(last["count"], whole["count"])
    _close(model.v, complete.v)
    receiver = _model(require_grad=True)
    receiver.restore_dict_from_checkpoint(saved_model)
    receiving = _plan(receiver)
    restored = receiving.restore_state(saved_callbacks)
    tail = _execute(receiver, "longrun", plan=receiving, state=restored, steps=4)
    for field in ("v", "hh.m", "t"):
        _close(
            tail["trace"][field],
            torch.cat((middle["trace"][field], last["trace"][field])),
        )
    _close(tail["energy"], last["energy"])
    _close(tail["count"], last["count"])
    _close(receiver.v, model.v)
    with pytest.raises(dn.func.FunctionalizationError, match="different callback plan"):
        _execute(model, "run", plan=_plan(model), state=first.state, steps=0)


@pytest.mark.parametrize("method", ("run", "longrun"))
def test_native_zero_step_observations_preserve_existing_hook_behavior_and_phase(
    method,
):
    class Hooks(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.calls = []

        def pre_loop_hook(self, model):
            self.calls.append("pre_loop")

        def post_loop_hook(self, model):
            self.calls.append("post_loop")

    model = _model()
    callbacks, native = _plan(model), Hooks()
    fresh = _execute(model, method, plan=callbacks, steps=0, callbacks=[native])
    assert fresh["trace"]["v"].shape == (1, *model.shape)
    assert fresh["energy"]["steps"].item() == 0
    assert native.calls == (["pre_loop", "post_loop"] if method == "run" else [])
    native.calls.clear()
    resumed = _execute(
        model, method, plan=callbacks, state=fresh.state, steps=0, callbacks=[native]
    )
    assert resumed["trace"]["v"].shape == (0, *model.shape)
    assert resumed["views"]["trace"]["v"].shape == (0, *model.shape)
    _close(resumed.state.carries, fresh.state.carries)
    assert native.calls == (["pre_loop", "post_loop"] if method == "run" else [])
    assert model.t.item() == 0


@pytest.mark.parametrize("method", ("run", "longrun"))
def test_native_pre_loop_keeps_its_existing_position_relative_to_integrator_initialization(
    method,
):
    class TimestepAtPreLoop(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.values = []

        def pre_loop_hook(self, model):
            self.values.append(model.integrator.dt)

    model = _model()
    model.run(tstop=0, dt=3 * DT)
    assert model.integrator.dt == 3 * DT
    callbacks, native = _plan(model), TimestepAtPreLoop()
    _execute(model, method, plan=callbacks, callbacks=[native], steps=1)
    assert native.values == ([3 * DT] if method == "run" else [DT])
    assert model.integrator.dt == DT


@pytest.mark.parametrize("method", ("run", "longrun"))
def test_pure_views_observe_before_native_post_step_writes_and_own_retained_storage(
    method,
):
    class Overwrite(dn.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.voltages, self.times = [], []

        def post_step_hook(self, model):
            self.voltages.append(model.v.detach().clone())
            self.times.append(model.t.detach().clone())
            model.v.add_(5)

    model = _model()
    model.eval()
    initial = model.v.detach().clone()
    callbacks, native = _plan(model), Overwrite()
    result = _execute(model, method, plan=callbacks, callbacks=[native], steps=3)
    expected = torch.stack((initial, *native.voltages))
    _close(result["views"]["trace"]["v"], expected)
    _close(result["views"]["trace"]["t"][1:], torch.stack(native.times))
    _close(result["views"]["initial"], initial)
    _close(result["views"]["latest"], native.voltages[-1][..., ::2])
    assert not result["views"]["initial"].is_set_to(model.v)
    frozen = _snapshot(result.outputs)
    with torch.no_grad():
        model.v.fill_(-999)
        model.t.fill_(999)
    _close(result.outputs, frozen)
    _close(result.state["views"]["initial"], initial)


def test_backward_through_older_native_observations_preserves_later_forward_model_state():
    model = _model(require_grad=True, stimulated=True)
    callbacks = _plan(model)
    first = _execute(model, "run", plan=callbacks, steps=2)
    last = _execute(model, "longrun", plan=callbacks, state=first.state, steps=3)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    first["views"]["trace"]["v"].square().mean().backward()
    assert (
        torch.isfinite(_gnabar(model).grad).all()
        and _gnabar(model).grad.abs().sum() > 0
    )
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    assert last["energy"]["steps"].item() == 5


@pytest.mark.parametrize("method", ("run", "longrun"))
def test_native_extra_and_registered_intracellular_stimulation_keep_timing(method):
    def extra(model):
        return (
            torch.linspace(-0.1, 0.1, model.v.numel(), dtype=model.v.dtype).reshape(
                model.shape
            ),
            dn.mono_rect(amp=0.01, delay=0, pw=0.04, tau=0.001),
        )

    reference = _model(require_grad=True, stimulated=True)
    native = dn.Recorder(("v", "t"))
    _execute(reference, method, callbacks=[native], extra=extra(reference))
    expected = native.stack("v")
    expected_gradient = torch.autograd.grad(
        expected.square().mean(), _gnabar(reference)
    )[0]
    model = _model(require_grad=True, stimulated=True)
    actual = _execute(model, method, plan=_plan(model), extra=extra(model))
    _close(actual["views"]["trace"]["v"], expected)
    _close(actual["views"]["trace"]["t"], native.stack("t"))
    observed = torch.autograd.grad(
        actual["views"]["trace"]["v"].square().mean(), _gnabar(model)
    )[0]
    _close(observed, expected_gradient)


def test_native_run_precomputed_voltage_length_keeps_callback_horizon_and_duration_remainder():
    model = _model()
    callbacks = _plan(model)
    model._set_duration_remainder(0.4 * DT)
    voltage = torch.zeros((3, *model.shape), dtype=model.v.dtype)
    result = model.run(ve=voltage, tstop=100, dt=DT, functional_callbacks=callbacks)
    assert result["energy"]["steps"].item() == 3
    assert result["views"]["trace"]["v"].shape[0] == 4
    assert model.t.item() == pytest.approx(3 * DT)
    assert model._duration_remainder.item() == pytest.approx(0.4 * DT)


@pytest.mark.parametrize("method", ("run", "longrun"))
def test_ordinary_native_default_returns_and_validation_remain_compatible(method):
    model = _model()
    assert _execute(model, method, steps=1) is None
    callbacks = _plan(model)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    with pytest.raises(ValueError, match="requires functional_callbacks"):
        _execute(model, method, state=object(), steps=0)
    with pytest.raises(TypeError, match="FunctionalCallbacks"):
        _execute(model, method, plan={}, steps=0)
    with pytest.raises(dn.func.FunctionalizationError, match="exact source Population"):
        _execute(_model(), method, plan=callbacks, steps=0)
    with pytest.raises(dn.func.FunctionalizationError, match="dt|timestep"):
        options = {"chunklength": 2} if method == "longrun" else {}
        getattr(model, method)(
            tstop=0, dt=2 * DT, functional_callbacks=callbacks, **options
        )
    with pytest.raises(TypeError, match="FunctionalCallbackState"):
        _execute(model, method, plan=callbacks, state=object(), steps=0)
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)


def test_native_checkpoint_return_shapes_and_scalar_loss_coexist_with_shared_observations():
    model = _model(require_grad=True)
    callbacks = _plan(model)
    result = model.longrun_checkpointed(
        tstop=2 * DT,
        chunklength=2,
        dt=DT,
        callbacks=[LegacyVoltageLoss()],
        functional_callbacks=callbacks,
        safe_checkpoint=True,
        return_final_state=True,
    )
    assert isinstance(result, tuple) and len(result) == 3
    loss, final, observed = result
    assert type(observed) is dn.func.FunctionalCallbackResults
    _close(loss, observed["views"]["trace"]["v"][1:].square().mean((-1, -2)).sum())
    _assert_packet_equal(final, model.state_dict_for_checkpoint())
    (loss + observed["energy"]["energy"].sum()).backward()
    assert torch.isfinite(_gnabar(model).grad).all()


def test_native_bridge_retains_hdf5_rejection_without_starting_recording(
    tmp_path,
    local_hdf5_writer,  # noqa: F811 - pytest injects the imported fixture
):
    model = _model()
    configured = dn.Recorder("v").set_hdf5(
        str(tmp_path / "unsupported.h5"), cache_every=2
    )
    functional, _ = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks({"trace": configured})
    before = _snapshot(model.state_dict_for_checkpoint())
    try:
        for method in ("run", "longrun"):
            with pytest.raises(dn.func.FunctionalizationError, match="HDF5"):
                _execute(model, method, plan=callbacks, steps=0)
        assert configured.rec["v"] == [] and configured.run_number == 0
        _assert_packet_equal(model.state_dict_for_checkpoint(), before)
    finally:
        configured.close()


@pytest.mark.parametrize("method", _METHODS)
def test_native_observations_preserve_shared_stochastic_streams_and_registered_stimulation(
    method,
):
    observed, _, pulse = _stochastic_population()
    baseline, _, baseline_pulse = _stochastic_population()
    baseline.restore_dict_from_checkpoint(
        _snapshot(observed.state_dict_for_checkpoint())
    )
    functional, _ = _lower_stochastic(observed)
    callbacks = functional.make_callbacks(
        {"trace": dn.Recorder(("v", "t")), "energy": EnergyReducer()}
    )
    options = dict(tstop=3 * SHARED_DT, dt=SHARED_DT)
    if method != "run":
        options["chunklength"] = 2
    if method == "longrun_checkpointed":
        options.update(safe_checkpoint=True, restore_state_after_backward=True)
    global_rng = torch.get_rng_state().clone()
    initial_rng = pulse.generator.get_state().clone()
    expected = getattr(baseline, method)(**options)
    result = getattr(observed, method)(**options, functional_callbacks=callbacks)
    if method == "longrun_checkpointed":
        assert expected is None
        assert isinstance(result, tuple) and result[0] is None
        result = result[1]
    else:
        assert expected is None
    assert type(result) is dn.func.FunctionalCallbackResults
    assert result["trace"]["v"].shape == (4, *observed.shape)
    assert result["energy"]["steps"].item() == 3
    _assert_packet_equal(
        observed.state_dict_for_checkpoint(), baseline.state_dict_for_checkpoint()
    )
    _close(pulse.generator.get_state(), baseline_pulse.generator.get_state())
    assert not torch.equal(pulse.generator.get_state(), initial_rng)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.mps
def test_mps_native_pure_callbacks_keep_device_cadence_and_cross_method_continuation():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"

    def model():
        population = dn.SingleCompartment(
            N=1, C=2, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(pas, g=0.001, e=-60.0)
        population.eval()
        population.initialize()
        return population

    with dn.ctx(DEVICE="mps", DTYPE=torch.float32, REQUIRE_GRAD=0):
        actual, reference = model(), model()
        callbacks = _plan(actual, fields=("v", "t"))
        first = _execute(actual, "run", plan=callbacks, steps=2)
        last = _execute(actual, "longrun", plan=callbacks, state=first.state, steps=3)
        recorder = dn.Recorder(("v", "t"))
        _execute(reference, "run", callbacks=[recorder], steps=5)
        _close(
            torch.cat((first["trace"]["v"], last["trace"]["v"])),
            recorder.stack("v")[::2],
        )
        _close(actual.v, reference.v)
        assert last["views"]["trace"]["v"].device.type == "mps"
        assert last["energy"]["steps"].item() == 5
