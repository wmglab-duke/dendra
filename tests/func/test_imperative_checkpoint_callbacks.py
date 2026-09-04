from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra.models.callbacks import Recorder as ImperativeRecorder
from dendra.models.mod import hh
from dendra.models.stim.waveform import mono_rect

DT = 0.01
STEPS = 5


class EnergyReducer(dn.func.FunctionalCallback):
    """Keep a differentiable post-step voltage loss in explicit carry."""

    @staticmethod
    def _voltage(state):
        return state["integrator"]["v"]

    def initialize(self, state, auxiliary):
        assert auxiliary is None
        voltage = self._voltage(state)
        return {
            "steps": voltage.new_zeros((), dtype=torch.int64),
            "energy": voltage.new_zeros(voltage.shape[:-1]),
        }, None

    def update(self, carry, state, auxiliary):
        del state
        voltage = auxiliary["v"]
        return {
            "steps": carry["steps"] + 1,
            "energy": carry["energy"] + voltage.square().mean(dim=-1),
        }, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


class LegacyVoltageLoss(dn.callbacks.Callback):
    def post_step_hook(self, model):
        return model.v.square().mean()


def _model(*, require_grad=False, stimulated=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=int(require_grad)):
        model = dn.SingleCompartment(
            N=1,
            C=3,
            v_init=torch.tensor([-64.0, -59.0, -62.0], dtype=torch.float64),
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(hh, gnabar=0.05, gkbar=0.05)
        if stimulated:
            model[:, 0].inject(mono_rect(amp=1.0e-4, delay=0.0, pw=0.04, tau=0.001))
        model.initialize()
    model.train()
    return model


def _callbacks(model, callbacks=None):
    if callbacks is None:
        callbacks = {
            "trace": dn.func.Recorder(["v"]),
            "energy": EnergyReducer(),
            "anomalous": dn.func.AnomalyDetector(),
        }
    functional, _tensors = dn.func.make_functional(model, dt=DT)
    return functional.make_callbacks(callbacks)


def _gnabar(model):
    return model.integrator.mech.mechanisms.hh.gnabar_param


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {name: _clone_tree(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return value


def _assert_tree_close(actual, expected):
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        assert torch.is_tensor(actual) and torch.is_tensor(expected)
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and isinstance(expected, Mapping)
        assert tuple(actual) == tuple(expected)
        for name in actual:
            _assert_tree_close(actual[name], expected[name])
        return
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_tree_close(actual_item, expected_item)
        return
    assert actual == expected


def _checkpoint(
    model,
    callbacks,
    *,
    steps=STEPS,
    callback_state=None,
    legacy_callbacks=None,
    return_final_state=False,
):
    return model.longrun_checkpointed(
        tstop=steps * DT,
        chunklength=2,
        dt=DT,
        callbacks=legacy_callbacks,
        safe_checkpoint=True,
        restore_state_after_backward=True,
        return_final_state=return_final_state,
        functional_callbacks=callbacks,
        functional_callback_state=callback_state,
    )


def test_functional_recorder_bridge_matches_imperative_trace_and_gradient():
    reference = _model(require_grad=True)
    recorder = ImperativeRecorder(["v", "t"])
    reference.run(tstop=STEPS * DT, dt=DT, callbacks=[recorder])
    expected_trace = recorder.stack("v")
    expected_time = recorder.stack("t")
    expected_loss = expected_trace.square().mean()
    expected_loss.backward()
    expected_gradient = _gnabar(reference).grad.detach().clone()

    actual = _model(require_grad=True)
    plan = _callbacks(actual, {"trace": dn.func.Recorder(["v", "t"])})
    native_loss, results = _checkpoint(actual, plan)
    actual_trace = results["trace"]["v"]
    actual_time = results["trace"]["t"]
    actual_loss = actual_trace.square().mean()
    actual_loss.backward()

    assert native_loss is None
    torch.testing.assert_close(actual_trace, expected_trace)
    torch.testing.assert_close(actual_time, expected_time)
    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(_gnabar(actual).grad, expected_gradient)


def test_bridge_preserves_registered_intra_and_runtime_extra_execution():
    def extra(model):
        field = torch.linspace(
            -0.1,
            0.1,
            model.v.numel(),
            dtype=model.dtype(),
            device=model.device(),
        ).reshape(model.shape)
        return field, mono_rect(amp=0.01, delay=0.0, pw=0.04, tau=0.001)

    reference = _model(require_grad=True, stimulated=True)
    reference.longrun_checkpointed(
        tstop=4 * DT,
        chunklength=2,
        dt=DT,
        extra=extra(reference),
        safe_checkpoint=True,
    )
    reference_final = _clone_tree(reference.state_dict_for_checkpoint())
    reference.v.square().mean().backward()
    reference_gradient = _gnabar(reference).grad.detach().clone()

    actual = _model(require_grad=True, stimulated=True)
    plan = _callbacks(actual, {"trace": dn.func.Recorder(["v"])})
    native_loss, results = actual.longrun_checkpointed(
        tstop=4 * DT,
        chunklength=2,
        dt=DT,
        extra=extra(actual),
        safe_checkpoint=True,
        functional_callbacks=plan,
    )
    actual_final = _clone_tree(actual.state_dict_for_checkpoint())
    actual_loss = results["trace"]["v"][-1].square().mean()
    actual_loss.backward()

    assert native_loss is None
    _assert_tree_close(actual_final, reference_final)
    _assert_tree_close(actual.state_dict_for_checkpoint(), actual_final)
    torch.testing.assert_close(_gnabar(actual).grad, reference_gradient)


def test_custom_no_emission_callback_coexists_with_legacy_scalar_loss():
    expected = _model(require_grad=True)
    expected_loss = expected.longrun_checkpointed(
        tstop=STEPS * DT,
        chunklength=2,
        dt=DT,
        callbacks=[LegacyVoltageLoss()],
        safe_checkpoint=True,
    )

    actual = _model(require_grad=True)
    plan = _callbacks(actual)
    actual_loss, results = _checkpoint(
        actual,
        plan,
        legacy_callbacks=[LegacyVoltageLoss()],
    )

    torch.testing.assert_close(actual_loss, expected_loss)
    assert results["energy"]["steps"].item() == STEPS
    torch.testing.assert_close(
        results["energy"]["energy"],
        results["trace"]["v"][1:].square().mean(dim=-1).sum(dim=0),
    )
    assert not results["anomalous"].any()

    (actual_loss + results["energy"]["energy"].sum()).backward()
    assert torch.isfinite(_gnabar(actual).grad)


def test_checkpointed_return_shape_is_additive_only_when_plan_is_present():
    legacy = _model(require_grad=True)
    legacy_loss = legacy.longrun_checkpointed(
        tstop=DT,
        chunklength=1,
        dt=DT,
        callbacks=[LegacyVoltageLoss()],
        safe_checkpoint=True,
    )
    assert torch.is_tensor(legacy_loss)

    legacy_with_state = _model(require_grad=True)
    unchanged = legacy_with_state.longrun_checkpointed(
        tstop=DT,
        chunklength=1,
        dt=DT,
        callbacks=[LegacyVoltageLoss()],
        safe_checkpoint=True,
        return_final_state=True,
    )
    assert isinstance(unchanged, tuple) and len(unchanged) == 2
    assert torch.is_tensor(unchanged[0])
    assert isinstance(unchanged[1], Mapping)

    bridged = _model()
    plan = _callbacks(bridged)
    with_plan = _checkpoint(bridged, plan, steps=1)
    assert isinstance(with_plan, tuple) and len(with_plan) == 2
    assert with_plan[0] is None
    assert isinstance(with_plan[1], dn.func.FunctionalCallbackResults)

    bridged_with_state = _model()
    state_plan = _callbacks(bridged_with_state)
    expanded = _checkpoint(
        bridged_with_state,
        state_plan,
        steps=1,
        return_final_state=True,
    )
    assert isinstance(expanded, tuple) and len(expanded) == 3
    assert expanded[0] is None
    assert isinstance(expanded[1], Mapping)
    assert isinstance(expanded[2], dn.func.FunctionalCallbackResults)


def test_imperative_bridge_rejects_unbound_state_and_another_source_model():
    source = _model()
    plan = _callbacks(source)
    _loss, results = _checkpoint(source, plan, steps=0)

    with pytest.raises(ValueError, match="requires functional_callbacks"):
        source.longrun_checkpointed(
            tstop=0.0,
            chunklength=1,
            dt=DT,
            functional_callback_state=results.state,
        )

    other = _model()
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="exact source Population",
    ):
        _checkpoint(other, plan, steps=0)


def test_fresh_and_resumed_zero_step_callback_lifecycle():
    model = _model()
    plan = _callbacks(model)
    native_loss, fresh = _checkpoint(model, plan, steps=0)

    assert native_loss is None
    assert fresh["trace"]["v"].shape == (1, 1, 3)
    assert fresh["energy"]["steps"].item() == 0
    assert not fresh["anomalous"].any()

    native_loss, resumed = _checkpoint(
        model,
        plan,
        steps=0,
        callback_state=fresh.state,
    )
    assert native_loss is None
    assert resumed["trace"]["v"].shape == (0, 1, 3)
    _assert_tree_close(resumed.state, fresh.state)
    assert model.t.item() == pytest.approx(0.0)


def test_segmented_imperative_bridge_resume_matches_one_execution():
    segmented = _model()
    segmented_plan = _callbacks(segmented)
    _first_loss, first = _checkpoint(segmented, segmented_plan, steps=2)
    _second_loss, second = _checkpoint(
        segmented,
        segmented_plan,
        steps=3,
        callback_state=first.state,
    )

    complete = _model()
    complete_plan = _callbacks(complete)
    _complete_loss, expected = _checkpoint(complete, complete_plan, steps=5)

    resumed_trace = torch.cat((first["trace"]["v"], second["trace"]["v"]))
    torch.testing.assert_close(resumed_trace, expected["trace"]["v"])
    _assert_tree_close(second["energy"], expected["energy"])
    _assert_tree_close(second["anomalous"], expected["anomalous"])
    torch.testing.assert_close(segmented.v, complete.v)
    assert segmented.t.item() == pytest.approx(complete.t.item())


def test_threshold_callback_bridge_preserves_raster_and_count_carry():
    segmented = _model()
    segmented_plan = _callbacks(
        segmented,
        {
            "raster": dn.func.Raster(threshold=-100.0, node_check=[0, -1]),
            "count": dn.func.APCount(threshold=-100.0, node_check=[0, -1]),
        },
    )
    _first_loss, first = _checkpoint(segmented, segmented_plan, steps=2)
    _second_loss, second = _checkpoint(
        segmented,
        segmented_plan,
        steps=3,
        callback_state=first.state,
    )

    complete = _model()
    complete_plan = _callbacks(
        complete,
        {
            "raster": dn.func.Raster(threshold=-100.0, node_check=[0, -1]),
            "count": dn.func.APCount(threshold=-100.0, node_check=[0, -1]),
        },
    )
    _complete_loss, expected = _checkpoint(complete, complete_plan, steps=5)

    resumed_raster = torch.cat((first["raster"], second["raster"]))
    torch.testing.assert_close(resumed_raster, expected["raster"])
    torch.testing.assert_close(second["count"], expected["count"])
    torch.testing.assert_close(expected["count"], expected["raster"].sum(dim=0))
    assert torch.all(expected["raster"][0])
    assert not torch.any(expected["raster"][1:])


def test_functional_anomaly_detector_observes_nonfinite_post_step_voltage():
    model = _model()
    with torch.no_grad():
        model.v[..., 0] = torch.nan
    plan = _callbacks(model, {"anomalous": dn.func.AnomalyDetector()})
    _loss, results = _checkpoint(model, plan, steps=1)
    assert results["anomalous"].all()


def test_backward_through_only_functional_output_restores_forward_final_model():
    model = _model(require_grad=True)
    plan = _callbacks(model, {"energy": EnergyReducer()})
    native_loss, results = _checkpoint(model, plan, steps=4)
    forward_final = _clone_tree(model.state_dict_for_checkpoint())

    assert native_loss is None
    results["energy"]["energy"].sum().backward()

    _assert_tree_close(model.state_dict_for_checkpoint(), forward_final)
    assert _gnabar(model).grad is not None
    assert torch.isfinite(_gnabar(model).grad)


def test_resumed_callback_backward_restores_latest_forward_final_model():
    model = _model(require_grad=True)
    plan = _callbacks(model, {"energy": EnergyReducer()})

    first_loss, first = _checkpoint(model, plan, steps=2)
    first_forward_final = _clone_tree(model.state_dict_for_checkpoint())
    second_loss, second = _checkpoint(
        model,
        plan,
        steps=3,
        callback_state=first.state,
    )
    second_forward_final = _clone_tree(model.state_dict_for_checkpoint())

    assert first_loss is second_loss is None
    assert first_forward_final["t"].item() == pytest.approx(2 * DT)
    assert second_forward_final["t"].item() == pytest.approx(5 * DT)
    second["energy"]["energy"].sum().backward()

    _assert_tree_close(model.state_dict_for_checkpoint(), second_forward_final)
    assert _gnabar(model).grad is not None
    assert torch.isfinite(_gnabar(model).grad)


def test_backward_through_older_result_preserves_later_zero_step_state():
    model = _model(require_grad=True)
    plan = _callbacks(model, {"energy": EnergyReducer()})
    first_loss, first = _checkpoint(model, plan, steps=3)

    zero_loss, zero = model.longrun_checkpointed(
        tstop=0.5 * DT,
        chunklength=2,
        dt=DT,
        safe_checkpoint=True,
        restore_state_after_backward=True,
        functional_callbacks=plan,
        functional_callback_state=first.state,
    )
    latest_state = _clone_tree(model.state_dict_for_checkpoint())

    assert first_loss is zero_loss is None
    assert zero["energy"]["steps"].item() == 3
    assert latest_state["t"].item() == pytest.approx(3 * DT)
    assert latest_state["duration_remainder"].item() == pytest.approx(0.5 * DT)
    first["energy"]["energy"].sum().backward()

    _assert_tree_close(model.state_dict_for_checkpoint(), latest_state)
    assert _gnabar(model).grad is not None
    assert torch.isfinite(_gnabar(model).grad)


def test_higher_order_functional_output_gradients_preserve_forward_final_model():
    model = _model(require_grad=True)
    plan = _callbacks(model, {"energy": EnergyReducer()})
    native_loss, results = _checkpoint(model, plan, steps=4)
    forward_final = _clone_tree(model.state_dict_for_checkpoint())
    parameter = _gnabar(model)
    objective = results["energy"]["energy"].sum()

    assert native_loss is None
    first_gradient = torch.autograd.grad(
        objective,
        parameter,
        create_graph=True,
    )[0]
    _assert_tree_close(model.state_dict_for_checkpoint(), forward_final)

    second_gradient = torch.autograd.grad(first_gradient, parameter)[0]
    _assert_tree_close(model.state_dict_for_checkpoint(), forward_final)
    assert torch.isfinite(first_gradient)
    assert torch.isfinite(second_gradient)
