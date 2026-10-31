"""Population execution hosts share explicit step and physical-time horizons."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import dendra as dn
from dendra.models.backend import Backend
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_native_handoff import _capture, _unchanged
from tests.func.test_runtime_noise import DT as NOISE_DT
from tests.func.test_runtime_noise import _population as _noisy_population
from tests.test_native_runtime_inputs import DT
from tests.test_native_runtime_inputs import _environment as _environment
from tests.test_native_runtime_inputs import _model, _reference, _tapes

pytestmark = pytest.mark.cpu
_METHODS = ("run", "longrun", "longrun_checkpointed")


def _execute(model, method, *, dt=DT, **arguments):
    if dt is not None:
        arguments["dt"] = dt
    if method != "run":
        arguments["chunklength"] = 3
    if method == "longrun_checkpointed":
        arguments["safe_checkpoint"] = True
    return getattr(model, method)(**arguments)


def _slice(inputs, start, stop):
    return dn.func.Inputs(inputs.ve[start:stop], inputs.intra[start:stop])


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("horizon", ("steps", "duration"))
def test_explicit_population_horizons_match_functional_state_and_all_vjps(
    method, horizon
):
    model, _ = _model()
    inputs = _tapes(model, 5)
    functional, expected = _reference(model, inputs, 5)
    targets = model.mech.leak.g_param, inputs.ve, inputs.intra
    wanted = torch.autograd.grad(expected["integrator"]["v"].square().mean(), targets)
    original_inputs = _snapshot(inputs)
    result = _execute(
        model,
        method,
        inputs=inputs,
        **{horizon: 5 if horizon == "steps" else 5 * DT},
    )
    assert result is None
    _close(functional.extract(model).state, expected)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    actual = torch.autograd.grad(model.v.square().mean(), targets)
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    _assert_packet_equal(inputs, original_inputs)


@pytest.mark.parametrize("method", _METHODS)
def test_exact_steps_preserve_pending_duration_and_checkpoint_continuation(
    method,
):
    model, _ = _model()
    source, _ = _model()
    inputs = _tapes(model, 3)
    functional, expected = _reference(source, inputs, 3)
    empty = _slice(inputs, 0, 0)
    _execute(model, method, duration=0.4 * DT, inputs=empty)
    remainder = model._duration_remainder
    version = remainder._version
    assert remainder.item() == pytest.approx(0.4 * DT)
    for steps in (0, 2):
        _execute(model, method, steps=steps, inputs=_slice(inputs, 0, steps))
        if method != "longrun_checkpointed":
            assert model._duration_remainder is remainder
        assert remainder._version == version
        assert remainder.item() == pytest.approx(0.4 * DT)
        assert model._duration_remainder.item() == remainder.item()
    checkpoint = _snapshot(model.state_dict_for_checkpoint())
    restored, _ = _model()
    restored.restore_dict_from_checkpoint(checkpoint)
    for owner in (model, restored):
        _execute(owner, method, duration=0.6 * DT, inputs=_slice(inputs, 2, 3))
        _close(functional.extract(owner).state, expected)
        assert owner._duration_remainder.item() == 0.0


@pytest.mark.parametrize("method", _METHODS)
def test_population_horizons_keep_legacy_positional_arguments(method):
    legacy, _ = _model()
    canonical, _ = _model()
    if method == "run":
        ve = _tapes(legacy, 5).ve
        legacy.run(ve, None, 99 * DT, DT)
        canonical.run(ve=ve, steps=5, dt=DT)
    else:
        options = {"safe_checkpoint": True} if method.endswith("checkpointed") else {}
        getattr(legacy, method)(5 * DT, 3, DT, **options)
        _execute(canonical, method, duration=5 * DT)
    _assert_packet_equal(
        legacy.state_dict_for_checkpoint(), canonical.state_dict_for_checkpoint()
    )


@pytest.mark.parametrize("horizon", ("steps", "duration"))
def test_canonical_run_horizons_validate_coercive_legacy_voltage_tapes(horizon):
    model, _ = _model()
    inputs = _tapes(model, 3)
    legacy_ve = inputs.ve.detach().float().requires_grad_()
    functional, expected = _reference(
        model, dn.func.Inputs(legacy_ve.to(model.dtype()), inputs.intra), 3
    )
    targets = model.mech.leak.g_param, legacy_ve, inputs.intra
    wanted = torch.autograd.grad(expected["integrator"]["v"].square().mean(), targets)
    arguments = {horizon: 3 if horizon == "steps" else 3 * DT}
    model.run(
        ve=legacy_ve, inputs=dn.func.Inputs(intra=inputs.intra), dt=DT, **arguments
    )
    _close(functional.extract(model).state, expected)
    _close(torch.autograd.grad(model.v.square().mean(), targets), wanted)

    rejected, _ = _model()
    saved = _capture(rejected)
    with pytest.raises(ValueError, match="steps|length|expects|horizon"):
        rejected.run(ve=legacy_ve[:-1], dt=DT, **arguments)
    _unchanged(rejected, saved)


@pytest.mark.parametrize("method", _METHODS)
def test_population_default_dt_uses_backend_default(method, monkeypatch):
    model, _ = _model()
    reference, _ = _model()
    default_dt = DT / 2
    monkeypatch.setattr(Backend, "dt", default_dt)
    reference.step()
    reference.step()
    _execute(model, method, dt=None, steps=2)
    torch.testing.assert_close(model.v, reference.v)
    assert model.t.item() == pytest.approx(2 * default_dt)


@pytest.mark.parametrize("method", _METHODS)
def test_canonical_input_lengths_use_the_accepted_horizon_before_state_changes(method):
    model, _ = _model()
    model.run(duration=0.4 * DT, dt=DT)
    inputs = _tapes(model, 2)
    saved = _capture(model)
    for arguments in ({"steps": 3}, {"duration": 2.6 * DT}):
        with pytest.raises(ValueError, match="2 steps.*expects 3"):
            _execute(model, method, inputs=inputs, **arguments)
        _unchanged(model, saved)


@pytest.mark.parametrize("method", _METHODS)
def test_population_horizon_errors_precede_waveform_evaluation_and_preserve_state(
    method, monkeypatch
):
    model, waveform = _model(injection="solver")
    _execute(model, method, duration=0.4 * DT)
    saved = _capture(model)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "an invalid horizon must reject before waveform evaluation"
        )

    monkeypatch.setattr(waveform, "forward", forbidden)
    for arguments, error, message in (
        ({"steps": True}, ValueError, "steps"),
        ({"steps": np.int64(2)}, ValueError, "steps"),
        ({"steps": 1.5}, ValueError, "steps"),
        ({"steps": -1}, ValueError, "steps"),
        ({"duration": True}, TypeError, "duration"),
        ({"duration": -DT}, ValueError, "duration"),
        ({"duration": float("nan")}, ValueError, "duration"),
        ({"duration": float("inf")}, ValueError, "duration"),
        ({"steps": 2, "duration": DT}, ValueError, "steps|duration"),
        ({"steps": 2, "tstop": DT}, ValueError, "tstop"),
        ({"duration": DT, "tstop": DT}, ValueError, "tstop"),
    ):
        with pytest.raises(error, match=message):
            _execute(model, method, **arguments)
        _unchanged(model, saved)


@pytest.mark.parametrize("method", _METHODS)
def test_missing_population_horizon_preserves_existing_required_arguments(method):
    model, _ = _model()
    saved = _capture(model)
    error = ValueError if method == "run" else TypeError
    with pytest.raises(error, match="tstop|steps|duration"):
        _execute(model, method)
    _unchanged(model, saved)
    if method != "run":
        with pytest.raises((TypeError, ValueError), match="chunklength"):
            getattr(model, method)(steps=0, dt=DT)
        _unchanged(model, saved)


def test_mixed_step_duration_horizons_restore_noise_rng_and_pending_time():
    model, reference, restored = (_noisy_population().train() for _ in range(3))
    model.run(duration=0.4 * NOISE_DT, dt=NOISE_DT)
    frozen = _snapshot(model.state_dict_for_checkpoint())
    model.longrun_checkpointed(
        steps=0, chunklength=2, dt=NOISE_DT, safe_checkpoint=True
    )
    _assert_packet_equal(model.state_dict_for_checkpoint(), frozen)
    model.longrun(steps=2, chunklength=2, dt=NOISE_DT)
    checkpoint = _snapshot(model.state_dict_for_checkpoint())
    restored.restore_dict_from_checkpoint(checkpoint)
    model.longrun_checkpointed(
        duration=0.6 * NOISE_DT, chunklength=2, dt=NOISE_DT, safe_checkpoint=True
    )
    restored.run(duration=0.6 * NOISE_DT, dt=NOISE_DT)
    reference.run(steps=3, dt=NOISE_DT)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    _assert_packet_equal(
        restored.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )


def test_population_horizon_interleaving_preserves_pure_recorder_phase_and_gradients():
    model, _ = _model()
    inputs = _tapes(model, 7)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    configurations = {"trace": dn.Recorder(("v", "t"), dt=2 * DT)}
    callbacks = dn.func.make_callbacks(model, configurations, dt=DT)
    expected, auxiliary = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        inputs=inputs,
        steps=7,
        callbacks=functional.make_callbacks(configurations),
    )
    targets = model.mech.leak.g_param, inputs.ve, inputs.intra
    wanted = torch.autograd.grad(
        auxiliary["callbacks"]["trace"]["v"].square().sum(), targets
    )
    first = model.run(
        duration=0.4 * DT, dt=DT, inputs=_slice(inputs, 0, 0), callbacks=callbacks
    )
    middle = model.longrun(
        steps=3,
        chunklength=2,
        dt=DT,
        inputs=_slice(inputs, 0, 3),
        callbacks=callbacks,
        callback_state=first.state,
    )
    native_loss, last = model.longrun_checkpointed(
        duration=3.6 * DT,
        chunklength=3,
        dt=DT,
        inputs=_slice(inputs, 3, 7),
        callbacks=callbacks,
        callback_state=middle.state,
        safe_checkpoint=True,
    )
    assert native_loss is None
    trace = {
        name: torch.cat(
            tuple(result["trace"][name] for result in (first, middle, last))
        )
        for name in ("v", "t")
    }
    _close(trace, auxiliary["callbacks"]["trace"])
    _close(last.state.carries, auxiliary["callbacks"].state.carries)
    _close(functional.extract(model).state, expected)
    actual = torch.autograd.grad(trace["v"].square().sum(), targets)
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_horizons_preserve_fractional_time_and_checkpointed_source_input_vjps():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        model, _ = _model()
        inputs = _tapes(model, 4)
        _, expected = _reference(model, inputs, 4)
        targets = model.mech.leak.g_param, inputs.ve, inputs.intra
        wanted = torch.autograd.grad(
            expected["integrator"]["v"].square().mean(), targets
        )
        original_inputs = _snapshot(inputs)
        model.run(duration=0.4 * DT, dt=DT, inputs=_slice(inputs, 0, 0))
        pending = model._duration_remainder
        version = pending._version
        model.longrun(steps=3, chunklength=2, dt=DT, inputs=_slice(inputs, 0, 3))
        assert model._duration_remainder is pending
        assert pending._version == version
        assert pending.item() == pytest.approx(0.4 * DT)
        model.longrun_checkpointed(
            duration=0.6 * DT,
            chunklength=2,
            dt=DT,
            inputs=_slice(inputs, 3, 4),
            safe_checkpoint=True,
        )
        assert model._duration_remainder.item() == 0.0
        torch.testing.assert_close(model.t, model.t.new_tensor(4 * DT))
        torch.testing.assert_close(
            model.v, expected["integrator"]["v"], rtol=3e-5, atol=3e-5
        )
        saved = _snapshot(model.state_dict_for_checkpoint())
        actual = torch.autograd.grad(model.v.square().mean(), targets)
        for value, reference in zip(actual, wanted, strict=True):
            torch.testing.assert_close(value, reference, rtol=3e-4, atol=3e-6)
            assert value.device.type == "mps" and value.isfinite().all()
            assert value.abs().sum() > 0
        _assert_packet_equal(model.state_dict_for_checkpoint(), saved)
        _assert_packet_equal(inputs, original_inputs)
