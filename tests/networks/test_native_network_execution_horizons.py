"""Native Network horizons share exact steps and retained physical duration."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _snapshot
from tests.func.test_network import DT, DTYPE, _close, _mixed_network
from tests.func.test_network import _native_loss as _native_mixed_loss
from tests.func.test_network_rollout import _driven_network
from tests.func.test_stimulation_rng_sharing import DT as STOCHASTIC_DT
from tests.func.test_stimulation_rng_sharing import _network as _stochastic_network
from tests.networks.test_native_network_runtime_inputs import _drives, _inputs, _loss

pytestmark = pytest.mark.cpu
_METHODS = ("run", "longrun", "longrun_checkpointed")


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    rng = torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def _call(model, method, **arguments):
    if method != "run":
        arguments["chunklength"] = 3
    if method == "longrun_checkpointed":
        arguments["safe_checkpoint"] = True
    return getattr(model, method)(**arguments)


def _callbacks(model):
    return dn.func.make_callbacks(
        model,
        {"trace": dn.Recorder(("left.v", "right.v", "t"), dt=2 * DT)},
        dt=DT,
    )


def _result(value, method):
    if method == "longrun_checkpointed":
        loss, value = value
        assert loss is None
    assert isinstance(value, dn.func.FunctionalCallbackResults)
    return value


def _recorded_loss(result):
    return sum(result["trace"][name].square().mean() for name in ("left.v", "right.v"))


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("horizon", ("steps", "duration"))
def test_network_canonical_horizons_match_legacy_tapes_callbacks_and_vjps(
    method, horizon
):
    model, reference = _driven_network(), _driven_network()
    conductance = model.left.mech.pas.g_param.requires_grad_()
    reference_conductance = reference.left.mech.pas.g_param.requires_grad_()
    ve, intra = _drives(model, 7)
    reference_ve = ve.detach().clone().requires_grad_()
    reference_intra = intra.detach().clone().requires_grad_()
    arguments = {horizon: 7 if horizon == "steps" else 7 * DT}
    actual = _result(
        _call(
            model,
            method,
            **arguments,
            inputs=_inputs(ve, intra),
            callbacks=_callbacks(model),
        ),
        method,
    )
    expected = _result(
        _call(
            reference,
            method,
            tstop=7 * DT,
            inputs=_inputs(reference_ve, reference_intra),
            callbacks=_callbacks(reference),
        ),
        method,
    )
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    _close(actual.outputs, expected.outputs)
    _close(actual.state.carries, expected.state.carries)
    assert actual["trace"]["t"].tolist() == [0.0, 2 * DT, 4 * DT, 6 * DT]
    saved = _snapshot(model.state_dict_for_checkpoint())
    observed = torch.autograd.grad(
        _recorded_loss(actual) + _loss(cell.v for cell in model.populations.values()),
        (ve, intra, conductance),
    )
    wanted = torch.autograd.grad(
        _recorded_loss(expected)
        + _loss(cell.v for cell in reference.populations.values()),
        (reference_ve, reference_intra, reference_conductance),
    )
    _close(observed, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in observed)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("steps", (0, 5))
def test_network_exact_steps_bypass_duration_budget_and_preserve_pending_time(
    method, steps, monkeypatch
):
    model, reference = _driven_network(), _driven_network()
    for network in (model, reference):
        network.run(duration=0.25 * DT)
    pending = model._duration_remainder
    version = pending._version
    snapshot = pending.clone()

    def unexpected(*args):
        raise AssertionError(
            "exact steps must not resolve or publish a duration budget"
        )

    monkeypatch.setattr(model, "_duration_budget", unexpected)
    monkeypatch.setattr(model, "_set_duration_remainder", unexpected)
    zeros = torch.zeros((steps, *model.left.shape), dtype=DTYPE)
    assert (
        _call(model, method, steps=steps, inputs={"left": dn.func.Inputs(intra=zeros)})
        is None
    )
    for _ in range(steps):
        reference.step()
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    assert model._duration_remainder.item() == 0.25 * DT
    assert pending._version == version
    _close(pending, snapshot)
    if method != "longrun_checkpointed" or steps == 0:
        assert model._duration_remainder is pending


@pytest.mark.parametrize("method", _METHODS)
def test_network_fractional_duration_interleaves_steps_and_runtime_checkpoint_continuation(
    method,
):
    model, receiver, reference = (_driven_network() for _ in range(3))
    ve, intra = _drives(model, 4)
    tapes = _inputs(ve, intra)
    callbacks = _callbacks(model)
    model.run(duration=0.25 * DT)
    first = _result(
        _call(
            model,
            method,
            steps=3,
            inputs={
                name: packet._replace(ve=packet.ve[:3], intra=packet.intra[:3])
                for name, packet in tapes.items()
            },
            callbacks=callbacks,
        ),
        method,
    )
    assert model._duration_remainder.item() == 0.25 * DT
    receiver.restore_dict_from_checkpoint(_snapshot(model.state_dict_for_checkpoint()))
    receiver_callbacks = _callbacks(receiver)
    receiver_state = receiver_callbacks.restore_state(
        callbacks.export_state(first.state)
    )
    suffix = {
        name: packet._replace(ve=packet.ve[3:], intra=packet.intra[3:])
        for name, packet in tapes.items()
    }
    actual = _result(
        _call(
            model,
            method,
            duration=0.75 * DT,
            inputs=suffix,
            callbacks=callbacks,
            callback_state=first.state,
        ),
        method,
    )
    resumed = _result(
        _call(
            receiver,
            method,
            duration=0.75 * DT,
            inputs=suffix,
            callbacks=receiver_callbacks,
            callback_state=receiver_state,
        ),
        method,
    )
    reference.run(4 * DT, inputs=tapes)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), receiver.state_dict_for_checkpoint()
    )
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    _close(actual.outputs, resumed.outputs)
    _close(actual.state.carries, resumed.state.carries)
    assert actual["trace"]["t"].tolist() == [4 * DT]
    assert model._duration_remainder.item() == 0.0


@pytest.mark.parametrize("method", _METHODS)
def test_network_horizon_validation_precedes_sampling_and_preserves_runtime(method):
    model, pulses = _stochastic_network()
    generator = pulses[0].generator
    state = _snapshot(model.state_dict_for_checkpoint())
    rng = generator.get_state().clone()
    invalid = (
        ({"steps": True}, ValueError, "steps must be a non-negative integer"),
        ({"steps": -1}, ValueError, "steps must be a non-negative integer"),
        ({"steps": 1.5}, ValueError, "steps must be a non-negative integer"),
        (
            {"steps": torch.tensor(2)},
            ValueError,
            "steps must be a non-negative integer",
        ),
        (
            {"steps": 1, "duration": DT},
            ValueError,
            "steps and duration are mutually exclusive",
        ),
        ({"tstop": DT, "steps": 1}, ValueError, "tstop cannot be combined"),
        ({"tstop": DT, "duration": DT}, ValueError, "tstop cannot be combined"),
        ({"duration": -1.0}, ValueError, "duration"),
        ({"duration": True}, TypeError, "duration"),
        ({"duration": float("nan")}, ValueError, "duration"),
        ({"duration": float("inf")}, ValueError, "duration"),
        ({}, TypeError, "tstop"),
    )
    for arguments, error, message in invalid:
        with pytest.raises(error, match=message):
            _call(model, method, **arguments)
        _assert_packet_equal(model.state_dict_for_checkpoint(), state)
        _close(generator.get_state(), rng)


@pytest.mark.parametrize("method", _METHODS)
def test_network_canonical_horizon_is_authoritative_for_input_length(method):
    model = _driven_network()
    model.run(duration=0.75 * DT)
    state = _snapshot(model.state_dict_for_checkpoint())
    three = torch.zeros((3, *model.left.shape), dtype=DTYPE)
    with pytest.raises(ValueError, match="steps"):
        _call(model, method, steps=2, inputs={"left": dn.func.Inputs(intra=three)})
    _assert_packet_equal(model.state_dict_for_checkpoint(), state)
    with pytest.raises(ValueError, match="steps"):
        _call(
            model,
            method,
            duration=1.25 * DT,
            inputs={"left": dn.func.Inputs(intra=three)},
        )
    _assert_packet_equal(model.state_dict_for_checkpoint(), state)
    with pytest.raises(TypeError, match="tstop"):
        _call(model, method, inputs={"left": dn.func.Inputs(intra=three)})
    _assert_packet_equal(model.state_dict_for_checkpoint(), state)


@pytest.mark.parametrize("method", _METHODS)
def test_network_exact_steps_keep_joint_poisson_noise_sde_rng_order(method):
    model, pulses = _stochastic_network()
    reference, reference_pulses = _stochastic_network()
    generator = pulses[0].generator
    global_rng = torch.get_rng_state().clone()
    for network in (model, reference):
        network.run(duration=0.25 * STOCHASTIC_DT)
    _call(model, method, steps=5)
    _call(reference, method, tstop=5 * STOCHASTIC_DT)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    _call(model, method, duration=0.75 * STOCHASTIC_DT)
    _call(reference, method, tstop=0.75 * STOCHASTIC_DT)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    assert model._clock_step.item() == 6
    assert all(pulse.generator is generator for pulse in pulses)
    _close(generator.get_state(), reference_pulses[0].generator.get_state())
    _close(torch.get_rng_state(), global_rng)


def test_network_exact_checkpoint_horizon_preserves_active_delivery_and_weight_vjps():
    model, reference = _mixed_network(), _mixed_network()
    current = (
        torch.linspace(0.5e-8, 1.2e-8, 8 * model.cell.v.numel(), dtype=DTYPE)
        .reshape(8, *model.cell.shape)
        .requires_grad_()
    )
    reference_current = current.detach().clone().requires_grad_()
    plans = [
        dn.func.make_callbacks(
            network,
            {"trace": dn.Recorder(("cell.v", "cell.mech.relay.level", "t"), dt=2 * DT)},
            dt=DT,
        )
        for network in (model, reference)
    ]
    slots = tuple(
        (collection, name)
        for collection in ("synapses", "continuous_synapses")
        for name in getattr(model, collection)
    )
    weights = tuple(
        tuple(
            getattr(network, collection)[name].weight.weights[0].param.rho
            for collection, name in slots
        )
        for network in (model, reference)
    )
    for network in (model, reference):
        network.run(duration=0.4 * DT)
    first = _result(
        model.longrun_checkpointed(
            steps=7,
            chunklength=3,
            inputs={"cell": dn.func.Inputs(intra=current[:7])},
            callbacks=plans[0],
            safe_checkpoint=True,
        ),
        "longrun_checkpointed",
    )
    expected_first = reference.run(
        7 * DT,
        inputs={"cell": dn.func.Inputs(intra=reference_current[:7])},
        callbacks=plans[1],
    )
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    _close(first.outputs, expected_first.outputs)
    _close(first.state.carries, expected_first.state.carries)
    assert model._duration_remainder.item() == pytest.approx(0.4 * DT)
    assert model.cell.mech.relay.ledger.abs().sum() > 0
    assert any(
        connection.delivery_buffer.abs().sum() > 0
        for connection in model.continuous_synapses.values()
    )
    final = _result(
        model.longrun_checkpointed(
            duration=0.6 * DT,
            chunklength=3,
            inputs={"cell": dn.func.Inputs(intra=current[7:])},
            callbacks=plans[0],
            callback_state=first.state,
            safe_checkpoint=True,
        ),
        "longrun_checkpointed",
    )
    expected_final = reference.run(
        0.6 * DT,
        inputs={"cell": dn.func.Inputs(intra=reference_current[7:])},
        callbacks=plans[1],
        callback_state=expected_first.state,
    )
    _assert_packet_equal(
        model.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    _close(final.outputs, expected_final.outputs)
    _close(final.state.carries, expected_final.state.carries)
    assert final["trace"]["t"].tolist() == [8 * DT]
    assert model._duration_remainder.item() == 0.0
    saved = _snapshot(model.state_dict_for_checkpoint())
    actual = torch.autograd.grad(
        _native_mixed_loss(model)
        + first["trace"]["cell.v"].square().mean()
        + final["trace"]["cell.v"].square().mean(),
        (current, *weights[0]),
    )
    expected = torch.autograd.grad(
        _native_mixed_loss(reference)
        + expected_first["trace"]["cell.v"].square().mean()
        + expected_final["trace"]["cell.v"].square().mean(),
        (reference_current, *weights[1]),
    )
    _close(actual, expected)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    _assert_packet_equal(model.state_dict_for_checkpoint(), saved)


@pytest.mark.parametrize("method", ("longrun", "longrun_checkpointed"))
def test_network_canonical_calls_still_require_a_positive_chunklength(method):
    model = _driven_network()
    snapshot = _snapshot(model.state_dict_for_checkpoint())
    for value in (None, True, 0, -1, 1.5):
        error = (
            TypeError
            if method == "longrun_checkpointed"
            and (isinstance(value, bool) or not isinstance(value, int))
            else ValueError
        )
        with pytest.raises(error, match="chunklength must be a positive integer"):
            getattr(model, method)(steps=0, chunklength=value)
        _assert_packet_equal(model.state_dict_for_checkpoint(), snapshot)
