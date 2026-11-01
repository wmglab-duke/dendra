"""One training workflow crosses native, compiled and portable boundaries."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_continuation_checkpoints import _disk
from tests.test_native_runtime_inputs import DT
from tests.test_native_runtime_inputs import _environment as _environment
from tests.test_native_runtime_inputs import _model, _tapes

pytestmark = pytest.mark.cpu
G = "integrator.mech.mechanisms.leak.g_param"


def _population():
    model, _ = _model()
    model.insert(pas.rename("paired"), g=0.001, e=-65.0)
    model.initialize()
    model.mech.paired.g_param = model.mech.leak.g_param
    model.initialize()
    return model


def _configurations():
    return {"trace": dn.Recorder(("v", "t"), dt=2 * DT)}


def _program(model):
    return dn.compile(
        model,
        dt=DT,
        steps=3,
        chunk_steps=3,
        inference=False,
        trainable="*g_param",
        callbacks=_configurations(),
        backend="eager",
    )


def _drives(model):
    whole = _tapes(model, 10)
    return tuple(
        dn.func.Inputs(
            whole.ve[start:stop].detach().clone().requires_grad_(),
            whole.intra[start:stop].detach().clone().requires_grad_(),
        )
        for start, stop in ((0, 2), (2, 5), (5, 6), (6, 8), (8, 10))
    )


def _reference(drives):
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    callbacks = functional.make_callbacks(_configurations())
    continuation = dn.func.Continuation(
        tensors.state, tensors.parameters, tensors.constants
    )
    stages = []
    with torch.no_grad():
        for inputs, horizon in zip(
            drives,
            (
                {"duration": 2.4 * DT},
                {"steps": 3},
                {"duration": DT},
                {"steps": 2},
                {"duration": 1.6 * DT},
            ),
            strict=True,
        ):
            state, auxiliary = dn.func.run(
                functional,
                continuation,
                inputs=inputs,
                callbacks=callbacks,
                **horizon,
            )
            stages.append((state, auxiliary["callbacks"]))
            continuation = auxiliary["continuation"]
    return stages


def _check_stage(actual_state, actual_results, expected):
    state, results = expected
    _close(actual_state, state)
    _close(actual_results.outputs, results.outputs)
    _close(actual_results.state.carries, results.state.carries)


def _loss(voltage, results):
    return voltage.square().mean() + results["trace"]["v"].square().mean()


def test_population_native_compiled_checkpoint_handoff_preserves_training_lifecycle(
    tmp_path,
):
    source = _population()
    drives = _drives(source)
    original_drives = _snapshot(drives)
    expected = _reference(drives)
    callbacks = dn.func.make_callbacks(source, _configurations(), dt=DT)
    prefix = source.run(duration=2.4 * DT, dt=DT, inputs=drives[0], callbacks=callbacks)
    prefix_voltage = source.v
    prefix_parameter = source.mech.leak.g_param
    frozen_prefix = _snapshot(source.state_dict_for_checkpoint())
    assert source.mech.paired.g_param is prefix_parameter
    assert source._duration_remainder.item() == pytest.approx(0.4 * DT)

    program = _program(source)
    _check_stage(program.initial_state, prefix, expected[0])
    selected = tuple(program.trainable_parameters.values())
    assert len(selected) == 1 and selected[0] is program.parameters[G]
    assert selected[0] is not prefix_parameter
    carry = program.callbacks.restore_state(callbacks.export_state(prefix.state))
    state, first = dn.func.run(program, steps=3, inputs=drives[1], callback_state=carry)
    _check_stage(state, first["callbacks"], expected[1])
    state, second = dn.func.run(
        program, first["continuation"], duration=DT, inputs=drives[2]
    )
    _check_stage(state, second["callbacks"], expected[2])
    saved = second["continuation"]
    assert saved.state["control"]["duration_remainder"].item() == pytest.approx(
        0.4 * DT
    )

    receiver = _program(_population())
    portable = _disk(program.export_continuation(saved), tmp_path / "population.pt")
    restored = receiver.restore_continuation(portable)
    restored_parameter = restored.parameters[G]
    assert restored_parameter.is_leaf and restored_parameter.requires_grad
    assert restored_parameter is not program.parameters[G]
    assert restored.callback_state.plan_token is receiver.callbacks._state_token
    _close(restored.state, saved.state)
    _close(restored.callback_state.carries, saved.callback_state.carries)
    state, third = dn.func.run(receiver, restored, steps=2, inputs=drives[3])
    _check_stage(state, third["callbacks"], expected[3])
    continuation = third["continuation"]
    assert continuation.parameters[G] is restored_parameter
    compiled_prefix = state["integrator"]["v"]
    assert compiled_prefix.grad_fn is not None

    target = _population()
    parameter_bindings = dict(target.named_parameters(remove_duplicate=False))
    target_parameter = target.mech.leak.g_param
    target_callbacks = dn.func.make_callbacks(target, _configurations(), dt=DT)
    native_carry = target_callbacks.restore_state(
        receiver.callbacks.export_state(continuation.callback_state)
    )
    assert dn.func.commit_continuation_(receiver, target, continuation) is target
    for name, value in target.named_parameters(remove_duplicate=False):
        assert value is parameter_bindings[name]
    assert target.mech.paired.g_param is target.mech.leak.g_param is target_parameter
    assert target_parameter.requires_grad and target_parameter.is_leaf
    _close(receiver.functional.extract(target).state, continuation.state)
    assert target._duration_remainder.item() == pytest.approx(0.4 * DT)

    reference = receiver.restore_continuation(
        receiver.export_continuation(continuation)
    )
    reference_inputs = dn.func.Inputs(
        drives[4].ve.detach().clone().requires_grad_(),
        drives[4].intra.detach().clone().requires_grad_(),
    )
    suffix_state, suffix_auxiliary = dn.func.run(
        receiver.functional,
        reference,
        duration=1.6 * DT,
        inputs=reference_inputs,
        callbacks=receiver.callbacks,
    )
    wanted = torch.autograd.grad(
        _loss(suffix_state["integrator"]["v"], suffix_auxiliary["callbacks"]),
        (reference.parameters[G], *reference_inputs),
    )
    native_loss, suffix = target.longrun_checkpointed(
        duration=1.6 * DT,
        chunklength=2,
        dt=DT,
        inputs=drives[4],
        callbacks=target_callbacks,
        callback_state=native_carry,
        safe_checkpoint=True,
    )
    assert native_loss is None
    _check_stage(receiver.functional.extract(target).state, suffix, expected[4])
    _close(suffix.outputs, suffix_auxiliary["callbacks"].outputs)
    _close(suffix.state.carries, suffix_auxiliary["callbacks"].state.carries)
    assert target._duration_remainder.item() == 0.0
    records = (
        prefix,
        first["callbacks"],
        second["callbacks"],
        third["callbacks"],
        suffix,
    )
    for field in ("v", "t"):
        joined = torch.cat(tuple(result["trace"][field] for result in records))
        golden = torch.cat(tuple(result["trace"][field] for _, result in expected))
        _close(joined, golden)
        assert joined.shape[0] == 6
    torch.testing.assert_close(
        torch.cat(tuple(result["trace"]["t"] for result in records)),
        torch.arange(0, 11, 2, dtype=target.dtype(), device=target.device()) * DT,
    )

    frozen_target = _snapshot(target.state_dict_for_checkpoint())
    gradients = torch.autograd.grad(
        _loss(target.v, suffix),
        (
            target_parameter,
            *drives[4],
            restored_parameter,
            program.parameters[G],
            compiled_prefix,
        ),
        allow_unused=True,
    )
    _close(gradients[:3], wanted)
    assert all(
        value.isfinite().all() and value.abs().sum() > 0 for value in gradients[:3]
    )
    assert gradients[3:] == (None, None, None)
    retained = torch.autograd.grad(
        compiled_prefix.square().mean(), (restored_parameter, *drives[3])
    )
    original = torch.autograd.grad(
        prefix_voltage.square().mean(), (prefix_parameter, *drives[0])
    )
    assert all(
        value.isfinite().all() and value.abs().sum() > 0
        for value in (*retained, *original)
    )
    _assert_packet_equal(target.state_dict_for_checkpoint(), frozen_target)
    _assert_packet_equal(source.state_dict_for_checkpoint(), frozen_prefix)
    _assert_packet_equal(drives, original_drives)
