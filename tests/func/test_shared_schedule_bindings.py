"""Dual reference declarations retain the existing explicit schedule packets."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import expsyn
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _close, _snapshot
from tests.func.test_netstim_schedules import _carry
from tests.networks.test_native_shared_schedule_bindings import (
    DT,
    _bind,
    _owner,
    _schedule,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64, IMEM=False),
    ):
        yield


def _configured(kind):
    native = _owner(kind)
    times = torch.tensor([0.15, 0.25], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([2.0, -0.5, 0.0], dtype=torch.float64, requires_grad=True)
    _bind(native, times, weights)
    _schedule(native, [0, 0, 1, 2], time_idx=[0, 1, 0, 1], weight_idx=[0, 1, 2, 0])
    return native, times, weights


def _netcon_trajectory(native, *, compiled=False):
    projection, tensors = dn.func.make_functional_netcon(native)
    values = torch.zeros(
        projection.source_shape,
        device=projection.source_device,
        dtype=projection.source_dtype,
    )

    def transition(state, schedule):
        return projection.step(tensors.weights, state, values, schedule=schedule)

    advance = (
        torch.compile(transition, backend="aot_eager", fullgraph=True)
        if compiled
        else transition
    )
    state, actual_loss, native_loss = tensors.state, 0, 0
    initial = _snapshot(tensors)
    for index in range(6):
        state, payload = advance(state, tensors.schedule)
        previous = native.syn.g.clone()
        native.advance()
        _close(state, native.state_dict_for_checkpoint())
        _close(payload, native.syn.g - previous)
        actual_loss = actual_loss + (index + 1) * payload.square().sum()
        native_loss = (
            native_loss + (index + 1) * (native.syn.g - previous).square().sum()
        )
    _assert_packet_equal(tensors, initial)
    return tensors, actual_loss, native_loss


@pytest.mark.parametrize("kind", ("netstim", "netcon"))
@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "aot_fullgraph"))
def test_dual_schedule_packets_match_native_carry_and_both_source_gradients(
    kind, compiled
):
    native, times, weights = _configured(kind)
    if kind == "netcon":
        tensors, actual_loss, native_loss = _netcon_trajectory(
            native, compiled=compiled
        )
    else:
        plan, tensors = dn.func.make_functional_netstim(native)

        def transition(state, time, draw, schedule):
            return plan.step(
                tensors.parameters,
                tensors.constants,
                state,
                time,
                draw,
                bptt=True,
                schedule=schedule,
            )

        advance = (
            torch.compile(transition, backend="aot_eager", fullgraph=True)
            if compiled
            else transition
        )
        state, actual_loss, native_loss = tensors.state, 0, 0
        initial = _snapshot(tensors)
        for index in range(6):
            time = torch.tensor(index * DT, dtype=torch.float64)
            state, outputs = advance(
                state, time, plan.sample_step_draw(state), tensors.schedule
            )
            native(time, bptt=True)
            _close(state, _carry(native))
            _close(outputs.spike_gate, native.spike_gate)
            actual_loss = actual_loss + (index + 1) * outputs.spike_gate.square().sum()
            native_loss = native_loss + (index + 1) * native.spike_gate.square().sum()
        _assert_packet_equal(tensors, initial)
    assert type(tensors.schedule) is dn.func.TensorScheduleTensors
    assert tensors.schedule._fields == (
        "times",
        "amplitudes",
        "time_source",
        "amplitude_source",
    )
    assert (
        tensors.schedule.time_source.is_leaf
        and tensors.schedule.amplitude_source.is_leaf
    )
    actual = torch.autograd.grad(
        actual_loss, (tensors.schedule.time_source, tensors.schedule.amplitude_source)
    )
    wanted = torch.autograd.grad(native_loss, (times, weights))
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    assert native._sched_t_source is times and native._sched_w_source is weights


@pytest.mark.parametrize("kind", ("netstim", "netcon"))
def test_explicit_dual_sources_are_independent_replacements_without_new_packet_fields(
    kind,
):
    native, times, weights = _configured(kind)
    if kind == "netstim":
        plan, tensors = dn.func.make_functional_netstim(native)

        def evaluate(schedule):
            return plan.step(
                tensors.parameters,
                tensors.constants,
                tensors.state,
                0.2,
                plan.sample_step_draw(tensors.state),
                bptt=True,
                schedule=schedule,
            )[1].spike_gate
    else:
        plan, tensors = dn.func.make_functional_netcon(native)
        values = torch.zeros(plan.source_shape, dtype=plan.source_dtype)

        def evaluate(schedule):
            state = tensors.state
            payloads = []
            for _ in range(6):
                state, payload = plan.step(
                    tensors.weights, state, values, schedule=schedule
                )
                payloads.append(payload)
            return torch.stack(payloads)

    original = _snapshot((tensors, times, weights, native.state_dict_for_checkpoint()))
    changed = tensors.schedule._replace(
        time_source=tensors.schedule.time_source + 0.02,
        amplitude_source=1.4 * tensors.schedule.amplitude_source,
    )
    assert not torch.equal(evaluate(changed), evaluate(tensors.schedule))
    _assert_packet_equal(
        (tensors, times, weights, native.state_dict_for_checkpoint()), original
    )
    native.sched_weight_idx[0] = 1
    with pytest.raises(dn.func.FunctionalizationError, match="topology|configuration"):
        plan.extract()


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_dual_reference_projection_has_native_values_and_source_gradients():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        pre = dn.SingleCompartment(
            N=1, C=2, v_init=0.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        post = dn.SingleCompartment(N=1, C=2, integrator=dn.bwd_euler_sc(imem=False))
        post.insert(expsyn.rename("syn"))
        model = dn.Network(
            {"pre": pre, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        model.connect_one_to_one(
            pre[:], post[:], post.mech.syn, threshold=None, weight=0.003, delay=DT
        )
        model.train()
        model.initialize(DT)
        model.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=False,
            diff_scheduled_times=True,
            train_delay_backend="dense",
        )
        native = next(iter(model.synapses.values()))
        times = torch.tensor(
            [0.15, 0.25], device="mps", dtype=torch.float32, requires_grad=True
        )
        weights = torch.tensor(
            [0.7, -0.2], device="mps", dtype=torch.float32, requires_grad=True
        )
        _bind(native, times, weights)
        _schedule(native, [0, 1], time_idx=[0, 1], weight_idx=[0, 1])
        tensors, actual_loss, native_loss = _netcon_trajectory(native)
        actual = torch.autograd.grad(
            actual_loss,
            (tensors.schedule.time_source, tensors.schedule.amplitude_source),
        )
        wanted = torch.autograd.grad(native_loss, (times, weights))
        for value, reference in zip(actual, wanted, strict=True):
            torch.testing.assert_close(value, reference, rtol=3e-4, atol=3e-6)
            assert (
                value.device.type == "mps"
                and value.isfinite().all()
                and value.abs().sum() > 0
            )
