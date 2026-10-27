"""Call-local extracellular drives share the host runner's accepted clocks."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.func import _runners
from dendra.models.mod import pas

DT = 0.01
STEPS = 8
SPAN = 3
pytestmark = pytest.mark.cpu


def _case():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float32):
        model = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=torch.float32,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=1.0e-4, e=-65.0)
        # Only the repeatedly added float32 clock reaches this final-step pulse.
        waveform = dn.mono_rect(amp=1.0, delay=1000.070068359375, pw=0.001)
        model.initialize()
        model.t.fill_(1000.0)
        model.train()
    field = torch.linspace(-1.0, 1.0, model.v.numel()).reshape(model.shape)
    field.requires_grad_()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = {name: value.detach() for name, value in tensors.parameters.items()}
    prepared = functional.prepare(parameters, tensors.constants)
    bound = functional.bind(parameters, prepared)
    amplitude = dict(waveform.named_parameters())["amp"]
    return bound, tensors.state, (field, waveform), (field, amplitude)


@pytest.mark.parametrize("checkpointed", (False, True))
def test_runtime_extra_uses_accepted_clocks_and_bounded_checkpoint_assembly(
    checkpointed, monkeypatch
):
    bound, state, extra, targets = _case()
    field, waveform = extra
    expected = state
    for _ in range(STEPS):
        ve = field * waveform(expected["clock"]["t"])
        expected, _ = bound(expected, dn.func.StepInput(ve=ve))
    expected_v = expected["integrator"]["v"]
    expected_gradients = torch.autograd.grad(expected_v.square().mean(), targets)
    assert all(gradient.abs().max().item() > 0 for gradient in expected_gradients)

    assembled = []
    assemble = _runners._assemble_extra_chunk

    def bounded_assemble(functional, plan, tensors, current, start, steps, *args, **kw):
        assembled.append(steps)
        return assemble(functional, plan, tensors, current, start, steps, *args, **kw)

    monkeypatch.setattr(_runners, "_assemble_extra_chunk", bounded_assemble)
    options = {"checkpoint_every" if checkpointed else "host_span_steps": SPAN}
    actual, _ = dn.func.run(bound, state, steps=STEPS, extra=extra, **options)
    assert assembled == [SPAN, SPAN, STEPS % SPAN]
    actual_v = actual["integrator"]["v"]
    actual_gradients = torch.autograd.grad(actual_v.square().mean(), targets)

    torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)
    torch.testing.assert_close(actual["clock"]["t"], expected["clock"]["t"])
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert max(assembled) == SPAN
    if checkpointed:
        assert len(assembled) > 3


def test_declared_runtime_none_supplies_canonical_zero_chunks(monkeypatch):
    bound, state, _extra, _targets = _case()
    expected, _ = dn.func.run(bound, state, steps=STEPS, host_span_steps=SPAN)
    observed = []
    execute = _runners._execute_steps

    def execute_with_slot(step, current, inputs, steps):
        assert tuple(inputs.ve.shape) == (steps, *bound.functional.shape)
        assert inputs.ve.device == bound.functional.device
        assert inputs.ve.dtype == bound.functional.dtype
        assert torch.count_nonzero(inputs.ve).item() == 0
        observed.append(steps)
        return execute(step, current, inputs, steps)

    monkeypatch.setattr(_runners, "_execute_steps", execute_with_slot)
    actual, _ = dn.func.run(
        bound,
        state,
        steps=STEPS,
        host_span_steps=SPAN,
        extra=None,
        _runtime_extra_slot=True,
    )
    assert observed == [SPAN, SPAN, STEPS % SPAN]
    torch.testing.assert_close(actual["integrator"]["v"], expected["integrator"]["v"])


def test_supplied_network_clock_is_used_for_runtime_assembly(monkeypatch):
    bound, state, extra, _targets = _case()
    functional = bound.functional
    plan, tensors = _runners._prepare_extra(functional, extra, dn.func.RolloutInput())
    times = state["clock"]["t"].new_tensor([1000.070068359375])
    observed = []
    assemble = plan.assemble_tensors

    def assemble_at_network_clock(tensors, sampled_times):
        observed.append(sampled_times)
        return assemble(tensors, sampled_times)

    monkeypatch.setattr(plan, "assemble_tensors", assemble_at_network_clock)
    actual = _runners._assemble_extra_chunk(
        functional, plan, tensors, state, 0, 1, 1, DT, times=times
    )
    assert len(observed) == 1 and observed[0] is times
    field, waveform = extra
    expected = field.unsqueeze(0) * waveform(times).reshape(1, 1, 1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mutation", ("field", "waveform", "rebind_waveform"))
def test_declared_runtime_checkpoint_rejects_changed_authored_extra(mutation):
    bound, state, extra, targets = _case()
    field, waveform = extra
    actual, _ = dn.func.run(
        bound,
        state,
        steps=STEPS,
        checkpoint_every=SPAN,
        extra=extra,
        _runtime_extra_slot=True,
    )
    with torch.no_grad():
        if mutation == "field":
            field.add_(0.1)
        elif mutation == "waveform":
            waveform.amp.add_(0.1)
        else:
            waveform.amp = torch.nn.Parameter(waveform.amp.detach().clone())
    # An in-place Parameter write may be caught while PyTorch unpacks saved
    # checkpoint operands, before the explicit source guard can run.
    error = (
        (dn.func.FunctionalizationError, RuntimeError)
        if mutation == "waveform"
        else dn.func.FunctionalizationError
    )
    with pytest.raises(error, match="changed|modified|version"):
        torch.autograd.grad(actual["integrator"]["v"].square().mean(), targets)


@pytest.mark.parametrize("checkpointed", (False, True))
def test_runtime_waveform_clock_retains_prepared_dt_gradients(checkpointed):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64):
        model = dn.Unmyelinated(
            [2.0],
            L=3.0,
            dx=1.0,
            v_init=-65.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=1.0e-4, e=-65.0)
        model.initialize()
        model.t.fill_(0.175)
        model.train()
        waveform = dn.sin(amp=0.3, freq=0.9, phase=0.2)
    positions = torch.linspace(-1.0, 1.0, model.v.numel(), dtype=model.dtype())
    field = (positions.square() + 0.2 * positions).reshape(model.shape)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = {name: value.detach() for name, value in tensors.parameters.items()}
    constants = {name: value.detach() for name, value in tensors.constants.items()}
    expected_dt = constants["dt"].clone().requires_grad_()
    actual_dt = constants["dt"].clone().requires_grad_()
    expected_bound = functional.bind(
        parameters, functional.prepare(parameters, {**constants, "dt": expected_dt})
    )
    actual_bound = functional.bind(
        parameters, functional.prepare(parameters, {**constants, "dt": actual_dt})
    )
    expected_state = tensors.state
    frames = [
        {"v": expected_state["integrator"]["v"], "t": expected_state["clock"]["t"]}
    ]
    for _ in range(STEPS):
        ve = field * waveform(expected_state["clock"]["t"])
        expected_state, _ = expected_bound(expected_state, dn.func.StepInput(ve=ve))
        frames.append(
            {"v": expected_state["integrator"]["v"], "t": expected_state["clock"]["t"]}
        )
    expected = {
        name: torch.stack([frame[name] for frame in frames]) for name in ("v", "t")
    }
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("v", "t"))})
    options = {"checkpoint_every" if checkpointed else "host_span_steps": SPAN}
    actual_state, auxiliary = dn.func.run(
        actual_bound,
        tensors.state,
        steps=STEPS,
        extra=(field, waveform),
        _runtime_extra_slot=True,
        callbacks=callbacks,
        **options,
    )
    actual = dict(auxiliary["callbacks"]["trace"])
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    weights = torch.linspace(0.7, 1.3, model.shape[-1], dtype=model.dtype())
    expected_gradient = torch.autograd.grad(
        (expected["v"] * weights).square().mean(), expected_dt
    )[0]
    actual_gradient = torch.autograd.grad(
        (actual["v"] * weights).square().mean(), actual_dt
    )[0]
    assert torch.isfinite(expected_gradient).all()
    assert expected_gradient.abs().item() > 0
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=2e-9, atol=2e-10
    )
