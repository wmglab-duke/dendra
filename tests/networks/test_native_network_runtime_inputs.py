"""Native Network deterministic inputs share functional tapes and ownership."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _snapshot
from tests.func.test_network import DT, DTYPE, _close
from tests.func.test_network import _loss as _mixed_loss
from tests.func.test_network import _mixed_network
from tests.func.test_network import _native_loss as _native_mixed_loss
from tests.func.test_network_rollout import _driven_network, _native_driven_step
from tests.func.test_stimulation_rng_sharing import DT as STOCHASTIC_DT
from tests.func.test_stimulation_rng_sharing import _network as _stochastic_network

pytestmark = pytest.mark.cpu
STEPS = 7


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


def _drives(model, steps):
    shape = (steps, *model.left.shape)
    count = steps * model.left.v.numel()
    ve = torch.linspace(-0.3, 0.7, count, dtype=DTYPE).reshape(shape).requires_grad_()
    intra = (
        torch.linspace(0.2, 0.8, count, dtype=DTYPE)
        .mul(1e-8)
        .reshape(shape)
        .requires_grad_()
    )
    return ve, intra


def _inputs(ve, intra):
    return {
        "left": dn.func.Inputs(ve=ve, intra=intra),
        "right": dn.func.Inputs(ve=ve, intra=-0.5 * intra),
    }


def _frame(inputs, index):
    return {
        name: packet._replace(
            ve=None if packet.ve is None else packet.ve[index],
            intra=None if packet.intra is None else packet.intra[index],
        )
        for name, packet in inputs.items()
    }


def _execute(model, method, inputs, *, steps=STEPS, extra=None):
    if method == "step":
        for index in range(steps):
            assert model.step(inputs=_frame(inputs, index), extra=extra) is model
    elif method == "run":
        assert model.run(steps * DT, inputs=inputs, extra=extra) is None
    elif method == "longrun":
        assert model.longrun(steps * DT, 3, inputs=inputs, extra=extra) is None
    else:
        assert (
            model.longrun_checkpointed(
                steps * DT, 3, inputs=inputs, extra=extra, safe_checkpoint=True
            )
            is None
        )


def _loss(populations):
    return sum(
        (value * torch.linspace(0.7, 1.3, value.shape[-1], dtype=value.dtype))
        .square()
        .mean()
        for value in populations
    )


@pytest.mark.parametrize("method", ("step", "run", "longrun", "checkpointed"))
def test_network_inputs_match_functional_values_shared_alias_gradients_and_tails(
    method,
):
    target, source = _driven_network(), _driven_network()
    functional, tensors = dn.func.make_functional_network(source)
    ve, intra = _drives(target, STEPS)
    reference_ve, reference_intra = (
        ve.detach().clone().requires_grad_(),
        intra.detach().clone().requires_grad_(),
    )
    inputs = _inputs(ve, intra)
    frozen = _snapshot(inputs)
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        inputs=_inputs(reference_ve, reference_intra),
        steps=STEPS,
    )
    wanted = torch.autograd.grad(
        _loss(local["integrator"]["v"] for local in expected["populations"].values()),
        (reference_ve, reference_intra),
    )
    _execute(target, method, inputs)
    _close(functional.extract(target).state, expected)
    final_state = _snapshot(target.state_dict_for_checkpoint())
    actual = torch.autograd.grad(
        _loss(cell.v for cell in target.populations.values()), (ve, intra)
    )
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)
    _assert_packet_equal(target.state_dict_for_checkpoint(), final_state)
    _close(inputs, frozen)
    assert inputs["left"].ve is inputs["right"].ve is ve
    assert inputs["left"].intra is intra


@pytest.mark.parametrize("method", ("run", "longrun", "checkpointed"))
def test_network_input_families_coexist_with_disjoint_legacy_stimulation(method):
    target, source = _driven_network(), _driven_network()
    for model in (target, source):
        model.left[:, 1].inject(dn.mono_rect(amp=0.2e-8, pw=10.0))
        model.initialize(DT)
    field = torch.linspace(-0.2, 0.4, target.right.v.numel(), dtype=DTYPE).reshape(
        target.right.shape
    )
    legacy = {"right": (field, dn.mono_rect(amp=1.0, pw=10.0))}
    functional, tensors = dn.func.make_functional_network(source, extra=legacy)
    ve, intra = _drives(target, 5)
    inputs = {
        "left": dn.func.Inputs(ve=ve),  # keeps its registered solver injection
        "right": dn.func.Inputs(intra=intra),  # keeps its legacy extracellular input
    }
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        inputs=inputs,
        steps=5,
    )
    wanted = torch.autograd.grad(
        _loss(local["integrator"]["v"] for local in expected["populations"].values()),
        (ve, intra),
    )
    _execute(target, method, inputs, steps=5, extra=legacy)
    _close(functional.extract(target).state, expected)
    actual = torch.autograd.grad(
        _loss(cell.v for cell in target.populations.values()), (ve, intra)
    )
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)


@pytest.mark.parametrize("method", ("step", "run", "longrun", "checkpointed"))
@pytest.mark.parametrize("invalid", ("unknown", "packet", "dtype", "ve", "intra"))
def test_network_input_rejection_precedes_waveform_sampling_and_clock_changes(
    method, invalid
):
    model = _driven_network()
    generator = torch.Generator().manual_seed(812)
    waveform = dn.mono_rect(amp=1.0, pw=0.05).poisson(
        interval=0.1,
        n=3,
        noise=0.7,
        randomize_every_call=True,
        generator=generator,
    )
    shape = model.left.shape if method == "step" else (2, *model.left.shape)
    value = torch.zeros(shape, dtype=DTYPE)
    extra = {"left": (torch.ones(model.left.shape, dtype=DTYPE), waveform)}
    inputs = {"left": dn.func.Inputs(intra=value)}
    if invalid == "unknown":
        inputs = {"missing": dn.func.Inputs(intra=value)}
        error = KeyError
    elif invalid == "packet":
        inputs = {"left": value}
        error = TypeError
    elif invalid == "dtype":
        inputs = {"left": dn.func.Inputs(intra=value.float())}
        error = ValueError
    elif invalid == "ve":
        inputs = {"left": dn.func.Inputs(ve=value)}
        error = ValueError
    else:
        model.left[:, 1].inject(waveform)
        error = ValueError
    state = _snapshot(model.state_dict_for_checkpoint())
    rng, global_rng = generator.get_state().clone(), torch.get_rng_state().clone()
    with pytest.raises(error):
        if method == "step":
            model.step(inputs=inputs, extra=extra)
        else:
            _execute(model, method, inputs, steps=2, extra=extra)
    _assert_packet_equal(model.state_dict_for_checkpoint(), state)
    _close(generator.get_state(), rng)
    _close(torch.get_rng_state(), global_rng)


def test_network_duration_remainder_defines_input_length_including_zero_steps():
    model, reference = _driven_network(), _driven_network()
    empty = torch.empty((0, *model.left.shape), dtype=DTYPE)
    assert model.run(0.4 * DT, inputs={"left": dn.func.Inputs(intra=empty)}) is None
    assert model.t.item() == 0
    previous = _snapshot(model.state_dict_for_checkpoint())
    current = torch.full((1, *model.left.shape), 0.3e-8, dtype=DTYPE)
    with pytest.raises(ValueError, match="steps"):
        model.longrun(0.1 * DT, 3, inputs={"left": dn.func.Inputs(intra=current)})
    _assert_packet_equal(model.state_dict_for_checkpoint(), previous)
    model.longrun(0.6 * DT, 3, inputs={"left": dn.func.Inputs(intra=current)})
    reference.step(inputs={"left": dn.func.Inputs(intra=current[0])})
    _close(model.left.v, reference.left.v)
    _close(model.right.v, reference.right.v)
    _close(model.t, reference.t)


def test_network_checkpoint_input_operands_detect_mutation_before_recomputation():
    model = _driven_network()
    _, intra = _drives(model, 5)
    model.longrun_checkpointed(
        5 * DT,
        2,
        inputs={"left": dn.func.Inputs(intra=intra)},
        safe_checkpoint=True,
    )
    loss = model.left.v.square().sum()
    with torch.no_grad():
        intra.add_(0.1e-8)
    with pytest.raises(RuntimeError, match="modified by an inplace|version"):
        loss.backward()


@pytest.mark.parametrize("collision", (False, True), ids=("batch_only", "trailing"))
def test_network_step_inputs_use_trailing_broadcast_before_batch_fallback(collision):
    target, oracle = _driven_network(), _driven_network()
    compartments = target.left.shape[-1]
    batch = compartments if collision else compartments + 1
    for model in (target, oracle):
        model.batch(batch)
        model.initialize(DT)
    ve = torch.linspace(-0.4, 0.6, batch, dtype=DTYPE)
    # A matching batch/compartment size selects spatial trailing broadcasting.
    expected = (
        ve.reshape(1, 1, compartments).expand(target.left.shape)
        if collision
        else ve.reshape(batch, 1, 1).expand(target.left.shape)
    )
    target.step(inputs={"left": dn.func.Inputs(ve=ve)})
    _native_driven_step(oracle, {"left": dn.func.Inputs(ve=expected)})
    _close(target.left.v, oracle.left.v)
    _close(target.right.v, oracle.right.v)


def test_checkpointed_inputs_preserve_active_event_and_continuous_projection_vjps():
    target, source = _mixed_network(), _mixed_network()
    functional, tensors = dn.func.make_functional_network(source)
    current = (
        torch.linspace(0.5e-8, 1.2e-8, STEPS * target.cell.v.numel(), dtype=DTYPE)
        .reshape(STEPS, *target.cell.shape)
        .requires_grad_()
    )
    reference_current = current.detach().clone().requires_grad_()
    expected, _ = dn.func.run(
        functional,
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants),
        inputs={"cell": dn.func.Inputs(intra=reference_current)},
        steps=STEPS,
    )
    slots = tuple(
        (collection, name)
        for collection in ("synapses", "continuous_synapses")
        for name in getattr(target, collection)
    )
    wanted = torch.autograd.grad(
        _mixed_loss(expected),
        (
            reference_current,
            *(
                tensors.parameters[collection][name]["weight.weights.0.param.rho"]
                for collection, name in slots
            ),
        ),
    )
    target.longrun_checkpointed(
        STEPS * DT,
        3,
        inputs={"cell": dn.func.Inputs(intra=current)},
        safe_checkpoint=True,
    )
    _close(functional.extract(target).state, expected)
    assert target.cell.mech.relay.ledger.abs().sum() > 0
    assert any(
        connection.delivery_buffer.abs().sum() > 0
        for connection in target.continuous_synapses.values()
    )
    actual = torch.autograd.grad(
        _native_mixed_loss(target),
        (
            current,
            *(
                getattr(target, collection)[name].weight.weights[0].param.rho
                for collection, name in slots
            ),
        ),
    )
    _close(actual, wanted)
    assert all(value.isfinite().all() and value.abs().sum() > 0 for value in actual)


def test_network_explicit_inputs_preserve_mechanism_owned_poisson_and_joint_rng_order():
    model, pulses = _stochastic_network()
    reference, reference_pulses = _stochastic_network()
    zero = torch.zeros((5, *model.left.shape), dtype=model.left.dtype())
    generator = pulses[0].generator
    global_rng = torch.get_rng_state().clone()
    # Mechanism ownership leaves the solver input free for an explicit tape.
    model.longrun(
        5 * STOCHASTIC_DT,
        3,
        inputs={"left": dn.func.Inputs(intra=zero)},
    )
    reference.longrun(5 * STOCHASTIC_DT, 3)
    _assert_packet_equal(
        model.state_dict_for_checkpoint(),
        _snapshot(reference.state_dict_for_checkpoint()),
    )
    assert pulses[0].generator is generator
    _close(generator.get_state(), reference_pulses[0].generator.get_state())
    _close(torch.get_rng_state(), global_rng)
