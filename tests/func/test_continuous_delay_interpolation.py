"""Fractional continuous delivery through the native and functional Network APIs."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import expsyn, graded_syn, pas
from dendra.models.networks._continuous_delivery import continuous_delay_taps
from dendra.models.parametric import Bounded
from tests.func.test_network import _clone, _close

pytestmark = pytest.mark.cpu
DT = 0.125
DTYPE = torch.float64
MAX_DELAY = 6 * DT
WEIGHT = 0.002
RAW_DELAY = "delay_ms.weights.0.param.rho"


class _Drive(Mechanism):
    Mechanism.CARRY("gate")


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
            torch_compiler_warning_context(),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _network(delay, *, interpolate=True, training=True, event=False):
    pre = dn.SingleCompartment(N=1, C=1)
    post = dn.SingleCompartment(N=1, C=1)
    pre.insert(_Drive.rename("drive"))
    post.insert(pas, g=0.001, e=-65.0)
    post.insert(graded_syn, e=0.0)
    if event:
        post.insert(expsyn.rename("event"), e=0.0, tau=2.0)
    network = dn.Network({"pre": pre, "post": post}, netcon_train_backend="dense")
    network.connect_continuous_one_to_one(
        pre[:],
        post[:],
        post.mech.graded_syn,
        pre_var="mech.drive.gate",
        weight=WEIGHT,
        delay=delay,
        transform=torch.tanh,
    )
    if event:
        network.connect_one_to_one(pre[:], post[:], post.mech.event, delay=DT)
    network.train(training)
    network.build(DT, max_delay_ms=MAX_DELAY)
    if interpolate:
        network.set_synaptic_diff_config(kind="continuous", diff_delays=True, taps=2)
    if event:
        network.set_synaptic_diff_config(diff_spiking=False)
    network.initialize(DT, max_delay_ms=MAX_DELAY)
    return network


@pytest.mark.parametrize("compiled", (False, True))
def test_continuous_delay_value_gradient_and_checkpoint_parity(compiled):
    source = Bounded(2.35 * DT, min_val=DT, max_val=MAX_DELAY, requires_grad=True)
    native = _network(source, event=True)
    resumed = _network(
        Bounded(2.35 * DT, min_val=DT, max_val=MAX_DELAY, requires_grad=True),
        event=True,
    )
    assert next(iter(native.synapses.values())).train_flags[1] is False
    plan, tensors = dn.func.make_functional_network(native)
    name = next(iter(native.continuous_synapses))
    parameters = tensors.independent_parameters(
        (f"continuous_synapses.{name}.{RAW_DELAY}",)
    )
    execute = plan.prepare_and_step
    if compiled:
        execute = torch.compile(execute, backend="aot_eager", fullgraph=True)

    drives = torch.linspace(0.2, 1.5, 12, dtype=DTYPE).reshape(-1, 1, 1)
    state = tensors.state
    native_deliveries = []
    functional_deliveries = []
    for index, drive in enumerate(drives):
        native.pre.mech.drive.gate = drive
        native.step()
        state["populations"]["pre"]["mechanism_buffers"]["drive"]["gate"] = drive
        state, _ = execute(parameters, tensors.constants, state)
        observed = state["populations"]["post"]["mechanism_buffers"]["graded_syn"]
        functional_deliveries.append(observed["g_pre"])
        native_deliveries.append(native.post.mech.graded_syn.g_pre)
        _close(state, plan.extract(native).state)
        if index == 3:
            resumed.restore_dict_from_checkpoint(
                _clone(native.state_dict_for_checkpoint(), detached=True)
            )
        elif index > 3:
            resumed.pre.mech.drive.gate = drive
            resumed.step()
            _close(state, plan.extract(resumed).state)

    native_values = torch.stack(native_deliveries)
    functional_values = torch.stack(functional_deliveries)
    signal = WEIGHT * drives.tanh()
    recent = torch.cat((torch.zeros_like(signal[:2]), signal[:-2]))
    older = torch.cat((torch.zeros_like(signal[:3]), signal[:-3]))
    expected = torch.lerp(recent, older, source._compute() / DT - 2)
    torch.testing.assert_close(functional_values, expected)
    torch.testing.assert_close(native_values, expected)
    raw_delay = parameters["continuous_synapses"][name][RAW_DELAY]
    (functional_gradient,) = torch.autograd.grad(
        functional_values.square().sum(), raw_delay
    )
    (native_gradient,) = torch.autograd.grad(native_values.square().sum(), source.rho)
    (expected_gradient,) = torch.autograd.grad(expected.square().sum(), source.rho)
    assert expected_gradient.abs() > 0
    torch.testing.assert_close(functional_gradient, expected_gradient)
    torch.testing.assert_close(native_gradient, expected_gradient)


def test_linear_integer_delay_preserves_zero_delivery_presence_in_eval():
    network = _network(2 * DT, interpolate=False, training=False)
    network.set_synaptic_diff_config(kind="continuous", diff_delays=True)
    connection = next(iter(network.continuous_synapses.values()))
    network.pre.mech.drive.gate = torch.zeros((1, 1), dtype=DTYPE)
    network.step()
    # The zero sample is delivered at step 2. Its zero-coefficient upper tap
    # contributes neither payload nor presence.
    assert connection.delivery_buffer.count_nonzero().item() == 0
    assert connection.delivery_mask[:, 0].nonzero().flatten().tolist() == [2]


def test_continuous_delay_learns_fractional_target_from_integer_initialization():
    initial_delay = 2 * DT
    target_fraction = 0.5
    target_delay = initial_delay + target_fraction * DT
    delay = Bounded(initial_delay, requires_grad=True)
    network = _network(delay)
    connection = next(iter(network.continuous_synapses.values()))
    assert connection.delay_ms().item() == initial_delay

    drives = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE).reshape(-1, 1, 1)
    impulse = WEIGHT * drives[0].tanh()
    optimizer = torch.optim.SGD([delay.rho], lr=0.75 * DT**2)

    def late_fraction():
        network.initialize(DT)
        for drive in drives:
            network.pre.mech.drive.gate = drive
            network.step()
        return network.post.mech.graded_syn.g_pre / impulse

    for iteration in range(10):
        optimizer.zero_grad()
        fraction = late_fraction()
        # The second sample is zero but marks this slot present via its lower
        # tap. Only the first sample's upper tap can provide a delay gradient.
        if iteration == 0:
            assert fraction.item() == 0.0
        loss = 0.5 * (fraction - target_fraction).square().sum()
        loss.backward()
        optimizer.step()
        if iteration == 0:
            assert initial_delay < delay.rho.item() < target_delay

    assert delay.rho.item() == pytest.approx(target_delay, abs=1e-7)
    with torch.no_grad():
        assert late_fraction().item() == pytest.approx(target_fraction, abs=1e-6)


def test_roundoff_correction_preserves_right_hand_delay_gradient():
    dt = torch.tensor(DT, dtype=DTYPE)
    knot = 2 * dt
    delays = torch.stack((torch.nextafter(knot, dt), torch.nextafter(knot, 3 * dt)))
    delays.requires_grad_()
    steps, fraction = continuous_delay_taps(
        delays, dt, max_delay=MAX_DELAY, dtype=DTYPE
    )
    assert steps.tolist() == [2, 2]
    torch.testing.assert_close(fraction, torch.zeros_like(delays), rtol=0, atol=0)
    (gradient,) = torch.autograd.grad(fraction.sum(), delays)
    torch.testing.assert_close(gradient, torch.full_like(delays, 1 / DT))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_roundoff_correction_preserves_low_precision_fraction(dtype):
    dt = torch.tensor(DT, dtype=dtype)
    delay = torch.nextafter(2 * dt, dt).reshape(1)
    steps, fraction = continuous_delay_taps(delay, dt, max_delay=MAX_DELAY, dtype=dtype)
    assert steps.tolist() == [1]
    torch.testing.assert_close(fraction, delay / dt - 1, rtol=0, atol=0)


def test_delay_change_after_portable_restore_only_routes_new_samples():
    old_delay = Bounded(2.25 * DT, requires_grad=True)
    original = _network(old_delay)
    plan, _ = dn.func.make_functional_network(original)
    original.pre.mech.drive.gate = torch.tensor([[0.6]], dtype=DTYPE)
    original.step()
    captured = dn.func.capture_continuation(plan, original)
    saved = dn.func.export_continuation(plan, captured)

    new_delay = Bounded(4.5 * DT, requires_grad=True)
    receiver, tensors = dn.func.make_functional_network(_network(new_delay))
    restored = dn.func.restore_continuation(receiver, saved)
    state = restored.state
    delivered = []
    for index in range(1, 8):
        drive = torch.tensor([[0.9 if index == 1 else 0.0]], dtype=DTYPE)
        state["populations"]["pre"]["mechanism_buffers"]["drive"]["gate"] = drive
        state, _ = receiver.prepare_and_step(
            tensors.parameters, restored.constants, state
        )
        delivered.append(
            state["populations"]["post"]["mechanism_buffers"]["graded_syn"]["g_pre"]
        )

    old_sample, new_sample = WEIGHT * torch.tensor([0.6, 0.9], dtype=DTYPE).tanh()
    expected = torch.zeros(7, dtype=DTYPE)
    expected[1:3] = old_sample * torch.tensor([0.75, 0.25], dtype=DTYPE)
    expected[4:6] = new_sample * 0.5
    torch.testing.assert_close(torch.stack(delivered).flatten(), expected)


def test_continuous_delay_interpolation_is_explicit_and_checks_bounds():
    network = _network(0.5 * DT, interpolate=False)
    connection = next(iter(network.continuous_synapses.values()))
    assert connection.delay_steps.tolist() == [0]
    with pytest.raises(ValueError, match="delay"):
        network.set_synaptic_diff_config(kind="continuous", diff_delays=True)
    network.step()
    assert connection.delay_steps.tolist() == [0]
