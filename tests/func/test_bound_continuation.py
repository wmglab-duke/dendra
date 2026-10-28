"""Raw-bound host schedules share graph-preserving continuation packets."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import (
    _assign,
    _callbacks,
    _close,
    _drive,
    _joined,
    _plain,
    _runner,
    _snapshot,
)
from tests.func.test_lowering_api import DT, G, _network, _population, _target, _voltage
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network import _stim_network
from tests.func.test_network_initialization_bindings import _cross_population_binding

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _case(network, *, compiled=False):
    model = _network() if network else _population()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = tensors.independent_parameters(trainable="*g_param")
    constants = _plain(tensors.constants)
    owner = (
        functional.compile_rollout_chunk(2, backend="aot_eager")
        if compiled
        else functional
    )
    bound = owner.bind(parameters, constants=constants)
    callbacks = functional.make_callbacks(_callbacks(network))
    return functional, tensors, parameters, constants, bound, callbacks


def _inputs(network, current):
    packet = dn.func.Inputs(intra=current)
    return {"right": packet} if network else packet


def _voltage_leaf(state, network):
    return (
        state["populations"]["left"]["integrator"]["v"]
        if network
        else state["integrator"]["v"]
    )


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("compiled", (False, True))
def test_raw_schedule_publishes_owned_frames_and_resumes_callbacks_without_duplicate_samples(
    network, compiled
):
    functional, tensors, parameters, constants, bound, callbacks = _case(
        network, compiled=compiled
    )
    native = _snapshot(functional.extract().state)
    expected, reference = dn.func.run(
        bound, tensors.state, steps=5, callbacks=callbacks, host_span_steps=3
    )
    prefix, first = dn.func.run(bound, tensors.state, steps=2, callbacks=callbacks)
    continuation = first["continuation"]
    assert isinstance(continuation, dn.func.Continuation)
    assert continuation.state is not prefix
    assert _voltage_leaf(continuation.state, network) is _voltage_leaf(prefix, network)
    assert continuation.parameters is not parameters
    assert continuation.constants is not constants
    assert _target(continuation.parameters, network) is _target(parameters, network)
    assert continuation.callback_state.plan_token is callbacks._state_token
    assert continuation.callback_state.plan_token is first["callbacks"].state.plan_token
    if network:
        assert (
            continuation.parameters["populations"]["left"][G]
            is continuation.parameters["populations"]["right"][G]
        )
    actual, second = dn.func.run(
        bound, continuation, steps=3, callbacks=callbacks, host_span_steps=2
    )
    explicit, legacy = dn.func.run(
        bound,
        prefix,
        steps=3,
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
        host_span_steps=2,
    )
    _close(actual, expected)
    _close(actual, explicit)
    _close(second["callbacks"].outputs, legacy["callbacks"].outputs)
    _close(_joined(first, second), reference["callbacks"]["trace"])
    _close(second["callbacks"]["count"], reference["callbacks"]["count"])
    for trace in second["callbacks"]["trace"].values():
        assert trace.shape[0] == 3
    assert _target(second["continuation"].parameters, network) is _target(
        parameters, network
    )
    _close(functional.extract().state, native)
    # Editing returned containers cannot redirect the immutable binding.
    _assign(continuation.parameters, network, None)
    assert _target(bound.parameters, network) is _target(parameters, network)


@pytest.mark.parametrize("network", (False, True))
def test_bound_packet_preserves_nonleaf_parameter_and_input_gradients_across_checkpoint_tails(
    network,
):
    functional, tensors, parameters, constants, bound, callbacks = _case(
        network, compiled=True
    )
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    replacement = upstream * 1.2
    _assign(parameters, network, replacement)
    bound = bound.compiled_chunk.bind(parameters, constants=constants)
    current, full_inputs = _drive(bound, network, 6, requires_grad=True)
    _, first = dn.func.run(
        bound,
        tensors.state,
        inputs=_inputs(network, current[:3]),
        steps=3,
        checkpoint_every=2,
        callbacks=callbacks,
    )
    continuation = first["continuation"]
    assert _target(continuation.parameters, network) is replacement
    assert replacement.grad_fn is not None
    actual, last = dn.func.run(
        bound,
        continuation,
        inputs=_inputs(network, current[3:]),
        steps=3,
        checkpoint_every=2,
        callbacks=callbacks,
    )
    trace = _joined(first, last)
    loss = _voltage(actual, network).square().sum() + sum(
        value.square().mean() for name, value in trace.items() if name != "t"
    )
    observed = torch.autograd.grad(loss, (upstream, current), retain_graph=True)
    expected, reference = functional.prepare_and_rollout(
        parameters,
        constants,
        tensors.state,
        full_inputs,
        steps=6,
        callbacks=callbacks,
    )
    reference_trace = reference["callbacks"]["trace"]
    reference_loss = _voltage(expected, network).square().sum() + sum(
        value.square().mean() for name, value in reference_trace.items() if name != "t"
    )
    gradients = torch.autograd.grad(reference_loss, (upstream, current))
    _close(actual, expected)
    _close(trace, reference_trace)
    _close(observed, gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert _target(last["continuation"].parameters, network) is replacement


@pytest.mark.parametrize("network", (False, True))
def test_zero_step_packets_preserve_callback_phase_and_fractional_duration(network):
    _, tensors, _, _, bound, callbacks = _case(network)
    zero, first = dn.func.run(bound, tensors.state, steps=0, callbacks=callbacks)
    assert first["continuation"].callback_state is not None
    _close(zero, tensors.state)
    same, repeated = dn.func.run(
        bound, first["continuation"], steps=0, callbacks=callbacks
    )
    _close(same, zero)
    assert all(trace.shape[0] == 0 for trace in repeated["callbacks"]["trace"].values())
    partial, half = dn.func.run(
        bound, repeated["continuation"], duration=0.5 * DT, callbacks=callbacks
    )
    _close(_voltage(partial, network), _voltage(zero, network))
    actual, last = dn.func.run(
        bound, half["continuation"], duration=1.5 * DT, callbacks=callbacks
    )
    expected, reference = dn.func.run(
        bound, tensors.state, steps=2, callbacks=callbacks
    )
    _close(actual, expected)
    _close(_joined(first, last), reference["callbacks"]["trace"])
    _close(last["callbacks"]["count"], reference["callbacks"]["count"])


@pytest.mark.parametrize("network", (False, True))
def test_zero_step_without_callbacks_publishes_an_empty_auxiliary_continuation(network):
    _, tensors, parameters, _, bound, _ = _case(network)
    state, auxiliary = dn.func.run(bound, tensors.state, steps=0)
    packet = auxiliary["continuation"]
    assert packet.callback_state is None
    assert _target(packet.parameters, network) is _target(parameters, network)
    _close(state, tensors.state)
    resumed, again = dn.func.run(bound, packet, steps=0)
    _close(resumed, state)
    assert again["continuation"].callback_state is None


def test_network_packet_continues_explicit_netstim_draws_and_rng_carry():
    model = _stim_network()
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    bound = functional.bind(tensors.parameters, constants=tensors.constants)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("cell.v", "t"))})
    native_rng = model.netstim._rng.get_state().clone()
    draws = functional.sample_runtime_draws(tensors.state, steps=8)

    def selected(start, stop):
        return pytree.tree_map_only(
            torch.Tensor, lambda value: value[start:stop], draws
        )

    expected, reference = dn.func.run(
        bound, tensors.state, steps=8, callbacks=callbacks, draws=draws
    )
    _, first = dn.func.run(
        bound, tensors.state, steps=3, callbacks=callbacks, draws=selected(0, 3)
    )
    actual, last = dn.func.run(
        bound, first["continuation"], steps=5, callbacks=callbacks, draws=selected(3, 8)
    )
    _close(actual, expected)
    _close(_joined(first, last), reference["callbacks"]["trace"])
    torch.testing.assert_close(model.netstim._rng.get_state(), native_rng)


@pytest.mark.parametrize("network", (False, True))
def test_restored_sources_require_explicit_rebinding_before_raw_schedule_resume(
    network, tmp_path
):
    source, receiver = (
        _runner(network, callbacks=False),
        _runner(network, callbacks=False),
    )
    _, first = source()
    path = tmp_path / "bound.pt"
    torch.save(source.export_continuation(first["continuation"]), path)
    restored = receiver.restore_continuation(torch.load(path, weights_only=True))
    original = receiver.functional.bind(
        receiver.parameters, constants=receiver.constants
    )
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError),
        match="bound|binding|source|identit",
    ):
        dn.func.run(original, restored, steps=3)
    bound = receiver.functional.bind(restored.parameters, constants=restored.constants)
    actual, last = dn.func.run(bound, restored, steps=3)
    expected, _ = receiver(restored)
    _close(actual, expected)
    assert _target(last["continuation"].parameters, network) is _target(
        restored.parameters, network
    )
    assert last["continuation"].callback_state is None


def test_network_packet_publishes_normalized_reader_aliases_without_rebinding_authored_constants():
    model, key = _cross_population_binding()
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    parameters = tensors.independent_parameters(trainable=f"populations.left.{key}")
    constants = _plain(tensors.constants)
    canonical = parameters["populations"]["left"][key]
    authored = constants["populations"]["right"]["parametrizations.factor"]
    assert authored is not canonical
    with torch.no_grad():
        canonical.add_(0.2)
    bound = functional.bind(parameters, constants=constants)
    expected, _ = functional.prepare_and_rollout(
        parameters, constants, tensors.state, steps=3
    )
    _, first = dn.func.run(bound, tensors.state, steps=1)
    packet = first["continuation"]
    assert (
        packet.constants["populations"]["right"]["parametrizations.factor"] is canonical
    )
    assert (
        bound.constants["populations"]["right"]["parametrizations.factor"] is authored
    )
    assert constants["populations"]["right"]["parametrizations.factor"] is authored
    actual, last = dn.func.run(bound, packet, steps=2)
    _close(actual, expected)
    assert (
        last["continuation"].constants["populations"]["right"][
            "parametrizations.factor"
        ]
        is canonical
    )
    gradient = torch.autograd.grad(_voltage(actual, True).square().sum(), canonical)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_normalized_reader_packet_does_not_hide_original_checkpoint_operand_mutation():
    model, key = _cross_population_binding()
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    parameters = tensors.independent_parameters(trainable=f"populations.left.{key}")
    constants = _plain(tensors.constants)
    authored = constants["populations"]["right"]["parametrizations.factor"]
    canonical = parameters["populations"]["left"][key]
    assert authored is not canonical
    bound = functional.bind(parameters, constants=constants)
    state, auxiliary = dn.func.run(bound, tensors.state, steps=3, checkpoint_every=1)
    assert (
        auxiliary["continuation"].constants["populations"]["right"][
            "parametrizations.factor"
        ]
        is canonical
    )
    with torch.no_grad():
        authored.add_(0.1)
    with pytest.raises(
        (dn.func.FunctionalizationError, RuntimeError),
        match="changed after checkpoint forward|modified by an inplace operation",
    ):
        _voltage(state, True).square().sum().backward()


def test_direct_prepared_and_legacy_execution_keep_their_explicit_result_contracts():
    functional, tensors, parameters, constants, raw, _ = _case(False)
    _, direct = raw(tensors.state)
    assert "continuation" not in direct
    prepared = functional.bind(parameters, functional.prepare(parameters, constants))
    expected, auxiliary = dn.func.run(prepared, tensors.state, steps=2)
    assert "continuation" not in auxiliary
    legacy, metadata = dn.func.run(functional, prepared, tensors.state, tstop=2 * DT)
    _close(legacy, expected)
    assert "continuation" not in metadata
    _, raw_result = dn.func.run(raw, tensors.state, steps=2)
    with pytest.raises(
        (ValueError, TypeError, dn.func.FunctionalizationError),
        match="raw|prepared|constant|Continuation|continuation",
    ):
        dn.func.run(prepared, raw_result["continuation"], steps=2)
