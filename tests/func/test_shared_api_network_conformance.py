"""One connected stochastic Network crosses native and functional lifecycles."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _snapshot
from tests.func.test_network import _clone, _close
from tests.func.test_network_rng_sharing import PREFIX, RAW
from tests.func.test_network_stochastic_conformance import DT, _assert_aliases, _network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    rng = torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                IMEM=False,
                DTYPE=torch.float64,
                DEVICE="cpu",
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def _recording():
    return {"trace": dn.Recorder(("cable.v", "cell.v", "t"), dt=2 * DT)}


def _inputs(current):
    return {name: dn.func.Inputs(intra=current) for name in ("cable", "cell")}


def _voltage_loss(state):
    return sum(
        values["integrator"]["v"].square().mean()
        for values in state["populations"].values()
    )


def test_connected_stochastic_network_shared_api_roundtrip_and_graph_ownership(
    tmp_path,
):
    native, reference, receiving_model = (_network(training=True) for _ in range(3))
    reference.restore_dict_from_checkpoint(
        _clone(native.state_dict_for_checkpoint(), detached=True)
    )
    native_callbacks = dn.func.make_callbacks(native, _recording(), dt=DT)
    reference_callbacks = dn.func.make_callbacks(reference, _recording(), dt=DT)
    options = dict(
        dt=DT,
        steps=2,
        chunk_steps=2,
        initialization="snapshot",
        inference=False,
        trainable=("*g_mu_param", "*sigma_param"),
        backend="eager",
        fullgraph=True,
    )
    receiver = dn.compile(receiving_model, callbacks=_recording(), **options)
    original_parameters = dict(native.named_parameters(remove_duplicate=False))
    original_flags = {
        name: value.requires_grad for name, value in original_parameters.items()
    }
    streams = (native.cable.mech.probe.primary, native.cable.mech.probe.secondary)
    generators = tuple(stream.rng for stream in streams)
    netstim_generator = native.netstim._rng
    global_rng = torch.get_rng_state().clone()
    values = torch.linspace(0.2e-9, 0.8e-9, 11, dtype=torch.float64).reshape(11, 1, 1)
    prefix_current, first_current, resumed_current, suffix_current = (
        values[start:stop].clone().requires_grad_()
        for start, stop in ((0, 3), (3, 6), (6, 8), (8, 11))
    )
    whole = reference.run(
        steps=11, inputs=_inputs(values), callbacks=reference_callbacks
    )

    # The two native calls execute three steps and retain 0.0004 ms.
    prefix = native.run(
        steps=2, inputs=_inputs(prefix_current[:2]), callbacks=native_callbacks
    )
    boundary = native.longrun(
        duration=0.0014,
        chunklength=2,
        inputs=_inputs(prefix_current[2:]),
        callbacks=native_callbacks,
        callback_state=prefix.state,
    )
    assert native._clock_step.item() == 3
    assert native._duration_remainder.item() == 0.0004
    frozen_native = _snapshot(native.state_dict_for_checkpoint())

    program = dn.compile(native, callbacks=_recording(), **options)
    assert len(program.initial_state["rng"]) == 2
    compiled_carry = program.callbacks.restore_state(
        native_callbacks.export_state(boundary.state)
    )
    initial = dn.func.Continuation(
        program.initial_state, program.parameters, program.constants, compiled_carry
    )
    first_state, first = dn.func.run(
        program,
        initial,
        duration=0.00325,
        inputs=_inputs(first_current),
    )
    continuation = first["continuation"]
    assert first_state["clock"]["step"].item() == 6
    assert first_state["clock"]["duration_remainder"].item() == 0.00065
    assert first_state["populations"]["cable"]["integrator"]["v"].grad_fn is not None
    assert program.execution_report().strategy == "compiled_scan"
    assert (
        continuation.parameters["populations"]["cable"][RAW]
        is continuation.parameters["populations"]["cell"][RAW]
    )

    checkpoint = tmp_path / "network-continuation.pt"
    torch.save(program.export_continuation(continuation), checkpoint)
    restored = receiver.restore_continuation(torch.load(checkpoint, weights_only=True))
    restored_shared = restored.parameters["populations"]["cable"][RAW]
    assert restored_shared is restored.parameters["populations"]["cell"][RAW]
    assert restored_shared is not continuation.parameters["populations"]["cable"][RAW]
    assert restored_shared.is_leaf and restored_shared.requires_grad
    assert (
        restored.callback_state.plan_token is not continuation.callback_state.plan_token
    )
    _close(restored.state, continuation.state)
    _close(restored.callback_state.carries, continuation.callback_state.carries)
    sigma_name = f"{PREFIX}DE._SharedGate.sigma_param"
    restored_sigma = restored.parameters["populations"]["cable"][sigma_name]
    assert (
        restored_sigma.is_leaf
        and restored_sigma.requires_grad
        and restored_sigma.grad_fn is None
    )

    resumed_state, resumed = dn.func.run(
        receiver, restored, steps=2, inputs=_inputs(resumed_current)
    )
    incoming = resumed["continuation"]
    assert resumed_state["clock"]["step"].item() == 8
    assert resumed_state["clock"]["duration_remainder"].item() == 0.00065
    sensitivity, old_segment = torch.autograd.grad(
        _voltage_loss(resumed_state),
        (restored_sigma, first_current),
        allow_unused=True,
        retain_graph=True,
    )
    assert sensitivity.isfinite().all() and sensitivity.abs().sum() > 0
    assert old_segment is None
    _assert_packet_equal(native.state_dict_for_checkpoint(), frozen_native)
    frozen_incoming = _snapshot(
        (incoming.state, incoming.parameters, incoming.constants)
    )

    # Callback carry crosses through its own portable protocol; handoff transfers
    # sources, runtime state and RNG without installing callback objects.
    native_carry = native_callbacks.restore_state(
        receiver.callbacks.export_state(incoming.callback_state)
    )
    assert dn.func.commit_continuation_(receiver, native, incoming) is native
    _assert_aliases(native)
    current_parameters = dict(native.named_parameters(remove_duplicate=False))
    assert current_parameters.keys() == original_parameters.keys()
    assert all(
        current_parameters[name] is value
        and value.requires_grad == original_flags[name]
        for name, value in original_parameters.items()
    )
    assert native.cable.mech.probe.primary is streams[0]
    assert native.cable.mech.probe.secondary is streams[1]
    assert all(
        stream.rng is generator
        for stream, generator in zip(streams, generators, strict=True)
    )
    assert native.netstim._rng is netstim_generator
    assert native._duration_remainder.item() == 0.00065
    finished_fraction = native.run(
        duration=0.00035,
        inputs=_inputs(suffix_current[:1]),
        callbacks=native_callbacks,
        callback_state=native_carry,
    )
    final = native.longrun(
        steps=2,
        chunklength=2,
        inputs=_inputs(suffix_current[1:]),
        callbacks=native_callbacks,
        callback_state=finished_fraction.state,
    )

    _assert_packet_equal(
        native.state_dict_for_checkpoint(), reference.state_dict_for_checkpoint()
    )
    assert native._clock_step.item() == 11
    assert native._duration_remainder.item() == 0.0
    traces = (
        prefix,
        boundary,
        first["callbacks"],
        resumed["callbacks"],
        finished_fraction,
        final,
    )
    for name, expected in whole["trace"].items():
        _close(torch.cat(tuple(result["trace"][name] for result in traces)), expected)
    _close(final.state.carries, whole.state.carries)
    assert whole["trace"]["t"].tolist() == [
        0.0,
        2 * DT,
        4 * DT,
        6 * DT,
        8 * DT,
        10 * DT,
    ]
    assert native.cell.mech.events.g.abs().sum() > 0
    assert any(
        connection.delivery_buffer.abs().sum() > 0
        for connection in native.continuous_synapses.values()
    )
    _assert_aliases(native)
    for source, expected in zip(
        streams,
        (reference.cable.mech.probe.primary, reference.cable.mech.probe.secondary),
        strict=True,
    ):
        _close(source.rng.get_state(), expected.rng.get_state())
    _close(native.netstim._rng.get_state(), reference.netstim._rng.get_state())

    saved = _snapshot(native.state_dict_for_checkpoint())
    native_sigma = native.cable.mech.probe.DE._SharedGate.sigma_param
    gradient, *upstream = torch.autograd.grad(
        sum(population.v.square().mean() for population in native.populations.values()),
        (
            native_sigma,
            prefix_current,
            first_current,
            resumed_current,
            restored_sigma,
            suffix_current,
        ),
        allow_unused=True,
    )
    assert gradient.isfinite().all() and gradient.abs().sum() > 0
    assert all(value is None for value in upstream[:-1])
    assert upstream[-1].isfinite().all() and upstream[-1].abs().sum() > 0
    _assert_packet_equal(native.state_dict_for_checkpoint(), saved)
    _close((incoming.state, incoming.parameters, incoming.constants), frozen_incoming)
    _close(torch.get_rng_state(), global_rng)
