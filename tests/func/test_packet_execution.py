"""Source-free plans and chunks schedule explicit continuation packets."""

from __future__ import annotations

import gc
import os
import weakref

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_bound_continuation import _case, _inputs
from tests.func.test_continuation import (
    _assign,
    _callbacks,
    _close,
    _drive,
    _failure,
    _joined,
    _snapshot,
)
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_explicit_initialization import _case as _initialization_case
from tests.func.test_lowering_api import DT, G, _network, _target, _voltage
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network_initialization_bindings import _cross_population_binding
from tests.func.test_packet_binding import _geometry
from tests.func.test_stimulation_rng_sharing import (
    _lower,
)
from tests.func.test_stimulation_rng_sharing import _network as _random_network
from tests.func.test_stimulation_rng_sharing import (
    _one_stream,
)
from tests.func.test_stimulation_rng_sharing import _population as _random_population
from tests.func.test_stimulation_rng_sharing import (
    _slice,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _packet_case(network, *, compiled=False):
    functional, tensors, parameters, constants, bound, callbacks = _case(
        network, compiled=compiled
    )
    owner = bound.compiled_chunk if compiled else functional
    initial = dn.func.Continuation(tensors.state, parameters, constants)
    return functional, tensors, owner, initial, callbacks


def _same(actual, expected):
    _close(actual[0], expected[0])
    assert ("callbacks" in actual[1]) == ("callbacks" in expected[1])
    if "callbacks" in actual[1]:
        _close(actual[1]["callbacks"].outputs, expected[1]["callbacks"].outputs)
        _close(
            actual[1]["callbacks"].state.carries,
            expected[1]["callbacks"].state.carries,
        )
    packet = actual[1]["continuation"]
    assert type(packet) is dn.func.Continuation
    assert packet._lifecycle == "active"
    _close(packet.state, actual[0])


@pytest.mark.parametrize("network", (False, True), ids=("population", "network"))
@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "compiled"))
def test_plan_and_chunk_packets_match_explicit_binding_across_tails_and_continuation(
    network, compiled
):
    functional, tensors, owner, initial, callbacks = _packet_case(
        network, compiled=compiled
    )
    native = _snapshot(functional.extract())
    incoming = _snapshot((initial.state, initial.parameters, initial.constants))
    bound = owner.bind(initial)
    expected = dn.func.run(
        bound, initial, steps=5, callbacks=callbacks, host_span_steps=3
    )
    _, first = dn.func.run(owner, initial, steps=2, callbacks=callbacks)
    actual = dn.func.run(
        owner,
        state=first["continuation"],
        steps=3,
        callbacks=callbacks,
        host_span_steps=2,
    )
    _same(
        actual,
        dn.func.run(
            bound,
            first["continuation"],
            steps=3,
            callbacks=callbacks,
            host_span_steps=2,
        ),
    )
    _close(actual[0], expected[0])
    _close(_joined(first, actual[1]), expected[1]["callbacks"]["trace"])
    _close(actual[1]["callbacks"]["count"], expected[1]["callbacks"]["count"])
    assert all(
        value.shape[0] == 3 for value in actual[1]["callbacks"]["trace"].values()
    )
    packet = actual[1]["continuation"]
    assert _target(packet.parameters, network) is _target(initial.parameters, network)
    assert (
        _geometry(packet.constants, network)["diam"]
        is _geometry(initial.constants, network)["diam"]
    )
    assert packet.callback_state.plan_token is callbacks._state_token
    if network:
        assert packet.parameters["populations"]["right"][G] is _target(
            initial.parameters, network
        )
    _close((initial.state, initial.parameters, initial.constants), incoming)
    _close(tensors.state, incoming[0])
    _close(functional.extract(), native)


@pytest.mark.parametrize("network", (False, True))
def test_packet_zero_steps_and_duration_keep_callback_phase_and_fractional_remainder(
    network,
):
    _, _, owner, initial, callbacks = _packet_case(network, compiled=True)
    zero, first = dn.func.run(owner, state=initial, steps=0, callbacks=callbacks)
    _close(zero, initial.state)
    assert all(value.shape[0] == 1 for value in first["callbacks"]["trace"].values())
    same, repeated = dn.func.run(
        owner, first["continuation"], steps=0, callbacks=callbacks
    )
    _close(same, zero)
    assert all(value.shape[0] == 0 for value in repeated["callbacks"]["trace"].values())
    partial, half = dn.func.run(
        owner, repeated["continuation"], duration=0.5 * DT, callbacks=callbacks
    )
    _close(_voltage(partial, network), _voltage(zero, network))
    clock = "clock" if network else "control"
    assert partial[clock]["duration_remainder"].item() == pytest.approx(0.5 * DT)
    actual, last = dn.func.run(
        owner, half["continuation"], duration=1.5 * DT, callbacks=callbacks
    )
    expected, whole = dn.func.run(
        owner.bind(initial), initial, steps=2, callbacks=callbacks
    )
    _close(actual, expected)
    _close(_joined(first, last), whole["callbacks"]["trace"])
    assert actual[clock]["duration_remainder"].item() == pytest.approx(0.0)
    empty, auxiliary = dn.func.run(owner, initial, steps=0)
    _close(empty, initial.state)
    assert "callbacks" not in auxiliary
    assert auxiliary["continuation"].callback_state is None


@pytest.mark.parametrize("network", (False, True))
def test_packet_checkpointed_tails_preserve_parameter_constant_and_input_gradients(
    network,
):
    functional, _, owner, original, callbacks = _packet_case(network, compiled=True)
    parameters, constants = original.parameters, original.constants
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    replacement = upstream * 1.2
    _assign(parameters, network, replacement)
    geometry = _geometry(constants, network)
    diameter = geometry["diam"].detach().clone().requires_grad_()
    geometry["diam"] = diameter * 1.1
    initial = dn.func.Continuation(original.state, parameters, constants)
    drive, inputs = _drive(owner.bind(initial), network, 5, requires_grad=True)
    _, first = dn.func.run(
        owner,
        initial,
        inputs=_inputs(network, drive[:2]),
        steps=2,
        callbacks=callbacks,
        checkpoint_every=2,
    )
    actual, last = dn.func.run(
        owner,
        first["continuation"],
        inputs=_inputs(network, drive[2:]),
        steps=3,
        callbacks=callbacks,
        checkpoint_every=2,
    )
    expected, reference = functional.prepare_and_rollout(
        parameters,
        constants,
        initial.state,
        inputs,
        steps=5,
        callbacks=callbacks,
    )
    _close(actual, expected)
    _close(_joined(first, last), reference["callbacks"]["trace"])
    targets = (upstream, diameter, drive)
    observed = torch.autograd.grad(
        _voltage(actual, network).square().sum(), targets, retain_graph=True
    )
    gradients = torch.autograd.grad(_voltage(expected, network).square().sum(), targets)
    _close(observed, gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert _target(last["continuation"].parameters, network) is replacement
    assert (
        _geometry(last["continuation"].constants, network)["diam"] is geometry["diam"]
    )


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("origin", ("initialized", "restored"))
def test_effective_initialized_and_restored_sources_execute_without_default_reversion(
    network, origin, tmp_path, monkeypatch
):
    functional, tensors = _initialization_case(network, scaled=True)
    tensors = tensors._replace(
        parameters=tensors.independent_parameters(trainable="*g_param")
    )
    packet = dn.func.initialize(functional, tensors)
    assert _target(packet.parameters, network) is not _target(
        tensors.parameters, network
    )
    callbacks = functional.make_callbacks(_callbacks(network))
    if origin == "restored":
        _, first = dn.func.run(
            functional.bind(packet), packet, steps=2, callbacks=callbacks
        )
        receiving, _ = _initialization_case(network, scaled=True)
        receiving_callbacks = receiving.make_callbacks(_callbacks(network))
        packet = dn.func.restore_continuation(
            receiving,
            _disk(
                dn.func.export_continuation(
                    functional, first["continuation"], callbacks=callbacks
                ),
                tmp_path / "effective.pt",
            ),
            callbacks=receiving_callbacks,
        )
        functional, callbacks = receiving, receiving_callbacks
    owner = functional.compile_rollout_chunk(2, backend="aot_eager")
    bound = owner.bind(packet)
    expected = dn.func.run(
        bound, packet, steps=3, callbacks=callbacks, checkpoint_every=2
    )

    def reject(*args, **kwargs):
        raise AssertionError("packet execution must not extract or initialize sources")

    with monkeypatch.context() as guarded:
        guarded.setattr(functional, "extract", reject)
        guarded.setattr(functional, "initialize", reject)
        guarded.setattr(dn.func, "initialize", reject)
        actual = dn.func.run(
            owner, packet, steps=3, callbacks=callbacks, checkpoint_every=2
        )
    _same(actual, expected)
    trace = "left.v" if network else "v"
    assert actual[1]["callbacks"]["trace"][trace].shape[0] == (
        4 if origin == "initialized" else 3
    )
    assert _target(actual[1]["continuation"].parameters, network) is _target(
        packet.parameters, network
    )
    if network and origin == "restored":
        selected = packet.constants["populations"]["left"][
            "materializations.values.integrator.mech.mechanisms.leak.g"
        ]
    else:
        selected = _target(packet.parameters, network)
    observed = torch.autograd.grad(
        _voltage(actual[0], network).square().sum(), selected, retain_graph=True
    )[0]
    reference = torch.autograd.grad(
        _voltage(expected[0], network).square().sum(), selected
    )[0]
    _close(observed, reference)
    assert torch.isfinite(observed).all() and observed.abs().sum() > 0


@pytest.mark.parametrize("network", (False, True))
def test_packet_automatic_and_explicit_draws_continue_shared_rng_without_native_mutation(
    network,
):
    if network:
        model, pulses = _random_network()
    else:
        model, _, pulse = _random_population()
        pulses = [pulse]
    functional, tensors = _lower(model)
    initial = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    source_rng, global_rng = (
        pulses[0].generator.get_state().clone(),
        torch.get_rng_state().clone(),
    )
    native = _snapshot(functional.extract())
    tape = functional.sample_runtime_draws(initial.state, steps=4)
    frozen = _snapshot(tape)
    callbacks = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(
                ("left.v", "right.v", "t") if network else ("v", "t")
            )
        }
    )
    automatic = dn.func.run(functional, initial, steps=4, callbacks=callbacks)
    explicit = dn.func.run(
        functional, initial, draws=tape, steps=4, callbacks=callbacks
    )
    _same(automatic, explicit)
    _, first = dn.func.run(
        functional, initial, draws=_slice(tape, 0, 1), steps=1, callbacks=callbacks
    )
    actual = dn.func.run(
        functional,
        first["continuation"],
        draws=_slice(tape, 1, 4),
        steps=3,
        callbacks=callbacks,
    )
    _close(actual[0], automatic[0])
    _close(_joined(first, actual[1]), automatic[1]["callbacks"]["trace"])
    _close(_one_stream(actual[0]), _one_stream(automatic[0]))
    assert not torch.equal(_one_stream(actual[0]), _one_stream(initial.state))
    _close(tape, frozen)
    _close(functional.extract(), native)
    _close(pulses[0].generator.get_state(), source_rng)
    _close(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("network", (False, True))
def test_packet_inputs_infer_horizon_and_keep_existing_validation_errors(network):
    _, _, owner, initial, _ = _packet_case(network)
    bound = owner.bind(initial)
    _, inputs = _drive(bound, network, 3)
    _same(
        dn.func.run(owner, initial, inputs=inputs),
        dn.func.run(bound, initial, inputs=inputs),
    )
    options = (
        {"inputs": inputs, "steps": 2},
        {"steps": 1, "duration": DT},
        {"steps": 1, "checkpoint_every": 1, "host_span_steps": 1},
        {"steps": 1, "host_span_steps": 0},
        {"steps": 1, "tstop": DT},
    )
    for keywords in options:
        assert _failure(lambda: dn.func.run(owner, initial, **keywords)) == _failure(
            lambda: dn.func.run(bound, initial, **keywords)
        )


@pytest.mark.parametrize("compiled", (False, True))
def test_packet_execution_rebinds_per_call_without_retaining_binding_or_preparation(
    compiled, monkeypatch
):
    _, _, owner, initial, _ = _packet_case(False, compiled=compiled)
    original_bind = owner.bind
    bindings = []

    def capture(packet):
        bound = original_bind(packet)
        bindings.append(weakref.ref(bound))
        assert bound.parameters[G] is packet.parameters[G]
        assert bound.prepared is None
        return bound

    monkeypatch.setattr(owner, "bind", capture)
    with torch.no_grad():
        first = dn.func.run(owner, initial, steps=3)
        target = initial.parameters[G]
        target.add_(0.0002)
        changed = dn.func.run(owner, initial, steps=3)
        expected = dn.func.run(original_bind(initial), initial, steps=3)
    _same(changed, expected)
    assert not torch.equal(first[0]["integrator"]["v"], changed[0]["integrator"]["v"])
    gc.collect()
    assert len(bindings) == 2
    assert all(binding() is None for binding in bindings)
    assert changed[1]["continuation"].parameters[G] is target


@pytest.mark.parametrize("network", (False, True))
def test_packet_execution_rejects_competing_operands_without_changing_bound_ownership(
    network,
):
    _, _, owner, initial, callbacks = _packet_case(network)
    bound = owner.bind(initial)
    for keyword in ("parameters", "constants", "initialization"):
        with pytest.raises(TypeError, match=keyword):
            dn.func.run(owner, initial, steps=0, **{keyword: None})
    with pytest.raises(TypeError, match="more than once"):
        dn.func.run(owner, initial, state=initial, steps=0)
    _, first = dn.func.run(owner, initial, steps=1, callbacks=callbacks)
    packet = first["continuation"]
    with pytest.raises(
        ValueError, match="continuation cannot be combined with callback_state"
    ):
        dn.func.run(
            owner,
            packet,
            steps=0,
            callbacks=callbacks,
            callback_state=packet.callback_state,
        )
    replacement = _target(initial.parameters, network).clone()
    _assign(packet.parameters, network, replacement)
    with pytest.raises(
        dn.func.FunctionalizationError, match="parameters.*bound sources"
    ):
        dn.func.run(bound, packet, steps=0, callbacks=callbacks)
    _, rebound = dn.func.run(owner, packet, steps=0, callbacks=callbacks)
    assert _target(rebound["continuation"].parameters, network) is replacement
    assert _target(bound.parameters, network) is _target(initial.parameters, network)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("compiled", (False, True))
def test_packet_execution_accepts_exact_targets_and_continuations_only(
    network, compiled
):
    _, tensors, owner, initial, _ = _packet_case(network, compiled=compiled)
    with pytest.raises((TypeError, ValueError)):
        dn.func.run(owner, tensors, steps=0)
    with pytest.raises((TypeError, ValueError)):
        dn.func.run(owner, initial.state, steps=0)
    with pytest.raises((TypeError, ValueError)):
        dn.func.run(owner, steps=0)

    class PacketSubclass(dn.func.Continuation):
        pass

    subclass = PacketSubclass(initial.state, initial.parameters, initial.constants)
    with pytest.raises(TypeError, match="exact|Continuation"):
        dn.func.run(owner, subclass, steps=0)

    class TargetSubclass(type(owner)):
        pass

    inherited = object.__new__(TargetSubclass)
    inherited.__dict__.update(owner.__dict__)
    with pytest.raises(TypeError, match="exact|target"):
        dn.func.run(inherited, initial, steps=0)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("context", ("compile", "grad"))
def test_packet_execution_retains_existing_host_boundary_rejections(
    network, context, monkeypatch
):
    _, _, owner, initial, _ = _packet_case(network)
    with pytest.raises(
        dn.func.FunctionalizationError, match="torch.compile|torch.func|host"
    ):
        if context == "compile":
            monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
            dn.func.run(owner, initial, steps=1)
        else:
            torch.func.grad(
                lambda value: _voltage(
                    dn.func.run(owner, initial, steps=1)[0], network
                ).sum()
                + value
            )(torch.tensor(0.0))


def test_network_packet_execution_keeps_canonical_reader_validation():
    model, key = _cross_population_binding()
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    parameters = tensors.independent_parameters(trainable=f"populations.left.{key}")
    incoming = dn.func.Continuation(tensors.state, parameters, tensors.constants)
    bound = functional.bind(incoming)
    # The new form retains strict packet identity validation; constructing a
    # binding does not silently rewrite a packet's authored constant readers.
    assert _failure(lambda: dn.func.run(functional, incoming, steps=0)) == _failure(
        lambda: dn.func.run(bound, incoming, steps=0)
    )
    _, first = dn.func.run(bound, tensors.state, steps=0)
    canonical = first["continuation"]
    actual = dn.func.run(functional, canonical, steps=1)
    expected = dn.func.run(functional.bind(canonical), canonical, steps=1)
    _same(actual, expected)
    assert (
        actual[1]["continuation"].constants["populations"]["right"][
            "parametrizations.factor"
        ]
        is parameters["populations"]["left"][key]
    )


def test_legacy_callable_and_prepared_population_contracts_remain_separate():
    functional, tensors, parameters, constants, raw, _ = _case(False)
    prepared = functional.bind(parameters, functional.prepare(parameters, constants))
    expected, auxiliary = dn.func.run(prepared, tensors.state, steps=2)
    assert "continuation" not in auxiliary
    actual, legacy = dn.func.run(functional, prepared, tensors.state, tstop=2 * DT)
    _close(actual, expected)
    assert "continuation" not in legacy

    class CallablePacket(dn.func.Continuation):
        def __call__(self, state, inputs):
            return prepared(state, inputs)

    adapter = CallablePacket(tensors.state, parameters, constants)
    actual, legacy = dn.func.run(functional, adapter, tensors.state, tstop=2 * DT)
    _close(actual, expected)
    assert "continuation" not in legacy
    with pytest.raises(TypeError, match="exact Continuation"):
        dn.func.run(functional, adapter, steps=0)
    packet = dn.func.Continuation(tensors.state, parameters, constants)
    with pytest.raises(dn.func.FunctionalizationError, match="raw binding|prepared"):
        dn.func.run(prepared, packet, steps=0)
    _, direct = raw(tensors.state)
    assert "continuation" not in direct

    class Callable:
        def __call__(self, *args, **kwargs):
            raise AssertionError("unrelated callables must not become packet targets")

    with pytest.raises((TypeError, ValueError)):
        dn.func.run(Callable(), packet, steps=0)


def test_packet_target_admission_uses_class_identity_without_metaclass_equality():
    _, _, owner, initial, _ = _packet_case(False)
    comparisons = []

    class Metadata(type):
        def __eq__(cls, other):
            comparisons.append(other)
            return True

    class Inherited(type(owner), metaclass=Metadata):
        pass

    inherited = object.__new__(Inherited)
    inherited.__dict__.update(owner.__dict__)
    with pytest.raises(TypeError, match="exact FunctionalPopulation"):
        dn.func.run(inherited, initial, steps=0)
    assert comparisons == []


@pytest.mark.mps
def test_mps_network_packet_while_execution_preserves_tails_and_callback_continuation():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DTYPE=torch.float32, DEVICE="mps"):
        functional, tensors = dn.func.make_functional(_network(), dt=DT)
        packet = dn.func.Continuation(
            tensors.state, tensors.parameters, tensors.constants
        )
        callbacks = functional.make_callbacks(
            {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
        )
        owner = functional.compile_rollout_chunk(
            4, execution="while", backend="aot_eager"
        )
        with torch.inference_mode():
            actual = dn.func.run(owner, packet, steps=5, callbacks=callbacks)
            expected = dn.func.run(
                owner.bind(packet), packet, steps=5, callbacks=callbacks
            )
            _same(actual, expected)
            resumed = dn.func.run(
                owner, actual[1]["continuation"], steps=1, callbacks=callbacks
            )
            reference = dn.func.run(
                owner.bind(actual[1]["continuation"]),
                actual[1]["continuation"],
                steps=1,
                callbacks=callbacks,
            )
            _same(resumed, reference)
        assert _voltage(actual[0], True).device.type == "mps"
        assert _voltage(actual[0], True).dtype == torch.float32
        assert resumed[1]["callbacks"]["trace"]["left.v"].shape[0] == 1
