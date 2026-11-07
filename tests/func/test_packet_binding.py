"""Source packets bind effective raw frames without changing execution ownership."""

from __future__ import annotations

import os

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_bound_continuation import _case
from tests.func.test_continuation import (
    _assign,
    _callbacks,
    _close,
    _drive,
    _joined,
    _plain,
    _snapshot,
)
from tests.func.test_continuation_checkpoints import _disk
from tests.func.test_explicit_initialization import _case as _initialization_case
from tests.func.test_lowering_api import G, _target, _voltage
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network_initialization_bindings import _cross_population_binding

pytestmark = pytest.mark.cpu
_KINDS = ("plan", "kernel", "eager_constructor", "compiled_constructor")


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _factory(functional, network, kind):
    chunk = (
        functional.compile_rollout_chunk(2, backend="aot_eager")
        if kind in ("kernel", "compiled_constructor")
        else None
    )
    if kind in ("plan", "kernel"):
        return (functional if chunk is None else chunk).bind
    constructor = dn.func.BoundNetwork if network else dn.func.BoundPopulation
    return lambda *args, **kwargs: constructor(
        functional, *args, compiled_chunk=chunk, **kwargs
    )


def _inputs(network, current):
    inputs = dn.func.Inputs(intra=current)
    return {"right": inputs} if network else inputs


def _geometry(constants, network):
    return constants["populations"]["right"] if network else constants


def _training_case(network, kind):
    functional, tensors, parameters, constants, _, callbacks = _case(network)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    replacement = upstream * 1.2
    _assign(parameters, network, replacement)
    geometry = _geometry(constants, network)
    diameter = geometry["diam"].detach().clone().requires_grad_()
    geometry["diam"] = diameter * 1.1
    bundle = tensors._replace(parameters=parameters, constants=constants)
    before = _snapshot((parameters, constants, tensors.state))
    bind = _factory(functional, network, kind)
    bound = bind(bundle)
    explicit = bind(parameters, constants=constants)
    assert _target(bound.parameters, network) is replacement
    assert _geometry(bound.constants, network)["diam"] is geometry["diam"]
    if network:
        assert replacement is bound.parameters["populations"]["right"][G]
    else:
        assert bound.prepared is None
    drive, inputs = _drive(bound, network, 5, requires_grad=True)
    expected, whole = dn.func.run(
        explicit, tensors.state, steps=5, inputs=inputs, callbacks=callbacks
    )
    _, first = dn.func.run(
        bound,
        bundle.state,
        steps=2,
        inputs=_inputs(network, drive[:2]),
        callbacks=callbacks,
        checkpoint_every=2,
    )
    actual, last = dn.func.run(
        bound,
        first["continuation"],
        steps=3,
        inputs=_inputs(network, drive[2:]),
        callbacks=callbacks,
        checkpoint_every=2,
    )
    _close(actual, expected)
    _close(_joined(first, last), whole["callbacks"]["trace"])
    _close(last["callbacks"]["count"], whole["callbacks"]["count"])
    trace = "left.v" if network else "v"
    assert first["callbacks"]["trace"][trace].shape[0] == 3
    assert last["callbacks"]["trace"][trace].shape[0] == 3
    targets = (upstream, diameter, drive)
    observed = torch.autograd.grad(
        _voltage(actual, network).square().sum(), targets, retain_graph=True
    )
    reference = torch.autograd.grad(_voltage(expected, network).square().sum(), targets)
    _close(observed, reference)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed
    )
    assert _target(last["continuation"].parameters, network) is replacement
    _close((parameters, constants, tensors.state), before)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("kind", _KINDS)
def test_lowering_packet_binding_matches_explicit_sources_and_preserves_training_tails(
    network, kind
):
    _training_case(network, kind)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("origin", ("initialized", "restored"))
def test_effective_initialization_and_portable_packets_bind_without_default_source_reversion(
    network, origin, tmp_path
):
    functional, tensors = _initialization_case(network, scaled=True)
    tensors = tensors._replace(
        parameters=tensors.independent_parameters(trainable="*g_param")
    )
    initial = dn.func.initialize(functional, tensors)
    callbacks = functional.make_callbacks(_callbacks(network))
    if origin == "restored":
        _, first = dn.func.run(
            functional.bind(initial), initial, steps=2, callbacks=callbacks
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
    else:
        packet = initial
    kernel = functional.compile_rollout_chunk(2, backend="aot_eager")
    bound = kernel.bind(packet)
    assert _target(bound.parameters, network) is _target(packet.parameters, network)
    actual, auxiliary = dn.func.run(
        bound, packet, steps=3, callbacks=callbacks, checkpoint_every=2
    )
    expected, reference = functional.prepare_and_rollout(
        packet.parameters,
        packet.constants,
        packet.state,
        steps=3,
        callbacks=callbacks,
        callback_state=packet.callback_state,
    )
    _close(actual, expected)
    _close(auxiliary["callbacks"].outputs, reference["callbacks"].outputs)
    trace = "left.v" if network else "v"
    assert auxiliary["callbacks"]["trace"][trace].shape[0] == (
        4 if origin == "initialized" else 3
    )
    if network and origin == "restored":
        selected = packet.constants["populations"]["left"][
            "materializations.values.integrator.mech.mechanisms.leak.g"
        ]
    else:
        selected = _target(packet.parameters, network)
    observed = torch.autograd.grad(
        _voltage(actual, network).square().sum(), selected, retain_graph=True
    )[0]
    gradient = torch.autograd.grad(
        _voltage(expected, network).square().sum(), selected
    )[0]
    _close(observed, gradient)
    assert torch.isfinite(observed).all() and observed.abs().sum() > 0


@pytest.mark.parametrize("reader_first", (False, True))
def test_network_bundle_binding_preserves_authored_constants_and_canonical_readers(
    reader_first,
):
    model, key = _cross_population_binding(reader_first=reader_first)
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    parameters = tensors.independent_parameters(trainable=f"populations.left.{key}")
    constants = _plain(tensors.constants)
    bundle = tensors._replace(parameters=parameters, constants=constants)
    canonical = parameters["populations"]["left"][key]
    authored = constants["populations"]["right"]["parametrizations.factor"]
    assert authored is not canonical
    bound = functional.bind(bundle)
    explicit = functional.bind(parameters, constants=constants)
    reader_path = "constants.populations.right.parametrizations.factor"
    entries = {
        entry.path: entry.tensor for entry in dn.func.inspect_sources(bound).entries
    }
    references = {
        entry.path: entry.tensor for entry in dn.func.inspect_sources(explicit).entries
    }
    assert list(entries) == list(references)
    assert all(value is references[name] for name, value in entries.items())
    assert entries[reader_path] is canonical
    assert (
        bundle.constants["populations"]["right"]["parametrizations.factor"] is authored
    )
    assert (
        bound.constants["populations"]["right"]["parametrizations.factor"] is authored
    )
    state, auxiliary = dn.func.run(bound, bundle.state, steps=2)
    assert (
        auxiliary["continuation"].constants["populations"]["right"][
            "parametrizations.factor"
        ]
        is canonical
    )
    gradient = torch.autograd.grad(_voltage(state, True).square().sum(), canonical)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    # Binding sources does not redirect an incoming packet's continuation carry.
    incoming = dn.func.Continuation(bundle.state, parameters, constants)
    again = functional.bind(incoming)
    assert (
        incoming.constants["populations"]["right"]["parametrizations.factor"]
        is authored
    )
    with pytest.raises(
        dn.func.FunctionalizationError, match="constants.*bound sources"
    ):
        dn.func.run(again, incoming, steps=1)


def test_packet_population_binding_reuses_inference_preparation_and_refreshes_live_mutations():
    functional, tensors, parameters, constants, _, _ = _case(False)
    packet = tensors._replace(parameters=parameters, constants=constants)
    bound = functional.bind(packet)
    target = _target(parameters, False)
    with torch.no_grad():
        first, _ = dn.func.run(bound, packet.state, steps=2)
        cached = bound._inference_prepared
        assert cached is not None
        repeated, _ = dn.func.run(bound, packet.state, steps=2)
        _close(repeated, first)
        assert bound._inference_prepared is cached
        target.add_(0.0002)
        changed, _ = dn.func.run(bound, packet.state, steps=2)
        assert bound._inference_prepared is not cached
        expected, _ = functional.prepare_and_rollout(
            parameters, constants, packet.state, steps=2
        )
        _close(changed, expected)
        assert not torch.equal(changed["integrator"]["v"], first["integrator"]["v"])
        packet.parameters[G] = target.clone()
        assert bound.parameters[G] is target


@pytest.mark.parametrize("network", (False, True))
def test_packet_binding_keeps_explicit_checkpoint_mutation_diagnostics(network):
    functional, tensors, parameters, constants, _, _ = _case(network)
    kernel = functional.compile_rollout_chunk(2, backend="aot_eager")
    bound = kernel.bind(tensors._replace(parameters=parameters, constants=constants))
    explicit = kernel.bind(parameters, constants=constants)
    actual, _ = dn.func.run(bound, tensors.state, steps=3, checkpoint_every=2)
    expected, _ = dn.func.run(explicit, tensors.state, steps=3, checkpoint_every=2)
    with torch.no_grad():
        _target(parameters, network).add_(0.001)
    failures = []
    for state in (actual, expected):
        try:
            _voltage(state, network).square().sum().backward()
        except (dn.func.FunctionalizationError, RuntimeError) as error:
            failures.append((type(error), str(error)))
    assert len(failures) == 2
    assert failures[0] == failures[1]
    assert "changed" in failures[0][1] or "modified" in failures[0][1]


@pytest.mark.mps
def test_mps_packet_binding_preserves_nonleaf_sources_and_aot_training_tails():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    from tests.func.test_runtime_extra_backends import (
        _environment as backend_environment,
    )

    with backend_environment("mps"):
        _training_case(True, "compiled_constructor")
