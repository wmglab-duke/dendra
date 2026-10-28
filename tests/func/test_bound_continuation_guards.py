"""Packet admission preserves authoritative bindings and cold error boundaries."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
import dendra.func._continuation as continuation_module
from tests.func.test_bound_continuation import _case
from tests.func.test_bound_continuation import _environment as _environment
from tests.func.test_continuation import _assign, _callbacks, _close, _plain, _snapshot
from tests.func.test_lowering_api import G, _target
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network_initialization_bindings import _cross_population_binding

pytestmark = pytest.mark.cpu


def _reject_model_work(monkeypatch, functional):
    def reject(*args, **kwargs):
        raise AssertionError(
            "invalid continuation must reject before preparation or sampling"
        )

    for name in (
        "prepare",
        "_prepare_source_validated",
        "_prepare_values",
        "sample_runtime_draws",
        "_sample_missing_runtime_draws",
        "sample_noise_draw_tape",
    ):
        if hasattr(functional, name):
            monkeypatch.setattr(functional, name, reject)


@pytest.mark.parametrize(
    "network, changed",
    (
        (False, "parameters"),
        (True, "parameters"),
        (False, "constants"),
        (True, "constants"),
        (True, "alias"),
    ),
)
def test_packet_source_mismatches_reject_before_preparation_sampling_or_mutation(
    network, changed, monkeypatch
):
    functional, tensors, parameters, constants, bound, _ = _case(network)
    packet = dn.func.Continuation(tensors.state, parameters, constants)
    if changed == "parameters":
        _assign(packet.parameters, network, _target(parameters, network).clone())
    elif changed == "constants":
        owner = packet.constants["populations"]["left"] if network else packet.constants
        owner["dt"] = owner["dt"].clone()
    else:
        # Equal values and shared storage do not establish bound object identity.
        original = packet.parameters["populations"]["left"][G]
        packet.parameters["populations"]["right"][G] = original.view_as(original)
    sources = _snapshot((bound.parameters, bound.constants, tensors.state))
    rng = torch.random.get_rng_state().clone()
    _reject_model_work(monkeypatch, functional)
    with pytest.raises(
        dn.func.FunctionalizationError, match="bound sources|raw binding"
    ):
        dn.func.run(bound, packet, steps=2)
    _close((bound.parameters, bound.constants, tensors.state), sources)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("network", (False, True))
def test_packet_callback_state_keyword_conflicts_reject_before_model_work(
    network, monkeypatch
):
    functional, tensors, _, _, bound, callbacks = _case(network)
    _, first = dn.func.run(bound, tensors.state, steps=1, callbacks=callbacks)
    _reject_model_work(monkeypatch, functional)
    with pytest.raises(ValueError, match="callback_state"):
        dn.func.run(
            bound,
            first["continuation"],
            steps=1,
            callbacks=callbacks,
            callback_state=first["callbacks"].state,
        )


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("other_collection", (False, True))
def test_packet_callback_carry_requires_its_explicit_owning_collection_before_model_work(
    network, other_collection, monkeypatch
):
    functional, tensors, _, _, bound, callbacks = _case(network)
    _, first = dn.func.run(bound, tensors.state, steps=1, callbacks=callbacks)
    receiver = (
        functional.make_callbacks(_callbacks(network)) if other_collection else None
    )
    _reject_model_work(monkeypatch, functional)
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="callback|collection|plan"
    ):
        dn.func.run(bound, first["continuation"], steps=1, callbacks=receiver)


@pytest.mark.parametrize("network", (False, True))
def test_continuation_publication_failure_restores_the_compiled_kernel_report(
    network, monkeypatch
):
    functional, tensors, _, _, bound, callbacks = _case(network, compiled=True)
    _, first = dn.func.run(bound, tensors.state, steps=2, callbacks=callbacks)
    previous = bound.compiled_chunk.execution_report()
    assert previous is not None
    packet_state = _snapshot(first["continuation"].state)

    def fail(*args, **kwargs):
        raise RuntimeError("continuation publication failed")

    monkeypatch.setattr(continuation_module, "_complete_continuation", fail)
    with pytest.raises(RuntimeError, match="continuation publication failed"):
        dn.func.run(bound, first["continuation"], steps=1, callbacks=callbacks)
    assert bound.compiled_chunk.execution_report() is previous
    _close(first["continuation"].state, packet_state)


def test_network_does_not_normalize_away_an_invalid_incoming_reader_alias(monkeypatch):
    model, key = _cross_population_binding()
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    parameters = tensors.independent_parameters(trainable=f"populations.left.{key}")
    constants = _plain(tensors.constants)
    bound = functional.bind(parameters, constants=constants)
    _, first = dn.func.run(bound, tensors.state, steps=1)
    packet = first["continuation"]
    canonical = parameters["populations"]["left"][key]
    reader = packet.constants["populations"]["right"]
    assert reader["parametrizations.factor"] is canonical
    reader["parametrizations.factor"] = canonical.clone()
    _reject_model_work(monkeypatch, functional)
    with pytest.raises(
        dn.func.FunctionalizationError, match="constants.*bound sources"
    ):
        dn.func.run(bound, packet, steps=1)


@pytest.mark.parametrize("network", (False, True))
def test_equivalent_mapping_containers_and_explicit_none_callback_keyword_are_accepted(
    network,
):
    _, tensors, parameters, constants, bound, _ = _case(network)
    packet = dn.func.Continuation(tensors.state, parameters, constants)
    # Container ownership is independent; Tensor ownership is authoritative.
    items = tuple(packet.parameters.items())
    packet.parameters.clear()
    packet.parameters.update(reversed(items))
    actual, auxiliary = dn.func.run(bound, packet, steps=1, callback_state=None)
    expected, _ = dn.func.run(bound, tensors.state, steps=1)
    _close(actual, expected)
    assert auxiliary["continuation"].callback_state is None
