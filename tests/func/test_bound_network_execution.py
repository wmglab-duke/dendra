"""Explicit Network bindings retain raw inputs and share host scheduling."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_network import (
    _RAW_WEIGHT,
    DT,
    _clone,
    _close,
    _loss,
    _mixed_network,
    _stim_network,
)
from tests.func.test_network_rollout import _driven_network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _mutable(tree):
    if isinstance(tree, Mapping):
        return {name: _mutable(value) for name, value in tree.items()}
    if isinstance(tree, tuple):
        values = tuple(_mutable(value) for value in tree)
        return type(tree)(*values) if hasattr(tree, "_fields") else values
    return tree


def _bound(functional, tensors, compiled):
    if not compiled:
        return functional.bind(tensors.parameters, tensors.constants)
    kernel = functional.compile_rollout_chunk(3, backend="aot_eager")
    return kernel.bind(tensors.parameters, tensors.constants)


@pytest.mark.parametrize("compiled", (False, True))
def test_bound_network_native_order_matches_direct_and_canonical_execution(compiled):
    model, native = _mixed_network(), _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    bound = _bound(functional, tensors, compiled)
    assert isinstance(bound, dn.func.BoundNetwork)
    assert bound.functional is functional
    assert bound.steps == (3 if compiled else 1)
    assert (bound.compiled_chunk is not None) is compiled
    direct, _ = bound(tensors.state)
    for _ in range(bound.steps):
        native.step()
    _close(direct, functional.extract(native).state)
    canonical, _ = dn.func.run(bound, tensors.state, steps=7, host_span_steps=4)
    for _ in range(7 - bound.steps):
        native.step()
    _close(canonical, functional.extract(native).state)
    _close(tensors.state, functional.extract().state)


@pytest.mark.parametrize("compiled", (False, True))
def test_direct_bound_network_consumes_explicit_draw_tapes_without_sampling(compiled):
    model, native = _stim_network(), _stim_network()
    native.netstim.restore_dict_from_checkpoint(
        _clone(model.netstim.state_dict_for_checkpoint(), detached=True)
    )
    functional, tensors = dn.func.make_functional_network(model)
    bound = _bound(functional, tensors, compiled)
    draws = functional.sample_draw_tape(tensors.state, bound.steps)
    inputs = dn.func.NetworkRolloutInput(netstim=draws)
    actual, _ = bound(tensors.state, inputs)
    for _ in range(bound.steps):
        native.step()
    _close(actual, functional.extract(native).state)
    _close(tensors.state, functional.extract().state)


def test_bound_network_snapshots_nested_mappings_and_preserves_shared_live_leaves():
    model = _driven_network()
    model.right.mech.pas.g_param = model.left.mech.pas.g_param
    model.initialize(DT)
    functional, tensors = dn.func.make_functional_network(model)
    parameters, constants = _mutable(tensors.parameters), _mutable(tensors.constants)
    key = "integrator.mech.mechanisms.pas.g_param"
    original = parameters["populations"]["left"][key]
    original_dt = constants["populations"]["left"]["dt"]
    bound = functional.bind(parameters, constants)
    assert bound.parameters["populations"]["left"][key] is original
    assert bound.parameters["populations"]["right"][key] is original
    parameters["populations"]["left"][key] = original.clone()
    constants["populations"]["left"]["dt"] = 2 * original_dt
    assert bound.parameters["populations"]["left"][key] is original
    assert bound.constants["populations"]["left"]["dt"] is original_dt
    with pytest.raises(TypeError):
        bound.parameters["populations"]["left"][key] = original.clone()
    with pytest.raises(TypeError):
        bound.constants["populations"]["left"]["dt"] = original_dt.clone()
    with torch.no_grad():
        original.mul_(1.1)
    actual, _ = bound(tensors.state)
    expected, _ = functional.prepare_and_rollout(
        bound.parameters, bound.constants, tensors.state, steps=1
    )
    _close(actual, expected)


@pytest.mark.parametrize("compiled", (False, True))
def test_bound_network_repeated_backward_derives_fresh_materialization_graphs(compiled):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    names = tuple(
        name
        for name in tensors.named_parameters
        if name.startswith("synapses.") and name.endswith(f".{_RAW_WEIGHT}")
    )
    parameters = tensors.independent_parameters(trainable=names)
    packet = tensors._replace(parameters=parameters)
    bound = _bound(functional, packet, compiled)
    targets = tuple(
        value for name, value in packet.named_parameters.items() if name in names
    )
    steps = 9
    expected, _ = functional.prepare_and_rollout(
        parameters, tensors.constants, tensors.state, steps=steps
    )
    expected_gradients = torch.autograd.grad(_loss(expected), targets)
    outputs, gradients = [], []
    for _ in range(2):
        # Advance past delayed deliveries before differentiating. Each complete
        # forward starts from the same carry and derives its own coefficients.
        state = tensors.state
        for _ in range(steps // bound.steps):
            state, _ = bound(state)
        outputs.append(state)
        gradients.append(torch.autograd.grad(_loss(state), targets))
        _close(state, expected)
        _close(gradients[-1], expected_gradients)
    _close(outputs[0], outputs[1])
    _close(gradients[0], gradients[1])
    assert all(torch.isfinite(value).all() for value in gradients[0])
    assert all(value.abs().sum() > 0 for value in gradients[0])
    assert not hasattr(bound, "_prepared")


def test_bound_network_rechecks_source_after_mapping_materialization():
    model = _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)

    class MutatingMapping(dict):
        def items(self):
            model.eval()
            return super().items()

    with pytest.raises(dn.func.FunctionalizationError, match="mode|training|changed"):
        functional.bind(MutatingMapping(tensors.parameters), tensors.constants)


def test_bound_network_rejects_wrong_chunk_owner_and_nonowned_eager_method(monkeypatch):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    other, _ = dn.func.make_functional_network(_mixed_network())
    chunk = other.compile_rollout_chunk(2, backend="eager")
    with pytest.raises(dn.func.FunctionalizationError, match="different.*plan"):
        functional.bind(tensors.parameters, tensors.constants, compiled_chunk=chunk)
    original = functional.prepare_and_rollout
    monkeypatch.setattr(
        functional, "prepare_and_rollout", lambda *a, **k: original(*a, **k)
    )
    with pytest.raises(dn.func.FunctionalizationError, match="owned"):
        functional.bind(tensors.parameters, tensors.constants)


def test_bound_network_duration_accumulates_and_explicit_steps_preserve_remainder():
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    bound = functional.bind(tensors.parameters, tensors.constants)
    first, _ = dn.func.run(bound, tensors.state, duration=0.6 * DT)
    assert first["clock"]["step"].item() == 0
    second, _ = dn.func.run(bound, first, duration=0.6 * DT)
    assert second["clock"]["step"].item() == 1
    assert second["clock"]["duration_remainder"].item() == pytest.approx(0.2 * DT)
    final, _ = dn.func.run(bound, second, steps=2)
    assert final["clock"]["step"].item() == 3
    torch.testing.assert_close(
        final["clock"]["duration_remainder"], second["clock"]["duration_remainder"]
    )


def test_bound_network_checkpoint_continuation_preserves_rng_and_recordings():
    functional, tensors = dn.func.make_functional_network(_stim_network())
    bound = functional.compile_rollout_chunk(3, backend="aot_eager").bind(
        tensors.parameters, tensors.constants
    )
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("cell.v", "t"))})
    prefix, prefix_aux = dn.func.run(
        bound, tensors.state, steps=4, checkpoint_every=2, callbacks=callbacks
    )
    final, final_aux = dn.func.run(
        bound,
        prefix,
        steps=3,
        checkpoint_every=2,
        callbacks=callbacks,
        callback_state=prefix_aux["callbacks"].state,
    )
    whole, whole_aux = functional.run(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=7,
        callbacks=callbacks,
    )
    _close(final, whole)
    for name, values in whole_aux["callbacks"]["trace"].items():
        continued = torch.cat(
            (
                prefix_aux["callbacks"]["trace"][name],
                final_aux["callbacks"]["trace"][name],
            )
        )
        _close(continued, values)
        assert continued.shape[0] == 8


@pytest.mark.parametrize(
    "options",
    (
        {"steps": 1, "duration": DT},
        {"steps": 1, "checkpoint_every": 2, "host_span_steps": 2},
        {"steps": 1, "dt": 2 * DT},
    ),
)
def test_bound_network_rejects_ambiguous_schedule_or_clock_override(options):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    bound = functional.bind(tensors.parameters, tensors.constants)
    with pytest.raises((ValueError, dn.func.FunctionalizationError)):
        dn.func.run(bound, tensors.state, **options)


def test_bound_network_inferred_input_horizon_must_match_explicit_duration():
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    bound = functional.bind(tensors.parameters, tensors.constants)
    population = tensors.state["populations"]["cell"]["integrator"]["v"]
    inputs = dn.func.NetworkRolloutInput(
        {"cell": dn.func.RolloutInput(ve=population.new_zeros((2, *population.shape)))}
    )
    actual, _ = dn.func.run(bound, tensors.state, inputs=inputs)
    assert actual["clock"]["step"].item() == 2
    with pytest.raises(ValueError, match="steps|samples|horizon"):
        dn.func.run(bound, tensors.state, inputs=inputs, duration=3 * DT)


@pytest.mark.parametrize("pending", (float("nan"), float("inf"), -1.0))
def test_bound_network_rejects_invalid_duration_remainder_even_for_zero_steps(pending):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    bound = functional.bind(tensors.parameters, tensors.constants)
    state = {
        **tensors.state,
        "clock": {
            **tensors.state["clock"],
            "duration_remainder": torch.tensor(pending, dtype=torch.float64),
        },
    }
    with pytest.raises(ValueError, match="duration_remainder.*finite.*non-negative"):
        dn.func.run(bound, state, steps=0)


def test_bound_network_runtime_extra_uses_the_shared_bounded_assembly():
    model = _driven_network()
    functional, tensors = dn.func.make_functional_network(model)
    bound = functional.bind(tensors.parameters, tensors.constants)
    field = torch.linspace(-1.0, 1.5, model.left.v.numel(), dtype=model.left.dtype())
    field = field.reshape(model.left.shape).requires_grad_()
    actual, _ = dn.func.run(
        bound,
        tensors.state,
        steps=3,
        host_span_steps=2,
        extra={"left": (field, dn.constant(value=0.2))},
    )
    inputs = dn.func.NetworkRolloutInput(
        {"left": dn.func.RolloutInput(ve=(0.2 * field).expand(3, *field.shape))}
    )
    expected, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, inputs, steps=3
    )
    _close(actual, expected)
    actual_loss = actual["populations"]["left"]["integrator"]["v"].square().sum()
    expected_loss = expected["populations"]["left"]["integrator"]["v"].square().sum()
    _close(
        torch.autograd.grad(actual_loss, field),
        torch.autograd.grad(expected_loss, field),
    )
    with pytest.raises(ValueError, match="unknown Populations"):
        dn.func.run(bound, tensors.state, steps=0, extra={"missing": None})
