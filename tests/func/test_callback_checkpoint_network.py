"""Network checkpoints share callback restoration and complete RNG continuation."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from tests.func.test_callback_checkpoints import (
    _assert_packet_equal,
    _clone,
    _disk,
)
from tests.func.test_callback_checkpoints import _environment as _environment
from tests.func.test_callback_checkpoints import (
    _model,
    _plain,
)
from tests.func.test_network import DT, _stim_network

STEPS = 8
HALF = STEPS // 2
pytestmark = pytest.mark.cpu


def _specifications():
    crossing = dict(
        node_check=[0], threshold=-64.0, t_start_check=DT, t_end_check=7 * DT
    )
    return {
        "trace": dn.func.Recorder(("cell.v", "cell.mech.syn.g", "t")),
        "raster": dn.func.PopulationCallback("cell", dn.func.Raster(**crossing)),
        "count": dn.func.PopulationCallback("cell", dn.func.APCount(**crossing)),
        "active": dn.func.PopulationCallback("cell", dn.func.Active(**crossing)),
        "anomaly": dn.func.PopulationCallback("cell", dn.func.AnomalyDetector()),
    }


def _bound(functional, tensors, execution):
    owner = (
        functional
        if execution == "eager"
        else functional.compile_rollout_chunk(
            HALF, execution="scan", backend="aot_eager"
        )
    )
    return owner.bind(tensors.parameters, constants=tensors.constants)


def _assert_joined(first, second, expected):
    for field, values in expected["trace"].items():
        torch.testing.assert_close(
            torch.cat((first["trace"][field], second["trace"][field])), values
        )
    torch.testing.assert_close(
        torch.cat((first["raster"], second["raster"])), expected["raster"]
    )
    for name in ("count", "active", "anomaly"):
        torch.testing.assert_close(second[name], expected[name])


@pytest.mark.parametrize("execution", ("eager", "scan"))
def test_network_disk_roundtrip_matches_uninterrupted_state_rng_and_scoped_callbacks(
    tmp_path, execution
):
    functional, tensors = dn.func.make_functional_network(_stim_network())
    receiver, received = dn.func.make_functional_network(_stim_network())
    callbacks = functional.make_callbacks(_specifications())
    receiver_callbacks = receiver.make_callbacks(_specifications())
    bound, receiver_bound = (
        _bound(functional, tensors, execution),
        _bound(receiver, received, execution),
    )
    source = _clone(tensors.state)
    with torch.no_grad():
        expected, whole = dn.func.run(
            bound, tensors.state, steps=STEPS, callbacks=callbacks
        )
        prefix, first = dn.func.run(
            bound, tensors.state, steps=HALF, callbacks=callbacks
        )
        saved = _disk(
            {
                "model": _clone(prefix),
                "callbacks": callbacks.export_state(first["callbacks"].state),
            },
            tmp_path,
        )
        _plain(saved)
        restored = receiver_callbacks.restore_state(saved["callbacks"])
        middle, zero = dn.func.run(
            receiver_bound,
            saved["model"],
            steps=0,
            callbacks=receiver_callbacks,
            callback_state=restored,
        )
        final, second = dn.func.run(
            receiver_bound,
            middle,
            steps=HALF,
            callbacks=receiver_callbacks,
            callback_state=zero["callbacks"].state,
        )
    torch.testing.assert_close(final, expected)
    _assert_joined(first["callbacks"], second["callbacks"], whole["callbacks"])
    # Subsequent draws prove that model checkpointing retained the full RNG
    # stream rather than merely reproducing the observed voltage prefix.
    _assert_packet_equal(
        receiver.sample_runtime_draws(final, steps=3),
        functional.sample_runtime_draws(expected, steps=3),
    )
    torch.testing.assert_close(functional.extract().state, source)
    assert zero["callbacks"]["trace"]["cell.v"].shape[0] == 0
    assert zero["callbacks"]["raster"].shape[0] == 0
    assert second["callbacks"]["trace"]["cell.v"].shape[0] == HALF
    assert whole["callbacks"]["trace"]["cell.mech.syn.g"].abs().sum() > 0


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_network_runner_callbacks_restore_into_rebuilt_inference_runner(
    tmp_path, execution
):
    options = dict(
        dt=DT,
        steps=HALF,
        callbacks=_specifications(),
        execution=execution,
        backend="eager",
    )
    model = _stim_network()
    runner = dn.compile(model, **options)
    receiver = dn.compile(
        _stim_network(), **{**options, "callbacks": _specifications()}
    )
    assert isinstance(runner.callbacks, dn.func.FunctionalCallbacks)
    with pytest.raises(AttributeError):
        runner.callbacks = None
    prefix, first = runner()
    saved = _disk(
        {
            "model": _clone(prefix),
            "callbacks": runner.callbacks.export_state(first["callbacks"].state),
        },
        tmp_path,
    )
    final, second = receiver(
        saved["model"],
        callback_state=receiver.callbacks.restore_state(saved["callbacks"]),
    )
    expected, whole = dn.compile(
        model, **{**options, "steps": STEPS, "callbacks": _specifications()}
    )()
    torch.testing.assert_close(final, expected)
    _assert_joined(first["callbacks"], second["callbacks"], whole["callbacks"])
    plain = dn.compile(_stim_network(), dt=DT, steps=HALF, backend="eager")
    assert plain.callbacks is None


def _scoped_network(*, names=("left", "right")):
    network = dn.Network({names[0]: _model(), names[1]: _model()})
    network.train()
    network.initialize(DT)
    return network


@pytest.mark.parametrize("kind", ("scope", "qualified_path", "region"))
def test_network_checkpoint_rejects_changed_population_scope_or_selection(kind):
    model = _scoped_network()
    functional, tensors = dn.func.make_functional_network(model)
    if kind == "scope":
        left = dn.func.PopulationCallback("left", dn.func.APCount(node_check=[0, 3]))
        right = dn.func.PopulationCallback("right", dn.func.APCount(node_check=[0, 3]))
    elif kind == "qualified_path":
        left = dn.func.Recorder(("left.v",))
        right = dn.func.Recorder(("right.v",))
    else:
        left = dn.func.Recorder(("left.v",), region=model.left[:, [0, 2]])
        right = dn.func.Recorder(("left.v",), region=model.left[:, [1, 3]])
    callbacks = functional.make_callbacks({"analysis": left})
    receiver = functional.make_callbacks({"analysis": right})
    _, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=0,
        callbacks=callbacks,
    )
    packet = callbacks.export_state(auxiliary["callbacks"].state)
    with pytest.raises(
        dn.func.FunctionalizationError, match="configuration|compatible|checkpoint"
    ):
        receiver.restore_state(packet)


def test_equivalent_rebuilt_network_named_region_checkpoint_keeps_selection(tmp_path):
    model, other = _scoped_network(), _scoped_network()
    functional, tensors = dn.func.make_functional_network(model)
    receiver, received = dn.func.make_functional_network(other)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("left.v", "t"), region=model.left.even)}
    )
    receiver_callbacks = receiver.make_callbacks(
        {"trace": dn.func.Recorder(("left.v", "t"), region=other.left.even)}
    )
    with torch.no_grad():
        prefix, first = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=2,
            callbacks=callbacks,
        )
        saved = _disk(
            {
                "model": _clone(prefix),
                "callbacks": callbacks.export_state(first["callbacks"].state),
            },
            tmp_path,
        )
        actual, second = receiver.prepare_and_rollout(
            received.parameters,
            received.constants,
            saved["model"],
            steps=2,
            callbacks=receiver_callbacks,
            callback_state=receiver_callbacks.restore_state(saved["callbacks"]),
        )
        expected, whole = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=4,
            callbacks=callbacks,
        )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.cat(
            (
                first["callbacks"]["trace"]["left.v"],
                second["callbacks"]["trace"]["left.v"],
            )
        ),
        whole["callbacks"]["trace"]["left.v"],
    )
    assert second["callbacks"]["trace"]["left.v"].shape == (2, 1, 2)


class _PopulationName(str):
    pass


def test_network_population_string_subclasses_export_plain_owner_and_scope_names(
    tmp_path,
):
    model = _scoped_network(names=(_PopulationName("left"), _PopulationName("right")))
    functional, tensors = dn.func.make_functional_network(model)
    callbacks = functional.make_callbacks(
        {
            "anomaly": dn.func.PopulationCallback(
                _PopulationName("left"), dn.func.AnomalyDetector()
            )
        }
    )
    receiver, received = dn.func.make_functional_network(_scoped_network())
    receiver_callbacks = receiver.make_callbacks(
        {"anomaly": dn.func.PopulationCallback("left", dn.func.AnomalyDetector())}
    )
    with torch.no_grad():
        _, initial = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=0,
            callbacks=callbacks,
        )
        packet = callbacks.export_state(initial["callbacks"].state)
        _plain(packet)
        restored = receiver_callbacks.restore_state(_disk(packet, tmp_path))
        actual, auxiliary = receiver.prepare_and_rollout(
            received.parameters,
            received.constants,
            received.state,
            steps=1,
            callbacks=receiver_callbacks,
            callback_state=restored,
        )
        expected, reference = receiver.prepare_and_rollout(
            received.parameters,
            received.constants,
            received.state,
            steps=1,
            callbacks=receiver_callbacks,
        )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        dict(auxiliary["callbacks"]), dict(reference["callbacks"])
    )
