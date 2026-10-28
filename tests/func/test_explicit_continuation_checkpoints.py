"""Lowered plans share the portable continuation boundary with runners."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils import _pytree as pytree

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
    _runner,
    _snapshot,
)
from tests.func.test_continuation_checkpoint_compatibility import _node_at
from tests.func.test_continuation_checkpoint_initialization import (
    _fresh_program,
    _scheduled_network,
)
from tests.func.test_continuation_checkpoints import (
    _copy_checkpoint,
    _disk,
    _frames,
    _leaf_flags,
    _plain_checkpoint,
    _tensors,
)
from tests.func.test_lowering_api import G, _target, _voltage
from tests.func.test_membrane_current import DT as CACHE_DT
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network_initialization_bindings import _cross_population_binding
from tests.func.test_noise_workspaces import DT as NOISE_DT
from tests.func.test_noise_workspaces import _population as _noise_population
from tests.func.test_population_owned_cache import _cached as _cached_population

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("compiled", (False, True))
def test_explicit_disk_checkpoint_rebinds_owned_trainable_sources_and_callback_carry(
    network, compiled, tmp_path
):
    functional, tensors, parameters, constants, _, callbacks = _case(network)
    receiving, received, _, _, template, receiving_callbacks = _case(
        network, compiled=compiled
    )
    defaults = _snapshot(received)
    upstream = _target(parameters, network).detach().clone().requires_grad_()
    replacement = upstream * 1.3
    _assign(parameters, network, replacement)
    bound = functional.bind(parameters, constants=constants)
    _, first = dn.func.run(bound, tensors.state, steps=2, callbacks=callbacks)
    current = first["continuation"]
    checkpoint = dn.func.export_continuation(functional, current, callbacks=callbacks)
    _plain_checkpoint(checkpoint)
    with torch.inference_mode():
        restored = dn.func.restore_continuation(
            receiving,
            _disk(checkpoint, tmp_path / "explicit.pt"),
            callbacks=receiving_callbacks,
        )
    target = _target(restored.parameters, network)
    assert target.is_leaf and target.requires_grad and target.grad_fn is None
    assert target is not replacement and target.data_ptr() != replacement.data_ptr()
    if network:
        assert target is restored.parameters["populations"]["right"][G]
    assert restored.callback_state.plan_token is receiving_callbacks._state_token
    assert restored.callback_state.plan_token is not current.callback_state.plan_token
    assert _leaf_flags(restored.parameters) == _leaf_flags(current.parameters)
    assert _leaf_flags(restored.constants) == _leaf_flags(current.constants)
    for value in _tensors(_frames(restored)):
        assert type(value) is torch.Tensor and not value.is_inference()
        assert value.is_leaf and value.grad_fn is None
    owner = template.compiled_chunk if compiled else receiving
    receiving_bound = owner.bind(restored.parameters, constants=restored.constants)
    drive, inputs = _drive(receiving_bound, network, 3, requires_grad=True)
    actual, last = dn.func.run(
        receiving_bound,
        restored,
        steps=3,
        inputs=inputs,
        callbacks=receiving_callbacks,
        checkpoint_every=2,
    )
    expected, reference = receiving.prepare_and_rollout(
        restored.parameters,
        restored.constants,
        restored.state,
        inputs,
        steps=3,
        callbacks=receiving_callbacks,
        callback_state=restored.callback_state,
    )
    _close(actual, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    trace = "left.v" if network else "v"
    assert last["callbacks"]["trace"][trace].shape[0] == 3
    loss = _voltage(actual, network).square().sum()
    reference_loss = _voltage(expected, network).square().sum()
    observed = torch.autograd.grad(loss, (target, drive, upstream), allow_unused=True)
    gradients = torch.autograd.grad(reference_loss, (target, drive))
    assert observed[2] is None
    _close(observed[:2], gradients)
    assert all(
        torch.isfinite(value).all() and value.abs().sum() > 0 for value in observed[:2]
    )
    assert _target(last["continuation"].parameters, network) is target
    _close(received, defaults)


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("direction", ("runner_to_plan", "plan_to_runner"))
def test_runner_and_explicit_portable_formats_are_interoperable(
    network, direction, tmp_path
):
    source, receiver = _runner(network), _runner(network)
    _, first = source()
    current = first["continuation"]
    packet = (
        source.export_continuation(current)
        if direction == "runner_to_plan"
        else dn.func.export_continuation(
            source.functional, current, callbacks=source.callbacks
        )
    )
    saved = _disk(packet, tmp_path / "interoperable.pt")
    restored = (
        dn.func.restore_continuation(
            receiver.functional, saved, callbacks=receiver.callbacks
        )
        if direction == "runner_to_plan"
        else receiver.restore_continuation(saved)
    )
    expected, reference = source(current)
    if direction == "runner_to_plan":
        bound = receiver.kernel.bind(restored.parameters, constants=restored.constants)
        actual, last = dn.func.run(
            bound, restored, steps=3, callbacks=receiver.callbacks
        )
    else:
        actual, last = receiver(restored)
    _close(actual, expected)
    _close(last["callbacks"].outputs, reference["callbacks"].outputs)
    assert packet["format"] == "dendra.func.continuation" and packet["version"] == 1


@pytest.mark.parametrize("network", (False, True))
def test_receiving_metadata_needs_no_runner_or_live_frames_and_allocates_no_dummy_tensors(
    network, monkeypatch
):
    source, tensors, parameters, constants, bound, _ = _case(network)
    receiver, receiving, _, _, _, _ = _case(network)
    _, first = dn.func.run(bound, tensors.state, steps=2)
    current = first["continuation"]
    before = _snapshot(receiving)

    def reject(*args, **kwargs):
        raise AssertionError("checkpoint schema resolution must use frozen metadata")

    with monkeypatch.context() as guarded:
        guarded.setattr(dn, "compile", reject)
        for plan in (source, receiver):
            for name in (
                "extract",
                "prepare",
                "_prepare_values",
                "initialize",
                "initialize_host",
                "_initialize_host",
                "sample_runtime_draws",
            ):
                if hasattr(plan, name):
                    guarded.setattr(plan, name, reject)
            populations = plan.populations.values() if network else (plan,)
            for population in populations:
                for name in (
                    "extract",
                    "_extract_native_state",
                    "_prepare_values",
                    "initialize_host",
                ):
                    guarded.setattr(population, name, reject)
        for name in (
            "empty",
            "empty_like",
            "zeros",
            "zeros_like",
            "ones",
            "ones_like",
            "full",
            "full_like",
        ):
            guarded.setattr(torch, name, reject)
        packet = dn.func.export_continuation(source, current)
        restored = dn.func.restore_continuation(receiver, packet)
    _close(_frames(restored), _frames(current))
    _close(receiving, before)
    assert _target(restored.parameters, network) is not _target(parameters, network)
    assert restored.constants is not constants


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("changed", ("state_shape", "constant_dtype", "constant_keys"))
def test_metadata_decoder_rejects_incompatible_saved_shapes_dtypes_and_constant_keys(
    network, changed
):
    source, tensors, _, _, bound, _ = _case(network)
    receiver, _, _, _, _, _ = _case(network)
    _, first = dn.func.run(bound, tensors.state, steps=1)
    packet = _copy_checkpoint(
        dn.func.export_continuation(source, first["continuation"])
    )
    prefix = ("populations", "left") if network else ()
    if changed == "constant_keys":
        node = _node_at(packet["frames"]["tree"], ("constants", *prefix))
        index = node["keys"].index("dt")
        node["keys"].pop(index)
        node["children"].pop(index)
        message = "receiving frame keys"
    else:
        path = (
            ("state", *prefix, "integrator", "v")
            if changed == "state_shape"
            else ("constants", *prefix, "dt")
        )
        node = _node_at(packet["frames"]["tree"], path)
        index = node["index"]
        value = packet["frames"]["tensors"][index]
        packet["frames"]["tensors"][index] = (
            value.reshape(-1) if changed == "state_shape" else value.to(torch.float32)
        )
        message = "receiving shape, dtype and device"
    with pytest.raises(ValueError, match=message):
        dn.func.restore_continuation(receiver, packet)


@pytest.mark.parametrize("kind", ("point", "multi_point"))
def test_metadata_restores_active_population_cache_and_packed_geometry_without_initialization(
    kind, monkeypatch
):
    source, tensors = dn.func.make_functional(_cached_population(kind), dt=CACHE_DT)
    receiver, receiving = dn.func.make_functional(_cached_population(kind), dt=CACHE_DT)
    initialized = source.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    assert bool(initialized.constants["cache.active"])
    assert not bool(receiving.constants["cache.active"])
    bound = source.bind(initialized.parameters, constants=initialized.constants)
    _, first = dn.func.run(bound, initialized.state, steps=2)
    saved = dn.func.export_continuation(source, first["continuation"])

    def reject(*args, **kwargs):
        raise AssertionError("cache restore must use saved effective values")

    monkeypatch.setattr(receiver, "extract", reject)
    monkeypatch.setattr(receiver, "initialize", reject)
    monkeypatch.setattr(receiver, "initialize_host", reject)
    restored = dn.func.restore_continuation(receiver, saved)
    assert bool(restored.constants["cache.active"])
    _close(restored.constants, first["continuation"].constants)
    actual, _ = dn.func.run(
        receiver.bind(restored.parameters, constants=restored.constants),
        restored,
        steps=3,
    )
    expected, _ = dn.func.run(bound, first["continuation"], steps=3)
    _close(actual, expected)
    assert not bool(receiving.constants["cache.active"])


def test_metadata_restores_noise_derived_and_timestep_workspaces_with_explicit_rng_carry(
    monkeypatch,
):
    source, tensors = dn.func.make_functional(_noise_population(), dt=NOISE_DT)
    receiver, _ = dn.func.make_functional(_noise_population(), dt=NOISE_DT)
    bound = source.bind(tensors.parameters, constants=tensors.constants)
    draws = source.sample_runtime_draws(tensors.state, steps=5)

    def selected(start, stop):
        return pytree.tree_map_only(
            torch.Tensor, lambda value: value[start:stop], draws
        )

    expected, _ = dn.func.run(bound, tensors.state, steps=5, draws=draws)
    _, first = dn.func.run(bound, tensors.state, steps=2, draws=selected(0, 2))
    current = first["continuation"]
    assert any(name.startswith("noise_workspaces.") for name in current.constants)

    def reject(*args, **kwargs):
        raise AssertionError("NOISE workspace restore must not recapture or sample")

    with monkeypatch.context() as guarded:
        guarded.setattr(receiver, "extract", reject)
        guarded.setattr(receiver, "_prepare_values", reject)
        guarded.setattr(receiver, "sample_runtime_draws", reject)
        guarded.setattr(receiver._random_plan, "extract", reject)
        restored = dn.func.restore_continuation(
            receiver, dn.func.export_continuation(source, current)
        )
    _close(restored.constants, current.constants)
    actual, _ = dn.func.run(
        receiver.bind(restored.parameters, constants=restored.constants),
        restored,
        steps=3,
        draws=selected(2, 5),
    )
    _close(actual, expected)


def test_explicit_scheduled_network_retains_namedtuples_rng_and_joined_recording(
    tmp_path, monkeypatch
):
    model, receiving_model = _scheduled_network(), _scheduled_network()
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    receiver, receiving = dn.func.make_functional(receiving_model, dt=NETWORK_DT)
    receiving_before = _snapshot(receiving.state)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("cell.v", "t"))})
    receiving_callbacks = receiver.make_callbacks(
        {"trace": dn.func.Recorder(("cell.v", "t"))}
    )
    parameters = _plain(tensors.parameters)
    schedule = parameters["netstim"]["schedule"]
    parameters["netstim"]["schedule"] = schedule._replace(
        times=schedule.times.detach().clone().requires_grad_(),
        amplitudes=schedule.amplitudes.detach().clone().requires_grad_(),
    )
    bound = functional.bind(parameters, constants=tensors.constants)
    draws = functional.sample_runtime_draws(tensors.state, steps=8)

    def selected(start, stop):
        return pytree.tree_map_only(
            torch.Tensor, lambda value: value[start:stop], draws
        )

    with torch.no_grad():
        expected, reference = dn.func.run(
            bound, tensors.state, steps=8, callbacks=callbacks, draws=draws
        )
        _, first = dn.func.run(
            bound, tensors.state, steps=3, callbacks=callbacks, draws=selected(0, 3)
        )
    native_rng = receiving_model.netstim._rng.get_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("restore must not extract native NetStim or RNG frames")

    with monkeypatch.context() as guarded:
        guarded.setattr(receiver, "_extract", reject)
        guarded.setattr(receiver, "extract", reject)
        restored = dn.func.restore_continuation(
            receiver,
            _disk(
                dn.func.export_continuation(
                    functional, first["continuation"], callbacks=callbacks
                ),
                tmp_path / "schedules.pt",
            ),
            callbacks=receiving_callbacks,
        )
    restored_schedule = restored.parameters["netstim"]["schedule"]
    assert type(restored_schedule) is dn.func.TensorScheduleTensors
    assert restored_schedule.times.is_leaf and restored_schedule.times.requires_grad
    assert (
        restored_schedule.amplitudes.is_leaf
        and restored_schedule.amplitudes.requires_grad
    )
    receiving_bound = receiver.bind(restored.parameters, constants=restored.constants)
    with torch.no_grad():
        actual, last = dn.func.run(
            receiving_bound,
            restored,
            steps=5,
            callbacks=receiving_callbacks,
            draws=selected(3, 8),
        )
    _close(actual, expected)
    _close(_joined(first, last), reference["callbacks"]["trace"])
    torch.testing.assert_close(receiving_model.netstim._rng.get_state(), native_rng)
    _close(receiving.state, receiving_before)


def test_explicit_restore_retains_cross_parameter_constant_aliases_and_trainability():
    model, key = _cross_population_binding()
    receiving_model, _ = _cross_population_binding()
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    receiver, _ = dn.func.make_functional(receiving_model, dt=NETWORK_DT)
    parameters = tensors.independent_parameters(trainable=f"populations.left.{key}")
    bound = functional.bind(parameters, constants=tensors.constants)
    _, first = dn.func.run(bound, tensors.state, steps=2)
    current = first["continuation"]
    restored = dn.func.restore_continuation(
        receiver, dn.func.export_continuation(functional, current)
    )
    selected = restored.parameters["populations"]["left"][key]
    assert (
        selected
        is restored.constants["populations"]["right"]["parametrizations.factor"]
    )
    assert selected.is_leaf and selected.requires_grad and selected.grad_fn is None
    assert selected is not current.parameters["populations"]["left"][key]
    expected, _ = dn.func.run(bound, current, steps=3)
    actual, _ = dn.func.run(
        receiver.bind(restored.parameters, constants=restored.constants),
        restored,
        steps=3,
    )
    _close(actual, expected)


def test_explicit_restore_preserves_active_initialization_values_as_independent_leaves(
    monkeypatch,
):
    source, receiver = _fresh_program(True), _fresh_program(True)
    parameters = _plain(source.parameters)
    upstream = _target(parameters, True).detach().clone().requires_grad_()
    _assign(parameters, True, upstream * 1.2)
    _, first = source(parameters=parameters)
    current = first["continuation"]
    restored = dn.func.restore_continuation(
        receiver.functional,
        dn.func.export_continuation(
            source.functional, current, callbacks=source.callbacks
        ),
        callbacks=receiver.callbacks,
    )
    key = "materializations.values.integrator.mech.mechanisms.leak.g"
    retained = restored.constants["populations"]["left"][key]
    assert retained.requires_grad and retained.is_leaf and retained.grad_fn is None
    assert retained is not current.constants["populations"]["left"][key]

    def reject(*args, **kwargs):
        raise AssertionError(
            "explicit continuation must retain effective initialization"
        )

    monkeypatch.setattr(receiver.functional, "_initialize_host", reject)
    actual, _ = dn.func.run(
        receiver.functional.bind(restored.parameters, constants=restored.constants),
        restored,
        steps=2,
        callbacks=receiver.callbacks,
    )
    expected, _ = source(current)
    _close(actual, expected)
    gradient, prefix_gradient = torch.autograd.grad(
        _voltage(actual, True).square().sum(),
        (retained, upstream),
        allow_unused=True,
    )
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert prefix_gradient is None


@pytest.mark.parametrize("operation", ("export", "restore"))
def test_explicit_callback_collection_must_belong_to_the_exact_functional_plan(
    operation,
):
    source, tensors, _, _, bound, callbacks = _case(False)
    receiver, _, _, _, _, receiving_callbacks = _case(False)
    _, first = dn.func.run(bound, tensors.state, steps=1, callbacks=callbacks)
    current = first["continuation"]
    saved = dn.func.export_continuation(source, current, callbacks=callbacks)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="callbacks belong to a different functional execution plan",
    ):
        if operation == "export":
            dn.func.export_continuation(source, current, callbacks=receiving_callbacks)
        else:
            dn.func.restore_continuation(receiver, saved, callbacks=callbacks)


@pytest.mark.parametrize("operation", ("export", "restore"))
def test_explicit_callback_carry_cannot_be_omitted_or_replaced_by_authored_specifications(
    operation,
):
    functional, tensors, _, _, bound, callbacks = _case(False)
    _, first = dn.func.run(bound, tensors.state, steps=1, callbacks=callbacks)
    current = first["continuation"]
    saved = dn.func.export_continuation(functional, current, callbacks=callbacks)
    function = (
        (lambda **options: dn.func.export_continuation(functional, current, **options))
        if operation == "export"
        else (
            lambda **options: dn.func.restore_continuation(functional, saved, **options)
        )
    )
    with pytest.raises(dn.func.FunctionalizationError, match="callback"):
        function()
    with pytest.raises(
        TypeError, match="callbacks must be a FunctionalCallbacks collection or None"
    ):
        function(callbacks=_callbacks(False))


def test_explicit_checkpoint_entry_points_accept_only_lowered_plans_and_validate_headers_first():
    functional, tensors, _, _, bound, _ = _case(False, compiled=True)
    current = dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    saved = dn.func.export_continuation(functional, current)
    runner = _runner(False, callbacks=False)
    for invalid in (
        runner,
        bound,
        bound.compiled_chunk,
        functional._population,
        object(),
    ):
        with pytest.raises(
            TypeError,
            match="functional must be a FunctionalPopulation or FunctionalNetwork",
        ):
            dn.func.export_continuation(invalid, current)
        with pytest.raises(
            TypeError,
            match="functional must be a FunctionalPopulation or FunctionalNetwork",
        ):
            dn.func.restore_continuation(invalid, saved)
    with pytest.raises(TypeError, match="export_continuation requires a Continuation"):
        dn.func.export_continuation(object(), object())
    with pytest.raises(
        ValueError, match="continuation checkpoint has missing or unknown keys"
    ):
        dn.func.restore_continuation(object(), {}, callbacks=object())


@pytest.mark.mps
def test_explicit_mps_aot_network_checkpoint_keeps_devices_and_suffix_gradients(
    tmp_path,
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    from tests.func.test_runtime_extra_backends import (
        _environment as backend_environment,
    )

    with backend_environment("mps"):
        source, tensors, _, _, bound, callbacks = _case(True, compiled=True)
        receiver, _, _, _, template, receiving_callbacks = _case(True, compiled=True)
        _, first = dn.func.run(bound, tensors.state, steps=2, callbacks=callbacks)
        restored = dn.func.restore_continuation(
            receiver,
            _disk(
                dn.func.export_continuation(
                    source, first["continuation"], callbacks=callbacks
                ),
                tmp_path / "mps.pt",
            ),
            callbacks=receiving_callbacks,
        )
        selected = _target(restored.parameters, True)
        assert selected.device.type == "mps" and selected.dtype is torch.float32
        assert selected.requires_grad and selected.is_leaf and selected.grad_fn is None
        actual, _ = dn.func.run(
            template.compiled_chunk.bind(
                restored.parameters, constants=restored.constants
            ),
            restored,
            steps=3,
            callbacks=receiving_callbacks,
        )
        gradient = torch.autograd.grad(
            _voltage(actual, True).square().mean(), selected
        )[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
