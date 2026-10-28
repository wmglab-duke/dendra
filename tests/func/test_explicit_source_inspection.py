"""Explicit bundles and raw bindings expose one read-only source inventory."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_continuation import _close, _plain, _snapshot
from tests.func.test_lowering_api import DT, G, _network, _population, _target
from tests.func.test_network import DT as NETWORK_DT
from tests.func.test_network_initialization_bindings import _cross_population_binding
from tests.func.test_runtime_extra_backends import _environment as _device_environment
from tests.func.test_runtime_extra_backends import _model as _device_model

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
    return (
        functional,
        tensors,
        parameters,
        constants,
        owner.bind(parameters, constants=constants),
    )


def _path(network):
    return f"parameters.populations.left.{G}" if network else f"parameters.{G}"


def _entries(report):
    return {entry.path: entry for entry in report.entries}


def _signature(report):
    return (
        tuple(
            (
                entry.path,
                entry.domain,
                id(entry.tensor),
                entry.aliases,
                entry.requires_grad,
                entry.is_leaf,
            )
            for entry in report.entries
        ),
        tuple(map(id, report.trainable_leaves)),
    )


@pytest.mark.parametrize("network", (False, True))
def test_lowered_bundle_reports_stored_sources_and_preserves_existing_selectors(
    network,
):
    _, tensors, _, _, _ = _case(network)
    selected = dict(tensors.trainable_parameters)
    report = dn.func.inspect_sources(tensors)
    expected = dn.func.inspect_sources(
        dn.func.Continuation(tensors.state, tensors.parameters, tensors.constants)
    )
    assert _signature(report) == _signature(expected)
    assert str(report) == str(expected)
    assert _entries(report)[_path(network)].tensor is _target(
        tensors.parameters, network
    )
    assert tuple(tensors.trainable_parameters) == tuple(selected)
    assert all(
        tensors.trainable_parameters[key] is value for key, value in selected.items()
    )
    if network:
        assert _entries(report)[_path(True)].aliases == (
            f"parameters.populations.right.{G}",
        )


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("compiled", (False, True))
def test_raw_binding_report_matches_its_successful_continuation(network, compiled):
    _, tensors, parameters, _, bound = _case(network, compiled=compiled)
    before = dn.func.inspect_sources(bound)
    _, auxiliary = dn.func.run(bound, tensors.state, steps=3)
    actual = dn.func.inspect_sources(auxiliary["continuation"])
    assert _signature(before) == _signature(actual)
    assert str(before) == str(actual)
    assert _entries(before)[_path(network)].tensor is _target(parameters, network)
    assert tuple(map(id, before.trainable_leaves)) == (
        id(_target(parameters, network)),
    )


@pytest.mark.parametrize("reader_first", (False, True))
def test_network_binding_reports_canonical_readers_without_rewriting_stored_frames(
    reader_first,
):
    model, key = _cross_population_binding(reader_first=reader_first)
    functional, tensors = dn.func.make_functional(model, dt=NETWORK_DT)
    parameters = tensors.independent_parameters(trainable=f"populations.left.{key}")
    constants = _plain(tensors.constants)
    packet = dn.func.NetworkTensors(parameters, constants, tensors.state)
    bound = functional.bind(parameters, constants=constants)
    canonical = parameters["populations"]["left"][key]
    authored = constants["populations"]["right"]["parametrizations.factor"]
    assert authored is not canonical
    parameter_path = f"parameters.populations.left.{key}"
    reader_path = "constants.populations.right.parametrizations.factor"

    stored = dn.func.inspect_sources(packet)
    effective = dn.func.inspect_sources(bound)
    assert _entries(stored)[reader_path].tensor is authored
    assert _entries(effective)[reader_path].tensor is canonical
    assert reader_path in _entries(effective)[parameter_path].aliases
    assert parameter_path in _entries(effective)[reader_path].aliases
    assert reader_path not in _entries(stored)[parameter_path].aliases
    assert sum(value is canonical for value in effective.trainable_leaves) == 1
    assert all(value is not authored for value in effective.trainable_leaves)
    assert constants["populations"]["right"]["parametrizations.factor"] is authored
    assert (
        packet.constants["populations"]["right"]["parametrizations.factor"] is authored
    )
    assert (
        bound.constants["populations"]["right"]["parametrizations.factor"] is authored
    )

    _, auxiliary = dn.func.run(bound, tensors.state, steps=1)
    assert _signature(effective) == _signature(
        dn.func.inspect_sources(auxiliary["continuation"])
    )


@pytest.mark.parametrize("network", (False, True))
def test_raw_inspection_preserves_warmed_cache_reports_rng_and_source_values(
    network, monkeypatch
):
    functional, tensors, _, _, bound = _case(network, compiled=True)
    with torch.no_grad():
        dn.func.run(bound, tensors.state, steps=2)
    cached = None if network else bound._inference_prepared
    if not network:
        assert cached is not None
    previous = bound.compiled_chunk.execution_report()
    values = _snapshot((bound.parameters, bound.constants, tensors.state))
    rng = torch.random.get_rng_state().clone()

    def reject(*args, **kwargs):
        raise AssertionError("inspection must not perform lifecycle work")

    for name in (
        "_validate_source",
        "_validate_parameters",
        "_validate_constants",
        "_validate_state",
        "prepare",
        "_prepare_values",
        "_prepare_source_validated",
        "prepare_and_rollout",
        "sample_runtime_draws",
        "_sample_missing_runtime_draws",
        "sample_noise_draw_tape",
        "_initialize_host" if network else "initialize_host",
    ):
        if hasattr(functional, name):
            monkeypatch.setattr(functional, name, reject)
    report = dn.func.inspect_sources(bound)
    assert report.entries and str(report)
    assert bound.compiled_chunk.execution_report() is previous
    if not network:
        assert bound._inference_prepared is cached
    _close((bound.parameters, bound.constants, tensors.state), values)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


def test_network_bundle_keeps_schedule_names_nonleaves_views_and_cross_frame_aliases():
    conductance = torch.ones(2, requires_grad=True)
    view = conductance.detach().requires_grad_()
    derived = conductance * 2
    schedule = dn.func.TensorScheduleTensors(conductance, derived, conductance, None)
    excluded = torch.ones(2, requires_grad=True)
    tensors = dn.func.NetworkTensors(
        {
            "netstim": {"schedule": schedule},
            "continuous_synapses": {"conductance": view},
        },
        {"conductance": conductance},
        {"clock": excluded},
    )
    report = dn.func.inspect_sources(tensors)
    assert [entry.path for entry in report.entries] == [
        "parameters.netstim.schedule.times",
        "parameters.netstim.schedule.amplitudes",
        "parameters.netstim.schedule.time_source",
        "parameters.continuous_synapses.conductance",
        "constants.conductance",
    ]
    assert tuple(map(id, report.trainable_leaves)) == (id(conductance), id(view))
    assert _entries(report)["parameters.netstim.schedule.amplitudes"].tensor is derived
    assert not _entries(report)["parameters.netstim.schedule.amplitudes"].is_leaf
    assert all(entry.tensor is not excluded for entry in report.entries)
    assert view is not conductance and view.data_ptr() == conductance.data_ptr()
    assert "constants.conductance" in report.entries[0].aliases


@pytest.mark.parametrize("network", (False, True))
def test_bundle_and_binding_reports_capture_flags_and_binding_container_ownership(
    network,
):
    _, tensors, parameters, constants, bound = _case(network)
    bundle = tensors._replace(parameters=parameters, constants=constants)
    old_bundle = dn.func.inspect_sources(bundle)
    old_bound = dn.func.inspect_sources(bound)
    original = _target(parameters, network)
    original.requires_grad_(False)
    assert _entries(old_bundle)[_path(network)].requires_grad
    assert _entries(old_bound)[_path(network)].requires_grad
    assert not _entries(dn.func.inspect_sources(bound))[_path(network)].requires_grad
    replacement = original.detach().clone().requires_grad_()
    owner = parameters["populations"]["left"] if network else parameters
    owner[G] = replacement
    assert (
        _entries(dn.func.inspect_sources(bundle))[_path(network)].tensor is replacement
    )
    assert _entries(dn.func.inspect_sources(bound))[_path(network)].tensor is original
    assert _entries(old_bundle)[_path(network)].tensor is original


@pytest.mark.parametrize("compiled", (False, True))
def test_prepared_population_inspection_rejects_without_touching_its_preparation(
    compiled, monkeypatch
):
    functional, _, parameters, constants, _ = _case(False)
    prepared = functional.prepare(parameters, constants)
    owner = (
        functional.compile_rollout_chunk(2, backend="eager") if compiled else functional
    )
    bound = owner.bind(parameters, prepared=prepared)

    def reject(*args, **kwargs):
        raise AssertionError("prepared inspection must not validate or prepare")

    monkeypatch.setattr(functional, "_validate_source", reject)
    monkeypatch.setattr(functional, "prepare", reject)
    with pytest.raises(
        dn.func.FunctionalizationError, match="raw binding|raw constants|constants="
    ):
        dn.func.inspect_sources(bound)
    assert bound.prepared is prepared and bound.constants is None


@pytest.mark.parametrize("base", (dn.func.BoundPopulation, dn.func.BoundNetwork))
def test_binding_subclasses_reject_before_reading_authored_source_properties(base):
    class Authored(base):
        @property
        def parameters(self):
            raise AssertionError("unsupported binding must not be read")

        @property
        def constants(self):
            raise AssertionError("unsupported binding must not be read")

    subject = object.__new__(Authored)
    with pytest.raises(TypeError, match="exact.*binding"):
        dn.func.inspect_sources(subject)


@pytest.mark.parametrize("network", (False, True))
def test_bundle_subclasses_do_not_read_state_or_initialization(network):
    base = dn.func.NetworkTensors if network else dn.func.PopulationTensors

    class Authored(base):
        @property
        def state(self):
            raise AssertionError("state is outside source inspection")

        @property
        def initialization(self):
            raise AssertionError("initialization is outside source inspection")

    conductance = torch.tensor(0.001, requires_grad=True)
    arguments = ({G: conductance}, {"dt": torch.tensor(DT)}, object())
    subject = Authored(*arguments) if network else Authored(*arguments, object())
    report = dn.func.inspect_sources(subject)
    assert [entry.path for entry in report.entries] == [
        f"parameters.{G}",
        "constants.dt",
    ]
    assert report.trainable_leaves[0] is conductance


def test_bundle_and_raw_binding_inspection_and_display_use_metadata_only():
    class NoValues(torch.Tensor):
        def __repr__(self):
            raise AssertionError("inspection must not render Tensor values")

        def __str__(self):
            raise AssertionError("inspection must not render Tensor values")

        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            if function.__name__ != "__get__":
                raise AssertionError(f"inspection called {function.__name__}")
            return super().__torch_function__(function, types, args, kwargs or {})

    class NoOperators(TorchDispatchMode):
        def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
            raise AssertionError(f"inspection dispatched {operation}")

    functional, tensors, parameters, constants, _ = _case(False)
    value = torch.Tensor._make_subclass(NoValues, parameters[G].detach(), True)
    parameters[G] = value
    bundle = tensors._replace(parameters=parameters, constants=constants)
    bound = functional.bind(parameters, constants=constants)
    with NoOperators():
        stored = dn.func.inspect_sources(bundle)
        effective = dn.func.inspect_sources(bound)
        assert str(stored) == repr(effective)
    assert _entries(stored)[_path(False)].tensor is value
    assert _entries(effective)[_path(False)].tensor is value


@pytest.mark.mps
def test_mps_explicit_sources_keep_live_devices_and_trainable_identity():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware is unavailable")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with _device_environment("mps"):
        model = _device_model("mps")
        functional, tensors = dn.func.make_functional(model, dt=0.01)
        parameters = tensors.independent_parameters(trainable="*g_param")
        bundle = tensors._replace(parameters=parameters)
        bound = functional.bind(parameters, constants=tensors.constants)
        stored, effective = (
            dn.func.inspect_sources(bundle),
            dn.func.inspect_sources(bound),
        )
        assert _signature(stored) == _signature(effective)
        target = next(
            value for name, value in parameters.items() if name.endswith("g_param")
        )
        assert target.device.type == "mps"
        assert any(value is target for value in effective.trainable_leaves)
        assert "mps" in str(effective)
