"""Native models expose live raw sources without entering a model lifecycle."""

from __future__ import annotations

import os
import re

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.stim.waveform.core import Waveform
from tests.func.test_explicit_source_inspection import _signature
from tests.func.test_lowering_api import DT, G, _network, _population
from tests.func.test_network_connection_parameters import _network as _parameter_network
from tests.func.test_network_initialization_bindings import _cross_population_binding
from tests.func.test_stimulation_rng_sharing import (
    _population as _stochastic_population,
)
from tests.networks.test_native_shared_schedule_bindings import _scheduled_network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    rng = torch.get_rng_state().clone()
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield
    torch.set_rng_state(rng)


def _entries(report):
    return {entry.path: entry for entry in report.entries}


def _tensor_ids(values):
    return tuple(map(id, values))


def _native_path(network):
    return f"parameters.left.{G}" if network else f"parameters.{G}"


@pytest.mark.parametrize("network", (False, True))
def test_native_reports_keep_registration_paths_live_objects_and_optimizer_leaves(
    network,
):
    model = _network() if network else _population()
    parameters = tuple(model.named_parameters(remove_duplicate=False))
    report = dn.func.inspect_sources(model)
    actual = tuple(entry for entry in report.entries if entry.domain == "parameters")
    assert tuple(entry.path for entry in actual) == tuple(
        "parameters." + name for name, _ in parameters
    )
    assert all(
        entry.tensor is tensor
        for entry, (_name, tensor) in zip(actual, parameters, strict=True)
    )
    expected, seen = [], set()
    for _name, value in parameters:
        if id(value) not in seen and value.requires_grad and value.is_leaf:
            expected.append(value)
        seen.add(id(value))
    assert _tensor_ids(report.trainable_leaves) == _tensor_ids(expected)
    source = model.left.mech.leak.g_param if network else model.mech.leak.g_param
    entry = _entries(report)[_native_path(network)]
    assert entry.tensor is source and entry.requires_grad and entry.is_leaf
    if network:
        assert f"parameters.right.{G}" in entry.aliases
        assert sum(value is source for value in report.trainable_leaves) == 1
    for name, population in model.populations.items() if network else (("", model),):
        prefix = name + "." if name else ""
        assert _entries(report)[f"constants.{prefix}diam"].tensor is population.diam
        assert _entries(report)[f"constants.{prefix}dx"].tensor is population.dx
        excluded = (population.v, population.t, population._duration_remainder)
        assert all(
            entry.tensor is not value for entry in report.entries for value in excluded
        )
    assert str(report).startswith("SourceReport\n")
    assert "trainable leaf" in str(report)


@pytest.mark.parametrize("network", (False, True))
def test_uninitialized_native_inspection_reports_registered_sources_without_building(
    network, monkeypatch
):
    model = _network(initialize=False) if network else _population(initialize=False)
    parameters = tuple(model.named_parameters(remove_duplicate=False))
    namespaces = {id(module): tuple(vars(module)) for module in model.modules()}

    def forbidden(*args, **kwargs):
        raise AssertionError("native inspection must not build or initialize")

    monkeypatch.setattr(type(model), "initialize", forbidden)
    monkeypatch.setattr(type(model), "build", forbidden)
    report = dn.func.inspect_sources(model)
    assert tuple(
        entry.path for entry in report.entries if entry.domain == "parameters"
    ) == tuple("parameters." + name for name, _ in parameters)
    assert {id(module): tuple(vars(module)) for module in model.modules()} == namespaces


def test_alias_module_paths_are_complete_and_ancestor_cycles_are_stopped():
    model = _population()
    model._modules["alias"] = model.integrator
    source = model.mech.leak.g_param
    expected = tuple(
        "parameters." + name
        for name, value in model.named_parameters(remove_duplicate=False)
        if value is source
    )
    model.integrator._modules["parent_cycle"] = model
    report = dn.func.inspect_sources(model)
    original = f"parameters.{G}"
    alias = "parameters.alias.mech.mechanisms.leak.g_param"
    entries = _entries(report)
    assert entries[original].tensor is entries[alias].tensor is model.mech.leak.g_param
    assert alias in entries[original].aliases and original in entries[alias].aliases
    assert not any("parent_cycle" in entry.path for entry in report.entries)
    selected = report.select(alias)
    assert tuple(entry.path for entry in selected.entries) == expected
    assert selected.trainable_leaves == (source,)


def test_cross_domain_aliases_selection_and_captured_flags_use_native_paths():
    model, key = _cross_population_binding()
    source = model.right.factor
    source.requires_grad_(True)
    report = dn.func.inspect_sources(model)
    parameter_path = f"parameters.left.{key}"
    constant_path = "constants.right.factor"
    entries = _entries(report)
    assert entries[parameter_path].tensor is entries[constant_path].tensor is source
    assert constant_path in entries[parameter_path].aliases
    assert parameter_path in entries[constant_path].aliases
    assert sum(value is source for value in report.trainable_leaves) == 1
    source.requires_grad_(False)
    selected = report.select(re.compile(r"^constants\.right\.factor$"))
    assert parameter_path in _entries(selected) and constant_path in _entries(selected)
    assert selected.trainable_leaves == (source,)
    assert all(entry.requires_grad and entry.is_leaf for entry in selected.entries)
    refreshed = dn.func.inspect_sources(model).select(constant_path)
    assert refreshed.trainable_leaves == ()
    assert all(not entry.requires_grad for entry in refreshed.entries)


def test_connection_and_torch_parametrization_raw_sources_keep_all_native_aliases():
    model, shared = _parameter_network(parametrized=True)
    report = dn.func.inspect_sources(model)
    original = shared.parametrizations.rho.original
    factor = shared.parametrizations.rho[0].factor
    offset = shared.parametrizations.rho[0].offset
    root_path = "parameters.cell.cm_param.parametrizations.rho.original"
    assert _entries(report)[root_path].tensor is original
    selected = report.select(root_path)
    assert any(
        entry.path.startswith("parameters.synapses.") for entry in selected.entries
    )
    assert any(
        entry.path.startswith("parameters.continuous_synapses.")
        for entry in selected.entries
    )
    assert selected.trainable_leaves == (original,)
    assert any(entry.tensor is factor for entry in report.entries)
    offset_entries = tuple(entry for entry in report.entries if entry.tensor is offset)
    assert offset_entries and all(
        entry.domain == "constants" for entry in offset_entries
    )
    assert all(
        entry.path.endswith("parametrizations.rho.0.offset") for entry in offset_entries
    )
    effective = tuple(
        value
        for collection in (model.synapses, model.continuous_synapses)
        for connection in collection.values()
        for value in (
            connection.weight.w,
            connection.delay_ms.w,
            connection.delivery_buffer,
        )
    )
    assert all(
        entry.tensor is not value for entry in report.entries for value in effective
    )


def test_indexed_schedules_expose_shared_sources_and_exclude_event_runtime():
    model, times, amplitudes = _scheduled_network()
    report = dn.func.inspect_sources(model)
    entries = _entries(report)
    for name, value in (("_sched_t_source", times), ("_sched_w_source", amplitudes)):
        path = f"parameters.netstim.{name}"
        assert entries[path].tensor is value
        aliases = report.select(path)
        assert any(
            entry.path.startswith("parameters.synapses.") for entry in aliases.entries
        )
        assert aliases.trainable_leaves == (value,)
    excluded = [
        model.netstim.next_stoch_time,
        model.netstim.spike_gate,
        model.netstim.t_last,
    ]
    excluded.extend(
        connection.delivery_buffer for connection in model.synapses.values()
    )
    assert all(
        entry.tensor is not value for entry in report.entries for value in excluded
    )


class _OffsetWaveform(Waveform):
    Waveform.PARAMETER(amplitude=1e-6)

    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.tensor(1e-7))

    def fn(self, time):
        return self.amplitude + self.offset + torch.zeros_like(time)


def test_registered_intracellular_waveform_sources_are_read_without_evaluation(
    monkeypatch,
):
    waveform = _OffsetWaveform()
    model = _population(initialize=False)
    model[:].inject(waveform)
    model.initialize(DT)

    def forbidden(*args, **kwargs):
        raise AssertionError("native inspection must not evaluate waveforms")

    monkeypatch.setattr(waveform, "forward", forbidden)
    monkeypatch.setattr(waveform, "fn", forbidden)
    report = dn.func.inspect_sources(model)
    amplitude = tuple(
        entry for entry in report.entries if entry.tensor is waveform.amplitude
    )
    offset = tuple(entry for entry in report.entries if entry.tensor is waveform.offset)
    assert amplitude and all(entry.domain == "parameters" for entry in amplitude)
    assert offset and all(entry.domain == "constants" for entry in offset)
    selected = report.select(amplitude[0].path)
    assert selected.trainable_leaves == (waveform.amplitude,)


def test_stochastic_state_samples_draws_and_poisson_history_are_not_sources():
    model, extra, waveform = _stochastic_population()
    assert extra is None
    probe = model.mech.probe
    gate = probe.DE._SharedStochasticGate
    generator = waveform.generator
    rng = generator.get_state().clone()
    excluded = (
        probe.offset,
        probe.eta,
        probe.x,
        gate.forcing,
        waveform._spike_times,
        model.v,
    )
    report = dn.func.inspect_sources(model)
    assert all(
        entry.tensor is not value for entry in report.entries for value in excluded
    )
    assert any(entry.tensor is gate.theta_param for entry in report.entries)
    torch.testing.assert_close(generator.get_state(), rng, rtol=0, atol=0)


class _NoTensorOperations(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        raise AssertionError(f"native source inspection performed {function}")


@pytest.mark.parametrize("network", (False, True))
@pytest.mark.parametrize("check_purity", (False, True))
def test_native_inspection_does_not_lower_validate_get_values_or_run_hooks(
    network, check_purity, monkeypatch
):
    import dendra.func._lowering as lowering

    model = _network() if network else _population()
    expected = dn.func.inspect_sources(model)
    modules = tuple(model.modules())
    rng = torch.get_rng_state().clone()
    versions = {id(value): value._version for value in model.parameters()}

    def forbidden(*args, **kwargs):
        raise AssertionError("native inspection must read registered metadata only")

    for module in modules:
        for name in (
            "named_parameters",
            "named_buffers",
            "named_modules",
            "parameters",
            "buffers",
            "children",
            "modules",
            "forward",
        ):
            monkeypatch.setattr(module, name, forbidden)
        for name in (
            "_compute",
            "initialize",
            "build",
            "populate_parameter_buffers",
            "derive_buffers",
        ):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(lowering, "make_functional", forbidden)
    monkeypatch.setattr(lowering, "_lower_network", forbidden)
    monkeypatch.setattr(dn.func.FunctionalPopulation, "__init__", forbidden)
    monkeypatch.setattr(dn.func.FunctionalNetwork, "__init__", forbidden)
    with dn.ctx(CHECK_PURITY=check_purity), _NoTensorOperations():
        actual = dn.func.inspect_sources(model)
        selected = actual.select("glob:*g_param")
        assert str(selected).startswith("SourceReport\n")
    assert _signature(actual) == _signature(expected)
    assert all(
        entry.tensor._version == versions[id(entry.tensor)]
        for entry in actual.entries
        if id(entry.tensor) in versions
    )
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_packed_population_inspection_uses_registered_component_source_paths():
    from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
    from tests.func.test_multi_population import _mixed_multi

    if not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("mixed MultiPopulation requires dendra-solvers")
    model = _mixed_multi()
    expected = tuple(model.named_parameters(remove_duplicate=False))
    report = dn.func.inspect_sources(model)
    parameters = tuple(
        entry for entry in report.entries if entry.domain == "parameters"
    )
    assert tuple(entry.path for entry in parameters) == tuple(
        "parameters." + name for name, _ in expected
    )
    assert all(
        entry.tensor is value
        for entry, (_name, value) in zip(parameters, expected, strict=True)
    )
    assert any(
        entry.path.startswith("parameters.populations.point.") for entry in parameters
    )
    assert any(
        entry.path.startswith("parameters.populations.tree.") for entry in parameters
    )
    assert all(entry.tensor is not model.v for entry in report.entries)
    constants = _entries(report)
    for name in ("diam", "dx"):
        assert f"constants.{name}" not in constants
        for component_name, component in model.populations.items():
            assert constants[
                f"constants.populations.{component_name}.{name}"
            ].tensor is getattr(component, name)


def test_native_inspection_leaves_existing_tensor_bundle_and_packet_reports_unchanged():
    model = _network()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    continuation = dn.func.Continuation(
        tensors.state, tensors.parameters, tensors.constants
    )
    binding = functional.bind(continuation)
    expected = tuple(
        _signature(dn.func.inspect_sources(subject))
        for subject in (tensors, continuation, binding)
    )
    native = dn.func.inspect_sources(model)
    assert _native_path(True) in _entries(native)
    for subject, previous in zip(
        (tensors, continuation, binding), expected, strict=True
    ):
        report = dn.func.inspect_sources(subject)
        assert _signature(report) == previous
        assert f"parameters.populations.left.{G}" in _entries(report)


def test_ion_and_material_initializer_buffers_are_live_sources_without_evaluation(
    monkeypatch,
):
    from dendra.models.mechanisms._ions import Ion
    from dendra.models.mechanisms._materials import Material
    from tests.func.test_initialization_sources_plan import _AffineSource

    model = _population()
    source = _AffineSource()
    ion = Ion("na", model.shape, einit=1, eadvance=1)
    del ion.o_init
    ion.o_init = source
    material = Material(
        "pool", model.shape, fields={"amount": source, "reserve": source}
    )
    model.mech.ions["na"] = ion
    model.mech.materials["pool"] = material

    def forbidden(*args, **kwargs):
        raise AssertionError("inspection must not evaluate initialization sources")

    monkeypatch.setattr(source, "forward", forbidden)
    report = dn.func.inspect_sources(model)
    offset = tuple(entry for entry in report.entries if entry.tensor is source.offset)
    assert offset and all(entry.domain == "constants" for entry in offset)
    for suffix in (
        ".ions.na.o_init.offset",
        ".materials.pool._initial_sources.amount.offset",
        ".materials.pool._initial_sources.reserve.offset",
    ):
        assert any(entry.path.endswith(suffix) for entry in offset)
    selected = report.select(offset[0].path)
    assert tuple(entry.path for entry in selected.entries) == tuple(
        entry.path for entry in offset
    )
    current = (
        *(ion._buffers[name] for name in ion._material_fields),
        *(
            ion._buffers[name + "_"]
            for name in ion._material_fields
            if name + "_" in ion._buffers
        ),
        material.amount,
        material.reserve,
    )
    assert all(
        entry.tensor is not value for entry in report.entries for value in current
    )
    for name in ("e_init", "i_init"):
        initializer = getattr(ion, name)
        assert isinstance(initializer, torch.Tensor)
        domain = "parameters" if name in ion._parameters else "constants"
        assert any(
            entry.tensor is initializer
            and entry.domain == domain
            and entry.path.endswith(f".ions.na.{name}")
            for entry in report.entries
        )
    assert any(entry.tensor is source.value for entry in report.entries)


@pytest.mark.parametrize("learnable", (False, True))
def test_table_sources_include_axis_and_ordinates_without_interpolation_workspaces(
    learnable, monkeypatch
):
    from tests.func.test_table_execution import (
        _FixedTableLeak,
        _LearnableTableLeak,
    )
    from tests.func.test_table_execution import _population as table_population

    model = table_population(_LearnableTableLeak if learnable else _FixedTableLeak)
    table = model.mech.tabulated.curve_table
    axis = table._x_search
    ordinates = table.y if learnable else table._y
    derived = tuple(
        value
        for name, value in table._buffers.items()
        if name not in {"_x_search", "_y"}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("inspection must not interpolate a TABLE")

    monkeypatch.setattr(table, "forward", forbidden)
    report = dn.func.inspect_sources(model)
    axis_entries = tuple(entry for entry in report.entries if entry.tensor is axis)
    y_entries = tuple(entry for entry in report.entries if entry.tensor is ordinates)
    assert axis_entries and all(entry.domain == "constants" for entry in axis_entries)
    assert y_entries and all(
        entry.domain == ("parameters" if learnable else "constants")
        for entry in y_entries
    )
    assert all(
        entry.tensor is not value for entry in report.entries for value in derived
    )
    selected = report.select(y_entries[0].path)
    assert selected.trainable_leaves == ((ordinates,) if learnable else ())


def test_runtime_slot_alias_does_not_hide_or_publish_its_raw_configuration_source():
    from tests.func.test_api import _InplaceBufferMechanism
    from tests.func.test_network_initialization_bindings import _ScaleByArgument

    model = _population(initialize=False)
    model.insert(_InplaceBufferMechanism.rename("carry"))
    model.register_buffer("factor", torch.ones(model.shape))
    model.register_parametrization_in_graph("cm", _ScaleByArgument(), args=("factor",))
    model.initialize(DT)
    mechanism = model.mech.carry
    mechanism._buffers["gain"] = model.factor
    report = dn.func.inspect_sources(model)
    matching = tuple(entry for entry in report.entries if entry.tensor is model.factor)
    assert matching and "constants.factor" in {entry.path for entry in matching}
    assert all(not entry.path.endswith(".gain") for entry in matching)
    assert not any(entry.path.endswith(".carry.gain") for entry in report.entries)


class _GetterGuardPopulation(dn.SingleCompartment):
    def __getattribute__(self, name):
        if name == "__class__":
            raise AssertionError("inspection must not read an authored __class__")
        return super().__getattribute__(name)


def test_native_subclass_inspection_bypasses_authored_class_and_metadata_getters(
    monkeypatch,
):
    model = _population()
    expected = dn.func.inspect_sources(model)
    original_class = type(model)
    class_slot = vars(object)["__class__"]

    def forbidden(*args):
        raise AssertionError("inspection must not evaluate authored metadata getters")

    # Parameter declaration dictionaries must be ordinary mappings while the
    # Population subclass is created. Install hostile descriptors only after
    # that normal class-construction lifecycle has completed.
    monkeypatch.setattr(_GetterGuardPopulation, "_global", property(forbidden))
    monkeypatch.setattr(_GetterGuardPopulation, "area", property(forbidden))
    try:
        class_slot.__set__(model, _GetterGuardPopulation)
        with _NoTensorOperations():
            report = dn.func.inspect_sources(model)
    finally:
        class_slot.__set__(model, original_class)
    assert _signature(report) == _signature(expected)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_native_metadata_inspection_performs_no_tensor_operations():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
    with dn.ctx(DEVICE="mps", DTYPE=torch.float32):
        model = _population()
        source = model.mech.leak.g_param
        with _NoTensorOperations():
            report = dn.func.inspect_sources(model)
            selected = report.select("glob:*g_param")
            assert str(selected).startswith("SourceReport\n")
        assert selected.trainable_leaves == (source,)
        assert all(entry.tensor.device.type == "mps" for entry in report.entries)


def test_custom_material_process_configuration_buffer_is_a_live_trainable_source(
    monkeypatch,
):
    from tests.func.test_custom_material_processes import (
        _population as material_population,
    )

    model = material_population(compartments=2, physiology=False)
    process = model.mech.material_processes["local"]
    process.offset.requires_grad_(True)
    raw = process.offset
    current = model.mech.materials.pool.amount, model.mech.materials.pool.reserve

    def forbidden(*args, **kwargs):
        raise AssertionError("inspection must not prepare or query custom materials")

    monkeypatch.setattr(process, "prepare_materials", forbidden)
    monkeypatch.setattr(type(process), "material_fields", property(forbidden))
    with _NoTensorOperations():
        report = dn.func.inspect_sources(model)
        selected = report.select("glob:constants.*material_processes.local.offset")
    assert selected.entries and all(entry.tensor is raw for entry in selected.entries)
    assert all(
        entry.domain == "constants" and entry.requires_grad and entry.is_leaf
        for entry in selected.entries
    )
    assert selected.trainable_leaves == (raw,)
    assert all(
        entry.tensor is not value for entry in report.entries for value in current
    )
