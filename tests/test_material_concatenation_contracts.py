"""Lightweight ownership, initialization, and topology contracts for chemistry."""

import pickle

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import (
    ClampProcess,
    ClearanceProcess,
    DiffusionProcess,
    ExchangeProcess,
)
from dendra.models.mod import pas
from dendra.models.multi import MultiPopulation

pytestmark = pytest.mark.cpu
DTYPE = torch.float64


class _RegionalClearance(ClearanceProcess):
    ClearanceProcess.RANGE(rate=0.5)
    ClearanceProcess.CLEAR("pool", field="c", rate="rate")


class _TensorClamp(ClampProcess):
    ClampProcess.GLOBAL(value=1.5)
    ClampProcess.SET(
        "pool", field="c", value="value", where=torch.tensor([[True, False, True]])
    )


class _RangedMaskClamp(ClampProcess):
    ClampProcess.RANGE(enabled=0.0)
    ClampProcess.SET("pool", field="c", value=0.0, where="enabled")


class _IonClearance(ClearanceProcess):
    ClearanceProcess.GLOBAL(rate=0.2)
    ClearanceProcess.CLEAR("na", field="nai", rate="rate")


class _IonClamp(ClampProcess):
    ClampProcess.SET("na", field="nai", value=12.0)


class _IonDiffusion(DiffusionProcess):
    DiffusionProcess.DIFFUSE("na", field="nai", D=0.5)
    DiffusionProcess.METHOD("implicit", solver="dense")


class _IonExchange(ExchangeProcess):
    ExchangeProcess.EXCHANGE("na.nai", "na.nao", rate=0.3, volume_a=1.0, volume_b=2.0)


class _MissingMaterial(ClearanceProcess):
    ClearanceProcess.CLEAR("unregistered_concat_test_material", field="c", rate=0.1)


class _RelaxPopulationSources(DiffusionProcess):
    DiffusionProcess.DIFFUSE("pool", field="c", D=0.5)
    DiffusionProcess.RELAX(
        "pool", field="c", rate="decay", target="reservoir", where="reservoir_mask"
    )


def _point(compartments=3):
    return dn.Population(N=1, C=compartments, integrator=dn.bwd_euler_sc(), dtype=DTYPE)


def _initialize(population):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population.train().initialize()
    return population


def _advance_processes(population, dt=0.1):
    for process in population.mech.material_processes.values():
        process.advance_materials(dt)
    return population.mech.materials["pool"].c


def test_missing_material_fields_keep_zero_storage_and_independent_minima():
    first, second, missing = _point(), _point(), _point()
    first.material("pool", fields={"c": 0.2}, min_values={"c": 0.7})
    second.material("pool", fields={"c": 0.3, "other": 2.0})
    packed = _initialize(MultiPopulation(first=first, second=second, missing=missing))
    torch.testing.assert_close(
        packed.mech.materials["pool"].c,
        torch.tensor([[0.7] * 3 + [0.3] * 3 + [0.0] * 3], dtype=DTYPE),
    )
    torch.testing.assert_close(
        packed.mech.materials["pool"].other,
        torch.tensor([[0.0] * 3 + [2.0] * 3 + [0.0] * 3], dtype=DTYPE),
    )
    assert packed._material_field_support[("pool", "c")].tolist() == [
        [True] * 6 + [False] * 3
    ]
    assert packed._material_field_support[("pool", "other")].tolist() == [
        [False] * 3 + [True] * 3 + [False] * 3
    ]


def test_pending_child_rebuild_uses_reconfigured_material_initial_source():
    child = _point(2)
    old_initial = torch.nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
    new_initial = torch.nn.Parameter(torch.tensor(2.0, dtype=DTYPE))
    child.material("pool", fields={"c": old_initial})
    _initialize(child)
    child.material("pool", fields={"c": new_initial})
    assert child._flag_rebuild

    packed = _initialize(MultiPopulation(child=child))
    actual = packed.mech.materials["pool"].c
    torch.testing.assert_close(actual, torch.full((1, 2), 2.0, dtype=DTYPE))
    new_gradient, old_gradient = torch.autograd.grad(
        actual.sum(), (new_initial, old_initial), allow_unused=True
    )
    torch.testing.assert_close(new_gradient, new_initial.new_tensor(2.0))
    assert old_gradient is None
    torch.testing.assert_close(_initialize(child).mech.materials["pool"].c, actual)


def test_regional_rate_parameter_and_tensor_where_remain_component_local():
    first, second = _point(), _point()
    rate = torch.nn.Parameter(torch.tensor([0.2, 0.7], dtype=DTYPE))
    first.material("pool", fields={"c": 2.0})
    second.material("pool", fields={"c": 3.0})
    first[:, [0, 2]].insert(_RegionalClearance, rate=rate)
    second.insert(_TensorClamp)
    _initialize(first)
    _initialize(second)
    expected = torch.cat(
        (_advance_processes(first), _advance_processes(second)), dim=-1
    )
    source_rate = next(iter(first.mech.material_processes.values())).rate_param
    expected_gradient = torch.autograd.grad(expected.sum(), source_rate)[0]
    packed = _initialize(MultiPopulation(first=first, second=second))
    actual = _advance_processes(packed)
    packed_rate = next(
        process.rate_param
        for process in packed.mech.material_processes.values()
        if isinstance(process, _RegionalClearance)
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), packed_rate)[0], expected_gradient
    )


@pytest.mark.parametrize("threads", (1, 4))
@pytest.mark.parametrize("process", (_RegionalClearance, _RangedMaskClamp))
def test_nonrectangular_local_operands_do_not_broadcast_as_component_columns(
    threads, process
):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        first = dn.Population(N=2, C=3, integrator=dn.bwd_euler_sc(), dtype=DTYPE)
        second = _point(2)
        first.material("pool", fields={"c": 1.0})
        second.material("pool", fields={"c": 1.0})
        clearance = process is _RegionalClearance
        name = "rate" if clearance else "enabled"
        first_values = torch.nn.Parameter(
            torch.tensor([0.1, 0.2] if clearance else [1.0, 0.0], dtype=DTYPE)
        )
        last_value = torch.nn.Parameter(
            torch.tensor([0.3] if clearance else [1.0], dtype=DTYPE)
        )
        first[0, [0, 2]].insert(process, **{name: first_values})
        first[1, [1]].insert(process, **{name: last_value})
        packed = _initialize(MultiPopulation(first=first, second=second))
        actual = _advance_processes(packed)
        expected = torch.ones_like(actual)
        if clearance:
            expected[0, [0, 2]] = torch.exp(-0.1 * first_values)
            expected[0, 4] = torch.exp(-0.1 * last_value[0])
            operator = next(iter(packed.mech.material_processes.values()))
            local_rates = operator.rate
            actual_gradient = torch.autograd.grad(actual.sum(), local_rates)[0]
            torch.testing.assert_close(
                actual_gradient, -0.1 * torch.exp(-0.1 * local_rates)
            )
        else:
            expected[0, [0, 4]] = 0.0
        torch.testing.assert_close(actual, expected)
    finally:
        torch.set_num_threads(previous)


def test_ion_initial_sources_keep_values_and_original_parameter_gradients():
    first, second = _point(2), _point(3)
    initial = torch.nn.Parameter(torch.tensor(12.0, dtype=DTYPE))
    first.concentrations(nai0=initial)
    second.concentrations(nai0=23.0)
    first.insert(_IonClearance)
    second.insert(_IonClearance)
    packed = _initialize(MultiPopulation(first=first, second=second))
    concentration = packed.mech.ions["na"].nai
    torch.testing.assert_close(
        concentration, torch.tensor([[12.0, 12.0, 23.0, 23.0, 23.0]], dtype=DTYPE)
    )
    torch.testing.assert_close(
        torch.autograd.grad(concentration.sum(), initial)[0], initial.new_tensor(2.0)
    )


@pytest.mark.parametrize("configured", (False, True))
@pytest.mark.parametrize(
    ("field", "source_name", "value"),
    (("nai", "i_init", 17.0), ("nao", "o_init", 123.0), ("ena", "e_init", 45.0)),
)
def test_trained_ion_initial_sources_survive_concatenation(
    configured, field, source_name, value
):
    child = _point(2)
    child.insert(_IonClearance)
    if configured:
        child.concentrations(nai0=12.0, nao0=130.0)
        child.equilibria(ena=55.0)
    _initialize(child)
    source = getattr(child.mech.ions["na"], source_name)
    with torch.no_grad():
        source.fill_(value)
    packed = _initialize(MultiPopulation(child=child, empty=_point(1)))
    actual = getattr(packed.mech.ions["na"], field)
    torch.testing.assert_close(actual[:, :2], source.new_full((1, 2), value))
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), source)[0], source.new_tensor(2.0)
    )


@pytest.mark.parametrize("setter", ("concentrations", "equilibria"))
@pytest.mark.parametrize("parameter", (False, True))
def test_ion_reconfiguration_replaces_sources_without_rebuilding_mechanisms(
    setter, parameter
):
    child = _point(2)
    child.insert(_IonClearance)
    child.insert(pas)
    _initialize(child)
    conductance = child.mech.mechanisms["pas"].g_param
    with torch.no_grad():
        conductance.fill_(0.003)
    value = 22.123456789012345
    initial = (
        torch.nn.Parameter(torch.tensor(value, dtype=DTYPE)) if parameter else value
    )
    field, keyword = ("nai", "nai0") if setter == "concentrations" else ("ena", "ena")
    getattr(child, setter)(**{keyword: initial})
    assert not child._flag_rebuild
    source = getattr(child.mech.ions["na"], "i_init" if field == "nai" else "e_init")
    assert source.dtype == child.dtype() and source.device == child.device()
    assert source.requires_grad
    if parameter:
        assert source is initial
    packed = _initialize(MultiPopulation(child=child))
    actual = getattr(packed.mech.ions["na"], field)
    torch.testing.assert_close(actual, source.new_full((1, 2), value), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), source)[0], source.new_tensor(2.0)
    )
    torch.testing.assert_close(
        getattr(_initialize(child).mech.ions["na"], field), actual
    )
    assert child.mech.mechanisms["pas"].g_param is conductance
    torch.testing.assert_close(conductance, conductance.new_tensor(0.003))


@pytest.mark.parametrize(
    "process", (_IonClearance, _IonClamp, _IonDiffusion, _IonExchange)
)
def test_builtin_process_can_be_the_only_ion_user(process):
    child = _point()
    child.insert(process)
    _initialize(child)
    assert "na" in child.mech.ions
    packed = _initialize(MultiPopulation(child=child, empty=_point(2)))
    assert "na" in packed.mech.ions
    torch.testing.assert_close(
        packed.mech.ions["na"].nai, torch.full((1, 5), 10.0, dtype=DTYPE)
    )


def test_undeclared_nonionic_process_target_fails_clearly():
    child = _point()
    child.insert(_MissingMaterial)
    with pytest.raises(ValueError, match="no (registered )?fields"):
        _initialize(child)
    with pytest.raises(ValueError, match="no fields"):
        MultiPopulation(child=child, empty=_point(2))


def test_named_geometry_helper_tracks_reinitialization_after_dtype_move():
    first, second = _point(), _point(2)
    volume = torch.nn.Parameter(torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE))
    first.register_material_geometry(
        "chemical",
        domain="intracellular",
        volume=volume,
        edge_factor=torch.ones(2, dtype=DTYPE),
    )
    packed = MultiPopulation(first=first, second=second).float()
    source = first.material_geometry("chemical")["edge_factor"]
    getattr(first, source).fill_(3.0)
    _initialize(packed)
    values = packed._material_geometry(packed.diam, packed.dx)
    for name, value in values.items():
        torch.testing.assert_close(value, getattr(packed, name))
    edge_name = packed.material_geometry("packed_0_chemical")["edge_factor"]
    torch.testing.assert_close(values[edge_name], torch.tensor([[3.0, 3.0, 0.0]]))
    packed.batch(2)
    _initialize(packed)
    for name, value in packed._material_geometry(packed.diam, packed.dx).items():
        torch.testing.assert_close(value, getattr(packed, name))


def test_foreign_same_size_forest_checkpoint_is_rejected_without_replacing_topology():
    target = MultiPopulation(first=_point(2), second=_point(3))
    foreign = MultiPopulation(first=_point(3), second=_point(2))
    expected = target.diff_parent_index.clone()
    with pytest.raises(RuntimeError, match="component chemical forest"):
        target.load_state_dict(foreign.state_dict())
    torch.testing.assert_close(target.diff_parent_index, expected)


def test_scoped_process_class_roundtrip_preserves_component_and_declarations():
    child = _point()
    child.material("pool", fields={"c": 1.0})
    child.insert(_RegionalClearance)
    first = MultiPopulation(child=child)
    second = MultiPopulation(child=child)
    first_class = next(iter(first._mech_data))
    second_class = next(iter(second._mech_data))
    assert first_class is not second_class
    restored = pickle.loads(pickle.dumps(first_class))
    assert restored._material_process_component == "child"
    assert restored._material_process_original_class is _RegionalClearance
    assert restored._clearance_specs == _RegionalClearance._clearance_specs
    assert (
        first_class._material_process_geometry_names
        is not second_class._material_process_geometry_names
    )


def test_population_owned_relaxation_sources_are_scoped_without_detaching():
    child = _point()
    child.material("pool", fields={"c": 1.0})
    child.register_parameter(
        "decay", torch.nn.Parameter(torch.tensor(0.2, dtype=DTYPE))
    )
    child.register_buffer("reservoir", torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE))
    child.register_buffer("reservoir_mask", torch.tensor([True, False, True]))
    child.insert(_RelaxPopulationSources)
    packed = MultiPopulation(child=child, empty=_point(2))
    scoped = next(iter(packed._mech_data))
    for source in ("decay", "reservoir", "reservoir_mask"):
        mapped = scoped._material_process_quantity_names[source]
        assert getattr(packed, mapped) is getattr(child, source)


@pytest.mark.parametrize("keyword", ("fields", "initial_values"))
def test_parent_material_reconfiguration_redeclares_only_touched_fields(keyword):
    first, missing = _point(), _point(2)
    first.material(
        "pool",
        fields={"c": 1.0, "other": 2.0},
        min_values={"c": 0.1},
        domain="intracellular",
        units="mM",
    )
    packed = MultiPopulation(first=first, missing=missing)
    packed.material("pool", **{keyword: {"c": 4.0}})
    _initialize(packed)
    torch.testing.assert_close(
        packed.mech.materials["pool"].c, torch.full((1, 5), 4.0, dtype=DTYPE)
    )
    torch.testing.assert_close(
        packed.mech.materials["pool"].other,
        torch.tensor([[2.0, 2.0, 2.0, 0.0, 0.0]], dtype=DTYPE),
    )
    assert packed._material_field_support[("pool", "c")].all()
    assert packed._material_field_support[("pool", "other")].tolist() == [
        [True, True, True, False, False]
    ]
    assert packed.mech.materials["pool"].field_spec("c").domain == "intracellular"
    assert packed.mech.materials["pool"].field_spec("c").units == "mM"


def test_parent_owned_named_geometry_is_independent_of_component_adapter():
    packed = MultiPopulation(first=_point(), second=_point(2))
    volume = torch.nn.Parameter(torch.arange(1.0, 6.0, dtype=DTYPE))
    packed.register_material_geometry(
        "shared",
        domain="intracellular",
        volume=volume,
        edge_factor=torch.ones(3, dtype=DTYPE),
    )
    config = packed.material_geometry("shared")
    _initialize(packed)
    assert getattr(packed, config["volume"]) is volume
    assert config["volume"] not in packed._material_geometry(packed.diam, packed.dx)
    torch.testing.assert_close(
        torch.autograd.grad(getattr(packed, config["volume"]).sum(), volume)[0],
        torch.ones_like(volume),
    )


@pytest.mark.parametrize("threads", (1, 4))
def test_explicit_ion_style_preserves_full_concentration_reversal_traces_and_gradients(
    threads,
):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        first, second = _point(2), _point(3)
        sources = (
            torch.nn.Parameter(torch.tensor(12.0, dtype=DTYPE)),
            torch.nn.Parameter(torch.tensor(23.0, dtype=DTYPE)),
        )
        for child, source in zip((first, second), sources):
            child.concentrations(nai0=source)
            child.ion_style("na", 1, 1)
            child.insert(_IonClearance)

        def trace(population):
            _initialize(population)
            frames = []
            ion = population.mech.ions["na"]
            for step in range(4):
                frames.append(torch.stack((ion.nai, ion.ena)))
                if step != 3:
                    for process in population.mech.material_processes.values():
                        process.advance_materials(0.1)
                    ion.advance(population.celsius)
            return torch.stack(frames)

        expected = torch.cat((trace(first), trace(second)), dim=-1)
        expected_gradients = torch.autograd.grad(expected.sum(), sources)
        packed = MultiPopulation(first=first, second=second, celsius=first.celsius)
        actual = trace(packed)
        torch.testing.assert_close(actual, expected)
        for actual_gradient, expected_gradient in zip(
            torch.autograd.grad(actual.sum(), sources), expected_gradients
        ):
            torch.testing.assert_close(actual_gradient, expected_gradient)
    finally:
        torch.set_num_threads(old_threads)


def test_conflicting_effective_ion_styles_reject_instead_of_changing_reversals():
    first, second = _point(), _point()
    first.insert(_IonClearance)
    second.insert(_IonClearance)
    first.ion_style("na", 1, 1)
    with pytest.raises(ValueError, match="incompatible effective ion_style"):
        MultiPopulation(first=first, second=second)
