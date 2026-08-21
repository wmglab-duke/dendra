"""Contracts for opt-in population-shaped sparse mechanism storage.

Only ordinary distributed mechanisms installed on the same ordered compartment
columns in every population row may opt into ``(N, K)`` local storage.  Every
other support and mechanism category must retain its legacy packed ABI.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import (
    MaterialProcess,
    Mechanism,
    PointProcess,
    State,
    VoltageProcess,
)
from dendra.models.mechanisms._support import SupportKind

DTYPE = torch.float64
N_POPULATIONS = 3
N_COMPARTMENTS = 6
COLUMNS = torch.tensor([1, 4], dtype=torch.long)
N_LOCAL_COLUMNS = int(COLUMNS.numel())
EXPLICIT_BATCH = 4


class _AxisState(State):
    State.STATE("x")
    State.RANGE(rate=0.15)
    State.BATCH(state_scale=1.0)
    State.DERIVATIVE("x' = -state_scale * rate * x")


class _AxisDensity(Mechanism):
    Mechanism.RANGE(g=2.0e-4, e=-52.0)
    Mechanism.BATCH(scale=1.0)
    Mechanism.STATE(_AxisState)
    Mechanism.INIT(x=0.4)
    Mechanism.BUFFER("scratch")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.SAVE("i")
    Mechanism.AFFINE("i")

    def initial(self, v):
        self.scratch = torch.zeros_like(v)

    def breakpoint(self, v):
        self.scratch = 0.5 * v

    def i(self, v):
        return self.scale * self.g * self.x * (v - self.e)


class _PopulationBatchProbe(Mechanism):
    # A real per-population default catches accidental collapse back to (1,).
    Mechanism.BATCH(scale=torch.tensor([[0.5], [1.0], [1.5]], dtype=DTYPE))


class _AxisPoint(PointProcess):
    PointProcess.RANGE(g=1.0e-4)
    PointProcess.BATCH(scale=1.0)
    PointProcess.NONSPECIFIC_CURRENT("i")
    PointProcess.AFFINE("i")

    def i(self, v):
        return self.scale * self.g * v


class _AxisMaterialProcess(MaterialProcess):
    MaterialProcess.RANGE(rate=0.1)
    MaterialProcess.BATCH(scale=1.0)


class _AxisVoltageProcess(VoltageProcess):
    VoltageProcess.RANGE(offset=0.0)
    VoltageProcess.BATCH(scale=1.0)

    def update_v(self, v):
        return v + self.scale * self.offset


class _AxisSensitiveDensity(Mechanism):
    supports_population_axis_layout = False
    Mechanism.RANGE(g=1.0)


class _FlatDefaultDensity(Mechanism):
    Mechanism.RANGE(value=torch.arange(6, dtype=DTYPE))


class _FlatInitDensity(Mechanism):
    Mechanism.INIT(x=torch.arange(6, dtype=DTYPE))


class _NoBatchDensity(Mechanism):
    Mechanism.RANGE(value=0.0)


class _MechanismBatchOnly(Mechanism):
    Mechanism.BATCH(scale=1.0)


class _NestedBatchState(State):
    State.STATE("x")
    State.BATCH(scale=1.0)
    State.DERIVATIVE("x' = -scale * x")


class _NestedBatchOnly(Mechanism):
    Mechanism.STATE(_NestedBatchState)
    Mechanism.INIT(x=1.0)


class _BatchRandOnly(Mechanism):
    Mechanism.BATCHRAND("sample", distribution="normal", mu=0.0, sigma=1.0, seed=7)


class _BatchNoiseOnly(Mechanism):
    Mechanism.BATCHNOISE("noise", distribution="normal", mu=0.0, sigma=1.0, seed=11)


def _mechanism(population, cls):
    name = cls._name or cls.__name__
    return getattr(population.mech, name)


def _shared_column_population(*, preserve=True, batch_size=None, cls=_AxisDensity):
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        v_init=-65.0,
        dtype=DTYPE,
        integrator=dn.euler(),
        preserve_mechanism_population_axis=preserve,
    )
    population[:, COLUMNS].insert(cls)
    if batch_size is not None:
        # This is deliberately done after insertion and before build: pending
        # structural selectors must be promoted correctly by Population.batch.
        population.batch(batch_size)
    population.build()
    return population, _mechanism(population, cls)


@pytest.mark.parametrize(
    ("batch_size", "expected_parameter_shape", "expected_full_shape"),
    [
        (None, (N_POPULATIONS, N_LOCAL_COLUMNS), (N_POPULATIONS, N_LOCAL_COLUMNS)),
        (
            EXPLICIT_BATCH,
            (1, N_POPULATIONS, N_LOCAL_COLUMNS),
            (EXPLICIT_BATCH, N_POPULATIONS, N_LOCAL_COLUMNS),
        ),
    ],
    ids=("unbatched", "prebuild_explicit_batch"),
)
def test_opt_in_shared_columns_preserve_all_mechanism_storage_axes(
    batch_size, expected_parameter_shape, expected_full_shape
):
    population, mechanism = _shared_column_population(batch_size=batch_size)
    state = mechanism.DE[_AxisState.__name__]

    assert mechanism.support_spec.kind is SupportKind.SHARED_COLUMNS
    assert mechanism.support_spec.all_populations
    assert mechanism.support_spec.preserves_population_axis
    assert mechanism.support_spec.runtime_local_shape == (
        N_POPULATIONS,
        N_LOCAL_COLUMNS,
    )
    assert mechanism.shape_p == expected_parameter_shape
    assert mechanism.shape_f == expected_full_shape

    # RANGE parameters, state storage, SAVE mirrors, and mechanism BUFFERs all
    # use shape_p at construction.  Explicit runtime batches stay broadcastable
    # rather than being redundantly materialized in parameter storage.
    for value in (
        mechanism.g,
        mechanism.e,
        mechanism.x,
        mechanism.i_,
        mechanism.scratch,
        state.rate,
    ):
        assert tuple(value.shape) == expected_parameter_shape

    # BATCH means every structural axis except the final compartment axis.
    # Preserving N therefore changes (1,) into (N, 1), with explicit batch
    # dimensions retained as singleton parameter axes.
    expected_batch_shape = expected_parameter_shape[:-1] + (1,)
    assert tuple(mechanism.scale.shape) == expected_batch_shape
    assert tuple(state.state_scale.shape) == expected_batch_shape

    # Population-owned geometry remains full-field; only the local view changes.
    assert tuple(population.v.shape) == (
        (N_POPULATIONS, N_COMPARTMENTS)
        if batch_size is None
        else (EXPLICIT_BATCH, N_POPULATIONS, N_COMPARTMENTS)
    )

    # Initialization materializes dynamic state over explicit runtime batches;
    # parameter buffers keep their singleton broadcast prefix.  A real step
    # then exercises both SAVE and user BUFFER rebinding at shape_f.
    population.initialize()
    assert tuple(mechanism.x.shape) == expected_full_shape
    assert tuple(mechanism.scratch.shape) == expected_full_shape
    population.step(dt=0.025)
    assert tuple(mechanism.x.shape) == expected_full_shape
    assert tuple(mechanism.i_.shape) == expected_full_shape
    assert tuple(mechanism.scratch.shape) == expected_full_shape
    assert tuple(mechanism.g.shape) == expected_parameter_shape
    assert tuple(state.rate.shape) == expected_parameter_shape


@pytest.mark.parametrize(
    ("batch_size", "expected_shape"),
    [
        (None, (N_POPULATIONS, 1)),
        (EXPLICIT_BATCH, (1, N_POPULATIONS, 1)),
    ],
    ids=("unbatched", "prebuild_explicit_batch"),
)
def test_batch_parameter_keeps_population_values_and_shape(batch_size, expected_shape):
    _, mechanism = _shared_column_population(
        batch_size=batch_size,
        cls=_PopulationBatchProbe,
    )

    assert mechanism.support_spec.preserves_population_axis
    assert tuple(mechanism.scale.shape) == expected_shape
    expected = torch.tensor([[0.5], [1.0], [1.5]], dtype=DTYPE)
    if batch_size is not None:
        expected = expected.unsqueeze(0)
    torch.testing.assert_close(mechanism.scale, expected)


def test_slice_batch_parametrization_targets_population_rows_once_and_rebuilds():
    population, mechanism = _shared_column_population()
    selected_rows = torch.tensor([0, 2], dtype=torch.long)

    population[selected_rows, 1].mech._AxisDensity.parametrize(
        "scale", 7.0, alias="selected_rows"
    )
    mechanism.populate_parameter_buffers()
    torch.testing.assert_close(
        mechanism.scale,
        torch.tensor([[7.0], [1.0], [7.0]], dtype=DTYPE),
    )
    assert torch.equal(mechanism.keys["scale"], selected_rows)

    population.build(force_rebuild=True)
    mechanism = _mechanism(population, _AxisDensity)
    mechanism.populate_parameter_buffers()
    torch.testing.assert_close(
        mechanism.scale,
        torch.tensor([[7.0], [1.0], [7.0]], dtype=DTYPE),
    )


def test_delete_rejects_population_batch_scope_collapse_atomically():
    population, mechanism = _shared_column_population()
    selected_rows = torch.tensor([0, 2], dtype=torch.long)
    population[selected_rows, 1].mech._AxisDensity.parametrize(
        "scale", 7.0, alias="selected_rows"
    )
    mechanism.populate_parameter_buffers()
    expected_key = mechanism.key.clone()
    expected_scale = mechanism.scale.clone()
    expected_records = len(population._slice_mechanism_parametrizations)

    with pytest.raises(ValueError, match="grouped.*packed.*BATCH"):
        population[0, 1].delete(_AxisDensity)

    assert not population._flag_rebuild
    assert population.mech._AxisDensity is mechanism
    assert torch.equal(mechanism.key, expected_key)
    assert len(population._slice_mechanism_parametrizations) == expected_records
    mechanism.populate_parameter_buffers()
    torch.testing.assert_close(mechanism.scale, expected_scale)


def test_delete_whole_column_keeps_population_batch_groups():
    population, mechanism = _shared_column_population()
    selected_rows = torch.tensor([0, 2], dtype=torch.long)
    population[selected_rows, 4].mech._AxisDensity.parametrize(
        "scale", 7.0, alias="surviving_rows"
    )
    mechanism.populate_parameter_buffers()

    population[:, 1].delete(_AxisDensity)
    population.build()
    mechanism = _mechanism(population, _AxisDensity)
    mechanism.populate_parameter_buffers()

    assert mechanism.support_spec.runtime_local_shape == (N_POPULATIONS, 1)
    assert mechanism.shape_p == (N_POPULATIONS, 1)
    assert mechanism.scale.shape == (N_POPULATIONS, 1)
    torch.testing.assert_close(
        mechanism.scale,
        torch.tensor([[7.0], [1.0], [7.0]], dtype=DTYPE),
    )
    assert torch.equal(
        mechanism.get(population.v),
        population.v.index_select(-1, torch.tensor([4])),
    )


def test_indexed_delete_of_entire_structured_support_succeeds():
    population, _ = _shared_column_population()

    population[:, COLUMNS].delete(_AxisDensity)
    population.build()

    assert _AxisDensity not in population._mech_data
    assert _AxisDensity.__name__ not in population.mech.mechanisms


@pytest.mark.parametrize(
    "mechanism",
    (
        _MechanismBatchOnly,
        _NestedBatchOnly,
        _BatchRandOnly,
        _BatchNoiseOnly,
    ),
    ids=("mechanism_batch", "state_batch", "batch_rand", "batch_noise"),
)
@pytest.mark.parametrize("build_first", (False, True), ids=("prebuild", "built"))
def test_batch_contract_alone_rejects_structured_to_packed_delete(
    mechanism, build_first
):
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=True,
    )
    population[:, COLUMNS].insert(mechanism)
    if build_first:
        population.build()

    with pytest.raises(ValueError, match="grouped.*packed.*BATCH"):
        population[0, 1].delete(mechanism)

    population.build()
    compiled = _mechanism(population, mechanism)
    assert compiled.support_spec.preserves_population_axis
    assert torch.equal(compiled.key, torch.tensor([1, 4, 7, 10, 13, 16]))


def test_declaration_free_mechanism_may_transition_to_exact_packed_support():
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=True,
    )
    values = torch.arange(N_POPULATIONS * N_LOCAL_COLUMNS, dtype=DTYPE)
    population[:, COLUMNS].insert(_NoBatchDensity, value=values)
    population.build()
    assert _mechanism(
        population, _NoBatchDensity
    ).support_spec.preserves_population_axis

    population[0, 1].delete(_NoBatchDensity)
    population.build()
    mechanism = _mechanism(population, _NoBatchDensity)

    assert mechanism.support_spec.kind is SupportKind.PACKED_FLAT
    assert not mechanism.support_spec.preserves_population_axis
    assert torch.equal(mechanism.key, torch.tensor([4, 7, 10, 13, 16]))
    torch.testing.assert_close(mechanism.value, values[1:])


def test_delete_all_rolls_back_earlier_class_when_batch_gate_rejects_later_class():
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=True,
    )
    population[:, COLUMNS].insert(_NoBatchDensity)
    population[:, COLUMNS].insert(_MechanismBatchOnly)
    population.build()
    expected_no_batch = population.mech._NoBatchDensity.key.clone()
    expected_batch = population.mech._MechanismBatchOnly.key.clone()

    with pytest.raises(ValueError, match="grouped.*packed.*BATCH"):
        population.delete_all(index=(0, 1))

    assert not population._flag_rebuild
    population.build()
    assert torch.equal(population.mech._NoBatchDensity.key, expected_no_batch)
    assert torch.equal(population.mech._MechanismBatchOnly.key, expected_batch)


@pytest.mark.parametrize("batch_size", [None, EXPLICIT_BATCH])
def test_population_shaped_support_gather_add_and_put_are_exact(batch_size):
    population, mechanism = _shared_column_population(batch_size=batch_size)
    field = torch.arange(
        population.v.numel(), dtype=DTYPE, device=population.v.device
    ).reshape_as(population.v)
    expected_local = field.index_select(-1, COLUMNS)

    actual_local = mechanism.get(field)
    assert tuple(actual_local.shape) == tuple(mechanism.shape_f)
    assert torch.equal(actual_local, expected_local)

    local_update = torch.arange(
        actual_local.numel(), dtype=DTYPE, device=population.v.device
    ).reshape_as(actual_local)
    expected_add = torch.zeros_like(field)
    expected_add[..., COLUMNS] = local_update

    destination = torch.zeros_like(field)
    returned = mechanism.add_(destination, local_update)
    assert returned is destination
    assert torch.equal(destination, expected_add)
    assert torch.equal(
        mechanism.add(torch.zeros_like(field), local_update), expected_add
    )

    original = torch.full_like(field, -7.0)
    expected_put = original.clone()
    expected_put[..., COLUMNS] = local_update + 100.0
    actual_put = mechanism.put(local_update + 100.0, original, field)
    assert torch.equal(actual_put, expected_put)
    # The default clone contract must leave caller-owned destination storage intact.
    assert torch.equal(original, torch.full_like(field, -7.0))


def _rollout(*, preserve, jit):
    with dn.ctx(
        DTYPE=DTYPE,
        JIT=int(jit),
        BACKEND="eager",
        FULLGRAPH=0,
    ):
        population, mechanism = _shared_column_population(preserve=preserve)
        population.train(True)
        population.initialize()

        # Use the live RANGE buffer as the differentiated input.  This mirrors
        # optimizer/in-graph parametrization execution while keeping unrelated
        # declared defaults detached from the compiled graph.
        conductance = mechanism.g.detach().clone().requires_grad_()
        mechanism._buffers["g"] = conductance
        # Present the compiled kernel with its eventual autograd contract on
        # the first call.  Otherwise Dynamo quite reasonably specializes once
        # for a detached initial voltage and again for the differentiable
        # recurrent voltage produced by that call.
        population.v = population.v.detach().clone().requires_grad_()

        trajectory = [population.v.clone()]
        state_trajectory = [mechanism.x.reshape(N_POPULATIONS, -1).clone()]
        for _ in range(4):
            population.step(dt=0.025)
            trajectory.append(population.v.clone())
            state_trajectory.append(mechanism.x.reshape(N_POPULATIONS, -1).clone())

        trajectory = torch.stack(trajectory)
        state_trajectory = torch.stack(state_trajectory)
        loss = trajectory.square().mean() + 0.1 * state_trajectory.square().mean()
        (gradient,) = torch.autograd.grad(loss, conductance)
        return (
            trajectory.detach(),
            state_trajectory.detach(),
            gradient.reshape(N_POPULATIONS, -1).detach(),
            mechanism.i_.reshape(N_POPULATIONS, -1).detach().clone(),
            mechanism.scratch.reshape(N_POPULATIONS, -1).detach().clone(),
        )


@pytest.mark.filterwarnings(
    "ignore:The \\.grad attribute of a Tensor that is not a leaf Tensor.*:UserWarning"
)
@pytest.mark.parametrize("jit", [False, True], ids=("eager", "jit_eager_backend"))
def test_opt_in_trajectory_and_gradient_match_legacy_flat_storage(jit):
    legacy = _rollout(preserve=False, jit=jit)
    structured = _rollout(preserve=True, jit=jit)

    for actual, expected in zip(structured, legacy):
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


@dataclass(frozen=True)
class _FallbackCase:
    name: str
    mechanism: type[Mechanism]
    build: object
    expected_kind: SupportKind
    expected_slots: int


def _insert_shared(population, mechanism):
    population[:, COLUMNS].insert(mechanism)


def _insert_duplicates(population, mechanism):
    population[:, COLUMNS].insert(mechanism, copies=2)


def _insert_ragged(population, mechanism):
    rows = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
    columns = torch.tensor([0, 3, 2, 1, 4, 5], dtype=torch.long)
    population[rows, columns].insert(mechanism)


def _insert_forced_flat(population, mechanism):
    population.insert(mechanism)
    population[:, 1].delete(mechanism)


FALLBACK_CASES = (
    _FallbackCase(
        "point_process",
        _AxisPoint,
        _insert_shared,
        SupportKind.SHARED_COLUMNS,
        N_POPULATIONS * N_LOCAL_COLUMNS,
    ),
    _FallbackCase(
        "material_process",
        _AxisMaterialProcess,
        _insert_shared,
        SupportKind.SHARED_COLUMNS,
        N_POPULATIONS * N_LOCAL_COLUMNS,
    ),
    _FallbackCase(
        "voltage_process",
        _AxisVoltageProcess,
        _insert_shared,
        SupportKind.SHARED_COLUMNS,
        N_POPULATIONS * N_LOCAL_COLUMNS,
    ),
    _FallbackCase(
        "duplicates",
        _AxisDensity,
        _insert_duplicates,
        SupportKind.PACKED_FLAT,
        2 * N_POPULATIONS * N_LOCAL_COLUMNS,
    ),
    _FallbackCase(
        "ragged",
        _AxisDensity,
        _insert_ragged,
        SupportKind.PACKED_FLAT,
        6,
    ),
    _FallbackCase(
        "forced_flat",
        _AxisDensity,
        _insert_forced_flat,
        SupportKind.PACKED_FLAT,
        N_POPULATIONS * (N_COMPARTMENTS - 1),
    ),
)


@pytest.mark.parametrize("case", FALLBACK_CASES, ids=lambda case: case.name)
def test_ineligible_mechanisms_and_supports_strictly_keep_legacy_flat_storage(case):
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=True,
    )
    case.build(population, case.mechanism)
    population.build()
    mechanism = _mechanism(population, case.mechanism)

    assert mechanism.support_spec.kind is case.expected_kind
    assert not mechanism.support_spec.preserves_population_axis
    assert mechanism.support_spec.runtime_local_shape == (case.expected_slots,)
    assert mechanism.shape_p == (case.expected_slots,)
    assert mechanism.shape_f == (case.expected_slots,)

    if case.name == "duplicates":
        assert mechanism.support_spec.preserves_multiplicity
        assert mechanism.support_spec.has_duplicates
    if case.name == "forced_flat":
        assert mechanism.support_spec.force_packed


def test_single_population_and_partial_population_rows_keep_legacy_storage():
    single = dn.Population(
        N=1,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=True,
    )
    single[:, COLUMNS].insert(_AxisDensity)
    single.build()
    single_mechanism = _mechanism(single, _AxisDensity)
    assert single_mechanism.support_spec.kind is SupportKind.SHARED_COLUMNS
    assert single_mechanism.support_spec.all_populations
    assert not single_mechanism.support_spec.preserves_population_axis
    assert single_mechanism.shape_p == (N_LOCAL_COLUMNS,)

    partial = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=True,
    )
    rows = torch.tensor([0, 0, 2, 2], dtype=torch.long)
    columns = COLUMNS.repeat(2)
    partial[rows, columns].insert(_AxisDensity)
    partial.build()
    partial_mechanism = _mechanism(partial, _AxisDensity)
    assert partial_mechanism.support_spec.kind is SupportKind.SHARED_COLUMNS
    assert not partial_mechanism.support_spec.all_populations
    assert not partial_mechanism.support_spec.preserves_population_axis
    assert partial_mechanism.shape_p == (4,)


@pytest.mark.parametrize(
    "mechanism_type",
    (_AxisSensitiveDensity, _FlatDefaultDensity, _FlatInitDensity),
    ids=("class_opt_out", "flat_only_default", "flat_only_initial_state"),
)
def test_axis_sensitive_or_flat_only_mechanisms_fall_back(mechanism_type):
    population = dn.Population(
        N=N_POPULATIONS,
        C=N_COMPARTMENTS,
        dtype=DTYPE,
        preserve_mechanism_population_axis=True,
    )
    population[:, COLUMNS].insert(mechanism_type)
    population.build()
    mechanism = _mechanism(population, mechanism_type)

    assert mechanism.support_spec.kind is SupportKind.SHARED_COLUMNS
    assert not mechanism.support_spec.preserves_population_axis
    assert mechanism.shape_p == (N_POPULATIONS * N_LOCAL_COLUMNS,)


def test_multistream_delay_requires_an_explicit_axis_contract():
    _, mechanism = _shared_column_population()

    with pytest.raises(RuntimeError, match="explicit population-axis contract"):
        mechanism.register_delayed_states(
            "streams",
            torch.zeros(
                N_POPULATIONS,
                N_LOCAL_COLUMNS,
                dtype=DTYPE,
            ),
            torch.zeros(N_POPULATIONS, dtype=torch.long),
            stream_axis=0,
        )


def test_population_axis_policy_resolves_at_construction_with_clear_precedence():
    class DefaultOnPopulation(dn.Population):
        preserve_mechanism_population_axis_default = True

    assert not dn.Population(N=2, C=2).preserve_mechanism_population_axis
    assert DefaultOnPopulation(N=2, C=2).preserve_mechanism_population_axis

    with dn.ctx(PRESERVE_MECHANISM_POPULATION_AXIS=True):
        assert dn.Population(N=2, C=2).preserve_mechanism_population_axis
        assert not dn.Population(
            N=2,
            C=2,
            preserve_mechanism_population_axis=False,
        ).preserve_mechanism_population_axis

    with dn.ctx(PRESERVE_MECHANISM_POPULATION_AXIS=False):
        assert not DefaultOnPopulation(
            N=2,
            C=2,
        ).preserve_mechanism_population_axis
        assert DefaultOnPopulation(
            N=2,
            C=2,
            preserve_mechanism_population_axis=True,
        ).preserve_mechanism_population_axis


def test_population_axis_policy_is_snapshotted_before_build():
    with dn.ctx(PRESERVE_MECHANISM_POPULATION_AXIS=True):
        enabled = dn.Population(N=N_POPULATIONS, C=N_COMPARTMENTS)
    enabled[:, COLUMNS].insert(_AxisDensity)
    with dn.ctx(PRESERVE_MECHANISM_POPULATION_AXIS=False):
        enabled.build()
    assert enabled.preserve_mechanism_population_axis
    assert _mechanism(
        enabled,
        _AxisDensity,
    ).support_spec.preserves_population_axis

    disabled = dn.Population(N=N_POPULATIONS, C=N_COMPARTMENTS)
    disabled[:, COLUMNS].insert(_AxisDensity)
    with dn.ctx(PRESERVE_MECHANISM_POPULATION_AXIS=True):
        disabled.build()
    assert not disabled.preserve_mechanism_population_axis
    assert not _mechanism(
        disabled,
        _AxisDensity,
    ).support_spec.preserves_population_axis


def test_population_axis_policy_snapshot_is_read_only():
    population = dn.Population(N=2, C=2)

    with pytest.raises(AttributeError):
        population.preserve_mechanism_population_axis = True


def test_population_axis_explicit_flag_requires_a_boolean_or_none():
    with pytest.raises(TypeError, match="must be a boolean or None"):
        dn.Population(N=2, C=2, preserve_mechanism_population_axis=1)


def test_population_axis_model_family_default_requires_a_boolean():
    class InvalidDefaultPopulation(dn.Population):
        preserve_mechanism_population_axis_default = "yes"

    with pytest.raises(TypeError, match="default must be a boolean"):
        InvalidDefaultPopulation(N=2, C=2)
