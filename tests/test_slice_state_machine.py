"""Stateful property tests for retained Slice selections and lifecycle changes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

import dendra as dn
from dendra.models.mod import pas

DTYPE = torch.float64
_KEY_KINDS = (
    "all",
    "new_axis",
    "empty_last",
    "first_integer",
    "last_integer",
    "last_slice",
    "last_integer_tensor",
    "last_boolean_mask",
    "scalar",
    "paired_advanced",
)


@dataclass
class _SelectionRecord:
    selection: Any
    coordinate_ids: torch.Tensor


def _selection_key(shape, kind: str, first: int, second: int):
    """Construct a valid PyTorch index for a runtime-selected shape."""
    shape = tuple(int(size) for size in shape)
    ndim = len(shape)

    if kind == "all":
        return Ellipsis
    if kind == "new_axis":
        return (None, Ellipsis)
    if kind == "empty_last":
        return (Ellipsis, slice(0, 0)) if ndim else Ellipsis

    if kind == "first_integer":
        if ndim and shape[0] > 0:
            return first % shape[0]
        return Ellipsis

    if kind == "last_integer":
        if ndim and shape[-1] > 0:
            return (Ellipsis, second % shape[-1])
        return Ellipsis

    if kind == "last_slice":
        if not ndim:
            return Ellipsis
        size = shape[-1]
        left = first % (size + 1)
        right = second % (size + 1)
        start, stop = sorted((left, right))
        return (Ellipsis, slice(start, stop, 1 + (first + second) % 2))

    if kind == "last_integer_tensor":
        if not ndim or shape[-1] == 0:
            return Ellipsis
        candidates = [first % shape[-1], second % shape[-1]]
        indices = list(dict.fromkeys(candidates))
        return (Ellipsis, torch.tensor(indices, dtype=torch.long))

    if kind == "last_boolean_mask":
        if not ndim:
            return Ellipsis
        positions = torch.arange(shape[-1])
        mask = torch.remainder(positions + first, 2) == 0
        return (Ellipsis, mask)

    if kind == "scalar":
        if ndim and all(size > 0 for size in shape):
            return tuple(
                (first + axis * second) % size for axis, size in enumerate(shape)
            )
        return Ellipsis

    if kind == "paired_advanced":
        if ndim >= 2 and shape[-2] > 0 and shape[-1] > 0:
            row = first % shape[-2]
            column = second % shape[-1]
            return (
                Ellipsis,
                torch.tensor([row], dtype=torch.long),
                torch.tensor([column], dtype=torch.long),
            )
        return Ellipsis

    raise AssertionError(f"Unknown key kind: {kind}")


class SliceLifecycleStateMachine(RuleBasedStateMachine):
    """Compare retained Dendra selections with an independent coordinate oracle."""

    @initialize(
        n_cells=st.integers(min_value=1, max_value=3),
        n_compartments=st.integers(min_value=1, max_value=5),
    )
    def create_population(self, n_cells: int, n_compartments: int):
        self.initial_compartment_values = torch.linspace(
            -70.0,
            -70.0 + n_compartments - 1,
            n_compartments,
            dtype=DTYPE,
        )
        self.population = dn.Population(
            N=n_cells,
            C=n_compartments,
            v_init=self.initial_compartment_values,
            dtype=DTYPE,
        )
        self.initial_pas_g = 0.125
        self.population.insert(pas, g=self.initial_pas_g, e=-65.0)
        self.oracle = torch.arange(n_cells * n_compartments, dtype=DTYPE).reshape(
            n_cells, n_compartments
        )
        self.population.v.copy_(self.oracle)

        full_ids = self._coordinate_grid()
        self.records = [_SelectionRecord(self.population[...], full_ids.clone())]
        self.mechanism_records = []
        self.pas_slice = None
        self.g_oracle = None
        self.label_bindings = []
        self.label_counter = 0
        self.batch_count = 0

    def _coordinate_grid(self):
        return torch.arange(self.oracle.numel(), dtype=torch.long).reshape(
            self.oracle.shape
        )

    def _record(self, slot: int) -> _SelectionRecord:
        return self.records[slot % len(self.records)]

    def _expected_selection(self, record: _SelectionRecord):
        flat = self.oracle.reshape(-1)
        return flat.index_select(0, record.coordinate_ids.reshape(-1)).reshape(
            record.coordinate_ids.shape
        )

    def _initialized_oracle(self):
        singleton_prefix = (1,) * (self.oracle.ndim - 1)
        return (
            self.initial_compartment_values.reshape(
                *singleton_prefix, self.initial_compartment_values.numel()
            )
            .expand(self.oracle.shape)
            .clone()
        )

    def _record_built_mechanism(self):
        if self.pas_slice is None:
            self.pas_slice = self.records[0].selection.mech.pas
            self.mechanism_records.append(
                _SelectionRecord(self.pas_slice, self._coordinate_grid().clone())
            )
        self.g_oracle = torch.full(self.oracle.shape, self.initial_pas_g, dtype=DTYPE)

    def _expected_mechanism(self, record: _SelectionRecord):
        flat = self.g_oracle.reshape(-1)
        return flat.index_select(0, record.coordinate_ids.reshape(-1)).reshape(
            record.coordinate_ids.shape
        )

    @precondition(lambda self: len(self.records) < 8)
    @rule(
        kind=st.sampled_from(_KEY_KINDS),
        first=st.integers(min_value=0, max_value=12),
        second=st.integers(min_value=0, max_value=12),
    )
    def retain_direct_selection(self, kind: str, first: int, second: int):
        key = _selection_key(self.oracle.shape, kind, first, second)
        selection = self.population[key]
        coordinate_ids = self._coordinate_grid()[key].clone()
        self.records.append(_SelectionRecord(selection, coordinate_ids))

    @precondition(lambda self: len(self.records) < 8)
    @rule(
        slot=st.integers(min_value=0, max_value=30),
        kind=st.sampled_from(_KEY_KINDS),
        first=st.integers(min_value=0, max_value=12),
        second=st.integers(min_value=0, max_value=12),
    )
    def retain_nested_selection(self, slot: int, kind: str, first: int, second: int):
        parent = self._record(slot)
        key = _selection_key(parent.coordinate_ids.shape, kind, first, second)
        selection = parent.selection[key]
        coordinate_ids = parent.coordinate_ids[key].clone()
        self.records.append(_SelectionRecord(selection, coordinate_ids))

    @precondition(lambda self: self.batch_count < 2)
    @rule(size=st.integers(min_value=1, max_value=3))
    def add_leading_batch_axis(self, size: int):
        old_numel = self.oracle.numel()
        was_built = self.population.is_built
        self.population.batch(size)
        self.oracle = self.oracle.unsqueeze(0).expand(size, *self.oracle.shape).clone()

        for record in self.records:
            replicas = [
                record.coordinate_ids + batch * old_numel for batch in range(size)
            ]
            record.coordinate_ids = torch.stack(replicas, dim=0)
        for record in self.mechanism_records:
            replicas = [
                record.coordinate_ids + batch * old_numel for batch in range(size)
            ]
            record.coordinate_ids = torch.stack(replicas, dim=0)
        if was_built:
            # Batching a built Population rebuilds its compiled mechanism tree
            # so support metadata acquires the new leading axis.  Mutable
            # Slice.set writes belong to that disposable instance; only
            # authored insertion values and persistent parametrizations replay.
            self._record_built_mechanism()
        elif self.g_oracle is not None:
            self.g_oracle = (
                self.g_oracle.unsqueeze(0).expand(size, *self.g_oracle.shape).clone()
            )
        self.batch_count += 1

    @rule(
        slot=st.integers(min_value=0, max_value=30),
        writer=st.sampled_from(("set", "attribute")),
        value_kind=st.sampled_from(("scalar", "exact", "trailing")),
        seed=st.integers(min_value=-20, max_value=20),
    )
    def write_compatible_values(
        self, slot: int, writer: str, value_kind: str, seed: int
    ):
        record = self._record(slot)
        shape = tuple(record.coordinate_ids.shape)

        if value_kind == "scalar" or not shape:
            value = torch.tensor(float(seed), dtype=DTYPE)
        elif value_kind == "trailing":
            value = torch.arange(shape[-1], dtype=DTYPE) + float(seed)
        else:
            value = torch.arange(math.prod(shape), dtype=DTYPE).reshape(shape)
            value = value + float(seed)

        if writer == "set":
            result = record.selection.set("v", value)
        else:
            record.selection.v = value
            result = record.selection
        assert result is record.selection

        assigned = torch.empty(shape, dtype=DTYPE)
        assigned[...] = value
        flat_ids = record.coordinate_ids.reshape(-1)
        if flat_ids.numel():
            self.oracle.reshape(-1)[flat_ids] = assigned.reshape(-1)

    @rule(slot=st.integers(min_value=0, max_value=30))
    def attach_collision_free_label(self, slot: int):
        record = self._record(slot)
        name = f"region_{self.label_counter}"
        self.label_counter += 1

        parent = record.selection.parent_slice
        owner = self.population if parent is None else parent
        assert record.selection.label(name) is record.selection
        self.label_bindings.append((owner, name, record.selection))

    @precondition(lambda self: self.pas_slice is not None)
    @precondition(lambda self: len(self.mechanism_records) < 5)
    @rule(slot=st.integers(min_value=0, max_value=30))
    def retain_mechanism_selection(self, slot: int):
        record = self._record(slot)
        mechanism_slice = record.selection.mech.pas
        self.mechanism_records.append(
            _SelectionRecord(mechanism_slice, record.coordinate_ids.clone())
        )

    @precondition(lambda self: self.pas_slice is not None)
    @rule(
        writer=st.sampled_from(("set", "attribute")),
        value=st.integers(min_value=1, max_value=20),
    )
    def write_full_mechanism_field(self, writer: str, value: int):
        scalar = torch.tensor(value / 1000.0, dtype=DTYPE)
        if writer == "set":
            result = self.pas_slice.set("g", scalar)
        else:
            self.pas_slice.g = scalar
            result = self.pas_slice
        assert result is self.pas_slice
        self.g_oracle.fill_(scalar.item())

    @rule(force_rebuild=st.booleans())
    def build_or_rebuild(self, force_rebuild: bool):
        before = self.oracle.clone()
        was_built = self.population.is_built
        self.population.build(force_rebuild=force_rebuild)
        assert torch.equal(self.population.v, before)
        if not was_built or force_rebuild:
            self._record_built_mechanism()

    @rule(force_rebuild=st.booleans())
    def initialize_or_reinitialize(self, force_rebuild: bool):
        self.population.initialize(force_rebuild=force_rebuild)
        self.oracle = self._initialized_oracle()
        self._record_built_mechanism()

    @rule(
        slot=st.integers(min_value=0, max_value=30),
        writer=st.sampled_from(("set", "attribute")),
    )
    def incompatible_write_is_atomic(self, slot: int, writer: str):
        record = self._record(slot)
        before = self.population.v.clone()
        bad_value = torch.zeros(*record.coordinate_ids.shape, 2, dtype=DTYPE)

        with pytest.raises(ValueError):
            if writer == "set":
                record.selection.set("v", bad_value)
            else:
                record.selection.v = bad_value

        assert torch.equal(self.population.v, before)

    @precondition(lambda self: self.pas_slice is not None)
    @rule(
        slot=st.integers(min_value=0, max_value=30),
        writer=st.sampled_from(("set", "attribute")),
    )
    def incompatible_mechanism_write_is_atomic(self, slot: int, writer: str):
        record = self.mechanism_records[slot % len(self.mechanism_records)]
        before = self.g_oracle.clone()
        bad_value = torch.zeros(*record.coordinate_ids.shape, 2, dtype=DTYPE)

        with pytest.raises(ValueError):
            if writer == "set":
                record.selection.set("g", bad_value)
            else:
                record.selection.g = bad_value

        assert torch.equal(record.selection.g, self._expected_mechanism(record))
        assert torch.equal(self.g_oracle, before)

    @rule(slot=st.integers(min_value=0, max_value=30))
    def invalid_nested_index_is_atomic(self, slot: int):
        record = self._record(slot)
        before = self.population.v.clone()
        size = int(record.coordinate_ids.shape[0]) if record.coordinate_ids.ndim else 0

        with pytest.raises((IndexError, RuntimeError)):
            _ = record.selection[size]

        assert torch.equal(self.population.v, before)

    @rule(slot=st.integers(min_value=0, max_value=30))
    def invalid_label_is_atomic(self, slot: int):
        record = self._record(slot)
        parent = record.selection.parent_slice
        owner = self.population if parent is None else parent
        registry = dict(owner._labels)
        before = self.population.v.clone()

        with pytest.raises(ValueError):
            record.selection.label("v")

        assert owner._labels == registry
        assert torch.equal(self.population.v, before)

    @invariant()
    def retained_selections_match_coordinate_oracle(self):
        assert tuple(self.population.v.shape) == tuple(self.oracle.shape)
        assert torch.equal(self.population.v, self.oracle)

        for record in self.records:
            expected = self._expected_selection(record)
            assert tuple(record.selection.shape) == tuple(expected.shape)
            assert record.selection.is_scalar is (expected.ndim == 0)
            assert record.selection.is_empty is (expected.numel() == 0)
            assert torch.equal(record.selection.v, expected)
            if expected.ndim:
                assert len(record.selection) == expected.shape[0]
            else:
                with pytest.raises(TypeError):
                    len(record.selection)

        for owner, name, selection in self.label_bindings:
            assert getattr(owner, name) is selection

        if self.pas_slice is not None:
            for record in self.mechanism_records:
                expected = self._expected_mechanism(record)
                assert tuple(record.selection.shape) == tuple(expected.shape)
                assert torch.equal(record.selection.g, expected)


TestSliceLifecycleStateMachine = SliceLifecycleStateMachine.TestCase
TestSliceLifecycleStateMachine.settings = settings(
    max_examples=30,
    stateful_step_count=18,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)


def test_postbuild_batch_rebinds_mechanism_slice_to_authored_configuration():
    population = dn.Population(N=1, C=1, dtype=DTYPE)
    population.insert(pas, g=0.125, e=-65.0)
    population.build()

    retained = population[...].mech.pas
    old_mechanism = retained.model
    retained.set("g", torch.tensor(0.001, dtype=DTYPE))
    assert torch.equal(retained.g, torch.tensor([[0.001]], dtype=DTYPE))

    population.batch(1)

    assert retained.model is population.mech.pas
    assert retained.model is not old_mechanism
    assert torch.equal(retained.g, torch.tensor([[[0.125]]], dtype=DTYPE))
