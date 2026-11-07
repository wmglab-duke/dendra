"""Retained projection selects complete caller frames without copying tensors."""

from __future__ import annotations

from collections.abc import MutableMapping
from types import SimpleNamespace

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_functional_initialization import _assert_tree_close
from tests.func.test_initialization_execution import _assert_snapshot, _snapshot
from tests.func.test_network_initialization_ordering import DTYPE, _packed_network
from tests.func.test_noise_workspaces import DT as NOISE_DT
from tests.func.test_noise_workspaces import _population as _noise_population
from tests.func.test_retained_materialization_inventory import (
    _INTEGRATOR_FIELDS,
    _PACKED_PATHS,
    _PAS,
    _POPULATION_FIELDS,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(
            JIT=0,
            REQUIRE_GRAD=True,
            IMEM=False,
            DEVICE="cpu",
            DTYPE=DTYPE,
            CHECK_PURITY=False,
        ),
    ):
        yield


class _Observed(MutableMapping):
    def __init__(self, name, entries, observe):
        self.name, self.entries, self.observe = name, entries, observe

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        self.observe(f"{self.name}:get:{key}")
        return self.entries[key]

    def __setitem__(self, key, value):
        self.observe(f"{self.name}:set:{key}")
        if self.name == "values" and key == "integrator":
            value = _Observed("integrator", value, self.observe)
        self.entries[key] = value

    def __delitem__(self, key):
        del self.entries[key]

    def get(self, key, default=None):
        self.observe(f"{self.name}:get-default:{key}")
        return self.entries.get(key, default)


def _prepared_reference(path):
    if path in _POPULATION_FIELDS:
        return "population", path
    if path in (_PAS + "g", _PAS + "e"):
        return "mechanisms", path
    if path in _INTEGRATOR_FIELDS:
        return "integrator", path.removeprefix("integrator.")
    return None


def _case(*, extras_none=False, failure=None):
    functional, tensors = dn.func.make_functional_network(_packed_network())
    plan = functional.populations["packed"]
    constants = tensors.constants["populations"]["packed"]
    prepared = plan.prepare(tensors.parameters["populations"]["packed"], constants)
    raw = pytree.tree_map(lambda value: value, prepared.values)
    previous = raw["materializations"]
    integrator_names = tuple(raw["integrator"])
    reference = raw["population"]["cm"]
    shared = torch.arange(2 * reference.numel() + 1, dtype=reference.dtype)[1::2]
    shared = shared.reshape_as(reference).requires_grad_()
    probe = shared.detach().clone().requires_grad_()
    geometry_alias = shared.view_as(shared)
    raw["population"]["cm"] = shared
    raw["geometry"] = {
        **previous,
        **raw["geometry"],
        "diam": probe,
        "dx": geometry_alias,
    }
    entries = {**previous, "cm": probe, "diam": probe, "dx": None}
    del entries["_component_area"]
    events = []
    error = (
        KeyError("population" if failure == "values:get:population" else "cm")
        if failure in ("values:get:population", "population:get:cm")
        else RuntimeError(failure)
    )

    def observe(phase):
        events.append(phase)
        if phase == failure:
            raise error

    sections = {
        name: _Observed(name, values, observe)
        for name, values in raw.items()
        if name in ("population", "mechanisms", "geometry", "integrator")
    }
    raw.update(sections)
    if failure == "values:get:population":
        del raw["population"]
    elif failure == "population:get:cm":
        del sections["population"].entries["cm"]
    values = _Observed("values", raw, observe)
    original_integrator = raw["integrator"]
    reads = {}

    class Extras(_Observed):
        def __bool__(self):
            observe("extras:bool")
            return False

        def __contains__(self, path):
            observe(f"extras:contains:{path}")
            return path in self.entries

        def __getitem__(self, path):
            number = reads.get(path, 0) + 1
            reads[path] = number
            observe(f"extras:read{number}:{path}")
            if path == "diam":
                assert number <= 2
                return probe if number == 1 else shared
            return self.entries[path]

    extras = None if extras_none else Extras("extras", entries, observe)
    expected = [
        "values:get:integrator",
        *(f"integrator:get:{name}" for name in integrator_names),
        "values:set:integrator",
    ]
    for path in _PACKED_PATHS:
        prepared_path = _prepared_reference(path)
        if prepared_path is not None:
            section, key = prepared_path
            expected += [f"values:get:{section}", f"{section}:get:{key}"]
            continue
        if extras_none:
            expected += ["values:get:geometry", f"geometry:get:{path}"]
            continue
        expected.append(f"extras:contains:{path}")
        if path in entries:
            expected.append(f"extras:read1:{path}")
            if path != "dx":
                expected.append(f"extras:read2:{path}")
                continue
        expected += ["values:get:geometry", f"geometry:get:{path}"]
    expected.append("values:set:materializations")
    return SimpleNamespace(
        plan=plan,
        constants=constants,
        values=values,
        raw=raw,
        extras=extras,
        expected=expected,
        events=events,
        previous=previous,
        error=error,
        original_integrator=original_integrator,
        shared=shared,
        probe=probe,
        geometry_alias=geometry_alias,
        reads=reads,
    )


def test_projection_keeps_dynamic_double_read_falsey_mapping_and_tensor_routes():
    case = _case()
    supplied = _snapshot((case.constants, case.shared, case.probe))
    result = case.plan._finish_retained_preparation(
        case.values, case.constants, extras=case.extras
    )
    assert result is case.values
    assert case.events == case.expected
    projected = case.raw["materializations"]
    assert projected is not case.previous and tuple(projected) == _PACKED_PATHS
    assert projected["cm"] is projected["diam"] is case.shared
    assert projected["dx"] is case.geometry_alias
    assert projected["dx"] is not case.shared
    assert torch._C._is_alias_of(projected["dx"], case.shared)
    assert not case.shared.is_contiguous() and case.shared.storage_offset() > 0
    assert projected["diam"].stride() == case.shared.stride()
    assert case.reads["diam"] == 2 and case.reads["dx"] == 1
    gradients = torch.autograd.grad(
        sum(projected[path].sum() for path in ("cm", "diam", "dx")),
        (case.shared, case.probe),
        allow_unused=True,
    )
    torch.testing.assert_close(gradients[0], torch.full_like(case.shared, 3.0))
    assert gradients[1] is None
    _assert_snapshot((case.constants, case.shared, case.probe), supplied)


def test_none_extras_uses_each_geometry_fallback_after_prepared_paths():
    case = _case(extras_none=True)
    case.plan._finish_retained_preparation(case.values, case.constants)
    assert case.events == case.expected
    projected = case.raw["materializations"]
    assert projected["cm"] is case.shared
    assert projected["diam"] is case.probe
    assert projected["dx"] is case.geometry_alias
    for name in ("one", "two"):
        path = f"populations.{name}.cm"
        assert projected[path] is case.previous[path]


@pytest.mark.parametrize(
    "failure",
    (
        "values:get:population",
        "population:get:cm",
        "extras:contains:diam",
        "extras:read1:diam",
        "extras:read2:diam",
        "geometry:get:dx",
        "extras:contains:populations.one.celsius",
    ),
)
def test_projection_failure_preserves_exception_prefix_and_prior_frame(failure):
    case = _case(failure=failure)
    supplied = _snapshot((case.constants, case.shared, case.probe))
    with pytest.raises((KeyError, RuntimeError)) as caught:
        case.plan._finish_retained_preparation(
            case.values, case.constants, extras=case.extras
        )
    assert caught.value is case.error
    assert case.events == case.expected[: case.expected.index(failure) + 1]
    assert case.raw["materializations"] is case.previous
    assert case.raw["integrator"] is not case.original_integrator
    _assert_snapshot((case.constants, case.shared, case.probe), supplied)


def test_empty_retained_projection_preserves_earlier_noise_publication():
    plan, tensors = dn.func.make_functional(_noise_population(), dt=NOISE_DT)
    prepared = plan.prepare(tensors.parameters, tensors.constants)
    previous, sentinel = object(), object()
    events = []
    values = _Observed(
        "values",
        {"noise_workspaces": sentinel, "materializations": previous},
        events.append,
    )

    class UnusedExtras:
        def __bool__(self):
            pytest.fail("Empty retained projection inspected extras")

        def __contains__(self, key):
            pytest.fail("Empty retained projection selected an extra")

    rng = torch.random.get_rng_state().clone()
    result = plan._finish_retained_preparation(
        values, tensors.constants, extras=UnusedExtras()
    )
    assert result is values and events == ["values:set:noise_workspaces"]
    assert values.entries["materializations"] is previous
    assert values.entries["noise_workspaces"] is not sentinel
    _assert_tree_close(
        values.entries["noise_workspaces"], prepared.values["noise_workspaces"]
    )
    assert torch.equal(torch.random.get_rng_state(), rng)
