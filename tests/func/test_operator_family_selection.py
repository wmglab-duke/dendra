"""Operator selection preserves family contracts and qualification phases."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import _population as population_api
from dendra.models.integrators.explicit import (
    _dufort_frankel,
    _dufort_frankel_homogeneous,
    _euler,
    _rk2,
    _rk4,
)
from dendra.models.integrators.implicit import (
    _bwd_euler_bt,
    _bwd_euler_sc,
    _bwd_euler_sc_multi,
    _bwd_euler_sc_skip,
    _bwd_euler_ub,
)
from dendra.models.integrators.tree import _dhs, _dhs_multi
from dendra.models.integrators.tree_bt import _dhs_bt, _dhs_bt_multi
from dendra.models.mod import pas
from dendra.models.multi import MultiPopulation
from tests.func.test_extcell_multi_population import _model as _block_multi
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _module_snapshot,
)
from tests.func.test_multi_population import _mixed_multi
from tests.func.test_population_subclasses import _model as _subclass_model

pytestmark = pytest.mark.cpu
DT = 0.01


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                IMEM=False,
                DTYPE=torch.float64,
                DEVICE="cpu",
                CHECK_PURITY=False,
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _model(family, method=None):
    if method is None:
        if family == "multi_scalar":
            return _mixed_multi()
        if family == "multi_block":
            return _block_multi()
        return _subclass_model(family)
    integrator = getattr(dn, method)(imem=False)
    if family == "single_compartment":
        model = dn.SingleCompartment(N=1, C=3, integrator=integrator)
    elif family == "unmyelinated":
        model = dn.Unmyelinated([2.0], L=3.0, dx=1.0, integrator=integrator)
    else:
        model = dn.Myelinated([8.0], n_node=3, integrator=integrator)
    model.insert(pas)
    model.initialize()
    return model


@pytest.mark.parametrize(
    "family,method,base,topology,operator,implementation",
    (
        (
            "population",
            None,
            dn.Population,
            "single_compartment",
            "scalar_point",
            _bwd_euler_sc,
        ),
        (
            "single_compartment",
            None,
            dn.SingleCompartment,
            "single_compartment",
            "scalar_point",
            _bwd_euler_sc,
        ),
        ("cable", None, dn.Cable, "native_cable", "scalar_path", _bwd_euler_ub),
        ("tree", None, dn.Tree, "scalar_tree", "scalar_tree", _dhs),
        (
            "unmyelinated",
            None,
            dn.Unmyelinated,
            "unmyelinated",
            "scalar_path",
            _bwd_euler_ub,
        ),
        ("myelinated", None, dn.Myelinated, "myelinated", "scalar_path", _bwd_euler_ub),
        (
            "extcell_axon",
            None,
            dn.ExtCellAxon,
            "extcell_axon",
            "block_path",
            _bwd_euler_bt,
        ),
        ("extcell_tree", None, dn.ExtCellTree, "extcell_tree", "block_tree", _dhs_bt),
        (
            "multi_population",
            None,
            MultiPopulation,
            "multi_point",
            "scalar_multi_point",
            _bwd_euler_sc_multi,
        ),
        (
            "multi_scalar",
            None,
            MultiPopulation,
            "multi_scalar",
            "scalar_multi",
            _dhs_multi,
        ),
        (
            "multi_block",
            None,
            MultiPopulation,
            "multi_block",
            "block_multi_tree",
            _dhs_bt_multi,
        ),
        (
            "single_compartment",
            "euler",
            dn.SingleCompartment,
            "single_compartment",
            "scalar_explicit_path",
            _euler,
        ),
        (
            "single_compartment",
            "rk2",
            dn.SingleCompartment,
            "single_compartment",
            "scalar_explicit_path",
            _rk2,
        ),
        (
            "single_compartment",
            "rk4",
            dn.SingleCompartment,
            "single_compartment",
            "scalar_explicit_path",
            _rk4,
        ),
        (
            "single_compartment",
            "df",
            dn.SingleCompartment,
            "single_compartment",
            "scalar_dufort_path",
            _dufort_frankel,
        ),
        (
            "single_compartment",
            "dfh",
            dn.SingleCompartment,
            "single_compartment",
            "scalar_dufort_path",
            _dufort_frankel_homogeneous,
        ),
        (
            "single_compartment",
            "scnv",
            dn.SingleCompartment,
            "single_compartment",
            "scalar_point_skip",
            _bwd_euler_sc_skip,
        ),
        (
            "unmyelinated",
            "rk4",
            dn.Unmyelinated,
            "unmyelinated",
            "scalar_explicit_path",
            _rk4,
        ),
        (
            "myelinated",
            "df",
            dn.Myelinated,
            "myelinated",
            "scalar_dufort_path",
            _dufort_frankel,
        ),
    ),
)
def test_initialization_and_lowering_keep_native_family_and_workspace_contracts(
    family, method, base, topology, operator, implementation
):
    source = _model(family, method)
    before = _module_snapshot(source)
    assert population_api._functional_initialization_profile(source) == (
        base,
        operator,
        implementation,
    )
    assert population_api._functional_topology_kind(source) == topology
    native = population_api._resolve_functional_integrator_spec(
        source.integrator, operator
    )
    assert native is not None and native.implementation is implementation
    functional, _ = dn.func.make_functional(source, dt=DT)
    assert functional._topology_kind == topology
    assert functional._integrator_workspace_schema == native.workspace_schema
    _assert_module_snapshot(source, before)


@pytest.mark.parametrize("operator", (None, "unsupported", []))
def test_unknown_packed_operator_retains_absence_and_hash_error_behavior(
    operator, monkeypatch
):
    source = _model("multi_population")
    monkeypatch.setattr(source.integrator, "_FUNCTIONAL_OPERATOR_KIND", operator)
    if isinstance(operator, list):
        for function in (
            population_api._functional_initialization_profile,
            population_api._functional_topology_kind,
        ):
            with pytest.raises(TypeError, match="unhashable type: 'list'"):
                function(source)
    else:
        assert population_api._functional_initialization_profile(source) is None
        assert population_api._functional_topology_kind(source) is None


def test_packed_operator_string_subclass_retains_authored_marker_identity(monkeypatch):
    class OperatorMarker(str):
        pass

    source = _model("multi_population")
    marker = OperatorMarker("scalar_multi_point")
    monkeypatch.setattr(source.integrator, "_FUNCTIONAL_OPERATOR_KIND", marker)
    profile = population_api._functional_initialization_profile(source)
    assert profile is not None
    assert profile[0] is MultiPopulation
    assert profile[1] is marker
    assert profile[2] is _bwd_euler_sc_multi
    assert population_api._functional_topology_kind(source) == "multi_point"


@pytest.mark.parametrize(
    "family,operator,path_probe",
    (
        ("unmyelinated", "scalar_path", True),
        ("tree", "scalar_tree", False),
        ("multi_population", "scalar_multi_point", False),
    ),
)
@pytest.mark.parametrize("failure_phase", ("qualification", "solver"))
def test_clone_selection_retains_probe_qualification_and_failure_order(
    family, operator, path_probe, failure_phase, monkeypatch
):
    source = _model(family)
    before = _module_snapshot(source)
    original_clone = population_api._clone_execution_population
    original_strip = population_api._strip_packed_component_runtime_aliases
    original_point = population_api._scalar_point_integrator_profile
    original_path = population_api._scalar_path_integrator_profile
    original_resolve = population_api._resolve_functional_integrator_spec
    integrator = None
    stripped = False
    events = []
    failure = RuntimeError("deliberate cloned solver failure")

    def clone(population):
        nonlocal integrator
        result = original_clone(population)
        integrator = result.integrator
        return result

    def strip(population):
        nonlocal stripped
        original_strip(population)
        stripped = True
        events.append("strip")

    def point(candidate):
        if candidate is integrator:
            events.append("point")
        return original_point(candidate)

    def path(candidate):
        if candidate is integrator:
            events.append("path")
        return original_path(candidate)

    def solver():
        events.append("solver")
        raise failure

    def resolve(candidate, kind):
        if candidate is not integrator:
            return original_resolve(candidate, kind)
        events.append(f"resolve:{kind}")
        result = original_resolve(candidate, kind)
        if kind == operator:
            if failure_phase == "qualification":
                return None
            assert result is not None
            monkeypatch.setattr(candidate, "_functional_solver", solver)
        return result

    for name in (
        "_is_multi",
        "_is_scalar_point",
        "_is_native_cable",
        "_is_scalar_tree",
        "_is_extcell_tree",
        "_functional_topology_kind",
    ):
        original = getattr(population_api, name)

        def admit(population, original=original):
            assert not stripped, "operator selection reclassified stripped geometry"
            return original(population)

        monkeypatch.setattr(population_api, name, admit)
    monkeypatch.setattr(population_api, "_clone_execution_population", clone)
    monkeypatch.setattr(
        population_api, "_strip_packed_component_runtime_aliases", strip
    )
    monkeypatch.setattr(population_api, "_scalar_point_integrator_profile", point)
    monkeypatch.setattr(population_api, "_scalar_path_integrator_profile", path)
    monkeypatch.setattr(population_api, "_resolve_functional_integrator_spec", resolve)
    with pytest.raises(dn.func.FunctionalizationError) as caught:
        dn.func.make_functional(source, dt=DT)
    expected = ["strip"] if family == "multi_population" else []
    expected.extend(
        ("point", "path", "resolve:scalar_dufort_path", "resolve:scalar_explicit_path")
    )
    if path_probe:
        expected.extend(
            ("path", "resolve:scalar_dufort_path", "resolve:scalar_explicit_path")
        )
    expected.append(f"resolve:{operator}")
    if failure_phase == "solver":
        expected.append("solver")
        assert str(caught.value) == str(failure)
        assert caught.value.__cause__ is failure
    else:
        assert (
            str(caught.value)
            == "The cloned integrator no longer satisfies its functional contract."
        )
    assert events == expected
    _assert_module_snapshot(source, before)
