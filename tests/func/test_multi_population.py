"""Functional contracts for exact packed scalar ``MultiPopulation`` models.

The functional representation deliberately follows the packed runtime rather
than treating the child populations as independently advancing simulations.
State and call-local drives therefore retain the flat owner layout, while
component physical inputs remain explicit, namespaced leaves of preparation.
"""

from __future__ import annotations

import copy
import math
import random
from functools import partial

import networkx as nx
import numpy as np
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population import _unregistered_module_alias_paths
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import hh
from dendra.units import nA

DT = 0.01
DTYPE = torch.float64
STEPS = 3

POINT_CM = "populations.point.cm_param.rho"
CABLE_RHOA = "populations.cable.rhoa_param"
HH_GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE,
        reason="functional scalar MultiPopulation execution requires dendra-solvers",
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _ScaleByArgument(torch.nn.Module):
    def forward(self, value, factor):
        return value * factor


class _CrossCategoryScale(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        shared = torch.nn.Parameter(torch.as_tensor(factor, dtype=DTYPE))
        self.register_parameter("factor_parameter", shared)
        self.register_buffer("factor_buffer", shared)

    def forward(self, value):
        return value * self.factor_buffer


def _tree_graph(*, edge_resistance_scale=1.0):
    """Return one tiny exact branch with nonuniform physical geometry."""
    graph = nx.DiGraph()
    for node, (area, cm, diam) in enumerate(
        (
            (180.0, 0.85, 3.0),
            (95.0, 1.15, 1.6),
            (125.0, 0.95, 1.2),
        )
    ):
        graph.add_node(
            node,
            name=f"branch.{node}",
            L=9.0 + 2.0 * node,
            diam=diam,
            Ra=83.0 + 17.0 * node,
            cm=cm,
            area=area,
        )
    graph.add_edge(0, 1, R_ohm=edge_resistance_scale * 1.1e8)
    graph.add_edge(0, 2, R_ohm=edge_resistance_scale * 1.7e8)
    return graph


def _point_multi(
    *,
    batch_calls=(2, 3),
    write_back=True,
    component_order=("left", "right"),
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        left = dn.SingleCompartment(
            N=1,
            C=2,
            cm=0.75,
            celsius=6.3,
            v_init=torch.tensor([-66.0, -61.0], dtype=DTYPE),
            dtype=DTYPE,
        )
        right = dn.SingleCompartment(
            N=2,
            C=1,
            cm=1.35,
            celsius=6.3,
            v_init=torch.tensor([-58.0], dtype=DTYPE),
            dtype=DTYPE,
        )
        left.insert(hh)
        right.insert(hh)
        components = {"left": left, "right": right}
        model = dn.concat_models(
            {name: components[name] for name in component_order},
            celsius=6.3,
            write_back=write_back,
        )
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _point_multi_with_dx_alias(*, target_right_dx=None):
    """Build a two-point model with source-shared or target-independent dx."""
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        left = dn.SingleCompartment(
            N=1,
            C=1,
            celsius=6.3,
            dtype=DTYPE,
        )
        right = dn.SingleCompartment(
            N=1,
            C=1,
            celsius=6.3,
            dtype=DTYPE,
        )
        left.insert(hh)
        right.insert(hh)
        left_dx = left.dx.new_full(left.dx.shape, 1.0)
        left._buffers["dx"] = left_dx
        right._buffers["dx"] = (
            left_dx
            if target_right_dx is None
            else right.dx.new_full(right.dx.shape, target_right_dx)
        )
        model = dn.concat_models(
            {"left": left, "right": right},
            celsius=6.3,
        )
        model.initialize()
        model.train()
    return model


def _buffer_argument_multi(factor):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        point = dn.SingleCompartment(
            N=1,
            C=1,
            celsius=6.3,
            dtype=DTYPE,
        )
        cable = dn.Unmyelinated(
            [1.5],
            L=3.0,
            dx=1.0,
            celsius=6.3,
            dtype=DTYPE,
        )
        for component in (point, cable):
            component.insert(hh)
        cable.register_buffer(
            "hidden_factor",
            torch.tensor(float(factor), dtype=DTYPE),
        )
        cable.register_parametrization_in_graph(
            "cm",
            _ScaleByArgument(),
            args=("hidden_factor",),
        )
        model = dn.concat_models(
            {"point": point, "cable": cable},
            celsius=6.3,
        )
        model.initialize()
        model.train()
    return model


def _parameter_argument_multi():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        point = dn.SingleCompartment(
            N=1,
            C=1,
            celsius=6.3,
            dtype=DTYPE,
        )
        cable = dn.Unmyelinated(
            [1.5],
            L=3.0,
            dx=1.0,
            celsius=6.3,
            dtype=DTYPE,
        )
        for component in (point, cable):
            component.insert(hh)
        cable.register_parameter(
            "spatial_factor",
            torch.nn.Parameter(
                torch.linspace(0.8, 1.2, cable.v.numel(), dtype=DTYPE).reshape(
                    cable.shape
                )
            ),
        )
        cable.register_parametrization_in_graph(
            "cm",
            _ScaleByArgument(),
            args=("spatial_factor",),
        )
        model = dn.concat_models(
            {"point": point, "cable": cable},
            celsius=6.3,
        )
        model.initialize()
        model.train()
    return model


def _cross_category_transform_multi():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        point = dn.SingleCompartment(N=1, C=1, celsius=6.3, dtype=DTYPE)
        cable = dn.Unmyelinated(
            [1.5],
            L=3.0,
            dx=1.0,
            celsius=6.3,
            dtype=DTYPE,
        )
        for component in (point, cable):
            component.insert(hh)
        cable.register_parametrization_in_graph(
            "cm",
            _CrossCategoryScale(2.0),
        )
        model = dn.concat_models(
            {"point": point, "cable": cable},
            celsius=6.3,
        )
        model.initialize()
        model.train()
    return model


def _mixed_multi(
    *,
    batch_calls=(),
    write_back=True,
    stimulation=False,
    tree_graph=None,
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        point = dn.SingleCompartment(
            N=1,
            C=1,
            cm=0.72,
            celsius=6.3,
            v_init=-66.0,
            dtype=DTYPE,
        )
        cable = dn.Unmyelinated(
            diameters=[1.7],
            L=3.0,
            dx=1.0,
            cm=1.18,
            rhoa=71.0,
            celsius=6.3,
            v_init=torch.tensor([-64.0, -60.0, -57.0], dtype=DTYPE),
            dtype=DTYPE,
        )
        tree = dn.Tree.from_graph(
            _tree_graph() if tree_graph is None else tree_graph,
            N=1,
            celsius=6.3,
            v_init=torch.tensor([-63.0, -59.0, -55.0], dtype=DTYPE),
            dtype=DTYPE,
        )
        for component in (point, cable, tree):
            component.insert(hh)
        if stimulation:
            point[..., 0].inject(dn.mono_rect(amp=0.45 * nA, delay=0.0, pw=0.05))
            tree[..., 2].inject(dn.mono_rect(amp=0.18 * nA, delay=0.01, pw=0.04))

        model = dn.concat_models(
            {"point": point, "cable": cable, "tree": tree},
            celsius=6.3,
            threads=2,
            write_back=write_back,
        )
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _real_peripheral_multi():
    peripheral = pytest.importorskip("dendra_models.models.cells.peripheral")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        rattay = peripheral.Rattay1993(
            [1.0],
            L=20.0,
            dx=10.0,
        ).double()
        sundt = peripheral.Sundt2015(
            [1.0],
            L=20.0,
            dx=10.0,
        ).double()
        model = dn.concat_models({"rattay": rattay, "sundt": sundt})
        model.batch(2)
        model.initialize()
        model.train()
    return model


def _myelinated_multi():
    peripheral = pytest.importorskip("dendra_models.models.cells.peripheral")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        myelin = peripheral.Sweeney1987(
            diameters=[8.0],
            n_node=3,
        ).double()
        point = dn.SingleCompartment(1, dtype=DTYPE)
        point.insert(hh)
        model = dn.concat_models(
            {"myelin": myelin, "point": point},
            threads=2,
        )
        model.initialize()
        model.train()
    return model


def _native_cable_multi():
    morphology = dn.Morphology(rhoa=90.0, cm=1.0)
    root = morphology.section("root", L=20.0, diam=4.0, nseg=2)
    tail = morphology.section("tail", L=30.0, diam=2.0, nseg=2)
    tail.connect(root.at(1.0))
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        cable = dn.Cable.from_morphology(
            morphology,
            N=1,
            dtype=DTYPE,
            v_init=torch.tensor([-70.0, -60.0, -50.0, -40.0], dtype=DTYPE),
        )
        point = dn.SingleCompartment(1, dtype=DTYPE)
        cable.insert(hh)
        point.insert(hh)
        model = dn.concat_models(
            {"native": cable, "point": point},
            threads=2,
        )
        model.initialize()
        model.train()
    return model


def _drives(model, *, steps=STEPS):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -3.0,
        4.0,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -2.5e-9,
        3.5e-9,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _initialize_imperative(model):
    dt = torch.as_tensor(DT, device=model.device(), dtype=model.dtype())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    return dt


def _imperative_step(model, dt, ve, intra):
    model.integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


def _clone_tree(value):
    return torch.utils._pytree.tree_map(
        lambda tensor: tensor.detach().clone(),
        value,
    )


def _assert_tree_close(actual, expected, *, rtol=2.0e-10, atol=2.0e-11):
    actual_with_paths, actual_spec = torch.utils._pytree.tree_flatten_with_path(actual)
    expected_with_paths, expected_spec = torch.utils._pytree.tree_flatten_with_path(
        expected
    )
    assert actual_spec == expected_spec
    for (actual_path, actual_leaf), (expected_path, expected_leaf) in zip(
        actual_with_paths,
        expected_with_paths,
        strict=True,
    ):
        assert actual_path == expected_path
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
            msg=lambda message: f"tensor leaf {actual_path}: {message}",
        )


def _canonical_multi_parameter_names(model):
    return {
        name
        for name, _parameter in model.named_parameters()
        if name == "celsius_param"
        or name.startswith("integrator.")
        or (
            name.startswith("populations.")
            and not name.endswith(".celsius_param")
            and ".integrator." not in name
        )
    }


def _component_voltage_views(model, packed_voltage):
    """Yield component names, models, and their exact packed voltage views."""
    offset = 0
    for name, component in model.populations.items():
        width = math.prod(component.core_shape())
        view = packed_voltage[..., 0, offset : offset + width].reshape(component.shape)
        yield name, component, view
        offset += width
    assert offset == packed_voltage.shape[-1]


def test_point_multi_keeps_one_flat_carry_and_matches_every_imperative_leaf():
    source = _point_multi()
    reference = _point_multi()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source)

    assert type(source.integrator).__name__ == "_bwd_euler_sc_multi"
    assert source.shape == (3, 2, 1, 4)
    assert functional.shape == source.shape
    assert tensors.state["integrator"]["v"].shape == source.shape
    assert "populations" not in tensors.state
    for value in tensors.state["mechanisms"]["hh"].values():
        assert value.shape == source.shape

    assert set(tensors.parameters) == _canonical_multi_parameter_names(source)
    assert set(tensors.constants) == {
        "dt",
        "populations.left.diam",
        "populations.left.dx",
        "populations.right.diam",
        "populations.right.dx",
    }
    assert not {
        "cm_param.rho",
        "rhoa_param.rho",
        "cm_scale_param.rho",
        "rhoa_scale_param.rho",
        "area_scale_param.rho",
    } & set(tensors.parameters)
    assert not any(
        name.startswith("populations.") and name.endswith(".celsius_param")
        for name in tensors.parameters
    )

    dt = _initialize_imperative(reference)
    state = tensors.state
    last_auxiliary = None
    for index in range(STEPS):
        state, last_auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(reference, dt, ve[index], intra[index])
        _assert_tree_close(state, functional.extract(reference).state)

    rolled = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_tree_close(rolled, (state, last_auxiliary))


def test_mixed_multi_matches_packed_imperative_step_and_rollout_every_leaf():
    source = _mixed_multi(batch_calls=(2,))
    reference = _mixed_multi(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source)

    assert type(source.integrator).__name__ == "_dhs_multi"
    assert source.shape == (2, 1, 7)
    assert functional.shape == source.shape
    assert not functional._transition.population.populations
    assert not functional._preparation.population.populations
    assert not _unregistered_module_alias_paths(functional._transition.population)
    assert not _unregistered_module_alias_paths(functional._preparation.population)
    for adapter in functional._preparation.component_physical_preparations:
        assert not _unregistered_module_alias_paths(adapter)
    for mapping in (
        functional._base_mapping,
        functional._preparation_base_mapping,
    ):
        assert not any(name.startswith("population.populations.") for name in mapping)
    for name in functional._multi_component_physical_parameter_names:
        assert functional._transition_parameter_slots[name] == ()
        assert functional._preparation_parameter_slots[name] == ()
    for name in (
        set(functional._population_parameter_names)
        - functional._multi_component_physical_parameter_names
    ):
        assert functional._transition_parameter_slots[name]
        assert functional._preparation_parameter_slots[name]
    assert set(tensors.parameters) == _canonical_multi_parameter_names(source)
    assert {
        "populations.point.diam",
        "populations.point.dx",
        "populations.cable.diam",
        "populations.cable.dx",
        "populations.tree.diam",
        "populations.tree.dx",
        "populations.tree.canonical_area_cm2",
        "populations.tree.canonical_edge_resistance_ohm",
        "dt",
    } == set(tensors.constants)

    dt = _initialize_imperative(reference)
    state = tensors.state
    auxiliaries = []
    for index in range(STEPS):
        state, auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        auxiliaries.append(auxiliary)
        _imperative_step(reference, dt, ve[index], intra[index])
        _assert_tree_close(state, functional.extract(reference).state)

    rolled_state, rolled_auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_tree_close(rolled_state, state)
    _assert_tree_close(rolled_auxiliary, auxiliaries[-1])


def test_mixed_multi_rollout_uses_only_one_packed_solver_call_per_step(monkeypatch):
    model = _mixed_multi()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model)

    packed_calls = 0
    packed_solver = functional._transition.solver

    def counted_packed_solver(*args, **kwargs):
        nonlocal packed_calls
        packed_calls += 1
        return packed_solver(*args, **kwargs)

    monkeypatch.setattr(
        functional._transition,
        "solver",
        counted_packed_solver,
    )

    with torch.no_grad():
        functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )

    assert packed_calls == STEPS


def test_multi_component_preparation_adapters_retain_no_runtime_state():
    model = _mixed_multi()
    functional, _tensors = dn.func.make_functional(model, dt=DT)
    adapters = functional._preparation.component_physical_preparations

    assert len(adapters) == len(model.populations)
    assert not any(
        name.startswith("component_physical_preparations.")
        for name in functional._preparation_base_mapping
    )
    for record, adapter in zip(
        functional._multi_component_plans,
        adapters,
        strict=True,
    ):
        module_names = {name for name, _module in adapter.named_modules()}
        parameter_names = {name for name, _value in adapter.named_parameters()}
        buffer_names = {name for name, _value in adapter.named_buffers()}
        registered_names = parameter_names | buffer_names

        assert not hasattr(record, "base_parameters")
        assert {name for name, _public_name in record.parameter_sources} == {
            name for name, _value in adapter.population.named_parameters()
        }
        assert {slot for _name, slots in record.parameter_slots for slot in slots} == {
            name for name, _value in adapter.named_parameters(remove_duplicate=False)
        }

        assert not any(
            name.startswith(("population.integrator", "population.mech"))
            for name in module_names | registered_names
        )
        assert {"population.v", "population.t", "population._dummy"}.isdisjoint(
            buffer_names
        )
        assert adapter.population._m_list == []
        assert adapter.population.injections == []
        assert adapter.population.mechanism_injections == []
        assert adapter.population.intra is None
        assert adapter.population.i_membrane is None


def test_multi_component_owner_buffer_transform_argument_is_explicit_and_fresh():
    source = _buffer_argument_multi(2.0)
    target = _buffer_argument_multi(3.0)
    functional, tensors = dn.func.make_functional(source, dt=DT)

    constant_name = "populations.cable.parametrizations.hidden_factor"
    assert constant_name in tensors.constants
    refreshed = functional.extract(target)
    assert refreshed.constants[constant_name].item() == 3.0

    prepared = functional.prepare(refreshed.parameters, refreshed.constants)
    torch.testing.assert_close(
        prepared.values["population"]["cm"],
        target.cm,
        rtol=0.0,
        atol=0.0,
    )

    factor = refreshed.constants[constant_name].detach().clone().requires_grad_()
    differentiated = functional.prepare(
        refreshed.parameters,
        {**refreshed.constants, constant_name: factor},
    )
    gradient = torch.autograd.grad(
        differentiated.values["population"]["cm"].sum(),
        factor,
    )[0]
    assert torch.isfinite(gradient)
    assert gradient.item() > 0.0


def test_multi_spatial_transform_parameter_supports_jacrev_and_vmap():
    model = _parameter_argument_multi()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameter_name = "populations.cable.spatial_factor"
    factor = tensors.parameters[parameter_name]

    def materialized_cm(local_factor):
        parameters = {**tensors.parameters, parameter_name: local_factor}
        return functional.prepare(parameters, tensors.constants).values["population"][
            "cm"
        ]

    torch.testing.assert_close(
        materialized_cm(factor),
        model.cm,
        rtol=0.0,
        atol=0.0,
    )
    jacobian = torch.func.jacrev(materialized_cm)(factor)
    assert torch.isfinite(jacobian).all()
    assert torch.count_nonzero(jacobian) == factor.numel()

    lanes = torch.stack((0.9 * factor, 1.1 * factor))
    actual = torch.vmap(materialized_cm)(lanes)
    expected = torch.stack(tuple(materialized_cm(lane) for lane in lanes))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_multi_transform_internal_parameter_buffer_alias_has_one_public_leaf():
    model = _cross_category_transform_multi()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameter_name = next(
        name for name in tensors.parameters if name.endswith("factor_parameter")
    )
    assert not any("factor_buffer" in name for name in tensors.constants)

    def materialized_cm(factor):
        parameters = {**tensors.parameters, parameter_name: factor}
        return functional.prepare(parameters, tensors.constants).values["population"][
            "cm"
        ]

    factor = tensors.parameters[parameter_name]
    changed = factor.detach().new_tensor(6.0).requires_grad_()
    actual = materialized_cm(changed)
    point_width = math.prod(model.populations["point"].core_shape())
    torch.testing.assert_close(
        actual[..., point_width:],
        3.0 * model.cm[..., point_width:],
        rtol=0.0,
        atol=0.0,
    )
    jacobian = torch.func.jacrev(materialized_cm)(changed)
    assert torch.isfinite(jacobian).all()
    assert torch.count_nonzero(jacobian[..., point_width:]) > 0


def test_mixed_multi_registered_intra_and_bound_extra_match_public_step():
    source = _mixed_multi(batch_calls=(2,), stimulation=True)
    reference = _mixed_multi(batch_calls=(2,), stimulation=True)
    field_source = torch.linspace(-1.5, 2.0, source.shape[-1], dtype=DTYPE).reshape(
        source.core_shape()
    )
    field_reference = field_source.clone()
    extra_source = (
        field_source,
        dn.mono_rect(amp=1.4, delay=0.0, pw=0.05),
    )
    extra_reference = (
        field_reference,
        dn.mono_rect(amp=1.4, delay=0.0, pw=0.05),
    )
    functional, tensors = dn.func.make_functional(
        source,
        dt=DT,
        extra=extra_source,
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    assert any(name.startswith("stimulation.intra.") for name in tensors.parameters)
    assert any(name.startswith("stimulation.extra.") for name in tensors.parameters)

    state = tensors.state
    for _index in range(STEPS):
        state, _auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
        )
        reference.step(dt=DT, extra=extra_reference)
        _assert_tree_close(state, functional.extract(reference).state)


def test_real_dendra_models_multi_matches_every_leaf_and_keeps_physics_gradient():
    source = _real_peripheral_multi()
    reference = _real_peripheral_multi()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=2)

    reference_dt = _initialize_imperative(reference)
    state = tensors.state
    for index in range(2):
        state, _auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        _imperative_step(reference, reference_dt, ve[index], intra[index])
        _assert_tree_close(state, functional.extract(reference).state)

    cm_name = "populations.rattay.cm_param"
    raw_cm = tensors.parameters[cm_name].detach().clone().requires_grad_()

    def response(local_cm):
        parameters = {**tensors.parameters, cm_name: local_cm}
        final_state, _auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )
        weights = torch.linspace(
            0.5,
            1.5,
            source.v.numel(),
            dtype=source.dtype(),
            device=source.device(),
        ).reshape(source.shape)
        return (final_state["integrator"]["v"] * weights).sum()

    gradient = torch.autograd.grad(response(raw_cm), raw_cm)[0]
    assert torch.isfinite(gradient)
    assert gradient != 0


def test_multi_component_regional_physics_is_explicit_and_differentiable():
    override = torch.tensor([[2.0], [3.0]], dtype=DTYPE)
    override_key = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8])
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        cable = dn.Unmyelinated(
            [1.5, 2.0],
            L=4.0,
            dx=1.0,
            dtype=DTYPE,
        )
        point = dn.SingleCompartment(1, dtype=DTYPE)
        cable.insert(hh)
        point.insert(hh)
        cable.parametrize("cm", override, key=override_key, alias="regional")
        model = dn.concat_models(
            {"cable": cable, "point": point},
            threads=2,
        )
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameter_name = "populations.cable.cm_regional"
    raw_cm = tensors.parameters[parameter_name]
    cable_width = cable.v.numel()

    def effective_cm(value):
        parameters = {**tensors.parameters, parameter_name: value}
        packed = functional.prepare(parameters, tensors.constants).values["population"][
            "cm"
        ]
        return packed[..., 0, :cable_width].reshape(cable.core_shape())

    expected = torch.cat((raw_cm.expand(2, 4), raw_cm.new_ones((2, 1))), dim=1)
    torch.testing.assert_close(effective_cm(raw_cm), expected, rtol=0.0, atol=0.0)

    expected_jacobian = raw_cm.new_zeros((2, 5, *raw_cm.shape))
    expected_jacobian[:, :4] = (
        torch.eye(
            raw_cm.numel(),
            device=raw_cm.device,
            dtype=raw_cm.dtype,
        )
        .reshape(*raw_cm.shape, *raw_cm.shape)
        .expand(2, 4, *raw_cm.shape)
    )
    torch.testing.assert_close(
        torch.func.jacrev(effective_cm)(raw_cm),
        expected_jacobian,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.func.jacfwd(effective_cm)(raw_cm),
        expected_jacobian,
        rtol=0.0,
        atol=0.0,
    )


def test_myelinated_multi_prepares_exact_parametrized_physics_and_gradient():
    model = _myelinated_multi()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    myelin = model.populations["myelin"]
    width = myelin.v.numel()
    adapter = functional._preparation.component_physical_preparations[0]

    assert {
        "populations.myelin.diameters",
        "populations.myelin.diam_original",
        "populations.myelin.dx",
    } <= set(tensors.constants)
    assert not {
        "population._dummy",
        "population.v",
        "population.t",
        "population.x",
        "population.y",
        "population.z",
    } & set(dict(adapter.named_buffers()))
    assert not any(
        name.startswith(("population.integrator.", "population.mech."))
        for name, _value in adapter.named_parameters()
    )
    torch.testing.assert_close(
        prepared.values["geometry"]["diam"][..., 0, :width],
        myelin.diam.reshape(-1),
    )
    torch.testing.assert_close(
        prepared.values["population"]["rhoa"][..., 0, :width],
        myelin.rhoa.reshape(-1),
    )

    raw_name = "populations.myelin.noded1"
    raw = tensors.parameters[raw_name].detach().clone().requires_grad_()
    ve, intra = _drives(model, steps=2)

    def response(local_raw):
        parameters = {**tensors.parameters, raw_name: local_raw}
        final_state, _auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )
        weights = torch.arange(
            1,
            model.v.numel() + 1,
            device=model.device(),
            dtype=model.dtype(),
        ).reshape(model.shape)
        return (final_state["integrator"]["v"] * weights).sum()

    gradient = torch.autograd.grad(response(raw), raw)[0]
    assert torch.isfinite(gradient)
    assert gradient != 0


def test_native_cable_multi_matches_packed_step_and_resistance_gradient():
    source = _native_cable_multi()
    reference = _native_cable_multi()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=1)

    actual, _auxiliary = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(ve=ve[0], intra=intra[0]),
    )
    reference_dt = _initialize_imperative(reference)
    _imperative_step(reference, reference_dt, ve[0], intra[0])
    _assert_tree_close(actual, functional.extract(reference).state)

    resistance_name = "populations.native.canonical_edge_resistance_ohm"
    resistance = tensors.constants[resistance_name].detach().clone().requires_grad_()
    constants = {**tensors.constants, resistance_name: resistance}
    perturbed, _auxiliary = functional.prepare_and_step(
        tensors.parameters,
        constants,
        tensors.state,
        dn.func.StepInput(ve=ve[0], intra=intra[0]),
    )
    weights = torch.arange(
        1,
        source.v.numel() + 1,
        device=source.device(),
        dtype=source.dtype(),
    ).reshape(source.shape)
    gradient = torch.autograd.grad(
        (perturbed["integrator"]["v"] * weights).sum(),
        resistance,
    )[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[..., 1:])


def test_mixed_multi_gradients_reach_component_physics_mechanism_state_and_drives():
    model = _mixed_multi()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps=2)

    arguments = tuple(
        value.detach().clone().requires_grad_()
        for value in (
            tensors.parameters[POINT_CM],
            tensors.parameters[CABLE_RHOA],
            tensors.parameters[HH_GNABAR],
            tensors.state["integrator"]["v"],
            ve,
            intra,
        )
    )

    def response(point_cm, cable_rhoa, gnabar, voltage, local_ve, local_intra):
        parameters = {
            **tensors.parameters,
            POINT_CM: point_cm,
            CABLE_RHOA: cable_rhoa,
            HH_GNABAR: gnabar,
        }
        state = {
            **tensors.state,
            "integrator": {
                **tensors.state["integrator"],
                "v": voltage,
            },
        }
        final_state, _auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=local_ve, intra=local_intra),
        )
        weights = torch.linspace(
            0.4,
            1.6,
            model.v.numel(),
            dtype=model.dtype(),
            device=model.device(),
        ).reshape(model.shape)
        return (final_state["integrator"]["v"] * weights).sum()

    loss = response(*arguments)
    gradients = torch.autograd.grad(loss, arguments)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0

    def voltage_from_cable_rhoa(cable_rhoa):
        parameters = {**tensors.parameters, CABLE_RHOA: cable_rhoa}
        return functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(ve=ve[0], intra=intra[0]),
        )[0]["integrator"]["v"]

    jacobian = torch.func.jacrev(voltage_from_cable_rhoa)(
        tensors.parameters[CABLE_RHOA]
    )
    point_width = math.prod(model.populations["point"].core_shape())
    cable_width = math.prod(model.populations["cable"].core_shape())
    assert torch.count_nonzero(jacobian[..., :point_width]) == 0
    assert (
        torch.count_nonzero(jacobian[..., point_width : point_width + cable_width]) > 0
    )


def _mixed_transform_case(*, batch_calls=(2,)):
    model = _mixed_multi(batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, steps=1)
    raw_rhoa = tensors.parameters[CABLE_RHOA]

    def response(local_rhoa, local_intra):
        parameters = {**tensors.parameters, CABLE_RHOA: local_rhoa}
        state, _auxiliary = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(ve=ve[0], intra=local_intra),
        )
        return state["integrator"]["v"]

    return model, response, raw_rhoa, intra[0]


def test_mixed_multi_jacrev_and_jacfwd_agree():
    _model, response, raw_rhoa, intra = _mixed_transform_case(batch_calls=())
    reverse = torch.func.jacrev(response, argnums=0)(raw_rhoa, intra)
    with torch_compiler_warning_context():
        forward = torch.func.jacfwd(response, argnums=0)(raw_rhoa, intra)

    assert torch.isfinite(reverse).all()
    assert torch.count_nonzero(reverse) > 0
    torch.testing.assert_close(reverse, forward, rtol=3.0e-9, atol=3.0e-11)


def test_mixed_multi_direct_nested_and_empty_vmap_compose_with_explicit_batch():
    model, response, raw_rhoa, intra = _mixed_transform_case(batch_calls=(2,))

    rhoa_lanes = raw_rhoa + raw_rhoa.new_tensor((-0.15, 0.0, 0.2))
    intra_lanes = torch.stack((0.7 * intra, intra, 1.3 * intra))
    actual = torch.vmap(response)(rhoa_lanes, intra_lanes)
    expected = torch.stack(
        tuple(
            response(local_rhoa, local_intra)
            for local_rhoa, local_intra in zip(
                rhoa_lanes,
                intra_lanes,
                strict=True,
            )
        )
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)

    nested_rhoa = raw_rhoa + raw_rhoa.new_tensor(((-0.2, 0.05), (0.15, 0.3)))
    offsets = intra.new_tensor(((-2.0e-10, 0.0), (1.0e-10, 3.0e-10)))
    nested_intra = intra.reshape((1, 1, *model.shape)) + offsets.reshape(
        (2, 2, *((1,) * len(model.shape)))
    )
    nested_actual = torch.vmap(torch.vmap(response))(nested_rhoa, nested_intra)
    nested_expected = torch.stack(
        tuple(
            torch.stack(
                tuple(
                    response(nested_rhoa[outer, inner], nested_intra[outer, inner])
                    for inner in range(2)
                )
            )
            for outer in range(2)
        )
    )
    torch.testing.assert_close(
        nested_actual,
        nested_expected,
        rtol=2.0e-10,
        atol=2.0e-11,
    )

    empty = torch.vmap(response)(
        raw_rhoa.new_empty((0,)),
        intra.new_empty((0, *model.shape)),
    )
    assert empty.shape == (0, *model.shape)


def test_mixed_multi_compile_of_jacrev_matches_eager():
    _model, response, raw_rhoa, intra = _mixed_transform_case(batch_calls=())
    transformed = torch.func.jacrev(response, argnums=0)
    expected = transformed(raw_rhoa, intra)

    with torch_compiler_warning_context():
        compiled = torch.compile(
            transformed,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled(raw_rhoa, intra)

    torch.testing.assert_close(actual, expected, rtol=3.0e-9, atol=3.0e-11)


@pytest.mark.parametrize("runner_name", ["run", "longrun", "longrun_checkpointed"])
def test_mixed_multi_host_runners_match_direct_rollout(runner_name):
    model = _mixed_multi(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model)
    inputs = dn.func.RolloutInput(ve=ve, intra=intra)
    expected = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        inputs,
    )
    step = partial(functional.step, tensors.parameters, prepared)

    if runner_name == "run":
        actual = dn.func.run(functional, step, tensors.state, inputs)
    else:
        actual = getattr(dn.func, runner_name)(
            functional,
            step,
            tensors.state,
            STEPS * DT,
            2,
            inputs,
        )

    _assert_tree_close(actual, expected)


@pytest.mark.parametrize("write_back", [True, False])
def test_mixed_multi_commit_is_atomic_syncs_expected_views_and_resumes(write_back):
    source = _mixed_multi(batch_calls=(2,), write_back=write_back)
    target = _mixed_multi(batch_calls=(2,), write_back=write_back)
    reference = _mixed_multi(batch_calls=(2,), write_back=write_back)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=5)

    checkpoint, _auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve[:3], intra=intra[:3]),
    )

    target_before = functional.extract(target).state
    child_before = {
        name: component.v.detach().clone()
        for name, component in target.populations.items()
    }
    invalid = _clone_tree(checkpoint)
    invalid["integrator"]["v"] = invalid["integrator"]["v"] + 10.0
    invalid["control"]["duration_remainder"] = invalid["control"][
        "duration_remainder"
    ].new_tensor(-1.0)
    with pytest.raises(ValueError, match="duration_remainder"):
        functional.commit_state_(target, invalid)
    _assert_tree_close(functional.extract(target).state, target_before)
    for name, component in target.populations.items():
        torch.testing.assert_close(component.v, child_before[name], rtol=0.0, atol=0.0)

    functional.commit_state_(target, checkpoint)
    _assert_tree_close(functional.extract(target).state, checkpoint)
    for name, component, expected in _component_voltage_views(
        target,
        checkpoint["integrator"]["v"],
    ):
        if write_back:
            torch.testing.assert_close(component.v, expected, rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(
                component.v,
                child_before[name],
                rtol=0.0,
                atol=0.0,
            )

    target_dt = _initialize_imperative(target)
    for index in range(3, 5):
        _imperative_step(target, target_dt, ve[index], intra[index])
    reference_dt = _initialize_imperative(reference)
    for index in range(5):
        _imperative_step(reference, reference_dt, ve[index], intra[index])
    _assert_tree_close(
        functional.extract(target).state,
        functional.extract(reference).state,
    )


def test_multi_functionalization_rejects_imem_recording():
    with dn.ctx(IMEM=1):
        model = _point_multi(batch_calls=())

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="i_membrane recording is not supported yet",
    ):
        dn.func.make_functional(model, dt=DT)


def test_failed_multi_child_admission_restores_class_and_global_rng_state():
    class ImpurePhysicalTransform(torch.nn.Module):
        calls = 0

        def forward(self, value):
            type(self).calls += 1
            torch.rand(())
            random.random()
            np.random.random()
            return value

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        cable = dn.Unmyelinated([1.5], L=3.0, dx=1.0, dtype=DTYPE)
        point = dn.SingleCompartment(1, dtype=DTYPE)
        cable.insert(hh)
        point.insert(hh)
        cable.register_parametrization_in_graph(
            "rhoa",
            ImpurePhysicalTransform(),
        )
        model = dn.concat_models({"cable": cable, "point": point}, threads=2)
        model.initialize()
        model.train()

    ImpurePhysicalTransform.calls = 0
    expected_torch_rng = torch.random.get_rng_state().clone()
    expected_python_rng = random.getstate()
    expected_numpy_rng = copy.deepcopy(np.random.get_state())
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="class state|RNG state",
    ):
        dn.func.make_functional(model, dt=DT)

    assert ImpurePhysicalTransform.calls == 0
    assert torch.equal(torch.random.get_rng_state(), expected_torch_rng)
    assert random.getstate() == expected_python_rng
    actual_numpy_rng = np.random.get_state()
    assert actual_numpy_rng[0] == expected_numpy_rng[0]
    assert np.array_equal(actual_numpy_rng[1], expected_numpy_rng[1])
    assert actual_numpy_rng[2:] == expected_numpy_rng[2:]


def test_multi_functionalization_rejects_noncanonical_component_subclasses():
    class CustomPoint(dn.SingleCompartment):
        @property
        def area(self):
            return super().area

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        custom = CustomPoint(N=1, C=1, dtype=DTYPE)
        ordinary = dn.SingleCompartment(N=1, C=1, dtype=DTYPE)
        custom.insert(hh)
        ordinary.insert(hh)
        model = dn.concat_models({"custom": custom, "ordinary": ordinary})
        model.initialize()
        model.train()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"component 'custom': the current supported topologies",
    ):
        dn.func.make_functional(model, dt=DT)


@pytest.mark.parametrize("incompatibility", ["component_order", "tree_geometry"])
def test_multi_extract_and_commit_reject_incompatible_targets_before_mutation(
    incompatibility,
):
    if incompatibility == "component_order":
        source = _point_multi(batch_calls=())
        target = _point_multi(
            batch_calls=(),
            component_order=("right", "left"),
        )
        error = "ordered component structure"
    else:
        source = _mixed_multi()
        target = _mixed_multi(
            tree_graph=_tree_graph(edge_resistance_scale=1.25),
        )
        error = "canonical component geometry values"

    functional, tensors = dn.func.make_functional(source, dt=DT)
    target_voltage = target.v.detach().clone()
    component_voltages = {
        name: component.v.detach().clone()
        for name, component in target.populations.items()
    }

    with pytest.raises(dn.func.FunctionalizationError, match=error):
        functional.extract(target)
    with pytest.raises(dn.func.FunctionalizationError, match=error):
        functional.commit_state_(target, tensors.state)

    torch.testing.assert_close(target.v, target_voltage, rtol=0.0, atol=0.0)
    for name, component in target.populations.items():
        torch.testing.assert_close(
            component.v,
            component_voltages[name],
            rtol=0.0,
            atol=0.0,
        )


def test_multi_extract_accepts_equal_values_for_source_aliased_constants():
    source = _point_multi_with_dx_alias()
    target = _point_multi_with_dx_alias(target_right_dx=1.0)
    assert source.populations["left"].dx is source.populations["right"].dx
    assert target.populations["left"].dx is not target.populations["right"].dx

    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert "populations.left.dx" in tensors.constants
    assert "populations.right.dx" not in tensors.constants

    target_tensors = functional.extract(target)
    prepared = functional.prepare(
        target_tensors.parameters,
        target_tensors.constants,
    )
    torch.testing.assert_close(
        prepared.values["geometry"]["dx"],
        target.dx,
        rtol=0.0,
        atol=0.0,
    )


def test_multi_extract_rejects_divergent_values_for_source_aliased_constants():
    source = _point_multi_with_dx_alias()
    target = _point_multi_with_dx_alias(target_right_dx=2.0)
    functional, _tensors = dn.func.make_functional(source, dt=DT)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="aliased constant 'populations.left.dx'.*exactly equal",
    ):
        functional.extract(target)


def test_multi_source_rejects_rebound_canonical_tree_and_stale_component_shape():
    model = _mixed_multi()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    tree = model.populations["tree"]
    tree._compartment_graph = copy.deepcopy(tree.compartment_graph)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="canonical MultiPopulation component geometry changed",
    ):
        functional.prepare(tensors.parameters, tensors.constants)

    model = _point_multi(batch_calls=())
    functional, tensors = dn.func.make_functional(model, dt=DT)
    model.populations["left"].batch(2)

    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)


def test_multi_step_rejects_corrupted_prepared_plan_and_malformed_state():
    model = _mixed_multi()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    workspace_name = next(
        name for name in prepared.values["integrator"] if name != "dt"
    )
    prepared.values["integrator"].pop(workspace_name)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="prepared workspaces changed",
    ):
        functional.step(tensors.parameters, prepared, tensors.state)

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    malformed_state = _clone_tree(tensors.state)
    malformed_state["integrator"]["v"] = malformed_state["integrator"]["v"][..., :-1]
    with pytest.raises(ValueError, match=r"state leaf 'integrator\.v'.*expected"):
        functional.step(tensors.parameters, prepared, malformed_state)


def test_multi_commit_rolls_back_a_failure_after_parent_restore(monkeypatch):
    source = _mixed_multi(batch_calls=(2,))
    target = _mixed_multi(batch_calls=(2,))
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, steps=2)
    checkpoint, _auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )

    # Exercise the exact rollback contract with a graph-bearing packed voltage
    # and the ordinary write-back views installed on every component.
    voltage_leaf = target.v.detach().clone().requires_grad_()
    target.v = voltage_leaf + target.v.new_zeros(())
    for _name, component, view in _component_voltage_views(target, target.v):
        component.v = view

    target_before = functional.extract(target).state
    registered_buffers = {
        name: value for name, value in target.named_buffers(remove_duplicate=False)
    }
    duration_remainder = target._duration_remainder
    parent_voltage = target.v
    parent_grad_fn = target.v.grad_fn
    component_voltages = {
        name: (
            component.v,
            component.v.grad_fn,
            component.v._base,
            component.v.data_ptr(),
            component.v.untyped_storage().data_ptr(),
            component.v.storage_offset(),
            component.v.stride(),
        )
        for name, component in target.populations.items()
    }

    original_restore = target.restore_dict_from_checkpoint
    restore_calls = 0

    def fail_after_first_restore(state):
        nonlocal restore_calls
        restore_calls += 1
        result = original_restore(state)
        if restore_calls == 1:
            raise RuntimeError("injected restore failure")
        return result

    monkeypatch.setattr(
        target,
        "restore_dict_from_checkpoint",
        fail_after_first_restore,
    )
    with pytest.raises(RuntimeError, match="injected restore failure"):
        functional.commit_state_(target, checkpoint)

    assert restore_calls == 1
    _assert_tree_close(functional.extract(target).state, target_before)
    assert target._duration_remainder is duration_remainder
    assert target.v is parent_voltage
    assert target.v.requires_grad
    assert target.v.grad_fn is parent_grad_fn
    for name, value in target.named_buffers(remove_duplicate=False):
        assert value is registered_buffers[name]
    for name, component in target.populations.items():
        (
            voltage,
            grad_fn,
            base,
            data_ptr,
            storage_data_ptr,
            storage_offset,
            stride,
        ) = component_voltages[name]
        assert component.v is voltage
        assert component.v.requires_grad
        assert component.v.grad_fn is grad_fn
        assert component.v._base is base is parent_voltage
        assert component.v.data_ptr() == data_ptr
        assert component.v.untyped_storage().data_ptr() == storage_data_ptr
        assert component.v.storage_offset() == storage_offset
        assert component.v.stride() == stride
        torch.testing.assert_close(
            component.v,
            voltage,
            rtol=0.0,
            atol=0.0,
        )
