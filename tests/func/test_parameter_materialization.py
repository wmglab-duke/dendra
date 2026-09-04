"""Contracts for shared raw-to-effective Population parameter materialization."""

from __future__ import annotations

import copy
import random
from types import MethodType

import numpy as np
import pytest
import torch
from torch.func import functional_call

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.core import Myelinated
from dendra.models.mod import hh, pas
from dendra.models.parametric import PositiveParam

DT = 0.01


def _single_compartment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.SingleCompartment(
            N=2,
            C=3,
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        population.insert(pas, g=0.001, e=-70.0)
        population.initialize()
    return population


def _unmyelinated():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(pas, g=0.001, e=-70.0)
        population.initialize()
    return population


class _ScaleTransform(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = float(factor)

    def forward(self, value):
        return self.factor * value


class _ScaleByArgument(torch.nn.Module):
    def forward(self, value, factor):
        return value * factor


class _CrossCategoryScale(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        shared = torch.nn.Parameter(torch.as_tensor(factor, dtype=torch.float64))
        self.register_parameter("factor_parameter", shared)
        self.register_buffer("factor_buffer", shared)

    def forward(self, value):
        return value * self.factor_buffer


class _ParameterScale(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.as_tensor(factor, dtype=torch.float64))

    def forward(self, value):
        return value * self.factor


def _scaled_unmyelinated(factor):
    population = _unmyelinated()
    transform = _ScaleTransform(factor)
    population.register_parametrization_in_graph("cm", transform)
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()
    return population, transform


def _buffer_argument_unmyelinated(factor):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.Unmyelinated(
            [1.5],
            L=3.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(hh)
        population.register_buffer(
            "hidden_factor",
            torch.tensor(float(factor), dtype=torch.float64),
        )
        population.register_parametrization_in_graph(
            "cm",
            _ScaleByArgument(),
            args=("hidden_factor",),
        )
        population.initialize()
        population.train()
    return population


def _parameter_argument_unmyelinated():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.Unmyelinated(
            [1.5],
            L=3.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(hh)
        population.register_parameter(
            "spatial_factor",
            torch.nn.Parameter(
                torch.linspace(0.8, 1.2, population.v.numel(), dtype=torch.float64)
                .reshape(population.shape)
                .requires_grad_()
            ),
        )
        population.register_parametrization_in_graph(
            "cm",
            _ScaleByArgument(),
            args=("spatial_factor",),
        )
        population.initialize()
        population.train()
    return population


def _cross_category_parameter_argument_unmyelinated():
    population = _unmyelinated()
    factor = torch.nn.Parameter(torch.tensor(2.0, dtype=population.dtype()))
    population.register_parameter("shared_factor", factor)
    population.register_buffer("hidden_factor", factor)
    population.register_parametrization_in_graph(
        "cm",
        _ScaleByArgument(),
        args=("hidden_factor",),
    )
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()
    population.train()
    return population


def _cross_category_transform_unmyelinated():
    population = _unmyelinated()
    population.register_parametrization_in_graph(
        "cm",
        _CrossCategoryScale(2.0),
    )
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()
    population.train()
    return population


def _parameter_transform_unmyelinated():
    population = _unmyelinated()
    transform = _ParameterScale(2.0)
    population.register_parametrization_in_graph("cm", transform)
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()
    population.train()
    return population, transform


@pytest.mark.parametrize("factory", [_single_compartment, _unmyelinated])
def test_pure_parameter_derivation_matches_imperative_without_mutation(factory):
    population = factory()
    parameter_snapshot = {
        name: (id(value), value._version, value.detach().clone())
        for name, value in population.named_parameters()
    }
    buffer_snapshot = {
        name: (id(value), value._version, value.detach().clone())
        for name, value in population.named_buffers()
    }

    values = population._derive_parameter_buffers()

    assert set(values) == {
        "celsius",
        "cm",
        "rhoa",
        "rhoa_scale",
        "cm_scale",
        "area_scale",
    }
    for name, value in values.items():
        torch.testing.assert_close(value, getattr(population, name), rtol=0.0, atol=0.0)

    for name, (identity, version, expected) in parameter_snapshot.items():
        actual = dict(population.named_parameters())[name]
        assert id(actual) == identity
        assert actual._version == version
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for name, (identity, version, expected) in buffer_snapshot.items():
        actual = dict(population.named_buffers())[name]
        assert id(actual) == identity
        assert actual._version == version
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_functional_preparation_uses_the_authored_bounded_parameter_transform():
    population = _single_compartment()
    population.cm_param = PositiveParam(
        torch.tensor(1.25, dtype=torch.float64),
        include_zero=True,
        max_val=2.0,
        cap_mode="hard-ste",
        requires_grad=True,
    )
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()

    functional, tensors = dn.func.make_functional(population, dt=DT)
    raw = (tensors.parameters["cm_param.rho"].detach().clone() + 0.25).requires_grad_()
    parameters = {**tensors.parameters, "cm_param.rho": raw}
    prepared = functional.prepare(parameters, tensors.constants)
    expected = functional_call(
        population.cm_param,
        {"rho": raw},
        (),
        tie_weights=False,
    ).expand(population._calc_shape_p())

    torch.testing.assert_close(
        prepared.values["population"]["cm"],
        expected,
        rtol=0.0,
        atol=0.0,
    )
    gradient = torch.autograd.grad(
        prepared.values["population"]["cm"].sum(),
        raw,
    )[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient)


def test_registered_owner_buffer_transform_argument_is_an_explicit_constant():
    source = _buffer_argument_unmyelinated(2.0)
    target = _buffer_argument_unmyelinated(3.0)
    functional, tensors = dn.func.make_functional(source, dt=DT)

    constant_name = "parametrizations.hidden_factor"
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


def test_unregistered_owner_tensor_transform_argument_fails_closed():
    population = _unmyelinated()
    population.hidden_factor = torch.tensor(2.0, dtype=population.dtype())
    population.register_parametrization_in_graph(
        "cm",
        _ScaleByArgument(),
        args=("hidden_factor",),
    )
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="must be a registered buffer or Parameter",
    ):
        dn.func.make_functional(population, dt=DT)


def test_unregistered_transform_target_cannot_hide_a_later_argument_dependency():
    population = _unmyelinated()
    population.hidden = torch.tensor(2.0, dtype=population.dtype())
    population.register_parametrization_in_graph(
        "cm",
        _ScaleByArgument(),
        args=("hidden",),
    )
    population.register_parametrization_in_graph(
        "hidden",
        _ScaleTransform(1.0),
    )
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="must be a registered buffer or Parameter",
    ):
        dn.func.make_functional(population, dt=DT)


def test_transform_argument_descriptor_is_rejected_without_touching_source():
    class DescriptorArgumentUnmyelinated(dn.Unmyelinated):
        @property
        def hidden_factor(self):
            self.arg_reads += 1
            return self._hidden_factor

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = DescriptorArgumentUnmyelinated(
            [1.5],
            L=3.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(hh)
        population.register_buffer(
            "_hidden_factor",
            torch.tensor(2.0, dtype=torch.float64),
        )
        population.arg_reads = 0
        population.register_parametrization_in_graph(
            "cm",
            _ScaleByArgument(),
            args=("hidden_factor",),
        )
        population.initialize()
        population.train()
    population.arg_reads = 0

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="not a directly registered Tensor",
    ):
        dn.func.make_functional(population, dt=DT)
    assert population.arg_reads == 0


def test_registered_buffer_transform_target_without_raw_base_fails_closed():
    population = _unmyelinated()
    population.register_buffer(
        "hidden",
        torch.tensor(2.0, dtype=population.dtype()),
    )
    population.register_parametrization_in_graph(
        "hidden",
        _ScaleTransform(1.0),
    )
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="has no reconstructed parameter base",
    ):
        dn.func.make_functional(population, dt=DT)


def test_transform_argument_alias_of_geometry_reuses_the_geometry_constant():
    population = _unmyelinated()
    population.register_buffer("hidden_factor", population.dx)
    population.register_parametrization_in_graph(
        "cm",
        _ScaleByArgument(),
        args=("hidden_factor",),
    )
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()
    functional, tensors = dn.func.make_functional(population, dt=DT)

    assert "dx" in tensors.constants
    assert not any(
        name.startswith("parametrizations.") and "dx" in name
        for name in tensors.constants
    )
    baseline = functional.prepare(tensors.parameters, tensors.constants)
    constants = {**tensors.constants, "dx": 1.5 * tensors.constants["dx"]}
    changed = functional.prepare(tensors.parameters, constants)
    torch.testing.assert_close(
        changed.values["population"]["cm"],
        1.5 * baseline.values["population"]["cm"],
        rtol=0.0,
        atol=0.0,
    )


def test_spatial_registered_parameter_transform_argument_supports_func_transforms():
    population = _parameter_argument_unmyelinated()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    factor = tensors.parameters["spatial_factor"]

    def materialized_cm(local_factor):
        parameters = {**tensors.parameters, "spatial_factor": local_factor}
        return functional.prepare(parameters, tensors.constants).values["population"][
            "cm"
        ]

    torch.testing.assert_close(
        materialized_cm(factor),
        population.cm,
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


def test_parameter_buffer_alias_routes_from_one_public_parameter_leaf():
    population = _cross_category_parameter_argument_unmyelinated()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    assert "shared_factor" in tensors.parameters
    assert not any("hidden_factor" in name for name in tensors.constants)

    def materialized_cm(factor):
        parameters = {**tensors.parameters, "shared_factor": factor}
        return functional.prepare(parameters, tensors.constants).values["population"][
            "cm"
        ]

    factor = tensors.parameters["shared_factor"]
    changed = factor.detach().new_tensor(6.0).requires_grad_()
    actual = materialized_cm(changed)
    torch.testing.assert_close(
        actual,
        3.0 * population.cm,
        rtol=0.0,
        atol=0.0,
    )
    jacobian = torch.func.jacrev(materialized_cm)(changed)
    assert torch.isfinite(jacobian).all()
    assert torch.count_nonzero(jacobian) == population.cm.numel()

    def next_voltage(local_factor):
        parameters = {**tensors.parameters, "shared_factor": local_factor}
        state, _auxiliary = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return state["integrator"]["v"]

    transformed = torch.func.jacrev(next_voltage)
    expected_voltage_jacobian = transformed(changed)
    with torch_compiler_warning_context():
        compiled = torch.compile(
            transformed,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        compiled_voltage_jacobian = compiled(changed)
    torch.testing.assert_close(
        compiled_voltage_jacobian,
        expected_voltage_jacobian,
        rtol=0.0,
        atol=0.0,
    )


def test_transform_internal_parameter_buffer_alias_has_one_public_leaf():
    population = _cross_category_transform_unmyelinated()
    functional, tensors = dn.func.make_functional(population, dt=DT)
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
    torch.testing.assert_close(actual, 3.0 * population.cm, rtol=0.0, atol=0.0)
    jacobian = torch.func.jacrev(materialized_cm)(changed)
    assert torch.isfinite(jacobian).all()
    assert torch.count_nonzero(jacobian) == population.cm.numel()

    def next_voltage(local_factor):
        parameters = {**tensors.parameters, parameter_name: local_factor}
        state, _auxiliary = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return state["integrator"]["v"]

    transformed = torch.func.jacrev(next_voltage)
    expected = transformed(changed)
    with torch_compiler_warning_context():
        compiled = torch.compile(
            transformed,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled(changed)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_extract_rejects_compatible_target_parameter_dtype_mismatch_immediately():
    source, _source_transform = _parameter_transform_unmyelinated()
    target, target_transform = _parameter_transform_unmyelinated()
    functional, _tensors = dn.func.make_functional(source, dt=DT)
    target_transform.factor = torch.nn.Parameter(
        target_transform.factor.detach().float()
    )

    with pytest.raises(
        ValueError,
        match=r"parameter .*factor.* must use cpu/torch.float64",
    ):
        functional.extract(target)


def test_parameter_alias_of_canonical_geometry_fails_closed():
    population = _unmyelinated()
    shared_dx = torch.nn.Parameter(population.dx.detach().clone())
    population._buffers["dx"] = shared_dx
    population.register_parameter("geometry_parameter", shared_dx)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="canonical geometry tensors cannot also be registered Parameters",
    ):
        dn.func.make_functional(population, dt=DT)


def test_transform_parameter_alias_of_runtime_state_fails_closed():
    population = _cross_category_parameter_argument_unmyelinated()
    population._buffers["t"] = population.shared_factor

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"Parameter/buffer aliases.*conflicting ownership=.*t",
    ):
        dn.func.make_functional(population, dt=DT)


class _MyelinatedMaterializer(torch.nn.Module):
    def __init__(self, population):
        super().__init__()
        self.population = population

    def forward(self):
        values = self.population._derive_parameter_buffers()
        return values["diam"], values["rhoa"]


def _myelinated():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = Myelinated(
            [8.0, 12.0],
            5,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(pas, g=0.001, e=-70.0)
        population.initialize()
    return population


def _alias_mapping(module, replacements):
    canonical = dict(module.named_parameters())
    canonical_by_identity = {id(value): name for name, value in canonical.items()}
    mapping = {}
    for path, value in module.named_parameters(remove_duplicate=False):
        name = canonical_by_identity[id(value)]
        mapping[path] = replacements.get(name, canonical[name])
    return mapping


def test_myelinated_parameter_transforms_are_registered_and_purely_materialized():
    population = _myelinated()
    module_names = dict(population.named_modules())
    transform_name = "_in_graph_parametrization_rhoa_0"
    assert transform_name in module_names
    assert (
        module_names[transform_name]
        is population.in_graph_parametrizations["rhoa"][0][0]
    )
    assert f"{transform_name}.axond2" in population.state_dict()
    assert f"{transform_name}.deltax2" in population.state_dict()

    values = population._derive_parameter_buffers()
    torch.testing.assert_close(values["diam"], population.diam, rtol=0.0, atol=0.0)
    torch.testing.assert_close(values["rhoa"], population.rhoa, rtol=0.0, atol=0.0)

    cloned = copy.deepcopy(population).float()
    assert transform_name in dict(cloned.named_modules())
    assert getattr(cloned, transform_name).axond2.dtype == torch.float32
    cloned.populate_parameter_buffers()
    torch.testing.assert_close(
        cloned._derive_parameter_buffers()["rhoa"],
        cloned.rhoa,
        rtol=0.0,
        atol=0.0,
    )


def test_myelinated_strict_load_accepts_checkpoint_without_new_transform_aliases():
    source = _myelinated()
    state = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith("_in_graph_parametrization_rhoa_0.")
    }
    target = _myelinated()

    result = target.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    target.populate_parameter_buffers()
    torch.testing.assert_close(target.diam, source.diam, rtol=0.0, atol=0.0)
    torch.testing.assert_close(target.rhoa, source.rhoa, rtol=0.0, atol=0.0)


def test_myelinated_shared_alias_substitution_has_exact_geometry_gradients():
    population = _myelinated()
    materializer = _MyelinatedMaterializer(population)
    public = dict(materializer.named_parameters())
    replacements = {
        name: public[name].detach().clone().requires_grad_()
        for name in ("population.noded2", "population.axond2", "population.deltax2")
    }
    diam, rhoa = functional_call(
        materializer,
        _alias_mapping(materializer, replacements),
        (),
        tie_weights=False,
    )

    fiber_diameter = population.diameters.unsqueeze(1).expand_as(diam)
    torch.testing.assert_close(
        diam,
        replacements["population.noded2"] * fiber_diameter,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    expected_rhoa = (
        public["population.rhoa_param"]
        * (replacements["population.noded2"] / replacements["population.axond2"]) ** 2
        * (replacements["population.deltax2"] * fiber_diameter / population.dx)
    )
    torch.testing.assert_close(rhoa, expected_rhoa, rtol=1.0e-12, atol=1.0e-12)

    gradients = torch.autograd.grad(
        rhoa.sum(),
        tuple(replacements.values()),
    )
    expected_gradients = (
        (2.0 * rhoa / replacements["population.noded2"]).sum(),
        (-2.0 * rhoa / replacements["population.axond2"]).sum(),
        (rhoa / replacements["population.deltax2"]).sum(),
    )
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
        assert torch.count_nonzero(actual)


def test_myelinated_preserves_independent_geometry_sources_and_gradients():
    population = _myelinated()
    original = population.parametrizations.diam.original
    with torch.no_grad():
        original.copy_(1.15 * original)
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()

    functional, tensors = dn.func.make_functional(population, dt=DT)
    assert set(tensors.constants) == {
        "diameters",
        "diam_original",
        "dx",
        "dt",
    }
    torch.testing.assert_close(
        tensors.constants["diam_original"],
        population.parametrizations.diam.original,
        rtol=0.0,
        atol=0.0,
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    torch.testing.assert_close(
        prepared.values["population"]["diam"],
        population.diam,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        prepared.values["population"]["rhoa"],
        population.rhoa,
        rtol=0.0,
        atol=0.0,
    )

    changed_diameters = 1.1 * tensors.constants["diameters"]
    diameter_constants = {**tensors.constants, "diameters": changed_diameters}
    diameter_prepared = functional.prepare(tensors.parameters, diameter_constants)
    torch.testing.assert_close(
        diameter_prepared.values["population"]["diam"],
        prepared.values["population"]["diam"],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        diameter_prepared.values["population"]["rhoa"],
        prepared.values["population"]["rhoa"],
    )

    changed_original = 0.9 * tensors.constants["diam_original"]
    original_constants = {**tensors.constants, "diam_original": changed_original}
    original_prepared = functional.prepare(tensors.parameters, original_constants)
    assert not torch.equal(
        original_prepared.values["population"]["diam"],
        prepared.values["population"]["diam"],
    )

    ve = torch.linspace(
        -2.0,
        2.0,
        population.v.numel(),
        dtype=population.dtype(),
    ).reshape(population.shape)

    def next_voltage(diameters, diam_original):
        constants = {
            **tensors.constants,
            "diameters": diameters,
            "diam_original": diam_original,
        }
        state, _aux = functional.prepare_and_step(
            tensors.parameters,
            constants,
            tensors.state,
            dn.func.StepInput(ve=ve),
        )
        return state["integrator"]["v"]

    jacobians = torch.func.jacrev(next_voltage, argnums=(0, 1))(
        tensors.constants["diameters"],
        tensors.constants["diam_original"],
    )
    for jacobian in jacobians:
        assert torch.isfinite(jacobian).all()
        assert torch.count_nonzero(jacobian)


@pytest.mark.parametrize("override_kind", ["instance", "subclass"])
def test_myelinated_rejects_nonstandard_parametrization_hooks(override_kind):
    population = _myelinated()
    transform_name = "_in_graph_parametrization_rhoa_0"
    transform, args = population.in_graph_parametrizations["rhoa"][0]
    if override_kind == "instance":
        transform.forward = lambda rhoa, _dx, _diameters, _diam: rhoa
    else:

        class AlteredRhoa(type(transform)):
            def forward(self, rhoa, _dx, _diameters, _diam):
                return rhoa

        transform = AlteredRhoa(
            transform.deltax1,
            transform.deltax2,
            transform.deltax3,
            transform.axond1,
            transform.axond2,
            transform.axond3,
        )
        setattr(population, transform_name, transform)
        population.in_graph_parametrizations["rhoa"][0] = (transform, args)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="standard registered diameter and axial-resistivity parametrizations",
    ):
        dn.func.make_functional(population, dt=DT)


def _myelinated_transform(population, transform_name):
    if transform_name == "rhoa":
        return population.in_graph_parametrizations["rhoa"][0][0]
    if transform_name == "diam":
        return population.parametrizations.diam[0]
    raise AssertionError(f"unknown transform {transform_name!r}")


def _register_module_hook(module, hook_kind):
    if hook_kind == "forward_pre":
        return module.register_forward_pre_hook(lambda _module, _inputs: None)
    if hook_kind == "forward":
        return module.register_forward_hook(
            lambda _module, _inputs, output: output,
        )
    if hook_kind == "backward":
        return module.register_full_backward_hook(
            lambda _module, _grad_input, _grad_output: None,
        )
    raise AssertionError(f"unknown hook kind {hook_kind!r}")


@pytest.mark.parametrize("transform_name", ["rhoa", "diam"])
@pytest.mark.parametrize("hook_kind", ["forward_pre", "forward", "backward"])
def test_myelinated_rejects_registered_transform_hooks(transform_name, hook_kind):
    population = _myelinated()
    transform = _myelinated_transform(population, transform_name)
    _register_module_hook(transform, hook_kind)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="standard registered diameter and axial-resistivity parametrizations",
    ):
        dn.func.make_functional(population, dt=DT)


@pytest.mark.parametrize("transform_name", ["rhoa", "diam"])
@pytest.mark.parametrize("hook_kind", ["forward_pre", "forward", "backward"])
def test_myelinated_transform_hook_changes_invalidate_existing_plan(
    transform_name,
    hook_kind,
):
    population = _myelinated()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    transform = _myelinated_transform(population, transform_name)
    handle = _register_module_hook(transform, hook_kind)

    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)

    handle.remove()
    functional.prepare(tensors.parameters, tensors.constants)


@pytest.mark.parametrize("mutation", ["hook", "arguments"])
def test_myelinated_parametrization_changes_invalidate_existing_plan(mutation):
    population = _myelinated()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    transform, args = population.in_graph_parametrizations["rhoa"][0]
    if mutation == "hook":
        transform.forward = lambda rhoa, _dx, _diameters, _diam: rhoa
    else:
        population.in_graph_parametrizations["rhoa"][0] = (
            transform,
            tuple(reversed(args)),
        )

    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        functional.prepare(tensors.parameters, tensors.constants)


@pytest.mark.parametrize("one_shot", [False, True])
def test_unsupported_stateful_parameter_transform_does_not_touch_source(one_shot):
    class StatefulTransform(torch.nn.Module):
        def __init__(self, one_shot):
            super().__init__()
            self.one_shot = one_shot
            self.register_buffer("calls", torch.zeros((), dtype=torch.int64))

        def forward(self, value):
            if not self.one_shot or self.calls.item() == 0:
                self.calls.add_(1)
            return value

    population = _unmyelinated()
    transform = StatefulTransform(one_shot)
    population.register_parametrization_in_graph("rhoa", transform)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="_derive_parameter_buffers.*mutated or rebound",
    ):
        dn.func.make_functional(population, dt=DT)
    assert transform.calls.item() == 0


def test_derived_buffer_mutation_is_rejected_without_touching_the_source():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(hh)
        population.initialize()

    state = population.mech.hh.DE["mhn"]
    expected_am1 = state.am1.detach().clone()

    def mutating_derive_buffers(self):
        self.am1.add_(1.0)
        return {"q10": torch.ones_like(self.am1)}

    state.derive_buffers = MethodType(mutating_derive_buffers, state)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"derive_buffers.*mutated or rebound.*am1",
    ):
        dn.func.make_functional(population, dt=DT)
    torch.testing.assert_close(state.am1, expected_am1, rtol=0.0, atol=0.0)


def test_transform_literal_configuration_invalidates_source_and_compatible_models():
    source, transform = _scaled_unmyelinated(2.0)
    compatible, _compatible_transform = _scaled_unmyelinated(3.0)
    functional, tensors = dn.func.make_functional(source, dt=DT)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure does not match",
    ):
        functional.extract(compatible)

    transform.factor = 3.0
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure changed",
    ):
        functional.prepare(tensors.parameters, tensors.constants)


def test_transform_class_literal_configuration_invalidates_existing_plan():
    class ClassScale(torch.nn.Module):
        factor = 2.0

        def forward(self, value):
            return type(self).factor * value

    population = _unmyelinated()
    population.register_parametrization_in_graph("cm", ClassScale())
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()
    functional, tensors = dn.func.make_functional(population, dt=DT)

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    torch.testing.assert_close(
        prepared.values["population"]["cm"],
        population.cm,
        rtol=0.0,
        atol=0.0,
    )

    ClassScale.factor = 3.0
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure changed",
    ):
        functional.prepare(tensors.parameters, tensors.constants)


def test_transform_class_tensor_dependency_must_be_registered():
    class HiddenClassTensor(torch.nn.Module):
        factor = torch.tensor(2.0)

        def forward(self, value):
            return self.factor * value

    population = _unmyelinated()
    population.register_parametrization_in_graph("rhoa", HiddenClassTensor())

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="unregistered Tensor or array",
    ):
        dn.func.make_functional(population, dt=DT)


def test_transform_class_helper_rebinding_invalidates_existing_plan():
    def times_two(value):
        return 2.0 * value

    def times_three(value):
        return 3.0 * value

    class ClassHelper(torch.nn.Module):
        operation = staticmethod(times_two)

        def forward(self, value):
            return self.operation(value)

    population = _unmyelinated()
    population.register_parametrization_in_graph("cm", ClassHelper())
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()
    functional, tensors = dn.func.make_functional(population, dt=DT)

    ClassHelper.operation = staticmethod(times_three)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure changed",
    ):
        functional.prepare(tensors.parameters, tensors.constants)


def test_transform_class_level_module_must_be_an_instance_child():
    class ScaleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("factor", torch.tensor(2.0))

        def forward(self, value):
            return self.factor * value

    class HiddenModule(torch.nn.Module):
        operation = ScaleModule()

        def forward(self, value):
            return self.operation(value)

    population = _unmyelinated()
    population.register_parametrization_in_graph("cm", HiddenModule())
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="unregistered class-level Module",
    ):
        dn.func.make_functional(population, dt=DT)


def test_transform_stateful_class_callable_is_rejected_without_execution():
    class StatefulCallable:
        def __init__(self):
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            return value

    operation = StatefulCallable()

    class HiddenCallable(torch.nn.Module):
        op = operation

        def forward(self, value):
            return self.op(value)

    population = _unmyelinated()
    population.register_parametrization_in_graph("cm", HiddenCallable())
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="callable object",
    ):
        dn.func.make_functional(population, dt=DT)
    assert operation.calls == 0


def test_transform_configuration_mixin_after_module_is_fingerprinted():
    class ConfigurationMixin:
        factor = 2.0

    class MixedTransform(torch.nn.Module, ConfigurationMixin):
        def forward(self, value):
            return type(self).factor * value

    population = _unmyelinated()
    population.register_parametrization_in_graph("cm", MixedTransform())
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()
    functional, tensors = dn.func.make_functional(population, dt=DT)

    ConfigurationMixin.factor = 3.0
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="structure changed",
    ):
        functional.prepare(tensors.parameters, tensors.constants)


def test_standard_torch_child_transform_is_admitted():
    class StandardChildren(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.operation = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.GELU(),
                torch.nn.SiLU(),
                torch.nn.Softplus(),
            )

        def forward(self, value):
            return self.operation(value)

    population = _unmyelinated()
    population.register_parametrization_in_graph("cm", StandardChildren())
    population.populate_parameter_buffers()
    population.integrator.initialized = False
    population.initialize()

    functional, tensors = dn.func.make_functional(population, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    assert torch.isfinite(prepared.values["population"]["cm"]).all()


def test_dunder_class_state_is_restored_after_rejected_lowering():
    class DunderMutation(torch.nn.Module):
        __calls__ = 0

        def forward(self, value):
            type(self).__calls__ += 1
            return value

    population = _unmyelinated()
    population.register_parametrization_in_graph("cm", DunderMutation())
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="class state mutation",
    ):
        dn.func.make_functional(population, dt=DT)
    assert DunderMutation.__calls__ == 0


def test_class_snapshot_rejects_custom_deepcopy_without_executing_it():
    class DeepcopyTrap:
        def __deepcopy__(self, memo):
            SnapshotOwner.calls += 1
            raise RuntimeError("must not execute")

    class SnapshotOwner(torch.nn.Module):
        calls = 0
        configuration = DeepcopyTrap()

        def forward(self, value):
            return value

    population = _unmyelinated()
    population.register_parametrization_in_graph("cm", SnapshotOwner())
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="cannot safely audit mutable class state",
    ):
        dn.func.make_functional(population, dt=DT)
    assert SnapshotOwner.calls == 0


@pytest.mark.parametrize("hidden_kind", ["callable", "module"])
def test_mechanism_class_hidden_executable_state_is_rejected_without_leaking(
    hidden_kind,
):
    class StatefulCallable:
        def __init__(self):
            self.calls = 0

        def __call__(self, value):
            self.calls += 1
            return value

    class CounterModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("calls", torch.zeros((), dtype=torch.int64))

        def forward(self, value):
            self.calls.add_(1)
            return value

    operation = StatefulCallable() if hidden_kind == "callable" else CounterModule()

    class BadPas(pas):
        op = operation

        def i(self, v):
            self.op(v)
            return self.g * (v - self.e)

        def i_with_conductance(self, v):
            self.op(v)
            return self.g * (v - self.e), self.g.expand_as(v)

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(BadPas)
        population.initialize()

    expected_calls = (
        operation.calls
        if hidden_kind == "callable"
        else operation.calls.detach().clone()
    )
    expected_message = (
        "class-level callable object"
        if hidden_kind == "callable"
        else "unregistered class-level Module"
    )
    with pytest.raises(dn.func.FunctionalizationError, match=expected_message):
        dn.func.make_functional(population, dt=DT)
    if hidden_kind == "callable":
        assert operation.calls == expected_calls
    else:
        torch.testing.assert_close(operation.calls, expected_calls, rtol=0.0, atol=0.0)


def test_class_snapshot_rejects_custom_descriptor_without_repr_execution():
    class SideEffectDescriptor:
        __slots__ = ()

        def __get__(self, instance, owner=None):
            return self

        def __set__(self, instance, value):
            raise AttributeError

        def __repr__(self):
            DescriptorPas.calls += 1
            raise RuntimeError("must not execute")

    class DescriptorPas(pas):
        calls = 0
        hidden = SideEffectDescriptor()

    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        population = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        population.insert(DescriptorPas)
        population.initialize()

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="cannot safely audit mutable class descriptor",
    ):
        dn.func.make_functional(population, dt=DT)
    assert DescriptorPas.calls == 0


@pytest.mark.parametrize(
    "configuration",
    [[2.0], np.asarray(2.0)],
    ids=["mutable-list", "unregistered-array"],
)
def test_transform_hidden_mutable_or_numerical_configuration_is_rejected(
    configuration,
):
    class HiddenConfiguration(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.factor = value

        def forward(self, value):
            return value

    population = _unmyelinated()
    population.register_parametrization_in_graph(
        "rhoa",
        HiddenConfiguration(configuration),
    )
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="transform state.*(mutable Python|unregistered Tensor or array)",
    ):
        dn.func.make_functional(population, dt=DT)


def test_failed_eligibility_restores_transform_class_and_all_global_rng_state():
    class ImpureTransform(torch.nn.Module):
        calls = 7

        def forward(self, value):
            type(self).calls += 1
            torch.rand(())
            random.random()
            np.random.random()
            return value

    population = _unmyelinated()
    population.register_parametrization_in_graph("rhoa", ImpureTransform())
    expected_torch_rng = torch.random.get_rng_state().clone()
    expected_python_rng = random.getstate()
    expected_numpy_rng = copy.deepcopy(np.random.get_state())

    with pytest.raises(
        dn.func.FunctionalizationError,
        match="class state|RNG state",
    ):
        dn.func.make_functional(population, dt=DT)

    assert ImpureTransform.calls == 7
    assert torch.equal(torch.random.get_rng_state(), expected_torch_rng)
    assert random.getstate() == expected_python_rng
    actual_numpy_rng = np.random.get_state()
    assert actual_numpy_rng[0] == expected_numpy_rng[0]
    assert np.array_equal(actual_numpy_rng[1], expected_numpy_rng[1])
    assert actual_numpy_rng[2:] == expected_numpy_rng[2:]
