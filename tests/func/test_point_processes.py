"""Functional contracts for lumped PointProcess currents."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import PointProcess
from dendra.models.mod import alphasynapse_d, pas

DT = 0.01
DTYPE = torch.float64
LAYOUTS = ("dense", "rectangular", "shared_columns", "packed", "duplicates")

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _PointAffine(PointProcess):
    PointProcess.RANGE(g=3.0e-7, e=-10.0)
    PointProcess.NONSPECIFIC_CURRENT("i")
    PointProcess.AFFINE("i")

    def i(self, v):
        return self.g * (v - self.e)

    def i_with_conductance(self, v):
        return self.i(v), self.g.expand_as(v)


def _diameters(scale=1.0):
    return scale * torch.tensor(
        [
            [1.00, 1.10, 1.20, 1.30, 1.40],
            [1.45, 1.35, 1.25, 1.15, 1.05],
        ],
        dtype=DTYPE,
    )


def _target(model, layout):
    if layout == "dense":
        return model, False
    if layout == "rectangular":
        return model[:, 1:4], False
    if layout == "shared_columns":
        return model[:, torch.tensor([1, 3])], False
    if layout == "packed":
        return (
            model[torch.tensor([0, 0, 1]), torch.tensor([0, 4, 2])],
            False,
        )
    if layout == "duplicates":
        return (
            model[
                torch.tensor([0, 0, 1, 1]),
                torch.tensor([1, 1, 3, 3]),
            ],
            True,
        )
    raise AssertionError(f"unknown PointProcess layout {layout!r}")


def _model(layout="dense", *, diameters=None, batch_calls=()):
    diameters = _diameters() if diameters is None else diameters
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            diameters,
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=1.0e-8, e=-65.0)
        target, preserve_multiplicity = _target(model, layout)
        target.insert(
            _PointAffine,
            preserve_multiplicity=preserve_multiplicity,
            g=3.0e-7,
            e=-10.0,
        )
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _alpha_model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.2],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=1.0e-8, e=-65.0)
        model.insert(
            alphasynapse_d,
            onset=0.015,
            tau=0.08,
            gmax=3.0e-7,
            e=0.0,
            smooth_eps=1.0e-3,
        )
        model.initialize()
        model.train()
    return model


def _imperative_steps(model, steps):
    dt = torch.as_tensor(DT, dtype=model.dtype(), device=model.device())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    for _ in range(steps):
        model.integrator.step(model, dt)
        model.t = model.t + dt


def _assert_tree_close(actual, expected, *, rtol=2.0e-10, atol=2.0e-11):
    actual_values, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_values, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_value, expected_value in zip(
        actual_values,
        expected_values,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=rtol,
            atol=atol,
        )


def _tensor_snapshot(value):
    return (
        id(value),
        value.untyped_storage().data_ptr(),
        value._version,
        value.detach().clone(),
    )


def _assert_tensor_unchanged(value, snapshot):
    identity, storage, version, expected = snapshot
    assert id(value) == identity
    assert value.untyped_storage().data_ptr() == storage
    assert value._version == version
    torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)


def _point_g_name(parameters):
    candidates = [
        name for name in parameters if ".mechanisms._PointAffine.g_param" in name
    ]
    regional = [name for name in candidates if name.endswith("g_param_0")]
    if regional:
        return regional[0]
    assert len(candidates) == 1
    return candidates[0]


def _prepared_point_area_factor(prepared):
    factors = [
        value
        for name, value in prepared.values["mechanisms"].items()
        if name.endswith("._point_area_factor")
    ]
    assert len(factors) == 1
    return factors[0]


def _cylindrical_area_cm2(model):
    return model.diam * 1.0e-4 * torch.pi * model.dx * 1.0e-4


def _assert_exact_factor_and_one_step_parity(source, reference, physical_area):
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    factor = _prepared_point_area_factor(prepared)
    expected_factor = 1.0e6 * physical_area

    assert torch.all(source.area_scale != 1.0)
    torch.testing.assert_close(factor, expected_factor, rtol=0.0, atol=0.0)
    assert not torch.equal(factor, expected_factor * source.area_scale)

    actual, _aux = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    _imperative_steps(reference, 1)
    _assert_tree_close(actual, functional.extract(reference).state)


def _single_compartment_point_model():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=2,
            C=3,
            v_init=-65.0,
            area_scale=2.5,
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.diam.copy_(
            torch.tensor(
                [[250.0, 400.0, 550.0], [300.0, 450.0, 600.0]],
                dtype=DTYPE,
            )
        )
        model.dx.copy_(
            torch.tensor(
                [[50.0, 75.0, 100.0], [60.0, 85.0, 110.0]],
                dtype=DTYPE,
            )
        )
        model.insert(_PointAffine)
        model.initialize()
        model.train()
    return model


def _native_cable_point_model():
    morphology = dn.Morphology(rhoa=91.0, cm=1.0)
    morphology.section(
        "tapered",
        points=[
            (0.0, 0.0, 0.0, 7.0),
            (11.0, 4.0, 2.0, 3.5),
            (29.0, 13.0, -1.0, 1.2),
        ],
        nseg=3,
    )
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Cable.from_morphology(
            morphology,
            N=2,
            v_init=-65.0,
            area_scale=3.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(_PointAffine)
        model.initialize()
        model.train()
    return model


def _real_myelinated_point_model():
    peripheral = pytest.importorskip("dendra_models.models.cells.peripheral")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = peripheral.Sweeney1987(
            [8.0, 12.0],
            n_node=3,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        ).double()
        model.area_scale_param.set(4.0)
        model.insert(_PointAffine)
        model.initialize()
        model.train()
    return model


@pytest.mark.parametrize("layout", LAYOUTS)
def test_point_process_every_leaf_parity_across_support_layouts(layout):
    source = _model(layout)
    reference = _model(layout)
    functional, tensors = dn.func.make_functional(source, dt=DT)

    actual, _aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=3,
    )
    _imperative_steps(reference, 3)

    _assert_tree_close(actual, functional.extract(reference).state)


@pytest.mark.parametrize(
    ("batch_calls", "batch_shape"),
    [((), ()), ((3, 2), (2, 3))],
)
def test_point_process_population_batch_keeps_factor_shared_and_matches_imperative(
    batch_calls,
    batch_shape,
):
    source = _model("shared_columns", batch_calls=batch_calls)
    reference = _model("shared_columns", batch_calls=batch_calls)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    factor = _prepared_point_area_factor(prepared)
    expected_factor = 1.0e6 * _cylindrical_area_cm2(source)[
        :, torch.tensor([1, 3])
    ].reshape(-1)
    assert source.shape == (*batch_shape, *source.core_shape())
    assert factor.shape == expected_factor.shape == (4,)
    torch.testing.assert_close(factor, expected_factor, rtol=0.0, atol=0.0)

    actual, _aux = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    _imperative_steps(reference, 1)
    _assert_tree_close(actual, functional.extract(reference).state)


def test_point_process_area_uses_compatible_populations_live_geometry():
    source = _model("shared_columns", diameters=_diameters(1.0))
    target = _model("shared_columns", diameters=_diameters(1.7))
    reference = _model("shared_columns", diameters=_diameters(1.7))
    functional, _source_tensors = dn.func.make_functional(source, dt=DT)
    tensors = functional.extract(target)

    actual, _aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=3,
    )
    _imperative_steps(reference, 3)

    _assert_tree_close(actual, functional.extract(reference).state)


def test_point_process_parameter_and_geometry_jacobians_agree():
    model = _model("shared_columns")
    source_area_factor = model.mech._PointAffine._point_area_factor
    source_area_factor_snapshot = _tensor_snapshot(source_area_factor)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    g_name = _point_g_name(tensors.parameters)
    g = tensors.parameters[g_name]
    diam = tensors.constants["diam"]

    def response(local_g, local_diam):
        parameters = {**tensors.parameters, g_name: local_g}
        constants = {**tensors.constants, "diam": local_diam}
        return functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )[0]["integrator"]["v"]

    reverse = torch.func.jacrev(response, argnums=(0, 1))(g, diam)
    forward = torch.func.jacfwd(response, argnums=(0, 1))(g, diam)

    for reverse_value, forward_value in zip(reverse, forward, strict=True):
        torch.testing.assert_close(
            reverse_value,
            forward_value,
            rtol=2.0e-10,
            atol=2.0e-11,
        )
        assert torch.isfinite(reverse_value).all()
        assert torch.count_nonzero(reverse_value) > 0

    _assert_tensor_unchanged(source_area_factor, source_area_factor_snapshot)


def test_point_process_parameter_and_geometry_vmap_including_zero_lanes():
    model = _model("packed")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    g_name = _point_g_name(tensors.parameters)
    g = tensors.parameters[g_name]
    diam = tensors.constants["diam"]

    def response(local_g, local_diam):
        parameters = {**tensors.parameters, g_name: local_g}
        constants = {**tensors.constants, "diam": local_diam}
        return functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )[0]["integrator"]["v"]

    g_lanes = g + g.new_tensor([-0.4e-7, 0.0, 0.6e-7])
    diameter_scale = diam.new_tensor([0.85, 1.0, 1.2]).reshape(3, 1, 1)
    diam_lanes = diam.unsqueeze(0) * diameter_scale
    expected = torch.stack(
        [
            response(local_g, local_diam)
            for local_g, local_diam in zip(g_lanes, diam_lanes, strict=True)
        ]
    )

    torch.testing.assert_close(
        torch.vmap(response)(g_lanes, diam_lanes),
        expected,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    empty = torch.vmap(response)(
        g.new_empty((0, *g.shape)),
        diam.new_empty((0, *diam.shape)),
    )
    assert empty.shape == (0, *model.shape)


def test_point_process_compile_jacrev_composes_with_live_geometry():
    model = _model("duplicates")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    g_name = _point_g_name(tensors.parameters)
    g = tensors.parameters[g_name]
    diam = tensors.constants["diam"]

    def response(local_g, local_diam):
        parameters = {**tensors.parameters, g_name: local_g}
        constants = {**tensors.constants, "diam": local_diam}
        return functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
        )[0]["integrator"]["v"]

    transformed = torch.func.jacrev(response, argnums=(0, 1))
    expected = transformed(g, diam)
    with torch_compiler_warning_context():
        compiled = torch.compile(transformed, backend="eager", fullgraph=True)
        actual = compiled(g, diam)

    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-10,
            atol=2.0e-11,
        )


def test_alphasynapse_d_reads_the_explicit_functional_clock_each_step():
    source = _alpha_model()
    reference = _alpha_model()
    initial_voltage = source.v.clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)

    actual, _aux = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=8,
    )
    _imperative_steps(reference, 8)

    _assert_tree_close(actual, functional.extract(reference).state)
    assert not torch.equal(actual["integrator"]["v"], initial_voltage)


def test_single_compartment_point_process_uses_exact_unscaled_physical_area():
    source = _single_compartment_point_model()
    reference = _single_compartment_point_model()

    _assert_exact_factor_and_one_step_parity(
        source,
        reference,
        _cylindrical_area_cm2(source),
    )


def test_native_cable_point_process_uses_exact_unscaled_canonical_area():
    source = _native_cable_point_model()
    reference = _native_cable_point_model()

    _assert_exact_factor_and_one_step_parity(
        source,
        reference,
        source._canonical_area_cm2,
    )


def test_real_myelinated_point_process_uses_exact_unscaled_physical_area():
    source = _real_myelinated_point_model()
    reference = _real_myelinated_point_model()

    _assert_exact_factor_and_one_step_parity(
        source,
        reference,
        _cylindrical_area_cm2(source),
    )
