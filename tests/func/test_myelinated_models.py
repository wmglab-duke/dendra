"""Functional lowering contracts for real Myelinated/Axon models."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context

peripheral = pytest.importorskip("dendra_models.models.cells.peripheral")

DT = 0.01
MODEL_NAMES = ("Sweeney1987", "FHM")

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


def _model(name, *, batch=None, dtype=torch.float64, method="pcr"):
    constructor = getattr(peripheral, name)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        kwargs = {}
        if method is not None:
            kwargs["integrator"] = dn.bwd_euler_ub(method=method, imem=False)
        model = constructor(
            diameters=[8.0, 12.0],
            n_node=5,
            **kwargs,
        ).to(dtype=dtype)
        if batch is not None:
            model.batch(batch)
        model.initialize()
        model.train()
    return model


def _drives(model, steps):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -1.0,
        1.0,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -1.0e-9,
        1.0e-9,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _assert_tree_equal(actual, expected):
    actual_paths, actual_spec = torch.utils._pytree.tree_flatten_with_path(actual)
    expected_paths, expected_spec = torch.utils._pytree.tree_flatten_with_path(expected)
    assert actual_spec == expected_spec
    for (actual_path, actual_value), (expected_path, expected_value) in zip(
        actual_paths,
        expected_paths,
        strict=True,
    ):
        assert actual_path == expected_path
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=0.0,
            atol=0.0,
            msg=lambda message: f"state leaf {actual_path}: {message}",
        )


def _assert_complete_state_schema(name, state):
    assert set(state) == {"integrator", "mechanisms", "ions", "clock", "control"}
    assert set(state["integrator"]) == {"v"}
    assert set(state["clock"]) == {"t"}
    assert set(state["control"]) == {"duration_remainder"}
    if name == "Sweeney1987":
        assert set(state["mechanisms"]) == {"sweeney"}
        assert set(state["mechanisms"]["sweeney"]) == {"m", "h"}
        assert set(state["ions"]) == {"na"}
        assert set(state["ions"]["na"]) == {"ina", "ena", "nai", "nao"}
    else:
        assert set(state["mechanisms"]) == {"fh"}
        assert set(state["mechanisms"]["fh"]) == {"m", "h", "n", "p"}
        assert set(state["ions"]) == {"na", "k"}
        assert set(state["ions"]["na"]) == {"ina", "ena", "nai", "nao"}
        assert set(state["ions"]["k"]) == {"ik", "ek", "ki", "ko"}


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("batch", [None, 3])
def test_real_myelinated_models_match_every_imperative_leaf(name, batch):
    functional_model = _model(name, batch=batch)
    imperative_model = _model(name, batch=batch)
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    _assert_complete_state_schema(name, tensors.state)
    if batch is not None:
        assert tensors.state["integrator"]["v"].shape == (batch, 2, 5)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state
    ve, intra = _drives(functional_model, 3)
    dt = torch.as_tensor(DT, dtype=imperative_model.dtype())
    imperative_model.integrator._initialize(
        imperative_model,
        dt,
        force=True,
        compile_scope="population",
    )

    for index in range(3):
        state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        imperative_model.integrator.step(
            imperative_model,
            dt,
            ve[index],
            intra[index],
        )
        imperative_model.t = imperative_model.t + dt
        expected = functional.extract(imperative_model).state
        _assert_tree_equal(state, expected)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("batch", [None, 3])
def test_myelinated_preparation_exactly_matches_imperative_workspace(name, batch):
    functional_model = _model(name, batch=batch)
    imperative_model = _model(name, batch=batch)
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    dt = torch.as_tensor(DT, dtype=imperative_model.dtype())
    imperative_model.integrator._initialize(
        imperative_model,
        dt,
        force=True,
        compile_scope="population",
    )

    torch.testing.assert_close(
        prepared.values["population"]["diam"],
        imperative_model.diam,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        prepared.values["population"]["rhoa"],
        imperative_model.rhoa,
        rtol=0.0,
        atol=0.0,
    )
    if batch is not None:
        assert prepared.values["population"]["diam"].shape == (2, 5)
        assert prepared.values["population"]["rhoa"].shape == (1, 2, 5)
        assert prepared.values["integrator"]["diag_base"].shape == (batch * 2, 5)
    for (
        workspace_name,
        _role,
    ) in imperative_model.integrator._prepared_workspace_schema():
        torch.testing.assert_close(
            prepared.values["integrator"][workspace_name],
            getattr(imperative_model.integrator, workspace_name),
            rtol=0.0,
            atol=0.0,
        )


def test_sweeney_geometry_parameters_support_reverse_forward_and_compiled_jacobians():
    model = _model("Sweeney1987")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = (value[0] for value in _drives(model, 1))
    names = ("noded2", "axond2", "deltax2")
    sources = tuple(tensors.parameters[name].detach().clone() for name in names)

    def final_voltage(noded2, axond2, deltax2):
        parameters = {
            **tensors.parameters,
            "noded2": noded2,
            "axond2": axond2,
            "deltax2": deltax2,
        }
        state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(ve=ve, intra=intra),
        )
        return state["integrator"]["v"]

    reverse = torch.func.jacrev(final_voltage, argnums=(0, 1, 2))
    forward = torch.func.jacfwd(final_voltage, argnums=(0, 1, 2))
    expected = reverse(*sources)
    forward_actual = forward(*sources)
    for name, reverse_value, forward_value in zip(
        names,
        expected,
        forward_actual,
        strict=True,
    ):
        assert torch.isfinite(reverse_value).all(), name
        assert torch.count_nonzero(reverse_value), name
        torch.testing.assert_close(
            forward_value,
            reverse_value,
            rtol=2.0e-10,
            atol=2.0e-11,
        )

    with torch_compiler_warning_context():
        compiled = torch.compile(reverse, backend="aot_eager", fullgraph=True)
        actual = compiled(*sources)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            rtol=2.0e-10,
            atol=2.0e-11,
        )


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_myelinated_geometry_sources_vmap_matches_explicit_lanes(name):
    model = _model(name)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = (value[0] for value in _drives(model, 1))
    parameter_names = ("noded2", "axond2", "deltax2")
    sources = tuple(tensors.parameters[name] for name in parameter_names) + (
        tensors.constants["diameters"],
        tensors.constants["diam_original"],
    )
    scales = (
        (0.95, 1.05),
        (0.9, 1.1),
        (0.85, 1.15),
        (0.92, 1.08),
        (0.97, 1.03),
    )
    lanes = tuple(
        torch.stack((low * source, high * source))
        for source, (low, high) in zip(sources, scales, strict=True)
    )

    def advance(noded2, axond2, deltax2, diameters, diam_original):
        parameters = {
            **tensors.parameters,
            "noded2": noded2,
            "axond2": axond2,
            "deltax2": deltax2,
        }
        constants = {
            **tensors.constants,
            "diameters": diameters,
            "diam_original": diam_original,
        }
        return functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
            dn.func.StepInput(ve=ve, intra=intra),
        )[0]

    actual = torch.vmap(advance)(*lanes)
    expected = torch.utils._pytree.tree_map(
        lambda *values: torch.stack(values),
        *(advance(*(values[index] for values in lanes)) for index in range(2)),
    )
    _assert_tree_equal(actual, expected)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_myelinated_geometry_sources_vmap_accept_zero_lanes(name):
    model = _model(name)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = (value[0] for value in _drives(model, 1))
    noded2 = tensors.parameters["noded2"]
    axond2 = tensors.parameters["axond2"]
    deltax2 = tensors.parameters["deltax2"]
    diameters = tensors.constants["diameters"]
    diam_original = tensors.constants["diam_original"]

    def advance(
        local_noded2,
        local_axond2,
        local_deltax2,
        local_diameters,
        local_diam_original,
    ):
        parameters = {
            **tensors.parameters,
            "noded2": local_noded2,
            "axond2": local_axond2,
            "deltax2": local_deltax2,
        }
        constants = {
            **tensors.constants,
            "diameters": local_diameters,
            "diam_original": local_diam_original,
        }
        return functional.prepare_and_step(
            parameters,
            constants,
            tensors.state,
            dn.func.StepInput(ve=ve, intra=intra),
        )[0]

    actual = torch.vmap(advance)(
        noded2.new_empty((0, *noded2.shape)),
        axond2.new_empty((0, *axond2.shape)),
        deltax2.new_empty((0, *deltax2.shape)),
        diameters.new_empty((0, *diameters.shape)),
        diam_original.new_empty((0, *diam_original.shape)),
    )
    for actual_leaf, state_leaf in zip(
        torch.utils._pytree.tree_leaves(actual),
        torch.utils._pytree.tree_leaves(tensors.state),
        strict=True,
    ):
        assert actual_leaf.shape == (0, *state_leaf.shape)


def test_sweeney_float32_matches_every_imperative_leaf():
    functional_model = _model("Sweeney1987", dtype=torch.float32)
    imperative_model = _model("Sweeney1987", dtype=torch.float32)
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state
    ve, intra = _drives(functional_model, 2)
    dt = torch.as_tensor(DT, dtype=torch.float32)
    imperative_model.integrator._initialize(
        imperative_model,
        dt,
        force=True,
        compile_scope="population",
    )

    for index in range(2):
        state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        imperative_model.integrator.step(
            imperative_model,
            dt,
            ve[index],
            intra[index],
        )
        imperative_model.t = imperative_model.t + dt
        _assert_tree_equal(state, functional.extract(imperative_model).state)


def test_sweeney_default_native_thomas_path_when_available():
    try:
        from dendra_solvers import thomas_solve_t
    except ImportError:
        pytest.skip("transform-compatible dendra-solvers facade is unavailable")

    functional_model = _model("Sweeney1987", method=None)
    imperative_model = _model("Sweeney1987", method=None)
    assert functional_model.integrator.method == "thomas"
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    assert functional._transition.solver is thomas_solve_t
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = (value[0] for value in _drives(functional_model, 1))
    actual, _aux = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.StepInput(ve=ve, intra=intra),
    )
    dt = torch.as_tensor(DT, dtype=imperative_model.dtype())
    imperative_model.integrator._initialize(
        imperative_model,
        dt,
        force=True,
        compile_scope="population",
    )
    imperative_model.integrator.step(imperative_model, dt, ve, intra)
    imperative_model.t = imperative_model.t + dt
    _assert_tree_equal(actual, functional.extract(imperative_model).state)


def test_functionalization_preserves_myelinated_imperative_hot_path_and_workspace():
    source = _model("Sweeney1987")
    reference = _model("Sweeney1987")
    source.eval()
    reference.eval()
    source.run(tstop=0.0, dt=DT)
    reference.run(tstop=0.0, dt=DT)

    workspace = {
        name: (
            id(getattr(source.integrator, name)),
            getattr(source.integrator, name).untyped_storage().data_ptr(),
            getattr(source.integrator, name)._version,
            getattr(source.integrator, name).detach().clone(),
        )
        for name, _role in source.integrator._prepared_workspace_schema()
    }
    functional, tensors = dn.func.make_functional(source, dt=DT)
    functional.prepare(tensors.parameters, tensors.constants)
    ve, _intra = _drives(source, 3)
    source.run(ve=ve, dt=DT)
    reference.run(ve=ve, dt=DT)

    _assert_tree_equal(
        functional.extract(source).state, functional.extract(reference).state
    )
    for name, (identity, storage, version, expected) in workspace.items():
        actual = getattr(source.integrator, name)
        assert id(actual) == identity
        assert actual.untyped_storage().data_ptr() == storage
        assert actual._version == version
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
