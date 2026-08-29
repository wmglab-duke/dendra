"""Pure functional-initialization contracts for the first HH capability slice."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import hh, pas

DT = 0.01
AM1 = "integrator.mech.mechanisms.hh.DE.mhn.am1_param"
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


class _RandomCurrent(Mechanism):
    """Unsupported random initializer used to exercise the admission boundary."""

    Mechanism.GLOBALRAND(
        "sample",
        distribution="normal",
        mu=0.0,
        sigma=1.0,
        seed=17,
    )
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.sample * torch.zeros_like(v)

    def i_with_conductance(self, v):
        zero = torch.zeros_like(v)
        return zero, zero


def _model(kind, *, batch_calls=(), mechanism=hh, dtype=torch.float64):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        if kind == "sc":
            model = dn.SingleCompartment(
                N=2,
                C=3,
                v_init=torch.tensor([-64.0, -59.0, -62.0], dtype=dtype),
                dtype=dtype,
                integrator=dn.bwd_euler_sc(imem=False),
            )
        elif kind == "ub":
            model = dn.Unmyelinated(
                [2.0, 2.5],
                L=4.0,
                dx=1.0,
                v_init=torch.tensor(
                    [-64.0, -60.0, -56.0, -59.0, -63.0],
                    dtype=dtype,
                ),
                dtype=dtype,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            )
        else:  # pragma: no cover - private test helper
            raise AssertionError(f"unknown model kind {kind!r}")
        model.insert(mechanism)
        for size in batch_calls:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _new_initial_voltage(model):
    return torch.linspace(
        -71.0,
        -53.0,
        model.v.numel(),
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(model.shape)


def _assert_tree_close(actual, expected, *, rtol=0.0, atol=0.0):
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


def _tensor_snapshot(tensor):
    return (
        id(tensor),
        tensor.untyped_storage().data_ptr(),
        tensor._version,
        tensor.detach().clone(),
    )


def _assert_tensor_snapshot(tensor, snapshot):
    identity, storage, version, value = snapshot
    assert id(tensor) == identity
    assert tensor.untyped_storage().data_ptr() == storage
    assert tensor._version == version
    torch.testing.assert_close(tensor, value, rtol=0.0, atol=0.0)


def _module_snapshot(module):
    return {
        "parameters": {
            name: _tensor_snapshot(value)
            for name, value in module.named_parameters(remove_duplicate=False)
        },
        "buffers": {
            name: _tensor_snapshot(value)
            for name, value in module.named_buffers(remove_duplicate=False)
        },
        "python": {
            "v_init": (
                _tensor_snapshot(module.v_init)
                if torch.is_tensor(module.v_init)
                else module.v_init
            ),
            "initialized": module.initialized,
            "integrator_initialized": module.integrator.initialized,
            "integrator_dt": module.integrator.dt,
            "integrator_shape": module.integrator.shape,
            "compiled_kernels": dict(module.integrator._compiled_kernels),
            "caches": copy.deepcopy(module._caches),
        },
    }


def _assert_module_snapshot(module, snapshot):
    for name, value in module.named_parameters(remove_duplicate=False):
        _assert_tensor_snapshot(value, snapshot["parameters"][name])
    for name, value in module.named_buffers(remove_duplicate=False):
        _assert_tensor_snapshot(value, snapshot["buffers"][name])
    if torch.is_tensor(module.v_init):
        _assert_tensor_snapshot(module.v_init, snapshot["python"]["v_init"])
    else:
        assert module.v_init == snapshot["python"]["v_init"]
    assert module.initialized == snapshot["python"]["initialized"]
    assert module.integrator.initialized == snapshot["python"]["integrator_initialized"]
    assert module.integrator.dt == snapshot["python"]["integrator_dt"]
    assert module.integrator.shape == snapshot["python"]["integrator_shape"]
    assert module.integrator._compiled_kernels == snapshot["python"]["compiled_kernels"]
    assert module._caches.keys() == snapshot["python"]["caches"].keys()
    for name in module._caches:
        _assert_tree_close(module._caches[name], snapshot["python"]["caches"][name])


def _imperative_initialized_state(functional, kind, v_init, *, batch_calls=()):
    reference = _model(kind, batch_calls=batch_calls, dtype=v_init.dtype)
    reference.set_v_init(v_init)
    reference.initialize()
    return functional.extract(reference).state


def _imperative_step(model, ve=None, intra=None):
    dt = torch.as_tensor(DT, device=model.device(), dtype=model.dtype())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    model.integrator.step(model, dt, ve, intra)
    model.t = model.t + dt


@pytest.mark.parametrize("kind", ["sc", "ub"])
@pytest.mark.parametrize("batch_calls", [(), (2,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initialize_matches_every_imperative_state_leaf(kind, batch_calls, dtype):
    model = _model(kind, batch_calls=batch_calls, dtype=dtype)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    v_init = _new_initial_voltage(model)
    inputs = dn.func.InitializationInput(v_init=v_init)

    actual = functional.initialize(tensors.parameters, tensors.constants, inputs)
    expected = _imperative_initialized_state(
        functional,
        kind,
        v_init,
        batch_calls=batch_calls,
    )

    _assert_tree_close(actual.state, expected)
    assert tuple(actual.parameters) == tuple(tensors.parameters)
    assert tuple(actual.constants) == tuple(tensors.constants)
    for name in tensors.parameters:
        assert actual.parameters[name] is tensors.parameters[name]
    for name in tensors.constants:
        assert actual.constants[name] is tensors.constants[name]
    assert actual.initialization.v_init is v_init
    assert actual.initialization.states == {}
    assert actual.state["integrator"]["v"] is not v_init
    assert (
        actual.state["integrator"]["v"].untyped_storage().data_ptr()
        != v_init.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(
        actual.state["clock"]["t"],
        torch.zeros_like(tensors.state["clock"]["t"]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual.state["control"]["duration_remainder"],
        torch.zeros_like(tensors.state["control"]["duration_remainder"]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("kind", ["sc", "ub"])
def test_extraction_exposes_an_independent_expanded_initialization_input(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    assert tensors.initialization.v_init.shape == functional.shape
    torch.testing.assert_close(
        tensors.initialization.v_init,
        model.expanded_v_init(),
        rtol=0.0,
        atol=0.0,
    )
    assert (
        tensors.initialization.v_init.untyped_storage().data_ptr()
        != model.v.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("kind", ["sc", "ub"])
def test_initialize_does_not_mutate_source_or_any_explicit_input(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    v_init = _new_initial_voltage(model).requires_grad_()
    inputs = dn.func.InitializationInput(v_init=v_init)
    source_snapshot = _module_snapshot(model)
    parameter_snapshots = {
        name: _tensor_snapshot(value) for name, value in tensors.parameters.items()
    }
    constant_snapshots = {
        name: _tensor_snapshot(value) for name, value in tensors.constants.items()
    }
    input_snapshot = _tensor_snapshot(v_init)
    rng_snapshot = torch.random.get_rng_state().clone()

    initialized = functional.initialize(tensors.parameters, tensors.constants, inputs)
    repeated = functional.initialize(tensors.parameters, tensors.constants, inputs)
    _assert_tree_close(repeated.state, initialized.state)
    gradient = torch.autograd.grad(
        initialized.state["integrator"]["v"].square().sum(),
        v_init,
    )[0]
    assert torch.count_nonzero(gradient) > 0

    _assert_module_snapshot(model, source_snapshot)
    for name, value in tensors.parameters.items():
        _assert_tensor_snapshot(value, parameter_snapshots[name])
    for name, value in tensors.constants.items():
        _assert_tensor_snapshot(value, constant_snapshots[name])
    _assert_tensor_snapshot(v_init, input_snapshot)
    assert torch.equal(torch.random.get_rng_state(), rng_snapshot)


@pytest.mark.parametrize("kind", ["sc", "ub"])
def test_initialize_grad_jacobians_and_hessian_are_composable(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    base_v = _new_initial_voltage(model)
    base_am1 = tensors.parameters[AM1]

    def initialized_m(v_init, am1):
        parameters = dict(tensors.parameters)
        parameters[AM1] = am1
        initialized = functional.initialize(
            parameters,
            tensors.constants,
            dn.func.InitializationInput(v_init=v_init),
        )
        return initialized.state["mechanisms"]["hh"]["m"]

    reverse = torch.func.jacrev(initialized_m, argnums=(0, 1))(base_v, base_am1)
    forward = torch.func.jacfwd(initialized_m, argnums=(0, 1))(base_v, base_am1)
    _assert_tree_close(reverse, forward, rtol=2.0e-10, atol=2.0e-11)
    for derivative in reverse:
        assert torch.isfinite(derivative).all()
        assert torch.count_nonzero(derivative) > 0

    def loss(v_init):
        initialized = initialized_m(v_init, base_am1)
        return initialized.square().mean()

    gradient, value = torch.func.grad_and_value(loss)(base_v)
    hessian = torch.func.hessian(loss)(base_v)
    assert value.ndim == 0
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert torch.isfinite(hessian).all()
    assert torch.count_nonzero(hessian) > 0
    hessian_matrix = hessian.reshape(base_v.numel(), base_v.numel())
    torch.testing.assert_close(
        hessian_matrix,
        hessian_matrix.mT,
        rtol=2.0e-10,
        atol=2.0e-11,
    )


@pytest.mark.parametrize("kind", ["sc", "ub"])
def test_initialize_vmap_matches_explicit_lanes_and_accepts_zero_lanes(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    base_v = _new_initial_voltage(model)
    base_am1 = tensors.parameters[AM1]

    def lane(v_init, am1):
        parameters = dict(tensors.parameters)
        parameters[AM1] = am1
        initialized = functional.initialize(
            parameters,
            tensors.constants,
            dn.func.InitializationInput(v_init=v_init),
        )
        return initialized.state["mechanisms"]["hh"]["m"]

    offsets = base_v.new_tensor([-1.0, 0.0, 1.0])
    scales = base_v.new_tensor([0.8, 1.0, 1.2])
    voltage_lanes = base_v.unsqueeze(0) + offsets.reshape(3, *([1] * base_v.ndim))
    parameter_lanes = base_am1 * scales
    actual = torch.vmap(lane)(voltage_lanes, parameter_lanes)
    expected = torch.stack(
        [lane(voltage_lanes[index], parameter_lanes[index]) for index in range(3)]
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(actual[0], actual[2])
    empty = torch.vmap(lane)(voltage_lanes[:0], parameter_lanes[:0])
    assert empty.shape == (0, *model.shape)


@pytest.mark.parametrize("kind", ["sc", "ub"])
def test_transform_then_compile_initialization_matches_eager(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    base_v = _new_initial_voltage(model)

    def loss(v_init):
        initialized = functional.initialize(
            tensors.parameters,
            tensors.constants,
            dn.func.InitializationInput(v_init=v_init),
        )
        m = initialized.state["mechanisms"]["hh"]["m"]
        return m.square().mean()

    transformed = torch.func.jacrev(loss)
    expected = transformed(base_v)
    compiled = torch.compile(
        transformed,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    with torch_compiler_warning_context():
        actual = compiled(base_v)
    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-11)


@pytest.mark.parametrize("kind", ["sc", "ub"])
def test_initialize_prepare_rollout_matches_imperative_and_retains_gradients(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    v_init = _new_initial_voltage(model)
    steps = 2
    count = steps * model.v.numel()
    ve = torch.linspace(-1.0, 1.0, count, dtype=model.dtype()).reshape(
        steps,
        *model.shape,
    )
    intra = torch.linspace(-1.0e-9, 1.0e-9, count, dtype=model.dtype()).reshape(
        steps,
        *model.shape,
    )

    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        dn.func.InitializationInput(v_init=v_init),
    )
    actual, _auxiliary = functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
        steps=steps,
    )

    reference = _model(kind)
    reference.set_v_init(v_init)
    reference.initialize()
    for index in range(steps):
        _imperative_step(reference, ve[index], intra[index])
    _assert_tree_close(
        actual,
        functional.extract(reference).state,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    def loss(local_v_init, gnabar):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        local = functional.initialize(
            parameters,
            tensors.constants,
            dn.func.InitializationInput(v_init=local_v_init),
        )
        final, _aux = functional.prepare_and_rollout(
            local.parameters,
            local.constants,
            local.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
            steps=steps,
        )
        return final["integrator"]["v"].square().mean()

    gradients = torch.func.grad(loss, argnums=(0, 1))(
        v_init, tensors.parameters[GNABAR]
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize(
    ("bad_v_init", "error_type", "message"),
    [
        (torch.tensor(-65.0, dtype=torch.float64), ValueError, "shape"),
        (torch.zeros(2, 3, dtype=torch.float32), ValueError, "torch.float64|dtype"),
        (-65.0, TypeError, "Tensor"),
    ],
)
def test_initialize_requires_an_exact_full_shape_tensor_input(
    bad_v_init,
    error_type,
    message,
):
    model = _model("sc")
    functional, tensors = dn.func.make_functional(model, dt=DT)

    with pytest.raises(error_type, match=message):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            dn.func.InitializationInput(v_init=bad_v_init),
        )


@pytest.mark.parametrize("hook_kind", ["pre", "post"])
def test_initialize_fails_closed_for_imperative_initialization_hooks(hook_kind):
    model = _model("sc")
    register = getattr(model, f"register_{hook_kind}_initialize_hook")
    register(lambda population: population.v.add_(1.0))

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"initialization hooks|initialize hooks",
    ):
        functional, tensors = dn.func.make_functional(model, dt=DT)
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


@pytest.mark.parametrize("mutation", ["pre_hook", "post_hook", "steady_state"])
def test_initialize_rechecks_source_initialization_semantics_after_lowering(mutation):
    model = _model("sc")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    if mutation == "steady_state":
        model.cache("_steady_state")
    else:
        hook_kind = mutation.removesuffix("_hook")
        register = getattr(model, f"register_{hook_kind}_initialize_hook")
        register(lambda population: population.v.add_(1.0))

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"initialization hooks|initialize hooks|steady.state|steady state",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


def test_initialize_fails_closed_for_a_cached_steady_state():
    model = _model("sc")
    model.cache("_steady_state")

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"steady.state|steady state",
    ):
        functional, tensors = dn.func.make_functional(model, dt=DT)
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


def test_initialize_rechecks_a_derived_buffer_builder_after_lowering():
    model = _model("sc")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    state = model.mech.hh.DE["mhn"]
    state.derive_buffers = lambda: {"q10": torch.ones_like(state.celsius)}

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"structure|derive_buffers|initialization",
    ):
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


def test_initialize_fails_closed_for_random_parameters():
    model = _model("sc", mechanism=_RandomCurrent)

    with pytest.raises(
        dn.func.FunctionalizationError,
        match=r"random parameters|_random_parameters",
    ):
        functional, tensors = dn.func.make_functional(model, dt=DT)
        functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
        )


def test_initializer_supports_stateless_mechanisms_without_special_cases():
    model = _model("sc", mechanism=pas)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    v_init = _new_initial_voltage(model)
    actual = functional.initialize(
        tensors.parameters,
        tensors.constants,
        dn.func.InitializationInput(v_init=v_init),
    )

    reference = copy.deepcopy(model)
    reference.set_v_init(v_init)
    reference.initialize()
    expected = functional.extract(reference)
    _assert_tree_close(actual.state, expected.state)
