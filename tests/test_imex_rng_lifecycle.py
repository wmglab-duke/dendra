import copy
import math

import pytest
import torch

import dendra as dn  # noqa: F401 - initialize Dendra before integrator imports
from dendra.models.integrators.core import (
    Integrator,
    _broadcast_to_shape,
    _expanded_v_init,
    _flatten_to_solve,
    ensure_model_buffer,
    flatten_model_tensor,
    flatten_optional_model_tensor,
    flatten_voltage,
    get_init_defaults,
    solver_shape_from_voltage_shape,
    unflatten_voltage,
)
from dendra.models.integrators.imex import _krylov_etd1
from dendra.models.integrators.tridiag.pcr import (
    _batched_thomas_inplace,
    pcr_solve_t,
)
from dendra.models.mechanisms import Mechanism
from dendra.models.rng import (
    RNGModule,
    _device_seed,
    _validate_base_seed,
    _validate_rng_checkpoint_payload,
    _validate_rng_state,
)

DTYPE = torch.float64


class _CableModel(torch.nn.Module):
    def __init__(self, shape=(2, 3), v_init=-0.2):
        super().__init__()
        self.shape = tuple(shape)
        self.v_init = v_init
        self.nc = self.shape[-1]
        self.temp_c = 36.0
        self.jit = False
        self.jit_network_solves = False
        self.backend = "inductor"
        self.fullgraph = False
        self.dynamic = False
        self.compile_mode = None
        self.compile_options = None

        self.register_buffer("v", self._full(v_init))
        self.register_buffer("diam", self._full(2.0))
        self.register_buffer("dx", self._full(10.0))
        self.register_buffer("cm", self._full(1.0))
        self.register_buffer("rhoa", self._full(100.0))

    def _full(self, value):
        return torch.as_tensor(value, dtype=DTYPE).expand(self.shape).clone()

    def device(self):
        return self.v.device

    def dtype(self):
        return self.v.dtype

    def expanded_v_init(self):
        return torch.as_tensor(
            self.v_init, dtype=self.v.dtype, device=self.v.device
        ).expand_as(self.v)

    def resize(self, shape):
        self.shape = tuple(shape)
        self.nc = self.shape[-1]
        for name, value in (
            ("v", self.v_init),
            ("diam", 2.0),
            ("dx", 10.0),
            ("cm", 1.0),
            ("rhoa", 100.0),
        ):
            setattr(self, name, self._full(value))


class _LinearMechanism(torch.nn.Module):
    def __init__(self, conductance=0.001, drive=0.0002):
        super().__init__()
        self.conductance = conductance
        self.drive = drive
        self.advance_calls = 0
        self.noise_calls = 0
        self.dt = None
        self.detached = False

    def advance(self, v, dt, temp):
        self.advance_calls += 1

    def i(self, v):
        conductance = torch.full_like(v, self.conductance)
        return conductance * v - self.drive, conductance

    def set_dt(self, dt):
        self.dt = float(dt)

    def sample_runtime_noises_(self, dt, phase):
        self.noise_calls += 1

    def detach(self):
        self.detached = True


class _FallbackCanonicalMechanism(Mechanism):
    Mechanism.TIMESTEP_BUFFER("coefficient")

    def derive_timestep_buffers(self, dt):
        return {"coefficient": torch.ones_like(self.diam) * dt}


def _physical_cable_terms(model):
    batch = math.prod(model.shape[:-1]) or 1
    size = model.shape[-1]
    diam = model.diam.reshape(batch, size)
    dx = model.dx.reshape(batch, size)
    cm_specific = model.cm.reshape(batch, size)
    rhoa = model.rhoa.reshape(batch, size)
    radius_cm = 1e-4 * diam / 2.0
    dx_cm = 1e-4 * dx
    area = 2.0 * torch.pi * radius_cm * dx_cm
    capacitance = 1e-6 * cm_specific * area
    axial_resistance = rhoa * dx_cm / (torch.pi * radius_cm.square())
    edge_conductance = 2.0 / (axial_resistance[:, :-1] + axial_resistance[:, 1:])
    left = edge_conductance / capacitance[:, :-1]
    right = edge_conductance / capacitance[:, 1:]

    matrix = torch.zeros(batch, size, size, dtype=model.dtype(), device=model.device())
    rows = torch.arange(size - 1, device=model.device())
    matrix[:, rows, rows] -= left
    matrix[:, rows + 1, rows + 1] -= right
    matrix[:, rows, rows + 1] = left
    matrix[:, rows + 1, rows] = right
    return matrix, area / capacitance, capacitance.reciprocal(), left, right


def _dense_imex_reference(model, mechanism, v, dt, ve, intra):
    batch = math.prod(model.shape[:-1]) or 1
    size = model.shape[-1]
    v_flat = v.reshape(batch, size)
    conductance = torch.full_like(v_flat, mechanism.conductance)
    current = conductance * v_flat - mechanism.drive
    matrix, scale, cm_inv, left, right = _physical_cable_terms(model)
    forcing = (conductance * v_flat - current) * scale
    matrix = matrix - torch.diag_embed(conductance * scale)

    if ve is not None:
        ve_flat = ve.expand(model.shape).reshape(batch, size)
        delta_ve = ve_flat[:, 1:] - ve_flat[:, :-1]
        source = torch.zeros_like(forcing)
        source[:, :-1] += left * delta_ve
        source[:, 1:] -= right * delta_ve
        forcing = forcing + source
    if intra is not None:
        forcing = forcing + intra.expand(model.shape).reshape(batch, size) * cm_inv

    dt_seconds = float(dt) * 1e-3
    outputs = []
    for batch_index in range(batch):
        augmented = torch.zeros(size + 1, size + 1, dtype=v.dtype)
        augmented[:size, :size] = matrix[batch_index]
        augmented[:size, size] = forcing[batch_index]
        transition = torch.matrix_exp(dt_seconds * augmented)
        outputs.append(
            transition[:size, :size] @ v_flat[batch_index] + transition[:size, size]
        )
    return torch.stack(outputs).reshape(model.shape)


@pytest.mark.parametrize("method", ["arnoldi", "lanczos"])
@pytest.mark.parametrize("guard", [False, True])
def test_imex_etd1_step_matches_dense_augmented_exponential(method, guard):
    model = _CableModel()
    mechanism = _LinearMechanism()
    integrator = _krylov_etd1(
        model, mechanism, m=model.nc, method=method, guard=guard
    ).to(dtype=DTYPE)
    dt = 0.01
    integrator._initialize(model, dt)
    model.v = torch.tensor([[-0.2, 0.1, 0.4], [0.3, -0.1, 0.2]], dtype=DTYPE)
    ve = torch.tensor([0.0, 0.02, -0.01], dtype=DTYPE)
    intra = torch.tensor([1.0e-14, -2.0e-14, 0.5e-14], dtype=DTYPE)
    expected = _dense_imex_reference(model, mechanism, model.v.clone(), dt, ve, intra)

    integrator.step(model, torch.tensor(dt, dtype=DTYPE), ve=ve, intra=intra)

    torch.testing.assert_close(model.v, expected, atol=2e-10, rtol=2e-10)
    assert mechanism.advance_calls == 1
    assert mechanism.noise_calls == 1


def test_imex_reinitialization_resizes_workspaces_for_new_leading_shape():
    model = _CableModel(shape=(1, 3))
    mechanism = _LinearMechanism()
    integrator = _krylov_etd1(model, mechanism, m=3).to(dtype=DTYPE)
    integrator._initialize(model, 0.01)
    assert integrator.V_buf.shape == (1, 3, 3)

    model.resize((2, 2, 3))
    integrator._initialize(model, 0.01)

    assert integrator.V_buf.shape == (4, 3, 3)
    assert integrator.H_buf.shape == (4, 3, 3)
    integrator.step(model, torch.tensor(0.01, dtype=DTYPE))
    assert model.v.shape == (2, 2, 3)
    assert torch.isfinite(model.v).all()


def test_imex_arnoldi_preserves_uniform_voltage_with_nonuniform_capacitance():
    model = _CableModel(shape=(1, 3), v_init=0.3)
    model.cm = torch.tensor([[1.0, 2.0, 0.5]], dtype=DTYPE)
    mechanism = _LinearMechanism(conductance=0.0, drive=0.0)
    integrator = _krylov_etd1(model, mechanism, m=3, method="arnoldi").to(dtype=DTYPE)
    integrator._initialize(model, 0.01)

    integrator.step(model, torch.tensor(0.01, dtype=DTYPE))

    torch.testing.assert_close(model.v, torch.full_like(model.v, 0.3))


def test_imex_arnoldi_nonuniform_cable_matches_independent_physical_reference():
    model = _CableModel(shape=(1, 3))
    model.diam = torch.tensor([[1.0, 2.0, 1.5]], dtype=DTYPE)
    model.dx = torch.tensor([[8.0, 12.0, 20.0]], dtype=DTYPE)
    model.cm = torch.tensor([[1.0, 1.7, 0.6]], dtype=DTYPE)
    model.rhoa = torch.tensor([[80.0, 120.0, 95.0]], dtype=DTYPE)
    model.v = torch.tensor([[-0.2, 0.1, 0.4]], dtype=DTYPE)
    mechanism = _LinearMechanism()
    integrator = _krylov_etd1(model, mechanism, m=3, method="arnoldi").to(dtype=DTYPE)
    dt = 0.01
    integrator._initialize(model, dt)
    ve = torch.tensor([0.0, 0.02, -0.01], dtype=DTYPE)
    intra = torch.tensor([1.0e-14, -2.0e-14, 0.5e-14], dtype=DTYPE)
    expected = _dense_imex_reference(model, mechanism, model.v.clone(), dt, ve, intra)

    integrator.step(model, torch.tensor(dt, dtype=DTYPE), ve=ve, intra=intra)

    torch.testing.assert_close(model.v, expected, atol=2e-10, rtol=2e-10)


def test_imex_lanczos_rejects_nonsymmetric_nonuniform_cable_operator():
    model = _CableModel(shape=(1, 3))
    model.cm = torch.tensor([[1.0, 2.0, 0.5]], dtype=DTYPE)
    integrator = _krylov_etd1(model, _LinearMechanism(), m=3, method="lanczos").to(
        dtype=DTYPE
    )

    with pytest.raises(ValueError, match="symmetric cable operator"):
        integrator._initialize(model, 0.01)


def test_imex_single_compartment_zero_operator_has_exact_finite_drive_step():
    model = _CableModel(shape=(1, 1), v_init=0.3)
    mechanism = _LinearMechanism(conductance=0.0, drive=0.0002)
    integrator = _krylov_etd1(model, mechanism, m=1, guard=False).to(dtype=DTYPE)
    dt = torch.tensor(0.01, dtype=DTYPE)
    integrator._initialize(model, dt)
    expected = model.v + dt * 1e-3 * mechanism.drive * integrator.scale.reshape_as(
        model.v
    )

    integrator.step(model, dt)

    assert torch.isfinite(model.v).all()
    torch.testing.assert_close(model.v, expected)


def test_imex_rejects_unsupported_membrane_current_reporting():
    model = _CableModel(shape=(1, 3))

    with pytest.raises(NotImplementedError, match="does not support imem"):
        _krylov_etd1(model, _LinearMechanism(), m=3, imem=True)


@pytest.mark.parametrize("m", [0, -1, 4, True])
def test_imex_rejects_invalid_krylov_dimensions(m):
    model = _CableModel(shape=(1, 3))
    with pytest.raises((TypeError, ValueError), match="Krylov dimension"):
        _krylov_etd1(model, _LinearMechanism(), m=m)


def test_imex_validates_method_and_revalidates_dimension_after_shape_change():
    model = _CableModel(shape=(1, 3))
    with pytest.raises(ValueError, match="Unknown method"):
        _krylov_etd1(model, _LinearMechanism(), m=2, method="unknown")

    integrator = _krylov_etd1(model, _LinearMechanism(), m=3).to(dtype=DTYPE)
    integrator._initialize(model, 0.01)
    model.resize((1, 2))
    with pytest.raises(ValueError, match="Krylov dimension"):
        integrator._initialize(model, 0.01)


def test_integrator_shape_helpers_cover_broadcast_reshape_and_validation():
    assert solver_shape_from_voltage_shape((2, 3, 4)) == (6, 4)
    with pytest.raises(ValueError, match="at least one dimension"):
        solver_shape_from_voltage_shape(())

    vector = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    flattened = flatten_model_tensor(vector, (2, 2, 3))
    assert flattened.shape == (4, 3)
    assert torch.equal(flattened[0], vector)
    assert flatten_optional_model_tensor(None, (2, 3)) is None

    block = torch.arange(6.0, dtype=DTYPE).reshape(3, 2)
    flat_block = flatten_model_tensor(block, (2, 3, 2), core_ndim=2)
    assert flat_block.shape == (2, 3, 2)
    assert torch.equal(flat_block[0], block)
    assert torch.equal(
        unflatten_voltage(flattened, (2, 2, 3)), flattened.reshape(2, 2, 3)
    )
    assert _broadcast_to_shape(torch.tensor(2.0), (2, 3)).shape == (2, 3)

    with pytest.raises(ValueError, match="core_ndim"):
        flatten_model_tensor(vector, (2, 3), core_ndim=0)
    with pytest.raises(RuntimeError, match="Cannot broadcast"):
        flatten_model_tensor(torch.ones(5), (2, 3))
    with pytest.raises(ValueError, match="Cannot broadcast"):
        _broadcast_to_shape(torch.ones(1, 2, 3), (2, 3))


def test_integrator_core_helpers_cover_defaults_buffers_and_v_init_fallbacks():
    class Defaults:
        def __init__(self, required, alpha=1, beta=None):
            pass

    assert get_init_defaults(Defaults) == {"alpha": 1, "beta": None}
    model = _CableModel(shape=(2, 3))
    first = ensure_model_buffer(model, "scratch", (2, 3))
    assert first.shape == (2, 3)
    assert ensure_model_buffer(model, "scratch", (2, 3)) is first
    resized = ensure_model_buffer(model, "scratch", (1, 3), dtype=torch.float32)
    assert resized.shape == (1, 3)
    assert resized.dtype == torch.float32

    values = flatten_model_tensor([[1.0, 2.0, 3.0]], (2, 3))
    assert values.shape == (2, 3)
    assert flatten_voltage(values).shape == (2, 3)
    same_numel = flatten_model_tensor(torch.arange(12.0).reshape(2, 6), (2, 2, 3))
    assert same_numel.shape == (4, 3)
    assert _flatten_to_solve(None) is None
    assert _flatten_to_solve(torch.ones(2, 3)).shape == (2, 3)

    class Fallback:
        pass

    fallback = Fallback()
    fallback.v = torch.zeros(2, 3, dtype=DTYPE)
    fallback.nc = 3
    for v_init, expected in (
        (-4.0, torch.full((2, 3), -4.0, dtype=DTYPE)),
        ([1.0, 2.0, 3.0], torch.tensor([[1.0, 2.0, 3.0]] * 2, dtype=DTYPE)),
        ([[1.0, 2.0, 3.0]], torch.tensor([[1.0, 2.0, 3.0]] * 2, dtype=DTYPE)),
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE),
        ),
    ):
        fallback.v_init = v_init
        assert torch.equal(_expanded_v_init(fallback), expected)

    fallback.v_init = [1.0, 2.0]
    with pytest.raises(ValueError, match="v_init has length"):
        _expanded_v_init(fallback)
    fallback.v_init = torch.ones(2, 2)
    with pytest.raises(ValueError, match="Unsupported v_init shape"):
        _expanded_v_init(fallback)


class _LifecycleIntegrator(Integrator):
    def __init__(self, model, mechanism):
        self.initialize_calls = 0
        super().__init__(model, mechanism, imem=True)

    def initialize(self, model, dt):
        self.initialize_calls += 1
        self._refresh_solver_shape(model)

    def _transition(self, value):
        return value + 1


def test_integrator_initialization_compile_cache_and_mutable_state(monkeypatch):
    model = _CableModel(shape=(1, 3))
    mechanism = _LinearMechanism()
    integrator = _LifecycleIntegrator(model, mechanism)

    integrator._initialize(model, 0.01)
    integrator._initialize(model, 0.01)
    assert integrator.initialize_calls == 1
    assert mechanism.dt == pytest.approx(0.01)
    integrator._initialize(model, 0.02)
    assert integrator.initialize_calls == 2

    compile_calls = []

    def fake_compile(function, **kwargs):
        compile_calls.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    model.jit = True
    integrator.configure_jit(model)
    value = torch.tensor(2.0)
    assert integrator._call_kernel("_transition", value).item() == 3.0
    assert integrator._call_kernel("_transition", value).item() == 3.0
    assert len(compile_calls) == 1

    model.backend = "eager"
    integrator.configure_jit(model)
    assert not integrator._compiled_kernels
    integrator._call_kernel("_transition", value)
    assert len(compile_calls) == 2
    assert mechanism.noise_calls == 3

    integrator.init_v(model)
    model.i_membrane.fill_(4.0)
    state = integrator.mutable_state_dict(model)
    model.v = torch.zeros_like(model.v)
    model.i_membrane = torch.zeros_like(model.i_membrane)
    integrator.restore_mutable_state_dict(model, state)
    assert torch.all(model.v == model.v_init)
    assert torch.all(model.i_membrane == 4.0)

    integrator._compiled_kernels["sentinel"] = object()
    assert integrator.__getstate__()["_compiled_kernels"] == {}
    with pytest.raises(ValueError, match="at most one"):
        integrator.pickleable(inplace=True, clone=True)
    assert integrator.pickleable(inplace=True) is integrator
    assert not integrator._compiled_kernels

    cloned = integrator.pickleable(clone=True)
    assert cloned is not integrator
    assert not cloned._compiled_kernels
    integrator.detach(model)
    assert mechanism.detached


def test_integrator_shape_views_force_reinit_and_handler_dt_fallback():
    model = _CableModel(shape=(2, 3))

    class Leaf:
        def __init__(self):
            self.dt = None

        def set_dt(self, dt):
            self.dt = float(dt)

    class Handler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            canonical = _FallbackCanonicalMechanism(
                "canonical",
                torch.full_like(model.diam, 36.0),
                model.diam,
                model.shape,
                model.shape,
            )
            self.mechanisms = {"a": Leaf(), "b": Leaf(), "canonical": canonical}

        def detach(self):
            pass

    handler = Handler()
    integrator = _LifecycleIntegrator(model, handler)
    integrator._initialize(model, 0.01, force=True)
    assert [handler.mechanisms[name].dt for name in ("a", "b")] == [0.01, 0.01]
    canonical = handler.mechanisms["canonical"]
    assert float(canonical.dt) == pytest.approx(0.01)
    torch.testing.assert_close(
        canonical.coefficient,
        torch.full_like(canonical.coefficient, 0.01),
    )
    assert integrator.needs_to_be_initialized(model, 0.01) is False
    assert integrator.needs_to_be_initialized(model, 0.01, force=True) is True

    values = torch.arange(6.0).reshape(2, 3)
    block = torch.arange(24.0).reshape(2, 3, 4)
    assert integrator._flat_voltage(None) is None
    assert integrator._flat_voltage(values).shape == (2, 3)
    assert integrator._flat_block_voltage(None, 4) is None
    assert integrator._flat_block_voltage(block, 4).shape == (2, 3, 4)
    assert torch.equal(integrator._restore_voltage(values), values)
    integrator._refresh_solver_shape(model, block_dim=4)
    assert torch.equal(integrator._restore_block_voltage(block), block)


def test_rng_state_dict_round_trip_reproduces_exact_suffix_and_can_be_ignored():
    source = RNGModule(1234, shape_p=(2, 3), shape_f=(2, 3))
    source.rand()
    checkpoint = copy.deepcopy(source.state_dict())
    expected_suffix = (source.rand(), source.randn())

    restored = RNGModule(999, shape_p=(2, 3), shape_f=(2, 3))
    restored.load_state_dict(checkpoint)
    actual_suffix = (restored.rand(), restored.randn())
    for actual, expected in zip(actual_suffix, expected_suffix):
        assert torch.equal(actual, expected)

    ignored = RNGModule(77, shape_p=(2, 3), shape_f=(2, 3))
    control = RNGModule(77, shape_p=(2, 3), shape_f=(2, 3))
    ignored.ignore_rng_on_load()
    ignored.load_state_dict(checkpoint)
    assert torch.equal(ignored.rand(), control.rand())


def test_public_network_load_restores_exact_rng_suffix():
    source = dn.Network({}, seed=1234)
    torch.rand(5, generator=source._rng("cpu"))
    checkpoint = copy.deepcopy(source.state_dict())
    expected_suffix = torch.rand(8, generator=source._rng("cpu"))

    restored = dn.Network({}, seed=999)
    assert restored.load(checkpoint) is restored
    actual_suffix = torch.rand(8, generator=restored._rng("cpu"))

    assert torch.equal(actual_suffix, expected_suffix)


def test_rng_unavailable_device_state_is_deferred_resaved_and_applied_lazily(
    monkeypatch,
):
    real_generator = torch.Generator
    deferred_generator = real_generator(device="cpu").manual_seed(314)
    deferred_state = deferred_generator.get_state().clone()
    expected_suffix = torch.rand(6, generator=deferred_generator)

    def unavailable_cuda_generator(device="cpu"):
        if torch.device(device).type == "cuda":
            raise RuntimeError("CUDA unavailable in this test")
        return real_generator(device=device)

    monkeypatch.setattr(torch, "Generator", unavailable_cuda_generator)
    source = RNGModule(11, shape_p=(1,), shape_f=(1,))
    source.set_rng_state({"cpu": source._cpu_gen.get_state(), "cuda:7": deferred_state})
    assert "cuda:7" in source._pending_device_states
    assert torch.equal(source.rng_state()["cuda:7"], deferred_state)

    checkpoint = copy.deepcopy(source.state_dict())
    restored = RNGModule(99, shape_p=(1,), shape_f=(1,))
    restored.load_state_dict(checkpoint)
    assert torch.equal(restored._pending_device_states["cuda:7"], deferred_state)
    assert torch.equal(
        restored.state_dict()["_extra_state"]["rng_state"]["cuda:7"],
        deferred_state,
    )

    for operation in ("reseed", "reset"):
        cleared = RNGModule(99, shape_p=(1,), shape_f=(1,))
        cleared.load_state_dict(checkpoint)
        if operation == "reseed":
            cleared.reseed(99)
        else:
            cleared.reset()
        assert cleared._pending_device_states == {}

    # Simulate the same device becoming available. A CPU generator is a safe
    # stand-in here because the lifecycle behavior, not a backend algorithm, is
    # under test.
    monkeypatch.setattr(
        torch, "Generator", lambda device="cpu": real_generator(device="cpu")
    )
    restored_generator = restored._rng("cuda:7")
    assert "cuda:7" not in restored._pending_device_states
    assert torch.equal(torch.rand(6, generator=restored_generator), expected_suffix)


def test_rng_reseed_reset_shapes_like_helpers_and_binomial_are_reproducible():
    fresh = RNGModule(42, shape_p=(1,), shape_f=(1,))
    fresh.reset()
    assert fresh.rng is fresh._cpu_gen

    rng = RNGModule(42, shape_p=(2, 1), shape_f=(2, 4))
    first_uniform = rng.rand(dtype=DTYPE)
    first_normal = rng.randn(dtype=DTYPE)
    assert first_uniform.shape == first_normal.shape == (2, 4)

    template = torch.empty((3, 2), dtype=torch.float32)
    assert rng.rand_like(template).shape == template.shape
    assert rng.randn_like(template).dtype == template.dtype

    rng.reseed(42)
    assert torch.equal(rng.rand(dtype=DTYPE), first_uniform)
    assert torch.equal(rng.randn(dtype=DTYPE), first_normal)

    rng.reseed(19)
    first_binomial = rng.binomial(10, 0.25, shape=(32,))
    rng.reset()
    assert torch.equal(rng.binomial(10, 0.25, shape=(32,)), first_binomial)
    assert rng._rng(None) is rng._cpu_gen


def test_rng_reset_restores_all_lazily_registered_device_generators():
    rng = RNGModule(7, shape_p=(1,), shape_f=(1,))
    surrogate = torch.Generator(device="cpu")
    device = torch.device("meta")
    rng._device_gens[device] = surrogate
    rng.reseed(7)
    initial_state = surrogate.get_state().clone()
    torch.rand(4, generator=surrogate)

    rng.reset()

    assert torch.equal(surrogate.get_state(), initial_state)


def test_rng_device_state_round_trip_and_stable_seed_bounds():
    source = RNGModule(11, shape_p=(1,), shape_f=(1,))
    device = torch.device("cpu:1")
    source._device_gens[device] = torch.Generator(device="cpu")
    source.reseed(11)
    torch.rand(3, generator=source._device_gens[device])
    state = source.rng_state()
    expected = torch.rand(3, generator=source._device_gens[device])

    restored = RNGModule(99, shape_p=(1,), shape_f=(1,))
    restored.set_rng_state(state)
    actual = torch.rand(3, generator=restored._device_gens[device])
    assert torch.equal(actual, expected)
    assert _device_seed((1 << 64) - 1, torch.device("cuda:17")) <= (1 << 64) - 1
    assert _device_seed(11, torch.device("cuda:0")) != _device_seed(
        11, torch.device("cuda:1")
    )


def test_rng_unseeded_extra_state_default_shapes_and_meta_fallback():
    unseeded = RNGModule(None, shape_p=(2,), shape_f=(2,))
    assert 0 <= unseeded._base_seed <= (1 << 64) - 1
    seeded = RNGModule(5, shape_p=(2,), shape_f=(2,))
    unseeded.set_extra_state(copy.deepcopy(seeded.get_extra_state()))
    expected = seeded.rand()
    assert torch.equal(unseeded.rand(), expected)

    assert unseeded.binomial(1.0, 0.5).shape == (2,)
    assert unseeded.rand((2,), device="meta").device.type == "meta"
    assert unseeded.randn((2,), device="meta").device.type == "meta"


def test_rng_payload_validation_rejects_ambiguous_or_malformed_metadata(
    monkeypatch,
):
    valid_cpu = torch.Generator(device="cpu").manual_seed(71).get_state()

    assert _validate_base_seed(-(1 << 63)) == -(1 << 63)
    assert _validate_base_seed((1 << 64) - 1) == (1 << 64) - 1
    for invalid in (True, 1.5, "7"):
        with pytest.raises(TypeError, match="must be an integer"):
            _validate_base_seed(invalid)
    for invalid in (-(1 << 63) - 1, 1 << 64):
        with pytest.raises(ValueError, match="must be between"):
            _validate_base_seed(invalid)

    with pytest.raises(TypeError, match="must be a mapping"):
        _validate_rng_state([])
    with pytest.raises(TypeError, match="device keys"):
        _validate_rng_state({0: valid_cpu})
    with pytest.raises(ValueError, match="Invalid RNG state device key"):
        _validate_rng_state({"not-a-device": valid_cpu})
    with pytest.raises(ValueError, match="Duplicate RNG state"):
        _validate_rng_state({"cpu": valid_cpu, torch.device("cpu"): valid_cpu.clone()})
    with pytest.raises(KeyError, match="cpu"):
        _validate_rng_state({})
    with pytest.raises(TypeError, match="must be a tensor"):
        _validate_rng_state({"cpu": [1, 2, 3]})
    with pytest.raises(TypeError, match="torch.uint8"):
        _validate_rng_state({"cpu": valid_cpu.to(torch.int64)})
    with pytest.raises(ValueError, match="stored on CPU"):
        _validate_rng_state(
            {"cpu": torch.empty(valid_cpu.shape, dtype=torch.uint8, device="meta")}
        )
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_rng_state({"cpu": torch.empty(0, dtype=torch.uint8)})
    with pytest.raises(RuntimeError, match="not a valid generator state"):
        _validate_rng_state({"cpu": torch.zeros(3, dtype=torch.uint8)})

    canonical = _validate_rng_state({torch.device("cpu"): valid_cpu})
    assert canonical.keys() == {"cpu"}
    assert canonical["cpu"] is not valid_cpu
    assert torch.equal(canonical["cpu"], valid_cpu)

    with pytest.raises(TypeError, match="payload must be a mapping"):
        _validate_rng_checkpoint_payload([], allow_legacy=True)
    with pytest.raises(KeyError, match="requires both"):
        _validate_rng_checkpoint_payload({"base_seed": 7}, allow_legacy=True)
    with pytest.raises(KeyError, match="extra state requires"):
        _validate_rng_checkpoint_payload({"cpu": valid_cpu}, allow_legacy=False)

    seed, state = _validate_rng_checkpoint_payload(
        {"base_seed": 7, "rng_state": {"cpu": valid_cpu}}, allow_legacy=False
    )
    assert seed == 7
    assert torch.equal(state["cpu"], valid_cpu)
    seed, state = _validate_rng_checkpoint_payload(
        {"cpu": valid_cpu}, allow_legacy=True
    )
    assert seed is None
    assert torch.equal(state["cpu"], valid_cpu)

    real_generator = torch.Generator

    def unavailable_cpu_generator(device="cpu"):
        raise RuntimeError(f"unavailable {device}")

    monkeypatch.setattr(torch, "Generator", unavailable_cpu_generator)
    with pytest.raises(RuntimeError, match="unavailable cpu"):
        _validate_rng_state({"cpu": valid_cpu})
    monkeypatch.setattr(torch, "Generator", real_generator)


@pytest.mark.parametrize("restore_kind", ("rng_state", "extra_state"))
def test_rng_restore_rolls_back_after_an_apply_failure(monkeypatch, restore_kind):
    rng = RNGModule(19, shape_p=(2,), shape_f=(2,))
    rng.rand()
    before_seed = rng._base_seed
    before_state = rng.rng_state()["cpu"].clone()
    replacement = RNGModule(91, shape_p=(2,), shape_f=(2,)).get_extra_state()
    real_apply = rng._apply_validated_rng_state
    calls = 0

    def fail_once(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            rng._cpu_gen.manual_seed(999)
            raise RuntimeError("injected apply failure")
        return real_apply(state)

    monkeypatch.setattr(rng, "_apply_validated_rng_state", fail_once)
    with pytest.raises(RuntimeError, match="injected apply failure"):
        if restore_kind == "rng_state":
            rng.set_rng_state(replacement["rng_state"])
        else:
            rng.set_extra_state(replacement)

    assert calls == 2
    assert rng._base_seed == before_seed
    assert torch.equal(rng.rng_state()["cpu"], before_state)


def test_rng_legacy_pickle_fallback_and_integer_binomial(monkeypatch):
    real_generator = torch.Generator
    rng = RNGModule(43, shape_p=(4,), shape_f=(4,))
    del rng._pending_device_states
    monkeypatch.setattr(
        torch, "Generator", lambda device="cpu": real_generator(device="cpu")
    )

    surrogate = rng._rng("meta")
    assert surrogate.device.type == "cpu"
    assert hasattr(rng, "_pending_device_states")

    fresh = RNGModule(8, shape_p=(4,), shape_f=(4,))
    sample = fresh.binomial(1, 1)
    assert sample.dtype == torch.get_default_dtype()
    assert torch.equal(sample, torch.ones(4))


def _dense_tridiagonal(a, b, c):
    matrix = torch.diag_embed(b)
    rows = torch.arange(b.shape[-1] - 1)
    matrix[..., rows + 1, rows] = a
    matrix[..., rows, rows + 1] = c
    return matrix


def test_pcr_handles_scalar_system_and_arbitrary_leading_batch_dimensions():
    scalar = pcr_solve_t(
        torch.empty(0, dtype=DTYPE),
        torch.tensor([2.0], dtype=DTYPE),
        torch.empty(0, dtype=DTYPE),
        torch.tensor([6.0], dtype=DTYPE),
    )
    torch.testing.assert_close(scalar, torch.tensor([3.0], dtype=DTYPE))

    shape = (2, 3, 4)
    a = torch.full(shape[:-1] + (3,), -0.2, dtype=DTYPE)
    b = torch.full(shape, 2.0, dtype=DTYPE)
    c = torch.full(shape[:-1] + (3,), 0.1, dtype=DTYPE)
    rhs = torch.arange(torch.tensor(shape).prod(), dtype=DTYPE).reshape(shape)
    expected = torch.linalg.solve(_dense_tridiagonal(a, b, c), rhs.unsqueeze(-1))
    torch.testing.assert_close(pcr_solve_t(a, b, c, rhs), expected.squeeze(-1))


@pytest.mark.parametrize(
    "a,b,c,rhs,message",
    [
        (torch.empty(0), torch.ones(2), torch.empty(0), torch.ones(3), "same shape"),
        (
            torch.empty(0),
            torch.tensor(1.0),
            torch.empty(0),
            torch.tensor(1.0),
            "at least one",
        ),
        (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), "K >= 1"),
        (torch.ones(2), torch.ones(3), torch.ones(1), torch.ones(3), "a and c"),
    ],
)
def test_pcr_validates_band_shapes(a, b, c, rhs, message):
    with pytest.raises(ValueError, match=message):
        pcr_solve_t(a, b, c, rhs)


def test_pcr_is_autograd_safe_for_all_bands_and_rhs():
    a = torch.tensor([[-0.2, 0.1]], dtype=DTYPE, requires_grad=True)
    b = torch.tensor([[2.0, 2.5, 3.0]], dtype=DTYPE, requires_grad=True)
    c = torch.tensor([[0.3, -0.1]], dtype=DTYPE, requires_grad=True)
    rhs = torch.tensor([[1.0, -2.0, 0.5]], dtype=DTYPE, requires_grad=True)

    assert torch.autograd.gradcheck(
        pcr_solve_t,
        (a, b, c, rhs),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_legacy_inplace_thomas_helper_matches_dense_and_handles_empty_system():
    a = torch.tensor([[0.0, -0.2, 0.1]], dtype=DTYPE)
    b = torch.tensor([[2.0, 2.5, 3.0]], dtype=DTYPE)
    c = torch.tensor([[0.3, -0.1, 0.0]], dtype=DTYPE)
    rhs = torch.tensor([[1.0, -2.0, 0.5]], dtype=DTYPE)
    matrix = _dense_tridiagonal(a[:, 1:], b, c[:, :-1])
    expected = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)

    _batched_thomas_inplace(a.clone(), b.clone(), c.clone(), rhs)

    torch.testing.assert_close(rhs, expected)
    empty = torch.empty((2, 0), dtype=DTYPE)
    _batched_thomas_inplace(empty, empty, empty, empty)
