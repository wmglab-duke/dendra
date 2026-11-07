"""Material operators share explicit RAND/NOISE sampling with native execution."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import (
    ClampProcess,
    ClearanceProcess,
    DiffusionProcess,
    ExchangeProcess,
    MaterialProcess,
    Mechanism,
)
from dendra.models.mod import expsyn, pas
from tests.func.test_network import _clone, _close
from tests.func.test_network_bitpacked_inference import _sync_stimulus

pytestmark = pytest.mark.cpu
DT = 0.125


class _MaterialCurrent(Mechanism):
    Mechanism.USEMATERIAL("pool", read=["amount"])
    Mechanism.RANGE(g=0.001)
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00001, seed=83, rng_name="stream")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        conductance = self.g * (1.0 + 0.1 * self.amount)
        return conductance, -65.0 * conductance + self.eta.clone()


class _RandomTransfer(MaterialProcess):
    MaterialProcess.PHASE("post_transport")
    MaterialProcess.GLOBALRAND(
        "gain", distribution="uniform", low=0.7, high=0.9, seed=41, rng_name="stream"
    )
    MaterialProcess.RANGERAND("rate", mu=0.4, sigma=0.03, seed=42, rng_name="stream")
    MaterialProcess.BATCHRAND(
        "factor",
        distribution="lognormal",
        mu=0.0,
        sigma=0.05,
        seed=43,
        rng_name="stream",
    )
    MaterialProcess.RANGENOISE(
        "drive", mu=0.1, sigma=0.03, scale="dW", seed=44, rng_name="stream"
    )
    MaterialProcess.GLOBALNOISE(
        "offset",
        distribution="uniform",
        low=-0.01,
        high=0.01,
        seed=45,
        rng_name="stream",
    )
    MaterialProcess.BATCHNOISE("nudge", mu=0.0, sigma=0.01, seed=46, rng_name="stream")

    @property
    def material_fields(self):
        return (("pool", "amount"), ("pool", "reserve"))

    def prepare_materials(self, fields, dt):
        like = fields[("pool", "amount")]
        coefficient = dt * self._resolve_quantity("gain", like)
        coefficient = coefficient * self._resolve_quantity("rate", like)
        coefficient = coefficient * self._resolve_quantity("factor", like)
        # Native sampling keeps buffer identity stable, so each step retains
        # its detached sample for backward before the next in-place refresh.
        drive = self._resolve_quantity("drive", like).clone()
        drive = drive + self._resolve_quantity("offset", like).clone()
        drive = drive + self._resolve_quantity("nudge", like).clone()
        return coefficient, drive

    def material_updates(self, fields, prepared):
        coefficient, drive = prepared
        amount, reserve = fields[("pool", "amount")], fields[("pool", "reserve")]
        transfer = coefficient * (amount - 0.25 * reserve) + drive
        return self._mask_material_updates(
            fields,
            {
                ("pool", "amount"): amount - transfer,
                ("pool", "reserve"): reserve + transfer,
            },
        )


class _StickyTransfer(_RandomTransfer):
    MaterialProcess.RANGERAND(
        "rate",
        mu=0.4,
        sigma=0.03,
        seed=42,
        rng_name="stream",
        resample_on_initialize=False,
    )


class _ReseededTransfer(_RandomTransfer):
    MaterialProcess.GLOBALRAND(
        "gain",
        distribution="uniform",
        low=0.7,
        high=0.9,
        seed=41,
        rng_name="stream",
        reseed_on_initialize=True,
    )


class _RetainedTransfer(_StickyTransfer):
    MaterialProcess.GLOBALRAND(
        "gain",
        distribution="uniform",
        low=0.7,
        high=0.9,
        seed=41,
        rng_name="stream",
        resample_on_initialize=False,
    )
    MaterialProcess.BATCHRAND(
        "factor",
        distribution="lognormal",
        mu=0.0,
        sigma=0.05,
        seed=43,
        rng_name="stream",
        resample_on_initialize=False,
    )


class _Transport(MaterialProcess):
    MaterialProcess.PHASE("transport")
    MaterialProcess.GLOBAL(rate=0.25)
    material_fields = (("pool", "amount"), ("pool", "reserve"))

    def prepare_materials(self, fields, dt):
        del fields
        return (dt * self.rate,)

    def material_updates(self, fields, prepared):
        (coefficient,) = prepared
        amount, reserve = fields[("pool", "amount")], fields[("pool", "reserve")]
        transfer = coefficient * (reserve - 0.5 * amount)
        return {
            ("pool", "amount"): amount + transfer,
            ("pool", "reserve"): reserve - transfer,
        }


class _NoisyDiffusion(DiffusionProcess):
    DiffusionProcess.GLOBAL(decay=0.17)
    DiffusionProcess.GLOBALNOISE(
        "D", distribution="uniform", low=0.3, high=0.7, seed=73
    )
    DiffusionProcess.GLOBALRAND(
        "relaxation",
        mu=0.1,
        sigma=0.01,
        seed=74,
        resample_on_initialize=False,
    )
    DiffusionProcess.METHOD("implicit", solver="dense")
    DiffusionProcess.DIFFUSE("pool", field="amount", D="D", geometry="chemical")
    DiffusionProcess.RELAX("pool", field="amount", rate="decay", target="relaxation")


class _NoisyThomasDiffusion(_NoisyDiffusion):
    DiffusionProcess.METHOD("implicit", solver="thomas")


class _NoisyClearance(ClearanceProcess):
    ClearanceProcess.PHASE("post_transport")
    ClearanceProcess.RANGERAND("target", mu=0.8, sigma=0.02, seed=91, rng_name="stream")
    ClearanceProcess.RANGENOISE(
        "rate", distribution="uniform", low=0.2, high=0.6, seed=92, rng_name="stream"
    )
    ClearanceProcess.CLEAR("pool", field="amount", rate="rate", target="target")


class _NoisyClamp(ClampProcess):
    ClampProcess.PHASE("post_transport")
    ClampProcess.RANGERAND("lower", mu=0.2, sigma=0.01, seed=93, rng_name="stream")
    ClampProcess.RANGENOISE(
        "upper", distribution="uniform", low=1.1, high=1.3, seed=94, rng_name="stream"
    )
    ClampProcess.BOUNDS("pool", field="amount", lower="lower", upper="upper")


class _NoisyExchange(ExchangeProcess):
    ExchangeProcess.PHASE("post_transport")
    ExchangeProcess.RANGERAND("rate", mu=0.2, sigma=0.01, seed=95, rng_name="stream")
    ExchangeProcess.RANGENOISE(
        "volume", distribution="uniform", low=1.1, high=1.3, seed=96, rng_name="stream"
    )
    ExchangeProcess.METHOD("exact")
    ExchangeProcess.EXCHANGE(
        "pool.amount", "pool.reserve", rate="rate", volume_a="volume", volume_b=1.0
    )


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"
            ),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _population(
    process=_RandomTransfer, *, batch=None, regional=False, shared_rng=True
):
    population = dn.SingleCompartment(
        N=2, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.material(
        "pool",
        fields={"amount": 1.7, "reserve": 0.4},
        min_values={"amount": 0.0, "reserve": 0.0},
    )
    population.insert(_MaterialCurrent.rename("current"))
    # Insertion and application order differ: transport runs before stochastic
    # post-transport even though its RAND/NOISE owner is visited second.
    if regional == "permuted":
        owner = population[:, [1, 0]]
    else:
        owner = population[:, [0]] if regional else population
    owner.insert(
        process.rename("random"), preserve_duplicate_indices=regional == "permuted"
    )
    population.insert(_Transport.rename("transport"))
    if batch is not None:
        population.batch(batch)
    population.train()
    population.initialize()
    if shared_rng:
        population.mech.material_processes[
            "random"
        ].stream = population.mech.current.stream
        population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def _native_step(population):
    population._step(
        population.integrator, population, population.t.new_tensor(DT), None, None
    )
    population.t = population.t + DT


def _fields(state):
    return state["materials"]["pool"]


def _parameter(tensors, suffix):
    names = [name for name in tensors.parameters if name.endswith(suffix)]
    assert len(names) == 1, names
    return names[0], tensors.parameters[names[0]]


@pytest.mark.parametrize("check_purity", (False, True))
def test_material_stochastic_draw_packets_are_required_and_names_are_validated(
    check_purity,
):
    population = _population()
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional(population, dt=DT)
    with pytest.raises(ValueError, match="draw packet"):
        functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
    with pytest.raises(ValueError, match="explicit RuntimeNoiseDraw"):
        functional.prepare_and_step(
            tensors.parameters, tensors.constants, tensors.state
        )
    packet = functional.sample_noise_draw(tensors.state)
    material_name = next(name for name in packet.draws if "material_processes" in name)
    malformed = packet._replace(
        draws={
            name: value for name, value in packet.draws.items() if name != material_name
        }
    )
    with pytest.raises(ValueError, match="names"):
        functional.prepare_and_step(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.StepInput(noise=malformed),
        )


def test_material_stochastic_range_operands_follow_full_regional_permutation():
    source, native = _population(regional="permuted"), _population(regional="permuted")
    process = source.mech.material_processes["random"]
    like = source.mech.materials["pool"].amount
    torch.testing.assert_close(
        process._resolve_quantity("rate", like), process.rate.reshape_as(like).flip(-1)
    )
    functional, tensors = dn.func.make_functional(source, dt=DT)
    state, _ = functional.prepare_and_step(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.StepInput(noise=functional.sample_noise_draw(tensors.state)),
    )
    before = _fields(tensors.state)
    _native_step(native)
    process = native.mech.material_processes["random"]
    torch.testing.assert_close(
        process._resolve_quantity("drive", like),
        process.drive.reshape_as(like).flip(-1),
    )
    transported = DT * 0.25 * (before["reserve"] - 0.5 * before["amount"])
    amount, reserve = before["amount"] + transported, before["reserve"] - transported
    coefficient = (
        DT * process.gain * process.rate.reshape_as(like).flip(-1) * process.factor
    )
    drive = process.drive.reshape_as(like).flip(-1) + process.offset + process.nudge
    expected = amount - coefficient * (amount - 0.25 * reserve) - drive
    torch.testing.assert_close(_fields(state)["amount"], expected)
    _close(state, functional.extract(native).state)


def test_material_stochastic_local_operand_mapping_ignores_matching_full_shape():
    population = _population()
    process = _RandomTransfer(
        name="permuted",
        celsius=population.celsius,
        diameters=population.diam,
        shape=population.shape,
        shape_f=population.shape,
        key=torch.tensor([[1, 0], [3, 2]]),
    )
    process._bind_materials(
        population.mech.materials.__getitem__, population=population
    )
    process.populate()
    process.sample_runtime_noises_(dt=population.t.new_tensor(DT), force=True)
    like = population.mech.materials["pool"].amount
    assert process.rate.shape == process.drive.shape == like.shape
    # Matching tensor shape does not identify the global compartment frame.
    for name in ("rate", "drive"):
        torch.testing.assert_close(
            process._resolve_quantity(name, like), getattr(process, name).flip(-1)
        )


@pytest.mark.parametrize("process", (_NoisyClearance, _NoisyClamp, _NoisyExchange))
def test_material_builtin_pointwise_operators_refresh_noise_after_preparation(process):
    source, native = _population(process), _population(process)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    tape = functional.sample_noise_draw_tape(tensors.state, 3)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state
    for index in range(3):
        state, _ = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(noise=pytree.tree_map(lambda value: value[index], tape)),
        )
        _native_step(native)
        _close(state, functional.extract(native).state)


def _diffusion_population(process=_NoisyDiffusion, *, warm=True):
    population = dn.SingleCompartment(
        N=1, C=3, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.diam.fill_(2.0)
    population.dx.fill_(1.0)
    population.material(
        "pool", fields={"amount": population.v.new_tensor([[1.0, 2.0, 4.0]])}
    )
    population.register_material_geometry(
        "chemical",
        domain="intracellular",
        volume=torch.nn.Parameter(population.v.new_tensor([[1.0, 1.2, 0.8]])),
        edge_factor=torch.nn.Parameter(population.v.new_tensor([[0.7, 0.9]])),
    )
    population.insert(pas, g=0.001, e=-65.0)
    population.insert(process.rename("diffusion"))
    population.train()
    population.initialize()
    if not warm:
        return population
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    _native_step(population)
    # Capture a nonzero sampled D in the native solver. Subsequent runtime
    # noise refreshes do not rebuild this timestep-owned coefficient.
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def test_material_diffusion_cold_snapshot_configures_without_mutating_native_source():
    source, native = (
        _diffusion_population(warm=False),
        _diffusion_population(warm=False),
    )
    saved = _clone(source.state_dict(), detached=True)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    _close(source.state_dict(), saved)
    native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    tape = functional.sample_noise_draw_tape(tensors.state, 3)
    actual, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.RolloutInput(noise=tape),
    )
    for _ in range(3):
        _native_step(native)
    _close(actual, functional.extract(native).state)
    _close(source.state_dict(), saved)


@pytest.mark.parametrize("fresh", (False, True))
def test_material_diffusion_noise_preserves_timestep_owned_workspace(fresh):
    source, native = _diffusion_population(), _diffusion_population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    if fresh:
        tensors = functional.initialize_host(
            tensors.parameters,
            tensors.constants,
            tensors.initialization,
            state=tensors.state,
        )
        native.initialize()
        native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    tape = functional.sample_noise_draw_tape(tensors.state, 4)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state = tensors.state
    for index in range(4):
        state, _ = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(noise=pytree.tree_map(lambda value: value[index], tape)),
        )
        _native_step(native)
        _close(state, functional.extract(native).state)


def test_material_diffusion_snapshot_preserves_held_noise_and_parameter_geometry_gradients():
    source, native = _diffusion_population(), _diffusion_population()
    # Refresh the runtime D after its timestep workspace has captured the
    # previous sample. Lowering must retain that capture rather than the latest
    # visible NOISE buffer, including for differentiation through its solver.
    _native_step(source)
    _native_step(native)
    for population in (source, native):
        material = population.mech.materials["pool"]
        material._buffers["amount"] = material.amount.detach()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    tape = functional.sample_noise_draw_tape(tensors.state, 2)
    actual, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.RolloutInput(noise=tape),
    )
    for _ in range(2):
        _native_step(native)
    _close(actual, functional.extract(native).state)
    name = next(
        name for name in tensors.parameters if name.endswith("diffusion.decay_param")
    )
    geometry_path = source.material_geometry("chemical")["volume"]
    geometry_name = next(
        name for name in tensors.parameters if name.endswith(geometry_path)
    )
    actual_derivatives = torch.autograd.grad(
        _fields(actual)["amount"].square().sum(),
        (tensors.parameters[name], tensors.parameters[geometry_name]),
    )
    native_derivatives = torch.autograd.grad(
        native.mech.materials["pool"].amount.square().sum(),
        (native.get_parameter(name), native.get_parameter(geometry_name)),
    )
    for actual_derivative, expected_derivative in zip(
        actual_derivatives, native_derivatives, strict=True
    ):
        assert actual_derivative.isfinite().all() and actual_derivative.abs().sum() > 0
        torch.testing.assert_close(
            actual_derivative, expected_derivative, rtol=2e-9, atol=2e-10
        )


def test_material_diffusion_owned_cache_restores_held_noise_before_continuation():
    source, native = _diffusion_population(), _diffusion_population()
    for population in (source, native):
        population.cache("_steady_state")
        for _ in range(2):
            _native_step(population)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initialized = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    tape = functional.sample_noise_draw_tape(initialized.state, 3)
    actual, _ = functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        dn.func.RolloutInput(noise=tape),
    )
    for _ in range(3):
        _native_step(native)
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("cache_kind", (None, "population", "network"))
def test_material_diffusion_network_initialization_tapes_and_cache_continuation(
    cache_kind,
):
    def network():
        model = dn.Network(
            {
                name: _diffusion_population(_NoisyThomasDiffusion)
                for name in ("left", "right")
            }
        )
        model.train()
        model.initialize(DT)
        for _ in range(2):
            model.step()
        if cache_kind == "population":
            for population in model.populations.values():
                population.cache("_steady_state")
        elif cache_kind == "network":
            model.cache_state()
        if cache_kind is not None:
            for _ in range(2):
                model.step()
        return model

    source, native = network(), network()
    functional, tensors = dn.func.make_functional_network(source)
    draws = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    defaults = functional.initialization_inputs
    inputs = defaults._replace(
        populations={
            name: value._replace(random_draw=draws[name])
            for name, value in defaults.populations.items()
        }
    )
    initialized = functional.initialize(*tensors, inputs)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized.state, functional.extract(native).state)
    tapes = functional.sample_population_noise_tapes(initialized.state, 4)
    rollout_input = dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(noise=tape) for name, tape in tapes.items()
        }
    )
    saved = _clone((initialized, rollout_input), detached=True)
    prepared = functional.prepare(initialized.parameters, initialized.constants)
    kernel = functional.compile_rollout_chunk(2, execution="scan", backend="aot_eager")
    state = initialized.state
    for start in (0, 2):
        state, _ = kernel(
            initialized.parameters,
            prepared,
            state,
            pytree.tree_map_only(
                torch.Tensor, lambda value: value[start : start + 2], rollout_input
            ),
        )
        for _ in range(2):
            native.step()
        _close(state, functional.extract(native).state)
    _close((initialized, rollout_input), saved)


@pytest.mark.parametrize("batch,regional", ((None, False), (2, False), (2, True)))
def test_material_stochastic_explicit_tape_matches_native_phases_shapes_and_rng(
    batch, regional
):
    source, native = (
        _population(batch=batch, regional=regional),
        _population(batch=batch, regional=regional),
    )
    functional, tensors = dn.func.make_functional(source, dt=DT)
    saved = _clone(tensors, detached=True)
    global_rng = torch.get_rng_state().clone()
    tape = functional.sample_noise_draw_tape(tensors.state, 4)
    _close(tape, functional.sample_noise_draw_tape(tensors.state, 4))
    state = tensors.state
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    for index in range(4):
        packet = pytree.tree_map(lambda value: value[index], tape)
        before = _fields(state)
        state, _ = functional.step(
            tensors.parameters, prepared, state, dn.func.StepInput(noise=packet)
        )
        _native_step(native)
        _close(state, functional.extract(native).state)
        torch.testing.assert_close(
            _fields(state)["amount"] + _fields(state)["reserve"],
            before["amount"] + before["reserve"],
        )
    assert len(state["random"]["rng"]) == 1
    assert all(not value.requires_grad for value in state["runtime_noise"].values())
    _close(tensors, saved)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize(
    "process", (_RandomTransfer, _StickyTransfer, _ReseededTransfer)
)
def test_material_random_fresh_draws_follow_native_policy_after_noise_continuation(
    process,
):
    source, native = _population(process), _population(process)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    for _ in range(2):
        draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
        previous_random = _clone(tensors.state["random"], detached=True)
        tensors = functional.initialize(
            tensors.parameters,
            tensors.constants,
            tensors.initialization._replace(random_draw=draw),
            state=tensors.state,
        )
        native.initialize()
        _close(tensors.state, functional.extract(native).state)
        if process is _StickyTransfer:
            retained = next(
                name
                for name in previous_random["values"]
                if name.startswith("random.draws.") and ".rate." in name
            )
            torch.testing.assert_close(
                tensors.state["random"]["values"][retained],
                previous_random["values"][retained],
            )
        tape = functional.sample_noise_draw_tape(tensors.state, 2)
        state, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(noise=tape),
        )
        for _ in range(2):
            _native_step(native)
        _close(state, functional.extract(native).state)
        tensors = tensors._replace(state=state)


def test_material_random_gradients_are_pathwise_and_runtime_noise_is_detached():
    source, native = _population(), _population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    tape = functional.sample_noise_draw_tape(tensors.state, 3)
    tape = tape._replace(
        draws={
            name: value.clone().requires_grad_() for name, value in tape.draws.items()
        }
    )
    final, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        dn.func.RolloutInput(noise=tape),
    )
    rate_name, rate_mu = _parameter(tensors, "random.rate_mu_param")
    _, drive_mu = _parameter(tensors, "random.drive_mu_param")
    loss = _fields(final)["amount"].square().sum()
    gradients = torch.autograd.grad(
        loss, (rate_mu, drive_mu, *tape.draws.values()), allow_unused=True
    )
    assert gradients[0].isfinite().all() and gradients[0].abs().sum() > 0
    assert all(value is None or value.count_nonzero() == 0 for value in gradients[1:])
    for _ in range(3):
        _native_step(native)
    expected = torch.autograd.grad(
        native.mech.materials["pool"].amount.square().sum(),
        native.get_parameter(rate_name),
    )[0]
    torch.testing.assert_close(gradients[0], expected, rtol=2e-10, atol=2e-11)


def test_material_random_explicit_base_draws_support_gradients_jvp_and_vmap():
    source = _population(shared_rng=False)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    packet = packet._replace(
        draws={
            name: value.clone().requires_grad_() for name, value in packet.draws.items()
        }
    )
    rate_name, rate_mu = _parameter(tensors, "random.rate_mu_param")
    baseline = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(random_draw=packet),
        state=tensors.state,
    )
    noise = functional.sample_noise_draw(baseline.state)

    def evaluate(value):
        initialized = functional.initialize(
            {**tensors.parameters, rate_name: value},
            tensors.constants,
            tensors.initialization._replace(random_draw=packet),
            state=tensors.state,
        )
        inputs = dn.func.StepInput(noise=noise)
        state, _ = functional.prepare_and_step(
            initialized.parameters, initialized.constants, initialized.state, inputs
        )
        return _fields(state)["amount"]

    primal, tangent = torch.func.jvp(evaluate, (rate_mu,), (torch.ones_like(rate_mu),))
    finite_difference = (evaluate(rate_mu + 1e-5) - evaluate(rate_mu - 1e-5)) / 2e-5
    torch.testing.assert_close(tangent, finite_difference, rtol=2e-7, atol=2e-8)
    assert tangent.abs().sum() > 0
    mapped = torch.vmap(evaluate)(torch.stack((rate_mu, rate_mu + 0.01)))
    torch.testing.assert_close(mapped[0], primal)
    torch.testing.assert_close(mapped[1], evaluate(rate_mu + 0.01))
    derivatives = torch.autograd.grad(
        primal.sum(), tuple(packet.draws.values()), allow_unused=True
    )
    rand_derivatives = [
        value
        for name, value in zip(packet.draws, derivatives, strict=True)
        if name.startswith("random.draws.")
    ]
    assert rand_derivatives and all(
        value is not None and value.abs().sum() > 0 for value in rand_derivatives
    )
    assert all(
        value is None or value.count_nonzero() == 0
        for name, value in zip(packet.draws, derivatives, strict=True)
        if not name.startswith("random.draws.")
    )


def _network(process=_RandomTransfer):
    cells = {name: _population(process) for name in ("left", "right")}
    for population in cells.values():
        population.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
    stimulus = dn.NetStim(
        N=4, interval=0.25, start=0.0, noise=0.0, max_spikes=20, seed=812
    )
    network = dn.Network(
        cells,
        netstim=stimulus,
        netcon_delay_backend="dense",
        netcon_train_backend="dense",
    )
    for population in cells.values():
        network.connect_one_to_one(
            stimulus[:], population[:], population.mech.syn, weight=0.003, delay=DT
        )
    network.train()
    network.initialize(DT)
    for population in cells.values():
        population.mech.material_processes[
            "random"
        ].stream = population.mech.current.stream
    cells["right"].mech.material_processes["random"].rate_mu_param = (
        cells["left"].mech.material_processes["random"].rate_mu_param
    )
    network.initialize(DT, reinit_weights=False, reinit_delays=False)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=False,
        train_delay_backend="dense",
    )
    return network


def test_material_stochastic_network_shares_rng_modules_between_populations():
    source, native = _network(), _network()
    for network in (source, native):
        network.right.mech.material_processes[
            "random"
        ].stream = network.left.mech.material_processes["random"].stream
        assert (
            network.right.mech.material_processes["random"].stream
            is network.left.mech.current.stream
        )
        assert network.right.mech.current.stream is not network.left.mech.current.stream
    _sync_stimulus(source, native)
    snapshot = _clone(source.state_dict_for_checkpoint(), detached=True)
    global_rng = torch.get_rng_state().clone()
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors, detached=True)
    # Shared streams have one Network-owned carry per physical RNGModule.
    # RIGHT's current retains its independent stream; both material operators
    # consume LEFT's stream in native Population/declaration order.
    assert len(tensors.state["rng"]) == 2
    assert all(
        "rng" not in population["random"]
        for population in tensors.state["populations"].values()
    )
    packets = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    defaults = functional.initialization_inputs
    initialization = defaults._replace(
        populations={
            name: value._replace(random_draw=packets[name])
            for name, value in defaults.populations.items()
        },
        netstim_draw=functional.sample_initialization_draw(
            tensors.parameters, tensors.state, constants=tensors.constants
        ),
        reinit_weights=False,
        reinit_delays=False,
    )
    initialized = functional.initialize(*tensors, initialization)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized.state, functional.extract(native).state)
    tapes = functional.sample_population_noise_tapes(initialized.state, 4)
    _close(tapes, functional.sample_population_noise_tapes(initialized.state, 4))
    inputs = dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(noise=tape) for name, tape in tapes.items()
        },
        netstim=functional.sample_draw_tape(initialized.state, 4),
    )
    prepared = functional.prepare(initialized.parameters, initialized.constants)
    state = initialized.state
    for index in range(4):
        selected = pytree.tree_map_only(
            torch.Tensor, lambda value: value[index : index + 1], inputs
        )
        state, _ = functional.rollout(
            initialized.parameters, prepared, state, selected, steps=1
        )
        native.step()
        _close(state, functional.extract(native).state)
        if index == 1:
            prefix = _clone(state, detached=True)
    continued, _ = functional.rollout(
        initialized.parameters,
        prepared,
        prefix,
        pytree.tree_map_only(torch.Tensor, lambda value: value[2:], inputs),
        steps=2,
    )
    _close(continued, state)
    _close(tensors, frozen)
    _close(source.state_dict_for_checkpoint(), snapshot)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_material_stochastic_network_runner_preserves_recording_restart_and_checkpoint(
    execution,
):
    source, native = _network(), _network()
    _sync_stimulus(source, native)
    runner = dn.compile(
        source,
        dt=DT,
        steps=3,
        initialization="fresh",
        execution=execution,
        inference=execution == "while",
        trainable=("*random.rate_mu_param",) if execution == "scan" else (),
        backend="aot_eager",
        callbacks={"trace": dn.func.Recorder(("left.v", "left.pool_material.amount"))},
        **({"checkpoint_every": 1} if execution == "scan" else {}),
    )
    saved = _clone(runner.functional.extract(), detached=True)
    first, auxiliary = runner()
    _close(runner()[0], first)
    final, _ = runner(
        first, parameters=auxiliary["parameters"], constants=auxiliary["constants"]
    )
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    frames = [native.left.mech.materials["pool"].amount.clone()]
    for _ in range(6):
        native.step()
        frames.append(native.left.mech.materials["pool"].amount.clone())
    _close(final, runner.functional.extract(native).state)
    torch.testing.assert_close(
        auxiliary["callbacks"]["trace"]["left.pool_material.amount"],
        torch.stack(frames[:4]),
    )
    _close(runner.functional.extract(), saved)
    if execution == "scan":
        raw = runner.get_parameter("populations.left.*random.rate_mu_param")
        assert raw is runner.get_parameter("populations.right.*random.rate_mu_param")
        loss = sum(
            _fields(state)["amount"].square().sum()
            for state in final["populations"].values()
        )
        actual = torch.autograd.grad(loss, raw)[0]
        expected = torch.autograd.grad(
            sum(
                pop.mech.materials["pool"].amount.square().sum()
                for pop in native.populations.values()
            ),
            native.left.mech.material_processes["random"].rate_mu_param,
        )[0]
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    runner.functional.commit_state_(source, final)
    restored = _network()
    restored.restore_dict_from_checkpoint(source.state_dict_for_checkpoint())
    continued, _ = runner(
        final, parameters=auxiliary["parameters"], constants=auxiliary["constants"]
    )
    for _ in range(3):
        restored.step()
    _close(continued, runner.functional.extract(restored).state)


@pytest.mark.parametrize("cache_kind", ("population", "population_resample", "network"))
def test_material_stochastic_caches_restore_draws_noise_and_rng_before_continuation(
    cache_kind,
):
    process = _RetainedTransfer if cache_kind == "population" else _RandomTransfer
    source, native = _network(process), _network(process)
    _sync_stimulus(source, native)
    for network in (source, native):
        for _ in range(2):
            network.step()
        if cache_kind.startswith("population"):
            for population in network.populations.values():
                population.cache("_steady_state")
        else:
            network.cache_state()
        for _ in range(2):
            network.step()
    runner = dn.compile(
        source,
        dt=DT,
        steps=3,
        initialization="fresh",
        inference=False,
        trainable=("*random.rate_mu_param",),
        backend="aot_eager",
    )
    actual, auxiliary = runner()
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    if cache_kind.startswith("population"):
        assert all(
            ("_steady_state" in population._caches) == (cache_kind == "population")
            for population in native.populations.values()
        )
    for _ in range(3):
        native.step()
    _close(actual, runner.functional.extract(native).state)
    raw = runner.get_parameter("populations.left.*random.rate_mu_param")
    native_raw = native.left.mech.material_processes["random"].rate_mu_param
    loss = sum(
        _fields(state)["amount"].square().sum()
        for state in actual["populations"].values()
    )
    native_loss = sum(
        population.mech.materials["pool"].amount.square().sum()
        for population in native.populations.values()
    )
    derivative = torch.autograd.grad(loss + 0.0 * raw.sum(), raw)[0]
    native_derivative = torch.autograd.grad(
        native_loss + 0.0 * native_raw.sum(), native_raw
    )[0]
    torch.testing.assert_close(derivative, native_derivative, rtol=2e-10, atol=2e-11)
    final, _ = runner(
        actual, parameters=auxiliary["parameters"], constants=auxiliary["constants"]
    )
    for _ in range(3):
        native.step()
    _close(final, runner.functional.extract(native).state)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_material_stochastic_inductor_fullgraph_uses_shared_rollout(execution):
    source, native = _population(), _population()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        execution=execution,
        inference=execution == "while",
        backend="inductor",
        fullgraph=True,
        callbacks={"trace": dn.func.Recorder(("v", "pool_material.amount"))},
        trainable=("*random.rate_mu_param",) if execution == "scan" else (),
    )
    actual, _ = runner()
    native.initialize()
    for _ in range(4):
        _native_step(native)
    _close(actual, runner.functional.extract(native).state)
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    if execution == "scan":
        derivative = torch.autograd.grad(
            _fields(actual)["amount"].square().sum(),
            runner.get_parameter("*random.rate_mu_param"),
        )[0]
        assert derivative.isfinite().all() and derivative.abs().sum() > 0


def test_material_diffusion_inductor_fullgraph_preserves_held_noise_and_gradients():
    source, native = (
        _diffusion_population(_NoisyThomasDiffusion),
        _diffusion_population(_NoisyThomasDiffusion),
    )
    geometry_path = source.material_geometry("chemical")["volume"]
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        execution="scan",
        inference=False,
        backend="inductor",
        fullgraph=True,
        callbacks={"trace": dn.func.Recorder(("v", "pool_material.amount"))},
        trainable=("*diffusion.decay_param", f"*{geometry_path}"),
    )
    actual, _ = runner()
    native.initialize()
    native.integrator._initialize(native, native.t.new_tensor(DT), force=True)
    for _ in range(4):
        _native_step(native)
    _close(actual, runner.functional.extract(native).state)
    actual_derivatives = torch.autograd.grad(
        _fields(actual)["amount"].square().sum(),
        (
            runner.get_parameter("*diffusion.decay_param"),
            runner.get_parameter(f"*{geometry_path}"),
        ),
    )
    native_derivatives = torch.autograd.grad(
        native.mech.materials["pool"].amount.square().sum(),
        (
            native.mech.material_processes["diffusion"].decay_param,
            native.get_parameter(geometry_path),
        ),
    )
    for derivative, expected in zip(
        actual_derivatives, native_derivatives, strict=True
    ):
        assert derivative.isfinite().all() and derivative.abs().sum() > 0
        torch.testing.assert_close(derivative, expected, rtol=2e-9, atol=2e-10)
