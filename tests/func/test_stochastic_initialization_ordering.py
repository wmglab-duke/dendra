"""Root sampling precedes pre transforms; child sampling observes their writes."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import MaterialProcess, Mechanism, State
from tests.func.test_network import _clone, _close

pytestmark = pytest.mark.cpu
DT = 0.125
RAW = "trial_mu_param"


class _Affine(torch.nn.Module):
    def forward(self, value, factor, offset):
        return (factor * value + offset,)


class _RandomPopulation(dn.SingleCompartment):
    dn.SingleCompartment.GLOBALRAND(
        "trial", mu=0.2, sigma=0.01, seed=17, rng_name="stream"
    )
    dn.SingleCompartment.GLOBALNOISE(
        "trial_noise", mu=0.2, sigma=0.02, seed=18, rng_name="stream"
    )


class _StickyPopulation(_RandomPopulation):
    dn.SingleCompartment.GLOBALRAND(
        "trial",
        mu=0.2,
        sigma=0.01,
        seed=17,
        rng_name="stream",
        resample_on_initialize=False,
    )


class _ReseededPopulation(_RandomPopulation):
    dn.SingleCompartment.GLOBALRAND(
        "trial",
        mu=0.2,
        sigma=0.01,
        seed=17,
        rng_name="stream",
        reseed_on_initialize=True,
    )


class _RandomGate(State):
    State.STATE("m")
    State.GLOBALRAND("tau", mu=0.2, sigma=0.03, seed=31, rng_name="stream")
    State.GLOBALNOISE("forcing", mu=0.2, sigma=0.04, seed=32, rng_name="stream")
    State.DERIVATIVE("m' = (forcing - m) / tau")

    def state_defaults(self, v, values):
        del values
        return {"m": torch.full_like(v, 0.5)}


class _RandomLeak(Mechanism):
    Mechanism.STATE_BUNDLE(_RandomGate)
    Mechanism.GLOBALRAND("g", mu=0.2, sigma=0.05, seed=21, rng_name="stream")
    Mechanism.GLOBALNOISE("eta", mu=0.2, sigma=0.06, seed=22, rng_name="stream")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g * self.m, -65 * self.g * self.m + self.eta.clone()


class _RandomTransfer(MaterialProcess):
    MaterialProcess.GLOBALRAND("rate", mu=0.2, sigma=0.07, seed=41, rng_name="stream")
    MaterialProcess.GLOBALNOISE("drive", mu=0.2, sigma=0.08, seed=42, rng_name="stream")
    material_fields = (("pool", "amount"),)

    def prepare_materials(self, fields, dt):
        like = fields[("pool", "amount")]
        return (
            self._resolve_quantity("rate", like).expand_as(like) * dt,
            self._resolve_quantity("drive", like).expand_as(like).clone(),
        )

    def material_updates(self, fields, prepared):
        rate, drive = prepared
        return {("pool", "amount"): fields[("pool", "amount")] + rate + drive}


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


def _population(population_type=_RandomPopulation, *, shift=True):
    population = population_type(
        N=1, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    population.material("pool", fields={"amount": 1.0})
    population.insert(_RandomLeak.rename("leak"))
    population.insert(_RandomTransfer.rename("transfer"))
    population.initialize()
    leak = population.mech.leak
    gate = leak.DE._RandomGate
    process = population.mech.material_processes["transfer"]
    for owner, random_name, noise_name in (
        (population, "trial", "trial_noise"),
        (leak, "g", "eta"),
        (gate, "tau", "forcing"),
        (process, "rate", "drive"),
    ):
        setattr(owner, f"{random_name}_mu_param", population.trial_mu_param)
        setattr(owner, f"{noise_name}_mu_param", population.trial_mu_param)
        owner.stream = population.stream
    if shift:
        population.register_pre_initialize_transform(
            "affine",
            _Affine(),
            reads=(f"parameters.{RAW}",),
            writes=(f"parameters.{RAW}",),
            inputs={
                "factor": torch.tensor(1.5, dtype=torch.float64),
                "offset": torch.tensor(0.1, dtype=torch.float64),
            },
        )
    population.train()
    population.initialize()
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


def _native_close(actual, expected):
    # The pure pre-write distribution frame retains its graph; extracting a
    # native snapshot instead reconnects detached provenance to current inputs.
    # This execution metadata can differ while all values and RNG agree.
    def normalize(left, right):
        if isinstance(right, dict):
            return {
                key: left[key]
                if key.startswith("random.initialization_frame.")
                else normalize(left[key], value)
                for key, value in right.items()
            }
        return right

    _close(actual, normalize(actual, expected))


def _step(population):
    population._step(
        population.integrator, population, population.t.new_tensor(DT), None, None
    )
    population.t = population.t + DT


def _initialize(functional, tensors, packet=None):
    if packet is None:
        packet = functional.sample_initialization_draw(
            tensors.parameters, tensors.state
        )
    return functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(random_draw=packet),
        state=tensors.state,
    )


@pytest.mark.parametrize(
    "population_type", (_RandomPopulation, _StickyPopulation, _ReseededPopulation)
)
def test_fresh_stochastic_owner_order_matches_native_values_and_rng(population_type):
    source, native = _population(population_type), _population(population_type)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    frozen = _clone(tensors, detached=True)
    for _ in range(2):
        tensors = _initialize(functional, tensors)
        native.initialize()
        expected = functional.extract(native)
        _native_close(tensors.state, expected.state)
        _close(tensors.parameters, expected.parameters)
        for name in (
            "trial",
            "integrator.mech.mechanisms.leak.g",
            "integrator.mech.mechanisms.leak.DE._RandomGate.tau",
            "integrator.mech.material_processes.transfer.rate",
        ):
            torch.testing.assert_close(
                tensors.constants[f"random.values.{name}"],
                expected.constants[f"random.values.{name}"],
            )
        assert len(tensors.state["random"]["rng"]) == 1
        tape = functional.sample_noise_draw_tape(tensors.state, 2)
        state, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(noise=tape),
        )
        for _index in range(2):
            _step(native)
        _native_close(state, functional.extract(native).state)
        tensors = tensors._replace(state=state)
    _close(functional.extract(source), frozen)


def test_explicit_draws_distinguish_root_before_write_from_children_after_write():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    packet = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    initialized = _initialize(functional, tensors, packet)
    original = tensors.parameters[RAW]
    transformed = 1.5 * original + 0.1
    for name, sigma in (
        ("trial", 0.01),
        ("integrator.mech.mechanisms.leak.g", 0.05),
        ("integrator.mech.mechanisms.leak.DE._RandomGate.tau", 0.03),
        ("integrator.mech.material_processes.transfer.rate", 0.07),
    ):
        mean = original if name == "trial" else transformed
        expected = mean + sigma * packet.draws[f"random.draws.{name}.0"]
        torch.testing.assert_close(
            initialized.constants[f"random.values.{name}"], expected
        )
        torch.testing.assert_close(
            initialized.constants[f"random.parameters.{name}.mu"], mean
        )
    for name, sigma in (
        ("trial_noise", 0.02),
        ("integrator.mech.mechanisms.leak.eta", 0.06),
        ("integrator.mech.mechanisms.leak.DE._RandomGate.forcing", 0.04),
        ("integrator.mech.material_processes.transfer.drive", 0.08),
    ):
        mean = original if name == "trial_noise" else transformed
        expected = mean + sigma * packet.draws[f"noise.draws.{name}.0"]
        torch.testing.assert_close(initialized.state["runtime_noise"][name], expected)
        assert not initialized.state["runtime_noise"][name].requires_grad
    torch.testing.assert_close(initialized.parameters[RAW], transformed)


def test_stochastic_ordering_preserves_parameter_draw_and_transform_gradients():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    packet = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    packet = packet._replace(
        draws={
            name: value.clone().requires_grad_() for name, value in packet.draws.items()
        }
    )
    factor = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    offset = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    inputs = tensors.initialization._replace(
        random_draw=packet,
        transforms={"pre.affine.factor": factor, "pre.affine.offset": offset},
    )
    initial = functional.initialize(
        tensors.parameters, tensors.constants, inputs, state=tensors.state
    )
    prepared = functional.prepare(initial.parameters, initial.constants).values
    root = prepared["population"]["trial"]
    child = prepared["mechanisms"]["integrator.mech.mechanisms.leak.g"]
    raw = tensors.parameters[RAW]
    derivatives = torch.autograd.grad(
        root + 2 * child,
        (raw, factor, offset, *packet.draws.values()),
        allow_unused=True,
    )
    torch.testing.assert_close(derivatives[0], torch.ones_like(raw) * (1 + 2 * factor))
    torch.testing.assert_close(derivatives[1], 2 * raw)
    torch.testing.assert_close(derivatives[2], offset.new_tensor(2.0))
    for name, derivative in zip(packet.draws, derivatives[3:], strict=True):
        if name == "random.draws.trial.0":
            torch.testing.assert_close(derivative, torch.full_like(derivative, 0.01))
        elif name == "random.draws.integrator.mech.mechanisms.leak.g.0":
            torch.testing.assert_close(derivative, torch.full_like(derivative, 0.1))
        else:
            assert derivative is None or derivative.count_nonzero() == 0
    assert all(
        not value.requires_grad for value in initial.state["runtime_noise"].values()
    )


def test_stochastic_ordering_supports_jvp_vmap_and_fullgraph_initialization():
    functional, tensors = dn.func.make_functional(_population(), dt=DT)
    packet = functional.sample_initialization_draw(tensors.parameters, tensors.state)

    def initialize(raw):
        return functional.initialize(
            {**tensors.parameters, RAW: raw},
            tensors.constants,
            tensors.initialization._replace(random_draw=packet),
            state=tensors.state,
        )

    def values(raw):
        initial = initialize(raw)
        prepared = functional.prepare(initial.parameters, initial.constants).values
        return torch.stack(
            (
                prepared["population"]["trial"],
                prepared["mechanisms"]["integrator.mech.mechanisms.leak.g"],
            )
        )

    raw = tensors.parameters[RAW]
    primal, tangent = torch.func.jvp(values, (raw,), (torch.ones_like(raw),))
    torch.testing.assert_close(tangent, raw.new_tensor([1.0, 1.5]))
    lanes = torch.stack((raw, raw + 0.2))
    torch.testing.assert_close(
        torch.vmap(values)(lanes), torch.stack([values(value) for value in lanes])
    )
    assert torch.vmap(values)(lanes[:0]).shape == (0, 2)
    compiled = torch.compile(initialize, backend="aot_eager", fullgraph=True)
    _close(compiled(raw), initialize(raw))


def test_owned_cache_restores_samples_before_stochastic_ordered_continuation():
    source, native = _population(), _population()
    for population in (source, native):
        population.cache("_steady_state")
        population.initialize()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initial = functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    _native_close(initial.state, functional.extract(native).state)
    noise = functional.sample_noise_draw_tape(initial.state, 2)
    final, _ = functional.prepare_and_rollout(
        initial.parameters,
        initial.constants,
        initial.state,
        dn.func.RolloutInput(noise=noise),
    )
    for _ in range(2):
        _step(native)
    _native_close(final, functional.extract(native).state)


def _network(*, cache_kind=None):
    cells = {name: _population(shift=False) for name in ("left", "right")}
    model = dn.Network(cells)
    shared = cells["left"].trial_mu_param
    for index, population in enumerate(cells.values()):
        leak = population.mech.leak
        for owner, names in (
            (population, ("trial", "trial_noise")),
            (leak, ("g", "eta")),
            (leak.DE._RandomGate, ("tau", "forcing")),
            (population.mech.material_processes["transfer"], ("rate", "drive")),
        ):
            for name in names:
                setattr(owner, f"{name}_mu_param", shared)
        population.register_pre_initialize_transform(
            "affine",
            _Affine(),
            reads=(f"parameters.{RAW}",),
            writes=(f"parameters.{RAW}",),
            inputs={
                "factor": torch.tensor(1.5 + 0.5 * index),
                "offset": torch.tensor(0.1 + 0.1 * index),
            },
        )
    model.train()
    model.initialize(DT)
    if cache_kind == "population":
        model.right.cache("_steady_state")
    elif cache_kind == "network":
        model.cache_state()
    return model


@pytest.mark.parametrize("cache_kind", (None, "population", "network"))
def test_network_stochastic_owner_order_preserves_shared_parameters_caches_and_rng(
    cache_kind,
):
    source, native = _network(cache_kind=cache_kind), _network(cache_kind=cache_kind)
    functional, tensors = dn.func.make_functional_network(source)
    inputs = functional.initialization_inputs
    packets = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    inputs = inputs._replace(
        populations={
            name: value._replace(random_draw=packets[name])
            for name, value in inputs.populations.items()
        }
    )
    initial = functional.initialize(*tensors, inputs)
    native.initialize(DT)
    expected = functional.extract(native)
    _native_close(initial.state, expected.state)
    _close(initial.parameters, expected.parameters)
    assert (
        initial.parameters["populations"]["left"][RAW]
        is initial.parameters["populations"]["right"][RAW]
    )
    if cache_kind is None:
        left = initial.constants["populations"]["left"]
        right = initial.constants["populations"]["right"]
        original = tensors.parameters["populations"]["left"][RAW]
        after_left = 1.5 * original + 0.1
        torch.testing.assert_close(left["random.parameters.trial.mu"], original)
        torch.testing.assert_close(right["random.parameters.trial.mu"], after_left)
        torch.testing.assert_close(
            left["random.parameters.integrator.mech.mechanisms.leak.g.mu"], after_left
        )
    tape = functional.sample_population_noise_tapes(initial.state, 3)
    final, _ = functional.prepare_and_rollout(
        *initial,
        dn.func.NetworkRolloutInput(
            populations={
                name: dn.func.RolloutInput(noise=packet)
                for name, packet in tape.items()
            }
        ),
    )
    for _ in range(3):
        native.step()
    _native_close(final, functional.extract(native).state)
    functional.commit_state_(native, final)
    checkpoint = native.state_dict_for_checkpoint()
    restored = copy.deepcopy(native)
    restored.step()
    restored.restore_dict_from_checkpoint(checkpoint)
    tape = functional.sample_population_noise_tapes(final, 2)
    continued, _ = functional.prepare_and_rollout(
        initial.parameters,
        initial.constants,
        final,
        dn.func.NetworkRolloutInput(
            populations={
                name: dn.func.RolloutInput(noise=packet)
                for name, packet in tape.items()
            }
        ),
    )
    for _ in range(2):
        restored.step()
    _native_close(continued, functional.extract(restored).state)


@pytest.mark.parametrize("initialization", ("fresh", "snapshot"))
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_stochastic_ordering_runner_restart_and_checkpoint_continuation(
    initialization, execution
):
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=2,
        initialization=initialization,
        execution=execution,
        inference=execution == "while",
        backend="aot_eager",
        trainable=("populations.left.trial_mu_param",) if execution == "scan" else (),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.pool_material.amount"))},
        **({"checkpoint_every": 1} if execution == "scan" else {}),
    )
    initial = _clone(runner.functional.extract(), detached=True)
    first, auxiliary = runner()
    _close(runner()[0], first)
    if initialization == "fresh":
        native.initialize(DT)
    for _ in range(2):
        native.step()
    _native_close(first, runner.functional.extract(native).state)
    final, _ = runner(
        first,
        parameters=auxiliary.get("parameters", runner.parameters),
        constants=auxiliary.get("constants", runner.constants),
    )
    for _ in range(2):
        native.step()
    _native_close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), initial)
    if execution == "scan":
        gradient = torch.autograd.grad(
            final["populations"]["left"]["integrator"]["v"].sum(),
            runner.get_parameter("populations.left.trial_mu_param"),
        )[0]
        assert gradient.isfinite().all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_stochastic_ordering_inductor_fullgraph_preserves_native_rollout(execution):
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
        trainable=("trial_mu_param",) if execution == "scan" else (),
    )
    actual, _ = runner()
    native.initialize()
    for _ in range(4):
        _step(native)
    _native_close(actual, runner.functional.extract(native).state)
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    if execution == "scan":
        gradient = torch.autograd.grad(
            actual["integrator"]["v"].sum(), runner.get_parameter("trial_mu_param")
        )[0]
        assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_post_write_preserves_pre_transform_root_sample_and_native_preparation():
    source, native = _population(), _population()
    for population in (source, native):
        population.register_post_initialize_transform(
            "post_affine",
            _Affine(),
            reads=(f"parameters.{RAW}",),
            writes=(f"parameters.{RAW}",),
            inputs={
                "factor": torch.tensor(1.1, dtype=torch.float64),
                "offset": torch.tensor(0.2, dtype=torch.float64),
            },
        )
    functional, tensors = dn.func.make_functional(source, dt=DT)
    packet = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    initial = _initialize(functional, tensors, packet)
    native.initialize()
    prepared = functional.prepare(initial.parameters, initial.constants).values
    torch.testing.assert_close(prepared["population"]["trial"], native.trial)
    torch.testing.assert_close(
        prepared["mechanisms"]["integrator.mech.mechanisms.leak.g"], native.mech.leak.g
    )
    expected_root = (
        tensors.parameters[RAW] + 0.01 * packet.draws["random.draws.trial.0"]
    )
    torch.testing.assert_close(prepared["population"]["trial"], expected_root)
    torch.testing.assert_close(
        torch.autograd.grad(prepared["population"]["trial"], tensors.parameters[RAW])[
            0
        ],
        torch.ones_like(tensors.parameters[RAW]),
    )
    _native_close(initial.state, functional.extract(native).state)


def test_retained_root_cache_skips_sampling_and_pre_transform_writes():
    from dendra.models.mod import pas

    def build():
        population = _StickyPopulation(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
        population.insert(pas)
        population.register_pre_initialize_transform(
            "affine",
            _Affine(),
            reads=(f"parameters.{RAW}",),
            writes=(f"parameters.{RAW}",),
            inputs={
                "factor": torch.tensor(1.5, dtype=torch.float64),
                "offset": torch.tensor(0.1, dtype=torch.float64),
            },
        )
        population.initialize()
        population.integrator._initialize(
            population, population.t.new_tensor(DT), force=True
        )
        population.cache("_steady_state")
        return population

    source, native = build(), build()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    assert (
        functional.sample_initialization_draw(tensors.parameters, tensors.state) is None
    )
    initial = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )
    native.initialize()
    _native_close(initial.state, functional.extract(native).state)
    _close(initial.state["random"]["rng"], tensors.state["random"]["rng"])
    torch.testing.assert_close(initial.parameters[RAW], tensors.parameters[RAW])
    torch.testing.assert_close(initial.constants["random.values.trial"], native.trial)


class _Square(torch.nn.Module):
    def forward(self, value):
        return value.square()


def test_parametrized_root_mean_samples_before_raw_transform():
    from torch.nn.utils.parametrize import register_parametrization

    def build():
        population = _population(shift=False)
        population.trial_mu_param = torch.nn.Parameter(
            population.trial_mu_param.detach().clone()
        )
        register_parametrization(population, "trial_mu_param", _Square())
        raw_reference = "parameters.parametrizations.trial_mu_param.original"
        population.register_pre_initialize_transform(
            "affine",
            _Affine(),
            reads=(raw_reference,),
            writes=(raw_reference,),
            inputs={
                "factor": torch.tensor(1.5, dtype=torch.float64),
                "offset": torch.tensor(0.1, dtype=torch.float64),
            },
        )
        population.initialize()
        population.integrator._initialize(
            population, population.t.new_tensor(DT), force=True
        )
        return population

    source, native = build(), build()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    raw = source.parametrizations.trial_mu_param.original
    raw_name = next(name for name, value in source.named_parameters() if value is raw)
    assert (
        native.mech.leak.g_mu_param
        is not native.parametrizations.trial_mu_param.original
    )
    packet = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    initial = _initialize(functional, tensors, packet)
    prepared = functional.prepare(initial.parameters, initial.constants).values
    original = tensors.parameters[raw_name]
    root = prepared["population"]["trial"]
    child = prepared["mechanisms"]["integrator.mech.mechanisms.leak.g"]
    torch.testing.assert_close(
        root, original.square() + 0.01 * packet.draws["random.draws.trial.0"]
    )
    torch.testing.assert_close(
        child,
        tensors.parameters["trial_noise_mu_param"]
        + 0.05 * packet.draws["random.draws.integrator.mech.mechanisms.leak.g.0"],
    )
    actual = torch.autograd.grad(root + child, original)[0]
    torch.testing.assert_close(actual, 2 * original)
    native.initialize()
    _native_close(initial.state, functional.extract(native).state)
    torch.testing.assert_close(root, native.trial)
    torch.testing.assert_close(child, native.mech.leak.g)
