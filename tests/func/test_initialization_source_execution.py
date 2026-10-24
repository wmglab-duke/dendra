"""Parameterized Ion/Material sources share fresh initialization and rollouts."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.parametric import PositiveParam
from tests.func.test_shared_field_initialization import _assert_tree_close

DT = 0.125
STEPS = 4
DTYPE = torch.float64


class _AffineInitializer(torch.nn.Module):
    def __init__(self, value=140.0, *, offset=0.5):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.as_tensor(value, dtype=DTYPE))
        self.register_buffer("offset", torch.as_tensor(offset, dtype=DTYPE))

    def forward(self):
        return self.raw + self.offset


class _InitialConcentration(State):
    State.STATE("nai")
    State.DERIVATIVE("nai' = 0")

    def state_defaults(self, v, values):
        del values
        return {"nai": torch.full_like(v, 5.0)}


class _ConcentrationWriter(Mechanism):
    Mechanism.STATE_BUNDLE(_InitialConcentration)
    Mechanism.USEION("na", write=["nai"])


class _InitialCurrent(State):
    State.STATE("seen")
    State.DERIVATIVE("seen' = 0")

    def initial_values(self, v, values):
        del v
        return {"seen": values["ina"].clone()}


class _CurrentReader(Mechanism):
    Mechanism.STATE_BUNDLE(_InitialCurrent)
    Mechanism.USEION("na", read=["ina"])


class _InitializerLeak(Mechanism):
    Mechanism.GLOBAL(g=0.001)
    Mechanism.USEION("na", read=["ena"], write=["ina"])
    Mechanism.USEMATERIAL("pool", read=["amount"])
    Mechanism.AFFINE("ina")

    def ina_affine_terms(self, v):
        del v
        conductance = self.g + 0.00001 * self.amount
        return conductance, conductance * self.ena


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


def _source(kind="positive", *, vector=False):
    value = torch.tensor([130.0, 140.0, 150.0] if vector else 140.0, dtype=DTYPE)
    if kind == "positive":
        return PositiveParam(value, requires_grad=True)
    return _AffineInitializer(value)


def _population(kind="positive", *, source=None, shape=(2, 3), batch=None):
    source = _source(kind) if source is None else source
    population = dn.SingleCompartment(
        N=shape[0],
        C=shape[1],
        v_init=-60.0,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    population.concentrations(nao0=source)
    population.material(
        "pool",
        fields={"amount": source, "reserve": source},
        min_values={"amount": 0.5},
        conserved=False,
    )
    population.ion_style("na", 1, 1)
    population.insert(_ConcentrationWriter.rename("writer"))
    population.insert(_InitializerLeak.rename("leak"))
    population.insert(_CurrentReader.rename("reader"))
    if batch is not None:
        population.batch(batch)
    population.train()
    with torch.no_grad():
        population.initialize()
    population.detach()
    return population


def _network(kind="positive", *, shared=True):
    source = _source(kind)
    network = dn.Network(
        {
            "left": _population(source=source),
            "right": _population(source=source if shared else _source(kind)),
        }
    )
    network.train()
    with torch.no_grad():
        network.initialize(DT)
    for population in network.populations.values():
        population.detach()
    return network


def _source_name(parameters, *, prefix=""):
    return next(
        name
        for name in parameters
        if name.startswith(prefix) and ".ions.na.o_init." in name
    )


@pytest.mark.cpu
@pytest.mark.parametrize("kind", ("positive", "custom"))
@pytest.mark.parametrize("batch", (None, 2))
def test_fresh_sources_preserve_native_ordering_nernst_and_source_immutability(
    kind, batch
):
    population = _population(kind, source=_source(kind, vector=True), batch=batch)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    before = torch.utils._pytree.tree_map(
        lambda value: value.detach().clone() if torch.is_tensor(value) else value,
        functional.extract(),
    )
    initial = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native = copy.deepcopy(population)
    native.initialize()
    _assert_tree_close(initial.state, functional.extract(native).state)
    _assert_tree_close(functional.extract(), before)
    ions = initial.state["ions"]["na"]
    source = population.mech.ions["na"].o_init()
    torch.testing.assert_close(ions["nao"], source.expand_as(ions["nao"]))
    torch.testing.assert_close(ions["nai"], torch.full_like(ions["nai"], 5))
    expected_reversal = (
        torch.log(ions["nao"] / ions["nai"])
        * population.mech.ions["na"].rzf
        * (273.15 + population.celsius)
    )
    torch.testing.assert_close(ions["ena"], expected_reversal)
    # State initialization reads the provisional current, before the writer's
    # accepted concentration replaces nai and refreshes the final current.
    assert not torch.equal(initial.state["mechanisms"]["reader"]["seen"], ions["ina"])
    for field in ("amount", "reserve"):
        torch.testing.assert_close(
            initial.state["materials"]["pool"][field], source.expand_as(ions["nao"])
        )


@pytest.mark.cpu
@pytest.mark.parametrize("kind", ("positive", "custom"))
def test_initial_source_gradients_jacobians_jvp_vmap_and_fullgraph(kind):
    functional, tensors = dn.func.make_functional(_population(kind), dt=DT)
    name = _source_name(tensors.parameters)
    original = tensors.parameters[name]
    aliases = tuple(
        key for key, value in tensors.parameters.items() if value is original
    )
    # One canonical name binds the shared source Module at all three fields.
    assert len(aliases) == 1

    def evaluate(raw, voltage=None):
        parameters = {**tensors.parameters, **dict.fromkeys(aliases, raw)}
        inputs = (
            tensors.initialization
            if voltage is None
            else tensors.initialization._replace(v_init=voltage)
        )
        initial = functional.initialize(parameters, tensors.constants, inputs)
        final, _ = functional.prepare_and_rollout(
            initial.parameters, initial.constants, initial.state, steps=STEPS
        )
        return torch.cat(
            (
                final["integrator"]["v"].flatten(),
                initial.state["materials"]["pool"]["amount"].flatten(),
            )
        )

    primal = original.detach()
    reverse, forward = (
        torch.func.jacrev(evaluate)(primal),
        torch.func.jacfwd(evaluate)(primal),
    )
    torch.testing.assert_close(reverse, forward)
    assert reverse.abs().sum() > 0
    value, tangent = torch.func.jvp(evaluate, (primal,), (torch.ones_like(primal),))
    torch.testing.assert_close(tangent, reverse)
    lanes = torch.stack((primal - 1, primal + 1))
    torch.testing.assert_close(
        torch.vmap(evaluate)(lanes), torch.stack(tuple(evaluate(raw) for raw in lanes))
    )
    if kind == "positive":
        empty = torch.vmap(evaluate)(lanes[:0])
    else:
        # Bare PyTorch scalar + unbatched scalar Tensor also fails when the
        # scalar batch has zero lanes. Preserve the source's arithmetic and
        # promotion; exercise empty state lanes with a shared source instead.
        voltages = tensors.initialization.v_init.unsqueeze(0)[:0]
        empty = torch.vmap(lambda voltage: evaluate(primal, voltage))(voltages)
    assert empty.shape == (0, value.numel())
    actual = torch.compile(
        torch.func.jacrev(evaluate), backend="aot_eager", fullgraph=True
    )(primal)
    torch.testing.assert_close(actual, reverse)


@pytest.mark.cpu
@pytest.mark.parametrize("kind", ("positive", "custom"))
def test_network_source_aliases_share_independent_training_leaf_and_native_gradients(
    kind,
):
    source, native = _network(kind), _network(kind)
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        backend="aot_eager",
        checkpoint_every=2,
        trainable=("*ions.na.o_init.*",),
    )
    leaf = runner.get_parameter("*left*ions.na.o_init.*")
    assert leaf is runner.get_parameter("*right*ions.na.o_init.*")
    assert source.left.mech.ions["na"].o_init is source.right.mech.materials[
        "pool"
    ].initial_source("reserve")
    final, auxiliary = runner()
    native.initialize(DT)
    for _ in range(STEPS):
        native.step()
    _assert_tree_close(
        final, runner.functional.extract(native).state, rtol=2e-10, atol=2e-11
    )
    loss = sum(
        value["integrator"]["v"].square().sum()
        for value in final["populations"].values()
    )
    actual = torch.autograd.grad(loss, leaf)[0]
    original = next(native.left.mech.ions["na"].o_init.parameters())
    expected = torch.autograd.grad(
        native.left.v.square().sum() + native.right.v.square().sum(), original
    )[0]
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    assert actual.abs().sum() > 0
    assert all(value is not leaf for value in source.parameters())
    assert (
        auxiliary["parameters"]["populations"]["left"][
            _source_name(auxiliary["parameters"]["populations"]["left"])
        ]
        is leaf
    )


@pytest.mark.cpu
@pytest.mark.parametrize("kind", ("positive", "custom"))
def test_packed_multipopulation_initializers_match_native(kind):
    left, right = _population(kind, shape=(1, 2)), _population(kind, shape=(1, 3))
    population = dn.concat_models({"left": left, "right": right})
    population.train()
    with torch.no_grad():
        population.initialize()
    population.detach()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    initial = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native = copy.deepcopy(population)
    native.initialize()
    _assert_tree_close(initial.state, functional.extract(native).state)
    final, _ = functional.prepare_and_rollout(
        initial.parameters, initial.constants, initial.state, steps=STEPS
    )
    for _ in range(STEPS):
        native.step(DT)
    _assert_tree_close(final, functional.extract(native).state, rtol=2e-10, atol=2e-11)


@pytest.mark.cpu
@pytest.mark.parametrize("cache_kind", ("population", "network"))
def test_cached_sources_keep_last_population_write_and_checkpoint_continuation(
    cache_kind,
):
    network = _network("custom")
    with torch.no_grad():
        for _ in range(2):
            network.step()
        network.left.mech.ions["na"].o_init.raw.fill_(120)
        if cache_kind == "population":
            network.left.cache("_steady_state")
        network.right.mech.ions["na"].o_init.raw.fill_(160)
        if cache_kind == "population":
            network.right.cache("_steady_state")
        else:
            network.cache_state()
            for name, saved in (("left", 120), ("right", 160)):
                for key, value in network._state_cache[name].items():
                    if key.endswith(".raw"):
                        value.fill_(saved)
        network.left.mech.ions["na"].o_init.raw.fill_(180)
    native = copy.deepcopy(network)
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        backend="aot_eager",
        trainable=("*ions.na.o_init.raw",),
    )
    prefix, first = runner()
    leaf = first["parameters"]["populations"]["left"][
        _source_name(first["parameters"]["populations"]["left"])
    ]
    torch.testing.assert_close(leaf, torch.full_like(leaf, 160))
    assert (
        leaf
        is first["parameters"]["populations"]["right"][
            _source_name(first["parameters"]["populations"]["right"])
        ]
    )
    torch.testing.assert_close(
        torch.autograd.grad(
            leaf.sum(), runner.get_parameter("*left*ions.na.o_init.raw")
        )[0],
        torch.ones_like(leaf),
    )
    native.initialize(DT)
    for _ in range(STEPS):
        native.step()
    _assert_tree_close(
        prefix, runner.functional.extract(native).state, rtol=2e-10, atol=2e-11
    )
    checkpoint = torch.utils._pytree.tree_map(
        lambda value: value.detach().clone() if torch.is_tensor(value) else value,
        native.state_dict_for_checkpoint(),
    )
    continued, _ = runner(
        prefix, parameters=first["parameters"], constants=first["constants"]
    )
    native.restore_dict_from_checkpoint(checkpoint)
    for _ in range(STEPS):
        native.step()
    _assert_tree_close(
        continued, runner.functional.extract(native).state, rtol=2e-10, atol=2e-11
    )


@pytest.mark.cpu
@pytest.mark.parametrize("execution", ("scan", "while"))
def test_inductor_fresh_sources_record_and_continue_network(execution):
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        execution=execution,
        inference=execution == "while",
        backend="inductor",
        trainable=("*ions.na.o_init.*",) if execution == "scan" else (),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
    )
    prefix, first = runner()
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    native.initialize(DT)
    expected = [native.left.v.clone()]
    for _ in range(2 * STEPS):
        native.step()
        expected.append(native.left.v.clone())
    _assert_tree_close(
        final, runner.functional.extract(native).state, rtol=2e-10, atol=2e-11
    )
    torch.testing.assert_close(
        torch.cat(
            (
                first["callbacks"]["trace"]["left.v"],
                second["callbacks"]["trace"]["left.v"],
            )
        ),
        torch.stack(expected),
    )
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    if execution == "scan":
        gradient = torch.autograd.grad(
            final["populations"]["left"]["integrator"]["v"].square().sum(),
            runner.get_parameter("*left*ions.na.o_init.*"),
        )[0]
        assert gradient.isfinite().all() and gradient.abs().sum() > 0


class _MutatingInitializer(_AffineInitializer):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=DTYPE))

    def forward(self):
        self.count.add_(1)
        return self.raw + self.offset


@pytest.mark.cpu
@dn.ctx(CHECK_PURITY=True)
def test_unsupported_init_only_source_preserves_snapshot_transition_admission():
    source = _population(source=_MutatingInitializer())
    native = copy.deepcopy(source)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    final, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=STEPS
    )
    for _ in range(STEPS):
        native.step(DT)
    _assert_tree_close(final, functional.extract(native).state)
    with pytest.raises(dn.func.FunctionalizationError, match="initial|source|mutat"):
        functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )


@pytest.mark.cpu
@pytest.mark.parametrize("kind", ("positive", "custom"))
def test_evaluation_initialization_detaches_fields_like_native(kind):
    population = _population(kind)
    population.eval()
    population.initialize()
    functional, tensors = dn.func.make_functional(population, dt=DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native = copy.deepcopy(population)
    native.initialize()
    _assert_tree_close(initialized.state, functional.extract(native).state)
    source_fields = (
        initialized.state["ions"]["na"]["nao"],
        *initialized.state["materials"]["pool"].values(),
    )
    assert all(
        not value.requires_grad and value.grad_fn is None for value in source_fields
    )
    # Subsequent pure current assembly can still depend on explicitly prepared
    # mechanism/temperature parameters. The source itself is detached at the
    # native Ion/Material initialization boundary.
    current = initialized.state["ions"]["na"]["ina"]
    if current.requires_grad:
        raw = tensors.parameters[_source_name(tensors.parameters)]
        assert torch.autograd.grad(current.sum(), raw, allow_unused=True)[0] is None


@pytest.mark.cpu
def test_custom_source_buffer_is_an_explicit_shared_constant():
    population = _population("custom")
    functional, tensors = dn.func.make_functional(population, dt=DT)
    names = tuple(name for name in tensors.constants if name.endswith(".offset"))
    assert names
    replacement = next(iter(tensors.constants[name] for name in names)).new_tensor(2.5)
    constants = {**tensors.constants, **dict.fromkeys(names, replacement)}
    initialized = functional.initialize(
        tensors.parameters, constants, tensors.initialization
    )
    for field in ("amount", "reserve"):
        actual = initialized.state["materials"]["pool"][field]
        torch.testing.assert_close(actual, torch.full_like(actual, 142.5))
    torch.testing.assert_close(
        initialized.state["ions"]["na"]["nao"],
        torch.full_like(initialized.state["ions"]["na"]["nao"], 142.5),
    )
    torch.testing.assert_close(
        population.mech.ions["na"].o_init.offset, replacement.new_tensor(0.5)
    )


class _NestedInitializer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = PositiveParam(torch.tensor(140.0, dtype=DTYPE), requires_grad=True)
        self.register_buffer("offset", torch.tensor(0.5, dtype=DTYPE))

    def forward(self):
        return self.value() + self.offset


@pytest.mark.cpu
@pytest.mark.parametrize("training", (False, True))
def test_nested_positive_source_uses_current_leaves_without_mutating_derived_cache(
    training,
):
    source = _NestedInitializer()
    population = _population(source=source)
    population.train(training)
    cached = torch.tensor(999.0, dtype=DTYPE)
    source.value._cache = cached
    functional, tensors = dn.func.make_functional(population, dt=DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    nao = initialized.state["ions"]["na"]["nao"]
    torch.testing.assert_close(nao, torch.full_like(nao, 140.5))
    assert source.value._cache is cached
    assert source.value.training is training
    if training:
        raw = tensors.parameters[_source_name(tensors.parameters)]
        gradient = torch.autograd.grad(nao.sum(), raw)[0]
        assert gradient.isfinite().all() and gradient.abs().sum() > 0


class _BufferAliasInitializer(torch.nn.Module):
    def __init__(self, parameter):
        super().__init__()
        self.register_buffer("binding", parameter)

    def forward(self):
        return self.binding + 140.0


@pytest.mark.cpu
@pytest.mark.parametrize("source_slot", ("parameter", "buffer"))
def test_initializer_leaves_share_mechanism_parameter_identity_and_gradients(
    source_slot,
):
    shared = torch.nn.Parameter(torch.tensor(0.001, dtype=DTYPE))
    if source_slot == "parameter":
        source = _AffineInitializer(0.001, offset=140.0)
        source.raw = shared
    else:
        source = _BufferAliasInitializer(shared)
    population = _population(source=source)
    population.mech.leak.g_param = shared
    with torch.no_grad():
        population.initialize()
    population.detach()
    native = copy.deepcopy(population)
    runner = dn.compile(
        population,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        trainable=("*g_param",),
        backend="aot_eager",
    )
    leaf = runner.get_parameter("*g_param")
    assert leaf is not shared
    assert all(value is not shared for value in runner.constants.values())
    final, _ = runner()
    native.initialize()
    for _ in range(STEPS):
        native.step(DT)
    _assert_tree_close(
        final, runner.functional.extract(native).state, rtol=2e-10, atol=2e-11
    )
    expected = torch.autograd.grad(native.v.square().sum(), native.mech.leak.g_param)[0]
    actual = torch.autograd.grad(final["integrator"]["v"].square().sum(), leaf)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    assert actual.abs().sum() > 0


@pytest.mark.cpu
@pytest.mark.parametrize("runtime_alias", (False, True))
def test_later_population_cache_preserves_prior_initializer_fields_and_runtime_alias_guard(
    runtime_alias,
):
    network = _network("custom")
    source = network.left.mech.ions["na"].o_init
    if runtime_alias:
        network.left.mech.leak.g_param = source.raw
    with torch.no_grad():
        source.raw.fill_(120)
        network.right.cache("_steady_state")
        source.raw.fill_(180)
    native = copy.deepcopy(network)
    functional, tensors = dn.func.make_functional_network(network)
    initialized = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT)
    _assert_tree_close(initialized.state, functional.extract(native).state)
    left = initialized.state["populations"]["left"]
    torch.testing.assert_close(
        left["ions"]["na"]["nao"], torch.full_like(left["ions"]["na"]["nao"], 180.5)
    )
    left_name = (
        "integrator.mech.mechanisms.leak.g_param"
        if runtime_alias
        else _source_name(initialized.parameters["populations"]["left"])
    )
    raw = initialized.parameters["populations"]["left"][left_name]
    torch.testing.assert_close(raw, raw.new_tensor(120))
    assert (
        raw
        is initialized.parameters["populations"]["right"][
            _source_name(initialized.parameters["populations"]["right"])
        ]
    )
    if runtime_alias:
        # LEFT retains the coefficient built before RIGHT restores the shared
        # raw source. Runtime preparation must use 180 even though raw is 120.
        torch.testing.assert_close(
            native.left.mech.leak.g,
            torch.full_like(native.left.mech.leak.g, 180),
        )
    final, _ = functional.prepare_and_rollout(*initialized, steps=STEPS)
    for _ in range(STEPS):
        native.step()
    _assert_tree_close(final, functional.extract(native).state)
    initial_raw = tensors.parameters["populations"]["left"][left_name]
    torch.testing.assert_close(initial_raw, initial_raw.new_tensor(180))
    torch.testing.assert_close(source.raw, source.raw.new_tensor(180))
    actual = torch.autograd.grad(
        sum(
            population["integrator"]["v"].square().sum()
            for population in final["populations"].values()
        ),
        initial_raw,
    )[0]
    expected = torch.autograd.grad(
        sum(population.v.square().sum() for population in native.populations.values()),
        native.left.mech.ions["na"].o_init.raw,
    )[0]
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    assert actual.isfinite().all() and actual.abs().sum() > 0


@pytest.mark.cpu
@pytest.mark.parametrize("alias", ("voltage", "derived_parameter"))
def test_initializer_config_buffer_ownership_preserves_snapshot_and_rejects_fresh(
    alias,
):
    population = _population("custom")
    population.mech.ions["na"].o_init.offset = (
        population.v if alias == "voltage" else population.mech.leak.g
    )
    native = copy.deepcopy(population)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    actual, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=STEPS
    )
    for _ in range(STEPS):
        native.step(DT)
    torch.testing.assert_close(actual["integrator"]["v"], native.v)
    torch.testing.assert_close(actual["clock"]["t"], native.t)
    for name, value in actual["ions"]["na"].items():
        torch.testing.assert_close(value, getattr(native.mech.ions["na"], name))
    for name, value in actual["materials"]["pool"].items():
        torch.testing.assert_close(value, getattr(native.mech.materials["pool"], name))
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="alias runtime state, reconstructed parameters",
    ):
        functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )


@pytest.mark.cpu
def test_cross_population_initializer_buffer_ownership_preserves_snapshot_and_rejects_fresh():
    network = _network("custom", shared=False)
    network.left.mech.ions["na"].o_init.offset = network.right.v
    native = copy.deepcopy(network)
    functional, tensors = dn.func.make_functional_network(network)
    actual, _ = functional.prepare_and_rollout(*tensors, steps=STEPS)
    for _ in range(STEPS):
        native.step()
    torch.testing.assert_close(actual["clock"]["t"], native.t)
    for name, population in native.populations.items():
        state = actual["populations"][name]
        torch.testing.assert_close(state["integrator"]["v"], population.v)
        for field, value in state["ions"]["na"].items():
            torch.testing.assert_close(
                value, getattr(population.mech.ions["na"], field)
            )
        for field, value in state["materials"]["pool"].items():
            torch.testing.assert_close(
                value, getattr(population.mech.materials["pool"], field)
            )
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="alias runtime state, reconstructed parameters",
    ):
        functional.initialize(*tensors, functional.initialization_inputs)


@pytest.mark.cpu
@pytest.mark.parametrize("kind", ("positive", "custom"))
@pytest.mark.parametrize("source_dtype", (torch.float32, torch.float64))
def test_float32_population_initial_sources_preserve_state_dtype_and_raw_gradients(
    kind, source_dtype
):
    value = torch.tensor(140.0, dtype=source_dtype)
    source = (
        PositiveParam(value, requires_grad=True)
        if kind == "positive"
        else _AffineInitializer(value).to(dtype=source_dtype)
    )
    with dn.ctx(DTYPE=torch.float32):
        population = _population(source=source)
        # Population construction aligns registered modules to model dtype.
        # Exercise explicit mixed-dtype sources after that native build phase.
        source.to(dtype=source_dtype)
        native = copy.deepcopy(population)
        functional, tensors = dn.func.make_functional(population, dt=DT)
        initialized = functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )
        actual, _ = functional.prepare_and_rollout(
            initialized.parameters,
            initialized.constants,
            initialized.state,
            steps=STEPS,
        )
        native.initialize()
        for _ in range(STEPS):
            native.step(DT)
    for state in (initialized.state, actual):
        numerical = {name: value for name, value in state.items() if name != "control"}
        assert all(
            value.dtype == torch.float32
            for value in torch.utils._pytree.tree_leaves(numerical)
            if torch.is_tensor(value) and value.is_floating_point()
        )
        # Duration accumulation retains Native's independent precision policy.
        assert (
            state["control"]["duration_remainder"].dtype
            == native._duration_remainder.dtype
        )
    _assert_tree_close(actual, functional.extract(native).state, rtol=5e-5, atol=5e-5)
    raw = tensors.parameters[_source_name(tensors.parameters)]
    observed = torch.autograd.grad(actual["integrator"]["v"].square().sum(), raw)[0]
    native_raw = next(native.mech.ions["na"].o_init.parameters())
    expected = torch.autograd.grad(native.v.square().sum(), native_raw)[0]
    assert raw.dtype == native_raw.dtype and observed.dtype == native_raw.dtype
    assert observed.isfinite().all() and observed.abs().sum() > 0
    torch.testing.assert_close(observed, expected, rtol=5e-5, atol=5e-5)
