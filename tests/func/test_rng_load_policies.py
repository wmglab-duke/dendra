"""Cache load policies retain live RNG positions without changing exact resumes."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from dendra.models.rng import RNGModule
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_rng_sharing import (
    PREFIX,
    RAW,
    _initialization,
    _network,
    _rollout_inputs,
    _SharedGate,
    _SharedLeak,
)
from tests.func.test_runtime_noise import _native_step
from tests.func.test_stochastic_initialization_ordering import _native_close

pytestmark = pytest.mark.cpu


class _RetainedGate(_SharedGate):
    State.GLOBALRAND(
        "tau",
        mu=1.0,
        sigma=0.05,
        seed=101,
        rng_name="secondary",
        resample_on_initialize=False,
    )


class _RetainedLeak(Mechanism):
    Mechanism.STATE_BUNDLE(_RetainedGate)
    Mechanism.GLOBALRAND(
        "g",
        mu=0.001,
        sigma=0.00005,
        seed=11,
        rng_name="primary",
        resample_on_initialize=False,
    )
    Mechanism.GLOBALRAND(
        "sticky",
        mu=0.02,
        sigma=0.001,
        seed=12,
        rng_name="secondary",
        resample_on_initialize=False,
    )
    Mechanism.GLOBALRAND(
        "offset",
        mu=0.03,
        sigma=0.001,
        seed=13,
        rng_name="primary",
        resample_on_initialize=False,
    )
    Mechanism.GLOBALNOISE(
        "e", distribution="uniform", low=-68.0, high=-62.0, seed=14, rng_name="primary"
    )
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00002, seed=15, rng_name="secondary")
    Mechanism.RANGENOISE("shift", mu=0.0, sigma=0.00001, seed=16, rng_name="primary")
    Mechanism.CARRY("sample")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    initial_values = _SharedLeak.initial_values
    i_affine_terms = _SharedLeak.i_affine_terms


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _populations(model):
    return model.populations.values() if isinstance(model, dn.Network) else (model,)


def _streams(model):
    return tuple(module for module in model.modules() if type(module) is RNGModule)


def _step(model):
    return model.step() if isinstance(model, dn.Network) else _native_step(model)


def _native_initialize(model):
    if isinstance(model, dn.Network):
        model.initialize(DT)
    else:
        model.initialize()
        model.integrator._initialize(model, model.t.new_tensor(DT), force=True)


def _lower(model):
    return (
        dn.func.make_functional_network(model)
        if isinstance(model, dn.Network)
        else dn.func.make_functional(model, dt=DT)
    )


def _pure_initialize(functional, tensors):
    if hasattr(functional, "populations"):
        return functional.initialize(*tensors, _initialization(functional, tensors))
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    return functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization._replace(random_draw=draw),
        state=tensors.state,
    )


def _host_initialize(functional, tensors):
    if hasattr(functional, "populations"):
        return functional._initialize_host(*tensors, functional.initialization_inputs)
    return functional.initialize_host(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
    )


def _inputs(functional, state, steps):
    if hasattr(functional, "populations"):
        return _rollout_inputs(functional, state, steps)
    return dn.func.RolloutInput(noise=functional.sample_noise_draw_tape(state, steps))


def _cached_model(kind="shared_network", cache_kind="population"):
    cells = {}
    for name in ("left", "right"):
        population = dn.SingleCompartment(
            N=1, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(_RetainedLeak.rename("probe"))
        population.initialize()
        cells[name] = population
    for index, population in enumerate(cells.values()):
        leak = population.mech.probe
        if kind == "shared_network":
            leak.primary = cells["left"].mech.probe.primary
            leak.secondary = cells["left"].mech.probe.secondary
        gate = leak.DE._RetainedGate
        gate.primary = leak.primary
        gate.secondary = leak.secondary
        gate.m_dW_rng = leak.primary
        leak.g_mu_param = cells["left"].mech.probe.g_mu_param
        if index == 0 or kind != "shared_network":
            leak.primary.reseed(1203 + index)
            leak.secondary.reseed(1207 + index)
    model = dn.Network(cells)
    model.train()
    model.initialize(DT)
    if kind == "population":
        model = model.left
    primary = set()
    for population in _populations(model):
        stream = population.mech.probe.primary
        stream.ignore_rng_on_load(True)
        primary.add(id(stream))
        population.mech.probe.secondary.ignore_rng_on_load(False)
    for _ in range(2):
        _step(model)
    if cache_kind == "network":
        model.cache_state()
    else:
        for population in _populations(model):
            population.cache("_steady_state")
    for index, stream in enumerate(_streams(model)):
        stream.reseed((5101 if id(stream) in primary else 6101) + index)
        stream.randn((5,), device="cpu", dtype=DTYPE)
    for _ in range(3):
        _step(model)
    return model


@pytest.mark.parametrize(
    "kind,cache_kind",
    (
        ("population", "population"),
        ("independent_network", "population"),
        ("shared_network", "population"),
        ("independent_network", "network"),
        ("shared_network", "network"),
    ),
)
def test_mixed_cache_rng_policies_match_native_initialization_and_continuation(
    kind, cache_kind
):
    source = _cached_model(kind, cache_kind)
    native = copy.deepcopy(source)
    before = _clone(source.state_dict_for_checkpoint(), detached=True)
    global_rng = torch.get_rng_state().clone()
    aliases = tuple(
        (name, module)
        for name, module in source.named_modules(remove_duplicate=False)
        if type(module) is RNGModule
    )
    functional, tensors = _lower(source)
    if cache_kind == "population":
        plans = (
            functional.populations.values()
            if hasattr(functional, "populations")
            else (functional,)
        )
        assert all(plan._owned_cache_bound for plan in plans)
    initial = _pure_initialize(functional, tensors)
    _close(initial, _pure_initialize(functional, tensors))
    _close(initial, _host_initialize(functional, tensors))
    _native_initialize(native)
    _native_close(initial.state, functional.extract(native).state)
    for population in _populations(native):
        assert population.mech.probe.primary._base_seed >= 5101
        assert population.mech.probe.secondary._base_seed < 6101
    prefix, _ = functional.prepare_and_rollout(
        initial.parameters,
        initial.constants,
        initial.state,
        _inputs(functional, initial.state, 2),
    )
    final, _ = functional.prepare_and_rollout(
        initial.parameters,
        initial.constants,
        prefix,
        _inputs(functional, prefix, 3),
    )
    for _ in range(5):
        _step(native)
    _native_close(final, functional.extract(native).state)
    _close(source.state_dict_for_checkpoint(), before)
    assert all(source.get_submodule(name) is module for name, module in aliases)
    assert all(
        population.mech.probe.primary._ignore_rng_on_load
        for population in _populations(source)
    )
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


def test_ignored_root_mechanism_state_and_material_rng_restore_samples_without_rewind():
    from tests.func.test_stochastic_initialization_ordering import _population

    population = _population()
    source = dn.Network({"cell": population})
    source.train()
    source.initialize(DT)
    population.stream.ignore_rng_on_load(True)
    source.cache_state()
    population.stream.reseed(7303)
    for _ in range(3):
        source.step()
    native = copy.deepcopy(source)
    before = _clone(source.cell.stream.get_extra_state(), detached=True)
    functional, tensors = _lower(source)
    initial = _pure_initialize(functional, tensors)
    _close(initial, _host_initialize(functional, tensors))
    _native_initialize(native)
    _native_close(initial.state, functional.extract(native).state)
    _close(native.cell.stream.get_extra_state(), before)
    for key in (
        "trial",
        "integrator.mech.mechanisms.leak.g",
        "integrator.mech.material_processes.transfer.rate",
    ):
        assert f"random.values.{key}" in initial.constants["populations"]["cell"]
    final, _ = functional.prepare_and_rollout(
        *initial,
        _inputs(functional, initial.state, 2),
    )
    native.step()
    native.step()
    _native_close(final, functional.extract(native).state)


@pytest.mark.parametrize("malformation", ("empty", "boolean_seed", "invalid_bytes"))
@pytest.mark.parametrize("ignore", (True, False))
def test_cache_policy_skips_ignored_rng_payload_but_validates_restored_payload(
    malformation, ignore
):
    source = _cached_model("population")
    stream = source.mech.probe.primary
    stream.ignore_rng_on_load(ignore)
    cache = source._caches["_steady_state"]
    for path, module in source.named_modules(remove_duplicate=False):
        if module is not stream:
            continue
        key = f"{path}._extra_state"
        original = cache[key]
        cache[key] = (
            {}
            if malformation == "empty"
            else {**original, "base_seed": True}
            if malformation == "boolean_seed"
            else {
                **original,
                "rng_state": {"cpu": torch.zeros_like(original["rng_state"]["cpu"])},
            }
        )
    native = copy.deepcopy(source)
    snapshot = _clone(source.state_dict_for_checkpoint(), detached=True)
    global_rng = torch.get_rng_state().clone()
    if ignore:
        functional, tensors = _lower(source)
        initial = _pure_initialize(functional, tensors)
        _native_initialize(native)
        _native_close(initial.state, functional.extract(native).state)
    else:
        with pytest.raises(
            (KeyError, TypeError, ValueError, RuntimeError),
            match="RNG|rng|base_seed|generator|requires",
        ):
            _native_initialize(native)
        with pytest.raises(
            (
                dn.func.FunctionalizationError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ),
            match="RNG|rng|base_seed|generator|requires",
        ):
            functional, tensors = _lower(source)
            _host_initialize(functional, tensors)
    _close(source.state_dict_for_checkpoint(), snapshot)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("ignore", (True, False))
def test_cache_rng_aliases_use_last_registered_saved_payload_unless_ignored(ignore):
    source = _cached_model("shared_network")
    for population in _populations(source):
        population.mech.probe.spare_stream = population.mech.probe.primary
        population.mech.probe.primary.ignore_rng_on_load(ignore)
        population.cache("_steady_state")
        cache = population._caches["_steady_state"]
        for index, (path, module) in enumerate(
            population.named_modules(remove_duplicate=False)
        ):
            if module is population.mech.probe.primary:
                replacement = copy.deepcopy(module)
                replacement.reseed(8001 + index)
                replacement.randn((3,), device="cpu", dtype=DTYPE)
                cache[f"{path}._extra_state"] = replacement.get_extra_state()
    native = copy.deepcopy(source)
    functional, tensors = _lower(source)
    initial = _pure_initialize(functional, tensors)
    _native_initialize(native)
    _native_close(initial.state, functional.extract(native).state)
    final, _ = functional.prepare_and_rollout(
        *initial, _inputs(functional, initial.state, 2)
    )
    native.step()
    native.step()
    _native_close(final, functional.extract(native).state)


@pytest.mark.parametrize(
    "kind", ("population", "independent_network", "shared_network")
)
def test_rng_load_policy_changes_require_new_lowering_and_accept_new_plan(kind):
    source = _cached_model(kind)
    functional, tensors = _lower(source)
    stream = next(iter(_streams(source)))
    stream.ignore_rng_on_load(not stream._ignore_rng_on_load)
    with pytest.raises(
        dn.func.FunctionalizationError, match="RNG|rng|policy|policies|load"
    ):
        functional.extract()
    with pytest.raises(
        dn.func.FunctionalizationError, match="RNG|rng|policy|policies|load"
    ):
        _host_initialize(functional, tensors)
    replacement, values = _lower(source)
    actual = _pure_initialize(replacement, values)
    native = copy.deepcopy(source)
    _native_initialize(native)
    _native_close(actual.state, replacement.extract(native).state)


@pytest.mark.parametrize("kind", ("population", "shared_network"))
def test_exact_runtime_restore_accepts_different_destination_load_policies(kind):
    source = _cached_model(kind)
    functional, tensors = _lower(source)
    initial = _pure_initialize(functional, tensors)
    state, _ = functional.prepare_and_rollout(
        initial.parameters,
        initial.constants,
        initial.state,
        _inputs(functional, initial.state, 2),
    )
    destination = copy.deepcopy(source)
    aliases = tuple(
        (name, module)
        for name, module in destination.named_modules(remove_duplicate=False)
        if type(module) is RNGModule
    )
    for index, stream in enumerate(_streams(destination)):
        stream.ignore_rng_on_load(not stream._ignore_rng_on_load)
        stream.reseed(9001 + index)
    policies = tuple(stream._ignore_rng_on_load for stream in _streams(destination))
    functional.commit_state_(destination, state)
    _native_close(functional.extract(destination).state, state)
    checkpoint = _clone(destination.state_dict_for_checkpoint(), detached=True)
    for _ in range(2):
        _step(destination)
    destination.restore_dict_from_checkpoint(checkpoint)
    _native_close(functional.extract(destination).state, state)
    assert policies == tuple(
        stream._ignore_rng_on_load for stream in _streams(destination)
    )
    assert all(destination.get_submodule(name) is module for name, module in aliases)
    final, _ = functional.prepare_and_rollout(
        initial.parameters, initial.constants, state, _inputs(functional, state, 2)
    )
    _step(destination)
    _step(destination)
    _native_close(final, functional.extract(destination).state)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_cached_rng_load_policies_keep_compiled_rollout_and_parameter_anchors(
    execution,
):
    source = _cached_model("shared_network", "network")
    native = copy.deepcopy(source)
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        initialization="fresh",
        inference=execution == "while",
        execution=execution,
        backend="aot_eager",
        trainable=(
            f"populations.*.{RAW}",
            f"populations.*.{PREFIX}DE._RetainedGate.sigma_param",
        )
        if execution == "scan"
        else (),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v"))},
        **({"checkpoint_every": 2} if execution == "scan" else {}),
    )
    frozen = _clone(runner.functional.extract(), detached=True)
    prefix, first = runner()
    _close(runner()[0], prefix)
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    _native_initialize(native)
    expected = {
        name: [population.v.clone()] for name, population in native.populations.items()
    }
    for _ in range(8):
        native.step()
        for name, population in native.populations.items():
            expected[name].append(population.v.clone())
    for name, values in expected.items():
        trace = torch.cat(
            (
                first["callbacks"]["trace"][f"{name}.v"],
                second["callbacks"]["trace"][f"{name}.v"],
            )
        )
        torch.testing.assert_close(trace, torch.stack(values))
    _native_close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), frozen)
    if execution == "scan":
        raw = runner.get_parameter(f"populations.left.{RAW}")
        sigma = runner.get_parameter(
            f"populations.left.{PREFIX}DE._RetainedGate.sigma_param"
        )
        loss = sum(
            state["integrator"]["v"].square().sum()
            for state in final["populations"].values()
        )
        actual = torch.autograd.grad(
            loss, (raw, sigma), allow_unused=True, retain_graph=True
        )
        native_loss = sum(
            population.v.square().sum() for population in native.populations.values()
        )
        # Native cache restoration detaches the materialized g/sigma buffers;
        # dynamics retain that boundary while returned raw sources keep their
        # incoming Parameter anchors and aliases.
        assert not native_loss.requires_grad
        assert all(value is None or value.count_nonzero() == 0 for value in actual)
        returned = first["parameters"]["populations"]
        assert returned["left"][RAW] is returned["right"][RAW]
        anchor = returned["left"][RAW].sum() + returned["right"][RAW].sum()
        torch.testing.assert_close(
            torch.autograd.grad(anchor, raw)[0], 2 * torch.ones_like(raw)
        )
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    source.left.mech.probe.primary.ignore_rng_on_load(False)
    with pytest.raises(
        dn.func.FunctionalizationError, match="RNG|rng|policy|policies|load"
    ):
        runner()


def test_reseeded_fresh_initialization_uses_seed_retained_by_ignored_model_load():
    source = _network(reseed=True).left
    saved = _clone(source.state_dict(), detached=True)
    source.mech.probe.primary.ignore_rng_on_load(True)
    source.mech.probe.primary.reseed(7403)
    source.mech.probe.primary.randn((5,), device="cpu", dtype=DTYPE)
    retained = _clone(source.mech.probe.primary.get_extra_state(), detached=True)
    source.load(saved)
    _close(source.mech.probe.primary.get_extra_state(), retained)
    native = copy.deepcopy(source)
    functional, tensors = _lower(source)
    first_sample = None
    for _ in range(2):
        tensors = _pure_initialize(functional, tensors)
        _native_initialize(native)
        _native_close(tensors.state, functional.extract(native).state)
        sample = tensors.constants[f"random.values.{PREFIX}g"]
        if first_sample is None:
            first_sample = sample
        else:
            torch.testing.assert_close(sample, first_sample)
        assert native.mech.probe.primary._base_seed == 7403


@pytest.mark.parametrize("consumer", ("noise_only", "sde_only"))
def test_ignored_cache_rng_supports_stochastic_owners_without_rand_declarations(
    consumer,
):
    if consumer == "noise_only":
        from tests.func.test_runtime_noise import _population

        source = _population()
    else:
        from tests.func.test_state_sde_execution import _population

        source = _population()
    for index, stream in enumerate(_streams(source)):
        stream.ignore_rng_on_load(index % 2 == 0)
    source.cache("_steady_state")
    for index, stream in enumerate(_streams(source)):
        stream.reseed(7601 + index)
    for _ in range(3):
        _native_step(source)
    native = copy.deepcopy(source)
    functional, tensors = _lower(source)
    assert functional._random_plan is not None
    assert not functional._random_plan.entries
    initial = _pure_initialize(functional, tensors)
    _close(initial, _host_initialize(functional, tensors))
    _native_initialize(native)
    _native_close(initial.state, functional.extract(native).state)
    final, _ = functional.prepare_and_rollout(
        initial.parameters,
        initial.constants,
        initial.state,
        _inputs(functional, initial.state, 3),
    )
    for _ in range(3):
        _native_step(native)
    _native_close(final, functional.extract(native).state)


def test_owned_cache_defaults_use_snapshot_rng_and_explicit_state_selects_current_rng():
    source = _cached_model("population")
    functional, tensors = _lower(source)
    expected = _pure_initialize(functional, tensors)
    default = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    host_default = functional.initialize_host(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    _close(default, expected)
    _close(host_default, expected)
    native = copy.deepcopy(source)
    native.mech.probe.primary.reseed(9703)
    native.mech.probe.primary.randn((4,), device="cpu", dtype=DTYPE)
    advanced = functional.extract(native).state
    selected = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization, state=advanced
    )
    _native_initialize(native)
    _native_close(selected.state, functional.extract(native).state)
    assert native.mech.probe.primary._base_seed == 9703
    assert any(
        not torch.equal(selected.state["random"]["rng"][name]["seed"], values["seed"])
        for name, values in expected.state["random"]["rng"].items()
    )
