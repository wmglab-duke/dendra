"""Network stream ownership follows native Population and declaration order."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism, State
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_stochastic_initialization_ordering import _Affine, _native_close

pytestmark = pytest.mark.cpu
PREFIX = "integrator.mech.mechanisms.probe."
RAW = PREFIX + "g_mu_param"


class _SharedGate(State):
    State.STATE("m")
    State.GLOBALRAND("tau", mu=1.0, sigma=0.05, seed=101, rng_name="secondary")
    State.RANGENOISE("forcing", mu=0.0, sigma=0.01, seed=102, rng_name="primary")
    State.RANGE(sigma=0.04)
    State.DERIVATIVE("m' = (0.4 - m + forcing) / tau")
    State.DIFFUSION("m = sigma")
    State.METHOD("euler_maruyama")

    def state_defaults(self, v, values):
        del values
        return {"m": torch.full_like(v, 0.4)}


class _SharedLeak(Mechanism):
    Mechanism.STATE_BUNDLE(_SharedGate)
    # Each declaration family visits primary, secondary, primary. Grouping
    # requests by owner or stream would change the native generator suffix.
    Mechanism.GLOBALRAND("g", mu=0.001, sigma=0.00005, seed=11, rng_name="primary")
    Mechanism.GLOBALRAND(
        "sticky",
        mu=0.02,
        sigma=0.001,
        seed=12,
        rng_name="secondary",
        resample_on_initialize=False,
    )
    Mechanism.GLOBALRAND("offset", mu=0.03, sigma=0.001, seed=13, rng_name="primary")
    Mechanism.GLOBALNOISE(
        "e", distribution="uniform", low=-68.0, high=-62.0, seed=14, rng_name="primary"
    )
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00002, seed=15, rng_name="secondary")
    Mechanism.RANGENOISE("shift", mu=0.0, sigma=0.00001, seed=16, rng_name="primary")
    Mechanism.CARRY("sample")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def initial_values(self, v, values):
        del values
        return {"sample": torch.zeros_like(v) + self.g + self.sticky + self.offset}

    def i_affine_terms(self, v):
        del v
        conductance = self.g * self.m
        return (
            conductance,
            conductance * self.e.clone() - self.eta.clone() - self.shift.clone(),
        )


class _ReseededLeak(_SharedLeak):
    Mechanism.GLOBALRAND(
        "g",
        mu=0.001,
        sigma=0.00005,
        seed=11,
        rng_name="primary",
        reseed_on_initialize=True,
    )
    Mechanism.GLOBALRAND(
        "offset",
        mu=0.03,
        sigma=0.001,
        seed=13,
        rng_name="primary",
        reseed_on_initialize=True,
    )


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


def _network(*, shared=True, reseed=False, prewrites=False, cache_kind=None):
    cells = {}
    for name in ("left", "right"):
        cell = dn.SingleCompartment(
            N=1, C=2, v_init=-45.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert((_ReseededLeak if reseed else _SharedLeak).rename("probe"))
        cell.initialize()
        cells[name] = cell
    for index, cell in enumerate(cells.values()):
        leak = cell.mech.probe
        if shared:
            leak.primary = cells["left"].mech.probe.primary
            leak.secondary = cells["left"].mech.probe.secondary
        gate = leak.DE._SharedGate
        gate.primary = leak.primary
        gate.secondary = leak.secondary
        gate.m_dW_rng = leak.primary
        leak.g_mu_param = cells["left"].mech.probe.g_mu_param
        if prewrites:
            cell.register_pre_initialize_transform(
                "affine",
                _Affine(),
                reads=(f"parameters.{RAW}",),
                writes=(f"parameters.{RAW}",),
                inputs={
                    "factor": torch.tensor(1.1 + 0.1 * index, dtype=DTYPE),
                    "offset": torch.tensor(0.0001, dtype=DTYPE),
                },
            )
    model = dn.Network(cells)
    model.train()
    for index, cell in enumerate(cells.values()):
        if index == 0 or not shared:
            cell.mech.probe.primary.reseed(1203 + index)
            cell.mech.probe.secondary.reseed(1207 + index)
    model.initialize(DT)
    if cache_kind:
        with torch.no_grad():
            model.step()
            model.step()
            if cache_kind in ("left", "both"):
                model.left.cache("_steady_state")
            model.step()
            if cache_kind in ("right", "both"):
                model.right.cache("_steady_state")
            if cache_kind == "network":
                model.cache_state()
            model.step()
            model.step()
    return model


def _initialization(functional, tensors):
    packets = functional.sample_population_initialization_draws(
        tensors.parameters, tensors.state
    )
    inputs = functional.initialization_inputs
    return inputs._replace(
        populations={
            name: value._replace(random_draw=packets[name])
            for name, value in inputs.populations.items()
        }
    )


def _rollout_inputs(functional, state, steps):
    return dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(noise=tape)
            for name, tape in functional.sample_population_noise_tapes(
                state, steps
            ).items()
            if tape is not None
        }
    )


def _slice(value, start, stop):
    return pytree.tree_map_only(torch.Tensor, lambda tensor: tensor[start:stop], value)


def _assert_aliases(model):
    left, right = model.left.mech.probe, model.right.mech.probe
    assert left.primary is right.primary
    assert left.secondary is right.secondary
    assert left.DE._SharedGate.primary is left.primary
    assert right.DE._SharedGate.m_dW_rng is left.primary
    assert left.primary is not left.secondary


def test_shared_network_has_one_carry_per_physical_stream_without_source_mutation():
    source = _network()
    snapshot = _clone(source.state_dict_for_checkpoint(), detached=True)
    global_rng = torch.get_rng_state().clone()
    functional, tensors = dn.func.make_functional_network(source)
    assert len(tensors.state["rng"]) == 2
    for state in tensors.state["populations"].values():
        assert "rng" not in state["random"]
    assert all(
        set(value) == {"seed", "negative", "cpu"}
        for value in tensors.state["rng"].values()
    )
    inputs = _rollout_inputs(functional, tensors.state, 4)
    _close(inputs, _rollout_inputs(functional, tensors.state, 4))
    _close(source.state_dict_for_checkpoint(), snapshot)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    _assert_aliases(source)
    standalone, packet = dn.func.make_functional(source.left, dt=DT)
    assert "rng" in packet.state["random"]
    assert len(packet.state["random"]["rng"]) == 2
    assert standalone is not None


def test_independent_network_streams_keep_population_owned_carry():
    functional, tensors = dn.func.make_functional_network(_network(shared=False))
    assert "rng" not in tensors.state
    assert all(
        len(state["random"]["rng"]) == 2
        for state in tensors.state["populations"].values()
    )
    assert functional is not None


@pytest.mark.parametrize("reseed", (False, True))
@pytest.mark.parametrize("prewrites", (False, True))
def test_shared_initialization_orders_rand_noise_reseeding_and_shared_writes(
    reseed, prewrites
):
    source = _network(reseed=reseed, prewrites=prewrites)
    native = _network(reseed=reseed, prewrites=prewrites)
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(tensors, detached=True)
    global_rng = torch.get_rng_state().clone()
    for _ in range(2):
        inputs = _initialization(functional, tensors)
        _close(inputs, _initialization(functional, tensors))
        tensors = functional.initialize(*tensors, inputs)
        native.initialize(DT)
        expected = functional.extract(native)
        _native_close(tensors.state, expected.state)
        _close(tensors.parameters, expected.parameters)
        for name in ("left", "right"):
            constants = tensors.constants["populations"][name]
            reference = expected.constants["populations"][name]
            for key in constants:
                if key.startswith("random.values."):
                    _close(constants[key], reference[key])
        assert (
            tensors.parameters["populations"]["left"][RAW]
            is tensors.parameters["populations"]["right"][RAW]
        )
        state, _ = functional.prepare_and_rollout(
            *tensors, _rollout_inputs(functional, tensors.state, 2)
        )
        native.step()
        native.step()
        _native_close(state, functional.extract(native).state)
        tensors = tensors._replace(state=state)
    _close(functional.extract(source), frozen)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("execution", ("eager", "scan"))
def test_shared_noise_sde_tapes_interleave_populations_and_continue_chunks(execution):
    source, native = _network(), _network()
    functional, tensors = dn.func.make_functional_network(source)
    inputs = _rollout_inputs(functional, tensors.state, 6)
    frozen = _clone(inputs, detached=True)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("left.v", "right.v", "t"))}
    )
    execute = (
        functional.rollout
        if execution == "eager"
        else functional.compile_rollout_chunk(3, backend="aot_eager")
    )
    options = {"steps": 3} if execution == "eager" else {}
    prefix, first = execute(
        tensors.parameters,
        prepared,
        tensors.state,
        _slice(inputs, 0, 3),
        callbacks=callbacks,
        **options,
    )
    final, second = execute(
        tensors.parameters,
        prepared,
        prefix,
        _slice(inputs, 3, 6),
        callbacks=callbacks,
        callback_state=first["callbacks"].state,
        **options,
    )
    with torch.no_grad():
        traces = {name: [cell.v.clone()] for name, cell in native.populations.items()}
        for _ in range(6):
            native.step()
            for name, cell in native.populations.items():
                traces[name].append(cell.v.clone())
    for name, trace in traces.items():
        actual = torch.cat(
            (
                first["callbacks"]["trace"][f"{name}.v"],
                second["callbacks"]["trace"][f"{name}.v"],
            )
        )
        torch.testing.assert_close(actual, torch.stack(trace))
    _close(final, functional.extract(native).state)
    _close(inputs, frozen)
    continued = _rollout_inputs(functional, prefix, 3)
    _close(continued, _slice(inputs, 3, 6))
    zero = functional.sample_population_noise_tapes(final, 0)
    assert all(
        value.shape[0] == 0
        for packet in zero.values()
        for value in pytree.tree_leaves(packet)
    )


@pytest.mark.parametrize("cache_kind", ("left", "right", "both", "network"))
def test_shared_rng_cache_restore_uses_native_population_publication_order(cache_kind):
    source, native = _network(cache_kind=cache_kind), _network(cache_kind=cache_kind)
    functional, tensors = dn.func.make_functional_network(source)
    frozen = _clone(functional.extract(), detached=True)
    initial = functional.initialize(*tensors, _initialization(functional, tensors))
    native.initialize(DT)
    _native_close(initial.state, functional.extract(native).state)
    final, _ = functional.prepare_and_rollout(
        *initial, _rollout_inputs(functional, initial.state, 3)
    )
    for _ in range(3):
        native.step()
    _native_close(final, functional.extract(native).state)
    _close(functional.extract(), frozen)


def test_shared_rng_checkpoint_commit_preserves_aliases_and_native_continuation():
    source, native = _network(), _network()
    functional, tensors = dn.func.make_functional_network(source)
    state, _ = functional.prepare_and_rollout(
        *tensors, _rollout_inputs(functional, tensors.state, 3)
    )
    functional.commit_state_(native, state)
    _assert_aliases(native)
    _close(functional.extract(native).state, state)
    restored = copy.deepcopy(native)
    checkpoint = _clone(native.state_dict_for_checkpoint(), detached=True)
    restored.step()
    restored.restore_dict_from_checkpoint(checkpoint)
    _assert_aliases(restored)
    final, _ = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        state,
        _rollout_inputs(functional, state, 2),
    )
    restored.step()
    restored.step()
    _close(final, functional.extract(restored).state)


@pytest.mark.parametrize("invalid", ("missing", "shape", "generator"))
def test_shared_rng_invalid_checkpoint_rejects_before_native_publication(invalid):
    source = _network()
    functional, tensors = dn.func.make_functional_network(source)
    saved = _clone(source.state_dict_for_checkpoint(), detached=True)
    global_rng = torch.get_rng_state().clone()
    rng = _clone(tensors.state["rng"], detached=True)
    stream = next(iter(rng))
    if invalid == "missing":
        del rng[stream]
    elif invalid == "shape":
        key = next(iter(rng[stream]))
        rng[stream][key] = rng[stream][key][:-1]
    else:
        rng[stream] = {
            name: torch.zeros_like(value) for name, value in rng[stream].items()
        }
    with pytest.raises(
        (ValueError, RuntimeError), match="[Rr][Nn][Gg]|[Ss]tate|schema|generator"
    ):
        functional.commit_state_(source, {**tensors.state, "rng": rng})
    _close(source.state_dict_for_checkpoint(), saved)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    _assert_aliases(source)


def test_shared_rng_alias_changes_reject_before_extraction_or_checkpoint_commit():
    source = _network()
    functional, tensors = dn.func.make_functional_network(source)
    source.right.mech.probe.primary = copy.deepcopy(source.right.mech.probe.primary)
    saved = _clone(source.state_dict_for_checkpoint(), detached=True)
    with pytest.raises(
        dn.func.FunctionalizationError, match="RNG|rng|alias|sharing|structure"
    ):
        functional.extract(source)
    with pytest.raises(
        dn.func.FunctionalizationError, match="RNG|rng|alias|sharing|structure"
    ):
        functional.commit_state_(source, tensors.state)
    _close(source.state_dict_for_checkpoint(), saved)


@pytest.mark.parametrize("checkpoint_every", (None, 2))
def test_shared_rng_gradients_preserve_rand_and_sde_paths_and_detach_noise(
    checkpoint_every,
):
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        inference=False,
        trainable=(
            f"populations.*.{RAW}",
            "*eta_mu_param",
            f"populations.*.{PREFIX}DE._SharedGate.sigma_param",
        ),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v"))},
        backend="aot_eager",
        checkpoint_every=checkpoint_every,
    )
    inputs = _rollout_inputs(runner.functional, runner.initial_state, 4)
    packet = inputs.populations["left"].noise
    draw_name = next(name for name in packet.draws if name.startswith("noise.draws."))
    sde_name = next(name for name in packet.draws if name.startswith("sde.draws."))
    noise_draw = packet.draws[draw_name].requires_grad_()
    sde_draw = packet.draws[sde_name].requires_grad_()
    raw = runner.get_parameter(f"populations.left.{RAW}")
    eta_mu = runner.get_parameter("populations.left.*eta_mu_param")
    sigma = runner.get_parameter(f"populations.left.{PREFIX}DE._SharedGate.sigma_param")
    final, auxiliary = runner(inputs=inputs)
    loss = sum(
        auxiliary["callbacks"]["trace"][f"{name}.v"].square().mean()
        for name in ("left", "right")
    )
    actual = torch.autograd.grad(
        loss, (raw, eta_mu, sigma, noise_draw, sde_draw), allow_unused=True
    )
    assert actual[0].abs().sum() > 0
    assert actual[2].abs().sum() > 0
    assert all(
        value is None or value.count_nonzero() == 0
        for value in (actual[1], actual[3], actual[4])
    )
    traces = {name: [cell.v.clone()] for name, cell in native.populations.items()}
    for _ in range(4):
        native.step()
        for name, cell in native.populations.items():
            traces[name].append(cell.v.clone())
    native_loss = sum(torch.stack(trace).square().mean() for trace in traces.values())
    expected = torch.autograd.grad(
        native_loss,
        (
            native.left.mech.probe.g_mu_param,
            native.left.mech.probe.DE._SharedGate.sigma_param,
        ),
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[2], expected[1])
    _close(final, runner.functional.extract(native).state)


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_shared_rng_inductor_recording_restart_and_explicit_continuation(execution):
    source, native = _network(), _network()
    runner = dn.compile(
        source,
        dt=DT,
        steps=4,
        execution=execution,
        inference=execution == "while",
        trainable=(f"populations.left.{RAW}",) if execution == "scan" else (),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
        backend="inductor",
        fullgraph=True,
    )
    frozen = _clone(runner.functional.extract(), detached=True)
    prefix, first = runner()
    _close(runner()[0], prefix)
    final, second = runner(prefix, callback_state=first["callbacks"].state)
    with torch.no_grad():
        traces = {name: [cell.v.clone()] for name, cell in native.populations.items()}
        for _ in range(8):
            native.step()
            for name, cell in native.populations.items():
                traces[name].append(cell.v.clone())
    for name, trace in traces.items():
        actual = torch.cat(
            (
                first["callbacks"]["trace"][f"{name}.v"],
                second["callbacks"]["trace"][f"{name}.v"],
            )
        )
        torch.testing.assert_close(actual, torch.stack(trace))
    _close(final, runner.functional.extract(native).state)
    _close(runner.functional.extract(), frozen)
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    if execution == "scan":
        gradient = torch.autograd.grad(
            final["populations"]["left"]["integrator"]["v"].sum(),
            runner.get_parameter(f"populations.left.{RAW}"),
        )[0]
        assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_shared_population_root_material_and_state_consumers_keep_initialization_order():
    from tests.func.test_stochastic_initialization_ordering import (
        _network as ordered_network,
    )

    def build():
        model = ordered_network()
        stream = model.left.stream
        for population in model.populations.values():
            population.stream = stream
            population.mech.leak.stream = stream
            population.mech.leak.DE._RandomGate.stream = stream
            population.mech.material_processes["transfer"].stream = stream
        model.initialize(DT)
        return model

    source, native = build(), build()
    functional, tensors = dn.func.make_functional_network(source)
    assert len(tensors.state["rng"]) == 1
    initial = functional.initialize(*tensors, _initialization(functional, tensors))
    native.initialize(DT)
    _native_close(initial.state, functional.extract(native).state)
    _close(initial.parameters, functional.extract(native).parameters)
    final, _ = functional.prepare_and_rollout(
        *initial, _rollout_inputs(functional, initial.state, 3)
    )
    for _ in range(3):
        native.step()
    _native_close(final, functional.extract(native).state)


@pytest.mark.parametrize("cache_kind", ("population", "network"))
def test_shared_rng_cache_last_registered_alias_controls_restored_suffix(cache_kind):
    from dendra.models.rng import RNGModule

    def build():
        model = _network()
        for population in model.populations.values():
            population.mech.probe.spare_stream = population.mech.probe.primary
        if cache_kind == "network":
            model.cache_state()
            caches = model._state_cache
        else:
            for population in model.populations.values():
                population.cache("_steady_state")
            caches = {
                name: population._caches["_steady_state"]
                for name, population in model.populations.items()
            }
        # A checkpoint may contain distinct occurrences of a shared module.
        # Native state_dict loading visits every registered alias; its last
        # occurrence, including a non-declaration alias, wins for that stream.
        for population_index, (name, population) in enumerate(
            model.populations.items()
        ):
            for alias_index, (path, module) in enumerate(
                population.named_modules(remove_duplicate=False)
            ):
                if isinstance(module, RNGModule):
                    replacement = copy.deepcopy(module)
                    replacement.reseed(9000 + 100 * population_index + alias_index)
                    replacement.randn((3,), dtype=DTYPE, device="cpu")
                    caches[name][f"{path}._extra_state"] = replacement.get_extra_state()
        return model

    source, native = build(), build()
    functional, tensors = dn.func.make_functional_network(source)
    initial = functional.initialize(*tensors, _initialization(functional, tensors))
    native.initialize(DT)
    _native_close(initial.state, functional.extract(native).state)
    final, _ = functional.prepare_and_rollout(
        *initial, _rollout_inputs(functional, initial.state, 2)
    )
    native.step()
    native.step()
    _native_close(final, functional.extract(native).state)


@pytest.mark.parametrize("provided_population", ("left", "right"))
def test_shared_rng_host_fills_missing_population_tape_in_native_order(
    provided_population,
):
    runner = dn.compile(
        _network(),
        dt=DT,
        steps=3,
        inference=False,
        backend="aot_eager",
    )
    inputs = _rollout_inputs(runner.functional, runner.initial_state, 3)
    expected, _ = runner(inputs=inputs)
    partial = inputs._replace(
        populations={provided_population: inputs.populations[provided_population]}
    )
    actual, _ = runner(inputs=partial)
    _close(actual, expected)


def test_shared_rng_pure_initialization_compiles_fullgraph_without_source_mutation():
    source = _network(reseed=True, prewrites=True)
    functional, tensors = dn.func.make_functional_network(source)
    inputs = _initialization(functional, tensors)
    expected = functional.initialize(*tensors, inputs)
    frozen = _clone(functional.extract(), detached=True)
    compiled = torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
    actual = compiled(*tensors, inputs)
    _close(actual, expected)
    _close(functional.extract(), frozen)


@pytest.mark.parametrize(
    "order",
    (
        ("quiet", "left"),
        ("left", "quiet"),
        ("quiet", "left", "right"),
        ("left", "quiet", "right"),
        ("left", "right", "quiet"),
    ),
)
@pytest.mark.parametrize("cache_kind", ("population", "network"))
def test_quiet_population_cache_can_restore_another_populations_shared_stream(
    order, cache_kind
):
    from dendra.models.mod import pas

    def build():
        owners = _network()
        quiet = dn.SingleCompartment(N=1, C=2, integrator=dn.bwd_euler_sc(imem=False))
        quiet.insert(pas)
        quiet.initialize()
        quiet.shared_stream = owners.left.mech.probe.primary
        cells = {"left": owners.left, "right": owners.right, "quiet": quiet}
        model = dn.Network({name: cells[name] for name in order})
        model.train()
        model.initialize(DT)
        model.step()
        model.step()
        if cache_kind == "network":
            model.cache_state()
            cache = model._state_cache["quiet"]
        else:
            quiet.cache("_steady_state")
            cache = quiet._caches["_steady_state"]
        saved_stream = copy.deepcopy(quiet.shared_stream)
        saved_stream.reseed(4703)
        saved_stream.randn((5,), device="cpu", dtype=DTYPE)
        cache["shared_stream._extra_state"] = saved_stream.get_extra_state()
        model.step()
        return model

    source, native = build(), build()
    functional, tensors = dn.func.make_functional_network(source)
    assert "random" not in tensors.state["populations"]["quiet"]
    inputs = _initialization(functional, tensors)
    assert inputs.populations["quiet"].random_draw is None
    initial = functional.initialize(*tensors, inputs)
    native.initialize(DT)
    _native_close(initial.state, functional.extract(native).state)
    final, _ = functional.prepare_and_rollout(
        *initial, _rollout_inputs(functional, initial.state, 3)
    )
    for _ in range(3):
        native.step()
    _native_close(final, functional.extract(native).state)


@pytest.mark.parametrize("provided_population", ("left", "right"))
def test_shared_rng_host_fills_missing_initialization_packet_in_population_order(
    provided_population,
):
    source = _network(reseed=True, prewrites=True)
    runner = dn.compile(
        source,
        dt=DT,
        steps=1,
        initialization="fresh",
        inference=False,
        disable=True,
    )
    tensors = runner.functional.extract()
    complete = _initialization(runner.functional, tensors)
    frozen = _clone(tensors, detached=True)
    rng = torch.get_rng_state().clone()
    expected, expected_auxiliary = runner(initialization=complete)
    partial = complete._replace(
        populations={
            name: value
            if name == provided_population
            else value._replace(random_draw=None)
            for name, value in complete.populations.items()
        }
    )
    actual, auxiliary = runner(initialization=partial)
    _close(actual, expected)
    _close(auxiliary["parameters"], expected_auxiliary["parameters"])
    _close(auxiliary["constants"], expected_auxiliary["constants"])
    _close(runner.functional.extract(), frozen)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    _assert_aliases(source)
