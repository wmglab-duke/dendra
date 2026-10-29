"""Scoped Network callbacks reuse Population reducers and explicit carry."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_network import DTYPE, _clone, _close
from tests.func.test_network_rollout import _slice

DT = 0.01
CONDUCTANCE = "integrator.mech.mechanisms.pas.g_param"
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _network(*, packed=False, batch=2):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=DTYPE, DEVICE="cpu"):
        shared = torch.nn.Parameter(torch.tensor(0.001, dtype=DTYPE))
        left = dn.Population(
            N=2,
            C=3,
            v_init=-63.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=True),
        )
        right = dn.SingleCompartment(
            N=1,
            C=2,
            v_init=-67.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=True),
        )
        for index, population in enumerate((left, right)):
            population.insert(pas, g=shared, e=-60.0 - 10 * index)
            population[:, 0].label("soma")
        network = dn.Network({"left": left, "right": right})
        if packed:
            network = network.concat(packed=["left", "right"])
        if batch is not None:
            network.batch(batch)
        network.train()
        network.initialize(DT)
    return network


def _inputs(network, steps=6, *, trainable=False):
    populations = {}
    for offset, (name, population) in enumerate(network.populations.items()):
        times = torch.arange(steps, dtype=DTYPE)
        pulses = torch.where(times.remainder(3) == offset, 4e-6, -2e-6)
        spatial = torch.linspace(0.7, 1.3, population.v.numel(), dtype=DTYPE)
        drive = (pulses[:, None] * spatial).reshape(steps, *population.shape)
        populations[name] = dn.func.RolloutInput(intra=drive.requires_grad_(trainable))
    return dn.func.NetworkRolloutInput(populations=populations)


def _fields(population, *, trainable=False):
    field = torch.linspace(
        0.7, 1.3, population.shape[-2] * population.shape[-1], dtype=DTYPE
    ).reshape(population.shape[-2:])
    return [
        field.detach().clone().requires_grad_(trainable),
        field.flip(-1).detach().clone().requires_grad_(trainable),
    ]


def _specifications(fields):
    threshold = dict(
        threshold=-63.5,
        node_check=[0, 0, -1],
        t_start_check=DT,
        t_end_check=7 * DT,
    )
    return {
        "trace": dn.func.Recorder(("v", "i_membrane", "t")),
        "lfp": dn.func.LFP(fields),
        "raster": dn.func.Raster(**threshold),
        "count": dn.func.APCount(**threshold),
        "active": dn.func.Active(**threshold),
        "sites": dn.func.ActiveAL(**threshold, at_least=2),
        "total": dn.func.ActiveALCount(**threshold, at_least=3),
        "anomaly": dn.func.AnomalyDetector(),
    }


def _scoped(specifications):
    return {
        f"{population}_{name}": dn.func.PopulationCallback(population, callback)
        for population, callbacks in specifications.items()
        for name, callback in callbacks.items()
    }


def _replay_population_callbacks(functional, tensors, inputs, specifications):
    """Run independent Population collections on captured Network boundaries."""
    collections = {
        name: functional.populations[name].make_callbacks(callbacks)
        for name, callbacks in specifications.items()
    }
    carries, emissions = {}, {}
    state = tensors.state
    for name, collection in collections.items():
        carries[name], initial = collection._initialize(state["populations"][name])
        emissions[name] = [initial]
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    steps = next(iter(inputs.populations.values())).intra.shape[0]
    for index in range(steps):
        state, auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            inputs={
                name: dn.func.StepInput(intra=tape.intra[index])
                for name, tape in inputs.populations.items()
            },
        )
        for name, collection in collections.items():
            carries[name], emitted = collection._update(
                carries[name],
                state["populations"][name],
                auxiliary["populations"][name],
            )
            emissions[name].append(emitted)
    return state, {
        name: dn.func.FunctionalCallbackResults(
            state=carries[name],
            outputs=collection._finalize(
                carries[name], collection._stack(emissions[name])
            ),
        )
        for name, collection in collections.items()
    }


def _chunk(functional, execution, steps=4):
    return (
        None
        if execution == "eager"
        else functional.compile_rollout_chunk(
            steps, execution=execution, backend="aot_eager"
        )
    )


@pytest.mark.parametrize("execution", ("eager", "scan", "while"))
def test_scoped_builtin_callbacks_match_population_replay_and_keep_continuation_phase(
    execution,
):
    network = _network()
    functional, tensors = dn.func.make_functional_network(network)
    specifications = {
        name: _specifications(_fields(population))
        for name, population in network.populations.items()
    }
    callbacks = functional.make_callbacks(_scoped(specifications))
    inputs = _inputs(network)
    frozen = _clone((tensors, inputs), detached=True)
    chunk = _chunk(functional, execution)
    with torch.no_grad():
        expected, oracle = _replay_population_callbacks(
            functional, tensors, inputs, specifications
        )
        fresh_zero, boundary = functional.run(
            *tensors, steps=0, callbacks=callbacks, compiled_chunk=chunk
        )
        prefix, first = functional.run(
            *tensors,
            _slice(inputs, 0, 4),
            steps=4,
            callbacks=callbacks,
            compiled_chunk=chunk,
        )
        if chunk is not None:
            assert chunk.execution_report().strategy == (
                "compiled_while_loop" if execution == "while" else "compiled_scan"
            )
        empty, zero = functional.run(
            tensors.parameters,
            tensors.constants,
            prefix,
            steps=0,
            callbacks=callbacks,
            callback_state=first["callbacks"].state,
            compiled_chunk=chunk,
        )
        final, tail = functional.run(
            tensors.parameters,
            tensors.constants,
            empty,
            _slice(inputs, 4, 6),
            steps=2,
            callbacks=callbacks,
            callback_state=zero["callbacks"].state,
            compiled_chunk=chunk,
        )
    _close(fresh_zero, tensors.state)
    _close(empty, prefix)
    _close(final, expected)
    _close(zero["callbacks"].state.carries, first["callbacks"].state.carries)
    for population, output in oracle.items():
        assert boundary["callbacks"][f"{population}_trace"]["v"].shape[0] == 1
        assert boundary["callbacks"][f"{population}_lfp"].shape[0] == 1
        assert boundary["callbacks"][f"{population}_raster"].shape[0] == 0
        assert not boundary["callbacks"][f"{population}_count"].any()
        for name in specifications[population]:
            key = f"{population}_{name}"
            actual = tail["callbacks"][key]
            if name == "trace":
                actual = {
                    field: torch.cat(
                        tuple(
                            part["callbacks"][key][field]
                            for part in (first, zero, tail)
                        )
                    )
                    for field in actual
                }
                assert zero["callbacks"][key]["v"].shape[0] == 0
            elif name in {"lfp", "raster"}:
                actual = torch.cat(
                    tuple(part["callbacks"][key] for part in (first, zero, tail))
                )
                assert zero["callbacks"][key].shape[0] == 0
            _close(actual, output[name])
            _close(tail["callbacks"].state[key], output.state[name])
        assert not tail["callbacks"][f"{population}_anomaly"].any()
    assert oracle["left"]["raster"].any()
    assert not torch.equal(
        oracle["left"]["count"][..., 0, :], oracle["right"]["count"][..., 0, :]
    )
    if execution == "while":
        assert chunk.execution_report().strategy == "compiled_unrolled"
    _close((tensors, inputs), frozen)
    _close(functional.extract().state, frozen[0].state)


def test_scoped_lfp_scan_checkpoint_keeps_field_initial_drive_and_shared_parameter_gradients():
    network = _network()
    functional, tensors = dn.func.make_functional_network(network)
    assert (
        tensors.parameters["populations"]["left"][CONDUCTANCE]
        is tensors.parameters["populations"]["right"][CONDUCTANCE]
    )
    fields = {
        name: _fields(population, trainable=True)
        for name, population in network.populations.items()
    }
    callbacks = functional.make_callbacks(
        _scoped(
            {
                name: {
                    "lfp": dn.func.LFP(fields[name]),
                    "trace": dn.func.Recorder("v"),
                }
                for name in fields
            }
        )
    )
    initial = {
        name: value["integrator"]["v"].detach().clone().requires_grad_()
        for name, value in tensors.state["populations"].items()
    }
    state = {
        **tensors.state,
        "populations": {
            name: {
                **value,
                "integrator": {**value["integrator"], "v": initial[name]},
            }
            for name, value in tensors.state["populations"].items()
        },
    }
    inputs = _inputs(network, trainable=True)
    frozen = _clone((state, inputs), detached=True)
    chunk = _chunk(functional, "scan")
    targets = (
        *initial.values(),
        *(field for values in fields.values() for field in values),
        *(tape.intra for tape in inputs.populations.values()),
        tensors.parameters["populations"]["left"][CONDUCTANCE],
    )

    def loss(auxiliary):
        output = auxiliary["callbacks"]
        return sum(
            (output[f"{name}_lfp"] * 1e6).square().mean()
            + output[f"{name}_trace"]["v"].square().mean() * 1e-4
            for name in initial
        )

    reference, oracle = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        state,
        inputs,
        steps=6,
        callbacks=callbacks,
    )
    expected = torch.autograd.grad(loss(oracle), targets)
    for checkpoint_every in (None, 4):
        actual, auxiliary = functional.run(
            tensors.parameters,
            tensors.constants,
            state,
            inputs,
            steps=6,
            chunklength=4,
            checkpoint_every=checkpoint_every,
            callbacks=callbacks,
            compiled_chunk=chunk,
        )
        _close(actual, reference)
        _close(auxiliary["callbacks"].outputs, oracle["callbacks"].outputs)
        gradients = torch.autograd.grad(loss(auxiliary), targets)
        for gradient, wanted in zip(gradients, expected, strict=True):
            torch.testing.assert_close(gradient, wanted, rtol=2e-8, atol=1e-12)
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    _close((state, inputs), frozen)
    assert (
        tensors.parameters["populations"]["left"][CONDUCTANCE]
        is tensors.parameters["populations"]["right"][CONDUCTANCE]
    )


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_scoped_packed_region_and_masked_lfp_select_the_same_named_component(execution):
    network = _network(packed=True)
    functional, tensors = dn.func.make_functional_network(network)
    region = network.packed.left.soma
    field = torch.zeros_like(network.packed.v)
    field[region.index] = 1.0
    callbacks = functional.make_callbacks(
        {
            "selected": dn.func.PopulationCallback(
                "packed", dn.func.Recorder(("v", "i_membrane", "t"), region=region)
            ),
            "lfp": dn.func.PopulationCallback("packed", dn.func.LFP([field])),
            "qualified": dn.func.Recorder(
                ("packed.v", "packed.i_membrane", "t"), region=region
            ),
        }
    )
    inputs = _inputs(network, 4)
    with torch.no_grad():
        expected, oracle = functional.prepare_and_rollout(
            *tensors, inputs, steps=4, callbacks=callbacks
        )
        actual, auxiliary = functional.run(
            *tensors,
            inputs,
            steps=4,
            callbacks=callbacks,
            compiled_chunk=_chunk(functional, execution),
        )
    _close(actual, expected)
    _close(auxiliary["callbacks"].outputs, oracle["callbacks"].outputs)
    selected, qualified = (
        auxiliary["callbacks"]["selected"],
        auxiliary["callbacks"]["qualified"],
    )
    torch.testing.assert_close(selected["v"], qualified["packed.v"])
    torch.testing.assert_close(selected["i_membrane"], qualified["packed.i_membrane"])
    torch.testing.assert_close(selected["t"], qualified["t"])
    torch.testing.assert_close(
        auxiliary["callbacks"]["lfp"].squeeze(-1),
        selected["i_membrane"].sum((-1, -2)),
        atol=1e-20,
        rtol=2e-11,
    )


@dataclass(frozen=True)
class _PopulationEnergy(dn.func.FunctionalCallback):
    def _bind(self, functional):
        assert isinstance(functional, dn.func.FunctionalPopulation)
        return self

    def initialize(self, state, auxiliary):
        assert auxiliary is None
        return state["integrator"]["v"].square().mean(), None

    def update(self, carry, state, auxiliary):
        voltage = state["integrator"]["v"]
        return carry + voltage.square().mean() + auxiliary["v"].mean(), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


def test_scoped_custom_callback_scan_receives_only_the_selected_population_and_auxiliary():
    network = _network()
    functional, tensors = dn.func.make_functional_network(network)
    specifications = {
        name: {"energy": _PopulationEnergy()} for name in functional.populations
    }
    callbacks = functional.make_callbacks(_scoped(specifications))
    inputs = _inputs(network, 4)
    expected, oracle = _replay_population_callbacks(
        functional, tensors, inputs, specifications
    )
    actual, auxiliary = functional.run(
        *tensors,
        inputs,
        steps=4,
        callbacks=callbacks,
        compiled_chunk=_chunk(functional, "scan"),
    )
    _close(actual, expected)
    for name, output in oracle.items():
        _close(auxiliary["callbacks"][f"{name}_energy"], output["energy"])


@pytest.mark.parametrize("steps", (1, 4))
def test_scoped_custom_callback_while_support_includes_short_tails(steps):
    functional, tensors = dn.func.make_functional_network(_network())
    callbacks = functional.make_callbacks(
        {"energy": dn.func.PopulationCallback("left", _PopulationEnergy())}
    )
    chunk = _chunk(functional, "while", steps)
    with torch.no_grad():
        actual, auxiliary = functional.run(
            *tensors,
            steps=steps,
            callbacks=callbacks,
            compiled_chunk=chunk,
        )
        expected, reference = functional.prepare_and_rollout(
            *tensors, steps=steps, callbacks=callbacks
        )
    _close(actual, expected)
    _close(auxiliary["callbacks"]["energy"], reference["callbacks"]["energy"])
    assert chunk.execution_report().strategy == (
        "compiled_while_loop" if steps >= 4 else "compiled_unrolled"
    )


@pytest.mark.slow
def test_cpp_while_scoped_lfp_and_threshold_callbacks_match_eager_population_results():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        network = _network(batch=1)
        runner = dn.compile(
            network,
            dt=DT,
            steps=4,
            execution="while",
            inference=True,
            callbacks={
                "left_lfp": dn.func.PopulationCallback(
                    "left", dn.func.LFP(_fields(network.left))
                ),
                "right_lfp": dn.func.PopulationCallback(
                    "right", dn.func.LFP(_fields(network.right))
                ),
                "count": dn.func.PopulationCallback(
                    "left", dn.func.APCount(threshold=-63.5, node_check=[0, -1])
                ),
            },
            backend="inductor",
            options={"cpp_wrapper": True, "compile_threads": 1},
        )
        inputs = _inputs(network, 4)
        with torch.no_grad():
            expected, oracle = runner.functional.prepare_and_rollout(
                runner._parameters,
                runner._constants,
                runner.initial_state,
                inputs,
                steps=4,
                callbacks=runner._callbacks,
            )
        actual, auxiliary = runner(inputs=inputs)
        _close(actual, expected)
        _close(auxiliary["callbacks"].outputs, oracle["callbacks"].outputs)
        assert runner.execution_report().strategy == "compiled_while_loop"
    finally:
        torch.set_num_threads(previous_threads)
