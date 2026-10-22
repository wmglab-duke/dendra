"""Network recording selects owned physical regions across scan and while."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.implicit import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import expsyn, pas
from tests.func.test_network import DTYPE, _clone, _close
from tests.func.test_network_rollout import _slice

DT = 0.01
pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _network(*, packed=False, batch=2, extracellular=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=DTYPE, DEVICE="cpu"):
        if extracellular:
            left = dn.ExtCellAxon(
                [5.0, 8.0],
                n_comp=3,
                dtype=DTYPE,
                integrator=dn.bwd_euler_bt(method="thomas", imem=True),
            )
            left.xraxial.fill_(2.0)
            left.xc.fill_(0.1)
            left.xg.fill_(2e-4)
        else:
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
            population.insert(pas, g=0.001, e=-60.0 - 10 * index)
            population.insert(expsyn.rename("syn"), tau=1.2, e=0.0)
            population[:, 0].label("soma")
        left[1, :].label("cell_1")
        network = dn.Network({"left": left, "right": right})
        if packed:
            network = network.concat(packed=["left", "right"])
        if batch is not None:
            network.batch(batch)
        network.train()
        network.initialize(DT)
        with torch.no_grad():
            for population in network.populations.values():
                ramp = torch.linspace(-0.5, 0.7, population.v.numel(), dtype=DTYPE)
                population.v.add_(ramp.reshape(population.shape))
    return network


def _population(network, packed):
    name = "packed" if packed else "left"
    return name, network.populations[name]


def _regions(network, packed):
    _name, population = _population(network, packed)
    mask = torch.arange(population.shape[-1]).remainder(2) == 0
    return {
        "named": population.left.soma if packed else population.soma,
        "repeated": population[..., [-1, 0, -1]],
        "boolean": population[..., mask],
        "empty": population[..., []],
        "partial_batch": population[[1, 0, 1], :, [-1, 0, -1]],
    }


def _inputs(network, steps, *, requires_grad=False):
    return dn.func.NetworkRolloutInput(
        populations={
            name: dn.func.RolloutInput(
                intra=(
                    torch.linspace(
                        -1e-6, 3e-6, steps * population.v.numel(), dtype=DTYPE
                    )
                    .reshape(steps, *population.shape)
                    .requires_grad_(requires_grad)
                )
            )
            for name, population in network.populations.items()
        }
    )


def _specifications(network, packed):
    name, _population_source = _population(network, packed)
    physical = (f"{name}.v", f"{name}.i_membrane")
    regions = _regions(network, packed)
    specifications = {"full": dn.func.Recorder((*physical, "t"))}
    for key, region in regions.items():
        specifications[key] = dn.func.Recorder(
            (
                physical[0],
                f"populations.{name}.integrator.v",
                physical[1],
                f"{name}.t",
                "t",
            ),
            region=region,
        )
    return name, regions, specifications


@pytest.mark.parametrize(
    "packed", (False, True), ids=("two_populations", "network_concat")
)
@pytest.mark.parametrize("execution", ("eager", "scan", "while"))
def test_network_regions_match_full_physical_traces_and_continue_without_duplicate_samples(
    packed, execution
):
    network = _network(packed=packed)
    functional, tensors = dn.func.make_functional_network(network)
    name, regions, specifications = _specifications(network, packed)
    callbacks = functional.make_callbacks(specifications)
    inputs = _inputs(network, 8)
    frozen = _clone((tensors, inputs), detached=True)
    chunk = (
        None
        if execution == "eager"
        else functional.compile_rollout_chunk(
            4, execution=execution, backend="aot_eager"
        )
    )
    with torch.no_grad():
        expected, oracle = functional.prepare_and_rollout(
            *tensors, inputs, steps=8, callbacks=callbacks
        )
        prefix, first = functional.run(
            *tensors,
            _slice(inputs, 0, 4),
            steps=4,
            chunklength=4,
            compiled_chunk=chunk,
            callbacks=callbacks,
        )
        final, second = functional.run(
            tensors.parameters,
            tensors.constants,
            prefix,
            _slice(inputs, 4, 8),
            steps=4,
            chunklength=4,
            compiled_chunk=chunk,
            callbacks=callbacks,
            callback_state=first["callbacks"].state,
        )
    _close(final, expected)
    full = oracle["callbacks"]["full"]
    for key, region in regions.items():
        joined = {
            field: torch.cat(
                (first["callbacks"][key][field], second["callbacks"][key][field])
            )
            for field in first["callbacks"][key]
        }
        _close(joined, oracle["callbacks"][key])
        voltage = torch.stack([value[region.index] for value in full[f"{name}.v"]])
        current = torch.stack(
            [value[region.index] for value in full[f"{name}.i_membrane"]]
        )
        torch.testing.assert_close(joined[f"{name}.v"], voltage)
        torch.testing.assert_close(joined[f"populations.{name}.integrator.v"], voltage)
        torch.testing.assert_close(joined[f"{name}.i_membrane"], current)
        torch.testing.assert_close(joined["t"], full["t"])
        torch.testing.assert_close(joined[f"{name}.t"], full["t"])
        assert joined[f"{name}.v"].shape == (9, *region.shape)
        assert joined["t"].shape == (9,)
    if chunk is not None:
        assert chunk.execution_report().strategy == (
            "compiled_while_loop" if execution == "while" else "compiled_scan"
        )
    _close((tensors, inputs), frozen)
    _close(functional.extract().state, frozen[0].state)


@pytest.mark.parametrize(
    "packed", (False, True), ids=("two_populations", "network_concat")
)
def test_network_repeated_regions_keep_scan_checkpoint_voltage_and_input_gradients(
    packed,
):
    network = _network(packed=packed)
    functional, tensors = dn.func.make_functional_network(network)
    name, population = _population(network, packed)
    region = population[[1, 0, 1], :, [-1, 0, -1]]
    callbacks = functional.make_callbacks(
        {
            "full": dn.func.Recorder(f"{name}.v"),
            "selected": dn.func.Recorder(f"{name}.v", region=region),
        }
    )
    initial = (
        tensors.state["populations"][name]["integrator"]["v"]
        .detach()
        .clone()
        .requires_grad_()
    )
    original_population = tensors.state["populations"][name]
    state = {
        **tensors.state,
        "populations": {
            **tensors.state["populations"],
            name: {
                **original_population,
                "integrator": {**original_population["integrator"], "v": initial},
            },
        },
    }
    inputs = _inputs(network, 6, requires_grad=True)
    frozen = _clone((state, inputs), detached=True)
    _expected, oracle = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        state,
        inputs,
        steps=6,
        callbacks=callbacks,
    )
    chunk = functional.compile_rollout_chunk(4, execution="scan", backend="aot_eager")
    _actual, auxiliary = functional.run(
        tensors.parameters,
        tensors.constants,
        state,
        inputs,
        steps=6,
        chunklength=4,
        checkpoint_every=4,
        compiled_chunk=chunk,
        callbacks=callbacks,
    )
    full = oracle["callbacks"]["full"][f"{name}.v"]
    expected = torch.stack([value[region.index] for value in full])
    recorded = auxiliary["callbacks"]["selected"][f"{name}.v"]
    torch.testing.assert_close(recorded, expected)
    targets = (initial, inputs.populations[name].intra)
    actual_gradients = torch.autograd.grad(recorded.square().sum(), targets)
    expected_gradients = torch.autograd.grad(expected.square().sum(), targets)
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    _close((state, inputs), frozen)


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="extracellular CPU block execution requires dendra-solvers",
)
def test_network_while_regions_keep_real_extracellular_layers_for_qualified_vc_aliases():
    network = _network(extracellular=True)
    functional, tensors = dn.func.make_functional_network(network)
    fields = ("left.v", "left.vc", "left.i_membrane", "left.t", "t")
    regions = {
        "soma": network.left.soma,
        "partial": network.left[[1, 0, 1], :, [2, 0, 2]],
    }
    callbacks = functional.make_callbacks(
        {
            "full": dn.func.Recorder(fields),
            **{
                key: dn.func.Recorder(
                    (*fields, "populations.left.integrator.vc"), region=region
                )
                for key, region in regions.items()
            },
        }
    )
    inputs = _inputs(network, 4)
    frozen = _clone((tensors, inputs), detached=True)
    chunk = functional.compile_rollout_chunk(4, execution="while", backend="aot_eager")
    with torch.no_grad():
        expected, oracle = functional.prepare_and_rollout(
            *tensors, inputs, steps=4, callbacks=callbacks
        )
        actual, auxiliary = functional.run(
            *tensors, inputs, steps=4, compiled_chunk=chunk, callbacks=callbacks
        )
    _close(actual, expected)
    full = oracle["callbacks"]["full"]
    for key, region in regions.items():
        observed = auxiliary["callbacks"][key]
        _close(observed, oracle["callbacks"][key])
        vc = torch.stack(
            [
                torch.stack(
                    [value[..., layer][region.index] for layer in range(3)], dim=-1
                )
                for value in full["left.vc"]
            ]
        )
        torch.testing.assert_close(observed["left.vc"], vc)
        torch.testing.assert_close(observed["populations.left.integrator.vc"], vc)
        assert observed["left.vc"].shape == (5, *region.shape, 3)
    assert chunk.execution_report().strategy == "compiled_while_loop"
    _close((tensors, inputs), frozen)


@pytest.mark.slow
def test_cpp_while_network_concat_named_region_matches_eager_recording():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        network = _network(packed=True, batch=1)
        region = network.packed.left.soma
        runner = dn.compile(
            network,
            dt=DT,
            steps=4,
            execution="while",
            inference=True,
            callbacks={"soma": dn.func.Recorder(("packed.v", "t"), region=region)},
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
        _close(auxiliary["callbacks"]["soma"], oracle["callbacks"]["soma"])
        assert runner.execution_report().strategy == "compiled_while_loop"
    finally:
        torch.set_num_threads(previous_threads)
