"""Portable checkpoint compatibility includes topology and static transforms."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import graded_syn, pas
from tests.func.test_continuation import _runner, _snapshot
from tests.func.test_continuation_checkpoints import _copy_checkpoint
from tests.func.test_continuous_con import DT as CONNECTION_DT
from tests.func.test_lowering_api import DT, G
from tests.func.test_scalar_tree import _reordered_root_graph

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with (
        torch_compiler_warning_context(),
        dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False),
    ):
        yield


def _program(model, *, dt=DT):
    return dn.compile(
        model, dt=dt, steps=2, inference=True, backend="eager", disable=True
    )


def _initial(program):
    return dn.func.Continuation(
        program.initial_state, program.parameters, program.constants
    )


def _tree(*, alternate=False):
    graph = _reordered_root_graph()
    if alternate:
        edge = dict(graph.edges[2, 1])
        graph.remove_edge(2, 1)
        graph.add_edge(0, 1, **edge)
    model = dn.Tree.from_graph(
        graph,
        N=1,
        v_init=-60.0,
        dtype=torch.float64,
        integrator=dn.dhs(threads=1, imem=False),
    )
    model.insert(pas, g=0.001, e=-65.0)
    model.train()
    model.initialize(DT)
    return model


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE, reason="Tree lowering requires dendra-solvers"
)
def test_checkpoint_rejects_a_same_size_tree_with_different_parent_topology():
    source, receiver = _program(_tree()), _program(_tree(alternate=True))
    assert source.functional.shape == receiver.functional.shape
    assert (
        source.functional._population.compartment_graph.topology.parent_index
        != receiver.functional._population.compartment_graph.topology.parent_index
    )
    packet = source.export_continuation(_initial(source))
    before = _snapshot(receiver.initial_state)
    with pytest.raises(
        dn.func.FunctionalizationError, match="owner|configuration|compatible"
    ):
        receiver.restore_continuation(packet)
    torch.testing.assert_close(receiver.initial_state, before)


def _multi(widths):
    populations = {}
    for name, width in zip(("left", "right"), widths, strict=True):
        model = dn.SingleCompartment(
            N=1,
            C=width,
            dtype=torch.float64,
            v_init=-60.0,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(pas, g=0.001, e=-65.0)
        populations[name] = model
    model = dn.concat_models(populations, write_back=False)
    model.train()
    model.initialize(DT)
    return model


def test_checkpoint_rejects_equal_flattened_multi_population_with_changed_partitions():
    source, receiver = _program(_multi((1, 3))), _program(_multi((2, 2)))
    assert source.functional.shape == receiver.functional.shape
    source_components = source.functional._population.populations
    receiving_components = receiver.functional._population.populations
    assert tuple(source_components) == tuple(receiving_components)
    assert tuple(source_components["left"].shape) != tuple(
        receiving_components["left"].shape
    )
    with pytest.raises(
        dn.func.FunctionalizationError, match="owner|configuration|compatible"
    ):
        receiver.restore_continuation(source.export_continuation(_initial(source)))


def _regional_population(columns):
    model = dn.Population(
        N=1,
        C=4,
        dtype=torch.float64,
        v_init=-60.0,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model[:, list(columns)].insert(pas.rename("regional"), g=0.001, e=-65.0)
    model.train()
    model.initialize(DT)
    return model


def test_checkpoint_rejects_same_shape_mechanism_with_different_support_placement():
    source = _program(_regional_population((0, 2)))
    receiver = _program(_regional_population((1, 3)))
    assert source.functional.shape == receiver.functional.shape
    assert set(source.parameters) == set(receiver.parameters)
    assert {name: tuple(value.shape) for name, value in source.parameters.items()} == {
        name: tuple(value.shape) for name, value in receiver.parameters.items()
    }
    assert (
        source.initial_state["integrator"]["v"].shape
        == receiver.initial_state["integrator"]["v"].shape
    )
    with pytest.raises(
        dn.func.FunctionalizationError, match="owner|configuration|compatible"
    ):
        receiver.restore_continuation(source.export_continuation(_initial(source)))


class _StaticGate(torch.nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.gain = gain

    def forward(self, value):
        return torch.sigmoid(value * self.gain)


def _closure_gate(gain):
    def gate(value):
        return torch.sigmoid(value * gain)

    return gate


def _connected_model(transform):
    populations = {}
    for name in ("pre", "post"):
        population = dn.SingleCompartment(
            N=1, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        population.insert(pas, g=0.001, e=-65.0)
        populations[name] = population
    populations["post"].insert(graded_syn)
    model = dn.Network(populations)
    model.connect_continuous_one_to_one(
        model.pre[:],
        model.post[:],
        model.post.mech.graded_syn,
        weight=0.002,
        delay=0.0,
        transform=transform,
    )
    model.train()
    model.initialize(CONNECTION_DT)
    return model


@pytest.mark.parametrize("factory", (_StaticGate, _closure_gate))
def test_checkpoint_compares_static_transform_configuration_for_same_implementation(
    factory,
):
    source_model = _connected_model(factory(0.7))
    equivalent_model = _connected_model(factory(0.7))
    changed_model = _connected_model(factory(1.2))
    source = _program(source_model, dt=CONNECTION_DT)
    equivalent = _program(equivalent_model, dt=CONNECTION_DT)
    changed = _program(changed_model, dt=CONNECTION_DT)
    packet = source.export_continuation(_initial(source))
    restored = equivalent.restore_continuation(packet)
    torch.testing.assert_close(restored.state, source.initial_state)
    with pytest.raises(
        dn.func.FunctionalizationError, match="owner|configuration|compatible"
    ):
        changed.restore_continuation(packet)


def _node_at(node, path):
    for key in path:
        assert node["kind"] == "dict"
        node = node["children"][node["keys"].index(key)]
    return node


def test_restore_rejects_a_well_formed_tensor_reference_that_breaks_mandatory_aliases():
    source, receiver = _runner(True, callbacks=False), _runner(True, callbacks=False)
    packet = _copy_checkpoint(source.export_continuation(_initial(source)))
    frames = packet["frames"]
    left = _node_at(frames["tree"], ("parameters", "populations", "left", G))
    right = _node_at(frames["tree"], ("parameters", "populations", "right", G))
    assert left["kind"] == right["kind"] == "tensor"
    assert left["index"] == right["index"]
    index = left["index"]
    with torch.inference_mode(False), torch.no_grad():
        duplicate = frames["tensors"][index].detach().clone()
        duplicate.requires_grad_(frames["requires_grad"][index])
    right["index"] = len(frames["tensors"])
    frames["tensors"].append(duplicate)
    frames["requires_grad"].append(duplicate.requires_grad)
    with pytest.raises((ValueError, dn.func.FunctionalizationError), match="alias"):
        receiver.restore_continuation(packet)


_PORTABLE_GATE_GAIN = 0.07
_PORTABLE_GATE_BIAS = -0.2
_PORTABLE_GATE_OFFSET = 0.03


def _portable_global_gate(value):
    return (
        torch.sigmoid(value * _PORTABLE_GATE_GAIN + _PORTABLE_GATE_BIAS)
        + _PORTABLE_GATE_OFFSET
    )


def _global_gate_program():
    model = _connected_model(_portable_global_gate)
    return _program(model, dt=CONNECTION_DT)


def test_checkpoint_function_globals_restore_across_distinct_python_hash_seeds(
    tmp_path,
):
    checkpoint = tmp_path / "cross-process.pt"
    result = tmp_path / "restored.pt"
    script = """
import sys
import torch
import dendra as dn
from tests.func.test_continuation_checkpoint_compatibility import _global_gate_program

with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=False):
    if sys.argv[1] == "export":
        source = _global_gate_program()
        _, auxiliary = source()
        continuation = auxiliary["continuation"]
        checkpoint = source.export_continuation(continuation)
        expected, _ = source(continuation)
        torch.save({"checkpoint": checkpoint, "expected": expected,
                    "hash": hash("dendra-portable-function-globals")}, sys.argv[2])
    else:
        saved = torch.load(sys.argv[2], weights_only=True)
        receiver = _global_gate_program()
        continuation = receiver.restore_continuation(saved["checkpoint"])
        actual, _ = receiver(continuation)
        torch.testing.assert_close(actual, saved["expected"])
        torch.save({"state": actual,
                    "hash": hash("dendra-portable-function-globals")}, sys.argv[3])
"""
    repository = Path(__file__).resolve().parents[2]
    for seed, mode in (("1", "export"), ("2", "restore")):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        completed = subprocess.run(
            [sys.executable, "-c", script, mode, str(checkpoint), str(result)],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    saved = torch.load(checkpoint, weights_only=True)
    restored = torch.load(result, weights_only=True)
    assert saved["hash"] != restored["hash"]
    torch.testing.assert_close(restored["state"], saved["expected"])
