"""Packed MPS execution retains HH training and public checkpoint semantics."""

from __future__ import annotations

import io

import networkx as nx
import pytest
import torch
from torch.utils import _pytree as pytree
from torch.utils.checkpoint import checkpoint

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators._native import _mps_solver_available
from dendra.models.mod import hh

try:
    import dendra_solvers
except ImportError:
    dendra_solvers = None

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not _mps_solver_available(dendra_solvers),
        reason="MPS-enabled dendra-solvers is unavailable",
    ),
]
DT = 0.01
STEPS = 8


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(DEVICE="cpu", DTYPE=torch.float32, JIT=0, REQUIRE_GRAD=True),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _graph(size):
    graph = nx.DiGraph()
    for node in range(size):
        graph.add_node(
            node,
            name=f"branch.{node}",
            kind="compartment",
            L=8.0 + node % 3,
            diam=1.2 + 0.05 * (node % 5),
            Ra=80.0 + 3.0 * (node % 7),
            cm=0.8 + 0.05 * (node % 4),
            area=85.0 + 7.0 * (node % 6),
        )
        if node:
            graph.add_edge((node - 1) // 2, node, R_ohm=1.7e8 + node * 1e5)
    return graph


def _model(device, *, write_back=True):
    common = dict(device=device, dtype=torch.float32, celsius=6.3, v_init=-62.0)
    parts = {
        "point": dn.SingleCompartment(N=1, C=1, cm=0.7, **common),
        "cable": dn.Unmyelinated([1.4], L=40.0, dx=10.0, cm=1.1, **common),
        "branch": dn.Tree.from_graph(_graph(7), N=2, **common),
        "wide": dn.Tree.from_graph(_graph(67), N=1, **common),
    }
    for part in parts.values():
        part.insert(hh)
    model = dn.concat_models(parts, celsius=6.3, threads=2, write_back=write_back)
    model.batch(2)
    model.initialize()
    model.train()
    model.run(tstop=0.0, dt=DT)
    return model


def _clone(value, *, detach=True):
    return pytree.tree_map_only(
        torch.Tensor,
        lambda tensor: (tensor.detach() if detach else tensor).clone(),
        value,
    )


def _close(actual, expected, *, rtol=4e-5, atol=4e-5):
    actual_leaves, actual_spec = pytree.tree_flatten(actual)
    expected_leaves, expected_spec = pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for left, right in zip(actual_leaves, expected_leaves, strict=True):
        if torch.is_tensor(left):
            assert torch.is_tensor(right)
            torch.testing.assert_close(
                left.detach().cpu(), right.detach().cpu(), rtol=rtol, atol=atol
            )
        else:
            assert left == right


def _drive(model, steps=STEPS):
    return torch.linspace(
        -0.4, 0.6, steps * model.v.numel(), dtype=torch.float32, device="cpu"
    ).reshape(steps, *model.v.shape)


def _record(model, voltage):
    recorder = dn.Recorder(("v",))
    model.run(ve=voltage.to(model.device()), dt=DT, callbacks=[recorder])
    return recorder.stack("v")


@pytest.mark.parametrize("write_back", (False, True))
def test_mps_packed_hh_serialized_native_checkpoint_replays_and_matches_cpu(write_back):
    cpu, mps = (_model(device, write_back=write_back) for device in ("cpu", "mps"))
    voltage = _drive(cpu)
    for model in (cpu, mps):
        model.eval()
    with torch.no_grad():
        expected = _record(cpu, voltage)
        head = _record(mps, voltage[:3])
        owner = mps.integrator._cpu_topology_plan
        assert owner is not None and owner._token.device.type == "cpu"
        saved = _clone(mps.state_dict_for_checkpoint())
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=True)
        _close(restored, saved, rtol=0, atol=0)
        tail = _record(mps, voltage[3:])
        final = _clone(mps.state_dict_for_checkpoint())
        mps.restore_dict_from_checkpoint(restored)
        _close(mps.state_dict_for_checkpoint(), saved, rtol=0, atol=0)
        assert mps.integrator._cpu_topology_plan is owner
        replay = _record(mps, voltage[3:])
        assert mps.integrator._cpu_topology_plan is owner
        _close(replay, tail)
        _close(mps.state_dict_for_checkpoint(), final)
        _close(torch.cat((head, tail[1:])), expected)
        _close(mps.state_dict_for_checkpoint(), cpu.state_dict_for_checkpoint())
    assert mps.t.item() == pytest.approx(STEPS * DT)


def _functional_gradients(device):
    source = _model(device)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    parameters = dict(tensors.parameters)
    names = [
        name
        for name in parameters
        if name.endswith("gnabar_param") or name.endswith("cm_param.rho")
    ]
    assert len(names) >= 3
    for name in names:
        parameters[name] = parameters[name].detach().clone().requires_grad_()
    state = _clone(tensors.state)
    state["integrator"]["v"].requires_grad_()
    prepared = functional.prepare(parameters, tensors.constants)
    ve = _drive(source).to(device).requires_grad_()
    intra = (
        torch.linspace(-2e-10, 5e-10, ve.numel(), device=device, dtype=torch.float32)
        .reshape_as(ve)
        .requires_grad_()
    )

    def block(current, field, current_tape):
        return functional.rollout(
            parameters,
            prepared,
            current,
            dn.func.RolloutInput(ve=field, intra=current_tape),
        )[0]

    whole = block(state, ve, intra)
    prefix = checkpoint(block, state, ve[:3], intra[:3], use_reentrant=False)
    split = checkpoint(block, prefix, ve[3:], intra[3:], use_reentrant=False)
    _close(split, whole)

    def loss(current):
        voltage = current["integrator"]["v"]
        weights = torch.tensor(
            [0.5 + (index * 7 % 13) / 10.0 for index in range(voltage.numel())],
            device=device,
            dtype=torch.float32,
        ).reshape_as(voltage)
        voltage = (voltage.add(65.0).square() * weights).mean()
        gates = current["mechanisms"]["hh"]
        return voltage + sum(gates[name].square().mean() for name in ("m", "h", "n"))

    arguments = [parameters[name] for name in names] + [
        state["integrator"]["v"],
        ve,
        intra,
    ]
    expected = torch.autograd.grad(loss(whole), arguments, retain_graph=True)
    actual = torch.autograd.grad(loss(split), arguments)
    for left, right in zip(actual, expected, strict=True):
        assert torch.isfinite(left).all().item()
        scale = max(right.detach().abs().max().item(), 1e-20)
        _close(left / scale, right / scale, rtol=3e-3, atol=3e-4)
    assert all(value.abs().max().item() > 0 for value in actual)
    return names, _clone(whole), tuple(value.detach() for value in actual)


def test_mps_packed_hh_checkpointed_chunk_gradients_match_uninterrupted_cpu():
    cpu_names, cpu_state, cpu_gradients = _functional_gradients("cpu")
    mps_names, mps_state, mps_gradients = _functional_gradients("mps")
    assert mps_names == cpu_names
    _close(mps_state, cpu_state)
    for actual, expected in zip(mps_gradients, cpu_gradients, strict=True):
        scale = max(expected.abs().max().item(), 1e-20)
        _close(actual / scale, expected / scale, rtol=3e-3, atol=3e-4)
