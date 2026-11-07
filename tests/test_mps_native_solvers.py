"""Native MPS solvers preserve physical Population and functional execution."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators import bwd_euler_bt, dhs_bt
from dendra.models.integrators._native import _mps_solver_available
from dendra.models.mechanisms._spatial import SpatialOperatorTree
from dendra.models.mod import pas

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
KINDS = ("thomas", "pcr", "block", "block_spd", "tree", "block_tree", "multi")


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=True, DEVICE="cpu", DTYPE=torch.float32),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _graph(*, block=False):
    graph = nx.DiGraph()
    for node in range(4):
        fields = dict(
            name=f"node{node}",
            kind="compartment",
            L=8.0 + node,
            diam=1.2 + 0.2 * node,
            Ra=80.0 + 7.0 * node,
            cm=0.75 + 0.1 * node,
            area=85.0 + 35.0 * node,
        )
        if block:
            fields.update(
                xraxial=[1.4 + 0.2 * node, 2.2 + 0.25 * node],
                xc=[0.06 + 0.01 * node, 0.11 + 0.015 * node],
                xg=[1.2e-4 + 0.2e-4 * node, 2.1e-4 + 0.3e-4 * node],
            )
        graph.add_node(node, **fields)
    for parent, child, resistance in ((3, 0, 1.7e8), (3, 2, 2.9e8), (2, 1, 4.3e8)):
        graph.add_edge(parent, child, R_ohm=resistance, L=1.0)
    return graph


def _model(kind, device):
    common = dict(dtype=torch.float32, device=device, v_init=-64.0)
    if kind in {"thomas", "pcr"}:
        model = dn.Unmyelinated(
            [1.2, 1.6],
            L=3.0,
            dx=1.0,
            integrator=dn.bwd_euler_ub(method=kind),
            **common,
        )
    elif kind in {"block", "block_spd"}:
        model = dn.ExtCellAxon(
            diameters=[1.2, 1.6],
            n_comp=4,
            integrator=bwd_euler_bt(
                method="spd" if kind == "block_spd" and device == "mps" else "thomas"
            ),
            **common,
        )
        model.xraxial.fill_(2.0)
        model.xc.fill_(0.25)
        model.xg.fill_(1.0e-4)
    elif kind in {"tree", "block_tree"}:
        block = kind == "block_tree"
        cls = dn.ExtCellTree if block else dn.Tree
        model = cls.from_graph(
            _graph(block=block),
            N=2,
            integrator=dhs_bt(threads=2) if block else dn.dhs(threads=2),
            **common,
        )
    else:
        tree = dn.Tree.from_graph(_graph(), N=2, **common)
        cable = dn.Unmyelinated([1.3], L=3.0, dx=1.0, **common)
        for component in (tree, cable):
            component.insert(pas, g=4.0e-4, e=-71.0)
        model = dn.concat_models({"tree": tree, "cable": cable}, threads=2)
    if kind != "multi":
        model.insert(pas, g=4.0e-4, e=-71.0)
    model.initialize()
    model.train()
    dt = torch.tensor(DT, dtype=torch.float32, device=device)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    if hasattr(model, "vc"):
        with torch.no_grad():
            shell = torch.linspace(
                -0.7, 0.8, model.v.numel(), device=device
            ).reshape_as(model.v)
            model.vc[..., 1] = shell
            model.vc[..., 2] = -0.3 * shell
            model.vc[..., 0] = model.v + shell
    return model


def _explicit(model):
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = dict(tensors.parameters)
    trainable = []
    for name, value in parameters.items():
        if name.endswith("g_param") or name.endswith("cm_param.rho"):
            parameters[name] = value.detach().clone().requires_grad_()
            trainable.append(name)
    assert trainable
    return functional, tensors, parameters, trainable


def _close(actual, expected, *, rtol=4e-5, atol=4e-5):
    torch.testing.assert_close(
        actual.detach().cpu(), expected.detach().cpu(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("kind", KINDS)
def test_mps_native_and_functional_rollouts_match_cpu_values_and_gradients(kind):
    cpu, native = _model(kind, "cpu"), _model(kind, "mps")
    f_cpu, t_cpu, p_cpu, targets = _explicit(cpu)
    f_mps, t_mps, p_mps, mps_targets = _explicit(native)
    assert targets == mps_targets
    prepared_cpu = f_cpu.prepare(p_cpu, t_cpu.constants)
    prepared_mps = f_mps.prepare(p_mps, t_mps.constants)
    state_cpu, state_mps = t_cpu.state, t_mps.state
    tape_cpu = (
        torch.linspace(-2e-9, 3e-9, 2 * native.v.numel())
        .reshape(2, *native.v.shape)
        .requires_grad_()
    )
    tape_mps = tape_cpu.detach().to("mps").requires_grad_()
    frames_cpu, frames_mps = [], []
    for index in range(2):
        state_cpu, aux_cpu = f_cpu.step(
            p_cpu, prepared_cpu, state_cpu, dn.func.StepInput(intra=tape_cpu[index])
        )
        state_mps, aux_mps = f_mps.step(
            p_mps, prepared_mps, state_mps, dn.func.StepInput(intra=tape_mps[index])
        )
        native.integrator.step(
            native, torch.tensor(DT, device="mps"), intra=tape_mps[index]
        )
        _close(aux_mps["v"], aux_cpu["v"])
        _close(aux_mps["v"], native.v)
        frames_cpu.append(aux_cpu["v"])
        frames_mps.append(aux_mps["v"])
    losses = [
        torch.stack(frames).add(65.0).square().sum()
        for frames in (frames_cpu, frames_mps)
    ]
    gradients = [
        torch.autograd.grad(
            loss, [parameters[name] for name in targets] + [tape], allow_unused=True
        )
        for loss, parameters, tape in zip(
            losses, (p_cpu, p_mps), (tape_cpu, tape_mps), strict=True
        )
    ]
    for expected, actual in zip(*gradients, strict=True):
        if expected is None:
            assert actual is None or not actual.count_nonzero().item()
            continue
        assert actual is not None and torch.isfinite(actual).all().item()
        scale = max(expected.detach().abs().max().item(), 1e-20)
        _close(actual / scale, expected / scale, rtol=3e-3, atol=3e-4)


@pytest.mark.parametrize("kind", ("tree", "multi"))
def test_mps_functional_prepare_and_step_compiles_fullgraph(kind):
    model = _model(kind, "mps")
    functional, tensors, parameters, _targets = _explicit(model)
    drive = torch.full_like(model.v, 1e-9, requires_grad=True)

    def step(parameters, constants, state, drive):
        return functional.prepare_and_step(
            parameters, constants, state, dn.func.StepInput(intra=drive)
        )

    expected = step(parameters, tensors.constants, tensors.state, drive)
    compiled = torch.compile(step, backend="aot_eager", fullgraph=True)
    actual = compiled(parameters, tensors.constants, tensors.state, drive)
    _close(actual[1]["v"], expected[1]["v"])
    got = torch.autograd.grad(actual[1]["v"].sum(), drive)[0]
    want = torch.autograd.grad(expected[1]["v"].sum(), drive)[0]
    _close(got, want, rtol=2e-4, atol=1e-3)


def test_mps_native_tree_plan_reuses_unchanged_topology_and_refreshes_after_write(
    monkeypatch,
):
    model = _model("tree", "mps")
    integrator = model.integrator
    first = integrator._cpu_topology_plan
    assert first is not None and first._token.device.type == "cpu"
    identities = {id(value) for value in integrator._cpu_plan_sources}
    original_cpu = torch.Tensor.cpu

    def cpu(value, *args, **kwargs):
        assert id(value) not in identities, "unchanged topology was copied from MPS"
        return original_cpu(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", cpu)
        integrator.step(model, torch.tensor(DT, device="mps"))
        assert integrator._cpu_topology_plan is first
    integrator.order.copy_(integrator.order)
    integrator.step(model, torch.tensor(DT, device="mps"))
    assert integrator._cpu_topology_plan is not first


def test_mps_tree_material_diffusion_uses_native_and_owned_functional_solves():
    model_cpu, model_mps = _model("tree", "cpu"), _model("tree", "mps")
    outputs, targets = [], []
    for model, solver in ((model_cpu, "dense"), (model_mps, "thomas")):
        concentration = (
            torch.linspace(0.4, 1.2, model.v.numel(), device=model.device())
            .reshape_as(model.v)
            .requires_grad_()
        )
        diffusion = torch.tensor(0.3, device=model.device(), requires_grad=True)
        operator = SpatialOperatorTree(solver=solver, threads=2)
        operator.configure_diffusion(concentration, DT, diffusion, model)
        native = operator.diffuse_implicit_configured(concentration)
        functional = operator.diffuse_implicit_prepared(
            concentration, operator._buffers, solve=operator._functional_solver()
        )
        _close(functional, native)
        if model.device().type == "mps":
            assert operator.solver_name == "dhs_mps"
        outputs.append(functional)
        targets.append((concentration, diffusion))
    _close(outputs[1], outputs[0])
    weights = torch.linspace(0.5, 1.5, outputs[0].numel()).reshape_as(outputs[0])
    gradients = [
        torch.autograd.grad((output * weights.to(output.device)).sum(), target)
        for output, target in zip(outputs, targets, strict=True)
    ]
    for expected, actual in zip(*gradients, strict=True):
        _close(actual, expected, rtol=3e-3, atol=3e-5)
