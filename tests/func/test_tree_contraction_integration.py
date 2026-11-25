"""Public scalar Tree integration for the exact tree contraction solver."""

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators import tree as tree_module
from dendra.models.mod import pas

DT = 0.025
DEVICE_DTYPES = (
    pytest.param("mps", torch.float32, id="mps-fp32", marks=pytest.mark.mps),
    pytest.param("cuda", torch.float32, id="cuda-fp32", marks=pytest.mark.cuda),
    pytest.param("cuda", torch.float64, id="cuda-fp64", marks=pytest.mark.cuda),
)


def _graph():
    graph = nx.DiGraph()
    for node in range(8):
        graph.add_node(
            node,
            name="soma.root" if node == 7 else f"dend.{node}",
            L=12.0 + node,
            diam=1.5 + 0.1 * node,
            Ra=97.0,
            cm=0.9,
            area=100.0 + 9.0 * node,
        )
    for parent, child in ((7, 1), (1, 2), (2, 3), (3, 4), (4, 0), (4, 5), (7, 6)):
        graph.add_edge(parent, child, R_ohm=1.5e8 + 1e7 * child, L=1.0)
    return graph


def _model(factory):
    model = dn.Tree.from_graph(_graph(), N=2, integrator=factory, v_init=-64.0)
    model.insert(pas, g=3e-4, e=-71.0)
    model.initialize()
    model.train()
    return model


@pytest.mark.cpu
@pytest.mark.parametrize("cap", (False, 0, -1, 2.5, "auto"))
def test_tree_contraction_rejects_invalid_cap(cap):
    with pytest.raises((TypeError, ValueError), match="max_segment_interior_nodes"):
        dn.tree_contraction(cap)


@pytest.mark.cpu
def test_tree_contraction_reports_unsupported_cpu_device():
    with dn.ctx(DEVICE="cpu", DTYPE=torch.float32, JIT=0, REQUIRE_GRAD=0):
        model = _model(dn.tree_contraction())
        with pytest.raises(NotImplementedError, match="CUDA or MPS"):
            dn.step(model, DT)


@pytest.mark.cpu
def test_tree_contraction_rejects_automatic_packing():
    with dn.ctx(DEVICE="cpu", DTYPE=torch.float32, JIT=0):
        model = _model(dn.tree_contraction())
        with pytest.raises(TypeError, match="tree_contraction requires a scalar Tree"):
            dn.concat_models({"cell": model})


@pytest.mark.cpu
@pytest.mark.parametrize("imem", (False, True))
def test_tree_contraction_explicit_imem_overrides_context(imem):
    with dn.ctx(DEVICE="cpu", DTYPE=torch.float32, IMEM=not imem, JIT=0):
        model = _model(dn.tree_contraction(imem=imem))
        assert model.imem is model.integrator.imem is imem
        assert (model.i_membrane is not None) is imem


def _result(model, *, functional, plan_type=None):
    scale = torch.linspace(
        0.25, 1.25, model.v.numel(), device=model.device(), dtype=model.dtype()
    )
    scale = scale.reshape_as(model.v)
    voltage = (-68 + 10 * scale).requires_grad_()
    field = (0.3 * scale.square()).requires_grad_()
    current = (1e-9 * scale.flip(-1)).requires_grad_()
    if functional:
        transition, tensors = dn.func.make_functional(model, dt=DT)
        if plan_type is not None:
            assert isinstance(transition._transition.solver.keywords["plan"], plan_type)
        prepared = transition.prepare(tensors.parameters, tensors.constants)
        state = tensors.state
        state["integrator"]["v"] = voltage
        for sign in (1, -1):
            state, _ = transition.step(
                tensors.parameters,
                prepared,
                state,
                dn.func.Inputs(ve=sign * field, intra=current),
            )
        v = state["integrator"]["v"]
        imem = state["integrator"]["i_membrane"]
    else:
        model.v = voltage
        solver = None
        for sign in (1, -1):
            dn.step(model, DT, inputs=dn.func.Inputs(ve=sign * field, intra=current))
            if plan_type is not None:
                current_solver = model.integrator._solve
                assert isinstance(current_solver.keywords["plan"], plan_type)
                assert solver is None or solver is current_solver
                solver = current_solver
        v = model.v
        imem = model.i_membrane
    loss = ((0.01 * v + 1e6 * imem) * scale).sum()
    gradients = torch.autograd.grad(loss, (voltage, field, current))
    return v, imem, gradients


def _require_contraction_device(device):
    if not getattr(torch, device).is_available():
        pytest.skip(f"{device} is unavailable")
    solvers = pytest.importorskip("dendra_solvers")
    if not hasattr(solvers, "TreeContractionPlan"):
        pytest.skip("requires dendra-solvers with tree contraction")
    if device == "mps" and not tree_module._mps_solver_available(solvers):
        pytest.skip("requires MPS-enabled dendra-solvers")
    if device == "cuda" and not tree_module.TRITON_AVAILABLE:
        pytest.skip("requires Triton")
    return solvers


def _assert_parity(result, reference, dtype):
    if dtype == torch.float64:
        tolerances = ((2e-11, 1e-11), (2e-9, 1e-20), (2e-9, 5e-12))
    else:
        tolerances = ((1e-6, 1e-5), (4e-4, 1e-12), (4e-4, 5e-6))
    (v, imem, gradients), (v_ref, imem_ref, reference_gradients) = result, reference
    (v_rtol, v_atol), (imem_rtol, imem_atol), (grad_rtol, grad_atol) = tolerances
    assert v.dtype == imem.dtype == dtype
    torch.testing.assert_close(v, v_ref, rtol=v_rtol, atol=v_atol)
    torch.testing.assert_close(imem, imem_ref, rtol=imem_rtol, atol=imem_atol)
    for gradient, expected in zip(gradients, reference_gradients, strict=True):
        assert gradient.dtype == dtype
        assert torch.count_nonzero(gradient).item() > 0
        torch.testing.assert_close(gradient, expected, rtol=grad_rtol, atol=grad_atol)


@pytest.mark.parametrize(("device", "dtype"), DEVICE_DTYPES)
@pytest.mark.parametrize("cap", (2, None))
def test_tree_contraction_matches_dhs_states_and_gradients(device, dtype, cap):
    solvers = _require_contraction_device(device)
    with dn.ctx(DEVICE=device, DTYPE=dtype, IMEM=True, JIT=0, REQUIRE_GRAD=1):
        reference = _result(_model(dn.dhs(imem=True)), functional=False)
        for functional in (False, True):
            result = _result(
                _model(dn.tree_contraction(cap, imem=True)),
                functional=functional,
                plan_type=solvers.TreeContractionPlan,
            )
            _assert_parity(result, reference, dtype)


@pytest.mark.slow
@pytest.mark.parametrize(("device", "dtype"), DEVICE_DTYPES)
@pytest.mark.parametrize("inference", (True, False))
def test_tree_contraction_public_compiled_scan(device, dtype, inference):
    _require_contraction_device(device)
    steps = 4
    options = {"compile_threads": 1}
    if device == "cuda":
        options["triton.cudagraphs"] = False
    with dn.ctx(DEVICE=device, DTYPE=dtype, IMEM=True, JIT=0, REQUIRE_GRAD=1):
        model = _model(dn.tree_contraction())
        reference = _model(dn.dhs())
        runner = dn.func.compile(
            model,
            dt=DT,
            steps=steps,
            inference=inference,
            backend="inductor",
            fullgraph=True,
            options=options,
        )
        functional, tensors = dn.func.make_functional(reference, dt=DT)
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        scale = torch.linspace(
            0.25, 1.25, steps * model.v.numel(), dtype=dtype, device=model.device()
        )
        scale = scale.reshape(steps, *model.v.shape)

        def evaluate(run):
            field = (0.3 * scale.square()).requires_grad_(not inference)
            current = (1e-9 * scale.flip(-1)).requires_grad_(not inference)
            with torch_compiler_warning_context():
                state, _ = run(dn.func.Inputs(ve=field, intra=current))
            v = state["integrator"]["v"]
            imem = state["integrator"]["i_membrane"]
            if inference:
                return v, imem, ()
            loss = (0.01 * v + 1e6 * imem).sum()
            return v, imem, torch.autograd.grad(loss, (field, current))

        actual = evaluate(lambda inputs: runner(inputs=inputs))
        expected = evaluate(
            lambda inputs: functional.rollout(
                tensors.parameters, prepared, tensors.state, inputs
            )
        )
        assert runner.execution_report().strategy == "compiled_scan"
        _assert_parity(actual, expected, dtype)
