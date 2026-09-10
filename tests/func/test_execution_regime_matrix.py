"""Cross-topology regression fence for the supported differentiation regimes.

Detailed state, preparation, batching, and solver contracts live in the focused
functional test modules.  This intentionally small matrix instead seals the two
public execution policies which must continue to compose across every admitted
operator family:

* eager fixed-step rollout under ``torch.func`` transformations; and
* fixed compiled chunks inside an eager host loop using ordinary autograd.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from importlib.util import find_spec

import networkx as nx
import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators import dhs_bt
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas

DT = 0.01
DTYPE = torch.float64
STEPS = 2
PAS_G = "integrator.mech.mechanisms.pas.g_param"


_DENDRA_MODELS_AVAILABLE = find_spec("dendra_models") is not None
if _DENDRA_MODELS_AVAILABLE:
    from dendra_models.models.cells.peripheral import Sweeney1987
else:  # pragma: no cover - exercised by the explicit skip below
    Sweeney1987 = None


pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


@dataclass(frozen=True)
class _RegimeCase:
    name: str
    build: Callable[[], dn.Population]
    parameter: str = PAS_G


def _finish(model):
    model.initialize()
    model.train()
    return model


def _single_compartment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=2,
            v_init=torch.tensor([-65.0, -59.0], dtype=DTYPE),
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(pas, g=5.0e-4, e=-71.0)
        return _finish(model)


def _unmyelinated():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0],
            L=2.0,
            dx=1.0,
            v_init=torch.tensor([-65.0, -61.0, -58.0], dtype=DTYPE),
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=5.0e-4, e=-71.0)
        return _finish(model)


def _native_cable():
    morphology = dn.Morphology(rhoa=91.0, cm=1.0)
    root = morphology.section("root", L=13.0, diam=4.0, nseg=1)
    branch = morphology.section("branch", L=17.0, diam=1.5, nseg=1)
    branch.connect(root.at(1.0))
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Cable.from_morphology(
            morphology,
            N=1,
            v_init=-63.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=5.0e-4, e=-71.0)
        return _finish(model)


def _tree_graph(*, extracellular=False):
    graph = nx.DiGraph()
    extra = (
        {"xraxial": [1.6, 2.3], "xc": [0.08, 0.13], "xg": [1.5e-4, 2.4e-4]}
        if extracellular
        else {}
    )
    graph.add_node(
        0,
        name="root",
        L=10.0,
        diam=3.0,
        Ra=91.0,
        cm=0.9,
        area=140.0,
        **extra,
    )
    graph.add_node(
        1,
        name="branch",
        L=14.0,
        diam=1.4,
        Ra=117.0,
        cm=1.1,
        area=95.0,
        **extra,
    )
    graph.add_edge(0, 1, R_ohm=1.4e8, L=1.0)
    return graph


def _scalar_tree():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Tree.from_graph(
            _tree_graph(),
            N=1,
            v_init=-63.0,
            dtype=DTYPE,
            integrator=dn.dhs(threads=2),
        )
        model.insert(pas, g=5.0e-4, e=-71.0)
        return _finish(model)


def _extcell_axon():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.ExtCellAxon(
            diameters=[6.0],
            n_comp=2,
            v_init=-63.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_bt(method="thomas", imem=False),
        )
        model.insert(pas, g=5.0e-4, e=-71.0)
        return _finish(model)


def _extcell_tree():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.ExtCellTree.from_graph(
            _tree_graph(extracellular=True),
            N=1,
            v_init=-63.0,
            dtype=DTYPE,
            integrator=dhs_bt(threads=2, imem=False),
        )
        model.insert(pas, g=5.0e-4, e=-71.0)
        return _finish(model)


def _mixed_multi_population():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        point = dn.SingleCompartment(1, C=1, dtype=DTYPE, v_init=-65.0)
        cable = dn.Unmyelinated(
            [2.0],
            L=1.0,
            dx=1.0,
            dtype=DTYPE,
            v_init=-61.0,
        )
        point.insert(pas, g=5.0e-4, e=-71.0)
        cable.insert(pas, g=5.0e-4, e=-71.0)
        model = dn.concat_models({"point": point, "cable": cable}, threads=2)
        return _finish(model)


def _sweeney():
    assert Sweeney1987 is not None
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = Sweeney1987(
            diameters=[8.0],
            n_node=3,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        ).to(dtype=DTYPE)
        return _finish(model)


_NATIVE_MARK = pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="native CPU tree and block operators require dendra-solvers",
)
_DENDRA_MODELS_MARK = pytest.mark.skipif(
    not _DENDRA_MODELS_AVAILABLE,
    reason="real Myelinated coverage requires dendra-models",
)

CASES = (
    pytest.param(_RegimeCase("sc", _single_compartment), id="single-compartment-sc"),
    pytest.param(_RegimeCase("ub", _unmyelinated), id="unmyelinated-ub"),
    pytest.param(_RegimeCase("native-cable", _native_cable), id="native-cable-ub"),
    pytest.param(_RegimeCase("tree", _scalar_tree), marks=_NATIVE_MARK, id="tree"),
    pytest.param(
        _RegimeCase("block-tridiagonal", _extcell_axon),
        marks=_NATIVE_MARK,
        id="extcell-axon-bt",
    ),
    pytest.param(
        _RegimeCase("block-tree", _extcell_tree),
        marks=_NATIVE_MARK,
        id="extcell-tree",
    ),
    pytest.param(
        _RegimeCase(
            "mixed-multi",
            _mixed_multi_population,
            parameter="integrator.mech.mechanisms.pas.g_point",
        ),
        marks=_NATIVE_MARK,
        id="mixed-multi-population",
    ),
    pytest.param(
        _RegimeCase("myelinated", _sweeney, parameter="noded2"),
        marks=_DENDRA_MODELS_MARK,
        id="real-myelinated-ub",
    ),
)


def _drives(model):
    count = STEPS * model.v.numel()
    ve = torch.linspace(
        -0.8,
        1.1,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(STEPS, *model.shape)
    intra = torch.linspace(
        -1.0e-9,
        1.5e-9,
        count,
        dtype=model.dtype(),
        device=model.device(),
    ).reshape(STEPS, *model.shape)
    return dn.func.RolloutInput(ve=ve, intra=intra)


def _clone_tree(value):
    return torch.utils._pytree.tree_map(lambda tensor: tensor.detach().clone(), value)


def _assert_tree_close(actual, expected):
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_leaf, expected_leaf, rtol=3.0e-9, atol=3.0e-11
        )


def _objective(functional, tensors, inputs, parameter_name):
    def loss(parameter):
        parameters = {**tensors.parameters, parameter_name: parameter}
        final, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            inputs,
        )
        return (
            final["integrator"]["v"].square().mean()
            + 0.01 * auxiliary["v"].sin().mean()
        )

    return loss


@pytest.mark.parametrize("case", CASES)
def test_eager_rollout_supports_torch_func_grad_and_hessian(case):
    model = case.build()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameter = tensors.parameters[case.parameter]
    loss = _objective(functional, tensors, _drives(model), case.parameter)

    gradient, value = torch.func.grad_and_value(loss)(parameter)
    hessian = torch.func.hessian(loss)(parameter)

    assert torch.isfinite(value)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert torch.isfinite(hessian).all()
    assert torch.count_nonzero(hessian) > 0


def _ordinary_gradient_run(
    functional, tensors, inputs, parameter_name, *, compiled, compiled_steps=1
):
    parameters: Mapping[str, torch.Tensor] = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    parameter = parameters[parameter_name].requires_grad_()
    state = _clone_tree(tensors.state)
    ve = inputs.ve.detach().clone().requires_grad_()
    intra = inputs.intra.detach().clone().requires_grad_()
    prepared = functional.prepare(parameters, tensors.constants)
    if compiled:
        kernel = functional.compile_rollout_chunk(compiled_steps, backend="aot_eager")
        step = partial(kernel, parameters, prepared)
    else:
        step = partial(functional.step, parameters, prepared)

    final, auxiliary = dn.func.run(
        functional,
        step,
        state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    loss = final["integrator"]["v"].square().mean() + 0.01 * auxiliary["v"].sin().mean()
    gradients = torch.autograd.grad(
        loss,
        (parameter, ve, intra),
        allow_unused=True,
        materialize_grads=True,
    )
    return final, auxiliary, loss, gradients


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("compiled_steps", [1, 2], ids=["width-1", "width-2"])
def test_compiled_fixed_chunks_host_loop_matches_eager_ordinary_bptt(
    case, compiled_steps
):
    model = case.build()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    inputs = _drives(model)
    expected = _ordinary_gradient_run(
        functional,
        tensors,
        inputs,
        case.parameter,
        compiled=False,
    )
    with torch_compiler_warning_context():
        actual = _ordinary_gradient_run(
            functional,
            tensors,
            inputs,
            case.parameter,
            compiled=True,
            compiled_steps=compiled_steps,
        )

    _assert_tree_close(actual[0], expected[0])
    _assert_tree_close(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=3.0e-9, atol=3.0e-11)
    for actual_gradient, expected_gradient in zip(
        actual[3],
        expected[3],
        strict=True,
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=3.0e-8,
            atol=3.0e-10,
        )
