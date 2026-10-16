"""Real scan kernels against an independently authored HH current/gate oracle.

The small pairwise matrix exercises both implicit backends, both CPU thread
counts, both floating dtypes, and scalar/batched populations without compiling
their entire Cartesian product. Recorder observes every solver voltage and
every physical HH gate, including the initial sample and chunk boundaries.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import hh
from tests.test_hh_rate_configuration import CHANNEL, GATES, LegacyHH
from tests.test_membrane_assembly_context import model as inconsistent_affine_model

DT = 0.025
pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        torch.__version__.split("+")[0] not in ("2.14.0", "2.14.1"),
        reason="scan requires a source-verified PyTorch 2.14.0 or 2.14.1 build",
    ),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


@pytest.fixture
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


@dataclass(frozen=True)
class _FrameDiagnostics(dn.func.FunctionalCallback):
    """Exercise floating, integer and Boolean carry cotangents together."""

    def initialize(self, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "frames": voltage.new_ones((), dtype=torch.int64),
            "seen_positive": (voltage > 0).any(),
            "energy": voltage.square().mean() / 1000.0,
        }, None

    def update(self, carry, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        return {
            "frames": carry["frames"] + 1,
            "seen_positive": carry["seen_positive"] | (voltage > 0).any(),
            "energy": carry["energy"] + voltage.square().mean() / 1000.0,
        }, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


def _clone(tree):
    return torch.utils._pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _case(kind, dtype, batch, *, legacy=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=dtype):
        if kind == "sc":
            model = dn.SingleCompartment(
                N=batch,
                C=3,
                dtype=dtype,
                celsius=6.3,
                v_init=-65.0,
                integrator=dn.bwd_euler_sc(imem=False),
            )
            insertion = model
        else:
            if not DENDRA_SOLVERS_AVAILABLE:
                pytest.skip("DHS validation requires dendra-solvers")
            morphology = dn.Morphology(rhoa=100.0, cm=1.0)
            soma = morphology.section("soma", L=20.0, diam=20.0, nseg=1)
            trunk = morphology.section("trunk", L=30.0, diam=2.0, nseg=2)
            left = morphology.section("left", L=20.0, diam=1.0, nseg=2)
            right = morphology.section("right", L=25.0, diam=1.5, nseg=2)
            trunk.connect(soma.at(1.0), child_end=0)
            left.connect(trunk.at(1.0), child_end=0)
            right.connect(trunk.at(1.0), child_end=0)
            model = dn.Tree.from_morphology(
                morphology,
                N=batch,
                dtype=dtype,
                celsius=6.3,
                v_init=-65.0,
                integrator=dn.dhs(threads=32, imem=False),
            )
            insertion = model.internal_nodes
        insertion.insert(LegacyHH if legacy else hh)
        model.initialize()
        model.train()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    mechanism_name = next(iter(model.mech.mechanisms))
    callbacks = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(
                ["v", *(f"{mechanism_name}.{gate}" for gate in GATES), "t"]
            ),
            "diagnostics": _FrameDiagnostics(),
        }
    )
    return SimpleNamespace(
        model=model,
        functional=functional,
        tensors=tensors,
        callbacks=callbacks,
        dtype=dtype,
        legacy=legacy,
        mechanism_name=mechanism_name,
    )


def _operands(case, steps, *, spiking=False, perturb=None):
    parameters = _clone(case.tensors.parameters)
    constants = _clone(case.tensors.constants)
    state = _clone(case.tensors.state)
    voltage = state["integrator"]["v"]
    # Hold gates fixed when differentiating V0, instead of silently including
    # equilibrium reinitialization in just one of the compared paths.
    voltage = (
        torch.linspace(-65.0, -62.0, voltage.numel(), dtype=case.dtype)
        .reshape_as(voltage)
        .requires_grad_()
    )
    state["integrator"]["v"] = voltage
    targets = {"initial_voltage": voltage}
    for name in CHANNEL:
        key = f"integrator.mech.mechanisms.{case.mechanism_name}.{name}_param"
        parameters[key].requires_grad_()
        targets[name] = parameters[key]
    density = (
        torch.linspace(0.01, 0.02, steps, dtype=case.dtype)
        .reshape(steps, 1, 1)
        .expand(steps, *case.model.shape)
        .clone()
    )
    if spiking:
        density.zero_()
        density[20:120] = 0.02
    density.requires_grad_()
    targets["stimulus"] = density
    if perturb is not None:
        name, delta = perturb
        with torch.no_grad():
            targets[name].add_(delta)
    # Absolute intracellular current, preserving zero-area algebraic nodes.
    inputs = dn.func.RolloutInput(intra=density * case.model.area)
    return parameters, constants, state, inputs, targets


def _evaluate(
    case,
    steps,
    *,
    kernel=None,
    policy="auto",
    spiking=False,
    gradients=True,
    perturb=None,
):
    parameters, constants, state, inputs, targets = _operands(
        case, steps, spiking=spiking, perturb=perturb
    )
    original = _clone((parameters, constants, state))
    with dn.ctx(MEMBRANE_ASSEMBLY=policy):
        prepared = case.functional.prepare(parameters, constants)
        binding = (case.functional if kernel is None else kernel).bind(
            parameters, prepared
        )
        final, auxiliary = dn.func.run(
            binding, state, steps=steps, inputs=inputs, callbacks=case.callbacks
        )
        results = auxiliary["callbacks"]
        recorded = results["trace"]
        # Region aliases do not rename the runtime mechanism. Normalize only
        # the returned oracle labels; its independently authored physics and
        # real parameter/state paths keep their own mechanism class name.
        trace = {"v": recorded["v"], "t": recorded["t"]}
        trace.update(
            {f"hh.{gate}": recorded[f"{case.mechanism_name}.{gate}"] for gate in GATES}
        )
        diagnostics = results["diagnostics"]
        loss = (
            trace["v"].square().mean() / 1000.0
            + sum(trace[f"hh.{gate}"].square().mean() for gate in GATES) / 10.0
            + diagnostics["energy"] / diagnostics["frames"] / 10.0
        )
        if gradients:
            computed = torch.autograd.grad(
                loss, tuple(targets.values()), retain_graph=True
            )
            repeated = torch.autograd.grad(loss, tuple(targets.values()))
            for first, second in zip(computed, repeated, strict=True):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            gradients = dict(zip(targets, computed, strict=True))
        else:
            gradients = None
    for actual, expected in zip(
        torch.utils._pytree.tree_leaves((parameters, constants, state)),
        torch.utils._pytree.tree_leaves(original),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert trace["v"].shape == (steps + 1, *case.model.shape)
    for gate in GATES:
        assert trace[f"hh.{gate}"].shape == (
            steps + 1,
            *state["mechanisms"][case.mechanism_name][gate].shape,
        )
    assert diagnostics["frames"].dtype == torch.int64
    assert diagnostics["frames"].item() == steps + 1
    assert diagnostics["seen_positive"].dtype == torch.bool
    assert diagnostics["seen_positive"].item() == bool((trace["v"] > 0).any())
    assert not diagnostics["frames"].requires_grad
    assert not diagnostics["seen_positive"].requires_grad
    assert not final["control"]["duration_remainder"].requires_grad
    torch.testing.assert_close(trace["v"][-1], final["integrator"]["v"])
    return SimpleNamespace(
        trace=_clone(trace),
        diagnostics=_clone(diagnostics),
        loss=loss.detach(),
        gradients=_clone(gradients),
    )


def _assert_close(actual, expected, dtype):
    for name, reference in expected.trace.items():
        value = actual.trace[name]
        assert torch.isfinite(value).all(), name
        assert torch.isfinite(reference).all(), name
        if name == "t":
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        else:
            torch.testing.assert_close(
                value,
                reference,
                rtol=5e-6 if dtype == torch.float32 else 2e-10,
                atol=(3e-4 if name == "v" else 3e-6)
                if dtype == torch.float32
                else 2e-9,
                msg=lambda message: f"full trace {name}: {message}",
            )
    assert actual.gradients.keys() == expected.gradients.keys()
    for name, reference in expected.gradients.items():
        value = actual.gradients[name]
        scale = reference.abs().amax()
        assert torch.isfinite(reference).all() and scale > 0, name
        assert torch.isfinite(value).all(), name
        torch.testing.assert_close(
            value / scale,
            reference / scale,
            rtol=5e-4 if dtype == torch.float32 else 2e-7,
            atol=2e-5 if dtype == torch.float32 else 2e-9,
            msg=lambda message: f"gradient {name}: {message}",
        )


@pytest.mark.parametrize(
    "kind,dtype,batch,cpu_threads",
    [
        pytest.param("sc", torch.float64, 1, 1, id="sc-f64-b1-t1"),
        pytest.param("sc", torch.float32, 2, 4, id="sc-f32-b2-t4"),
        pytest.param("dhs", torch.float32, 1, 1, id="dhs-f32-b1-t1"),
        pytest.param("dhs", torch.float64, 2, 4, id="dhs-f64-b2-t4"),
    ],
    indirect=["cpu_threads"],
)
def test_inductor_scan_full_traces_and_all_gradients(kind, dtype, batch, cpu_threads):
    del cpu_threads
    steps = 8
    case = _case(kind, dtype, batch)
    oracle = _case(kind, dtype, batch, legacy=True)
    torch.testing.assert_close(case.model.area, oracle.model.area, rtol=0, atol=0)
    for gate in GATES:
        torch.testing.assert_close(
            case.tensors.state["mechanisms"][case.mechanism_name][gate],
            oracle.tensors.state["mechanisms"][oracle.mechanism_name][gate],
            rtol=0,
            atol=0,
        )
    expected = _evaluate(oracle, steps, policy="off")
    scan = case.functional.compile_rollout_chunk(
        4, execution="scan", backend="inductor"
    )
    unrolled = case.functional.compile_rollout_chunk(
        4, execution="unrolled", backend="inductor"
    )
    actual = _evaluate(case, steps, kernel=scan)
    explicit = _evaluate(case, steps, kernel=unrolled)
    _assert_close(actual, expected, dtype)
    _assert_close(actual, explicit, dtype)
    assert scan.execution_report().strategy == "compiled_scan"
    assert scan.execution_report().callbacks and scan.execution_report().grad_enabled
    assert unrolled.execution_report().strategy == "compiled_unrolled"

    if kind == "sc" and dtype == torch.float64:
        # Central differences of the actual compiled scan objective. Perturb
        # one stable scalar conductance and all V0/stimulus entries together;
        # the latter are directional derivatives (sum of element gradients).
        for name, delta in (
            ("gnabar", 1e-5),
            ("initial_voltage", 1e-3),
            ("stimulus", 1e-5),
        ):
            plus = _evaluate(
                case, steps, kernel=scan, gradients=False, perturb=(name, delta)
            ).loss
            minus = _evaluate(
                case, steps, kernel=scan, gradients=False, perturb=(name, -delta)
            ).loss
            finite_difference = (plus - minus) / (2 * delta)
            torch.testing.assert_close(
                actual.gradients[name].sum(),
                finite_difference,
                rtol=2e-5,
                atol=2e-8,
                msg=lambda message: f"scan finite difference {name}: {message}",
            )


@pytest.mark.parametrize("cpu_threads", [1], indirect=True)
def test_inductor_scan_long_spiking_trace_and_all_gradients(cpu_threads):
    del cpu_threads
    steps = 400
    case = _case("dhs", torch.float64, 2)
    oracle = _case("dhs", torch.float64, 2, legacy=True)
    expected = _evaluate(oracle, steps, policy="off", spiking=True)
    scan = case.functional.compile_rollout_chunk(
        40, execution="scan", backend="inductor"
    )
    actual = _evaluate(case, steps, kernel=scan, spiking=True)
    assert actual.trace["v"].amax() > 0
    assert expected.trace["v"].amax() > 0
    _assert_close(actual, expected, torch.float64)
    assert scan.execution_report().strategy == "compiled_scan"


def test_real_scan_recorded_policy_switch_and_failed_capture_isolation():
    # Deliberately different declared coefficients make stale policy captures
    # observable, unlike HH where auto/off differ only by floating rounding.
    source = inconsistent_affine_model()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(["v", "t"]),
            "diagnostics": _FrameDiagnostics(),
        }
    )
    scan = functional.compile_rollout_chunk(4, execution="scan", backend="aot_eager")
    bound = scan.bind(tensors.parameters, prepared)
    eager = functional.bind(tensors.parameters, prepared)

    def run(binding):
        _final, auxiliary = dn.func.run(
            binding, tensors.state, steps=8, callbacks=callbacks
        )
        return auxiliary["callbacks"]

    results = {}
    for policy in ("auto", "off", "auto"):
        with dn.ctx(MEMBRANE_ASSEMBLY=policy), torch.no_grad():
            expected, actual = run(eager), run(bound)
        for name in ("v", "t"):
            torch.testing.assert_close(
                actual["trace"][name], expected["trace"][name], rtol=1e-12, atol=1e-12
            )
        if policy in results:
            torch.testing.assert_close(
                actual["trace"]["v"], results[policy], rtol=0, atol=0
            )
        results[policy] = actual["trace"]["v"].clone()
        assert actual["diagnostics"]["frames"].item() == 9
        assert scan.execution_report().strategy == "compiled_scan"
    assert not torch.allclose(results["auto"], results["off"])
    with dn.ctx(MEMBRANE_ASSEMBLY="validate"), torch.no_grad():
        with pytest.raises(RuntimeError, match="Invalid AFFINE"):
            run(bound)
    for policy in ("auto", "off"):
        with dn.ctx(MEMBRANE_ASSEMBLY=policy), torch.no_grad():
            actual = run(bound)
        torch.testing.assert_close(
            actual["trace"]["v"], results[policy], rtol=0, atol=0
        )
