"""Multi-step integration oracles for the shared membrane G,J frame.

The reference disables only direct coefficient eligibility. It retains the same
HH state updater, original current evaluator, and solver's exact legacy current
arithmetic. Thus these tests isolate assembly, not the separate HH rate rewrite.
Broad compiler coverage uses bounded scan graphs; one short-chunk smoke retains
unrolled compatibility over the same full trajectory and gradient targets. No
historical timing or performance assertion belongs in this suite.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import MethodType

import networkx as nx
import pytest
import torch

import dendra as dn
import dendra.func._population as functional_population
from dendra.models.integrators import dhs_bt
from dendra.models.integrators.implicit import _bwd_euler_sc_multi
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mod import hh, pas

DT = 0.01
STEPS = 8
FIELDS = ("gnabar", "gkbar", "gl", "ena", "ek", "el")
KINDS = (
    "sc",
    "packed_sc",
    "ub_pcr",
    "ub_native",
    "dhs",
    "packed_dhs",
    "block_ub",
    "block_dhs",
)
NATIVE = set(KINDS) - {"sc", "packed_sc", "ub_pcr"}

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


@pytest.fixture(autouse=True)
def _one_worker():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _graph(*, extracellular=False):
    """Branched storage-order topology with an algebraic zero-area junction."""
    graph = nx.DiGraph()
    for node, area in enumerate((100.0, 0.0, 130.0, 220.0, 90.0)):
        extra = (
            dict(xraxial=[1.7, 2.3], xc=[0.08, 0.13], xg=[1.5e-4, 2.4e-4])
            if extracellular
            else {}
        )
        graph.add_node(
            node,
            name=f"node{node}",
            kind="junction" if area == 0 else "compartment",
            L=0.0 if area == 0 else 10.0,
            diam=2.0,
            Ra=100.0,
            cm=1.0,
            area=area,
            **extra,
        )
    for parent, child, resistance in (
        (3, 1, 1.7e8),
        (1, 0, 2.1e8),
        (1, 2, 2.9e8),
        (2, 4, 3.7e8),
    ):
        graph.add_edge(parent, child, R_ohm=resistance, L=1.0)
    return graph


def _model(kind, dtype):
    if kind in NATIVE and not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("native CPU assembly cases require dendra-solvers")
    options = dict(dtype=dtype, v_init=-62.0, celsius=23.0)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
        if kind == "sc":
            model = dn.SingleCompartment(
                N=2, C=3, integrator=dn.bwd_euler_sc(imem=False), **options
            )
        elif kind.startswith("ub_"):
            method = "pcr" if kind == "ub_pcr" else "thomas"
            model = dn.Unmyelinated(
                [2.0, 2.5],
                L=3.0,
                dx=1.0,
                integrator=dn.bwd_euler_ub(method=method, imem=False),
                **options,
            )
        elif kind in ("dhs", "block_dhs"):
            extracellular = kind == "block_dhs"
            cls = dn.ExtCellTree if extracellular else dn.Tree
            solver = (
                dhs_bt(threads=32, imem=False)
                if extracellular
                else dn.dhs(threads=32, imem=False)
            )
            model = cls.from_graph(
                _graph(extracellular=extracellular), N=2, integrator=solver, **options
            )
        elif kind == "block_ub":
            model = dn.ExtCellAxon(
                [5.0, 8.0],
                n_comp=4,
                integrator=dn.bwd_euler_bt(method="thomas", imem=False),
                **options,
            )
            model.xraxial.fill_(2.0)
            model.xc.fill_(0.1)
            model.xg.fill_(2.0e-4)
        elif kind in ("packed_sc", "packed_dhs"):
            point = dn.SingleCompartment(N=1, C=2, **options)
            other = (
                dn.SingleCompartment(N=2, C=1, cm=1.3, **options)
                if kind == "packed_sc"
                else dn.Tree.from_graph(_graph(), N=1, **options)
            )
            point.insert(hh)
            other.insert(hh)
            model = dn.concat_models(
                {"point": point, "other": other},
                threads=32,
                celsius=23.0,
                write_back=True,
            )
        else:
            raise AssertionError(kind)
        if not kind.startswith("packed_"):
            model.insert(hh)
        model.initialize()
        model.train()
        if hasattr(model, "vc"):
            shell = torch.linspace(-1.0, 1.0, model.v.numel(), dtype=dtype).reshape(
                model.shape
            )
            with torch.no_grad():
                model.vc[..., 1].copy_(shell)
                model.vc[..., 2].copy_(-0.5 * shell)
                model.vc[..., 0].copy_(model.v + shell)
        model.integrator._initialize(
            model, torch.tensor(DT, dtype=dtype), force=True, compile_scope="population"
        )
        if kind == "packed_sc":
            assert isinstance(model.integrator, _bwd_euler_sc_multi)
    return model


def _clone(tree):
    return torch.utils._pytree.tree_map(lambda x: x.detach().clone(), tree)


def _parameter_names(parameters):
    names = tuple(
        name
        for name in parameters
        if ".hh." in name and name.rsplit(".", 1)[-1].split("_")[0] in FIELDS
    )
    assert {name.rsplit(".", 1)[-1].split("_")[0] for name in names} == set(FIELDS)
    return names


def _inputs(model, *, gradients=True):
    dtype = model.dtype()
    amplitude = torch.tensor(0.7, dtype=dtype)
    initial = torch.linspace(-65.0, -55.0, model.v.numel(), dtype=dtype).reshape(
        model.shape
    )
    initial.requires_grad_(gradients)
    drive = torch.linspace(
        -0.8e-9, 1.1e-9, STEPS * model.v.numel(), dtype=dtype
    ).reshape(STEPS, *model.shape)
    # Do not directly inject current into algebraic zero-area junctions.
    drive = drive * (model.area != 0).to(dtype)
    drive.requires_grad_(gradients)
    ve = torch.linspace(-0.6, 0.8, drive.numel(), dtype=dtype).reshape_as(drive)
    return amplitude, initial, drive, ve


def _loss(trace):
    # Exclude the initial sample: initial-voltage gradients must cross a step.
    values = trace[1:]
    weights = torch.linspace(0.7, 1.3, values.numel(), dtype=values.dtype).reshape_as(
        values
    )
    return (weights * ((values + 70.0) / 20.0).square()).mean()


def _initial_state(state, initial):
    state = _clone(state)
    state["integrator"]["v"] = initial
    if "vc" in state["integrator"]:
        old = state["integrator"]["vc"]
        state["integrator"]["vc"] = torch.cat(
            ((initial + old[..., 1]).unsqueeze(-1), old[..., 1:]), dim=-1
        )
    return state


def _evaluate(
    kind,
    dtype,
    mode,
    *,
    gradients=True,
    backend=None,
    execution="scan",
    chunk_steps=4,
):
    model = _model(kind, dtype)
    # This eager discriminator prevents a silently inactive direct branch from
    # making both sides of the numerical test use the legacy evaluator.
    frame = model.integrator._membrane_frame(model.v, call_local=True)
    direct_used = frame[2] is None
    functional, tensors = dn.func.make_functional(model, dt=DT)
    amplitude, initial, drive, ve = _inputs(model, gradients=gradients)
    parameters = (
        dict(model.named_parameters())
        if mode == "population"
        else {
            name: value.detach().clone() for name, value in tensors.parameters.items()
        }
    )
    names = _parameter_names(parameters)
    for name in names:
        parameters[name].requires_grad_(gradients)
    targets = (drive, initial, *(parameters[name] for name in names))
    state = _initial_state(tensors.state, initial)
    context = nullcontext() if gradients else torch.no_grad()
    with context:
        if mode == "population":
            model.v = initial
            if hasattr(model, "vc"):
                old = model.vc
                model.vc = torch.cat(
                    ((initial + old[..., 1]).unsqueeze(-1), old[..., 1:]), dim=-1
                )
            values = [initial]
            for index in range(STEPS):
                model.integrator.step(model, DT, ve[index], amplitude * drive[index])
                model.t = model.t + DT
                values.append(model.v)
            if kind.startswith("packed_"):
                offset = 0
                for component in model.populations.values():
                    width = component.v.numel()
                    torch.testing.assert_close(
                        component.v,
                        model.v.reshape(-1)[offset : offset + width].reshape_as(
                            component.v
                        ),
                        rtol=0,
                        atol=0,
                    )
                    offset += width
                assert offset == model.v.numel()
            trace, final = torch.stack(values), functional.extract(model).state
        else:
            prepared = functional.prepare(parameters, tensors.constants)
            callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
            kernel = None
            if mode == "chunk":
                kernel = functional.compile_rollout_chunk(
                    chunk_steps,
                    execution=execution,
                    backend=backend or "aot_eager",
                )
            elif mode == "while":
                kernel = functional.compile_rollout_chunk(4, backend=backend or "eager")
            elif mode != "functional":
                raise AssertionError(mode)
            binding = (kernel or functional).bind(parameters, prepared)
            final, auxiliary = dn.func.run(
                binding,
                state,
                steps=STEPS,
                inputs=dn.func.RolloutInput(intra=amplitude * drive, ve=ve),
                callbacks=callbacks,
            )
            trace = auxiliary["callbacks"]["trace"]["v"]
            if mode == "while":
                report = kernel.execution_report()
                assert report.requested_execution == "default"
                assert report.strategy == "compiled_while_loop"
            if mode == "chunk":
                report = kernel.execution_report()
                assert report.requested_execution == execution
                expected_strategy = {
                    "scan": "compiled_scan",
                    "unrolled": "compiled_unrolled",
                }[execution]
                assert report.strategy == expected_strategy
                assert report.steps == chunk_steps
                assert report.callbacks
                assert report.grad_enabled == gradients
        assert trace.shape == (STEPS + 1, *model.shape)
        loss = _loss(trace)
        grads = torch.autograd.grad(loss, targets) if gradients else ()
    return _clone((trace, final, loss, grads)), direct_used


def _assert_close(actual, expected, dtype):
    leaves, spec = torch.utils._pytree.tree_flatten(actual)
    others, other_spec = torch.utils._pytree.tree_flatten(expected)
    assert spec == other_spec
    rtol, atol = (3e-5, 3e-5) if dtype == torch.float32 else (2e-10, 2e-11)
    for value, reference in zip(leaves, others, strict=True):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, reference, rtol=rtol, atol=atol)
    for observed, reference in zip(actual[3], expected[3], strict=True):
        assert torch.count_nonzero(reference), "gradient target was not exercised"
        scale = reference.abs().max()
        torch.testing.assert_close(
            observed / scale, reference / scale, rtol=rtol, atol=rtol
        )


def _patch_eligibility(patch, method, *, require_direct_dtype=False):
    patch.setattr(MechanismHandler, "_can_evaluate_direct_membrane_frame", method)
    # These are intentional test-only reference methods. Keep functional
    # identity validation active for every other handler method.
    patch.setitem(
        functional_population._STANDARD_FUNCTIONAL_HANDLER_HOOKS,
        "_can_evaluate_direct_membrane_frame",
        method,
    )
    if require_direct_dtype:
        compatible = MechanismHandler._direct_membrane_dtype_compatible

        def checked_dtype(handler, voltage):
            assert compatible(handler, voltage), (
                "direct frame fell back after ASSIGNED dtype check"
            )
            return True

        patch.setattr(
            MechanismHandler, "_direct_membrane_dtype_compatible", checked_dtype
        )
        patch.setitem(
            functional_population._STANDARD_FUNCTIONAL_HANDLER_HOOKS,
            "_direct_membrane_dtype_compatible",
            checked_dtype,
        )


def _compare(
    monkeypatch,
    kind,
    dtype,
    mode,
    *,
    gradients=True,
    backend=None,
    execution="scan",
    chunk_steps=4,
):
    eligible = MechanismHandler._can_evaluate_direct_membrane_frame

    def require_direct(handler):
        # Check the actual functional/compiled clone, not just the original
        # imperative handler. A proxy-proof regression must not silently turn
        # the "direct" half of this test into a second legacy execution.
        assert eligible(handler), "direct membrane eligibility lost during execution"
        return True

    with monkeypatch.context() as patch:
        _patch_eligibility(patch, require_direct, require_direct_dtype=True)
        direct, used = _evaluate(
            kind,
            dtype,
            mode,
            gradients=gradients,
            backend=backend,
            execution=execution,
            chunk_steps=chunk_steps,
        )
    assert used, f"direct membrane frame inactive for {kind}"
    with monkeypatch.context() as patch:
        _patch_eligibility(patch, lambda self: False)
        legacy, used = _evaluate(
            kind,
            dtype,
            mode,
            gradients=gradients,
            backend=backend,
            execution=execution,
            chunk_steps=chunk_steps,
        )
        assert not used
    _assert_close(direct, legacy, dtype)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("mode", ("population", "functional"))
def test_all_implicit_integrators_match_legacy_full_trace_state_and_gradients(
    monkeypatch, kind, dtype, mode
):
    _compare(monkeypatch, kind, dtype, mode)


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_compiled_scan_backward_matches_legacy(monkeypatch, kind):
    _compare(monkeypatch, kind, torch.float64, "chunk")


@pytest.mark.parametrize("kind", KINDS)
def test_compiled_while_full_recorder_matches_legacy(monkeypatch, kind):
    _compare(monkeypatch, kind, torch.float32, "while", gradients=False)


@pytest.mark.parametrize("kind", ("ub_pcr", "dhs", "block_dhs"))
@pytest.mark.parametrize(
    "mode", ("population", pytest.param("chunk", marks=pytest.mark.slow), "while")
)
def test_four_worker_forward_and_backward_match_legacy(monkeypatch, kind, mode):
    torch.set_num_threads(4)
    _compare(monkeypatch, kind, torch.float64, mode, gradients=mode != "while")


@pytest.mark.slow
@pytest.mark.parametrize("workers", (1, 4))
@pytest.mark.parametrize("kind", ("sc", "dhs", "block_dhs"))
@pytest.mark.parametrize("mode", ("chunk", "while"))
def test_inductor_generated_numeric_paths(monkeypatch, workers, kind, mode):
    """Real generated numerical kernels, not just Dynamo/AOT graph capture."""
    torch.set_num_threads(workers)
    _compare(
        monkeypatch,
        kind,
        torch.float32,
        mode,
        gradients=mode != "while",
        backend="inductor",
    )


@pytest.mark.slow
def test_inductor_unrolled_smoke_matches_legacy(monkeypatch):
    """Retain genuine repeated-body codegen with the smallest unrolled graph."""
    _compare(
        monkeypatch,
        "dhs",
        torch.float32,
        "chunk",
        backend="inductor",
        execution="unrolled",
        chunk_steps=2,
    )


@pytest.mark.parametrize(
    "kind", ("sc", "packed_sc", "ub_pcr", "dhs", "packed_dhs", "block_ub", "block_dhs")
)
def test_parameter_only_vmap_including_empty_lanes(kind):
    model = _model(kind, torch.float64)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    name = next(n for n in _parameter_names(tensors.parameters) if ".gnabar_" in n)
    base = tensors.parameters[name].detach()
    amplitude, initial, drive, ve = _inputs(model, gradients=False)
    state = _initial_state(tensors.state, initial)

    def response(value):
        parameters = {**tensors.parameters, name: value}
        final, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(intra=amplitude * drive, ve=ve),
        )
        return final["integrator"]["v"], auxiliary["v"]

    values = torch.stack((base * 0.9, base, base * 1.1))
    actual = torch.vmap(response)(values)
    explicit = [response(value) for value in values]
    for index, observed in enumerate(actual):
        torch.testing.assert_close(
            observed, torch.stack([v[index] for v in explicit]), rtol=2e-10, atol=2e-11
        )
    empty = torch.vmap(response)(values[:0])
    for observed, reference in zip(empty, actual, strict=True):
        assert observed.shape == (0, *reference.shape[1:])


@pytest.mark.parametrize("kind", ("sc", "dhs", "block_dhs"))
def test_float64_directional_finite_difference_for_all_targets(kind):
    model = _model(kind, torch.float64)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    names = _parameter_names(tensors.parameters)
    amplitude, initial, drive, ve = _inputs(model)
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    targets = (drive, initial, *(parameters[name].requires_grad_() for name in names))

    def objective(values):
        parameters_local = {**parameters, **dict(zip(names, values[2:], strict=True))}
        prepared = functional.prepare(parameters_local, tensors.constants)
        state = _initial_state(tensors.state, values[1])
        callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
        _, auxiliary = dn.func.run(
            functional.bind(parameters_local, prepared),
            state,
            steps=STEPS,
            inputs=dn.func.RolloutInput(intra=amplitude * values[0], ve=ve),
            callbacks=callbacks,
        )
        return _loss(auxiliary["callbacks"]["trace"]["v"])

    gradients = torch.autograd.grad(objective(targets), targets)
    for index, (target, gradient) in enumerate(zip(targets, gradients, strict=True)):
        assert torch.count_nonzero(gradient)
        direction = torch.ones_like(target) * max(
            float(target.detach().abs().max()), 1e-12
        )
        eps = 1e-4
        upper, lower = list(targets), list(targets)
        upper[index] = target.detach() + eps * direction
        lower[index] = target.detach() - eps * direction
        with torch.no_grad():
            finite_difference = (objective(upper) - objective(lower)) / (2 * eps)
        analytic = (gradient * direction).sum()
        torch.testing.assert_close(analytic, finite_difference, rtol=3e-5, atol=2e-8)


def test_unknown_custom_handler_uses_unchanged_current_evaluator():
    class CustomHandler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def i(self, voltage):
            self.calls += 1
            return voltage.square() + 0.3, 2 * voltage

    model = _model("sc", torch.float64)
    custom = CustomHandler()
    model.integrator.mech = custom
    voltage = model.v.detach().clone().requires_grad_()
    conductance, source, current, ion_current, ion_conductance = (
        model.integrator._membrane_frame(voltage)
    )
    assert custom.calls == 1
    assert current is not None
    assert ion_current == ion_conductance == ()
    torch.testing.assert_close(current, voltage.square() + 0.3, rtol=0, atol=0)
    torch.testing.assert_close(source, conductance * voltage - current, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(source.sum(), voltage)[0], 2 * voltage
    )


def _public_run(kind, *, steps, compiled):
    model = _model(kind, torch.float64)
    with dn.ctx(REQUIRE_GRAD=1, DTYPE=torch.float64):
        waveform = dn.mono_rect(
            amp=0.5 * dn.units.nA, delay=DT, pw=min(0.8, (steps - 2) * DT)
        ).to(dtype=torch.float64)
        model[:, 0].inject(waveform)
        model.initialize()
    model.train()
    functional, _ = dn.func.make_functional(model, dt=DT)
    parameters = dict(model.named_parameters())
    names = _parameter_names(parameters)
    amplitude = dict(waveform.named_parameters())["amp"]
    initial = torch.linspace(
        -65.0, -55.0, model.v.numel(), dtype=torch.float64
    ).reshape(model.shape)
    initial.requires_grad_()
    model.v = initial
    recorder = dn.Recorder(["v"])
    with dn.ctx(JIT=int(compiled), BACKEND="aot_eager", FULLGRAPH=1, REQUIRE_GRAD=1):
        model.run(tstop=steps * DT, dt=DT, callbacks=[recorder], progressbar=False)
    trace = recorder.stack("v")
    assert trace.shape == (steps + 1, *model.shape)
    loss = _loss(trace)
    targets = (amplitude, initial, *(parameters[name] for name in names))
    gradients = torch.autograd.grad(loss, targets)
    return _clone((trace, functional.extract(model).state, loss, gradients))


@pytest.mark.parametrize(
    "kind,steps,compiled", (("sc", 8, True), ("dhs", 8, True), ("dhs", 400, False))
)
def test_public_population_run_recorder_and_long_spiking_trace(
    monkeypatch, kind, steps, compiled
):
    eligible = MechanismHandler._can_evaluate_direct_membrane_frame

    def require_direct(handler):
        assert eligible(handler), "public Population.run lost direct eligibility"
        return True

    with monkeypatch.context() as patch:
        _patch_eligibility(patch, require_direct, require_direct_dtype=True)
        direct = _public_run(kind, steps=steps, compiled=compiled)
    with monkeypatch.context() as patch:
        _patch_eligibility(patch, lambda self: False)
        legacy = _public_run(kind, steps=steps, compiled=compiled)
    _assert_close(direct, legacy, torch.float64)
    if steps >= 200:
        assert direct[0][1:].max() > 0, "long fixture must evoke a spike"


def test_passive_block_ub_retains_zero_gradient_to_shadow_membrane_voltage(monkeypatch):
    """Circuit carry owns capacitance; the redundant v input has a zero AD edge."""
    if not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("block UB requires dendra-solvers")

    def response(*, direct):
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False):
            model = dn.ExtCellAxon(
                [5.0],
                n_comp=3,
                dtype=torch.float64,
                integrator=dn.bwd_euler_bt(method="thomas", imem=False),
            )
            model.insert(pas, g=0.001, e=-70.0)
            model.initialize()
        model.integrator._initialize(model, DT, force=True)
        assert (
            model.integrator._membrane_frame(model.v, call_local=True)[2] is None
        ) == direct
        functional, tensors = dn.func.make_functional(model, dt=DT)
        state = _clone(tensors.state)
        voltage = state["integrator"]["v"].requires_grad_()
        # Deliberately do not change vc when marking this redundant input.
        final, _ = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(
                intra=torch.zeros((STEPS, *model.shape), dtype=torch.float64)
            ),
        )
        gradient = torch.autograd.grad(final["integrator"]["v"].sum(), voltage)[0]
        # The legacy g*v - itot formula cancels two floating-point AD paths;
        # the direct source has an explicit, exact zero edge to this input.
        atol = 0 if direct else torch.finfo(gradient.dtype).eps
        torch.testing.assert_close(
            gradient, torch.zeros_like(voltage), rtol=0, atol=atol
        )
        return final["integrator"]["v"].detach(), gradient.detach()

    direct = response(direct=True)
    with monkeypatch.context() as patch:
        _patch_eligibility(patch, lambda self: False)
        legacy = response(direct=False)
    for actual, expected in zip(direct, legacy, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)


class _StateOnlyAssignedProbe(Mechanism):
    Mechanism.CARRY("steps_seen")
    Mechanism.ASSIGNED("stage_snapshot")

    def initial_values(self, voltage, values):
        return {"steps_seen": torch.zeros_like(voltage)}

    def assigned_values(self, voltage, values):
        return {"stage_snapshot": values["steps_seen"].clone()}

    def advance(self, voltage, dt, values):
        return {"steps_seen": values["steps_seen"] + 1}


@pytest.mark.parametrize("imem", (False, True))
def test_scalar_dhs_without_currents_preserves_state_only_assigned_phase(
    monkeypatch, imem
):
    """Baseline DHS omits current evaluation when its imperative current map is empty."""
    if not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("scalar DHS requires dendra-solvers")
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, IMEM=imem):
        model = dn.Tree.from_graph(
            _graph(),
            N=1,
            dtype=torch.float64,
            v_init=torch.linspace(-67.0, -57.0, 5, dtype=torch.float64),
            integrator=dn.dhs(threads=32, imem=imem),
        )
        model.insert(_StateOnlyAssignedProbe)
        model.initialize()
        model.integrator._initialize(model, DT, force=True)
    handler = model.integrator.mech
    probe = next(iter(handler.mechanisms.values()))
    assert not handler.currents
    # This is deliberately a mechanism whose ASSIGNED work would run if the
    # current evaluator were entered despite the empty current map.
    assert handler._current_assigned_plan
    assigned = probe.assigned_values
    seen = []

    def counted_assigned(voltage, values):
        seen.append(values["steps_seen"].detach().clone())
        return assigned(voltage, values)

    monkeypatch.setattr(probe, "assigned_values", counted_assigned)
    for index in range(4):
        model.integrator.step(model, DT)
        assert len(seen) == index + 1
        torch.testing.assert_close(
            seen[-1], torch.full_like(probe.steps_seen, index), rtol=0, atol=0
        )
        torch.testing.assert_close(
            probe.stage_snapshot,
            torch.full_like(probe.steps_seen, index),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            probe.steps_seen,
            torch.full_like(probe.steps_seen, index + 1),
            rtol=0,
            atol=0,
        )
        assert torch.isfinite(model.v).all()
        if imem:
            assert torch.isfinite(model.i_membrane).all()


@pytest.mark.parametrize("kind", ("sc", "ub_pcr", "ub_native"))
def test_legacy_voltage_update_instance_override_receives_actual_current(
    monkeypatch, kind
):
    """An old-signature user hook remains the voltage-update authority."""
    actual = _model(kind, torch.float64)
    reference = _model(kind, torch.float64)
    assert actual.integrator._membrane_frame(actual.v)[2] is None
    original = actual.integrator._voltage_update
    calls = []

    # Deliberately no membrane_source keyword and no **kwargs escape hatch.
    def old_signature(
        owner, voltage, dt, itot, gtot, ve=None, intra=None, *, solver=None
    ):
        assert owner is actual.integrator
        assert itot is not None
        assert torch.isfinite(itot).all()
        calls.append(itot.detach().clone())
        return original(voltage, dt, itot, gtot, ve, intra, solver=solver)

    monkeypatch.setattr(
        actual.integrator,
        "_voltage_update",
        MethodType(old_signature, actual.integrator),
    )
    _, _, drive, ve = _inputs(actual, gradients=False)

    def rollout(model):
        trace = [model.v.detach().clone()]
        with torch.no_grad():
            for index in range(STEPS):
                model.integrator.step(model, DT, ve[index], drive[index])
                trace.append(model.v.detach().clone())
        return torch.stack(trace)

    observed = rollout(actual)
    assert len(calls) == STEPS
    with monkeypatch.context() as patch:
        _patch_eligibility(patch, lambda self: False)
        expected = rollout(reference)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
