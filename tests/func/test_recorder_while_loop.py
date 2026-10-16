"""Recorded inference uses a bounded loop without changing the rollout contract."""

from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import hh

DT = 0.01

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 14),
        reason="recorded inference loops require PyTorch 2.14 or newer",
    ),
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE,
        reason="branched Tree CPU execution requires dendra-solvers",
    ),
]


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _case(*, stimulus_kind=None, narrow_pulse=False):
    """Small batched HH tree with algebraic junctions, like the benchmark."""
    dtype = torch.float32 if narrow_pulse else torch.float64
    morphology = dn.Morphology(rhoa=100.0, cm=1.0)
    soma = morphology.section("soma", L=20.0, diam=20.0, nseg=1)
    trunk = morphology.section("trunk", L=30.0, diam=2.0, nseg=2)
    left = morphology.section("left", L=20.0, diam=1.0, nseg=2)
    right = morphology.section("right", L=25.0, diam=1.5, nseg=2)
    trunk.connect(soma.at(1.0), child_end=0)
    left.connect(trunk.at(1.0), child_end=0)
    right.connect(trunk.at(1.0), child_end=0)
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=dtype):
        model = dn.Tree.from_morphology(
            morphology,
            N=2,
            celsius=6.3,
            v_init=-62.0,
            dtype=dtype,
            integrator=dn.dhs(threads=32, imem=False),
        )
        model.internal_nodes.insert(hh)
        extra = None
        if stimulus_kind is not None:
            waveform = dn.mono_rect(
                amp=0.2 * dn.units.nA if stimulus_kind == "intra" else 1.0,
                delay=1000.070068359375 if narrow_pulse else 0.02,
                pw=0.001 if narrow_pulse else 0.04,
            )
            if stimulus_kind == "intra":
                model[:, 0].inject(waveform)
            else:
                field = torch.linspace(-1.0, 1.0, model.v.numel(), dtype=dtype)
                extra = (field.reshape(model.shape), waveform)
        model.requires_grad_(False)
        model.eval()
        model.initialize()
        if narrow_pulse:
            model.t.fill_(1000.0)
    functional, tensors = dn.func.make_functional(model, dt=DT, extra=extra)
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    prepared = functional.prepare(parameters, tensors.constants)
    return SimpleNamespace(
        model=model,
        functional=functional,
        tensors=tensors,
        parameters=parameters,
        prepared=prepared,
    )


@pytest.fixture
def case():
    return _case()


def _clone(tree):
    return torch.utils._pytree.tree_map(
        lambda value: value.detach().clone()
        if isinstance(value, torch.Tensor)
        else value,
        tree,
    )


def _close(actual, expected, *, exact=False):
    leaves, spec = torch.utils._pytree.tree_flatten(actual)
    reference, reference_spec = torch.utils._pytree.tree_flatten(expected)
    assert spec == reference_spec
    for value, other in zip(leaves, reference, strict=True):
        if isinstance(value, torch.Tensor):
            kwargs = {"rtol": 0.0, "atol": 0.0} if exact else {}
            torch.testing.assert_close(value, other, **kwargs)
        else:
            assert value == other


def _backend(graphs):
    def capture(graph, _examples):
        graphs.append(graph)
        return graph.forward

    return capture


def _has_while(graph):
    return any(
        node.op == "call_function" and node.target is torch.ops.higher_order.while_loop
        for node in graph.graph.nodes
    )


def _run(
    case, callbacks, steps, *, kernel=None, state=None, callback_state=None, inputs=None
):
    binding = (
        case.functional.bind(case.parameters, case.prepared)
        if kernel is None
        else kernel.bind(case.parameters, case.prepared)
    )
    return dn.func.run(
        binding,
        case.tensors.state if state is None else state,
        steps=steps,
        inputs=inputs,
        callbacks=callbacks,
        callback_state=callback_state,
    )


def _recorders(case):
    count = case.model.shape[-1]
    return case.functional.make_callbacks(
        {
            "full": dn.func.Recorder(["v", "hh.m", "t"]),
            "selected": dn.func.Recorder(["v"], node_indices=[0, count - 1]),
            "maximum": dn.func.Recorder(["v"], max_only=True),
            "partitioned": dn.func.Recorder(
                ["v"], max_only=True, partition=[count // 2, count - count // 2]
            ),
        }
    )


def test_recorded_while_matches_eager_and_explicit_unrolled_without_mutation(case):
    callbacks = _recorders(case)
    graphs = []
    kernel = case.functional.compile_rollout_chunk(8, backend=_backend(graphs))
    unrolled = case.functional.compile_rollout_chunk(
        8, execution="unrolled", backend="eager"
    )
    originals = _clone((case.parameters, case.tensors.constants, case.tensors.state))
    with torch.no_grad():
        expected = _run(case, callbacks, 8)
        actual = _run(case, callbacks, 8, kernel=kernel)
        saved = _clone((actual[0], dict(actual[1]["callbacks"])))
        repeated = _run(case, callbacks, 8, kernel=kernel)
        explicit = _run(case, callbacks, 8, kernel=unrolled)
    for result in (actual, repeated, explicit):
        _close(result[0], expected[0])
        _close(dict(result[1]["callbacks"]), dict(expected[1]["callbacks"]))
    _close((actual[0], dict(actual[1]["callbacks"])), saved, exact=True)
    _close(
        (case.parameters, case.tensors.constants, case.tensors.state),
        originals,
        exact=True,
    )
    assert actual[1]["callbacks"]["full"]["v"].shape[0] == 9
    assert actual[1]["callbacks"]["selected"]["v"].shape[-1] == 2
    assert case.model.t.item() == 0.0
    assert graphs and all(_has_while(graph) for graph in graphs)
    report = kernel.execution_report()
    assert report.strategy == "compiled_while_loop" and report.callbacks
    assert not report.grad_enabled and report.steps == 8
    assert unrolled.execution_report().strategy == "compiled_unrolled"


@pytest.mark.parametrize("steps", [0, 1, 3, 4, 5, 8, 9])
def test_initial_sample_zero_step_and_short_tail_contract(case, steps):
    callbacks = case.functional.make_callbacks(
        {"trace": dn.func.Recorder(["v", "hh.m", "t"])}
    )
    graphs = []
    kernel = case.functional.compile_rollout_chunk(4, backend=_backend(graphs))
    with torch.no_grad():
        expected = _run(case, callbacks, steps)
        actual = _run(case, callbacks, steps, kernel=kernel)
    _close(actual[0], expected[0])
    _close(dict(actual[1]["callbacks"]), dict(expected[1]["callbacks"]))
    assert actual[1]["callbacks"]["trace"]["v"].shape[0] == steps + 1
    if steps == 0:
        assert not graphs and kernel.execution_report() is None
    else:
        tail = steps % 4 or 4
        assert kernel.execution_report().steps == tail
        assert kernel.execution_report().strategy == (
            "compiled_while_loop" if tail == 4 else "compiled_unrolled"
        )
        assert any(_has_while(graph) for graph in graphs) == (steps >= 4)


def test_recorder_resume_does_not_duplicate_initial_or_chunk_boundary_samples(case):
    callbacks = case.functional.make_callbacks(
        {"trace": dn.func.Recorder(["v", "hh.m", "t"])}
    )
    kernel = case.functional.compile_rollout_chunk(4, backend="eager")
    with torch.no_grad():
        expected = _run(case, callbacks, 9)
        zero = _run(case, callbacks, 0, kernel=kernel)
        first = _run(
            case,
            callbacks,
            5,
            kernel=kernel,
            state=zero[0],
            callback_state=zero[1]["callbacks"].state,
        )
        second = _run(
            case,
            callbacks,
            4,
            kernel=kernel,
            state=first[0],
            callback_state=first[1]["callbacks"].state,
        )
        report = kernel.execution_report()
        empty = _run(
            case,
            callbacks,
            0,
            kernel=kernel,
            state=second[0],
            callback_state=second[1]["callbacks"].state,
        )
    assert kernel.execution_report() is report
    for name in ("v", "hh.m", "t"):
        combined = torch.cat(
            [
                part[1]["callbacks"]["trace"][name]
                for part in (zero, first, second, empty)
            ]
        )
        torch.testing.assert_close(combined, expected[1]["callbacks"]["trace"][name])
        assert combined.shape[0] == 10
    _close(second[0], expected[0])
    _close(empty[0], second[0], exact=True)


@pytest.mark.parametrize("stimulus_kind", ["intra", "extra"])
def test_bound_pulse_uses_recurrent_accepted_clock_inside_recorded_while(stimulus_kind):
    case = _case(stimulus_kind=stimulus_kind, narrow_pulse=True)
    callbacks = case.functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})
    graphs = []
    kernel = case.functional.compile_rollout_chunk(8, backend=_backend(graphs))
    with torch.no_grad():
        expected = _run(case, callbacks, 8)
        actual = _run(case, callbacks, 8, kernel=kernel)
        amplitude_name = next(
            name
            for name in case.parameters
            if name.startswith(f"stimulation.{stimulus_kind}.")
            and name.endswith(".amp")
        )
        amplitude = case.parameters[amplitude_name]
        case.parameters = {
            **case.parameters,
            amplitude_name: torch.zeros_like(amplitude),
        }
        absent = _run(case, callbacks, 8)
        # Reuse the captured kernel with a replacement tensor: make_fx must
        # not freeze the first call's parameter values into its step graph.
        compiled_absent = _run(case, callbacks, 8, kernel=kernel)
    trace = actual[1]["callbacks"]["trace"]
    reference = expected[1]["callbacks"]["trace"]
    torch.testing.assert_close(trace["t"], reference["t"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(trace["v"], reference["v"])
    assert not torch.equal(reference["v"][-1], absent[1]["callbacks"]["trace"]["v"][-1])
    _close(dict(compiled_absent[1]["callbacks"]), dict(absent[1]["callbacks"]))
    assert graphs and all(_has_while(graph) for graph in graphs)
    assert kernel.execution_report().strategy == "compiled_while_loop"


@pytest.mark.parametrize("steps", [5, 9])
def test_one_recorded_chunk_executes_its_internal_while_tail(case, steps):
    callbacks = case.functional.make_callbacks(
        {"trace": dn.func.Recorder(["v", "hh.m", "t"])}
    )
    graphs = []
    # A single invocation, not a width-four host schedule with a separate tail.
    kernel = case.functional.compile_rollout_chunk(steps, backend=_backend(graphs))
    with torch.no_grad():
        expected = _run(case, callbacks, steps)
        actual = _run(case, callbacks, steps, kernel=kernel)
    _close(actual[0], expected[0])
    _close(dict(actual[1]["callbacks"]), dict(expected[1]["callbacks"]))
    assert actual[1]["callbacks"]["trace"]["v"].shape[0] == steps + 1
    assert len(graphs) == 1 and _has_while(graphs[0])
    assert kernel.execution_report().strategy == "compiled_while_loop"
    assert kernel.execution_report().steps == steps


def test_recorded_while_accepts_inference_mode_with_ordinary_prepared_inputs(case):
    callbacks = case.functional.make_callbacks(
        {"trace": dn.func.Recorder(["v", "hh.m", "t"])}
    )
    graphs = []
    kernel = case.functional.compile_rollout_chunk(8, backend=_backend(graphs))
    original = _clone((case.parameters, case.tensors.constants, case.tensors.state))
    with torch.no_grad():
        expected = _run(case, callbacks, 8)
    with torch.inference_mode():
        actual = _run(case, callbacks, 8, kernel=kernel)
        repeated = _run(case, callbacks, 8, kernel=kernel)
    for result in (actual, repeated):
        _close(result[0], expected[0])
        _close(dict(result[1]["callbacks"]), dict(expected[1]["callbacks"]))
    _close(
        (case.parameters, case.tensors.constants, case.tensors.state),
        original,
        exact=True,
    )
    assert graphs and all(_has_while(graph) for graph in graphs)
    assert kernel.execution_report().strategy == "compiled_while_loop"
    assert not kernel.execution_report().grad_enabled


def test_explicit_unrolled_bypasses_final_state_inference_loop(case):
    graphs = []
    kernel = case.functional.compile_rollout_chunk(
        8, execution="unrolled", backend=_backend(graphs)
    )
    with torch.no_grad():
        expected = case.functional.rollout(
            case.parameters, case.prepared, case.tensors.state, steps=8
        )
        actual = kernel(case.parameters, case.prepared, case.tensors.state)
    _close(actual, expected)
    assert graphs and not any(_has_while(graph) for graph in graphs)
    assert kernel.execution_report().strategy == "compiled_unrolled"
    assert kernel.execution_report().requested_execution == "unrolled"
    assert kernel.capabilities.structured_inference == "none"


def test_gradient_mode_falls_back_and_preserves_trace_gradient(case):
    callbacks = case.functional.make_callbacks({"trace": dn.func.Recorder(["v"])})
    graphs = []
    kernel = case.functional.compile_rollout_chunk(4, backend=_backend(graphs))

    def evaluate(compiled):
        voltage = (
            case.tensors.state["integrator"]["v"].detach().clone().requires_grad_()
        )
        state = dict(case.tensors.state)
        state["integrator"] = dict(state["integrator"], v=voltage)
        result = _run(
            case, callbacks, 4, kernel=kernel if compiled else None, state=state
        )
        trace = result[1]["callbacks"]["trace"]["v"]
        gradient = torch.autograd.grad(trace.square().mean(), voltage)[0]
        return trace.detach(), gradient

    expected = evaluate(False)
    actual = evaluate(True)
    _close(actual, expected)
    assert torch.isfinite(actual[1]).all() and torch.count_nonzero(actual[1]) > 0
    assert graphs and not any(_has_while(graph) for graph in graphs)
    report = kernel.execution_report()
    assert report.strategy == "compiled_unrolled" and report.grad_enabled


def test_explicit_time_varying_drives_are_recorded_without_mutating_inputs(case):
    callbacks = case.functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})
    count = 8 * case.model.v.numel()
    shape = (8, *case.model.shape)
    ve = torch.linspace(-0.1, 0.1, count, dtype=case.model.dtype()).reshape(shape)
    intra = torch.linspace(-1.0e-9, 1.0e-9, count, dtype=case.model.dtype()).reshape(
        shape
    )
    inputs = dn.func.RolloutInput(ve=ve, intra=intra)
    original = _clone((ve, intra))
    graphs = []
    kernel = case.functional.compile_rollout_chunk(8, backend=_backend(graphs))
    with torch.no_grad():
        expected = _run(case, callbacks, 8, inputs=inputs)
        actual = _run(case, callbacks, 8, kernel=kernel, inputs=inputs)
    _close(actual[0], expected[0])
    _close(dict(actual[1]["callbacks"]), dict(expected[1]["callbacks"]))
    _close((ve, intra), original, exact=True)
    assert graphs and all(_has_while(graph) for graph in graphs)


def test_mixed_builtin_callback_plan_uses_while_with_explicit_carry(case):
    callbacks = case.functional.make_callbacks(
        {
            "trace": dn.func.Recorder(["v"]),
            "anomaly": dn.func.AnomalyDetector(),
        }
    )
    graphs = []
    kernel = case.functional.compile_rollout_chunk(4, backend=_backend(graphs))
    with torch.no_grad():
        expected = _run(case, callbacks, 4)
        actual = _run(case, callbacks, 4, kernel=kernel)
    _close(dict(actual[1]["callbacks"]), dict(expected[1]["callbacks"]))
    assert graphs and any(_has_while(graph) for graph in graphs)
    assert kernel.execution_report().strategy == "compiled_while_loop"


def test_recorded_while_graph_size_is_bounded_as_horizon_grows(case):
    callbacks = case.functional.make_callbacks(
        {"trace": dn.func.Recorder(["v", "hh.m", "t"])}
    )
    counts = []
    for steps in (8, 32):
        graphs = []
        kernel = case.functional.compile_rollout_chunk(steps, backend=_backend(graphs))
        with torch.no_grad():
            actual = _run(case, callbacks, steps, kernel=kernel)
            expected = _run(case, callbacks, steps)
        _close(dict(actual[1]["callbacks"]), dict(expected[1]["callbacks"]))
        assert len(graphs) == 1 and _has_while(graphs[0])
        counts.append(
            sum(
                len(list(module.graph.nodes))
                for module in graphs[0].modules()
                if isinstance(module, torch.fx.GraphModule)
            )
        )
    assert abs(counts[1] - counts[0]) <= 16, counts


def test_inductor_recorded_while_matches_eager(case):
    callbacks = case.functional.make_callbacks(
        {"trace": dn.func.Recorder(["v", "hh.m", "t"])}
    )
    kernel = case.functional.compile_rollout_chunk(4, backend="inductor")
    with torch.no_grad():
        expected = _run(case, callbacks, 4)
        actual = _run(case, callbacks, 4, kernel=kernel)
    _close(actual[0], expected[0])
    _close(dict(actual[1]["callbacks"]), dict(expected[1]["callbacks"]))
    assert kernel.execution_report().strategy == "compiled_while_loop"


def test_native_recorder_buffers_and_inference_gradient_mode_switch_are_independent(
    case,
):
    callbacks = case.functional.make_callbacks(
        {
            "trace": dn.func.Recorder(["v", "hh.m", "t"]),
        }
    )
    kernel = case.functional.compile_rollout_chunk(4, backend="inductor")
    original = _clone((case.parameters, case.tensors.constants, case.tensors.state))

    def initial_state(offset, *, requires_grad=False):
        voltage = (case.tensors.state["integrator"]["v"].detach() + offset).clone()
        voltage.requires_grad_(requires_grad)
        state = dict(case.tensors.state)
        state["integrator"] = dict(state["integrator"], v=voltage)
        return state, voltage

    first_state, _ = initial_state(0.0)
    second_state, _ = initial_state(0.5)
    with torch.no_grad():
        first = _run(case, callbacks, 4, kernel=kernel, state=first_state)
        assert kernel.execution_report().strategy == "compiled_while_loop"
        first_saved = _clone((first[0], dict(first[1]["callbacks"])))
        second = _run(case, callbacks, 4, kernel=kernel, state=second_state)
        second_expected = _run(case, callbacks, 4, state=second_state)
        assert kernel.execution_report().strategy == "compiled_while_loop"
    _close(second[0], second_expected[0])
    _close(dict(second[1]["callbacks"]), dict(second_expected[1]["callbacks"]))
    assert not torch.equal(
        first[1]["callbacks"]["trace"]["v"],
        second[1]["callbacks"]["trace"]["v"],
    )
    _close((first[0], dict(first[1]["callbacks"])), first_saved, exact=True)

    def differentiate(compiled):
        state, voltage = initial_state(-0.25, requires_grad=True)
        result = _run(
            case, callbacks, 4, kernel=kernel if compiled else None, state=state
        )
        trace = result[1]["callbacks"]["trace"]["v"]
        gradient = torch.autograd.grad(trace.square().mean(), voltage)[0]
        return trace, gradient

    with torch.enable_grad():
        expected_trace, expected_gradient = differentiate(False)
        training_trace, gradient = differentiate(True)
    torch.testing.assert_close(training_trace, expected_trace)
    torch.testing.assert_close(gradient, expected_gradient)
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    report = kernel.execution_report()
    assert report.strategy == "compiled_unrolled" and report.grad_enabled
    training_saved = training_trace.detach().clone()

    with torch.no_grad():
        final_inference = _run(case, callbacks, 4, kernel=kernel, state=first_state)
    _close((final_inference[0], dict(final_inference[1]["callbacks"])), first_saved)
    report = kernel.execution_report()
    assert report.strategy == "compiled_while_loop" and not report.grad_enabled
    _close((first[0], dict(first[1]["callbacks"])), first_saved, exact=True)
    torch.testing.assert_close(training_trace, training_saved, rtol=0.0, atol=0.0)
    _close(
        (case.parameters, case.tensors.constants, case.tensors.state),
        original,
        exact=True,
    )
