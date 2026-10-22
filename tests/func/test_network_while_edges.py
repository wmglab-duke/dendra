"""Network inference loops retain explicit policy, bounded graphs, and ownership."""

from __future__ import annotations

import gc
import weakref
from types import MappingProxyType

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_network import DT, DTYPE, _clone, _close, _stim_network

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _case(*, training=False, twins=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        cell = dn.SingleCompartment(
            N=2, C=1, v_init=-61.0, integrator=dn.bwd_euler_sc(imem=False)
        )
        cell.insert(pas, g=0.001, e=-65.0)
        populations = {"cell": cell}
        if twins:
            other = dn.SingleCompartment(
                N=1, C=1, v_init=-63.0, integrator=dn.bwd_euler_sc(imem=False)
            )
            other.insert(pas, g=0.002, e=-65.0)
            populations["other"] = other
        model = dn.Network(populations)
        model.train(training)
        model.initialize(DT)
    functional, tensors = dn.func.make_functional_network(model)
    return (
        functional,
        tensors,
        functional.prepare(tensors.parameters, tensors.constants),
    )


def _backend(graphs):
    def backend(graph, examples):
        del examples
        graphs.append(graph)
        return graph.forward

    return backend


def _has_while(graph):
    return any(
        node.op == "call_function" and node.target is torch.ops.higher_order.while_loop
        for module in graph.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    )


def _state_with_voltage(state, voltage):
    cell = state["populations"]["cell"]
    return {
        **state,
        "populations": {
            **state["populations"],
            "cell": {**cell, "integrator": {**cell["integrator"], "v": voltage}},
        },
    }


@pytest.mark.parametrize(
    "options",
    ({"inference": False}, {"checkpoint_every": 2}, {"disable": True}),
)
def test_while_runner_rejects_training_checkpoint_and_disabled_policies(options):
    functional, _, _ = _case(training=True)
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError),
        match="inference|while|checkpoint|disable",
    ):
        dn.compile(
            functional._source,
            dt=DT,
            steps=4,
            execution="while",
            backend="eager",
            **options,
        )


def test_training_plan_while_rejects_ordinary_grad_and_accepts_inference():
    functional, tensors, prepared = _case(training=True)
    graphs = []
    chunk = functional.compile_rollout_chunk(
        4, execution="while", backend=_backend(graphs)
    )
    frozen = _clone(tensors.state, detached=True)
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="inference|scan|grad"
    ):
        chunk(tensors.parameters, prepared, tensors.state)
    assert not graphs and chunk.execution_report() is None
    _close(tensors.state, frozen)
    with torch.no_grad():
        actual, _ = chunk(tensors.parameters, prepared, tensors.state)
        expected, _ = functional.rollout(
            tensors.parameters, prepared, tensors.state, steps=4
        )
    _close(actual, expected)
    assert any(_has_while(graph) for graph in graphs)
    assert chunk.execution_report().strategy == "compiled_while_loop"
    assert not chunk.execution_report().grad_enabled


@pytest.mark.parametrize("transform", ("grad", "jacfwd", "vmap"))
def test_while_transforms_use_the_eager_training_recurrence(transform):
    functional, tensors, prepared = _case(training=True)

    def reject_backend(graph, examples):
        raise AssertionError("Functional transforms must not invoke the while backend")

    chunk = functional.compile_rollout_chunk(
        4, execution="while", backend=reject_backend
    )

    def loss(voltage, *, compiled):
        state = _state_with_voltage(tensors.state, voltage)
        final, _ = (
            chunk(tensors.parameters, prepared, state)
            if compiled
            else functional.rollout(tensors.parameters, prepared, state, steps=4)
        )
        return final["populations"]["cell"]["integrator"]["v"].square().mean()

    voltage = tensors.state["populations"]["cell"]["integrator"]["v"].detach()
    if transform == "vmap":
        voltage = torch.stack((voltage, voltage + 1))
    operation = getattr(torch.func, transform)
    expected = operation(lambda value: loss(value, compiled=False))(voltage)
    actual = operation(lambda value: loss(value, compiled=True))(voltage)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    assert chunk.execution_report().strategy == "eager_transform_fallback"


def test_while_direct_forward_ad_uses_eager_fallback():
    functional, tensors, prepared = _case(training=True)

    def reject_backend(graph, examples):
        raise AssertionError("Forward AD must not invoke the while backend")

    chunk = functional.compile_rollout_chunk(
        4, execution="while", backend=reject_backend
    )
    voltage = tensors.state["populations"]["cell"]["integrator"]["v"].detach()
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(voltage, torch.ones_like(voltage))
        state = _state_with_voltage(tensors.state, dual)
        actual, _ = chunk(tensors.parameters, prepared, state)
        expected, _ = functional.rollout(tensors.parameters, prepared, state, steps=4)
        observed = torch.autograd.forward_ad.unpack_dual(
            actual["populations"]["cell"]["integrator"]["v"]
        )
        reference = torch.autograd.forward_ad.unpack_dual(
            expected["populations"]["cell"]["integrator"]["v"]
        )
        torch.testing.assert_close(observed.primal, reference.primal)
        torch.testing.assert_close(observed.tangent, reference.tangent)
        assert observed.tangent is not None and observed.tangent.abs().sum() > 0
    assert chunk.execution_report().strategy == "eager_forward_ad_fallback"


def test_disabled_low_level_while_uses_eager_recurrence():
    functional, tensors, prepared = _case()

    def reject_backend(graph, examples):
        raise AssertionError("Disabled compilation must not invoke the backend")

    chunk = functional.compile_rollout_chunk(
        4, execution="while", backend=reject_backend, disable=True
    )
    actual, _ = chunk(tensors.parameters, prepared, tensors.state)
    expected, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, steps=4
    )
    _close(actual, expected)
    assert chunk.execution_report().strategy == "eager_disabled"


@pytest.mark.parametrize("steps", (0, 1, 3, 4, 5, 9))
def test_while_zero_steps_and_short_tails_report_the_dispatched_strategy(steps):
    functional, tensors, _ = _case()
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(("cell.v", "t"))})
    graphs = []
    chunk = functional.compile_rollout_chunk(
        4, execution="while", backend=_backend(graphs)
    )
    expected, reference = functional.prepare_and_rollout(
        *tensors, steps=steps, callbacks=callbacks
    )
    actual, auxiliary = functional.run(
        *tensors, steps=steps, callbacks=callbacks, compiled_chunk=chunk, chunklength=4
    )
    _close(actual, expected)
    _close(auxiliary["callbacks"]["trace"], reference["callbacks"]["trace"])
    assert auxiliary["callbacks"]["trace"]["cell.v"].shape[0] == steps + 1
    if not steps:
        assert not graphs and chunk.execution_report() is None
    else:
        tail = steps % 4 or 4
        report = chunk.execution_report()
        assert report.steps == tail and report.requested_execution == "while"
        assert report.strategy == (
            "compiled_while_loop" if tail == 4 else "compiled_unrolled"
        )
        assert any(_has_while(graph) for graph in graphs) is (steps >= 4)


class _Counter(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["clock"]["step"].clone(), None

    def update(self, carry, state, auxiliary):
        del state, auxiliary
        return carry + 1, None

    def finalize(self, carry, emissions):
        del emissions
        return carry


@pytest.mark.parametrize(("steps", "disabled"), ((1, False), (4, False), (4, True)))
def test_while_rejects_custom_callbacks_before_backend_or_state_changes(
    steps, disabled
):
    functional, tensors, prepared = _case()
    callbacks = functional.make_callbacks({"counter": _Counter()})
    graphs = []
    chunk = functional.compile_rollout_chunk(
        steps, execution="while", backend=_backend(graphs), disable=disabled
    )
    frozen = _clone(tensors.state, detached=True)
    with pytest.raises(
        (ValueError, dn.func.FunctionalizationError), match="callback|built-in|Recorder"
    ):
        chunk(tensors.parameters, prepared, tensors.state, callbacks=callbacks)
    assert not graphs and chunk.execution_report() is None
    _close(tensors.state, frozen)


@pytest.mark.parametrize("missing", (True, False))
def test_while_netstim_requires_a_matching_explicit_draw_tape(missing):
    functional, tensors = dn.func.make_functional_network(_stim_network())
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(4, execution="while", backend="eager")
    inputs = (
        None
        if missing
        else dn.func.NetworkRolloutInput(
            netstim=functional.sample_draw_tape(tensors.state, 3)
        )
    )
    frozen = _clone(tensors.state, detached=True)
    with torch.no_grad(), pytest.raises(ValueError, match="tape|steps|samples|axis"):
        chunk(tensors.parameters, prepared, tensors.state, inputs)
    assert chunk.execution_report() is None
    _close(tensors.state, frozen)


def test_while_callback_caches_release_collections_and_live_tensor_operands():
    functional, tensors, _ = _case()
    chunk = functional.compile_rollout_chunk(4, execution="while", backend="eager")

    def dispatch():
        parameters = _clone(tensors.parameters, detached=True)
        prepared = functional.prepare(parameters, tensors.constants)
        state = _clone(tensors.state, detached=True)
        callbacks = functional.make_callbacks({"trace": dn.func.Recorder("cell.v")})
        final, auxiliary = chunk(parameters, prepared, state, callbacks=callbacks)
        return tuple(
            weakref.ref(value)
            for value in (
                callbacks,
                next(iter(parameters["populations"]["cell"].values())),
                state["populations"]["cell"]["integrator"]["v"],
                final["populations"]["cell"]["integrator"]["v"],
                auxiliary["callbacks"]["trace"]["cell.v"],
            )
        )

    references = dispatch()
    gc.collect()
    assert all(reference() is None for reference in references)
    assert len(chunk._callback_kernels) == 0


@pytest.mark.parametrize("readonly", (False, True))
def test_while_accepts_reordered_state_mappings_and_noncontiguous_voltage(readonly):
    functional, tensors, prepared = _case(twins=True)
    voltage = tensors.state["populations"]["cell"]["integrator"]["v"]
    storage = voltage.new_empty(2 * voltage.numel() + 3)
    view = storage[1 : 1 + 2 * voltage.numel() : 2].view(voltage.shape)
    view.copy_(voltage)

    def reorder(value):
        if not isinstance(value, dict):
            return value
        values = {
            name: reorder(child) for name, child in reversed(tuple(value.items()))
        }
        return MappingProxyType(values) if readonly else values

    state = reorder(_state_with_voltage(tensors.state, view))
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("cell.v", "other.v", "t"))}
    )
    chunk = functional.compile_rollout_chunk(4, execution="while", backend="eager")
    expected, reference = functional.rollout(
        tensors.parameters, prepared, tensors.state, steps=4, callbacks=callbacks
    )
    actual, auxiliary = chunk(tensors.parameters, prepared, state, callbacks=callbacks)
    _close(actual, expected)
    _close(auxiliary["callbacks"]["trace"], reference["callbacks"]["trace"])
    torch.testing.assert_close(view, voltage)


@pytest.mark.parametrize("record", (False, True))
def test_while_graph_size_is_bounded_as_the_network_horizon_grows(record):
    functional, tensors, prepared = _case()
    callbacks = (
        functional.make_callbacks({"trace": dn.func.Recorder("cell.v")})
        if record
        else None
    )
    sizes = []
    for steps in (8, 32):
        graphs = []
        chunk = functional.compile_rollout_chunk(
            steps, execution="while", backend=_backend(graphs)
        )
        actual, auxiliary = chunk(
            tensors.parameters, prepared, tensors.state, callbacks=callbacks
        )
        expected, reference = functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            steps=steps,
            callbacks=callbacks,
        )
        _close(actual, expected)
        if record:
            _close(auxiliary["callbacks"]["trace"], reference["callbacks"]["trace"])
        assert len(graphs) == 1 and _has_while(graphs[0])
        sizes.append(
            sum(
                len(list(module.graph.nodes))
                for module in graphs[0].modules()
                if isinstance(module, torch.fx.GraphModule)
            )
        )
        assert chunk.execution_report().strategy == "compiled_while_loop"
    assert abs(sizes[1] - sizes[0]) <= 16, sizes
