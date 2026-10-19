"""Training packs own the HOP carry without exposing aliased public leaves."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils import _pytree as pytree

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._callbacks import _update_callback_values
from dendra.func._carry_packing import _PackedCarryLayout
from dendra.func._scan import _CompiledScanPopulationChunk
from dendra.func._types import StepInput


def _leaves(tree):
    return [value for value in pytree.tree_leaves(tree) if torch.is_tensor(value)]


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected, *, exact=False):
    values, spec = pytree.tree_flatten(actual)
    references, reference_spec = pytree.tree_flatten(expected)
    assert spec == reference_spec
    for value, reference in zip(values, references, strict=True):
        kwargs = {"rtol": 0, "atol": 0} if exact else {"rtol": 2e-10, "atol": 2e-11}
        torch.testing.assert_close(value, reference, **kwargs)


class _PackedFunctional:
    device = torch.device("cpu")
    intra = SimpleNamespace(enabled=True)
    extra = SimpleNamespace(enabled=True)

    @staticmethod
    def _clone_state_tree(state):
        return pytree.tree_map_only(
            torch.Tensor,
            lambda value: value.clone(memory_format=torch.contiguous_format),
            state,
        )

    @staticmethod
    def _step_values(parameters, prepared, state, inputs):
        voltage = (
            state["integrator"]["v"]
            + parameters["gain"] * inputs.intra
            + prepared["field_gain"] * inputs.ve
        ).tanh()
        result = {
            "integrator": {"v": voltage, "alias": voltage.view_as(voltage)},
            "fixed": state["fixed"].view_as(state["fixed"]),
            "clock": {"t": state["clock"]["t"] + prepared["dt"]},
            "control": {"duration_remainder": state["control"]["duration_remainder"]},
        }
        return result, {"v": voltage}


class _MixedCallback:
    @staticmethod
    def update(carry, state, auxiliary):
        del auxiliary
        voltage = state["integrator"]["v"]
        updated = {
            "value": carry["value"].view_as(carry["value"]),
            "alias": carry["value"].view_as(carry["value"]),
            "single": carry["single"].view_as(carry["single"]),
            "integer": carry["integer"] + 1,
            "boolean": torch.logical_not(carry["boolean"]),
            "loss": carry["loss"] + voltage.square().mean(),
        }
        emitted = {
            "v": voltage,
            "duplicate": voltage.view_as(voltage),
            "t": state["clock"]["t"],
        }
        return updated, emitted


def _case(*, callbacks=True, steps=4):
    voltage = torch.linspace(-0.4, 0.3, 12, dtype=torch.float64).reshape(6, 2).T
    voltage = voltage.detach().requires_grad_()
    fixed = torch.arange(12, dtype=torch.float32).reshape(6, 2).T
    state = {
        "integrator": {"v": voltage, "alias": voltage.view_as(voltage)},
        "fixed": fixed,
        "clock": {"t": torch.tensor(0.0, dtype=torch.float64)},
        "control": {"duration_remainder": torch.tensor(0.001, dtype=torch.float64)},
    }
    intra = torch.linspace(-0.03, 0.05, steps * 12, dtype=torch.float64)
    intra = intra.reshape(steps, 6, 2).transpose(1, 2).detach().requires_grad_()
    ve = torch.linspace(0.02, -0.01, steps * 12, dtype=torch.float64)
    ve = ve.reshape(steps, 6, 2).transpose(1, 2).detach().requires_grad_()
    parameters = {"gain": torch.tensor(0.7, dtype=torch.float64, requires_grad=True)}
    prepared = {
        "field_gain": torch.tensor(0.3, dtype=torch.float64),
        "dt": torch.tensor(0.025, dtype=torch.float64),
    }
    carry_value = torch.arange(6, dtype=torch.float64).reshape(3, 2).T
    carries = (
        {
            "mixed": {
                "value": carry_value,
                "alias": carry_value.view_as(carry_value),
                "single": torch.arange(6, dtype=torch.float32).reshape(3, 2).T,
                "integer": torch.arange(6, dtype=torch.int64).reshape(3, 2).T,
                "boolean": torch.tensor([[True, False, True], [False, True, False]]).T,
                "loss": torch.tensor(0.0, dtype=torch.float64),
            }
        }
        if callbacks
        else {}
    )
    plans = (
        (
            SimpleNamespace(
                name="mixed",
                callback=_MixedCallback(),
                emission_schema=None,
                validate_updates=False,
            ),
        )
        if callbacks
        else None
    )
    return SimpleNamespace(
        functional=_PackedFunctional(),
        parameters=parameters,
        prepared=prepared,
        state=state,
        intra=intra,
        ve=ve,
        carries=carries,
        plans=plans,
        steps=steps,
    )


def _capture(case, options):
    chunk = _CompiledScanPopulationChunk(
        case.functional,
        steps=case.steps,
        compile_options=options,
        stepwise_stimulation=True,
        plans=case.plans,
    )
    with torch.no_grad():
        return chunk._capture(
            case.parameters,
            case.prepared,
            case.state,
            case.ve,
            case.intra,
            case.carries,
            block=1,
            grad_enabled=True,
        )


def _eager(case):
    state, carry, frames = case.state, case.carries, []
    for index in range(case.steps):
        state, auxiliary = case.functional._step_values(
            case.parameters,
            case.prepared,
            state,
            StepInput(case.ve[index], case.intra[index]),
        )
        if case.plans is not None:
            carry, emission = _update_callback_values(
                case.plans, carry, state, auxiliary
            )
            frames.append(emission)
    if case.plans is None:
        return state, auxiliary
    emitted = pytree.tree_map(lambda *values: torch.stack(values), *frames)
    return state, auxiliary, carry, emitted


def _loss(result):
    loss = result[0]["integrator"]["v"].square().mean()
    if len(result) == 4:
        loss = loss + result[2]["mixed"]["loss"]
        loss = loss + result[3]["mixed"]["duplicate"].square().mean()
    return loss


def _gradients(result, case):
    return torch.autograd.grad(
        _loss(result),
        (case.state["integrator"]["v"], case.parameters["gain"], case.ve, case.intra),
    )


def _assert_owned(result, inputs):
    # auxiliary['v'] intentionally refers to the final voltage API result.
    tree = (result[0], *result[2:])
    outputs = _leaves(tree)
    for index, value in enumerate(outputs):
        assert value.is_contiguous()
        for sibling in outputs[:index]:
            assert not torch._C._is_alias_of(value, sibling)
        for source in _leaves(inputs):
            assert not torch._C._is_alias_of(value, source)
    assert not result[0]["control"]["duration_remainder"].requires_grad


@pytest.mark.parametrize(
    "callbacks", [False, True], ids=["no_callbacks", "mixed_callbacks"]
)
def test_training_pack_hop_uses_dtype_groups_and_preserves_gradients(callbacks):
    case = _case(callbacks=callbacks)
    graphs = []

    def backend(graph, _examples):
        graphs.append(graph)
        return graph.forward

    kernel = _capture(case, {"backend": backend, "fullgraph": True})
    actual = kernel(
        case.parameters, case.prepared, case.state, case.ve, case.intra, case.carries
    )
    expected = _eager(case)
    _close(actual, expected)
    _close(_gradients(actual, case), _gradients(expected, case))
    hops = [
        node
        for graph in graphs
        for node in graph.graph.nodes
        if node.op == "call_function" and node.target is torch.ops.higher_order.scan
    ]
    layout = _PackedCarryLayout.from_tree((case.state, case.carries))
    assert len(hops) == 1 and len(hops[0].args[1]) == len(layout.groups)
    _assert_owned(actual, (case.state, case.carries, case.ve, case.intra))


@pytest.mark.slow
@pytest.mark.parametrize("threads", [1, 4], ids=["t1", "t4"])
@pytest.mark.parametrize(
    "callbacks", [False, True], ids=["no_callbacks", "mixed_callbacks"]
)
def test_cpp_training_pack_preserves_aliases_full_frames_and_gradients(
    threads, callbacks
):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        _check_cpp_pack(callbacks)
    finally:
        torch.set_num_threads(previous)


def _check_cpp_pack(callbacks):
    case = _case(callbacks=callbacks)
    inputs = (
        case.parameters,
        case.prepared,
        case.state,
        case.ve,
        case.intra,
        case.carries,
    )
    originals = _clone(inputs)
    versions = tuple(value._version for value in _leaves(inputs))
    options = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "options": {"cpp_wrapper": True, "compile_threads": 1, "fx_graph_cache": False},
    }
    kernel = _capture(case, options)
    assert callable(kernel.get_compiler_config)
    assert callable(kernel._dendra_scan_donation_diagnostics)
    assert callable(kernel.__wrapped__)
    with torch_compiler_warning_context():
        actual = kernel(
            case.parameters,
            case.prepared,
            case.state,
            case.ve,
            case.intra,
            case.carries,
        )
        expected = _eager(case)
        _close(actual, expected)
        _close(_gradients(actual, case), _gradients(expected, case))
        _assert_owned(actual, inputs)
        saved = _clone(actual)
        repeated = kernel(
            {"gain": case.parameters["gain"] + 0.2},
            case.prepared,
            case.state,
            -case.ve,
            case.intra,
            case.carries,
        )
        _assert_owned(repeated, (inputs, actual))
        _close(actual, saved, exact=True)
    _close(inputs, originals, exact=True)
    assert versions == tuple(value._version for value in _leaves(inputs))


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_training_pack_preserves_mixed_host_scheduling_and_gradients():
    case = _case(callbacks=True)

    def device_value(value):
        return value.detach().to("cuda").requires_grad_(value.requires_grad)

    case.functional.device = torch.device("cuda")
    case.parameters = pytree.tree_map_only(torch.Tensor, device_value, case.parameters)
    case.state["integrator"] = pytree.tree_map_only(
        torch.Tensor, device_value, case.state["integrator"]
    )
    case.state["fixed"] = device_value(case.state["fixed"])
    case.carries = pytree.tree_map_only(torch.Tensor, device_value, case.carries)
    case.prepared["field_gain"] = device_value(case.prepared["field_gain"])
    case.ve, case.intra = device_value(case.ve), device_value(case.intra)
    # Real functional models deliberately keep scheduling scalars on the host.
    assert case.state["clock"]["t"].device.type == "cpu"
    assert case.state["control"]["duration_remainder"].device.type == "cpu"
    assert case.prepared["dt"].device.type == "cpu"
    options = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "options": {"compile_threads": 1},
    }
    kernel = _capture(case, options)
    with torch_compiler_warning_context():
        actual = kernel(
            case.parameters,
            case.prepared,
            case.state,
            case.ve,
            case.intra,
            case.carries,
        )
        expected = _eager(case)
        _close(actual, expected)
        _close(_gradients(actual, case), _gradients(expected, case))
    _assert_owned(actual, (case.state, case.carries, case.ve, case.intra))
