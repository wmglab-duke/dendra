"""Inference owns carry at actual while-block and public-tail boundaries."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils import _pytree as pytree

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import StepInput, _population_while
from dendra.func._callbacks import (
    AnomalyDetector,
    Recorder,
    _prepare_callback,
    _update_callback_values,
)

pytestmark = pytest.mark.cpu


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _tensor_leaves(tree):
    return [value for value in pytree.tree_leaves(tree) if torch.is_tensor(value)]


def _offset_view(value):
    """Keep logical values in a non-dense view with a nonzero storage offset."""
    storage = torch.empty(2 * value.numel() + 5, dtype=value.dtype, device=value.device)
    view = storage[3 : 3 + 2 * value.numel() : 2].view(value.shape)
    view.copy_(value)
    return view


def _close(actual, expected, *, exact=False):
    values, spec = pytree.tree_flatten(actual)
    references, reference_spec = pytree.tree_flatten(expected)
    assert spec == reference_spec
    for value, reference in zip(values, references, strict=True):
        if torch.is_tensor(value):
            options = (
                {"rtol": 0, "atol": 0} if exact else {"rtol": 1e-12, "atol": 1e-12}
            )
            torch.testing.assert_close(value, reference, **options)
        else:
            assert value == reference


class _ViewFunctional:
    """Pure transitions may return unchanged views; emissions may alias state."""

    device = torch.device("cpu")

    @staticmethod
    def _step_values(parameters, prepared, state, inputs):
        voltage = (
            state["integrator"]["v"]
            + parameters["gain"] * inputs.intra
            + prepared["field_gain"] * inputs.ve
        ).tanh()
        result = {
            "integrator": {"v": voltage},
            "fixed": state["fixed"].view_as(state["fixed"]),
            "clock": {
                "t": state["clock"]["t"] + prepared["dt"],
                "duration_remainder": state["clock"]["duration_remainder"],
            },
        }
        return result, {"v": voltage}

    @staticmethod
    def _clone_state_tree(state):
        # Match the real functional population's canonical layout and strides.
        def owned(value):
            return value.clone(memory_format=torch.contiguous_format)

        return {
            "integrator": {
                "v": owned(state["integrator"]["v"]),
            },
            "fixed": owned(state["fixed"]),
            "clock": {
                "t": owned(state["clock"]["t"]),
                "duration_remainder": owned(state["clock"]["duration_remainder"]),
            },
        }


def _unchanged_carry_views(carry, state, auxiliary):
    del state, auxiliary
    return tuple(value.view_as(value) for value in carry), None


def _case(steps=9):
    batch, nodes = 2, 6
    voltage = torch.linspace(-0.4, 0.3, batch * nodes, dtype=torch.float64)
    voltage = voltage.reshape(nodes, batch).transpose(0, 1)
    fixed = torch.arange(batch * nodes, dtype=torch.float64).reshape(nodes, batch).T
    state = {
        "integrator": {"v": voltage},
        "fixed": fixed,
        "clock": {
            "t": torch.tensor(0.0, dtype=torch.float64),
            "duration_remainder": torch.tensor(0.001, dtype=torch.float64),
        },
    }
    intra = torch.linspace(-0.03, 0.05, steps * batch * nodes, dtype=torch.float64)
    intra = intra.reshape(steps, nodes, batch).transpose(1, 2)
    ve = torch.linspace(0.02, -0.01, steps * batch * nodes, dtype=torch.float64)
    ve = ve.reshape(steps, nodes, batch).transpose(1, 2)
    parameters = {"gain": torch.tensor(0.7, dtype=torch.float64)}
    prepared = {
        "field_gain": torch.tensor(0.3, dtype=torch.float64),
        "dt": torch.tensor(0.025, dtype=torch.float64),
    }

    def recorder(*, selected=False, maximum=False, clock=False):
        return _prepare_callback(
            Recorder,
            {
                "states": ("t",) if clock else ("v", "view", "fixed"),
                "paths": (("clock", "t"),)
                if clock
                # The duplicate Recorder path deliberately emits the same state
                # storage twice. Invocation-local history buffers must own both.
                else (("integrator", "v"), ("integrator", "v"), ("fixed",)),
                "max_only": maximum,
                "node_indices": torch.tensor([0, nodes - 1, 0]) if selected else None,
                "partition": None,
                "region": None,
            },
        )

    # Exercise the private executor's carry-ownership contract with an admitted
    # callback type. This does not expand public custom-callback support: the
    # separate real AnomalyDetector below retains its normal behavior.
    unchanged = AnomalyDetector()
    unchanged.update = _unchanged_carry_views
    callbacks = {
        "full": recorder(),
        "selected": recorder(selected=True),
        "maximum": recorder(maximum=True),
        "clock": recorder(clock=True),
        "anomaly": AnomalyDetector(),
        "unchanged": unchanged,
    }
    plans = tuple(
        SimpleNamespace(
            name=name,
            callback=callback,
            emission_schema=None,
            validate_updates=False,
        )
        for name, callback in callbacks.items()
    )
    floating = torch.arange(6, dtype=torch.float64).reshape(3, 2).T
    carries = {
        name: callback.initialize(state, None)[0]
        for name, callback in callbacks.items()
        if name != "unchanged"
    }
    carries["unchanged"] = (
        floating,
        floating + 0.125,
        torch.arange(6, dtype=torch.int64).reshape(3, 2).T,
        (torch.arange(6).reshape(3, 2).T % 2 == 0),
    )
    assert not voltage.is_contiguous() and not fixed.is_contiguous()
    assert not intra.is_contiguous() and not ve.is_contiguous()
    return SimpleNamespace(
        functional=_ViewFunctional(),
        plans=plans,
        parameters=parameters,
        prepared=prepared,
        state=state,
        ve=ve,
        intra=intra,
        carries=carries,
    )


def _capture(case, steps, options):
    executor = _population_while._PopulationWhileExecutor(
        case.functional, steps=steps, plans=case.plans, compile_options=options
    )
    kernel, reason = executor.kernel_for(
        case.parameters,
        case.prepared,
        case.state,
        case.ve[:steps],
        case.intra[:steps],
        case.carries,
    )
    assert reason is None, reason
    assert kernel is not None and executor._capture_errors == {}
    return kernel


def _eager(case, steps, *, state=None, carries=None, offset=0):
    state = case.state if state is None else state
    carries = case.carries if carries is None else carries
    emissions = []
    for index in range(offset, offset + steps):
        state, auxiliary = case.functional._step_values(
            case.parameters,
            case.prepared,
            state,
            StepInput(case.ve[index], case.intra[index]),
        )
        carries, emitted = _update_callback_values(
            case.plans, carries, state, auxiliary
        )
        emissions.append(emitted)
    leaves, spec = pytree.tree_flatten(emissions[0])
    flattened = [pytree.tree_leaves(emission) for emission in emissions]
    observed = pytree.tree_unflatten(
        [
            torch.stack([frame[index] for frame in flattened])
            for index in range(len(leaves))
        ],
        spec,
    )
    observed = {plan.name: observed.get(plan.name) for plan in case.plans}
    return state, {"v": state["integrator"]["v"]}, carries, observed


def _assert_owned(result, inputs):
    final, _auxiliary, carries, emitted = result
    # auxiliary["v"] intentionally aliases the final voltage API result.
    outputs = _tensor_leaves((final, carries, emitted))
    for index, value in enumerate(outputs):
        for other in outputs[index + 1 :]:
            assert not torch._C._is_alias_of(value, other)
        for original in _tensor_leaves(inputs):
            assert not torch._C._is_alias_of(value, original)


def test_authored_transition_does_not_clone_output_state_or_callback_carry(monkeypatch):
    case = _case(4)
    graphs = []
    original_make_fx = _population_while.make_fx

    def capture_graph(function):
        def trace(*arguments):
            graph = original_make_fx(function)(*arguments)
            graphs.append(graph)
            return graph

        return trace

    monkeypatch.setattr(_population_while, "make_fx", capture_graph)
    monkeypatch.setattr(
        _population_while, "_compile_while", lambda function, _options: function
    )
    _capture(case, 4, {})
    assert len(graphs) == 1
    assert not any(
        node.op == "call_function" and node.target is torch.ops.aten.clone.default
        for node in graphs[0].graph.nodes
    )


@pytest.mark.slow
@pytest.mark.parametrize("threads", [1, 4], ids=["t1", "t4"])
def test_cpp_while_block_and_tail_own_views_mixed_carries_and_full_recording(threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        _check_cpp_boundaries()
    finally:
        torch.set_num_threads(previous)


def _check_cpp_boundaries():
    case = _case()
    inputs = (
        case.parameters,
        case.prepared,
        case.state,
        case.ve,
        case.intra,
        case.carries,
    )
    originals = _clone(inputs)
    versions = tuple(value._version for value in _tensor_leaves(inputs))
    options = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "options": {"cpp_wrapper": True, "compile_threads": 1, "fx_graph_cache": False},
    }
    kernels = {steps: _capture(case, steps, options) for steps in (4, 5, 8, 9)}
    results = {}
    with torch.inference_mode(), torch_compiler_warning_context():
        for steps, kernel in kernels.items():
            actual = kernel(
                case.parameters,
                case.prepared,
                case.state,
                case.ve[:steps],
                case.intra[:steps],
                case.carries,
            )
            _close(actual, _eager(case, steps))
            _assert_owned(actual, inputs)
            assert actual[3]["full"]["v"].shape == (steps, 2, 6)
            torch.testing.assert_close(
                actual[3]["selected"]["v"],
                actual[3]["full"]["v"].index_select(-1, torch.tensor([0, 5, 0])),
            )
            results[steps] = actual

        first = results[4]
        first_saved = _clone(first)
        continued = kernels[5](
            case.parameters,
            case.prepared,
            first[0],
            case.ve[4:],
            case.intra[4:],
            first[2],
        )
        _close(continued, _eager(case, 5, state=first[0], carries=first[2], offset=4))
        _close((continued[0], continued[2]), (results[9][0], results[9][2]))
        for name in ("full", "selected", "maximum", "clock"):
            joined = {
                key: torch.cat((first[3][name][key], continued[3][name][key]))
                for key in first[3][name]
            }
            _close(joined, results[9][3][name])
        _assert_owned(continued, (inputs, first))
        _close(first, first_saved, exact=True)

        # A distinct invocation must not overwrite earlier output buffers.
        saved = _clone(results[9])
        changed_parameters = {"gain": case.parameters["gain"] + 0.2}
        repeated = kernels[9](
            changed_parameters,
            case.prepared,
            case.state,
            case.ve,
            -case.intra,
            case.carries,
        )
        _assert_owned(repeated, (inputs, results))
        assert not torch.equal(repeated[3]["full"]["v"], results[9][3]["full"]["v"])
        _close(results[9], saved, exact=True)
    _close(inputs, originals, exact=True)
    assert versions == tuple(value._version for value in _tensor_leaves(inputs))
