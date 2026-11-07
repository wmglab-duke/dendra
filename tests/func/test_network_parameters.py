"""Nested Network optimizer packets preserve identity ties and readable names."""

from types import MappingProxyType

import pytest
import torch

import dendra as dn
from tests.func.test_network import DT, DTYPE, _close, _loss, _mixed_network

pytestmark = pytest.mark.cpu


def _packet():
    base = torch.tensor([0.3, 0.7], dtype=DTYPE, requires_grad=True)
    shared = base.square()
    view = shared.view_as(shared)
    parameters = {
        "populations": {"cell": {"integrator.g_param": shared}},
        "synapses": {
            "event": {
                "weights": shared,
                "delays": None,
                "schedule": dn.func.TensorScheduleTensors(shared, view, shared, None),
            }
        },
        "continuous_synapses": {},
        "netstim": None,
    }
    return dn.func.NetworkTensors(parameters, {}, {}), base, shared, view


def test_parameter_names_keep_population_keys_and_schedule_field_names():
    tensors, _, shared, view = _packet()
    assert tuple(tensors.named_parameters) == (
        "populations.cell.integrator.g_param",
        "synapses.event.weights",
        "synapses.event.schedule.times",
        "synapses.event.schedule.amplitudes",
        "synapses.event.schedule.time_source",
    )
    assert tensors.get_parameter("g_param") is shared
    assert tensors.parameter_name("amplitudes") == "synapses.event.schedule.amplitudes"
    assert tensors.find_parameters("schedule") == {
        "synapses.event.schedule.times": shared,
        "synapses.event.schedule.amplitudes": view,
        "synapses.event.schedule.time_source": shared,
    }
    names = tensors.named_parameters
    names.clear()
    assert len(tensors.named_parameters) == 5


@pytest.mark.parametrize("selection", ((), ("schedule.times",), ("weights", "g_param")))
def test_independent_parameters_preserve_genuine_ties_but_split_distinct_views(
    selection,
):
    tensors, base, shared, view = _packet()
    independent = tensors.independent_parameters(selection)
    schedule = independent["synapses"]["event"]["schedule"]
    target = independent["populations"]["cell"]["integrator.g_param"]
    assert type(schedule) is dn.func.TensorScheduleTensors
    assert independent["synapses"]["event"]["weights"] is target
    assert schedule.times is target and schedule.time_source is target
    assert schedule.amplitude_source is None
    assert independent["synapses"]["event"]["delays"] is None
    assert independent["netstim"] is None
    assert schedule.amplitudes is not target
    assert target.data_ptr() != shared.data_ptr()
    assert schedule.amplitudes.data_ptr() != view.data_ptr()
    assert schedule.amplitudes.data_ptr() != target.data_ptr()
    torch.testing.assert_close(target, shared)
    torch.testing.assert_close(schedule.amplitudes, view)
    assert target.requires_grad == bool(selection)
    assert isinstance(target, torch.nn.Parameter) == bool(selection)
    assert not schedule.amplitudes.requires_grad
    assert target.is_leaf and target.grad_fn is None
    assert base.grad is None
    if selection:
        loss = target.square().sum() + schedule.times.sum() + schedule.time_source.sum()
        gradient = torch.autograd.grad(loss, target)[0]
        torch.testing.assert_close(gradient, 2 * target + 2)
        with torch.no_grad():
            target.add_(0.4)
        torch.testing.assert_close(shared, base.square())


def test_selection_accepts_one_string_and_exact_name_wins_over_partial_matches():
    values = {
        "weights": torch.tensor(0.1, dtype=DTYPE),
        "nested": {"weights": torch.tensor(0.2, dtype=DTYPE)},
    }
    tensors = dn.func.NetworkTensors(values, {}, {})
    assert tensors.parameter_name("weights") == "weights"
    assert tensors.get_parameter("weights") is values["weights"]
    selected = tensors.independent_parameters("weights")
    assert selected["weights"].requires_grad
    assert not selected["nested"]["weights"].requires_grad


@pytest.mark.parametrize(
    "query, error, message",
    (
        (None, TypeError, "must be a string"),
        ("", ValueError, "non-empty"),
        ("absent", KeyError, "no Network parameter"),
        ("schedule.", KeyError, "ambiguous"),
    ),
)
def test_parameter_query_errors_are_clear(query, error, message):
    tensors, *_ = _packet()
    with pytest.raises(error, match=message):
        tensors.parameter_name(query)
    if error is not KeyError:
        with pytest.raises(error, match=message):
            tensors.find_parameters(query)


@pytest.mark.parametrize("dtype", (torch.long, torch.bool))
def test_integer_and_boolean_leaves_cannot_be_selected_for_gradients(dtype):
    parameters = {"flags": torch.ones(2, dtype=dtype)}
    tensors = dn.func.NetworkTensors(parameters, {}, {})
    with pytest.raises(ValueError, match="floating-point or complex"):
        tensors.independent_parameters("flags")
    independent = tensors.independent_parameters()
    torch.testing.assert_close(independent["flags"], parameters["flags"])
    assert not independent["flags"].requires_grad


def test_complex_leaves_can_be_selected_for_gradients():
    value = torch.tensor([1 + 2j], dtype=torch.complex128)
    tensors = dn.func.NetworkTensors({"value": value}, {}, {})
    independent = tensors.independent_parameters("value")
    gradient = torch.autograd.grad(
        independent["value"].abs().square().sum(), independent["value"]
    )[0]
    torch.testing.assert_close(gradient, 2 * value)


@pytest.mark.parametrize("shared", (False, True))
def test_dotted_path_collisions_are_rejected_even_when_values_are_tied(shared):
    first = torch.tensor(1.0)
    second = first if shared else torch.tensor(2.0)
    tensors = dn.func.NetworkTensors(
        {"cell.weight": first, "cell": {"weight": second}}, {}, {}
    )
    with pytest.raises(ValueError, match="collide.*cell.weight"):
        _ = tensors.named_parameters
    with pytest.raises(ValueError, match="collide.*cell.weight"):
        tensors.independent_parameters()


def test_inference_construction_returns_versioned_independent_trainable_aliases():
    with torch.inference_mode():
        source = torch.tensor([0.4, 0.8], dtype=DTYPE)
        parameters = MappingProxyType(
            {"first": source, "nested": MappingProxyType({"second": source})}
        )
        tensors = dn.func.NetworkTensors(parameters, {}, {})
        independent = tensors.independent_parameters("nested.second")
    assert independent["first"] is independent["nested"]["second"]
    assert not independent["first"].is_inference()
    assert independent["first"]._version >= 0
    assert independent["first"].requires_grad
    assert independent["first"].data_ptr() != source.data_ptr()
    gradient = torch.autograd.grad(
        independent["first"].square().sum(), independent["first"]
    )[0]
    torch.testing.assert_close(gradient, 2 * source)


def test_native_shared_weights_keep_one_optimizer_leaf_and_summed_rollout_gradient():
    model = _mixed_network()
    connections = tuple(model.synapses.values())
    connections[1].weight.weights[0].param = connections[0].weight.weights[0].param
    model.initialize(DT)
    functional, tensors = dn.func.make_functional_network(model)
    event_names = [
        name
        for name in tensors.named_parameters
        if name.startswith("synapses.") and name.endswith(".weight.weights.0.param.rho")
    ]
    assert len(event_names) == 2
    independent = tensors.independent_parameters(event_names[1])
    packet = dn.func.NetworkTensors(independent, tensors.constants, tensors.state)
    assert packet.get_parameter(event_names[0]) is packet.get_parameter(event_names[1])
    functional._validate_parameters(independent)
    expected, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=7
    )
    actual, _ = functional.prepare_and_rollout(
        independent, tensors.constants, tensors.state, steps=7
    )
    _close(actual, expected)
    observed = torch.autograd.grad(_loss(actual), packet.get_parameter(event_names[0]))[
        0
    ]
    oracle = torch.autograd.grad(
        _loss(expected), tensors.get_parameter(event_names[0])
    )[0]
    torch.testing.assert_close(observed, oracle)
    assert torch.isfinite(observed).all() and observed.abs().sum() > 0


def test_optional_empty_tree_and_plain_tuple_structure_are_preserved():
    value = torch.tensor([0.5])
    tensors = dn.func.NetworkTensors(
        {"tuple": (value, None), "none": None, "empty": {}}, {}, {}
    )
    assert tensors.parameter_name("tuple.0") == "tuple.0"
    independent = tensors.independent_parameters()
    assert type(independent["tuple"]) is tuple and independent["tuple"][1] is None
    assert independent["none"] is None and independent["empty"] == {}
