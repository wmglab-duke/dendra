"""Public parameter selectors expand predictably and expose actual trainability."""

from __future__ import annotations

import re

import pytest
import torch

import dendra as dn

pytestmark = pytest.mark.cpu
DTYPE = torch.float64


def _population(parameters):
    return dn.func.PopulationTensors(
        parameters,
        {},
        {},
        dn.func.InitializationInput(torch.tensor([[-65.0]], dtype=DTYPE)),
    )


@pytest.mark.parametrize("scope", ("all", "model", "intra", "extra"))
@pytest.mark.parametrize("inference", (False, True))
def test_population_independent_parameters_preserve_cross_namespace_aliases(
    scope, inference
):
    source = torch.nn.Parameter(torch.tensor(0.7, dtype=DTYPE))
    names = (
        "integrator.mech.mechanisms.owner.injected_waveforms.0.amp",
        "stimulation.intra.waveforms.0.amp",
        "stimulation.extra.waveforms.0.amp",
    )
    tensors = _population({**dict.fromkeys(names, source), "separate": source.detach()})
    with torch.inference_mode(inference):
        parameters = tensors.independent_parameters("*amp", within=scope)
    shared = parameters[names[0]]
    assert isinstance(shared, torch.nn.Parameter)
    assert all(parameters[name] is shared for name in names)
    assert (
        parameters["separate"] is not shared
        and not parameters["separate"].requires_grad
    )
    assert shared.data_ptr() != source.data_ptr()
    assert shared._version == 0
    gradient = torch.autograd.grad(
        sum(parameters[name].square() for name in names), shared
    )[0]
    torch.testing.assert_close(gradient, 6 * source.detach())
    assert source.grad is None


def _flatten_network(parameters, prefix=()):
    result = {}
    for name, value in parameters.items():
        path = (*prefix, name)
        if isinstance(value, dict):
            result.update(_flatten_network(value, path))
        elif torch.is_tensor(value):
            result[".".join(path)] = value
    return result


@pytest.fixture(params=("population", "network"))
def tensors(request):
    def value(number):
        return torch.tensor([number], dtype=DTYPE)

    parameters = {
        "weights": value(0.1),
        "nested": {"weights": value(0.2)},
        "model": {
            "cell0": {"gbar": value(0.3), "rest": value(-65.0)},
            "cell1": {"gbar": value(0.4)},
            "cell10": {"gbar": value(0.5)},
        },
        "Model": {"cell2": {"gbar": value(0.6)}},
        "stimulation": {
            "intra": {"waveforms": {"0": {"amp": value(0.7)}}},
            "extra": {"waveforms": {"0": {"amp": value(0.8)}}},
        },
        "literal[0]": {"gain": value(0.9)},
        "literal0": {"gain": value(1.0)},
    }
    if request.param == "network":
        return dn.func.NetworkTensors(parameters, {}, {})
    return _population(_flatten_network(parameters))


def _named(tensors):
    if isinstance(tensors, dn.func.NetworkTensors):
        return tensors.named_parameters
    return tensors.parameters


def _selected_packet(tensors, selectors, **options):
    return tensors._replace(
        parameters=tensors.independent_parameters(selectors, **options)
    )


def test_plain_queries_keep_exact_priority_and_substring_search(tensors):
    named = _named(tensors)
    assert tensors.parameter_name("weights") == "weights"
    assert tensors.get_parameter("weights") is named["weights"]
    assert tuple(tensors.find_parameters("weights")) == ("weights", "nested.weights")
    selected = _selected_packet(tensors, "weights")
    assert tuple(selected.trainable_parameters) == ("weights",)
    assert tensors.parameter_name("cell0.rest") == "model.cell0.rest"
    with pytest.raises(KeyError, match="ambiguous"):
        tensors.parameter_name("gbar")
    with pytest.raises(KeyError, match="ambiguous"):
        tensors.independent_parameters("gbar")


@pytest.mark.parametrize(
    "selector,expected",
    (
        (
            "model.*.gbar",
            ("model.cell0.gbar", "model.cell1.gbar", "model.cell10.gbar"),
        ),
        ("model.cell?.gbar", ("model.cell0.gbar", "model.cell1.gbar")),
        ("model.cell[01].gbar", ("model.cell0.gbar", "model.cell1.gbar")),
        (
            "model*gbar",
            ("model.cell0.gbar", "model.cell1.gbar", "model.cell10.gbar"),
        ),
        ("Model.*", ("Model.cell2.gbar",)),
        ("glob:*weights", ("weights", "nested.weights")),
        ("glob:weights", ("weights",)),
        (
            r"re:cell[01]\.gbar$",
            ("model.cell0.gbar", "model.cell1.gbar"),
        ),
        (re.compile(r"CELL1\.GBAR$", re.IGNORECASE), ("model.cell1.gbar",)),
    ),
)
def test_glob_and_regex_select_multiple_canonical_names(tensors, selector, expected):
    matches = tensors.find_parameters(selector)
    assert tuple(matches) == expected
    named = _named(tensors)
    assert all(matches[name] is named[name] for name in expected)
    selected = _selected_packet(tensors, selector)
    assert tuple(selected.trainable_parameters) == expected
    for name, value in _named(selected).items():
        assert value.requires_grad is (name in expected)
        assert isinstance(value, torch.nn.Parameter) is (name in expected)
        assert value.is_leaf and value.grad_fn is None
        assert value.data_ptr() != named[name].data_ptr()


@pytest.mark.parametrize(
    "selector", ("*cell0.rest", r"re:cell0\.rest$", re.compile("rest$"))
)
def test_pattern_singular_lookup_resolves_one_name(tensors, selector):
    assert tensors.parameter_name(selector) == "model.cell0.rest"
    assert tensors.get_parameter(selector) is _named(tensors)["model.cell0.rest"]


@pytest.mark.parametrize(
    "selector", ("model.*.gbar", r"re:gbar$", re.compile("weights$"))
)
def test_pattern_singular_lookup_rejects_multiple_names(tensors, selector):
    with pytest.raises(KeyError, match="ambiguous"):
        tensors.parameter_name(selector)
    with pytest.raises(KeyError, match="ambiguous"):
        tensors.get_parameter(selector)


def test_pattern_union_preserves_canonical_order_without_duplicate_leaves(tensors):
    selected = _selected_packet(
        tensors,
        (re.compile(r"rest$"), "model.cell1.*", "model.cell[01].gbar", "model.cell0.*"),
    )
    assert tuple(selected.trainable_parameters) == (
        "model.cell0.gbar",
        "model.cell0.rest",
        "model.cell1.gbar",
    )
    assert len({id(value) for value in selected.trainable_parameters.values()}) == 3
    values = tuple(selected.trainable_parameters.values())
    gradients = torch.autograd.grad(
        sum(value.square().sum() for value in values), values
    )
    for value, gradient in zip(values, gradients, strict=True):
        torch.testing.assert_close(gradient, 2 * value)


@pytest.mark.parametrize(
    "selector", ("absent*", "glob:absent", "re:absent$", re.compile("absent$"))
)
def test_unmatched_selectors_fail_training_even_when_another_selector_matches(
    tensors, selector
):
    assert tensors.find_parameters(selector) == {}
    with pytest.raises(KeyError, match="no .*parameter"):
        tensors.parameter_name(selector)
    with pytest.raises(KeyError, match="no .*parameter"):
        tensors.independent_parameters(("weights", selector))


@pytest.mark.parametrize(
    "selector,error",
    (
        ("", ValueError),
        ("re:", ValueError),
        ("glob:", ValueError),
        ("re:[", ValueError),
        (re.compile(""), ValueError),
        (re.compile(b"gbar"), TypeError),
        (b"weights", TypeError),
        (None, TypeError),
    ),
)
def test_invalid_selectors_raise_clear_errors(tensors, selector, error):
    with pytest.raises(error, match="query|selector|regex|glob|pattern|string"):
        tensors.find_parameters(selector)
    with pytest.raises(error, match="query|selector|regex|glob|pattern|string"):
        tensors.independent_parameters((selector,))


def test_literal_pattern_characters_remain_addressable_by_exact_name(tensors):
    assert tensors.parameter_name("literal[0].gain") == "literal[0].gain"
    selected = _selected_packet(tensors, "literal[0].gain")
    assert tuple(selected.trainable_parameters) == ("literal[0].gain",)
    assert tuple(tensors.find_parameters("glob:literal[0].gain")) == ("literal0.gain",)
    assert tuple(tensors.find_parameters(r"re:^literal\[0\]\.gain$")) == (
        "literal[0].gain",
    )


def test_trainable_inspection_tracks_live_flags_without_copying_leaves(tensors):
    assert tensors.trainable_parameters == {}
    selected = _selected_packet(tensors, ("weights", "*rest"))
    named = _named(selected)
    assert tuple(selected.trainable_parameters) == ("weights", "model.cell0.rest")
    assert selected.trainable_parameters["weights"] is named["weights"]
    named["weights"].requires_grad_(False)
    named["model.cell1.gbar"].requires_grad_(True)
    assert tuple(selected.trainable_parameters) == (
        "model.cell0.rest",
        "model.cell1.gbar",
    )
    assert tensors.trainable_parameters == {}


def test_regex_selection_constructed_under_inference_mode_is_versioned(tensors):
    with torch.inference_mode():
        selected = _selected_packet(tensors, re.compile(r"^model\.cell[01]\.gbar$"))
    assert len(selected.trainable_parameters) == 2
    for name, value in _named(selected).items():
        assert not value.is_inference()
        assert value._version >= 0
        assert value.data_ptr() != _named(tensors)[name].data_ptr()
    values = tuple(selected.trainable_parameters.values())
    gradients = torch.autograd.grad(sum(value.sum() for value in values), values)
    assert all(
        torch.equal(gradient, torch.ones_like(gradient)) for gradient in gradients
    )


@pytest.mark.parametrize("selector", ("*amp", r"re:amp$", re.compile(r"\.amp$")))
def test_population_scope_limits_patterns_but_returns_the_complete_packet(selector):
    population = _population(
        {
            "integrator.amp": torch.tensor(1.0),
            "stimulation.intra.waveforms.0.amp": torch.tensor(2.0),
            "stimulation.extra.waveforms.0.amp": torch.tensor(3.0),
        }
    )
    expected = "stimulation.intra.waveforms.0.amp"
    assert tuple(population.find_parameters(selector, within="intra")) == (expected,)
    assert population.parameter_name(selector, within="intra") == expected
    selected = _selected_packet(population, selector, within="intra")
    assert tuple(selected.parameters) == tuple(population.parameters)
    assert tuple(selected.trainable_parameters) == (expected,)
    with pytest.raises(KeyError, match="no .*parameter"):
        population.independent_parameters("stimulation.*", within="model")


@pytest.mark.parametrize("dtype", (torch.int64, torch.bool))
@pytest.mark.parametrize("selector", ("*", "re:.*", re.compile(r"flags$")))
def test_patterns_do_not_silently_skip_nondifferentiable_leaves(dtype, selector):
    for packet in (
        _population({"flags": torch.ones(2, dtype=dtype)}),
        dn.func.NetworkTensors(
            {"nested": {"flags": torch.ones(2, dtype=dtype)}}, {}, {}
        ),
    ):
        with pytest.raises(ValueError, match="floating-point or complex"):
            packet.independent_parameters(selector)


def test_network_pattern_alias_selection_and_inspection_keep_one_optimizer_leaf():
    base = torch.tensor([0.2, 0.4], dtype=DTYPE, requires_grad=True)
    shared = base.square()
    distinct_view = shared.view_as(shared)
    packet = dn.func.NetworkTensors(
        {
            "populations": {"cell": {"gain": shared}},
            "synapses": {
                "left": {"weights": shared},
                "right": {"weights": shared},
                "independent": {"weights": distinct_view},
            },
        },
        {},
        {},
    )
    selected = _selected_packet(packet, ("synapses.[lr]*.weights", "re:^synapses.left"))
    names = (
        "populations.cell.gain",
        "synapses.left.weights",
        "synapses.right.weights",
    )
    assert tuple(selected.trainable_parameters) == names
    parameter = selected.get_parameter(names[0])
    assert all(value is parameter for value in selected.trainable_parameters.values())
    assert not selected.get_parameter("independent.weights").requires_grad
    assert not parameter.is_inference() and parameter.is_leaf
    assert parameter.data_ptr() != shared.data_ptr()
    gradient = torch.autograd.grad(
        sum(value.square().sum() for value in selected.trainable_parameters.values()),
        parameter,
    )[0]
    torch.testing.assert_close(gradient, 6 * parameter)
    with pytest.raises(KeyError, match="ambiguous"):
        selected.parameter_name("synapses.[lr]*.weights")
    parameter.requires_grad_(False)
    assert selected.trainable_parameters == {}
    assert packet.trainable_parameters.keys() == packet.named_parameters.keys()
