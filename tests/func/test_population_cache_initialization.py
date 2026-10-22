"""Shared cache copies preserve native values and parameter differentiation."""

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.networks._population_cache import restore_population_cache_values
from tests.func.test_membrane_current import _model
from tests.func.test_network import _close
from tests.func.test_network_initialization_bindings import _cross_population_binding

pytestmark = pytest.mark.cpu
DT = 0.01


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _restore(parameter, cache):
    return restore_population_cache_values(
        {"value": parameter}, {"value": cache}, parameter_names=("value",)
    )["value"]


def test_cached_parameter_copy_has_exact_values_and_destination_gradient_identity():
    current = torch.tensor([1e30, float("nan"), float("inf")], requires_grad=True)
    saved = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    actual = _restore(current, saved)
    torch.testing.assert_close(actual, saved)
    assert actual.data_ptr() != saved.data_ptr()
    gradients = torch.autograd.grad(actual.sum(), (current, saved), allow_unused=True)
    torch.testing.assert_close(gradients[0], torch.ones_like(current))
    assert gradients[1] is None
    primal, tangent = torch.func.jvp(
        _restore, (current, saved), (torch.ones_like(current), torch.ones_like(saved))
    )
    torch.testing.assert_close(primal, saved)
    torch.testing.assert_close(tangent, torch.ones_like(current))


def test_cached_parameter_copy_compiles_and_maps_without_input_mutation():
    current = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    saved = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    actual = torch.compile(_restore, backend="aot_eager", fullgraph=True)(
        current, saved
    )
    torch.testing.assert_close(actual, saved)
    torch.testing.assert_close(torch.vmap(_restore)(current, saved), saved)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), current)[0], torch.ones_like(current)
    )
    torch.testing.assert_close(current, torch.ones_like(current))


@pytest.mark.parametrize(
    "kind",
    (
        "path",
        "cable",
        "tree",
        "block_path",
        "block_tree",
        "multi_point",
        "multi_scalar",
        "multi_block",
    ),
)
def test_cached_population_topologies_match_native_restored_state_and_parameters(kind):
    population = _model(kind, batch=None)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=torch.float64, DEVICE="cpu"):
        network = dn.Network({"cell": population})
        network.batch(2)
        network.train()
        network.initialize(DT)
        with torch.no_grad():
            for _ in range(3):
                network.step()
            network.cache_state()
        network.initialize(DT)
    functional, tensors = dn.func.make_functional_network(network)
    assert functional._cache_error is None
    actual = functional.initialize(*tensors, functional.initialization_inputs)
    _close(actual.parameters, tensors.parameters)
    _close(actual.state, tensors.state)
    expected_constants = {
        **tensors.constants,
        "populations": {
            "cell": {
                **tensors.constants["populations"]["cell"],
                "cache.active": torch.tensor(True),
            }
        },
    }
    _close(actual.constants, expected_constants)
    actual_state, _ = functional.prepare_and_step(*actual)
    network.step()
    _close(actual_state, functional.extract(network).state)


def test_cached_shared_parameter_and_buffer_follow_all_population_loads_in_native_order():
    network, key = _cross_population_binding()
    network.cache_state()
    functional, tensors = dn.func.make_functional_network(network)
    tensors.parameters["populations"]["left"][key].requires_grad_()
    defaults = functional.initialization_inputs
    cache = defaults.cache
    left_key = functional._population_cache_plans["left"].parameters[key]
    right_key = functional._population_cache_plans["right"].constants[
        "parametrizations.factor"
    ]
    cache.populations["left"][left_key].fill_(3)
    cache.populations["right"][right_key].fill_(4)
    network._state_cache = cache.populations
    initialized = functional.initialize(*tensors, defaults)
    value = initialized.parameters["populations"]["left"][key]
    torch.testing.assert_close(value, torch.full_like(value, 4))
    torch.testing.assert_close(
        initialized.constants["populations"]["right"]["parametrizations.factor"], value
    )
    torch.testing.assert_close(
        torch.autograd.grad(
            value.sum(), tensors.parameters["populations"]["left"][key]
        )[0],
        torch.ones_like(value),
    )
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        network.initialize(functional.dt)
    torch.testing.assert_close(network.left.get_parameter(key), value)
    actual, _ = functional.prepare_and_step(*initialized)
    network.step()
    _close(actual, functional.extract(network).state)


def test_cached_population_frozen_coordinates_reject_same_shape_edits_without_mutation():
    network, _key = _cross_population_binding()
    network.cache_state()
    functional, tensors = dn.func.make_functional_network(network)
    inputs = functional.initialization_inputs
    original = tensors.state["populations"]["left"]["integrator"]["v"].clone()
    inputs.cache.populations["left"]["x"].add_(1)
    with pytest.raises(dn.func.FunctionalizationError, match="frozen topology"):
        functional.initialize(*tensors, inputs)
    torch.testing.assert_close(
        tensors.state["populations"]["left"]["integrator"]["v"], original
    )


@pytest.mark.parametrize("saved", (None, [], 0))
def test_falsy_connection_caches_are_ignored_like_native_initialization(saved):
    network, _key = _cross_population_binding()
    network.cache_state()
    network.load_state_cache(syn_cache=saved)
    functional, tensors = dn.func.make_functional_network(network)
    assert functional._cache_error is None
    actual = functional.initialize(*tensors, functional.initialization_inputs)
    assert not functional.initialization_inputs.cache.synapses
    assert not functional.initialization_inputs.cache.continuous_synapses
    _close(actual.state, tensors.state)


def test_malformed_connection_cache_defers_error_until_cache_initialization():
    network, _key = _cross_population_binding()
    network.cache_state()
    network.load_state_cache(syn_cache=["invalid"])
    functional, tensors = dn.func.make_functional_network(network)
    assert "must be a mapping" in functional._cache_error
    _close(functional.extract().state, tensors.state)
    with pytest.raises(dn.func.FunctionalizationError, match="must be a mapping"):
        functional.initialize(*tensors, functional.initialization_inputs)
