"""Population cache writes reach shared external Network parameter readers."""

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_network import (
    DT,
    DTYPE,
    _clone,
    _close,
    _mixed_network,
    _stim_network,
)
from tests.func.test_network_bitpacked_inference import _sync_stimulus
from tests.func.test_network_cache_initialization import _warm_cache

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            yield
    finally:
        torch.set_num_threads(previous)


class _Factor(torch.nn.Module):
    def forward(self, value, factor):
        return value * (1 + factor)


def _shared_network(kind):
    if kind in {"constant", "broadcast"}:
        network = _stim_network()
        network.netstim.noise.zero_()
        if kind == "constant":
            network.cell.register_buffer("factor", network.netstim.start)
            network.cell.register_parametrization_in_graph(
                "cm", _Factor(), args=("factor",)
            )
            network.initialize(DT, reinit_weights=False, reinit_delays=False)
            source = "factor"
            parameter = ("netstim", "parameters", "start")
        else:
            shared = torch.tensor(0.23, dtype=network.cell.dtype())
            network.cell.register_buffer("tau_binding", shared)
            network.netstim.tau = shared
            source = "tau_binding"
            parameter = ("netstim", "parameters", "tau")
    else:
        network = _mixed_network()
        name, connection = next(
            (name, connection)
            for name, connection in network.continuous_synapses.items()
            if isinstance(connection.transform, torch.nn.Module)
        )
        shared = network.cell.mech.relay.scale.detach()
        network.cell.mech.relay.scale = shared
        connection.transform.offset = shared
        source = "mech.relay.scale"
        parameter = ("continuous_synapses", name, "transform_parameters", "offset")
    return _warm_cache(network), source, parameter


def _at(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def _replace(tree, path, value):
    if not path:
        return value
    key, *tail = path
    return {**tree, key: _replace(tree[key], tail, value)}


@pytest.mark.parametrize("kind", ("constant", "materialization", "broadcast"))
def test_population_buffer_cache_write_reaches_external_parameters_and_native_state(
    kind,
):
    source, key, path = _shared_network(kind)
    native, _, _ = _shared_network(kind)
    if source.netstim is not None:
        _sync_stimulus(source, native)
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    defaults = defaults._replace(
        netstim_draw=functional.sample_initialization_draw(
            tensors.parameters, tensors.state
        )
    )
    assert not functional._population_cache_plans["cell"].parameters.get(key)
    defaults.cache.populations["cell"][key].fill_(0.9)
    frozen = _clone(tensors, detached=True)
    initialized = functional.initialize(*tensors, defaults)
    expected = torch.full_like(_at(tensors.parameters, path), 0.9)
    torch.testing.assert_close(_at(initialized.parameters, path), expected)
    if kind == "constant":
        assert functional._parameter_constant_aliases
        torch.testing.assert_close(
            initialized.constants["populations"]["cell"]["parametrizations.factor"],
            expected,
        )
    elif kind == "materialization":
        assert not any(
            path == parameter for parameter, _ in functional._parameter_constant_aliases
        )
        torch.testing.assert_close(
            initialized.constants["populations"]["cell"][
                "cache.values.integrator.mech.mechanisms.relay.scale"
            ],
            expected,
        )
    native._state_cache = _clone(defaults.cache.populations, detached=True)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    native_parameters = functional.extract(native).parameters
    _close(initialized.parameters, native_parameters)
    draw = (
        None
        if functional.netstim is None
        else functional.netstim.sample_step_draw(initialized.state["netstim"])
    )
    actual, _ = functional.prepare_and_step(*initialized, netstim_draw=draw)
    native.step()
    _close(actual, functional.extract(native).state)
    _close(tensors, frozen)


def test_materialized_buffer_alias_restore_compiles_and_preserves_parameter_gradient_identity():
    source, key, path = _shared_network("materialization")
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    defaults.cache.populations["cell"][key].fill_(0.9)
    upstream = _at(tensors.parameters, path).detach().clone().requires_grad_()
    parameters = _replace(tensors.parameters, path, upstream * 3)
    compiled = torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
    initialized = compiled(parameters, tensors.constants, tensors.state, defaults)
    value = _at(initialized.parameters, path)
    torch.testing.assert_close(value, torch.full_like(value, 0.9))
    torch.testing.assert_close(
        torch.autograd.grad(value.sum(), upstream)[0], torch.full_like(upstream, 3)
    )


@pytest.mark.parametrize("lanes", (0, 2))
def test_materialized_buffer_alias_restore_accepts_mapped_destination_parameters(lanes):
    source, key, path = _shared_network("materialization")
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    defaults.cache.populations["cell"][key].fill_(0.9)

    def initialize(value):
        initialized = functional.initialize(
            _replace(tensors.parameters, path, value),
            tensors.constants,
            tensors.state,
            defaults,
        )
        return {
            "value": _at(initialized.parameters, path),
            "v": initialized.state["populations"]["cell"]["integrator"]["v"],
        }

    values = _at(tensors.parameters, path).expand(lanes).clone()
    actual = torch.vmap(initialize)(values)
    assert actual["value"].shape == (lanes,)
    for lane in range(lanes):
        _close(
            pytree.tree_map(lambda value: value[lane], actual), initialize(values[lane])
        )


def test_physical_buffer_external_alias_uses_final_native_storage_write():
    source = _stim_network()
    source.netstim.noise.zero_()
    shared = source.cell.celsius.detach()
    source.cell.celsius = shared
    source.netstim.tau = shared
    _warm_cache(source)
    assert source.cell.celsius is source.netstim.tau
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    defaults.cache.populations["cell"]["celsius"].fill_(0.9)
    defaults = defaults._replace(
        netstim_draw=functional.sample_initialization_draw(
            tensors.parameters, tensors.state
        )
    )
    initialized = functional.initialize(*tensors, defaults)
    # Native detached views of root/mechanism celsius and its raw Parameter
    # share a storage frame. A later mechanism copy still writes cached 37.
    value = initialized.parameters["netstim"]["parameters"]["tau"]
    torch.testing.assert_close(value, torch.full_like(value, 37))
    source._state_cache = _clone(defaults.cache.populations, detached=True)
    source.initialize(DT, reinit_weights=False, reinit_delays=False)
    torch.testing.assert_close(value, source.netstim.tau.expand_as(value))


def test_distinct_parameters_sharing_storage_keep_independent_destination_gradients():
    source = _mixed_network()
    base = torch.tensor(1.0, dtype=source.cell.dtype())
    source.cell.register_parameter("first", torch.nn.Parameter(base))
    source.cell.register_parameter("second", torch.nn.Parameter(base.detach()))
    assert source.cell.first is not source.cell.second
    assert source.cell.first.data_ptr() == source.cell.second.data_ptr()
    _warm_cache(source)
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    defaults.cache.populations["cell"]["first"].fill_(3)
    defaults.cache.populations["cell"]["second"].fill_(4)
    left = tensors.parameters["populations"]["cell"]["first"]
    right = tensors.parameters["populations"]["cell"]["second"]
    assert left is not right
    initialized = functional.initialize(*tensors, defaults)
    actual_left = initialized.parameters["populations"]["cell"]["first"]
    actual_right = initialized.parameters["populations"]["cell"]["second"]
    assert actual_left is not actual_right
    for value in (actual_left, actual_right):
        torch.testing.assert_close(value, torch.full_like(value, 4))
    left_gradients = torch.autograd.grad(
        actual_left.sum(), (left, right), allow_unused=True
    )
    right_gradients = torch.autograd.grad(
        actual_right.sum(), (left, right), allow_unused=True
    )
    torch.testing.assert_close(left_gradients[0], torch.ones_like(left))
    assert left_gradients[1] is None
    assert right_gradients[0] is None
    torch.testing.assert_close(right_gradients[1], torch.ones_like(right))


def test_expanded_population_buffer_copy_does_not_overwrite_outside_original_reader():
    def network():
        model = _mixed_network()
        name, connection = next(
            (name, connection)
            for name, connection in model.continuous_synapses.items()
            if isinstance(connection.transform, torch.nn.Module)
        )
        shared = torch.tensor(0.23, dtype=model.cell.dtype()).expand(2)
        model.cell.register_buffer("expanded_binding", shared)
        connection.transform.offset = shared
        return _warm_cache(model), name

    source, name = network()
    native, _ = network()
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    defaults.cache.populations["cell"]["expanded_binding"].fill_(0.9)
    initialized = functional.initialize(*tensors, defaults)
    path = ("continuous_synapses", name, "transform_parameters", "offset")
    expected = torch.full_like(_at(tensors.parameters, path), 0.23)
    torch.testing.assert_close(_at(initialized.parameters, path), expected)
    native._state_cache = _clone(defaults.cache.populations, detached=True)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    torch.testing.assert_close(
        native.cell.expanded_binding, torch.full_like(expected, 0.9)
    )
    torch.testing.assert_close(
        native.continuous_synapses[name].transform.offset, expected
    )
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


@pytest.mark.parametrize("compiled", (False, True))
def test_partial_storage_views_across_populations_follow_native_load_order(compiled):
    def network():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
            left = dn.SingleCompartment(
                N=2, C=1, integrator=dn.bwd_euler_sc(imem=False)
            )
            right = dn.SingleCompartment(
                N=1, C=1, integrator=dn.bwd_euler_sc(imem=False)
            )
            left.insert(pas, g=0.001, e=-65.0)
            right.insert(pas, g=0.001, e=-65.0)
            model = dn.Network({"left": left, "right": right})
            model.train()
            model.initialize(DT)
            voltage = torch.tensor([[-62.0], [-63.0]], dtype=DTYPE)
            left.v = voltage
            right.v = voltage[1:]
            model.cache_state()
            return model

    source, native = network(), network()
    functional, tensors = dn.func.make_functional_network(source)
    assert functional._cache_storage_replays
    defaults = functional.initialization_inputs
    defaults.cache.populations["left"]["v"].copy_(
        torch.tensor([[-50.0], [-51.0]], dtype=DTYPE)
    )
    defaults.cache.populations["right"]["v"].fill_(-40)
    execute = (
        torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
        if compiled
        else functional.initialize
    )
    initialized = execute(*tensors, defaults)
    expected = torch.tensor([[-50.0], [-40.0]], dtype=DTYPE)
    torch.testing.assert_close(
        initialized.state["populations"]["left"]["integrator"]["v"], expected
    )
    if not compiled:

        def restore_left(voltage):
            cache = defaults.cache._replace(
                populations={
                    **defaults.cache.populations,
                    "left": {**defaults.cache.populations["left"], "v": voltage},
                }
            )
            return functional.initialize(
                *tensors, defaults._replace(cache=cache)
            ).state["populations"]["left"]["integrator"]["v"]

        for lanes in (0, 2):
            values = (
                defaults.cache.populations["left"]["v"].expand(lanes, -1, -1).clone()
            )
            actual = torch.vmap(restore_left)(values)
            expected_lanes = values.clone()
            expected_lanes[:, 1] = -40
            torch.testing.assert_close(actual, expected_lanes)
    native._state_cache = _clone(defaults.cache.populations, detached=True)
    native.initialize(DT, reinit_weights=False, reinit_delays=False)
    _close(initialized.state, functional.extract(native).state)


def test_external_parameter_partial_view_receives_native_cached_buffer_copy():
    source = _stim_network()
    source.netstim.noise.zero_()
    _warm_cache(source)
    source.netstim.start = source.cell.v.reshape(-1)
    functional, tensors = dn.func.make_functional_network(source)
    assert functional._cache_storage_replays
    defaults = functional.initialization_inputs
    defaults.cache.populations["cell"]["v"].fill_(-30)
    defaults = defaults._replace(
        netstim_draw=functional.sample_initialization_draw(
            tensors.parameters, tensors.state
        )
    )
    initialized = functional.initialize(*tensors, defaults)
    start = initialized.parameters["netstim"]["parameters"]["start"]
    torch.testing.assert_close(start, torch.full_like(start, -30))
    source._state_cache = _clone(defaults.cache.populations, detached=True)
    source.initialize(DT, reinit_weights=False, reinit_delays=False)
    torch.testing.assert_close(start, source.netstim.start)


def test_negative_external_view_rejects_fresh_cache_and_keeps_snapshot_execution():
    source = _stim_network()
    source.netstim.noise.zero_()
    _warm_cache(source)
    source.netstim.start = torch._neg_view(source.cell.v.reshape(-1))
    assert source.netstim.start.is_neg()
    functional, tensors = dn.func.make_functional_network(source)
    with pytest.raises(dn.func.FunctionalizationError, match="negative|conjugate"):
        functional.initialize(*tensors, functional.initialization_inputs)
    continued, _ = functional.prepare_and_step(
        *tensors, netstim_draw=functional.sample_step_draw(tensors.state)
    )
    runner = dn.compile(
        source, dt=DT, steps=1, initialization="snapshot", inference=False, disable=True
    )
    snapshot, _ = runner()
    _close(snapshot, continued)
