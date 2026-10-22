"""Custom operator configuration retains explicit tensor ownership."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_custom_material_processes import DT, DTYPE, _network, _population
from tests.func.test_functional_initialization import _assert_tree_close

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


def test_custom_config_buffer_alias_uses_same_population_raw_parameter_binding():
    population = _population(physiology=False)
    process = population.mech.material_processes["local"]
    process._buffers["offset"] = process.scale
    native = copy.deepcopy(population)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    key = next(
        name
        for name in tensors.parameters
        if name.endswith("material_processes.local.scale")
    )
    leaf = torch.tensor(1.1, dtype=DTYPE, requires_grad=True)
    parameters = {**tensors.parameters, key: leaf}
    final, _ = functional.prepare_and_step(parameters, tensors.constants, tensors.state)
    with torch.no_grad():
        native.mech.material_processes["local"].scale.fill_(1.1)
    native.step(DT)
    _assert_tree_close(final, functional.extract(native).state)
    actual = torch.autograd.grad(final["materials"]["pool"]["amount"].sum(), leaf)[0]
    expected = torch.autograd.grad(
        native.mech.materials["pool"].amount.sum(),
        native.mech.material_processes["local"].scale,
    )[0]
    torch.testing.assert_close(actual, expected)
    assert actual.abs() > 0


def test_custom_config_cannot_alias_population_runtime_field():
    population = _population(physiology=False)
    population.mech.material_processes["local"].offset = population.mech.materials[
        "pool"
    ].amount
    with pytest.raises(dn.func.FunctionalizationError, match="alias runtime state"):
        dn.func.make_functional(population, dt=DT)


def test_partial_storage_alias_of_noncached_config_fails_closed_for_cache_restore():
    population = _population(physiology=False)
    process = population.mech.material_processes["local"]
    process.register_buffer("cachebase", torch.tensor([0.75, 0.9], dtype=DTYPE))
    process.offset = process.cachebase[:1]
    process._non_persistent_buffers_set.add("offset")
    with torch.no_grad():
        population.cache("_steady_state")
    functional, tensors = dn.func.make_functional(population, dt=DT)
    with pytest.raises(dn.func.FunctionalizationError, match="partially overlapping"):
        functional.initialize(
            tensors.parameters, tensors.constants, tensors.initialization
        )


@pytest.mark.parametrize("alias", ("parameter", "runtime"))
def test_network_custom_config_cannot_capture_another_populations_owned_tensor(alias):
    network = _network()
    local = network.left.mech.material_processes["local"]
    if alias == "parameter":
        # This Parameter is registered only on right, unlike the normally
        # shared scale fixture, which is registered on both processes.
        external = torch.nn.Parameter(
            torch.tensor(0.7, dtype=DTYPE), requires_grad=False
        )
        network.right.mech.material_processes["local"].external = external
        local._buffers["offset"] = external
        message = "register the same Parameter on the process"
    else:
        local.offset = network.right.mech.materials["pool"].amount
        message = "aliases runtime state"
    with pytest.raises(dn.func.FunctionalizationError, match=message):
        dn.func.make_functional_network(network)


@pytest.mark.parametrize("cache_kind", ("population", "network"))
@pytest.mark.parametrize("view", (False, True))
def test_noncached_config_cannot_miss_another_populations_cache_write(cache_kind, view):
    network = _network()
    reader = network.left.mech.material_processes["local"]
    writer = network.right.mech.material_processes["local"]
    if view:
        writer.register_buffer("saved_config", torch.tensor([0.75, 0.75], dtype=DTYPE))
        writer.offset = writer.saved_config[:1]
        reader.offset = writer.saved_config[1:]
    else:
        reader.offset = writer.offset
    reader._non_persistent_buffers_set.add("offset")
    with torch.no_grad():
        if cache_kind == "network":
            network.cache_state()
        else:
            network.right.cache("_steady_state")
        writer.offset.fill_(2.0)
        if view:
            writer.saved_config.fill_(2.0)
    functional, tensors = dn.func.make_functional_network(network)
    with pytest.raises(
        dn.func.FunctionalizationError, match="nonpersistent custom MaterialProcess"
    ):
        functional.initialize(*tensors, functional.initialization_inputs)
    # Snapshot stepping never restores a steady-state cache.
    state, _ = functional.prepare_and_step(*tensors)
    assert torch.isfinite(state["populations"]["left"]["integrator"]["v"]).all()


@pytest.mark.parametrize(
    "attribute,value",
    (("_material_process_phase", "transport"), ("_material_process_method", "changed")),
)
def test_custom_scheduler_metadata_changes_require_lowering_again(attribute, value):
    population = _population(physiology=False)
    functional, tensors = dn.func.make_functional(population, dt=DT)
    process = population.mech.material_processes["local"]
    original = getattr(type(process), attribute)
    try:
        setattr(type(process), attribute, value)
        with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
            functional.prepare(tensors.parameters, tensors.constants)
    finally:
        setattr(type(process), attribute, original)


@pytest.mark.parametrize("cache_kind", ("population", "network"))
def test_persistent_alias_preserves_represented_noncached_config_reader(cache_kind):
    network = _network()
    reader = network.left.mech.material_processes["local"]
    writer = network.right.mech.material_processes["local"]
    # Isolate configuration sharing from the separate unsupported ordering
    # of a later cache overwriting an earlier reader's shared raw parameter.
    writer.scale = torch.nn.Parameter(writer.scale.detach().clone())
    reader.offset = writer.offset
    reader.register_buffer("persistent_offset", reader.offset)
    reader._non_persistent_buffers_set.add("offset")
    with torch.no_grad():
        if cache_kind == "network":
            network.cache_state()
        else:
            network.right.cache("_steady_state")
        writer.offset.fill_(2.0)
    native = copy.deepcopy(network)
    functional, tensors = dn.func.make_functional_network(network)
    initial = functional.initialize(*tensors, functional.initialization_inputs)
    native.initialize(DT)
    final, _ = functional.prepare_and_step(
        initial.parameters, initial.constants, initial.state
    )
    native.step()
    _assert_tree_close(final, functional.extract(native).state)
