"""Network sources use current explicit frames and preserve genuine tensor ties."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._network_bindings import AliasSnapshots, SourceBinding
from dendra.func._schedules import TensorScheduleTensors
from dendra.models.mod import graded_syn, pas
from tests.networks.test_functional_event_delivery import ThresholdRelay

pytestmark = pytest.mark.cpu
DT = 0.125
DTYPE = torch.float64


def _network(*, duplicate_owner=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        cell = dn.SingleCompartment(
            N=2, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-65.0
        )
        cell.insert(pas, g=0.001, e=-60.0)
        cell.insert(ThresholdRelay.rename("relay"))
        cell.insert(graded_syn.rename("analog"), e=-10.0)
        populations = {"cell": cell}
        if duplicate_owner:
            populations["alias"] = cell
        network = dn.Network(populations)
        network.train()
        network.initialize(DT)
    return network


@pytest.mark.parametrize(
    "reference",
    ("v", "t", "mech.relay.level", "mech.relay.mirror", "mech.analog.e"),
)
def test_registered_source_bindings_match_the_native_field(reference):
    network = _network()
    population, tensors = dn.func.make_functional(network.cell, dt=DT)
    prepared = population._prepare_values(tensors.parameters, tensors.constants)
    source = SourceBinding(population, network.cell, reference)
    expected = network.cell
    for field in reference.split("."):
        expected = getattr(expected, field)
    actual = source.resolve(tensors.parameters, prepared, tensors.state)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == expected.shape


def test_source_binding_observes_receiver_updated_carry_and_its_gradient():
    network = _network()
    population, tensors = dn.func.make_functional(network.cell, dt=DT)
    prepared = population._prepare_values(tensors.parameters, tensors.constants)
    source = SourceBinding(population, network.cell, "mech.relay.level")
    original = source.resolve(tensors.parameters, prepared, tensors.state).clone()
    payload = torch.full_like(original, 0.3, requires_grad=True)
    current = population.prepare_and_receive(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        target="relay",
        payload=payload,
    )
    observed = source.resolve(tensors.parameters, prepared, current)
    assert observed is current["mechanism_buffers"]["relay"]["level"]
    assert not torch.equal(observed, original)
    (gradient,) = torch.autograd.grad(observed.sum(), payload)
    torch.testing.assert_close(gradient, torch.ones_like(payload))
    torch.testing.assert_close(
        source.resolve(tensors.parameters, prepared, tensors.state), original
    )
    torch.testing.assert_close(network.cell.mech.relay.level, original)


def test_raw_parameter_source_uses_the_explicit_replacement_tensor():
    network = _network()
    population, tensors = dn.func.make_functional(network.cell, dt=DT)
    reference = next(iter(network.cell.named_parameters()))[0]
    source = SourceBinding(population, network.cell, reference)
    parameters = dict(tensors.parameters)
    replacement = (parameters[reference].detach() + 0.2).requires_grad_()
    parameters[reference] = replacement
    prepared = population._prepare_values(parameters, tensors.constants)
    observed = source.resolve(parameters, prepared, tensors.state)
    assert observed is replacement
    (gradient,) = torch.autograd.grad(observed.sum(), replacement)
    torch.testing.assert_close(gradient, torch.ones_like(replacement))


def test_unrepresented_integrator_workspace_source_rejects_cold():
    network = _network()
    network.cell.integrator.register_buffer(
        "network_probe_workspace", torch.zeros_like(network.cell.v)
    )
    population, _ = dn.func.make_functional(network.cell, dt=DT)
    with pytest.raises(dn.func.FunctionalizationError, match="no explicit functional"):
        SourceBinding(population, network.cell, "integrator.network_probe_workspace")


def test_current_voltage_and_prepared_field_binding_compile_with_gradients():
    network = _network()
    population, tensors = dn.func.make_functional(network.cell, dt=DT)
    voltage = SourceBinding(population, network.cell, "v")
    reversal = SourceBinding(population, network.cell, "mech.analog.e")

    def read(parameters, constants, state):
        prepared = population._prepare_values(parameters, constants)
        return voltage.resolve(parameters, prepared, state) * reversal.resolve(
            parameters, prepared, state
        )

    initial = tensors.state["integrator"]["v"].detach().clone().requires_grad_()
    state = {**tensors.state, "integrator": {"v": initial}}
    expected = read(tensors.parameters, tensors.constants, state)
    with torch_compiler_warning_context():
        actual = torch.compile(read, backend="aot_eager", fullgraph=True)(
            tensors.parameters, tensors.constants, state
        )
    torch.testing.assert_close(actual, expected)
    (actual_gradient,) = torch.autograd.grad(actual.sum(), initial, retain_graph=True)
    (expected_gradient,) = torch.autograd.grad(expected.sum(), initial)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_alias_snapshots_keep_one_independent_trainable_leaf_for_native_ties():
    native = torch.nn.Parameter(torch.tensor([0.2, 0.4], dtype=DTYPE))
    aliases = AliasSnapshots()
    explicit = aliases.tree(
        {"left": native, "nested": {"right": native}},
        {"left": native.clone(), "nested": {"right": native.clone()}},
    )
    left, right = explicit["left"], explicit["nested"]["right"]
    assert left is right
    assert left.is_leaf and left.requires_grad
    assert left.data_ptr() != native.data_ptr()
    (actual,) = torch.autograd.grad((left + 2 * right).square().sum(), left)
    (expected,) = torch.autograd.grad((native + 2 * native).square().sum(), native)
    torch.testing.assert_close(actual, expected)


def test_schedule_packet_aliases_use_their_native_named_correspondence():
    native = torch.nn.Parameter(torch.tensor([0.2, 0.4], dtype=DTYPE))
    packet = TensorScheduleTensors(native.clone(), native.clone())
    aliases = AliasSnapshots()
    selected = aliases.tree({"times": native, "amplitudes": native}, packet)
    assert selected.times is selected.amplitudes
    assert selected.time_source is None and selected.amplitude_source is None
    assert selected.times.is_leaf and selected.times.requires_grad
    torch.testing.assert_close(selected.times, native)


def test_alias_snapshots_do_not_tie_equal_values_from_different_native_tensors():
    left = torch.tensor([0.2, 0.4], dtype=DTYPE)
    right = left.clone()
    aliases = AliasSnapshots()
    selected = aliases.tree((left, right), (left, right))
    assert selected[0] is not selected[1]
    torch.testing.assert_close(selected[0], selected[1])


def test_alias_snapshots_reject_incompatible_effective_component_schemas():
    native = torch.nn.Parameter(torch.tensor([0.2, 0.4], dtype=DTYPE))
    aliases = AliasSnapshots()
    aliases.snapshot(native, native)
    with pytest.raises(dn.func.FunctionalizationError, match="incompatible explicit"):
        aliases.snapshot(native, native.reshape(2, 1))


@pytest.mark.parametrize("trainable_first", (False, True))
def test_alias_snapshots_merge_reader_trainability_without_mutating_sources(
    trainable_first,
):
    native = torch.tensor([0.2, 0.4], dtype=DTYPE, requires_grad=True)
    aliases = AliasSnapshots()
    first = native if trainable_first else native.detach()
    second = native.detach() if trainable_first else native
    with torch.inference_mode():
        left = aliases.snapshot(native, first)
        right = aliases.snapshot(native, second)
    assert left is right
    assert left.is_leaf and left.requires_grad
    assert not left.is_inference()
    assert left.data_ptr() != native.data_ptr()
    with torch.no_grad():
        left.add_(0.3)
    torch.testing.assert_close(native, torch.tensor([0.2, 0.4], dtype=DTYPE))
    assert native.requires_grad
    gradient = torch.autograd.grad((left + right).square().sum(), left)[0]
    torch.testing.assert_close(gradient, 8 * left)


def test_duplicate_population_owners_reject_instead_of_losing_native_state_aliases():
    network = _network(duplicate_owner=True)
    assert network.cell is network.alias
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_network(network)
