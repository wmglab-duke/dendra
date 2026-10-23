"""Snapshot baselines belong to extracted packets, including alternate Networks."""

import pytest
import torch
from torch.nn.utils.parametrize import register_parametrization

import dendra as dn
from dendra.models.mechanisms import Mechanism
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_initialization_ordering import (
    RAW_CM,
    RAW_G,
    _Affine,
    _native_voltage,
    _network,
    _packed_network,
    _SharedScale,
    _torch_physical_network,
    _voltage,
)

pytestmark = pytest.mark.cpu


class _BooleanDerivedLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.DERIVED_BUFFER("enabled", dtype=torch.bool)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def derive_buffers(self):
        return {"enabled": self.g > 0}

    def i(self, v):
        return self.enabled * self.g * (v - self.e)


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
            yield
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize(
    "factory", (_network, _packed_network, _torch_physical_network)
)
@pytest.mark.parametrize("stale_source", (False, True))
def test_alternate_extraction_preserves_its_own_native_preparation(
    factory, stale_source
):
    source, alternate = factory(), factory()
    if stale_source:
        source.initialize(DT)
    else:
        alternate.initialize(DT)
    functional, default = dn.func.make_functional_network(source)
    before, _ = functional.prepare_and_step(*default)
    default_frozen = _clone(default, detached=True)
    packet = functional.extract(alternate)
    packet_frozen = _clone(packet, detached=True)
    actual, _ = functional.prepare_and_rollout(*packet, steps=3)
    for _ in range(3):
        alternate.step()
    torch.testing.assert_close(_voltage(actual), _native_voltage(alternate))
    # Extraction changes neither the plan's default baseline nor earlier packets.
    after, _ = functional.prepare_and_step(*default)
    _close(after, before)
    _close(default, default_frozen)
    _close(packet, packet_frozen)


def test_extracting_changed_source_buffers_refreshes_only_the_returned_baseline():
    source, native = _network(), _network()
    functional, default = dn.func.make_functional_network(source)
    before, _ = functional.prepare_and_step(*default)
    # A raw numerical edit leaves native prepared capacitance and solver buffers
    # at their earlier values, without changing Network delivery generation.
    for model in (source, native):
        with torch.no_grad():
            model.left.cm_scale_param.rho.add_(0.4)
    packet = functional.extract()
    actual, _ = functional.prepare_and_step(*packet)
    native.step()
    torch.testing.assert_close(_voltage(actual), _native_voltage(native))
    after, _ = functional.prepare_and_step(*default)
    _close(after, before)


def test_alternate_snapshot_offsets_keep_explicit_raw_updates_differentiable():
    source, alternate = _network(), _network()
    functional, _ = dn.func.make_functional_network(source)
    alternate.initialize(DT)
    packet = functional.extract(alternate)
    original = packet.parameters["populations"]["left"][RAW_CM]

    def voltage(raw):
        parameters = torch.utils._pytree.tree_map(
            lambda value: raw if value is original else value, packet.parameters
        )
        state, _ = functional.prepare_and_rollout(
            parameters, packet.constants, packet.state, steps=3
        )
        return _voltage(state)

    shifted = original + 0.2
    assert torch.all((voltage(shifted) - voltage(original)).abs() > 1e-6)
    reverse = torch.func.jacrev(voltage)(shifted)
    forward = torch.func.jacfwd(voltage)(shifted)
    torch.testing.assert_close(reverse, forward)
    epsilon = shifted.new_tensor(1e-5)
    numerical = (voltage(shifted + epsilon) - voltage(shifted - epsilon)) / (
        2 * epsilon
    )
    torch.testing.assert_close(reverse, numerical, rtol=1e-7, atol=1e-8)


def _boolean_network(g=0.001):
    cell = dn.SingleCompartment(
        N=1, C=1, integrator=dn.bwd_euler_sc(imem=False), v_init=-50.0
    )
    cell.insert(_BooleanDerivedLeak, g=torch.nn.Parameter(torch.tensor(g, dtype=DTYPE)))
    network = dn.Network({"cell": cell})
    network.train()
    with torch.no_grad():
        network.initialize(DT)
    return network


@pytest.mark.parametrize("initialization", ("snapshot", "fresh"))
def test_boolean_derived_materializations_match_native_and_keep_raw_updates_live(
    initialization,
):
    source, native = _boolean_network(), _boolean_network()
    functional, packet = dn.func.make_functional_network(source)
    if initialization == "fresh":
        packet = functional.initialize(*packet, functional.initialization_inputs)
        native.initialize(DT)
    actual, _ = functional.prepare_and_rollout(*packet, steps=3)
    for _ in range(3):
        native.step()
    torch.testing.assert_close(_voltage(actual), _native_voltage(native))
    if initialization == "snapshot":
        name = "integrator.mech.mechanisms._BooleanDerivedLeak.g_param"
        parameters = {
            **packet.parameters,
            "populations": {
                "cell": {
                    **packet.parameters["populations"]["cell"],
                    name: torch.tensor(-0.001, dtype=DTYPE),
                }
            },
        }
        shifted, _ = functional.prepare_and_rollout(
            parameters, packet.constants, packet.state, steps=3
        )
        negative = _boolean_network(g=-0.001)
        for _ in range(3):
            negative.step()
        torch.testing.assert_close(_voltage(shifted), _native_voltage(negative))


def test_boolean_snapshot_offset_preserves_a_stale_derived_getter_baseline():
    def model():
        network = _network()
        shared = torch.nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
        network.right.register_parameter("conductance_scale", shared)
        network.left.insert(_BooleanDerivedLeak)
        network.initialize(DT)
        register_parametrization(
            network.left.mech.mechanisms._BooleanDerivedLeak,
            "g",
            _SharedScale(shared),
        )
        network.right.register_pre_initialize_transform(
            "negative_conductance",
            _Affine(),
            reads=("parameters.conductance_scale",),
            writes=("parameters.conductance_scale",),
            inputs={"factor": shared.new_tensor(-1.0), "offset": shared.new_zeros(())},
        )
        network.initialize(DT)
        return network

    source, native = model(), model()
    mechanism = native.left.mech.mechanisms._BooleanDerivedLeak
    assert mechanism.enabled.all() and (mechanism.g < 0).all()
    functional, packet = dn.func.make_functional_network(source)
    baseline = packet.constants["populations"]["left"][
        "materializations.baseline.integrator.mech.mechanisms._BooleanDerivedLeak.enabled"
    ]
    assert baseline.dtype == torch.bool and not baseline.any()
    actual, _ = functional.prepare_and_rollout(*packet, steps=3)
    for _ in range(3):
        native.step()
    torch.testing.assert_close(_voltage(actual), _native_voltage(native))


def test_snapshot_baseline_preserves_small_native_values_after_large_shared_writes():
    source, native = _network(), _network()
    for network in (source, native):
        factor = network.right._initialization_transforms["scale"].input_values()[0]
        with torch.no_grad():
            factor.fill_(1e20)
        network.initialize(DT)
    functional, packet = dn.func.make_functional_network(source)
    actual, _ = functional.prepare_and_rollout(*packet, steps=3)
    for _ in range(3):
        native.step()
    torch.testing.assert_close(_voltage(actual), _native_voltage(native))


def test_fresh_restart_discards_previous_materialization_autograd_graphs():
    functional, original = dn.func.make_functional_network(_network())
    parameter = original.parameters["populations"]["left"][RAW_G]
    constants = original.constants
    gradients = []
    for _ in range(2):
        initialized = functional.initialize(
            original.parameters,
            constants,
            original.state,
            functional.initialization_inputs,
        )
        state, _ = functional.prepare_and_rollout(*initialized, steps=3)
        gradients.append(torch.autograd.grad(_voltage(state).sum(), parameter)[0])
        constants = initialized.constants
    torch.testing.assert_close(gradients[0], gradients[1])
    assert gradients[1].abs() > 0
