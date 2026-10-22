"""Network bindings inherit Population stimulation admission and alias contracts."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.implicit import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas
from tests.func.test_network import _close
from tests.func.test_network_extra import _network as _cable_network
from tests.func.test_stimulation import (
    _MutatingPureWaveform,
    _StochasticPureWaveform,
    _UnmarkedCustomWaveform,
)

pytestmark = pytest.mark.cpu
DT = 0.01
DTYPE = torch.float64
FIELD = "stimulation.extra.contacts.0.field"


def _network(*, mixed_dtype=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        populations = {}
        for name in ("left", "right"):
            dtype = torch.float32 if mixed_dtype and name == "right" else DTYPE
            population = dn.SingleCompartment(
                N=1, C=2, dtype=dtype, integrator=dn.bwd_euler_sc(imem=False)
            )
            population.insert(pas, g=0.001, e=-65.0)
            populations[name] = population
        network = dn.Network(populations)
        network.train()
        network.initialize(DT)
    return network


def _waveform():
    with dn.ctx(REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=DTYPE):
        return dn.sin(amp=0.3, freq=0.7, phase=0.2)


@pytest.mark.parametrize("entrypoint", ("lower", "compile"))
@pytest.mark.parametrize(
    ("extra", "error", "match"),
    ((object(), TypeError, "extra.*map"), ({"missing": None}, ValueError, "unknown")),
)
def test_network_extra_mapping_rejects_invalid_owners(entrypoint, extra, error, match):
    network = _network()
    with pytest.raises(error, match=match):
        if entrypoint == "lower":
            dn.func.make_functional_network(network, extra=extra)
        else:
            dn.compile(network, dt=DT, steps=1, extra=extra, backend="eager")


@pytest.mark.parametrize("extra", (None, {}, {"left": None}))
def test_absent_extra_is_disabled_per_population(extra):
    network = _network()
    functional, tensors = dn.func.make_functional_network(network, extra=extra)
    assert not any(plan.extra.enabled for plan in functional.populations.values())
    assert not any("stimulation.extra" in name for name in tensors.named_parameters)


def test_extra_field_aliases_follow_object_identity_including_repeated_contacts():
    network = _network()
    field = torch.tensor([[0.2, -0.4]], dtype=DTYPE)
    view = field.view_as(field)
    waveform = _waveform()
    functional, tensors = dn.func.make_functional_network(
        network,
        extra={
            "left": [(field, waveform), (field, waveform)],
            "right": (view, waveform),
        },
    )
    left, right = (
        tensors.parameters["populations"][name] for name in ("left", "right")
    )
    assert left[FIELD] is left["stimulation.extra.contacts.1.field"]
    assert left[FIELD] is not right[FIELD]
    assert left[FIELD].data_ptr() != field.data_ptr()
    amplitude = "stimulation.extra.waveforms.0.amp"
    assert left[amplitude] is right[amplitude]
    independent = tensors.independent_parameters((f"populations.left.{amplitude}",))
    assert independent["populations"]["left"][amplitude].requires_grad
    assert (
        independent["populations"]["left"][amplitude]
        is independent["populations"]["right"][amplitude]
    )
    # Breaking one explicit alias is an invalid packet, even with equal values.
    independent["populations"]["right"][amplitude] = independent["populations"][
        "right"
    ][amplitude].clone()
    with pytest.raises(ValueError, match="aliases"):
        functional.prepare(independent, tensors.constants)


def test_shared_extra_rejects_incompatible_population_conversions():
    network = _network(mixed_dtype=True)
    field = torch.tensor([[0.2, -0.4]], dtype=DTYPE)
    with pytest.raises(
        dn.func.FunctionalizationError, match="shared Network Tensor.*incompatible"
    ):
        dn.func.make_functional_network(
            network, extra={name: (field, _waveform()) for name in network.populations}
        )


def test_bound_extra_and_explicit_ve_conflict_only_for_the_selected_population():
    network = _network()
    functional, tensors = dn.func.make_functional_network(
        network, extra={"left": (torch.ones_like(network.left.v), _waveform())}
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    valid = dn.func.NetworkRolloutInput(
        populations={
            "right": dn.func.RolloutInput(
                ve=torch.ones(2, *network.right.shape, dtype=DTYPE)
            )
        }
    )
    functional.rollout(tensors.parameters, prepared, tensors.state, valid, steps=2)
    invalid = dn.func.NetworkRolloutInput(
        populations={
            "left": dn.func.RolloutInput(
                ve=torch.ones(0, *network.left.shape, dtype=DTYPE)
            )
        }
    )
    with pytest.raises(ValueError, match="explicit ve.*bound extra"):
        functional.rollout(
            tensors.parameters, prepared, tensors.state, invalid, steps=0
        )
    with pytest.raises(ValueError, match="runtime extra.*bound"):
        functional.step(
            tensors.parameters,
            prepared,
            tensors.state,
            extra={"left": (torch.ones_like(network.left.v), _waveform())},
        )


@pytest.mark.parametrize(
    "waveform_type",
    (_UnmarkedCustomWaveform, _MutatingPureWaveform, _StochasticPureWaveform),
)
def test_network_extra_reuses_population_waveform_purity_checks(waveform_type):
    network = _network()
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional_network(
            network, extra={"left": (torch.ones_like(network.left.v), waveform_type())}
        )


def test_network_extra_reuses_population_tensor_temporal_restriction():
    network = _network()
    with pytest.raises(dn.func.FunctionalizationError, match="Tensor-temporal extra"):
        dn.func.make_functional_network(
            network, extra={"left": (torch.ones_like(network.left.v), torch.ones(3))}
        )


def test_network_extra_freezes_mapping_and_owns_its_waveform_recipe():
    network = _network()
    waveform = _waveform()
    extra = {"left": (torch.ones_like(network.left.v), waveform)}
    functional, packet = dn.func.make_functional_network(network, extra=extra)
    extra.clear()
    assert functional.populations["left"].extra.enabled
    waveform.register_buffer("new_buffer", torch.ones((), dtype=DTYPE))
    # The authored extra is an external recipe. Execution owns its audited
    # copy, and new authored buffers do not enter the frozen explicit schema.
    _close(functional.extract(), packet)


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="packed scalar cable execution requires native CPU dendra-solvers",
)
def test_packed_network_extra_uses_top_level_owner_and_packed_layout():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context(), torch.no_grad():
            network = _cable_network(batch=None).concat(packed=["left", "right"])
            network.batch(2)
            network.train()
            network.initialize(DT)
            network._reset_runtime_clock(0.175)
            population = network.packed
            steps = 3
            profile = torch.linspace(-1.2, 1.7, population.shape[-1], dtype=DTYPE)
            field = profile.reshape(1, -1)
            batch_field = torch.stack((profile.flip(-1), -0.4 * profile)).unsqueeze(1)
            with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu"):
                first = dn.sin(amp=0.2, freq=0.7, phase=0.3)
                # A singleton oscillator axis retains distinct batch amplitudes.
                second = dn.cos(
                    amp=torch.tensor([[0.1], [-0.2]], dtype=DTYPE),
                    freq=0.4,
                    phase=-0.2,
                )
            with pytest.raises(ValueError, match="unknown"):
                dn.func.make_functional_network(network, extra={"left": (field, first)})
            bound, packet = dn.func.make_functional_network(
                network,
                extra={"packed": [(field, first), (batch_field, second)]},
            )
            ordinary, tensors = dn.func.make_functional_network(network)
            times = network.t + torch.arange(steps, dtype=DTYPE) * DT
            # Assemble independently of FunctionalExtra, including its temporal
            # batch-only fallback for the second Waveform's (batch, time) output.
            ve = first(times).reshape(steps, 1, 1, 1) * field
            ve = ve + second(times).movedim(-1, 0).reshape(steps, 2, 1, 1) * batch_field
            assert ve.shape == (steps, *population.shape)
            inputs = dn.func.NetworkRolloutInput(
                populations={"packed": dn.func.RolloutInput(ve=ve)}
            )
            actual, _ = bound.prepare_and_rollout(*packet, steps=steps)
            expected, _ = ordinary.prepare_and_rollout(*tensors, inputs, steps=steps)
            baseline, _ = ordinary.prepare_and_rollout(*tensors, steps=steps)
        _close(actual, expected)
        voltage = actual["populations"]["packed"]["integrator"]["v"]
        unbound = baseline["populations"]["packed"]["integrator"]["v"]
        assert (voltage - unbound).abs().max() > 1e-5
        assert not torch.equal(voltage[0], voltage[1])
    finally:
        torch.set_num_threads(previous)
