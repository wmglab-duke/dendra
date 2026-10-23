"""Public Mapping access finishes before Network freshness checks/preparation."""

from collections.abc import Mapping
from types import MappingProxyType

import pytest
import torch

import dendra as dn
from tests.func.test_network import DTYPE, _close, _loss, _mixed_network

pytestmark = pytest.mark.cpu


class _OnItems(Mapping):
    def __init__(self, values, action):
        self.values = values
        self.action = action
        self.accessed = False

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def items(self):
        if not self.accessed:
            self.accessed = True
            self.action()
        return self.values.items()


def _change_capacitance(tensors):
    with torch.no_grad():
        tensors.parameters["populations"]["cell"]["cm_scale_param.rho"].mul_(2)


def _packet(parameters, prepared, state, inputs, selected, action):
    packets = dict(parameters=parameters, prepared=prepared, state=state, inputs=inputs)
    values = inputs.populations if selected == "inputs" else packets[selected]
    wrapped = _OnItems(values, action)
    packets[selected] = (
        dn.func.NetworkRolloutInput(wrapped) if selected == "inputs" else wrapped
    )
    return packets, wrapped


@pytest.mark.parametrize("execution", ("chunk", "rollout"))
@pytest.mark.parametrize("packet", ("parameters", "prepared", "state", "inputs"))
def test_mapping_access_rejects_preparation_made_stale_by_public_packet(
    execution, packet
):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    inputs = dn.func.NetworkRolloutInput(populations={})
    packets, wrapped = _packet(
        tensors.parameters,
        prepared,
        tensors.state,
        inputs,
        packet,
        lambda: _change_capacitance(tensors),
    )
    execute = (
        functional.compile_rollout_chunk(5, disable=True)
        if execution == "chunk"
        else lambda **values: functional.rollout(**values, steps=5)
    )
    with pytest.raises(dn.func.FunctionalizationError, match="changed after prepare"):
        execute(**packets)
    assert wrapped.accessed


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize("checkpoint_every", (None, 3))
@pytest.mark.parametrize("packet", ("parameters", "constants", "state", "inputs"))
def test_host_normalizes_public_packets_before_coefficient_preparation(
    compiled, checkpoint_every, packet
):
    functional, tensors = dn.func.make_functional_network(_mixed_network())
    packets = dict(
        parameters=tensors.parameters,
        constants=tensors.constants,
        state=tensors.state,
        inputs=dn.func.NetworkRolloutInput(populations={}),
    )
    values = packets[packet].populations if packet == "inputs" else packets[packet]
    wrapped = _OnItems(values, lambda: _change_capacitance(tensors))
    packets[packet] = (
        dn.func.NetworkRolloutInput(wrapped) if packet == "inputs" else wrapped
    )
    kernel = functional.compile_rollout_chunk(2, disable=True) if compiled else None
    actual, _ = functional.run(
        **packets,
        steps=5,
        chunklength=2,
        checkpoint_every=checkpoint_every,
        compiled_chunk=kernel,
    )
    assert wrapped.accessed
    expected, _ = functional.prepare_and_rollout(
        tensors.parameters, tensors.constants, tensors.state, steps=5
    )
    _close(actual, expected)
    weight = next(iter(tensors.parameters["synapses"].values()))[
        "weight.weights.0.param.rho"
    ]
    observed = torch.autograd.grad(_loss(actual), weight)[0]
    oracle = torch.autograd.grad(_loss(expected), weight)[0]
    torch.testing.assert_close(observed, oracle)
    assert observed.abs().sum() > 0


@pytest.mark.parametrize("packet", ("parameters", "constants", "inputs"))
def test_normalized_checkpoint_operands_keep_original_public_rebinding_guards(packet):
    model = _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    population_inputs = {
        "cell": dn.func.RolloutInput(
            intra=torch.zeros((5, *model.cell.shape), dtype=DTYPE)
        )
    }
    actual, _ = functional.run(
        MappingProxyType(tensors.parameters),
        MappingProxyType(tensors.constants),
        MappingProxyType(tensors.state),
        dn.func.NetworkRolloutInput(MappingProxyType(population_inputs)),
        steps=5,
        chunklength=2,
        checkpoint_every=3,
        compiled_chunk=functional.compile_rollout_chunk(2, disable=True),
    )
    if packet == "inputs":
        population_inputs["cell"] = dn.func.RolloutInput(
            intra=population_inputs["cell"].intra.clone()
        )
    else:
        owner = getattr(tensors, packet)
        populations = {
            name: dict(value) for name, value in owner["populations"].items()
        }
        values = populations["cell"]
        key = next(iter(values))
        values[key] = values[key].clone()
        owner["populations"] = populations
    weight = next(iter(tensors.parameters["synapses"].values()))[
        "weight.weights.0.param.rho"
    ]
    with pytest.raises(
        dn.func.FunctionalizationError, match="changed after checkpoint"
    ):
        torch.autograd.grad(_loss(actual), weight)


@pytest.mark.parametrize("execution", ("chunk", "run"))
def test_mapping_access_source_mutation_is_checked_before_execution(execution):
    model = _mixed_network()
    functional, tensors = dn.func.make_functional_network(model)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    inputs = dn.func.NetworkRolloutInput(_OnItems({}, model.eval))
    with pytest.raises(dn.func.FunctionalizationError):
        if execution == "chunk":
            functional.compile_rollout_chunk(2, disable=True)(
                tensors.parameters, prepared, tensors.state, inputs
            )
        else:
            functional.run(
                tensors.parameters, tensors.constants, tensors.state, inputs, steps=2
            )
