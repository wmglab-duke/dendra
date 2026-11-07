"""Fixed source delays retain checkpoint compatibility within one route bin."""

import pytest
import torch

import dendra as dn
from tests.func.test_network import DT, _clone, _close, _mixed_network

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("collection", ("synapses", "continuous_synapses"))
@pytest.mark.parametrize("same_bin", (False, True))
def test_native_fixed_delay_source_updates_validate_routing_before_commit(
    collection, same_bin
):
    functional, packet = dn.func.make_functional_network(_mixed_network())
    target = _mixed_network()
    prefix, _ = functional.prepare_and_rollout(*packet, steps=3)
    name, connection = next(iter(getattr(target, collection).items()))
    source = connection.delay_ms.weights[0].param
    physical = source._compute().detach() + (0.1 if same_bin else 1.0) * DT
    raw = source._inverse_transform(physical)
    with torch.no_grad():
        source.rho.copy_(raw)
    connection.delay_ms.init()
    frozen = _clone(target.state_dict_for_checkpoint(), detached=True)
    if not same_bin:
        with pytest.raises(dn.func.FunctionalizationError, match="topology|plan"):
            functional.extract(target)
        with pytest.raises(dn.func.FunctionalizationError, match="topology|plan"):
            functional.commit_state_(target, prefix)
        _close(target.state_dict_for_checkpoint(), frozen)
        return
    # Physical values can change while the integer route and ring stay fixed.
    extracted = functional.extract(target)
    torch.testing.assert_close(
        extracted.parameters[collection][name]["delay_ms.weights.0.param.rho"], raw
    )
    parameters = {
        **packet.parameters,
        collection: {
            **packet.parameters[collection],
            name: {
                **packet.parameters[collection][name],
                "delay_ms.weights.0.param.rho": raw,
            },
        },
    }
    functional.commit_state_(target, prefix)
    expected, _ = functional.prepare_and_rollout(
        parameters, packet.constants, prefix, steps=3
    )
    for _ in range(3):
        target.step()
    _close(functional.extract(target).state, expected)
