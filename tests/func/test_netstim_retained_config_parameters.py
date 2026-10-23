"""Retained connection values keep shared NetStim config-source Jacobians."""

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from tests.func.test_connection_parameter_initialization import _network
from tests.func.test_network import DT, DTYPE

pytestmark = pytest.mark.cpu


def _model(*, trainable_config=False):
    model, source = _network(config=True)
    source.scale.requires_grad_(trainable_config)
    model.netstim = dn.NetStim(
        N=1, interval=0.4, noise=0.0, tau=source.scale, seed=21, dtype=DTYPE
    )
    model.initialize(DT)
    model.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=False,
        diff_spiking=True,
        diff_scheduled_times=False,
        train_delay_backend="dense",
    )
    return model, source.scale


@pytest.mark.parametrize("collection", ("synapses", "continuous_synapses"))
@pytest.mark.parametrize("refresh", (False, True))
def test_shared_config_weight_retention_preserves_native_raw_vjp(collection, refresh):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        model, _ = _model()
        native, native_config = _model(trainable_config=True)
        functional, packet = dn.func.make_functional_network(model)
        parameters = packet.independent_parameters("netstim.parameters.tau")
        raw = parameters["netstim"]["parameters"]["tau"]
        initialized = functional._initialize_host(
            parameters,
            packet.constants,
            packet.state,
            functional.initialization_inputs._replace(reinit_weights=refresh),
        )
        name = next(iter(getattr(native, collection)))
        connection = getattr(native, collection)[name]
        previous = connection.weight.w
        native.initialize(DT, reinit_weights=refresh)
        if not refresh:
            assert connection.weight.w is previous
        weights = functional._materialize_connections(
            initialized.parameters, initialized.constants
        )[collection][name]["weights"]
        torch.testing.assert_close(weights, connection.weight.w)
        actual = torch.autograd.grad(weights.sum(), raw)[0]
        expected = torch.autograd.grad(connection.weight.w.sum(), native_config)[0]
        torch.testing.assert_close(actual, expected)
        assert expected.abs().item() > 0


@pytest.mark.parametrize("collection", ("synapses", "continuous_synapses"))
def test_retained_config_supports_forward_ad_and_batched_source_vjps(collection):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        model, _ = _model()
        native, native_config = _model(trainable_config=True)
        functional, packet = dn.func.make_functional_network(model)
        parameters = packet.independent_parameters("netstim.parameters.tau")
        original = parameters["netstim"]["parameters"]["tau"]
        initialized = functional._initialize_host(
            parameters,
            packet.constants,
            packet.state,
            functional.initialization_inputs._replace(reinit_weights=False),
        )
        name = next(iter(getattr(native, collection)))
        native.initialize(DT, reinit_weights=False)
        native_weights = getattr(native, collection)[name].weight.w
        expected = torch.autograd.grad(native_weights.sum(), native_config)[0]

        def materialize(value):
            replaced = pytree.tree_map_only(
                torch.Tensor,
                lambda leaf: value if leaf is original else leaf,
                initialized.parameters,
            )
            return functional._materialize_connections(replaced, initialized.constants)[
                collection
            ][name]["weights"].sum()

        # Forward AD must keep tangents from an ordinary non-requires-grad
        # primal; retained source values use the destination input's Jacobian.
        primal = original.detach()
        value, tangent = torch.func.jvp(
            materialize, (primal,), (torch.ones_like(primal),)
        )
        torch.testing.assert_close(value, native_weights.sum())
        torch.testing.assert_close(tangent, expected)
        batched = torch.vmap(torch.func.grad(materialize))(
            torch.stack((primal, primal + 0.2, primal + 0.4))
        )
        torch.testing.assert_close(batched, expected.expand(3))
