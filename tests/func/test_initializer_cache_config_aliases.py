"""Cache alias closures include nonpersistent parametrization config readers."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils import parametrize

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas
from tests.func.test_network import _clone, _close
from tests.func.test_network_bitpacked_inference import _sync_stimulus

pytestmark = pytest.mark.cpu
DT = 0.1


class _NonpersistentOffset(torch.nn.Module):
    def __init__(self, offset, *, persistent=False):
        super().__init__()
        self.register_buffer("offset", offset, persistent=persistent)

    def forward(self, value):
        return value + self.offset


def _network(cache_kind, *, expanded=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        populations = {}
        for name in ("left", "right"):
            population = dn.SingleCompartment(
                N=2 if expanded and name == "right" else 1,
                C=1,
                integrator=dn.bwd_euler_sc(imem=False),
            )
            population.insert(pas)
            populations[name] = population
        shared = torch.tensor(0.25, dtype=torch.float64)
        populations["left"].config = torch.nn.Module()
        populations["left"].config.register_buffer("saved_binding", shared)
        parametrize.register_parametrization(
            populations["right"],
            "cm",
            _NonpersistentOffset(
                shared.expand(2, 1) if expanded else shared, persistent=expanded
            ),
        )
        stimulus = dn.NetStim(
            N=1, interval=1.0, start=0.0, noise=0.0, tau=shared, seed=317
        )
        network = dn.Network(populations, netstim=stimulus)
        network.train()
        network.initialize(DT)
        if cache_kind == "network":
            network.cache_state()
        else:
            network.left.cache("_steady_state")
        return network


@pytest.mark.parametrize("cache_kind", ("network", "population"))
@pytest.mark.parametrize("compiled", (False, True))
def test_cached_parameter_alias_closure_restores_nonpersistent_config_readers(
    cache_kind, compiled
):
    source, native = _network(cache_kind), _network(cache_kind)
    _sync_stimulus(source, native)
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    if cache_kind == "network":
        defaults.cache.populations["left"]["config.saved_binding"].fill_(0.9)
        native._state_cache = _clone(defaults.cache.populations, detached=True)
    else:
        defaults.populations["left"].cache["config.saved_binding"].fill_(0.9)
        native.left._caches["_steady_state"] = _clone(
            defaults.populations["left"].cache, detached=True
        )
    defaults = defaults._replace(
        netstim_draw=functional.sample_initialization_draw(
            tensors.parameters, tensors.state, constants=tensors.constants
        )
    )
    with torch_compiler_warning_context():
        execute = (
            torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
            if compiled
            else functional.initialize
        )
        initialized = execute(*tensors, defaults)
    native.initialize(DT)
    expected = functional.extract(native)
    _close(initialized.parameters, expected.parameters)
    _close(initialized.state, expected.state)
    tau = initialized.parameters["netstim"]["parameters"]["tau"]
    offset = initialized.constants["populations"]["right"][
        "parametrizations.parametrizations.cm.0.offset"
    ]
    assert tau is offset
    torch.testing.assert_close(tau, torch.full_like(tau, 0.9))
    torch.testing.assert_close(offset, native.right.parametrizations.cm[0].offset)
    draw = functional.sample_step_draw(initialized.state)
    continued, _ = functional.prepare_and_step(*initialized, netstim_draw=draw)
    native.step()
    _close(continued, functional.extract(native).state)


@pytest.mark.parametrize("compiled", (False, True))
def test_expanded_cached_config_keeps_native_owned_storage_rebinding(compiled):
    source = _network("network", expanded=True)
    native = _network("network", expanded=True)
    _sync_stimulus(source, native)
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    defaults.cache.populations["left"]["config.saved_binding"].fill_(0.9)
    native._state_cache = _clone(defaults.cache.populations, detached=True)
    defaults = defaults._replace(
        netstim_draw=functional.sample_initialization_draw(
            tensors.parameters, tensors.state, constants=tensors.constants
        )
    )
    with torch_compiler_warning_context():
        execute = (
            torch.compile(functional.initialize, backend="aot_eager", fullgraph=True)
            if compiled
            else functional.initialize
        )
        initialized = execute(*tensors, defaults)
    native.initialize(DT)
    expected = functional.extract(native)
    _close(initialized.parameters, expected.parameters)
    _close(initialized.state, expected.state)
    offset = initialized.constants["populations"]["right"][
        "parametrizations.parametrizations.cm.0.offset"
    ]
    torch.testing.assert_close(offset, native.right.parametrizations.cm[0].offset)
    torch.testing.assert_close(offset, torch.full_like(offset, 0.25))
    tau = initialized.parameters["netstim"]["parameters"]["tau"]
    torch.testing.assert_close(tau, torch.full_like(tau, 0.9))
    assert tau is not offset
