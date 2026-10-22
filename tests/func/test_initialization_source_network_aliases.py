"""Network initialization sources retain canonical raw Parameter ownership."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_initialization_source_execution import (
    DT,
    DTYPE,
    _AffineInitializer,
    _BufferAliasInitializer,
    _population,
)
from tests.func.test_shared_field_initialization import _assert_tree_close

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


def _network(*, source_parameter=False, source_first=True):
    # The runner can select an initially frozen model Parameter for training.
    shared = torch.nn.Parameter(torch.tensor(0.001, dtype=DTYPE), requires_grad=False)
    if source_parameter:
        source = _AffineInitializer(0.001, offset=140.0)
        source.raw = shared
    else:
        source = _BufferAliasInitializer(shared)
    initializer = _population(source=source)
    owner = _population("custom")
    owner.mech.leak.g_param = shared
    populations = {"initializer": initializer, "owner": owner}
    if not source_first:
        populations = dict(reversed(tuple(populations.items())))
    network = dn.Network(populations)
    network.train()
    with torch.no_grad():
        network.initialize(DT)
    for population in network.populations.values():
        population.detach()
    return network


def _compile(network, *, initialization="fresh"):
    return dn.compile(
        network,
        dt=DT,
        steps=1,
        initialization=initialization,
        inference=False,
        trainable=("*owner*g_param",),
        backend="aot_eager",
    )


@pytest.mark.parametrize("source_first", (False, True))
def test_cross_population_initializer_buffer_alias_rejects_fresh_only(source_first):
    network = _network(source_first=source_first)
    fresh = _compile(network)
    leaf = fresh.get_parameter("*owner*g_param")
    with torch.no_grad():
        leaf.fill_(1.0)
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="register the same Parameter on the initializer module",
    ):
        fresh()

    # Snapshot/explicit-state execution does not reevaluate initializer recipes.
    snapshot = _compile(network, initialization="snapshot")
    with torch.no_grad():
        snapshot.get_parameter("*owner*g_param").fill_(1.0)
    expected, _ = snapshot()
    actual, _ = fresh(state=fresh.functional.extract().state)
    _assert_tree_close(actual, expected)
    nao = actual["populations"]["initializer"]["ions"]["na"]["nao"]
    torch.testing.assert_close(nao, torch.full_like(nao, 140.001))


@pytest.mark.parametrize("source_first", (False, True))
def test_registered_source_parameter_shares_updates_and_native_gradients(source_first):
    network = _network(source_parameter=True, source_first=source_first)
    native = copy.deepcopy(network)
    runner = _compile(network)
    leaf = runner.get_parameter("*owner*g_param")
    assert runner.get_parameter("*initializer*o_init.raw") is leaf
    with torch.no_grad():
        leaf.fill_(1.0)
        native.owner.mech.leak.g_param.fill_(1.0)
    native.owner.mech.leak.g_param.requires_grad_(True)
    actual, _ = runner()
    native.initialize(DT)
    native.step()
    _assert_tree_close(actual, runner.functional.extract(native).state)
    nao = actual["populations"]["initializer"]["ions"]["na"]["nao"]
    torch.testing.assert_close(nao, torch.full_like(nao, 141.0))
    loss = nao.sum() + sum(
        state["integrator"]["v"].square().sum()
        for state in actual["populations"].values()
    )
    native_loss = native.initializer.mech.ions["na"].nao.sum() + sum(
        population.v.square().sum() for population in native.populations.values()
    )
    observed = torch.autograd.grad(loss, leaf)[0]
    expected = torch.autograd.grad(native_loss, native.owner.mech.leak.g_param)[0]
    torch.testing.assert_close(observed, expected, rtol=2e-10, atol=2e-11)


def test_population_cache_skips_unevaluated_initializer_buffer_alias():
    network = _network(source_first=True)
    with torch.no_grad():
        network.initializer.cache("_steady_state")
    functional, tensors = dn.func.make_functional_network(network)
    assert not any(
        "Cross-Population initializer buffer/parameter" in failure
        for failure in functional._initialization_alias_failures
    )
    actual = functional.initialize(*tensors, functional.initialization_inputs)
    native = copy.deepcopy(network)
    native.initialize(DT)
    _assert_tree_close(actual.state, functional.extract(native).state)


def test_network_cache_skips_unevaluated_initializer_buffer_alias():
    network = _network()
    with torch.no_grad():
        network.cache_state()
    functional, tensors = dn.func.make_functional_network(network)
    actual = functional.initialize(*tensors, functional.initialization_inputs)
    native = copy.deepcopy(network)
    native.initialize(DT)
    _assert_tree_close(actual.state, functional.extract(native).state)
