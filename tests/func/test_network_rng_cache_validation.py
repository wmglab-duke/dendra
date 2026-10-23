"""Cold cache validation covers registered, non-declaring shared RNG aliases."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas
from tests.func.test_network import DT, DTYPE, _clone, _close
from tests.func.test_network_rng_sharing import _initialization, _network
from tests.func.test_stochastic_initialization_ordering import _native_close

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"):
        yield


def _cached_network(cache_kind="population", *, ignore=False):
    owners = _network()
    quiet = dn.SingleCompartment(N=1, C=2, integrator=dn.bwd_euler_sc(imem=False))
    quiet.insert(pas)
    quiet.initialize()
    quiet.shared_stream = owners.left.mech.probe.primary
    quiet.late_alias = quiet.shared_stream
    quiet.shared_stream.ignore_rng_on_load(ignore)
    model = dn.Network({"left": owners.left, "right": owners.right, "quiet": quiet})
    model.train()
    model.initialize(DT)
    model.step()
    if cache_kind == "network":
        model.cache_state()
        cache = model._state_cache["quiet"]
    else:
        quiet.cache("_steady_state")
        cache = quiet._caches["_steady_state"]
    model.step()
    return model, cache


@pytest.mark.parametrize("cache_kind", ("population", "network"))
@pytest.mark.parametrize("alias", ("shared_stream", "late_alias"))
def test_quiet_shared_rng_invalid_native_cache_rejects_at_cold_boundary(
    cache_kind, alias
):
    source, cache = _cached_network(cache_kind)
    cache[f"{alias}._extra_state"]["rng_state"]["cpu"] = torch.zeros_like(
        cache[f"{alias}._extra_state"]["rng_state"]["cpu"]
    )
    native = copy.deepcopy(source)
    rng_before = _clone(native.quiet.shared_stream.get_extra_state(), detached=True)
    with pytest.raises(RuntimeError, match="not a valid generator state"):
        native.quiet.load_state_dict(cache)
    _close(
        native.quiet.shared_stream.get_extra_state(),
        rng_before if alias == "shared_stream" else cache["shared_stream._extra_state"],
    )

    saved = _clone(source.state_dict_for_checkpoint(), detached=True)
    global_rng = torch.get_rng_state().clone()
    functional, tensors = dn.func.make_functional_network(source)
    assert functional.populations["quiet"]._random_plan is None
    with pytest.raises(
        dn.func.FunctionalizationError, match="not a valid generator state"
    ):
        functional.sample_population_initialization_draws(
            tensors.parameters, tensors.state
        )
    with pytest.raises(
        dn.func.FunctionalizationError, match="not a valid generator state"
    ):
        functional._initialize_host(*tensors, functional.initialization_inputs)
    _close(source.state_dict_for_checkpoint(), saved)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("alias", ("shared_stream", "late_alias"))
def test_quiet_shared_rng_normalized_cache_rejects_invalid_generator_bytes(alias):
    source, _ = _cached_network()
    functional, _ = dn.func.make_functional_network(source)
    cache_plan = functional.populations["quiet"]._owned_cache_plan
    cache = _clone(cache_plan.default, detached=True)
    cache[f"{alias}._extra_state"]["rng_state"]["cpu"].zero_()
    saved = _clone(source.state_dict_for_checkpoint(), detached=True)
    with pytest.raises(RuntimeError, match="not a valid generator state"):
        cache_plan.normalize(cache)
    _close(source.state_dict_for_checkpoint(), saved)


def test_quiet_shared_rng_cache_preserves_native_ignore_rng_on_load():
    source, cache = _cached_network(ignore=True)
    for alias in ("shared_stream", "late_alias"):
        cache[f"{alias}._extra_state"]["rng_state"]["cpu"].zero_()
    native = copy.deepcopy(source)
    functional, tensors = dn.func.make_functional_network(source)
    initial = functional.initialize(*tensors, _initialization(functional, tensors))
    native.initialize(DT)
    _native_close(initial.state, functional.extract(native).state)
